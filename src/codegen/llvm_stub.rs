//! LLVM IR emitter for scalar `IrFunction`s.
//!
//! Emits valid LLVM IR text for functions whose bodies consist of scalar
//! arithmetic, comparisons, unary operations, and control flow.
//! Tensor types are lowered to opaque `ptr` (LLVM 15+ style).
//! `TensorOp` and `Call` instructions are emitted as opaque extern calls.
//!
//! Block-parameter SSA → LLVM phi conversion:
//! A pre-pass collects which predecessor blocks pass which values to each
//! block param, then emits `phi` instructions at the start of non-entry blocks.

use std::collections::HashMap;
use std::fmt::Write;

use crate::error::CodegenError;
use crate::ir::block::BlockId;
use crate::ir::function::IrFunction;
use crate::ir::instr::{BinOp, IrInstr, ScalarUnaryOp};
use crate::ir::module::IrModule;
use crate::ir::types::{DType, IrType};
use crate::ir::value::ValueId;

/// Emits LLVM IR for all functions in the module.
///
/// Phase 48 improvements:
/// - Target triple and data layout header.
/// - Global string constants for all `ConstStr` values.
/// - `declare` statements for all iris runtime helper functions.
pub fn emit_llvm_stub(module: &IrModule) -> Result<String, CodegenError> {
    let mut out = String::new();

    // ── Header ────────────────────────────────────────────────────────────
    writeln!(out, "; IRIS LLVM IR — scalar arithmetic + control flow")?;
    writeln!(out, "; Tensor ops use opaque ptr (LLVM 15+ style).\n")?;
    writeln!(
        out,
        "target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\""
    )?;
    writeln!(out, "target triple = \"x86_64-unknown-linux-gnu\"\n")?;

    // ── Collect global string constants ───────────────────────────────────
    // Build a dedup table: string content → global index.
    let mut str_table: HashMap<String, usize> = HashMap::new();
    let mut str_vec: Vec<String> = Vec::new();
    for func in module.functions() {
        for block in func.blocks() {
            for instr in &block.instrs {
                if let IrInstr::ConstStr { value, .. } = instr {
                    if !str_table.contains_key(value) {
                        let idx = str_vec.len();
                        str_table.insert(value.clone(), idx);
                        str_vec.push(value.clone());
                    }
                }
            }
        }
    }

    // ── Emit global string constants ──────────────────────────────────────
    for (idx, content) in str_vec.iter().enumerate() {
        let escaped = llvm_escape_string(content);
        let len = content.len() + 1; // +1 for null terminator
        writeln!(
            out,
            "@.str.{} = private unnamed_addr constant [{} x i8] c\"{}\\00\", align 1",
            idx, len, escaped
        )?;
    }
    if !str_vec.is_empty() {
        writeln!(out)?;
    }

    // ── Runtime function declarations ─────────────────────────────────────
    emit_iris_runtime_declares(&mut out)?;

    // ── Function definitions ──────────────────────────────────────────────
    for func in module.functions() {
        let ret = llvm_type_name(&func.return_ty)?;

        let params: Result<Vec<String>, CodegenError> = func
            .params
            .iter()
            .map(|p| Ok(format!("{} %{}", llvm_type_name(&p.ty)?, p.name)))
            .collect();
        let params = params?.join(", ");

        writeln!(out, "define {} @{}({}) {{", ret, func.name, params)?;
        emit_llvm_body(func, &str_table, &mut out)?;
        writeln!(out, "}}\n")?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Body emitter
// ---------------------------------------------------------------------------

fn emit_llvm_body(
    func: &IrFunction,
    str_table: &HashMap<String, usize>,
    out: &mut String,
) -> Result<(), CodegenError> {
    // Sub-pass A: collect constant values for inline substitution.
    // Constants are never emitted as LLVM instructions; they are used as
    // literal operands wherever the ValueId is referenced.
    let mut consts: HashMap<ValueId, String> = HashMap::new();
    for block in func.blocks() {
        for instr in &block.instrs {
            match instr {
                IrInstr::ConstFloat { result, value, .. } => {
                    consts.insert(*result, fmt_float(*value));
                }
                IrInstr::ConstInt { result, value, .. } => {
                    consts.insert(*result, value.to_string());
                }
                IrInstr::ConstBool { result, value } => {
                    consts.insert(*result, if *value { "true" } else { "false" }.to_owned());
                }
                _ => {}
            }
        }
    }

    // Sub-pass B: collect phi sources.
    // phi_src[(dest_block_id, param_index)] = Vec<(pred_block_id, value)>
    let mut phi_src: HashMap<(BlockId, usize), Vec<(BlockId, ValueId)>> = HashMap::new();
    for block in func.blocks() {
        for instr in &block.instrs {
            match instr {
                IrInstr::Br { target, args } => {
                    for (i, v) in args.iter().enumerate() {
                        phi_src
                            .entry((*target, i))
                            .or_default()
                            .push((block.id, *v));
                    }
                }
                IrInstr::CondBr {
                    then_block,
                    then_args,
                    else_block,
                    else_args,
                    ..
                } => {
                    for (i, v) in then_args.iter().enumerate() {
                        phi_src
                            .entry((*then_block, i))
                            .or_default()
                            .push((block.id, *v));
                    }
                    for (i, v) in else_args.iter().enumerate() {
                        phi_src
                            .entry((*else_block, i))
                            .or_default()
                            .push((block.id, *v));
                    }
                }
                _ => {}
            }
        }
    }

    let entry_id = func.blocks()[0].id;

    let mut gep_counter: u32 = 0;

    for block in func.blocks() {
        // Block label
        let blabel = block_label(block.name.as_deref(), block.id);
        writeln!(out, "{}:", blabel)?;

        // Phi nodes for non-entry block params
        if block.id != entry_id {
            for (i, param) in block.params.iter().enumerate() {
                let ty_s = llvm_type_name(&param.ty)?;
                let phi_name = format!("%v{}", param.id.0);
                let arms: Vec<String> = phi_src
                    .get(&(block.id, i))
                    .map(|srcs| {
                        srcs.iter()
                            .map(|(pred_id, v)| {
                                let vstr = llvm_val(*v, &consts, func);
                                let pred = block_label_by_id(func.blocks(), *pred_id);
                                format!("[ {}, %{} ]", vstr, pred)
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                writeln!(out, "  {} = phi {} {}", phi_name, ty_s, arms.join(", "))?;
            }
        }

        // Instructions
        for instr in &block.instrs {
            emit_llvm_instr(instr, &consts, func, &mut gep_counter, str_table, out)?;
        }
    }
    Ok(())
}

fn emit_llvm_instr(
    instr: &IrInstr,
    consts: &HashMap<ValueId, String>,
    func: &IrFunction,
    gep_counter: &mut u32,
    str_table: &HashMap<String, usize>,
    out: &mut String,
) -> Result<(), CodegenError> {
    let val = |v: ValueId| llvm_val(v, consts, func);

    match instr {
        // Skip constants — they are inlined at use sites.
        IrInstr::ConstFloat { .. } | IrInstr::ConstInt { .. } | IrInstr::ConstBool { .. } => {}

        IrInstr::BinOp {
            result,
            op,
            lhs,
            rhs,
            ty,
        } => {
            let lv = val(*lhs);
            let rv = val(*rhs);
            // For comparisons the result `ty` is Bool; use the left operand's
            // type to choose float (fcmp) vs integer (icmp/add/sub/...) forms.
            let operand_ty = func.value_type(*lhs).unwrap_or(ty);
            let ty_s = llvm_type_name(operand_ty)?;
            let is_float = matches!(operand_ty, IrType::Scalar(DType::F32 | DType::F64));
            let llvm_op = match (op, is_float) {
                (BinOp::Add, true) => format!("fadd {} {}, {}", ty_s, lv, rv),
                (BinOp::Sub, true) => format!("fsub {} {}, {}", ty_s, lv, rv),
                (BinOp::Mul, true) => format!("fmul {} {}, {}", ty_s, lv, rv),
                (BinOp::Div, true) => format!("fdiv {} {}, {}", ty_s, lv, rv),
                (BinOp::Add, false) => format!("add {} {}, {}", ty_s, lv, rv),
                (BinOp::Sub, false) => format!("sub {} {}, {}", ty_s, lv, rv),
                (BinOp::Mul, false) => format!("mul {} {}, {}", ty_s, lv, rv),
                (BinOp::Div, false) | (BinOp::FloorDiv, _) => {
                    format!("sdiv {} {}, {}", ty_s, lv, rv)
                }
                (BinOp::Mod, true) => format!("frem {} {}, {}", ty_s, lv, rv),
                (BinOp::Mod, false) => format!("srem {} {}, {}", ty_s, lv, rv),
                (BinOp::CmpEq, true) => format!("fcmp oeq {} {}, {}", ty_s, lv, rv),
                (BinOp::CmpNe, true) => format!("fcmp one {} {}, {}", ty_s, lv, rv),
                (BinOp::CmpLt, true) => format!("fcmp olt {} {}, {}", ty_s, lv, rv),
                (BinOp::CmpLe, true) => format!("fcmp ole {} {}, {}", ty_s, lv, rv),
                (BinOp::CmpGt, true) => format!("fcmp ogt {} {}, {}", ty_s, lv, rv),
                (BinOp::CmpGe, true) => format!("fcmp oge {} {}, {}", ty_s, lv, rv),
                (BinOp::CmpEq, false) => format!("icmp eq {} {}, {}", ty_s, lv, rv),
                (BinOp::CmpNe, false) => format!("icmp ne {} {}, {}", ty_s, lv, rv),
                (BinOp::CmpLt, false) => format!("icmp slt {} {}, {}", ty_s, lv, rv),
                (BinOp::CmpLe, false) => format!("icmp sle {} {}, {}", ty_s, lv, rv),
                (BinOp::CmpGt, false) => format!("icmp sgt {} {}, {}", ty_s, lv, rv),
                (BinOp::CmpGe, false) => format!("icmp sge {} {}, {}", ty_s, lv, rv),
                // Math builtins lower to LLVM intrinsic calls
                (BinOp::Pow, true)  => format!("call {} @llvm.pow.f64({} {}, {} {})", ty_s, ty_s, lv, ty_s, rv),
                (BinOp::Pow, false) => format!("call i64 @iris_pow_i64(i64 {}, i64 {})", lv, rv),
                (BinOp::Min, true)  => format!("call {} @llvm.minnum.f64({} {}, {} {})", ty_s, ty_s, lv, ty_s, rv),
                (BinOp::Min, false) => format!("call i64 @iris_min_i64(i64 {}, i64 {})", lv, rv),
                (BinOp::Max, true)  => format!("call {} @llvm.maxnum.f64({} {}, {} {})", ty_s, ty_s, lv, ty_s, rv),
                (BinOp::Max, false) => format!("call i64 @iris_max_i64(i64 {}, i64 {})", lv, rv),
                // Bitwise ops — integers only
                (BinOp::BitAnd, false) => format!("and {} {}, {}", ty_s, lv, rv),
                (BinOp::BitOr,  false) => format!("or {} {}, {}", ty_s, lv, rv),
                (BinOp::BitXor, false) => format!("xor {} {}, {}", ty_s, lv, rv),
                (BinOp::Shl,    false) => format!("shl {} {}, {}", ty_s, lv, rv),
                (BinOp::Shr,    false) => format!("ashr {} {}, {}", ty_s, lv, rv),
                (BinOp::BitAnd, true) | (BinOp::BitOr, true) | (BinOp::BitXor, true)
                | (BinOp::Shl, true) | (BinOp::Shr, true) =>
                    format!("call {} @iris_bitop_float_unsupported()", ty_s),
            };
            writeln!(out, "  %v{} = {}", result.0, llvm_op)?;
        }

        IrInstr::UnaryOp {
            result,
            op,
            operand,
            ty,
        } => {
            let ov = val(*operand);
            let ty_s = llvm_type_name(ty)?;
            let is_float = matches!(ty, IrType::Scalar(DType::F32 | DType::F64));
            match op {
                ScalarUnaryOp::Neg if is_float => {
                    writeln!(out, "  %v{} = fneg {} {}", result.0, ty_s, ov)?;
                }
                ScalarUnaryOp::Neg => {
                    writeln!(out, "  %v{} = sub {} 0, {}", result.0, ty_s, ov)?;
                }
                ScalarUnaryOp::Not => {
                    writeln!(out, "  %v{} = xor i1 {}, true", result.0, ov)?;
                }
                ScalarUnaryOp::Sqrt => {
                    writeln!(out, "  %v{} = call {} @llvm.sqrt.f64({} {})", result.0, ty_s, ty_s, ov)?;
                }
                ScalarUnaryOp::Abs if is_float => {
                    writeln!(out, "  %v{} = call {} @llvm.fabs.f64({} {})", result.0, ty_s, ty_s, ov)?;
                }
                ScalarUnaryOp::Abs => {
                    writeln!(out, "  ; abs({}) -- iris runtime call", ov)?;
                    writeln!(out, "  %v{} = call i64 @iris_abs_i64(i64 {})", result.0, ov)?;
                }
                ScalarUnaryOp::Floor => {
                    writeln!(out, "  %v{} = call {} @llvm.floor.f64({} {})", result.0, ty_s, ty_s, ov)?;
                }
                ScalarUnaryOp::Ceil => {
                    writeln!(out, "  %v{} = call {} @llvm.ceil.f64({} {})", result.0, ty_s, ty_s, ov)?;
                }
                ScalarUnaryOp::BitNot => {
                    // Bitwise NOT: xor with all-ones (-1 in two's complement)
                    writeln!(out, "  %v{} = xor {} {}, -1", result.0, ty_s, ov)?;
                }
                // Phase 36: trig / transcendental builtins
                ScalarUnaryOp::Sin => {
                    writeln!(out, "  %v{} = call {} @llvm.sin.f64({} {})", result.0, ty_s, ty_s, ov)?;
                }
                ScalarUnaryOp::Cos => {
                    writeln!(out, "  %v{} = call {} @llvm.cos.f64({} {})", result.0, ty_s, ty_s, ov)?;
                }
                ScalarUnaryOp::Tan => {
                    writeln!(out, "  %v{} = call double @tan(double {})", result.0, ov)?;
                }
                ScalarUnaryOp::Exp => {
                    writeln!(out, "  %v{} = call {} @llvm.exp.f64({} {})", result.0, ty_s, ty_s, ov)?;
                }
                ScalarUnaryOp::Log => {
                    writeln!(out, "  %v{} = call {} @llvm.log.f64({} {})", result.0, ty_s, ty_s, ov)?;
                }
                ScalarUnaryOp::Log2 => {
                    writeln!(out, "  %v{} = call {} @llvm.log2.f64({} {})", result.0, ty_s, ty_s, ov)?;
                }
                ScalarUnaryOp::Round => {
                    writeln!(out, "  %v{} = call {} @llvm.round.f64({} {})", result.0, ty_s, ty_s, ov)?;
                }
                ScalarUnaryOp::Sign => {
                    writeln!(out, "  %v{} = call double @iris_sign_f64(double {})", result.0, ov)?;
                }
            }
        }

        IrInstr::Cast {
            result,
            operand,
            from_ty,
            to_ty,
        } => {
            let ov = val(*operand);
            let from_s = llvm_type_name(from_ty)?;
            let to_s = llvm_type_name(to_ty)?;
            let is_from_float = matches!(from_ty, IrType::Scalar(DType::F32 | DType::F64));
            let is_to_float = matches!(to_ty, IrType::Scalar(DType::F32 | DType::F64));
            let is_from_int = matches!(from_ty, IrType::Scalar(DType::I32 | DType::I64));
            let is_to_int = matches!(to_ty, IrType::Scalar(DType::I32 | DType::I64));
            let is_from_f64 = matches!(from_ty, IrType::Scalar(DType::F64));
            let is_to_f64 = matches!(to_ty, IrType::Scalar(DType::F64));
            let is_from_i64 = matches!(from_ty, IrType::Scalar(DType::I64));
            let is_to_i64 = matches!(to_ty, IrType::Scalar(DType::I64));
            if from_ty == to_ty {
                // No-op cast: emit identity
                writeln!(
                    out,
                    "  %v{} = bitcast {} {} to {}",
                    result.0, from_s, ov, to_s
                )?;
            } else if is_from_float && is_to_int {
                writeln!(
                    out,
                    "  %v{} = fptosi {} {} to {}",
                    result.0, from_s, ov, to_s
                )?;
            } else if is_from_int && is_to_float {
                writeln!(
                    out,
                    "  %v{} = sitofp {} {} to {}",
                    result.0, from_s, ov, to_s
                )?;
            } else if is_from_float && is_to_float {
                if !is_from_f64 && is_to_f64 {
                    writeln!(
                        out,
                        "  %v{} = fpext {} {} to {}",
                        result.0, from_s, ov, to_s
                    )?;
                } else {
                    writeln!(
                        out,
                        "  %v{} = fptrunc {} {} to {}",
                        result.0, from_s, ov, to_s
                    )?;
                }
            } else if is_from_int && is_to_int {
                if !is_from_i64 && is_to_i64 {
                    writeln!(out, "  %v{} = sext {} {} to {}", result.0, from_s, ov, to_s)?;
                } else {
                    writeln!(
                        out,
                        "  %v{} = trunc {} {} to {}",
                        result.0, from_s, ov, to_s
                    )?;
                }
            } else {
                writeln!(
                    out,
                    "  %v{} = bitcast {} {} to {}",
                    result.0, from_s, ov, to_s
                )?;
            }
        }

        IrInstr::Return { values } => {
            if values.is_empty() {
                writeln!(out, "  ret void")?;
            } else {
                let v = val(values[0]);
                let ty_s = llvm_type_name(&func.return_ty)?;
                writeln!(out, "  ret {} {}", ty_s, v)?;
            }
        }

        IrInstr::Br { target, .. } => {
            let lbl = block_label_by_id(func.blocks(), *target);
            writeln!(out, "  br label %{}", lbl)?;
        }

        IrInstr::CondBr {
            cond,
            then_block,
            else_block,
            ..
        } => {
            let cv = val(*cond);
            let tl = block_label_by_id(func.blocks(), *then_block);
            let el = block_label_by_id(func.blocks(), *else_block);
            writeln!(out, "  br i1 {}, label %{}, label %{}", cv, tl, el)?;
        }

        IrInstr::Call {
            result,
            callee,
            args,
            ..
        } => {
            let args_str: Vec<String> = args.iter().map(|a| format!("ptr {}", val(*a))).collect();
            if let Some(r) = result {
                writeln!(
                    out,
                    "  %v{} = call ptr @iris_call_{}({})",
                    r.0,
                    callee,
                    args_str.join(", ")
                )?;
            } else {
                writeln!(
                    out,
                    "  call void @iris_call_{}({})",
                    callee,
                    args_str.join(", ")
                )?;
            }
        }

        IrInstr::TensorOp { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_tensor_op()", result.0)?;
        }

        IrInstr::Load {
            result,
            tensor,
            indices,
            result_ty,
        } => {
            let tv = val(*tensor);
            let ty_s = llvm_type_name(result_ty)?;
            match indices.as_slice() {
                [] => {
                    // No index: load directly from the tensor pointer.
                    writeln!(out, "  %v{} = load {}, ptr {}", result.0, ty_s, tv)?;
                }
                [idx] => {
                    // Single index: GEP then load.
                    let gep = format!("%gep{}", *gep_counter);
                    *gep_counter += 1;
                    writeln!(
                        out,
                        "  {} = getelementptr {}, ptr {}, i64 {}",
                        gep,
                        ty_s,
                        tv,
                        val(*idx)
                    )?;
                    writeln!(out, "  %v{} = load {}, ptr {}", result.0, ty_s, gep)?;
                }
                _ => {
                    // Multi-index: delegate to opaque runtime helper with all operands.
                    let mut args = vec![format!("ptr {}", tv)];
                    for idx in indices {
                        args.push(format!("i64 {}", val(*idx)));
                    }
                    writeln!(
                        out,
                        "  %v{} = call {} @iris_tensor_load({})",
                        result.0,
                        ty_s,
                        args.join(", ")
                    )?;
                }
            }
        }

        IrInstr::Store {
            tensor,
            indices,
            value,
        } => {
            let tv = val(*tensor);
            let vv = val(*value);
            let ty_s = func
                .value_type(*value)
                .and_then(|ty| llvm_type_name(ty).ok())
                .unwrap_or_else(|| "ptr".to_owned());
            match indices.as_slice() {
                [] => {
                    // No index: store directly through the tensor pointer.
                    writeln!(out, "  store {} {}, ptr {}", ty_s, vv, tv)?;
                }
                [idx] => {
                    // Single index: GEP then store.
                    let gep = format!("%gep{}", *gep_counter);
                    *gep_counter += 1;
                    writeln!(
                        out,
                        "  {} = getelementptr {}, ptr {}, i64 {}",
                        gep,
                        ty_s,
                        tv,
                        val(*idx)
                    )?;
                    writeln!(out, "  store {} {}, ptr {}", ty_s, vv, gep)?;
                }
                _ => {
                    // Multi-index: delegate to opaque runtime helper with all operands.
                    let mut args = vec![format!("ptr {}", tv), format!("{} {}", ty_s, vv)];
                    for idx in indices {
                        args.push(format!("i64 {}", val(*idx)));
                    }
                    writeln!(out, "  call void @iris_tensor_store({})", args.join(", "))?;
                }
            }
        }

        // Struct ops: emit as opaque runtime calls.
        IrInstr::MakeStruct { result, fields, .. } => {
            let args_str: Vec<String> = fields.iter().map(|f| format!("ptr {}", val(*f))).collect();
            writeln!(
                out,
                "  %v{} = call ptr @iris_make_struct({})",
                result.0,
                args_str.join(", ")
            )?;
        }

        IrInstr::GetField {
            result,
            base,
            field_index,
            result_ty,
        } => {
            let ty_s = llvm_type_name(result_ty).unwrap_or_else(|_| "ptr".to_owned());
            writeln!(
                out,
                "  %v{} = call {} @iris_get_field(ptr {}, i32 {})",
                result.0,
                ty_s,
                val(*base),
                field_index
            )?;
        }

        IrInstr::MakeVariant {
            result,
            variant_idx,
            ..
        } => {
            // Emit tag as i64; payload fields are stored via runtime calls (stub).
            writeln!(out, "  %v{} = add i64 0, {}", result.0, variant_idx)?;
        }

        IrInstr::ExtractVariantField {
            result,
            operand,
            field_idx,
            ..
        } => {
            writeln!(
                out,
                "  %v{} = call i64 @iris_extract_variant_field({}, i64 {})",
                result.0, val(*operand), field_idx
            )?;
        }

        IrInstr::SwitchVariant {
            scrutinee,
            arms,
            default_block,
        } => {
            let sv = val(*scrutinee);
            let blocks = func.blocks();
            // Emit LLVM `switch` instruction.
            let default = default_block
                .map(|bb| format!("label %{}", block_label_by_id(blocks, bb)))
                .unwrap_or_else(|| {
                    // Reuse first arm as default for exhaustive match.
                    arms.first()
                        .map(|(_, bb)| format!("label %{}", block_label_by_id(blocks, *bb)))
                        .unwrap_or_else(|| "label %unreachable".to_owned())
                });
            write!(out, "  switch i64 {}, {} [", sv, default)?;
            for (idx, bb) in arms {
                write!(
                    out,
                    " i64 {}, label %{}",
                    idx,
                    block_label_by_id(blocks, *bb)
                )?;
            }
            writeln!(out, " ]")?;
        }

        // Tuple ops: emit as opaque runtime calls.
        IrInstr::MakeTuple {
            result, elements, ..
        } => {
            let args_str: Vec<String> = elements
                .iter()
                .map(|e| format!("ptr {}", val(*e)))
                .collect();
            writeln!(
                out,
                "  %v{} = call ptr @iris_make_tuple({})",
                result.0,
                args_str.join(", ")
            )?;
        }

        IrInstr::GetElement {
            result,
            base,
            index,
            result_ty,
        } => {
            let ty_s = llvm_type_name(result_ty).unwrap_or_else(|_| "ptr".to_owned());
            writeln!(
                out,
                "  %v{} = call {} @iris_get_element(ptr {}, i32 {})",
                result.0,
                ty_s,
                val(*base),
                index
            )?;
        }

        // Array ops: emit as opaque runtime calls.
        IrInstr::AllocArray { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_alloc_array()", result.0)?;
        }

        IrInstr::ArrayLoad { result, array, index, elem_ty } => {
            let ty_s = llvm_type_name(elem_ty).unwrap_or_else(|_| "i64".to_owned());
            writeln!(
                out,
                "  %v{} = call {} @iris_array_load(ptr {}, i64 {})",
                result.0,
                ty_s,
                val(*array),
                val(*index)
            )?;
        }

        IrInstr::ArrayStore { array, index, value } => {
            writeln!(
                out,
                "  call void @iris_array_store(ptr {}, i64 {}, ptr {})",
                val(*array),
                val(*index),
                val(*value)
            )?;
        }

        IrInstr::ParFor { body_fn, start, end, .. } => {
            writeln!(out, "  call void @iris_par_for(ptr @{}, i64 {}, i64 {})", body_fn, val(*start), val(*end))?;
        }

        // Channel ops: emit as opaque runtime calls.
        IrInstr::ChanNew { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_chan_new()", result.0)?;
        }
        IrInstr::ChanSend { chan, value } => {
            writeln!(out, "  call void @iris_chan_send(ptr {}, ptr {})", val(*chan), val(*value))?;
        }
        IrInstr::ChanRecv { result, chan, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_chan_recv(ptr {})", result.0, val(*chan))?;
        }
        IrInstr::Spawn { body_fn, .. } => {
            writeln!(out, "  call void @iris_spawn_fn(ptr @{})", body_fn)?;
        }

        // Atomic / Mutex ops: emit as opaque runtime calls.
        IrInstr::AtomicNew { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_atomic_new()", result.0)?;
        }
        IrInstr::AtomicLoad { result, atomic, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_atomic_load(ptr {})", result.0, val(*atomic))?;
        }
        IrInstr::AtomicStore { atomic, value } => {
            writeln!(out, "  call void @iris_atomic_store(ptr {}, ptr {})", val(*atomic), val(*value))?;
        }
        IrInstr::AtomicAdd { result, atomic, value, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_atomic_add(ptr {}, ptr {})", result.0, val(*atomic), val(*value))?;
        }
        IrInstr::MutexNew { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_mutex_new()", result.0)?;
        }
        IrInstr::MutexLock { result, mutex, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_mutex_lock(ptr {})", result.0, val(*mutex))?;
        }
        IrInstr::MutexUnlock { mutex } => {
            writeln!(out, "  call void @iris_mutex_unlock(ptr {})", val(*mutex))?;
        }

        // Option ops: emit as opaque runtime calls.
        IrInstr::MakeSome { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_make_some()", result.0)?;
        }
        IrInstr::MakeNone { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_make_none()", result.0)?;
        }
        IrInstr::IsSome { result, operand } => {
            writeln!(out, "  %v{} = call i1 @iris_is_some(ptr {})", result.0, val(*operand))?;
        }
        IrInstr::OptionUnwrap { result, operand, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_option_unwrap(ptr {})", result.0, val(*operand))?;
        }

        // Result ops: emit as opaque runtime calls.
        IrInstr::MakeOk { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_make_ok()", result.0)?;
        }
        IrInstr::MakeErr { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_make_err()", result.0)?;
        }
        IrInstr::IsOk { result, operand } => {
            writeln!(out, "  %v{} = call i1 @iris_is_ok(ptr {})", result.0, val(*operand))?;
        }
        IrInstr::ResultUnwrap { result, operand, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_result_unwrap(ptr {})", result.0, val(*operand))?;
        }
        IrInstr::ResultUnwrapErr { result, operand, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_result_unwrap_err(ptr {})", result.0, val(*operand))?;
        }

        // String ops: use getelementptr into the global string constant.
        IrInstr::ConstStr { result, value } => {
            if let Some(&idx) = str_table.get(value) {
                let len = value.len() + 1;
                writeln!(
                    out,
                    "  %v{} = getelementptr inbounds [{} x i8], ptr @.str.{}, i32 0, i32 0",
                    result.0, len, idx
                )?;
            } else {
                // Fallback: should not happen if emit_llvm_stub populated str_table correctly.
                writeln!(out, "  %v{} = call ptr @iris_const_str()", result.0)?;
            }
        }

        IrInstr::StrLen { result, operand } => {
            writeln!(
                out,
                "  %v{} = call i64 @iris_str_len(ptr {})",
                result.0,
                val(*operand)
            )?;
        }

        IrInstr::StrConcat { result, lhs, rhs } => {
            writeln!(
                out,
                "  %v{} = call ptr @iris_str_concat(ptr {}, ptr {})",
                result.0,
                val(*lhs),
                val(*rhs)
            )?;
        }

        IrInstr::MakeGrad { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_make_grad()", result.0)?;
        }

        IrInstr::GradValue { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_grad_value()", result.0)?;
        }

        IrInstr::GradTangent { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_grad_tangent()", result.0)?;
        }

        IrInstr::Sparsify { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_sparsify()", result.0)?;
        }

        IrInstr::Densify { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_densify()", result.0)?;
        }

        IrInstr::Barrier => {
            // barrier is a no-op in LLVM stub
            writeln!(out, "  ; barrier")?;
        }

        IrInstr::Print { operand } => {
            writeln!(out, "  call void @iris_print(ptr {})", val(*operand))?;
        }

        IrInstr::StrContains { result, haystack, needle } => {
            writeln!(out, "  %v{} = call i1 @iris_str_contains(ptr {}, ptr {})", result.0, val(*haystack), val(*needle))?;
        }
        IrInstr::StrStartsWith { result, haystack, prefix } => {
            writeln!(out, "  %v{} = call i1 @iris_str_starts_with(ptr {}, ptr {})", result.0, val(*haystack), val(*prefix))?;
        }
        IrInstr::StrEndsWith { result, haystack, suffix } => {
            writeln!(out, "  %v{} = call i1 @iris_str_ends_with(ptr {}, ptr {})", result.0, val(*haystack), val(*suffix))?;
        }
        IrInstr::StrToUpper { result, operand } => {
            writeln!(out, "  %v{} = call ptr @iris_str_to_upper(ptr {})", result.0, val(*operand))?;
        }
        IrInstr::StrToLower { result, operand } => {
            writeln!(out, "  %v{} = call ptr @iris_str_to_lower(ptr {})", result.0, val(*operand))?;
        }
        IrInstr::StrTrim { result, operand } => {
            writeln!(out, "  %v{} = call ptr @iris_str_trim(ptr {})", result.0, val(*operand))?;
        }
        IrInstr::StrRepeat { result, operand, count } => {
            writeln!(out, "  %v{} = call ptr @iris_str_repeat(ptr {}, i64 {})", result.0, val(*operand), val(*count))?;
        }

        IrInstr::Panic { msg } => {
            writeln!(out, "  call void @iris_panic(ptr {})", val(*msg))?;
            writeln!(out, "  unreachable")?;
        }

        IrInstr::ValueToStr { result, operand } => {
            writeln!(out, "  %v{} = call ptr @iris_value_to_str(ptr {})", result.0, val(*operand))?;
        }

        IrInstr::ReadLine { result } => {
            writeln!(out, "  %v{} = call ptr @iris_read_line()", result.0)?;
        }

        IrInstr::ReadI64 { result } => {
            writeln!(out, "  %v{} = call i64 @iris_read_i64()", result.0)?;
        }

        IrInstr::ReadF64 { result } => {
            writeln!(out, "  %v{} = call double @iris_read_f64()", result.0)?;
        }

        IrInstr::ParseI64 { result, operand } => {
            writeln!(out, "  %v{} = call ptr @iris_parse_i64(ptr {})", result.0, val(*operand))?;
        }

        IrInstr::ParseF64 { result, operand } => {
            writeln!(out, "  %v{} = call ptr @iris_parse_f64(ptr {})", result.0, val(*operand))?;
        }

        IrInstr::StrIndex { result, string, index } => {
            writeln!(out, "  %v{} = call i64 @iris_str_index(ptr {}, i64 {})", result.0, val(*string), val(*index))?;
        }

        IrInstr::StrSlice { result, string, start, end } => {
            writeln!(out, "  %v{} = call ptr @iris_str_slice(ptr {}, i64 {}, i64 {})", result.0, val(*string), val(*start), val(*end))?;
        }

        IrInstr::StrFind { result, haystack, needle } => {
            writeln!(out, "  %v{} = call ptr @iris_str_find(ptr {}, ptr {})", result.0, val(*haystack), val(*needle))?;
        }

        IrInstr::StrReplace { result, string, from, to } => {
            writeln!(out, "  %v{} = call ptr @iris_str_replace(ptr {}, ptr {}, ptr {})", result.0, val(*string), val(*from), val(*to))?;
        }

        IrInstr::ListNew { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_list_new()", result.0)?;
        }
        IrInstr::ListPush { list, value } => {
            writeln!(out, "  call void @iris_list_push(ptr {}, ptr {})", val(*list), val(*value))?;
        }
        IrInstr::ListLen { result, list } => {
            writeln!(out, "  %v{} = call i64 @iris_list_len(ptr {})", result.0, val(*list))?;
        }
        IrInstr::ListGet { result, list, index, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_list_get(ptr {}, i64 {})", result.0, val(*list), val(*index))?;
        }
        IrInstr::ListSet { list, index, value } => {
            writeln!(out, "  call void @iris_list_set(ptr {}, i64 {}, ptr {})", val(*list), val(*index), val(*value))?;
        }
        IrInstr::ListPop { result, list, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_list_pop(ptr {})", result.0, val(*list))?;
        }

        IrInstr::MapNew { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_map_new()", result.0)?;
        }
        IrInstr::MapSet { map, key, value } => {
            writeln!(out, "  call void @iris_map_set(ptr {}, ptr {}, ptr {})", val(*map), val(*key), val(*value))?;
        }
        IrInstr::MapGet { result, map, key, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_map_get(ptr {}, ptr {})", result.0, val(*map), val(*key))?;
        }
        IrInstr::MapContains { result, map, key } => {
            writeln!(out, "  %v{} = call i1 @iris_map_contains(ptr {}, ptr {})", result.0, val(*map), val(*key))?;
        }
        IrInstr::MapRemove { map, key } => {
            writeln!(out, "  call void @iris_map_remove(ptr {}, ptr {})", val(*map), val(*key))?;
        }
        IrInstr::MapLen { result, map } => {
            writeln!(out, "  %v{} = call i64 @iris_map_len(ptr {})", result.0, val(*map))?;
        }

        IrInstr::MakeClosure { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_make_closure()", result.0)?;
        }

        IrInstr::CallClosure { result, closure, args, .. } => {
            let args_str: Vec<String> = args.iter().map(|a| format!("ptr {}", val(*a))).collect();
            if let Some(r) = result {
                writeln!(out, "  %v{} = call ptr @iris_call_closure(ptr {}, {})", r.0, val(*closure), args_str.join(", "))?;
            } else {
                writeln!(out, "  call void @iris_call_closure_void(ptr {}, {})", val(*closure), args_str.join(", "))?;
            }
        }

        // Phase 56: File I/O
        IrInstr::FileReadAll { result, path } => {
            writeln!(out, "  %v{} = call ptr @iris_file_read_all(ptr {})", result.0, val(*path))?;
        }
        IrInstr::FileWriteAll { result, path, content } => {
            writeln!(out, "  %v{} = call ptr @iris_file_write_all(ptr {}, ptr {})", result.0, val(*path), val(*content))?;
        }
        IrInstr::FileExists { result, path } => {
            writeln!(out, "  %v{} = call i1 @iris_file_exists(ptr {})", result.0, val(*path))?;
        }
        IrInstr::FileLines { result, path } => {
            writeln!(out, "  %v{} = call ptr @iris_file_lines(ptr {})", result.0, val(*path))?;
        }

        // Phase 58: Extended collections
        IrInstr::ListContains { result, list, value } => {
            writeln!(out, "  %v{} = call i1 @iris_list_contains(ptr {}, ptr {})", result.0, val(*list), val(*value))?;
        }
        IrInstr::ListSort { list } => {
            writeln!(out, "  call void @iris_list_sort(ptr {})", val(*list))?;
        }
        IrInstr::MapKeys { result, map } => {
            writeln!(out, "  %v{} = call ptr @iris_map_keys(ptr {})", result.0, val(*map))?;
        }
        IrInstr::MapValues { result, map } => {
            writeln!(out, "  %v{} = call ptr @iris_map_values(ptr {})", result.0, val(*map))?;
        }
        IrInstr::ListConcat { result, lhs, rhs } => {
            writeln!(out, "  %v{} = call ptr @iris_list_concat(ptr {}, ptr {})", result.0, val(*lhs), val(*rhs))?;
        }
        IrInstr::ListSlice { result, list, start, end } => {
            writeln!(out, "  %v{} = call ptr @iris_list_slice(ptr {}, i64 {}, i64 {})", result.0, val(*list), val(*start), val(*end))?;
        }

        // Phase 59: Process / environment
        IrInstr::ProcessExit { code } => {
            writeln!(out, "  call void @exit(i32 {})", val(*code))?;
            writeln!(out, "  unreachable")?;
        }
        IrInstr::ProcessArgs { result } => {
            writeln!(out, "  %v{} = call ptr @iris_process_args()", result.0)?;
        }
        IrInstr::EnvVar { result, name } => {
            writeln!(out, "  %v{} = call ptr @iris_env_var(ptr {})", result.0, val(*name))?;
        }
        // Phase 61: Pattern matching helpers
        IrInstr::GetVariantTag { result, operand } => {
            writeln!(out, "  %v{} = call i64 @iris_get_variant_tag({})", result.0, val(*operand))?;
        }
        IrInstr::StrEq { result, lhs, rhs } => {
            writeln!(out, "  %v{} = call i1 @iris_str_eq(ptr {}, ptr {})", result.0, val(*lhs), val(*rhs))?;
        }
        // Phase 83: GC retain/release
        IrInstr::Retain { ptr } => {
            writeln!(out, "  call void @iris_retain(ptr {})", val(*ptr))?;
        }
        IrInstr::Release { ptr, .. } => {
            writeln!(out, "  call void @iris_release(ptr {})", val(*ptr))?;
        }
        // Phase 81: FFI extern calls
        IrInstr::CallExtern { result, name, args, .. } => {
            let arg_strs: Vec<String> = args.iter().map(|a| format!("ptr {}", val(*a))).collect();
            if let Some(r) = result {
                writeln!(out, "  %v{} = call ptr @{}({})", r.0, name, arg_strs.join(", "))?;
            } else {
                writeln!(out, "  call void @{}({})", name, arg_strs.join(", "))?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the LLVM name for a value: inline constant, function arg name, or %vN.
fn llvm_val(v: ValueId, consts: &HashMap<ValueId, String>, func: &IrFunction) -> String {
    // 1. Inline constants
    if let Some(c) = consts.get(&v) {
        return c.clone();
    }
    // 2. Function arguments — entry-block params have the original param name
    for param in &func.blocks()[0].params {
        if param.id == v {
            if let Some(name) = &param.name {
                return format!("%{}", name);
            }
        }
    }
    // 3. All other values
    format!("%v{}", v.0)
}

/// Formats a block label as "{name}{id}" (e.g. "entry0", "then1", "merge3").
fn block_label(name: Option<&str>, id: BlockId) -> String {
    format!("{}{}", name.unwrap_or("bb"), id.0)
}

/// Finds the label for a block by its id.
fn block_label_by_id(blocks: &[crate::ir::block::IrBlock], id: BlockId) -> String {
    blocks
        .iter()
        .find(|b| b.id == id)
        .map(|b| block_label(b.name.as_deref(), b.id))
        .unwrap_or_else(|| format!("bb{}", id.0))
}

/// Maps an `IrType` to its LLVM type string.
fn llvm_type_name(ty: &IrType) -> Result<String, CodegenError> {
    match ty {
        IrType::Scalar(DType::F32) => Ok("float".to_owned()),
        IrType::Scalar(DType::F64) => Ok("double".to_owned()),
        IrType::Scalar(DType::I32) => Ok("i32".to_owned()),
        IrType::Scalar(DType::I64) => Ok("i64".to_owned()),
        IrType::Scalar(DType::Bool) => Ok("i1".to_owned()),
        IrType::Scalar(DType::U8) => Ok("i8".to_owned()),
        IrType::Scalar(DType::I8) => Ok("i8".to_owned()),
        IrType::Scalar(DType::U32) => Ok("i32".to_owned()),
        IrType::Scalar(DType::U64) => Ok("i64".to_owned()),
        IrType::Scalar(DType::USize) => Ok("i64".to_owned()),
        IrType::Tensor { .. } => Ok("ptr".to_owned()),
        IrType::Struct { .. } => Ok("ptr".to_owned()),
        IrType::Enum { .. } => Ok("i64".to_owned()),
        IrType::Tuple(_) => Ok("ptr".to_owned()),
        IrType::Str => Ok("ptr".to_owned()),
        IrType::Array { .. } => Ok("ptr".to_owned()),
        IrType::Option(_) | IrType::ResultType(_, _) => Ok("ptr".to_owned()),
        IrType::Chan(_) | IrType::Atomic(_) | IrType::Mutex(_) | IrType::Grad(_) | IrType::Sparse(_) | IrType::List(_) | IrType::Map(_, _) => Ok("ptr".to_owned()),
        IrType::Fn { .. } | IrType::Infer => Err(CodegenError::Unsupported {
            backend: "llvm".into(),
            detail: format!("cannot lower type {} to LLVM", ty),
        }),
    }
}

/// Formats a float literal for LLVM IR (always includes a decimal point).
fn fmt_float(v: f64) -> String {
    let s = format!("{}", v);
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{}.0", s)
    }
}

/// Escapes a string for use as an LLVM IR string constant.
/// LLVM uses `\HH` hex escapes for special/non-printable bytes.
fn llvm_escape_string(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'"'  => out.push_str("\\22"),
            b'\\' => out.push_str("\\5C"),
            b'\n' => out.push_str("\\0A"),
            b'\r' => out.push_str("\\0D"),
            b'\t' => out.push_str("\\09"),
            0x20..=0x7E => out.push(b as char),
            other => out.push_str(&format!("\\{:02X}", other)),
        }
    }
    out
}

/// Emits `declare` statements for all iris runtime helper functions used by
/// the LLVM emitter, plus common C/system library functions.
fn emit_iris_runtime_declares(out: &mut String) -> Result<(), CodegenError> {
    let declares: &[&str] = &[
        // I/O
        "declare void @iris_print(ptr)",
        "declare void @iris_panic(ptr)",
        "declare ptr @iris_read_line()",
        "declare i64 @iris_read_i64()",
        "declare double @iris_read_f64()",
        // String ops
        "declare i64 @iris_str_len(ptr)",
        "declare ptr @iris_str_concat(ptr, ptr)",
        "declare i1 @iris_str_contains(ptr, ptr)",
        "declare i1 @iris_str_starts_with(ptr, ptr)",
        "declare i1 @iris_str_ends_with(ptr, ptr)",
        "declare ptr @iris_str_to_upper(ptr)",
        "declare ptr @iris_str_to_lower(ptr)",
        "declare ptr @iris_str_trim(ptr)",
        "declare ptr @iris_str_repeat(ptr, i64)",
        "declare ptr @iris_value_to_str(ptr)",
        "declare ptr @iris_parse_i64(ptr)",
        "declare ptr @iris_parse_f64(ptr)",
        "declare i64 @iris_str_index(ptr, i64)",
        "declare ptr @iris_str_slice(ptr, i64, i64)",
        "declare ptr @iris_str_find(ptr, ptr)",
        "declare ptr @iris_str_replace(ptr, ptr, ptr)",
        // Option / Result
        "declare ptr @iris_make_some()",
        "declare ptr @iris_make_none()",
        "declare i1 @iris_is_some(ptr)",
        "declare ptr @iris_option_unwrap(ptr)",
        "declare ptr @iris_make_ok()",
        "declare ptr @iris_make_err()",
        "declare i1 @iris_is_ok(ptr)",
        "declare ptr @iris_result_unwrap(ptr)",
        "declare ptr @iris_result_unwrap_err(ptr)",
        // Collections
        "declare ptr @iris_list_new()",
        "declare void @iris_list_push(ptr, ptr)",
        "declare i64 @iris_list_len(ptr)",
        "declare ptr @iris_list_get(ptr, i64)",
        "declare void @iris_list_set(ptr, i64, ptr)",
        "declare ptr @iris_list_pop(ptr)",
        "declare ptr @iris_map_new()",
        "declare void @iris_map_set(ptr, ptr, ptr)",
        "declare ptr @iris_map_get(ptr, ptr)",
        "declare i1 @iris_map_contains(ptr, ptr)",
        "declare void @iris_map_remove(ptr, ptr)",
        "declare i64 @iris_map_len(ptr)",
        // Arrays / Tensors
        "declare ptr @iris_alloc_array()",
        "declare ptr @iris_array_load(ptr, i64)",
        "declare void @iris_array_store(ptr, i64, ptr)",
        "declare ptr @iris_tensor_op()",
        "declare ptr @iris_tensor_load(ptr, ...)",
        "declare void @iris_tensor_store(ptr, ...)",
        // Channels / Concurrency
        "declare ptr @iris_chan_new()",
        "declare void @iris_chan_send(ptr, ptr)",
        "declare ptr @iris_chan_recv(ptr)",
        "declare void @iris_spawn_fn(ptr)",
        "declare void @iris_par_for(ptr, i64, i64)",
        // Atomics / Mutex
        "declare ptr @iris_atomic_new()",
        "declare ptr @iris_atomic_load(ptr)",
        "declare void @iris_atomic_store(ptr, ptr)",
        "declare ptr @iris_atomic_add(ptr, ptr)",
        "declare ptr @iris_mutex_new()",
        "declare ptr @iris_mutex_lock(ptr)",
        "declare void @iris_mutex_unlock(ptr)",
        // Structs / Tuples / Closures
        "declare ptr @iris_make_struct(...)",
        "declare ptr @iris_get_field(ptr, i32)",
        "declare ptr @iris_make_tuple(...)",
        "declare ptr @iris_get_element(ptr, i32)",
        "declare ptr @iris_make_closure()",
        "declare ptr @iris_call_closure(ptr, ...)",
        "declare void @iris_call_closure_void(ptr, ...)",
        // Grad / Sparse
        "declare ptr @iris_make_grad()",
        "declare ptr @iris_grad_value()",
        "declare ptr @iris_grad_tangent()",
        "declare ptr @iris_sparsify()",
        "declare ptr @iris_densify()",
        // Math helpers (non-llvm-intrinsic)
        "declare i64 @iris_pow_i64(i64, i64)",
        "declare i64 @iris_min_i64(i64, i64)",
        "declare i64 @iris_max_i64(i64, i64)",
        "declare i64 @iris_abs_i64(i64)",
        "declare double @iris_sign_f64(double)",
        "declare double @tan(double)",
    ];
    for decl in declares {
        writeln!(out, "{}", decl)?;
    }
    writeln!(out)?;
    Ok(())
}
