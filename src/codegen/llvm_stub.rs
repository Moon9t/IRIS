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
pub fn emit_llvm_stub(module: &IrModule) -> Result<String, CodegenError> {
    let mut out = String::new();
    writeln!(out, "; IRIS LLVM — scalar arithmetic + control flow")?;
    writeln!(
        out,
        "; Tensor types are lowered to opaque ptr (LLVM 15+ style)."
    )?;
    writeln!(out, "; TensorOp/Call are emitted as opaque extern calls.\n")?;

    for func in module.functions() {
        let ret = llvm_type_name(&func.return_ty)?;

        let params: Result<Vec<String>, CodegenError> = func
            .params
            .iter()
            .map(|p| Ok(format!("{} %{}", llvm_type_name(&p.ty)?, p.name)))
            .collect();
        let params = params?.join(", ");

        writeln!(out, "define {} @{}({}) {{", ret, func.name, params)?;
        emit_llvm_body(func, &mut out)?;
        writeln!(out, "}}\n")?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Body emitter
// ---------------------------------------------------------------------------

fn emit_llvm_body(func: &IrFunction, out: &mut String) -> Result<(), CodegenError> {
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
            emit_llvm_instr(instr, &consts, func, &mut gep_counter, out)?;
        }
    }
    Ok(())
}

fn emit_llvm_instr(
    instr: &IrInstr,
    consts: &HashMap<ValueId, String>,
    func: &IrFunction,
    gep_counter: &mut u32,
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
            writeln!(out, "  %v{} = add i64 0, {}", result.0, variant_idx)?;
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

        // String ops: emit as opaque runtime calls.
        IrInstr::ConstStr { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_const_str()", result.0)?;
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

        IrInstr::Print { operand } => {
            writeln!(out, "  call void @iris_print(ptr {})", val(*operand))?;
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
        IrType::Tensor { .. } => Ok("ptr".to_owned()),
        IrType::Struct { .. } => Ok("ptr".to_owned()),
        IrType::Enum { .. } => Ok("i64".to_owned()),
        IrType::Tuple(_) => Ok("ptr".to_owned()),
        IrType::Str => Ok("ptr".to_owned()),
        IrType::Array { .. } => Ok("ptr".to_owned()),
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
