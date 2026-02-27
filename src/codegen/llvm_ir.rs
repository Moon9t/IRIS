//! Complete LLVM IR backend for IRIS.
//!
//! Phase 49 — enhanced over the text stub with:
//!
//! - Named LLVM struct type declarations (`%Name = type { ... }`) at module level.
//! - Fixed-size scalar arrays lowered to `[N x T]` LLVM array types + `alloca`.
//! - Properly typed user-defined function call signatures (not opaque `ptr`).
//! - GEP-based struct field and array element access for inline scalar paths.
//! - `nounwind` / `willreturn` function attributes on pure functions.
//! - Enum tag comparisons lowered to `icmp eq i64`.
//! - Typed `AllocArray` → `alloca [N x T]` with initialiser stores for scalar elems.

use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use crate::error::CodegenError;
use crate::ir::block::BlockId;
use crate::ir::function::IrFunction;
use crate::ir::instr::{BinOp, IrInstr, ScalarUnaryOp};
use crate::ir::module::IrModule;
use crate::ir::types::{DType, IrType};
use crate::ir::value::ValueId;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Emits complete LLVM IR for all functions in the module.
///
/// Improvements over `emit_llvm_stub`:
/// 1. Named struct type declarations.
/// 2. Fixed scalar arrays use `[N x T]` + `alloca`.
/// 3. User function calls use real typed signatures.
/// 4. GEP-based inline struct/array access.
pub fn emit_llvm_ir(module: &IrModule) -> Result<String, CodegenError> {
    let mut out = String::new();

    // ── Header ────────────────────────────────────────────────────────────
    writeln!(out, "; IRIS Complete LLVM IR — phase 49")?;
    writeln!(out, "; Struct/array types lowered, typed calls, alloca for fixed arrays.\n")?;
    writeln!(
        out,
        "target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128\""
    )?;
    writeln!(out, "target triple = \"x86_64-unknown-linux-gnu\"\n")?;

    // ── Named struct type declarations ────────────────────────────────────
    // Collect struct names in stable order for deterministic output.
    let mut struct_names: Vec<&str> = module.struct_defs.keys().map(|s| s.as_str()).collect();
    struct_names.sort();
    for name in &struct_names {
        let fields = &module.struct_defs[*name];
        let field_tys: Result<Vec<String>, CodegenError> = fields
            .iter()
            .map(|(_, ft)| llvm_type_complete(ft))
            .collect();
        writeln!(out, "%{} = type {{ {} }}", name, field_tys?.join(", "))?;
    }
    if !struct_names.is_empty() {
        writeln!(out)?;
    }

    // ── Collect global string constants ───────────────────────────────────
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
    for (idx, content) in str_vec.iter().enumerate() {
        let escaped = llvm_escape_string(content);
        let len = content.len() + 1;
        writeln!(
            out,
            "@.str.{} = private unnamed_addr constant [{} x i8] c\"{}\\00\", align 1",
            idx, len, escaped
        )?;
    }
    if !str_vec.is_empty() {
        writeln!(out)?;
    }

    // ── Runtime declarations ──────────────────────────────────────────────
    emit_runtime_declares(&mut out)?;

    // ── Build function signature map for typed calls ──────────────────────
    // Maps function name → (return_type_string, Vec<param_type_string>)
    let mut fn_sigs: HashMap<String, (String, Vec<String>)> = HashMap::new();
    for func in module.functions() {
        let ret_s = llvm_type_complete(&func.return_ty).unwrap_or_else(|_| "ptr".to_owned());
        let param_ss: Vec<String> = func
            .params
            .iter()
            .map(|p| llvm_type_complete(&p.ty).unwrap_or_else(|_| "ptr".to_owned()))
            .collect();
        fn_sigs.insert(func.name.clone(), (ret_s, param_ss));
    }

    // ── Function definitions ──────────────────────────────────────────────
    for func in module.functions() {
        emit_function_ir(func, &str_table, &fn_sigs, module, &mut out)?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Function emission
// ---------------------------------------------------------------------------

fn emit_function_ir(
    func: &IrFunction,
    str_table: &HashMap<String, usize>,
    fn_sigs: &HashMap<String, (String, Vec<String>)>,
    module: &IrModule,
    out: &mut String,
) -> Result<(), CodegenError> {
    let ret = llvm_type_complete(&func.return_ty)?;
    let params: Result<Vec<String>, CodegenError> = func
        .params
        .iter()
        .map(|p| Ok(format!("{} %{}", llvm_type_complete(&p.ty)?, p.name)))
        .collect();

    // Determine if pure (no side-effecting instructions) for attributes.
    let is_pure = func.blocks().iter().all(|b| {
        b.instrs.iter().all(|i| !is_side_effecting(i))
    });
    let attrs = if is_pure { " nounwind willreturn" } else { "" };

    writeln!(out, "define {} @{}({}){} {{", ret, func.name, params?.join(", "), attrs)?;
    emit_function_body(func, str_table, fn_sigs, module, out)?;
    writeln!(out, "}}\n")?;
    Ok(())
}

fn emit_function_body(
    func: &IrFunction,
    str_table: &HashMap<String, usize>,
    fn_sigs: &HashMap<String, (String, Vec<String>)>,
    module: &IrModule,
    out: &mut String,
) -> Result<(), CodegenError> {
    // Sub-pass A: inline constants.
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

    // Sub-pass B: phi sources for block-param SSA → LLVM phi conversion.
    let mut phi_src: HashMap<(BlockId, usize), Vec<(BlockId, ValueId)>> = HashMap::new();
    for block in func.blocks() {
        for instr in &block.instrs {
            match instr {
                IrInstr::Br { target, args } => {
                    for (i, v) in args.iter().enumerate() {
                        phi_src.entry((*target, i)).or_default().push((block.id, *v));
                    }
                }
                IrInstr::CondBr { then_block, then_args, else_block, else_args, .. } => {
                    for (i, v) in then_args.iter().enumerate() {
                        phi_src.entry((*then_block, i)).or_default().push((block.id, *v));
                    }
                    for (i, v) in else_args.iter().enumerate() {
                        phi_src.entry((*else_block, i)).or_default().push((block.id, *v));
                    }
                }
                _ => {}
            }
        }
    }

    // Sub-pass C: collect AllocArray instructions that use scalar elem types.
    // These will be emitted as `alloca` at the entry block.
    let mut scalar_arrays: HashSet<ValueId> = HashSet::new();
    for block in func.blocks() {
        for instr in &block.instrs {
            if let IrInstr::AllocArray { result, elem_ty, .. } = instr {
                if is_scalar_type(elem_ty) {
                    scalar_arrays.insert(*result);
                }
            }
        }
    }

    let entry_id = func.blocks()[0].id;
    let mut gep_counter: u32 = 0;

    // ── Alloca preamble (entry block) ─────────────────────────────────────
    // Emit all scalar array allocas at the top of the entry block.
    let mut alloca_emitted: HashSet<ValueId> = HashSet::new();
    for block in func.blocks() {
        for instr in &block.instrs {
            if let IrInstr::AllocArray { result, elem_ty, size, .. } = instr {
                if scalar_arrays.contains(result) && !alloca_emitted.contains(result) {
                    // Will be emitted inline at the alloca instruction site.
                    alloca_emitted.insert(*result);
                    let _ = (elem_ty, size); // used below in emit_instr_ir
                }
            }
        }
    }

    for block in func.blocks() {
        let blabel = block_label(block.name.as_deref(), block.id);
        writeln!(out, "{}:", blabel)?;

        // Phi nodes for non-entry blocks.
        if block.id != entry_id {
            for (i, param) in block.params.iter().enumerate() {
                let ty_s = llvm_type_complete(&param.ty)?;
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

        for instr in &block.instrs {
            emit_instr_ir(
                instr,
                &consts,
                func,
                fn_sigs,
                module,
                &scalar_arrays,
                &mut gep_counter,
                str_table,
                out,
            )?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Boxing helper
// ---------------------------------------------------------------------------

/// If `value_ty` is a scalar type, emit a boxing call and return the resulting
/// `%boxN` ptr name. Otherwise, the value is already a ptr — return it unchanged.
fn box_to_ptr(
    out: &mut String,
    value_str: &str,
    value_ty: Option<&IrType>,
    counter: &mut u32,
) -> Result<String, CodegenError> {
    let idx = *counter;
    match value_ty {
        Some(IrType::Scalar(DType::I64)) => {
            *counter += 1;
            let boxed = format!("%box{}", idx);
            writeln!(out, "  {} = call ptr @iris_box_i64(i64 {})", boxed, value_str)?;
            Ok(boxed)
        }
        Some(IrType::Scalar(DType::I32)) => {
            *counter += 1;
            let boxed = format!("%box{}", idx);
            writeln!(out, "  {} = call ptr @iris_box_i32(i32 {})", boxed, value_str)?;
            Ok(boxed)
        }
        Some(IrType::Scalar(DType::F64)) => {
            *counter += 1;
            let boxed = format!("%box{}", idx);
            writeln!(out, "  {} = call ptr @iris_box_f64(double {})", boxed, value_str)?;
            Ok(boxed)
        }
        Some(IrType::Scalar(DType::F32)) => {
            *counter += 1;
            let boxed = format!("%box{}", idx);
            writeln!(out, "  {} = call ptr @iris_box_f32(float {})", boxed, value_str)?;
            Ok(boxed)
        }
        Some(IrType::Scalar(DType::Bool)) => {
            *counter += 1;
            let boxed = format!("%box{}", idx);
            writeln!(out, "  {} = call ptr @iris_box_bool(i1 {})", boxed, value_str)?;
            Ok(boxed)
        }
        _ => {
            // Not a scalar — already a ptr. No boxing needed.
            Ok(value_str.to_owned())
        }
    }
}

// ---------------------------------------------------------------------------
// Instruction emission
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn emit_instr_ir(
    instr: &IrInstr,
    consts: &HashMap<ValueId, String>,
    func: &IrFunction,
    fn_sigs: &HashMap<String, (String, Vec<String>)>,
    _module: &IrModule,
    scalar_arrays: &HashSet<ValueId>,
    gep_counter: &mut u32,
    str_table: &HashMap<String, usize>,
    out: &mut String,
) -> Result<(), CodegenError> {
    let val = |v: ValueId| llvm_val(v, consts, func);

    match instr {
        // Constants are inlined at use sites.
        IrInstr::ConstFloat { .. } | IrInstr::ConstInt { .. } | IrInstr::ConstBool { .. } => {}

        IrInstr::BinOp { result, op, lhs, rhs, ty } => {
            let lv = val(*lhs);
            let rv = val(*rhs);
            let operand_ty = func.value_type(*lhs).unwrap_or(ty);
            let ty_s = llvm_type_complete(operand_ty)?;
            let is_float = matches!(operand_ty, IrType::Scalar(DType::F32 | DType::F64));
            let llvm_op = match (op, is_float) {
                (BinOp::Add, true) => format!("fadd {} {}, {}", ty_s, lv, rv),
                (BinOp::Sub, true) => format!("fsub {} {}, {}", ty_s, lv, rv),
                (BinOp::Mul, true) => format!("fmul {} {}, {}", ty_s, lv, rv),
                (BinOp::Div, true) => format!("fdiv {} {}, {}", ty_s, lv, rv),
                (BinOp::Add, false) => format!("add nsw {} {}, {}", ty_s, lv, rv),
                (BinOp::Sub, false) => format!("sub nsw {} {}, {}", ty_s, lv, rv),
                (BinOp::Mul, false) => format!("mul nsw {} {}, {}", ty_s, lv, rv),
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
                (BinOp::Pow, true) => format!("call {} @llvm.pow.f64({} {}, {} {})", ty_s, ty_s, lv, ty_s, rv),
                (BinOp::Pow, false) => format!("call i64 @iris_pow_i64(i64 {}, i64 {})", lv, rv),
                (BinOp::Min, true) => format!("call {} @llvm.minnum.f64({} {}, {} {})", ty_s, ty_s, lv, ty_s, rv),
                (BinOp::Min, false) => format!("call i64 @iris_min_i64(i64 {}, i64 {})", lv, rv),
                (BinOp::Max, true) => format!("call {} @llvm.maxnum.f64({} {}, {} {})", ty_s, ty_s, lv, ty_s, rv),
                (BinOp::Max, false) => format!("call i64 @iris_max_i64(i64 {}, i64 {})", lv, rv),
                (BinOp::BitAnd, false) => format!("and {} {}, {}", ty_s, lv, rv),
                (BinOp::BitOr, false) => format!("or {} {}, {}", ty_s, lv, rv),
                (BinOp::BitXor, false) => format!("xor {} {}, {}", ty_s, lv, rv),
                (BinOp::Shl, false) => format!("shl {} {}, {}", ty_s, lv, rv),
                (BinOp::Shr, false) => format!("ashr {} {}, {}", ty_s, lv, rv),
                _ => format!("call i64 @iris_unsupported_binop()"),
            };
            writeln!(out, "  %v{} = {}", result.0, llvm_op)?;
        }

        IrInstr::UnaryOp { result, op, operand, ty } => {
            let ov = val(*operand);
            let ty_s = llvm_type_complete(ty)?;
            let is_float = matches!(ty, IrType::Scalar(DType::F32 | DType::F64));
            match op {
                ScalarUnaryOp::Neg if is_float => {
                    writeln!(out, "  %v{} = fneg {} {}", result.0, ty_s, ov)?;
                }
                ScalarUnaryOp::Neg => {
                    writeln!(out, "  %v{} = sub nsw {} 0, {}", result.0, ty_s, ov)?;
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
                    writeln!(out, "  %v{} = call i64 @iris_abs_i64(i64 {})", result.0, ov)?;
                }
                ScalarUnaryOp::Floor => {
                    writeln!(out, "  %v{} = call {} @llvm.floor.f64({} {})", result.0, ty_s, ty_s, ov)?;
                }
                ScalarUnaryOp::Ceil => {
                    writeln!(out, "  %v{} = call {} @llvm.ceil.f64({} {})", result.0, ty_s, ty_s, ov)?;
                }
                ScalarUnaryOp::BitNot => {
                    writeln!(out, "  %v{} = xor {} {}, -1", result.0, ty_s, ov)?;
                }
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

        IrInstr::Cast { result, operand, from_ty, to_ty } => {
            let ov = val(*operand);
            let from_s = llvm_type_complete(from_ty)?;
            let to_s = llvm_type_complete(to_ty)?;
            let is_from_float = matches!(from_ty, IrType::Scalar(DType::F32 | DType::F64));
            let is_to_float = matches!(to_ty, IrType::Scalar(DType::F32 | DType::F64));
            let is_from_int = matches!(from_ty, IrType::Scalar(DType::I32 | DType::I64));
            let is_to_int = matches!(to_ty, IrType::Scalar(DType::I32 | DType::I64));
            let is_from_f64 = matches!(from_ty, IrType::Scalar(DType::F64));
            let is_to_f64 = matches!(to_ty, IrType::Scalar(DType::F64));
            let is_from_i64 = matches!(from_ty, IrType::Scalar(DType::I64));
            let is_to_i64 = matches!(to_ty, IrType::Scalar(DType::I64));
            if from_ty == to_ty {
                writeln!(out, "  %v{} = bitcast {} {} to {}", result.0, from_s, ov, to_s)?;
            } else if is_from_float && is_to_int {
                writeln!(out, "  %v{} = fptosi {} {} to {}", result.0, from_s, ov, to_s)?;
            } else if is_from_int && is_to_float {
                writeln!(out, "  %v{} = sitofp {} {} to {}", result.0, from_s, ov, to_s)?;
            } else if is_from_float && is_to_float {
                if !is_from_f64 && is_to_f64 {
                    writeln!(out, "  %v{} = fpext {} {} to {}", result.0, from_s, ov, to_s)?;
                } else {
                    writeln!(out, "  %v{} = fptrunc {} {} to {}", result.0, from_s, ov, to_s)?;
                }
            } else if is_from_int && is_to_int {
                if !is_from_i64 && is_to_i64 {
                    writeln!(out, "  %v{} = sext {} {} to {}", result.0, from_s, ov, to_s)?;
                } else {
                    writeln!(out, "  %v{} = trunc {} {} to {}", result.0, from_s, ov, to_s)?;
                }
            } else {
                writeln!(out, "  %v{} = bitcast {} {} to {}", result.0, from_s, ov, to_s)?;
            }
        }

        IrInstr::Return { values } => {
            if values.is_empty() {
                writeln!(out, "  ret void")?;
            } else {
                let v = val(values[0]);
                let ty_s = llvm_type_complete(&func.return_ty)?;
                writeln!(out, "  ret {} {}", ty_s, v)?;
            }
        }

        IrInstr::Br { target, .. } => {
            let lbl = block_label_by_id(func.blocks(), *target);
            writeln!(out, "  br label %{}", lbl)?;
        }

        IrInstr::CondBr { cond, then_block, else_block, .. } => {
            let cv = val(*cond);
            let tl = block_label_by_id(func.blocks(), *then_block);
            let el = block_label_by_id(func.blocks(), *else_block);
            writeln!(out, "  br i1 {}, label %{}, label %{}", cv, tl, el)?;
        }

        // ── Typed user-defined function calls ─────────────────────────────
        IrInstr::Call { result, callee, args, result_ty } => {
            if let Some((ret_s, param_ss)) = fn_sigs.get(callee) {
                // Build typed arg list.
                let typed_args: Vec<String> = args
                    .iter()
                    .zip(param_ss.iter())
                    .map(|(a, ty_s)| format!("{} {}", ty_s, val(*a)))
                    .collect();
                if let Some(r) = result {
                    writeln!(
                        out,
                        "  %v{} = call {} @{}({})",
                        r.0, ret_s, callee, typed_args.join(", ")
                    )?;
                } else {
                    writeln!(out, "  call {} @{}({})", ret_s, callee, typed_args.join(", "))?;
                }
            } else {
                // Unknown callee (runtime intrinsic) — opaque call.
                let ret_ty_s = result_ty
                    .as_ref()
                    .and_then(|t| llvm_type_complete(t).ok())
                    .unwrap_or_else(|| "ptr".to_owned());
                let args_str: Vec<String> = args.iter().map(|a| format!("ptr {}", val(*a))).collect();
                if let Some(r) = result {
                    writeln!(
                        out,
                        "  %v{} = call {} @{}({})",
                        r.0, ret_ty_s, callee, args_str.join(", ")
                    )?;
                } else {
                    writeln!(out, "  call void @{}({})", callee, args_str.join(", "))?;
                }
            }
        }

        // ── Struct ops ─────────────────────────────────────────────────────
        IrInstr::MakeStruct { result, fields, result_ty } => {
            if let IrType::Struct { name, fields: field_tys } = result_ty {
                // Inline struct construction via alloca + GEP stores.
                let struct_ty = format!("%{}", name);
                writeln!(out, "  %struct_alloc{} = alloca {}, align 8", result.0, struct_ty)?;
                for (i, (fv, (_, fty))) in fields.iter().zip(field_tys.iter()).enumerate() {
                    let fty_s = llvm_type_complete(fty)?;
                    let gep_name = format!("%sgep{}_{}", result.0, i);
                    writeln!(
                        out,
                        "  {} = getelementptr inbounds {}, ptr %struct_alloc{}, i32 0, i32 {}",
                        gep_name, struct_ty, result.0, i
                    )?;
                    writeln!(out, "  store {} {}, ptr {}, align 8", fty_s, val(*fv), gep_name)?;
                }
                writeln!(out, "  %v{} = load ptr, ptr %struct_alloc{}", result.0, result.0)?;
            } else {
                let args_str: Vec<String> = fields.iter().map(|f| format!("ptr {}", val(*f))).collect();
                writeln!(out, "  %v{} = call ptr @iris_make_struct({})", result.0, args_str.join(", "))?;
            }
        }

        IrInstr::GetField { result, base, field_index, result_ty } => {
            let bv = val(*base);
            // Try to determine struct type from value type.
            let base_ty = func.value_type(*base);
            if let Some(IrType::Struct { name, fields: field_tys }) = base_ty {
                let struct_ty = format!("%{}", name);
                let fty_s = llvm_type_complete(result_ty)?;
                let gep_name = format!("%fgep{}_{}", result.0, field_index);
                writeln!(
                    out,
                    "  {} = getelementptr inbounds {}, ptr {}, i32 0, i32 {}",
                    gep_name, struct_ty, bv, field_index
                )?;
                writeln!(out, "  %v{} = load {}, ptr {}, align 8", result.0, fty_s, gep_name)?;
                let _ = field_tys; // suppress unused warning
            } else {
                let ty_s = llvm_type_complete(result_ty).unwrap_or_else(|_| "ptr".to_owned());
                writeln!(
                    out,
                    "  %v{} = call {} @iris_get_field(ptr {}, i32 {})",
                    result.0, ty_s, bv, field_index
                )?;
            }
        }

        // ── Enum ops ───────────────────────────────────────────────────────
        IrInstr::MakeVariant { result, variant_idx, .. } => {
            writeln!(out, "  %v{} = add i64 0, {}", result.0, variant_idx)?;
        }

        IrInstr::SwitchVariant { scrutinee, arms, default_block } => {
            let sv = val(*scrutinee);
            let blocks = func.blocks();
            let default = default_block
                .map(|bb| format!("label %{}", block_label_by_id(blocks, bb)))
                .unwrap_or_else(|| {
                    arms.first()
                        .map(|(_, bb)| format!("label %{}", block_label_by_id(blocks, *bb)))
                        .unwrap_or_else(|| "label %unreachable".to_owned())
                });
            write!(out, "  switch i64 {}, {} [", sv, default)?;
            for (idx, bb) in arms {
                write!(out, " i64 {}, label %{}", idx, block_label_by_id(blocks, *bb))?;
            }
            writeln!(out, " ]")?;
        }

        IrInstr::ExtractVariantField { result, operand, field_idx, .. } => {
            writeln!(
                out,
                "  %v{} = call i64 @iris_extract_variant_field({}, i64 {})",
                result.0, val(*operand), field_idx
            )?;
        }

        // ── Tuple ops ──────────────────────────────────────────────────────
        IrInstr::MakeTuple { result, elements, .. } => {
            let args_str: Vec<String> = elements.iter().map(|e| format!("ptr {}", val(*e))).collect();
            writeln!(out, "  %v{} = call ptr @iris_make_tuple({})", result.0, args_str.join(", "))?;
        }

        IrInstr::GetElement { result, base, index, result_ty } => {
            let ty_s = llvm_type_complete(result_ty).unwrap_or_else(|_| "ptr".to_owned());
            writeln!(
                out,
                "  %v{} = call {} @iris_get_element(ptr {}, i32 {})",
                result.0, ty_s, val(*base), index
            )?;
        }

        // ── Array ops ─────────────────────────────────────────────────────
        // Scalar-element fixed arrays: use alloca + GEP.
        IrInstr::AllocArray { result, elem_ty, size, init } => {
            if scalar_arrays.contains(result) {
                let ety_s = llvm_type_complete(elem_ty)?;
                writeln!(
                    out,
                    "  %v{} = alloca [{} x {}], align 16",
                    result.0, size, ety_s
                )?;
                // Emit stores for initial values.
                for (i, iv) in init.iter().enumerate() {
                    let gep = format!("%ainit{}_{}", result.0, i);
                    writeln!(
                        out,
                        "  {} = getelementptr inbounds [{} x {}], ptr %v{}, i64 0, i64 {}",
                        gep, size, ety_s, result.0, i
                    )?;
                    writeln!(out, "  store {} {}, ptr {}, align {}", ety_s, val(*iv), gep, ety_align(elem_ty))?;
                }
            } else {
                writeln!(out, "  %v{} = call ptr @iris_alloc_array()", result.0)?;
            }
        }

        IrInstr::ArrayLoad { result, array, index, elem_ty } => {
            if scalar_arrays.contains(array) {
                let ety_s = llvm_type_complete(elem_ty)?;
                // Need to know the array size for GEP type — look up from alloca.
                let arr_size = find_alloc_size(func, *array);
                let gep = format!("%agep{}_{}", result.0, gep_counter);
                *gep_counter += 1;
                if let Some(sz) = arr_size {
                    writeln!(
                        out,
                        "  {} = getelementptr inbounds [{} x {}], ptr %v{}, i64 0, i64 {}",
                        gep, sz, ety_s, array.0, val(*index)
                    )?;
                } else {
                    writeln!(
                        out,
                        "  {} = getelementptr inbounds {}, ptr %v{}, i64 {}",
                        gep, ety_s, array.0, val(*index)
                    )?;
                }
                writeln!(out, "  %v{} = load {}, ptr {}, align {}", result.0, ety_s, gep, ety_align(elem_ty))?;
            } else {
                let ety_s = llvm_type_complete(elem_ty).unwrap_or_else(|_| "i64".to_owned());
                writeln!(
                    out,
                    "  %v{} = call {} @iris_array_load(ptr {}, i64 {})",
                    result.0, ety_s, val(*array), val(*index)
                )?;
            }
        }

        IrInstr::ArrayStore { array, index, value } => {
            if scalar_arrays.contains(array) {
                let vty = func.value_type(*value);
                let ety_s = vty
                    .and_then(|t| llvm_type_complete(t).ok())
                    .unwrap_or_else(|| "i64".to_owned());
                let arr_size = find_alloc_size(func, *array);
                let gep = format!("%asgep_{}", gep_counter);
                *gep_counter += 1;
                if let Some(sz) = arr_size {
                    writeln!(
                        out,
                        "  {} = getelementptr inbounds [{} x {}], ptr %v{}, i64 0, i64 {}",
                        gep, sz, ety_s, array.0, val(*index)
                    )?;
                } else {
                    writeln!(
                        out,
                        "  {} = getelementptr inbounds {}, ptr %v{}, i64 {}",
                        gep, ety_s, array.0, val(*index)
                    )?;
                }
                writeln!(out, "  store {} {}, ptr {}", ety_s, val(*value), gep)?;
            } else {
                writeln!(
                    out,
                    "  call void @iris_array_store(ptr {}, i64 {}, ptr {})",
                    val(*array), val(*index), val(*value)
                )?;
            }
        }

        // ── Memory / Tensor ops ────────────────────────────────────────────
        IrInstr::TensorOp { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_tensor_op()", result.0)?;
        }

        IrInstr::Load { result, tensor, indices, result_ty } => {
            let tv = val(*tensor);
            let ty_s = llvm_type_complete(result_ty)?;
            match indices.as_slice() {
                [] => {
                    writeln!(out, "  %v{} = load {}, ptr {}", result.0, ty_s, tv)?;
                }
                [idx] => {
                    let gep = format!("%gep{}", gep_counter);
                    *gep_counter += 1;
                    writeln!(out, "  {} = getelementptr {}, ptr {}, i64 {}", gep, ty_s, tv, val(*idx))?;
                    writeln!(out, "  %v{} = load {}, ptr {}", result.0, ty_s, gep)?;
                }
                _ => {
                    let mut args = vec![format!("ptr {}", tv)];
                    for idx in indices {
                        args.push(format!("i64 {}", val(*idx)));
                    }
                    writeln!(out, "  %v{} = call {} @iris_tensor_load({})", result.0, ty_s, args.join(", "))?;
                }
            }
        }

        IrInstr::Store { tensor, indices, value } => {
            let tv = val(*tensor);
            let vv = val(*value);
            let ty_s = func
                .value_type(*value)
                .and_then(|ty| llvm_type_complete(ty).ok())
                .unwrap_or_else(|| "ptr".to_owned());
            match indices.as_slice() {
                [] => {
                    writeln!(out, "  store {} {}, ptr {}", ty_s, vv, tv)?;
                }
                [idx] => {
                    let gep = format!("%gep{}", gep_counter);
                    *gep_counter += 1;
                    writeln!(out, "  {} = getelementptr {}, ptr {}, i64 {}", gep, ty_s, tv, val(*idx))?;
                    writeln!(out, "  store {} {}, ptr {}", ty_s, vv, gep)?;
                }
                _ => {
                    let mut args = vec![format!("ptr {}", tv), format!("{} {}", ty_s, vv)];
                    for idx in indices {
                        args.push(format!("i64 {}", val(*idx)));
                    }
                    writeln!(out, "  call void @iris_tensor_store({})", args.join(", "))?;
                }
            }
        }

        // ── Concurrency ───────────────────────────────────────────────────
        IrInstr::ParFor { body_fn, start, end, .. } => {
            // Emit an OpenMP-compatible loop via iris_par_for runtime.
            // The body function is referenced by name.
            writeln!(
                out,
                "  call void @iris_par_for(ptr @{}, i64 {}, i64 {})",
                body_fn, val(*start), val(*end)
            )?;
        }

        IrInstr::ChanNew { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_chan_new()", result.0)?;
        }
        IrInstr::ChanSend { chan, value } => {
            let vv = val(*value);
            let vty = func.value_type(*value);
            let ptr_v = box_to_ptr(out, &vv, vty, gep_counter)?;
            writeln!(out, "  call void @iris_chan_send(ptr {}, ptr {})", val(*chan), ptr_v)?;
        }
        IrInstr::ChanRecv { result, chan, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_chan_recv(ptr {})", result.0, val(*chan))?;
        }
        IrInstr::Spawn { body_fn, .. } => {
            writeln!(out, "  call void @iris_spawn_fn(ptr @{})", body_fn)?;
        }

        // ── Atomics ───────────────────────────────────────────────────────
        IrInstr::AtomicNew { result, value, .. } => {
            let vv = val(*value);
            let vty = func.value_type(*value);
            let ptr_v = box_to_ptr(out, &vv, vty, gep_counter)?;
            writeln!(out, "  %v{} = call ptr @iris_atomic_new(ptr {})", result.0, ptr_v)?;
        }
        IrInstr::AtomicLoad { result, atomic, result_ty } => {
            if matches!(result_ty, IrType::Scalar(_)) {
                let ty_s = llvm_type_complete(result_ty)?;
                writeln!(out, "  %v{} = load atomic {} , ptr {} seq_cst, align 8", result.0, ty_s, val(*atomic))?;
            } else {
                writeln!(out, "  %v{} = call ptr @iris_atomic_load(ptr {})", result.0, val(*atomic))?;
            }
        }
        IrInstr::AtomicStore { atomic, value } => {
            let vty = func.value_type(*value);
            if let Some(ty) = vty {
                if matches!(ty, IrType::Scalar(_)) {
                    let ty_s = llvm_type_complete(ty)?;
                    writeln!(out, "  store atomic {} {}, ptr {} seq_cst, align 8", ty_s, val(*value), val(*atomic))?;
                } else {
                    writeln!(out, "  call void @iris_atomic_store(ptr {}, ptr {})", val(*atomic), val(*value))?;
                }
            } else {
                writeln!(out, "  call void @iris_atomic_store(ptr {}, ptr {})", val(*atomic), val(*value))?;
            }
        }
        IrInstr::AtomicAdd { result, atomic, value, result_ty } => {
            if matches!(result_ty, IrType::Scalar(DType::I32 | DType::I64)) {
                let ty_s = llvm_type_complete(result_ty)?;
                writeln!(
                    out,
                    "  %v{} = atomicrmw add ptr {}, {} {} seq_cst",
                    result.0, val(*atomic), ty_s, val(*value)
                )?;
            } else {
                writeln!(
                    out,
                    "  %v{} = call ptr @iris_atomic_add(ptr {}, ptr {})",
                    result.0, val(*atomic), val(*value)
                )?;
            }
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

        // ── Option / Result ────────────────────────────────────────────────
        IrInstr::MakeSome { result, value, .. } => {
            let vv = val(*value);
            let vty = func.value_type(*value);
            let ptr_v = box_to_ptr(out, &vv, vty, gep_counter)?;
            writeln!(out, "  %v{} = call ptr @iris_make_some(ptr {})", result.0, ptr_v)?;
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
        IrInstr::MakeOk { result, value, .. } => {
            let vv = val(*value);
            let vty = func.value_type(*value);
            let ptr_v = box_to_ptr(out, &vv, vty, gep_counter)?;
            writeln!(out, "  %v{} = call ptr @iris_make_ok(ptr {})", result.0, ptr_v)?;
        }
        IrInstr::MakeErr { result, value, .. } => {
            let vv = val(*value);
            let vty = func.value_type(*value);
            let ptr_v = box_to_ptr(out, &vv, vty, gep_counter)?;
            writeln!(out, "  %v{} = call ptr @iris_make_err(ptr {})", result.0, ptr_v)?;
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

        // ── Strings ───────────────────────────────────────────────────────
        IrInstr::ConstStr { result, value } => {
            if let Some(&idx) = str_table.get(value) {
                let len = value.len() + 1;
                writeln!(
                    out,
                    "  %v{} = getelementptr inbounds [{} x i8], ptr @.str.{}, i32 0, i32 0",
                    result.0, len, idx
                )?;
            } else {
                writeln!(out, "  %v{} = call ptr @iris_const_str()", result.0)?;
            }
        }
        IrInstr::StrLen { result, operand } => {
            writeln!(out, "  %v{} = call i64 @iris_str_len(ptr {})", result.0, val(*operand))?;
        }
        IrInstr::StrConcat { result, lhs, rhs } => {
            writeln!(out, "  %v{} = call ptr @iris_str_concat(ptr {}, ptr {})", result.0, val(*lhs), val(*rhs))?;
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

        // ── Collections ────────────────────────────────────────────────────
        IrInstr::ListNew { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_list_new()", result.0)?;
        }
        IrInstr::ListPush { list, value } => {
            let vv = val(*value);
            let vty = func.value_type(*value);
            let ptr_v = box_to_ptr(out, &vv, vty, gep_counter)?;
            writeln!(out, "  call void @iris_list_push(ptr {}, ptr {})", val(*list), ptr_v)?;
        }
        IrInstr::ListLen { result, list } => {
            writeln!(out, "  %v{} = call i64 @iris_list_len(ptr {})", result.0, val(*list))?;
        }
        IrInstr::ListGet { result, list, index, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_list_get(ptr {}, i64 {})", result.0, val(*list), val(*index))?;
        }
        IrInstr::ListSet { list, index, value } => {
            let vv = val(*value);
            let vty = func.value_type(*value);
            let ptr_v = box_to_ptr(out, &vv, vty, gep_counter)?;
            writeln!(out, "  call void @iris_list_set(ptr {}, i64 {}, ptr {})", val(*list), val(*index), ptr_v)?;
        }
        IrInstr::ListPop { result, list, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_list_pop(ptr {})", result.0, val(*list))?;
        }
        IrInstr::MapNew { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_map_new()", result.0)?;
        }
        IrInstr::MapSet { map, key, value } => {
            let vv = val(*value);
            let vty = func.value_type(*value);
            let ptr_v = box_to_ptr(out, &vv, vty, gep_counter)?;
            writeln!(out, "  call void @iris_map_set(ptr {}, ptr {}, ptr {})", val(*map), val(*key), ptr_v)?;
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

        // ── Closures ──────────────────────────────────────────────────────
        IrInstr::MakeClosure { result, fn_name, captures, .. } => {
            let mut args = vec![format!("ptr @{}", fn_name)];
            for c in captures {
                let cv = val(*c);
                let cty = func.value_type(*c);
                let ptr_c = box_to_ptr(out, &cv, cty, gep_counter)?;
                args.push(format!("ptr {}", ptr_c));
            }
            writeln!(out, "  %v{} = call ptr @iris_make_closure({})", result.0, args.join(", "))?;
        }
        IrInstr::CallClosure { result, closure, args, .. } => {
            let mut args_parts: Vec<String> = Vec::new();
            for a in args {
                let av = val(*a);
                let aty = func.value_type(*a);
                let ptr_a = box_to_ptr(out, &av, aty, gep_counter)?;
                args_parts.push(format!("ptr {}", ptr_a));
            }
            let args_str = args_parts.join(", ");
            if let Some(r) = result {
                writeln!(out, "  %v{} = call ptr @iris_call_closure(ptr {}, {})", r.0, val(*closure), args_str)?;
            } else {
                writeln!(out, "  call void @iris_call_closure_void(ptr {}, {})", val(*closure), args_str)?;
            }
        }

        // ── Grad / Sparse ─────────────────────────────────────────────────
        IrInstr::MakeGrad { result, value, tangent, .. } => {
            // value and tangent are f64 dual-number components.
            writeln!(
                out,
                "  %v{} = call ptr @iris_make_grad(double {}, double {})",
                result.0, val(*value), val(*tangent)
            )?;
        }
        IrInstr::GradValue { result, operand, .. } => {
            writeln!(out, "  %v{} = call double @iris_grad_value(ptr {})", result.0, val(*operand))?;
        }
        IrInstr::GradTangent { result, operand, .. } => {
            writeln!(out, "  %v{} = call double @iris_grad_tangent(ptr {})", result.0, val(*operand))?;
        }
        IrInstr::Sparsify { result, operand, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_sparsify(ptr {})", result.0, val(*operand))?;
        }
        IrInstr::Densify { result, operand, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_densify(ptr {})", result.0, val(*operand))?;
        }

        // ── I/O ────────────────────────────────────────────────────────────
        IrInstr::Print { operand } => {
            // Typed print: use specialised helper for scalars.
            let oty = func.value_type(*operand);
            match oty {
                Some(IrType::Scalar(DType::I64)) => {
                    writeln!(out, "  call void @iris_print_i64(i64 {})", val(*operand))?;
                }
                Some(IrType::Scalar(DType::I32)) => {
                    writeln!(out, "  call void @iris_print_i32(i32 {})", val(*operand))?;
                }
                Some(IrType::Scalar(DType::F64)) => {
                    writeln!(out, "  call void @iris_print_f64(double {})", val(*operand))?;
                }
                Some(IrType::Scalar(DType::F32)) => {
                    writeln!(out, "  call void @iris_print_f32(float {})", val(*operand))?;
                }
                Some(IrType::Scalar(DType::Bool)) => {
                    writeln!(out, "  call void @iris_print_bool(i1 {})", val(*operand))?;
                }
                _ => {
                    writeln!(out, "  call void @iris_print(ptr {})", val(*operand))?;
                }
            }
        }
        IrInstr::Panic { msg } => {
            writeln!(out, "  call void @iris_panic(ptr {})", val(*msg))?;
            writeln!(out, "  unreachable")?;
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
        IrInstr::ValueToStr { result, operand } => {
            let oty = func.value_type(*operand);
            match oty {
                Some(IrType::Scalar(DType::I64)) => {
                    writeln!(out, "  %v{} = call ptr @iris_i64_to_str(i64 {})", result.0, val(*operand))?;
                }
                Some(IrType::Scalar(DType::I32)) => {
                    writeln!(out, "  %v{} = call ptr @iris_i32_to_str(i32 {})", result.0, val(*operand))?;
                }
                Some(IrType::Scalar(DType::F64)) => {
                    writeln!(out, "  %v{} = call ptr @iris_f64_to_str(double {})", result.0, val(*operand))?;
                }
                Some(IrType::Scalar(DType::F32)) => {
                    writeln!(out, "  %v{} = call ptr @iris_f32_to_str(float {})", result.0, val(*operand))?;
                }
                Some(IrType::Scalar(DType::Bool)) => {
                    writeln!(out, "  %v{} = call ptr @iris_bool_to_str(i1 {})", result.0, val(*operand))?;
                }
                Some(IrType::Str) => {
                    writeln!(out, "  %v{} = call ptr @iris_str_to_str(ptr {})", result.0, val(*operand))?;
                }
                _ => {
                    writeln!(out, "  %v{} = call ptr @iris_value_to_str(ptr {})", result.0, val(*operand))?;
                }
            }
        }

        // ── Barrier ───────────────────────────────────────────────────────
        IrInstr::Barrier => {
            writeln!(out, "  call void @iris_barrier()")?;
        }

        // ── Phase 56: File I/O ─────────────────────────────────────────────
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

        // ── Phase 58: Extended collections ────────────────────────────────
        IrInstr::ListContains { result, list, value } => {
            let vv = val(*value);
            let vty = func.value_type(*value);
            let ptr_v = box_to_ptr(out, &vv, vty, gep_counter)?;
            writeln!(out, "  %v{} = call i1 @iris_list_contains(ptr {}, ptr {})", result.0, val(*list), ptr_v)?;
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

        // ── Phase 59: Process / environment ──────────────────────────────
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
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Complete LLVM type mapping — includes proper struct, array, and scalar types.
pub fn llvm_type_complete(ty: &IrType) -> Result<String, CodegenError> {
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
        // Named struct → pointer to named struct type.
        IrType::Struct { .. } => Ok("ptr".to_owned()), // pass by pointer
        // Enum → integer tag.
        IrType::Enum { .. } => Ok("i64".to_owned()),
        // Fixed scalar arrays → LLVM array type (via pointer for args).
        IrType::Array { .. } => Ok("ptr".to_owned()), // arrays passed as ptr to [N x T]
        IrType::Tuple(_) => Ok("ptr".to_owned()),
        IrType::Str => Ok("ptr".to_owned()),
        IrType::Tensor { .. } => Ok("ptr".to_owned()),
        IrType::Option(_) | IrType::ResultType(_, _) => Ok("ptr".to_owned()),
        IrType::Chan(_) | IrType::Atomic(_) | IrType::Mutex(_) | IrType::Grad(_) | IrType::Sparse(_) => Ok("ptr".to_owned()),
        IrType::List(_) | IrType::Map(_, _) => Ok("ptr".to_owned()),
        IrType::Fn { .. } => Ok("ptr".to_owned()), // function pointer
        IrType::Infer => Err(CodegenError::Unsupported {
            backend: "llvm-ir".into(),
            detail: "cannot lower Infer type to LLVM".into(),
        }),
    }
}

fn is_scalar_type(ty: &IrType) -> bool {
    matches!(ty, IrType::Scalar(_))
}

fn ety_align(ty: &IrType) -> usize {
    match ty {
        IrType::Scalar(DType::F64) | IrType::Scalar(DType::I64) => 8,
        IrType::Scalar(DType::F32) | IrType::Scalar(DType::I32) => 4,
        IrType::Scalar(DType::Bool) => 1,
        _ => 8,
    }
}

/// Look up the size of an AllocArray for a given result ValueId.
fn find_alloc_size(func: &IrFunction, array_id: ValueId) -> Option<usize> {
    for block in func.blocks() {
        for instr in &block.instrs {
            if let IrInstr::AllocArray { result, size, .. } = instr {
                if *result == array_id {
                    return Some(*size);
                }
            }
        }
    }
    None
}

/// Returns whether an instruction has side effects.
fn is_side_effecting(instr: &IrInstr) -> bool {
    matches!(
        instr,
        IrInstr::Print { .. }
            | IrInstr::Panic { .. }
            | IrInstr::ReadLine { .. }
            | IrInstr::ReadI64 { .. }
            | IrInstr::ReadF64 { .. }
            | IrInstr::Store { .. }
            | IrInstr::ArrayStore { .. }
            | IrInstr::AtomicStore { .. }
            | IrInstr::AtomicAdd { .. }
            | IrInstr::MutexLock { .. }
            | IrInstr::MutexUnlock { .. }
            | IrInstr::ChanSend { .. }
            | IrInstr::ListPush { .. }
            | IrInstr::ListSet { .. }
            | IrInstr::ListPop { .. }
            | IrInstr::ListSort { .. }
            | IrInstr::MapSet { .. }
            | IrInstr::MapRemove { .. }
            | IrInstr::Spawn { .. }
            | IrInstr::ParFor { .. }
            | IrInstr::Barrier
            | IrInstr::FileWriteAll { .. }
            | IrInstr::ProcessExit { .. }
    )
}

fn llvm_val(v: ValueId, consts: &HashMap<ValueId, String>, func: &IrFunction) -> String {
    if let Some(c) = consts.get(&v) {
        return c.clone();
    }
    for param in &func.blocks()[0].params {
        if param.id == v {
            if let Some(name) = &param.name {
                return format!("%{}", name);
            }
        }
    }
    format!("%v{}", v.0)
}

fn block_label(name: Option<&str>, id: BlockId) -> String {
    format!("{}{}", name.unwrap_or("bb"), id.0)
}

fn block_label_by_id(blocks: &[crate::ir::block::IrBlock], id: BlockId) -> String {
    blocks
        .iter()
        .find(|b| b.id == id)
        .map(|b| block_label(b.name.as_deref(), b.id))
        .unwrap_or_else(|| format!("bb{}", id.0))
}

fn fmt_float(v: f64) -> String {
    let s = format!("{}", v);
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{}.0", s)
    }
}

fn llvm_escape_string(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'"' => out.push_str("\\22"),
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

fn emit_runtime_declares(out: &mut String) -> Result<(), CodegenError> {
    let declares: &[&str] = &[
        // Typed print helpers
        "declare void @iris_print(ptr)",
        "declare void @iris_print_i64(i64)",
        "declare void @iris_print_i32(i32)",
        "declare void @iris_print_f64(double)",
        "declare void @iris_print_f32(float)",
        "declare void @iris_print_bool(i1)",
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
        "declare ptr @iris_const_str()",
        // Option / Result
        "declare ptr @iris_make_some(ptr)",
        "declare ptr @iris_make_none()",
        "declare i1 @iris_is_some(ptr)",
        "declare ptr @iris_option_unwrap(ptr)",
        "declare ptr @iris_make_ok(ptr)",
        "declare ptr @iris_make_err(ptr)",
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
        // Extended collections (Phase 58)
        "declare i1 @iris_list_contains(ptr, ptr)",
        "declare void @iris_list_sort(ptr)",
        "declare ptr @iris_map_keys(ptr)",
        "declare ptr @iris_map_values(ptr)",
        "declare ptr @iris_list_concat(ptr, ptr)",
        "declare ptr @iris_list_slice(ptr, i64, i64)",
        // File I/O (Phase 56)
        "declare ptr @iris_file_read_all(ptr)",
        "declare ptr @iris_file_write_all(ptr, ptr)",
        "declare i1 @iris_file_exists(ptr)",
        "declare ptr @iris_file_lines(ptr)",
        // Process / environment (Phase 59)
        "declare void @exit(i32)",
        "declare ptr @iris_process_args()",
        "declare ptr @iris_env_var(ptr)",
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
        "declare void @iris_barrier()",
        // Structs / Tuples / Closures
        "declare ptr @iris_make_struct(...)",
        "declare ptr @iris_get_field(ptr, i32)",
        "declare ptr @iris_make_tuple(...)",
        "declare ptr @iris_get_element(ptr, i32)",
        "declare ptr @iris_make_closure(...)",
        "declare ptr @iris_call_closure(ptr, ...)",
        "declare void @iris_call_closure_void(ptr, ...)",
        // Atomics / Mutex
        "declare ptr @iris_atomic_new(ptr)",
        "declare ptr @iris_atomic_load(ptr)",
        "declare void @iris_atomic_store(ptr, ptr)",
        "declare ptr @iris_atomic_add(ptr, ptr)",
        "declare ptr @iris_mutex_new()",
        "declare ptr @iris_mutex_lock(ptr)",
        "declare void @iris_mutex_unlock(ptr)",
        // Grad / Sparse
        "declare ptr @iris_make_grad(double, double)",
        "declare double @iris_grad_value(ptr)",
        "declare double @iris_grad_tangent(ptr)",
        "declare ptr @iris_sparsify(ptr)",
        "declare ptr @iris_densify(ptr)",
        // Boxing helpers (scalar → IrisVal*)
        "declare ptr @iris_box_i64(i64)",
        "declare ptr @iris_box_i32(i32)",
        "declare ptr @iris_box_f64(double)",
        "declare ptr @iris_box_f32(float)",
        "declare ptr @iris_box_bool(i1)",
        // Typed to-string conversions
        "declare ptr @iris_i64_to_str(i64)",
        "declare ptr @iris_i32_to_str(i32)",
        "declare ptr @iris_f64_to_str(double)",
        "declare ptr @iris_f32_to_str(float)",
        "declare ptr @iris_bool_to_str(i1)",
        "declare ptr @iris_str_to_str(ptr)",
        // Math helpers
        "declare i64 @iris_pow_i64(i64, i64)",
        "declare i64 @iris_min_i64(i64, i64)",
        "declare i64 @iris_max_i64(i64, i64)",
        "declare i64 @iris_abs_i64(i64)",
        "declare double @iris_sign_f64(double)",
        "declare double @tan(double)",
        // LLVM intrinsics
        "declare double @llvm.sqrt.f64(double)",
        "declare double @llvm.fabs.f64(double)",
        "declare double @llvm.floor.f64(double)",
        "declare double @llvm.ceil.f64(double)",
        "declare double @llvm.round.f64(double)",
        "declare double @llvm.sin.f64(double)",
        "declare double @llvm.cos.f64(double)",
        "declare double @llvm.exp.f64(double)",
        "declare double @llvm.log.f64(double)",
        "declare double @llvm.log2.f64(double)",
        "declare double @llvm.pow.f64(double, double)",
        "declare double @llvm.minnum.f64(double, double)",
        "declare double @llvm.maxnum.f64(double, double)",
    ];
    for decl in declares {
        writeln!(out, "{}", decl)?;
    }
    writeln!(out)?;
    Ok(())
}
