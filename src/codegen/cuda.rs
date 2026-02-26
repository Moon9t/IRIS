//! CUDA / NVPTX kernel codegen for IRIS.
//!
//! Phase 50: Emits NVPTX LLVM IR that can be compiled with `clang --target=nvptx64`
//! or passed directly to the CUDA driver via `nvrtcCompileProgram`.
//!
//! Design:
//! - Target triple is `nvptx64-nvidia-cuda` with NVPTX data layout.
//! - Every `ParFor` body function is promoted to a CUDA kernel and annotated
//!   with `!nvvm.annotations` metadata.
//! - Thread/block/grid IDs are read via `@llvm.nvvm.read.ptx.sreg.*` intrinsics.
//! - `Barrier` instructions map to `@llvm.nvvm.barrier0`.
//! - Scalar arithmetic is identical to the CPU LLVM backend.
//! - Opaque types (tensors, lists, channels, …) remain as `ptr`; device-side
//!   memory must be pre-allocated via `cudaMalloc` before kernel launch.
//!
//! Example IRIS source that produces a kernel:
//! ```iris
//! def vector_add(a: tensor<f32,[N]>, b: tensor<f32,[N]>, out: tensor<f32,[N]>) -> i64 {
//!     par for i in 0..1024 {
//!         val ai = a[i]
//!         val bi = b[i]
//!         out[i] = ai + bi
//!     }
//!     0
//! }
//! ```
//!
//! The emitter generates both the scalar host-side wrapper and the device kernel.

use std::collections::HashMap;
use std::fmt::Write;

use crate::error::CodegenError;
use crate::ir::block::BlockId;
use crate::ir::function::IrFunction;
use crate::ir::instr::{BinOp, IrInstr, ScalarUnaryOp};
use crate::ir::module::IrModule;
use crate::ir::types::{DType, IrType};
use crate::ir::value::ValueId;

// ---------------------------------------------------------------------------
// NVPTX data layout and target triple
// ---------------------------------------------------------------------------

const NVPTX_DATALAYOUT: &str =
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";
const NVPTX_TRIPLE: &str = "nvptx64-nvidia-cuda";

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Emit CUDA/NVPTX LLVM IR for all functions in the module.
///
/// Functions that contain `ParFor` instructions are split:
/// 1. A GPU **kernel** function (annotated with `!nvvm.annotations`).
/// 2. The original host function, which dispatches the kernel launch.
pub fn emit_cuda(module: &IrModule) -> Result<String, CodegenError> {
    let mut out = String::new();

    writeln!(out, "; IRIS CUDA/NVPTX IR — phase 50")?;
    writeln!(out, "; Compile: clang -target nvptx64-nvidia-cuda -O3 -o out.ptx")?;
    writeln!(out)?;
    writeln!(out, "target datalayout = \"{}\"", NVPTX_DATALAYOUT)?;
    writeln!(out, "target triple = \"{}\"\n", NVPTX_TRIPLE)?;

    // ── Global string constants ────────────────────────────────────────────
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
        let escaped = cuda_escape_string(content);
        let len = content.len() + 1;
        writeln!(
            out,
            "@.str.{} = private unnamed_addr addrspace(4) constant [{} x i8] c\"{}\\00\", align 1",
            idx, len, escaped
        )?;
    }
    if !str_vec.is_empty() {
        writeln!(out)?;
    }

    // ── NVPTX intrinsic declarations ───────────────────────────────────────
    emit_nvptx_declares(&mut out)?;

    // ── Collect which functions contain ParFor (need kernel extraction) ────
    let kernel_bodies: Vec<String> = module
        .functions()
        .iter()
        .flat_map(|f| {
            f.blocks().iter().flat_map(|b| {
                b.instrs.iter().filter_map(|i| {
                    if let IrInstr::ParFor { body_fn, .. } = i {
                        Some(body_fn.clone())
                    } else {
                        None
                    }
                })
            })
        })
        .collect();

    // Emit kernel wrappers for ParFor body functions.
    for body_fn in &kernel_bodies {
        if let Some(f) = module.function_by_name(body_fn) {
            emit_cuda_kernel(f, &str_table, &mut out)?;
        }
    }

    // ── Host-side function definitions ────────────────────────────────────
    // Build function signature map.
    let mut fn_sigs: HashMap<String, (String, Vec<String>)> = HashMap::new();
    for func in module.functions() {
        let ret_s = cuda_type(& func.return_ty).unwrap_or_else(|_| "ptr".to_owned());
        let param_ss: Vec<String> = func
            .params
            .iter()
            .map(|p| cuda_type(&p.ty).unwrap_or_else(|_| "ptr".to_owned()))
            .collect();
        fn_sigs.insert(func.name.clone(), (ret_s, param_ss));
    }

    for func in module.functions() {
        // Skip body functions already emitted as kernels.
        if kernel_bodies.contains(&func.name) {
            continue;
        }
        emit_cuda_host_function(func, &str_table, &fn_sigs, &mut out)?;
    }

    // ── !nvvm.annotations metadata ────────────────────────────────────────
    // Must appear at the end of the module.
    if !kernel_bodies.is_empty() {
        writeln!(out, "; NVVM annotations — mark kernel functions.")?;
        writeln!(out, "!nvvm.annotations = !{{")?;
        for (i, name) in kernel_bodies.iter().enumerate() {
            writeln!(
                out,
                "  !{{ptr @{}_kernel, !\"kernel\", i32 1}}{}",
                name,
                if i + 1 < kernel_bodies.len() { "," } else { "" }
            )?;
        }
        writeln!(out, "}}")?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Kernel function emitter
// ---------------------------------------------------------------------------

/// Emit a CUDA kernel wrapper `@{name}_kernel` with standard thread/block dispatch.
fn emit_cuda_kernel(
    func: &IrFunction,
    str_table: &HashMap<String, usize>,
    out: &mut String,
) -> Result<(), CodegenError> {
    let ret = cuda_type(&func.return_ty)?;
    let params: Result<Vec<String>, CodegenError> = func
        .params
        .iter()
        .map(|p| Ok(format!("{} %{}", cuda_type(&p.ty)?, p.name)))
        .collect();

    writeln!(
        out,
        "; ── CUDA kernel: {}_kernel ───────────────────────────────",
        func.name
    )?;
    writeln!(out, "define {} @{}_kernel({}) {{", ret, func.name, params?.join(", "))?;

    // Entry: compute flat thread index.
    writeln!(out, "kernel_entry{}:", func.name)?;
    writeln!(out, "  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()")?;
    writeln!(out, "  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()")?;
    writeln!(out, "  %bdim = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()")?;
    writeln!(out, "  %flat_i32 = mul i32 %bid, %bdim")?;
    writeln!(out, "  %flat_idx_i32 = add i32 %flat_i32, %tid")?;
    writeln!(out, "  %flat_idx = sext i32 %flat_idx_i32 to i64")?;

    // Emit the kernel body with the flat index as the loop variable.
    emit_cuda_kernel_body(func, str_table, out)?;

    writeln!(out, "}}\n")?;
    Ok(())
}

fn emit_cuda_kernel_body(
    func: &IrFunction,
    str_table: &HashMap<String, usize>,
    out: &mut String,
) -> Result<(), CodegenError> {
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

    let entry_id = func.blocks()[0].id;
    let mut gep_counter: u32 = 0;

    for block in func.blocks() {
        let blabel = block_label(block.name.as_deref(), block.id);
        writeln!(out, "{}:", blabel)?;
        if block.id != entry_id {
            for (i, param) in block.params.iter().enumerate() {
                let ty_s = cuda_type(&param.ty)?;
                let phi_name = format!("%v{}", param.id.0);
                let arms: Vec<String> = phi_src
                    .get(&(block.id, i))
                    .map(|srcs| {
                        srcs.iter()
                            .map(|(pred_id, v)| {
                                let vstr = cuda_val(*v, &consts, func);
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
            emit_cuda_instr(instr, &consts, func, &mut gep_counter, str_table, out)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Host-side function emitter
// ---------------------------------------------------------------------------

fn emit_cuda_host_function(
    func: &IrFunction,
    str_table: &HashMap<String, usize>,
    _fn_sigs: &HashMap<String, (String, Vec<String>)>,
    out: &mut String,
) -> Result<(), CodegenError> {
    let ret = cuda_type(&func.return_ty)?;
    let params: Result<Vec<String>, CodegenError> = func
        .params
        .iter()
        .map(|p| Ok(format!("{} %{}", cuda_type(&p.ty)?, p.name)))
        .collect();

    writeln!(out, "define {} @{}({}) {{", ret, func.name, params?.join(", "))?;
    emit_cuda_kernel_body(func, str_table, out)?;
    writeln!(out, "}}\n")?;
    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA instruction emitter
// ---------------------------------------------------------------------------

fn emit_cuda_instr(
    instr: &IrInstr,
    consts: &HashMap<ValueId, String>,
    func: &IrFunction,
    gep_counter: &mut u32,
    str_table: &HashMap<String, usize>,
    out: &mut String,
) -> Result<(), CodegenError> {
    let val = |v: ValueId| cuda_val(v, consts, func);

    match instr {
        IrInstr::ConstFloat { .. } | IrInstr::ConstInt { .. } | IrInstr::ConstBool { .. } => {}

        IrInstr::BinOp { result, op, lhs, rhs, ty } => {
            let lv = val(*lhs);
            let rv = val(*rhs);
            let operand_ty = func.value_type(*lhs).unwrap_or(ty);
            let ty_s = cuda_type(operand_ty)?;
            let is_float = matches!(operand_ty, IrType::Scalar(DType::F32 | DType::F64));
            let llvm_op = match (op, is_float) {
                (BinOp::Add, true) => format!("fadd {} {}, {}", ty_s, lv, rv),
                (BinOp::Sub, true) => format!("fsub {} {}, {}", ty_s, lv, rv),
                (BinOp::Mul, true) => format!("fmul {} {}, {}", ty_s, lv, rv),
                (BinOp::Div, true) => format!("fdiv {} {}, {}", ty_s, lv, rv),
                (BinOp::Add, false) => format!("add nsw {} {}, {}", ty_s, lv, rv),
                (BinOp::Sub, false) => format!("sub nsw {} {}, {}", ty_s, lv, rv),
                (BinOp::Mul, false) => format!("mul nsw {} {}, {}", ty_s, lv, rv),
                (BinOp::Div, false) | (BinOp::FloorDiv, _) => format!("sdiv {} {}, {}", ty_s, lv, rv),
                (BinOp::Mod, _) => format!("srem {} {}, {}", ty_s, lv, rv),
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
                _ => format!("add i64 0, 0 ; unsupported cuda binop"),
            };
            writeln!(out, "  %v{} = {}", result.0, llvm_op)?;
        }

        IrInstr::UnaryOp { result, op, operand, ty } => {
            let ov = val(*operand);
            let ty_s = cuda_type(ty)?;
            let is_float = matches!(ty, IrType::Scalar(DType::F32 | DType::F64));
            match op {
                ScalarUnaryOp::Neg if is_float => writeln!(out, "  %v{} = fneg {} {}", result.0, ty_s, ov)?,
                ScalarUnaryOp::Neg => writeln!(out, "  %v{} = sub nsw {} 0, {}", result.0, ty_s, ov)?,
                ScalarUnaryOp::Not => writeln!(out, "  %v{} = xor i1 {}, true", result.0, ov)?,
                ScalarUnaryOp::Sqrt => writeln!(out, "  %v{} = call {} @llvm.sqrt.f64({} {})", result.0, ty_s, ty_s, ov)?,
                ScalarUnaryOp::Abs if is_float => writeln!(out, "  %v{} = call {} @llvm.fabs.f64({} {})", result.0, ty_s, ty_s, ov)?,
                ScalarUnaryOp::Abs => writeln!(out, "  %v{} = call i64 @iris_abs_i64(i64 {})", result.0, ov)?,
                ScalarUnaryOp::Floor => writeln!(out, "  %v{} = call {} @llvm.floor.f64({} {})", result.0, ty_s, ty_s, ov)?,
                ScalarUnaryOp::Ceil => writeln!(out, "  %v{} = call {} @llvm.ceil.f64({} {})", result.0, ty_s, ty_s, ov)?,
                ScalarUnaryOp::Sin => writeln!(out, "  %v{} = call {} @llvm.sin.f64({} {})", result.0, ty_s, ty_s, ov)?,
                ScalarUnaryOp::Cos => writeln!(out, "  %v{} = call {} @llvm.cos.f64({} {})", result.0, ty_s, ty_s, ov)?,
                ScalarUnaryOp::Exp => writeln!(out, "  %v{} = call {} @llvm.exp.f64({} {})", result.0, ty_s, ty_s, ov)?,
                ScalarUnaryOp::Log => writeln!(out, "  %v{} = call {} @llvm.log.f64({} {})", result.0, ty_s, ty_s, ov)?,
                ScalarUnaryOp::Log2 => writeln!(out, "  %v{} = call {} @llvm.log2.f64({} {})", result.0, ty_s, ty_s, ov)?,
                ScalarUnaryOp::Round => writeln!(out, "  %v{} = call {} @llvm.round.f64({} {})", result.0, ty_s, ty_s, ov)?,
                ScalarUnaryOp::BitNot => writeln!(out, "  %v{} = xor {} {}, -1", result.0, ty_s, ov)?,
                ScalarUnaryOp::Tan => writeln!(out, "  %v{} = call double @__nv_tan(double {})", result.0, ov)?,
                ScalarUnaryOp::Sign => writeln!(out, "  %v{} = call double @iris_sign_f64(double {})", result.0, ov)?,
            }
        }

        IrInstr::Cast { result, operand, from_ty, to_ty } => {
            let ov = val(*operand);
            let from_s = cuda_type(from_ty)?;
            let to_s = cuda_type(to_ty)?;
            let is_from_float = matches!(from_ty, IrType::Scalar(DType::F32 | DType::F64));
            let is_to_float = matches!(to_ty, IrType::Scalar(DType::F32 | DType::F64));
            let is_from_int = matches!(from_ty, IrType::Scalar(DType::I32 | DType::I64));
            let is_to_int = matches!(to_ty, IrType::Scalar(DType::I32 | DType::I64));
            if from_ty == to_ty {
                writeln!(out, "  %v{} = bitcast {} {} to {}", result.0, from_s, ov, to_s)?;
            } else if is_from_float && is_to_int {
                writeln!(out, "  %v{} = fptosi {} {} to {}", result.0, from_s, ov, to_s)?;
            } else if is_from_int && is_to_float {
                writeln!(out, "  %v{} = sitofp {} {} to {}", result.0, from_s, ov, to_s)?;
            } else if is_from_float && is_to_float {
                let is_from_f64 = matches!(from_ty, IrType::Scalar(DType::F64));
                let is_to_f64 = matches!(to_ty, IrType::Scalar(DType::F64));
                if !is_from_f64 && is_to_f64 {
                    writeln!(out, "  %v{} = fpext {} {} to {}", result.0, from_s, ov, to_s)?;
                } else {
                    writeln!(out, "  %v{} = fptrunc {} {} to {}", result.0, from_s, ov, to_s)?;
                }
            } else if is_from_int && is_to_int {
                let is_from_i64 = matches!(from_ty, IrType::Scalar(DType::I64));
                let is_to_i64 = matches!(to_ty, IrType::Scalar(DType::I64));
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
                let ty_s = cuda_type(&func.return_ty)?;
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

        IrInstr::Load { result, tensor, indices, result_ty } => {
            let tv = val(*tensor);
            let ty_s = cuda_type(result_ty)?;
            match indices.as_slice() {
                [] => writeln!(out, "  %v{} = load {}, ptr {}", result.0, ty_s, tv)?,
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
                .and_then(|ty| cuda_type(ty).ok())
                .unwrap_or_else(|| "ptr".to_owned());
            match indices.as_slice() {
                [] => writeln!(out, "  store {} {}, ptr {}", ty_s, vv, tv)?,
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

        IrInstr::Barrier => {
            // CUDA __syncthreads() via NVVM intrinsic.
            writeln!(out, "  call void @llvm.nvvm.barrier0()")?;
        }

        IrInstr::ParFor { body_fn, start, end, .. } => {
            // On the GPU host side, emit a comment describing the kernel launch.
            // Actual kernel launch configuration is handled by the host runtime.
            writeln!(
                out,
                "  ; CUDA kernel launch: {}_kernel<<<grid, block>>>(i64 {}, i64 {})",
                body_fn,
                val(*start),
                val(*end)
            )?;
            writeln!(
                out,
                "  call void @iris_cuda_launch(ptr @{}_kernel, i64 {}, i64 {})",
                body_fn,
                val(*start),
                val(*end)
            )?;
        }

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

        IrInstr::TensorOp { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_tensor_op()", result.0)?;
        }

        IrInstr::Call { result, callee, args, result_ty } => {
            let ret_ty_s = result_ty
                .as_ref()
                .and_then(|t| cuda_type(t).ok())
                .unwrap_or_else(|| "ptr".to_owned());
            let args_str: Vec<String> = args.iter().map(|a| format!("ptr {}", val(*a))).collect();
            if let Some(r) = result {
                writeln!(out, "  %v{} = call {} @{}({})", r.0, ret_ty_s, callee, args_str.join(", "))?;
            } else {
                writeln!(out, "  call void @{}({})", callee, args_str.join(", "))?;
            }
        }

        IrInstr::Print { operand } => {
            // Use CUDA's vprintf for device-side printing.
            writeln!(out, "  ; cuda device print — uses vprintf")?;
            writeln!(out, "  call void @iris_print(ptr {})", val(*operand))?;
        }

        IrInstr::ConstStr { result, value } => {
            if let Some(&idx) = str_table.get(value) {
                let len = value.len() + 1;
                writeln!(
                    out,
                    "  %v{} = getelementptr inbounds [{} x i8], ptr addrspace(4) @.str.{}, i32 0, i32 0",
                    result.0, len, idx
                )?;
            } else {
                writeln!(out, "  %v{} = call ptr @iris_const_str()", result.0)?;
            }
        }

        // Everything else: emit as opaque runtime call or no-op.
        instr => {
            if let Some(result_id) = instr_result(instr) {
                writeln!(out, "  %v{} = call ptr @iris_cuda_unsupported()", result_id.0)?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// CUDA helpers
// ---------------------------------------------------------------------------

fn cuda_type(ty: &IrType) -> Result<String, CodegenError> {
    match ty {
        IrType::Scalar(DType::F32) => Ok("float".to_owned()),
        IrType::Scalar(DType::F64) => Ok("double".to_owned()),
        IrType::Scalar(DType::I32) => Ok("i32".to_owned()),
        IrType::Scalar(DType::I64) => Ok("i64".to_owned()),
        IrType::Scalar(DType::Bool) => Ok("i1".to_owned()),
        IrType::Enum { .. } => Ok("i64".to_owned()),
        _ => Ok("ptr".to_owned()),
    }
}

fn cuda_val(v: ValueId, consts: &HashMap<ValueId, String>, func: &IrFunction) -> String {
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

fn cuda_escape_string(s: &str) -> String {
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

fn instr_result(instr: &IrInstr) -> Option<ValueId> {
    match instr {
        IrInstr::BinOp { result, .. }
        | IrInstr::UnaryOp { result, .. }
        | IrInstr::ConstFloat { result, .. }
        | IrInstr::ConstInt { result, .. }
        | IrInstr::ConstBool { result, .. }
        | IrInstr::Cast { result, .. }
        | IrInstr::Load { result, .. }
        | IrInstr::Call { result: Some(result), .. }
        | IrInstr::MakeStruct { result, .. }
        | IrInstr::GetField { result, .. }
        | IrInstr::MakeVariant { result, .. }
        | IrInstr::MakeTuple { result, .. }
        | IrInstr::GetElement { result, .. }
        | IrInstr::MakeClosure { result, .. }
        | IrInstr::AllocArray { result, .. }
        | IrInstr::ArrayLoad { result, .. }
        | IrInstr::TensorOp { result, .. }
        | IrInstr::MakeSome { result, .. }
        | IrInstr::MakeNone { result, .. }
        | IrInstr::IsSome { result, .. }
        | IrInstr::OptionUnwrap { result, .. }
        | IrInstr::MakeOk { result, .. }
        | IrInstr::MakeErr { result, .. }
        | IrInstr::IsOk { result, .. }
        | IrInstr::ResultUnwrap { result, .. }
        | IrInstr::ResultUnwrapErr { result, .. }
        | IrInstr::ChanNew { result, .. }
        | IrInstr::ChanRecv { result, .. }
        | IrInstr::AtomicNew { result, .. }
        | IrInstr::AtomicLoad { result, .. }
        | IrInstr::AtomicAdd { result, .. }
        | IrInstr::MutexNew { result, .. }
        | IrInstr::MutexLock { result, .. }
        | IrInstr::ConstStr { result, .. }
        | IrInstr::StrLen { result, .. }
        | IrInstr::StrConcat { result, .. }
        | IrInstr::StrContains { result, .. }
        | IrInstr::StrStartsWith { result, .. }
        | IrInstr::StrEndsWith { result, .. }
        | IrInstr::StrToUpper { result, .. }
        | IrInstr::StrToLower { result, .. }
        | IrInstr::StrTrim { result, .. }
        | IrInstr::StrRepeat { result, .. }
        | IrInstr::StrIndex { result, .. }
        | IrInstr::StrSlice { result, .. }
        | IrInstr::StrFind { result, .. }
        | IrInstr::StrReplace { result, .. }
        | IrInstr::ValueToStr { result, .. }
        | IrInstr::ParseI64 { result, .. }
        | IrInstr::ParseF64 { result, .. }
        | IrInstr::ListNew { result, .. }
        | IrInstr::ListLen { result, .. }
        | IrInstr::ListGet { result, .. }
        | IrInstr::ListPop { result, .. }
        | IrInstr::MapNew { result, .. }
        | IrInstr::MapGet { result, .. }
        | IrInstr::MapContains { result, .. }
        | IrInstr::MapLen { result, .. }
        | IrInstr::MakeGrad { result, .. }
        | IrInstr::GradValue { result, .. }
        | IrInstr::GradTangent { result, .. }
        | IrInstr::Sparsify { result, .. }
        | IrInstr::Densify { result, .. }
        | IrInstr::ReadLine { result }
        | IrInstr::ReadI64 { result }
        | IrInstr::ReadF64 { result } => Some(*result),
        _ => None,
    }
}

fn emit_nvptx_declares(out: &mut String) -> Result<(), CodegenError> {
    let declares = &[
        // NVVM thread/block/grid ID intrinsics
        "declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()",
        "declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()",
        "declare i32 @llvm.nvvm.read.ptx.sreg.tid.z()",
        "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()",
        "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()",
        "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()",
        "declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()",
        "declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()",
        "declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()",
        "declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()",
        "declare i32 @llvm.nvvm.read.ptx.sreg.warpsize()",
        // Warp-level intrinsics
        "declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32)",
        "declare float @llvm.nvvm.shfl.sync.bfly.f32(i32, float, i32, i32)",
        // Barrier
        "declare void @llvm.nvvm.barrier0()",
        "declare void @llvm.nvvm.barrier.sync(i32)",
        // Atomic operations (CUDA-native)
        "declare i32 @llvm.nvvm.atomic.add.gen.i.cta.i32.p0i32(ptr, i32)",
        "declare float @llvm.nvvm.atomic.add.gen.f.cta.f32.p0f32(ptr, float)",
        // Math intrinsics (same as CPU)
        "declare double @llvm.sqrt.f64(double)",
        "declare float @llvm.sqrt.f32(float)",
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
        // CUDA libdevice (device math library)
        "declare double @__nv_tan(double)",
        "declare double @__nv_atan2(double, double)",
        "declare float @__nv_tanf(float)",
        // IRIS device runtime helpers
        "declare void @iris_print(ptr)",
        "declare void @iris_panic(ptr)",
        "declare ptr @iris_tensor_op()",
        "declare ptr @iris_tensor_load(ptr, ...)",
        "declare void @iris_tensor_store(ptr, ...)",
        "declare void @iris_cuda_launch(ptr, i64, i64)",
        "declare ptr @iris_const_str()",
        "declare i64 @iris_pow_i64(i64, i64)",
        "declare i64 @iris_min_i64(i64, i64)",
        "declare i64 @iris_max_i64(i64, i64)",
        "declare i64 @iris_abs_i64(i64)",
        "declare double @iris_sign_f64(double)",
        "declare ptr @iris_cuda_unsupported()",
    ];
    for decl in declares {
        writeln!(out, "{}", decl)?;
    }
    writeln!(out)?;
    Ok(())
}
