//! Native LLVM IR generation via `inkwell`.
//!
//! This module provides in-process LLVM code generation as an alternative
//! to the text-based `llvm_ir.rs` emitter. It is gated
//! behind the `native-llvm` Cargo feature.
//!
//! Capabilities when `native-llvm` is enabled:
//!
//! 1. **In-memory IR generation** — builds LLVM modules directly via
//!    inkwell's safe wrapper, avoiding string formatting.
//!
//! 2. **True in-process JIT** — uses LLVM's `ExecutionEngine` (MCJIT)
//!    to compile and execute IRIS functions without spawning `clang`.
//!
//! 3. **Native object file emission** — uses `TargetMachine` to emit
//!    `.o` / `.obj` files directly, skipping the `.ll` → `clang`
//!    intermediate step in the build pipeline.

#[cfg(feature = "native-llvm")]
mod backend {
    use std::collections::HashMap;
    use std::path::Path;

    use inkwell::attributes::{Attribute, AttributeLoc};
    use inkwell::basic_block::BasicBlock;
    use inkwell::builder::Builder;
    use inkwell::context::Context;
    use inkwell::execution_engine::{ExecutionEngine, JitFunction};
    use inkwell::module::Module;
    use inkwell::targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine};
    use inkwell::types::{BasicType, BasicTypeEnum, FunctionType};
    use inkwell::values::{
        BasicValue, BasicValueEnum, FunctionValue, InstructionOpcode, IntValue, PointerValue,
    };
    use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};

    use crate::error::CodegenError;
    use crate::ir::block::BlockId;
    use crate::ir::function::IrFunction;
    use crate::ir::instr::{BinOp, IrInstr, ScalarUnaryOp};
    use crate::ir::module::IrModule;
    use crate::ir::types::{DType, IrType};
    use crate::ir::value::ValueId;

    /// Configuration for the native LLVM backend.
    #[derive(Debug, Clone)]
    pub struct LlvmNativeConfig {
        /// Optimization level (0-3).
        pub opt_level: OptimizationLevel,
        /// Emit debug info.
        pub debug_info: bool,
        /// Target triple override.
        pub target_triple: Option<String>,
    }

    impl Default for LlvmNativeConfig {
        fn default() -> Self {
            Self {
                opt_level: OptimizationLevel::Aggressive,
                debug_info: false,
                target_triple: None,
            }
        }
    }

    /// A native LLVM compilation context that owns the LLVM `Context` and
    /// compiled `Module`.
    pub struct LlvmNativeCompiler<'ctx> {
        context: &'ctx Context,
        module: Module<'ctx>,
        builder: Builder<'ctx>,
        config: LlvmNativeConfig,
        /// ValueId → LLVM BasicValue (for SSA references).
        values: HashMap<ValueId, BasicValueEnum<'ctx>>,
        /// Label names for basic blocks.
        block_labels: HashMap<BlockId, BasicBlock<'ctx>>,
    }

    impl<'ctx> LlvmNativeCompiler<'ctx> {
        /// Create a new compiler attached to the given LLVM context.
        pub fn new(context: &'ctx Context, name: &str, config: LlvmNativeConfig) -> Self {
            let module = context.create_module(name);
            let builder = context.create_builder();
            if let Some(ref triple) = config.target_triple {
                module.set_triple(&inkwell::targets::TargetTriple::create(triple));
            }

            Self {
                context,
                module,
                builder,
                config,
                values: HashMap::new(),
                block_labels: HashMap::new(),
            }
        }

        /// Compile an `IrModule` into an in-memory LLVM module.
        pub fn compile_module(&mut self, ir_module: &IrModule) -> Result<(), CodegenError> {
            // Declare all functions first.
            for func in ir_module.functions() {
                self.declare_function(func)?;
            }

            // Emit vtable globals for trait objects.
            self.emit_vtable_globals(ir_module)?;

            // Define each function body.
            for func in ir_module.functions() {
                self.compile_function(func, ir_module)?;
            }

            Ok(())
        }

        fn emit_vtable_globals(&self, module: &IrModule) -> Result<(), CodegenError> {
            let ptr_ty = self.context.i8_type().ptr_type(AddressSpace::from(0));
            for (trait_name, methods) in module.trait_defs() {
                let Some(impl_list) = module.trait_impl_methods().get(trait_name) else {
                    continue;
                };
                let mut by_concrete: std::collections::BTreeMap<String, Vec<(String, String)>> =
                    std::collections::BTreeMap::new();
                for (concrete, mname, mangled) in impl_list {
                    by_concrete
                        .entry(concrete.clone())
                        .or_default()
                        .push((mname.clone(), mangled.clone()));
                }
                for (concrete, mut entries) in by_concrete {
                    entries.sort_by(|a, b| {
                        let ai = methods.iter().position(|m| m.name == a.0);
                        let bi = methods.iter().position(|m| m.name == b.0);
                        ai.cmp(&bi)
                    });
                    let n = entries.len();
                    if n == 0 {
                        continue;
                    }
                    let global_name = format!("vtable_{}__{}", trait_name, concrete);
                    // Build an array type [N x ptr] and collect fn ptr values.
                    let mut fn_ptrs = Vec::with_capacity(n);
                    for (_mname, mangled) in &entries {
                        if let Some(fn_val) = self.module.get_function(mangled) {
                            fn_ptrs.push(fn_val.as_global_value().as_pointer_value());
                        } else {
                            return Err(CodegenError::Unsupported {
                                backend: "llvm_native".into(),
                                detail: format!(
                                    "vtable: function '{}' not found for trait '{}' concrete '{}'",
                                    mangled, trait_name, concrete
                                ),
                            });
                        }
                    }
                    let array_ty = ptr_ty.array_type(n as u32);
                    let global = self.module.add_global(array_ty, None, &global_name);
                    global.set_linkage(inkwell::module::Linkage::Internal);
                    // Build a constant array of ptr values.
                    let vals: Vec<&dyn BasicValue<'ctx>> = fn_ptrs
                        .iter()
                        .map(|p| p as &dyn BasicValue)
                        .collect();
                    let init_val = ptr_ty.const_array(&vals);
                    global.set_initializer(&init_val);
                }
            }
            Ok(())
        }

        /// Verify the LLVM module.
        pub fn verify(&self) -> Result<(), CodegenError> {
            self.module
                .verify()
                .map_err(|msg| CodegenError::Unsupported {
                    backend: "llvm_native".into(),
                    detail: format!("LLVM module verification failed: {}", msg),
                })
        }

        /// Write the LLVM module as bitcode to a file.
        pub fn write_bitcode_to_file(&self, path: &Path) -> Result<(), CodegenError> {
            self.module
                .write_bitcode_to_path(path)
                .map_err(|e| CodegenError::Unsupported {
                    backend: "llvm_native".into(),
                    detail: format!("failed to write bitcode: {}", e),
                })
        }

        /// Emit an object file for the host target.
        pub fn emit_object(&self, path: &Path, target_triple: Option<&str>) -> Result<(), CodegenError> {
            let triple_str = target_triple
                .unwrap_or(&inkwell::targets::TargetMachine::get_default_triple().as_str().to_string_lossy())
                .to_string();

            let target = Target::from_triple(&inkwell::targets::TargetTriple::create(&triple_str))
                .map_err(|e| CodegenError::Unsupported {
                    backend: "llvm_native".into(),
                    detail: format!("unknown target triple '{}': {}", triple_str, e),
                })?;

            let target_machine = target
                .create_target_machine(
                    &inkwell::targets::TargetTriple::create(&triple_str),
                    "",
                    "",
                    self.config.opt_level,
                    RelocMode::Default,
                    CodeModel::Default,
                )
                .ok_or_else(|| CodegenError::Unsupported {
                    backend: "llvm_native".into(),
                    detail: format!("failed to create target machine for '{}'", triple_str),
                })?;

            target_machine
                .write_to_file(&self.module, FileType::Object, path)
                .map_err(|e| CodegenError::Unsupported {
                    backend: "llvm_native".into(),
                    detail: format!("failed to emit object file: {}", e),
                })
        }

        /// Consume the module and create a JIT execution engine.
        pub fn create_jit(self) -> Result<ExecutionEngine<'ctx>, CodegenError> {
            self.module
                .create_jit_execution_engine(self.config.opt_level)
                .map_err(|e| CodegenError::Unsupported {
                    backend: "llvm_native".into(),
                    detail: format!("failed to create JIT engine: {}", e),
                })
        }

        // ------------------------------------------------------------------
        // Internal: function declaration
        // ------------------------------------------------------------------

        fn declare_function(&mut self, func: &IrFunction) -> Result<FunctionValue<'ctx>, CodegenError> {
            let ret_type = ir_type_to_basic_type(self.context, &func.return_ty)?;
            let param_types: Vec<BasicTypeEnum> = func
                .params
                .iter()
                .map(|p| ir_type_to_basic_type(self.context, &p.ty))
                .collect::<Result<Vec<_>, _>>()?;

            let fn_type = match ret_type {
                Some(bt) => bt.fn_type(&param_types, false),
                None => self.context.void_type().fn_type(&param_types, false),
            };

            let llvm_fn = self.module.add_function(&func.name, fn_type, None);
            Ok(llvm_fn)
        }

        // ------------------------------------------------------------------
        // Internal: function body compilation
        // ------------------------------------------------------------------

        fn compile_function(&mut self, func: &IrFunction, module: &IrModule) -> Result<(), CodegenError> {
            let llvm_fn = self
                .module
                .get_function(&func.name)
                .ok_or_else(|| CodegenError::Unsupported {
                    backend: "llvm_native".into(),
                    detail: format!("function '{}' not declared", func.name),
                })?;

            // Create basic blocks.
            let entry_bb = self.context.append_basic_block(llvm_fn, "entry");
            self.builder.position_at_end(entry_bb);
            self.block_labels = HashMap::new();

            // Map block IDs to LLVM basic blocks.
            let blocks = func.blocks();
            for block in blocks {
                let name = block_label_name(block.name.as_deref(), block.id);
                let bb = self.context.append_basic_block(llvm_fn, &name);
                self.block_labels.insert(block.id, bb);
            }

            // Branch from entry to the first actual block.
            let first_bb = self.block_labels.get(&blocks[0].id).copied();
            if let Some(bb) = first_bb {
                if bb != entry_bb {
                    self.builder.build_unconditional_branch(bb)?;
                }
            }

            self.values.clear();

            // Compile each block.
            for block in blocks {
                if let Some(&bb) = self.block_labels.get(&block.id) {
                    self.builder.position_at_end(bb);

                    // Map block params (phi values).
                    for param in &block.params {
                        let ty = ir_type_to_basic_type(self.context, &param.ty)?;
                        if let Some(bt) = ty {
                            let phi = self.builder.build_phi(bt, &format!("v{}", param.id.0))?;
                            self.values.insert(param.id, phi.as_basic_value());
                        }
                    }

                    // Compile instructions.
                    for instr in &block.instrs {
                        self.compile_instr(instr, func, module)?;
                    }
                }
            }

            // Wire up phi incoming values.
            self.wire_phis(func)?;

            Ok(())
        }

        fn wire_phis(&self, func: &IrFunction) -> Result<(), CodegenError> {
            // Collect predecessor → block relationships from terminators.
            let mut pred_map: HashMap<BlockId, Vec<(BlockId, Vec<ValueId>)>> = HashMap::new();
            for block in func.blocks() {
                for instr in &block.instrs {
                    match instr {
                        IrInstr::Br { target, args } => {
                            pred_map
                                .entry(*target)
                                .or_default()
                                .push((block.id, args.clone()));
                        }
                        IrInstr::CondBr {
                            then_block,
                            then_args,
                            else_block,
                            else_args,
                            ..
                        } => {
                            pred_map
                                .entry(*then_block)
                                .or_default()
                                .push((block.id, then_args.clone()));
                            pred_map
                                .entry(*else_block)
                                .or_default()
                                .push((block.id, else_args.clone()));
                        }
                        _ => {}
                    }
                }
            }

            // Wire phis: for each block param, add incoming values.
            for block in func.blocks() {
                if block.id == func.blocks()[0].id {
                    continue; // entry block
                }
                let Some(&bb) = self.block_labels.get(&block.id) else {
                    continue;
                };

                for instr in bb.get_instructions() {
                    if instr.get_opcode() == InstructionOpcode::PHI {
                        let phi = instr.try_as_phi().unwrap();
                        let preds: Vec<(&str, Option<BasicValueEnum>)> = phi
                            .get_name()
                            .to_str()
                            .ok()
                            .map(|name| {
                                let val_id_str = name.trim_start_matches('v');
                                val_id_str.parse::<u32>().ok().map(|id| {
                                    let vid = ValueId(id);
                                    (name, self.values.get(&vid).copied())
                                })
                            })
                            .flatten()
                            .unwrap_or_default();

                        // For each predecessor, find the arg for this param index.
                        if let Some(entries) = pred_map.get(&block.id) {
                            let param_idx = block.params.iter().position(|p| {
                                let name = format!("v{}", p.id.0);
                                phi.get_name().to_str().map(|n| n == name).unwrap_or(false)
                            });

                            if let Some(idx) = param_idx {
                                for (pred_id, args) in entries {
                                    if let Some(&pred_bb) = self.block_labels.get(pred_id) {
                                        if let Some(&val_id) = args.get(idx) {
                                            if let Some(bv) = self.values.get(&val_id) {
                                                phi.add_incoming(&[(&bv.into(), pred_bb)]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            Ok(())
        }

        fn compile_instr(
            &mut self,
            instr: &IrInstr,
            func: &IrFunction,
            module: &IrModule,
        ) -> Result<(), CodegenError> {
            match instr {
                // Constants
                IrInstr::ConstFloat { result, value, .. } => {
                    let f64_type = self.context.f64_type();
                    let val = f64_type.const_float(*value);
                    self.values.insert(*result, val.into());
                }
                IrInstr::ConstInt { result, value, .. } => {
                    let i64_type = self.context.i64_type();
                    let val = i64_type.const_int(*value as u64, true);
                    self.values.insert(*result, val.into());
                }
                IrInstr::ConstBool { result, value } => {
                    let bool_type = self.context.bool_type();
                    let val = bool_type.const_int(*value as u64, false);
                    self.values.insert(*result, val.into());
                }

                // Binary operations
                IrInstr::BinOp {
                    result,
                    op,
                    lhs,
                    rhs,
                    ty,
                } => {
                    let lv = self.get_value(*lhs)?;
                    let rv = self.get_value(*rhs)?;
                    let is_float = matches!(ty, IrType::Scalar(DType::F32 | DType::F64));

                    // For comparisons, the result is always bool.
                    let result_val = if matches!(
                        op,
                        BinOp::CmpEq
                            | BinOp::CmpNe
                            | BinOp::CmpLt
                            | BinOp::CmpLe
                            | BinOp::CmpGt
                            | BinOp::CmpGe
                    ) {
                        let pred = match op {
                            BinOp::CmpEq => IntPredicate::EQ,
                            BinOp::CmpNe => IntPredicate::NE,
                            BinOp::CmpLt => IntPredicate::SLT,
                            BinOp::CmpLe => IntPredicate::SLE,
                            BinOp::CmpGt => IntPredicate::SGT,
                            BinOp::CmpGe => IntPredicate::SGE,
                            _ => unreachable!(),
                        };

                        if is_float {
                            let float_pred = match op {
                                BinOp::CmpEq => inkwell::FloatPredicate::OEQ,
                                BinOp::CmpNe => inkwell::FloatPredicate::ONE,
                                BinOp::CmpLt => inkwell::FloatPredicate::OLT,
                                BinOp::CmpLe => inkwell::FloatPredicate::OLE,
                                BinOp::CmpGt => inkwell::FloatPredicate::OGT,
                                BinOp::CmpGe => inkwell::FloatPredicate::OGE,
                                _ => unreachable!(),
                            };
                            self.builder
                                .build_float_compare(
                                    float_pred,
                                    lv.into_float_value(),
                                    rv.into_float_value(),
                                    &format!("v{}", result.0),
                                )?
                                .into()
                        } else {
                            self.builder
                                .build_int_compare(
                                    pred,
                                    lv.into_int_value(),
                                    rv.into_int_value(),
                                    &format!("v{}", result.0),
                                )?
                                .into()
                        }
                    } else if is_float {
                        let lhs_f = lv.into_float_value();
                        let rhs_f = rv.into_float_value();
                        match op {
                            BinOp::Add => self.builder.build_float_add(lhs_f, rhs_f, &format!("v{}", result.0))?.into(),
                            BinOp::Sub => self.builder.build_float_sub(lhs_f, rhs_f, &format!("v{}", result.0))?.into(),
                            BinOp::Mul => self.builder.build_float_mul(lhs_f, rhs_f, &format!("v{}", result.0))?.into(),
                            BinOp::Div => self.builder.build_float_div(lhs_f, rhs_f, &format!("v{}", result.0))?.into(),
                            BinOp::Mod => self.builder.build_float_rem(lhs_f, rhs_f, &format!("v{}", result.0))?.into(),
                            _ => {
                                // Fall back to text backend for unsupported ops.
                                return Err(CodegenError::Unsupported {
                                    backend: "llvm_native".into(),
                                    detail: format!("unsupported float op: {:?}", op),
                                });
                            }
                        }
                    } else {
                        let lhs_i = lv.into_int_value();
                        let rhs_i = rv.into_int_value();
                        match op {
                            BinOp::Add => self.builder.build_int_add(lhs_i, rhs_i, &format!("v{}", result.0))?.into(),
                            BinOp::Sub => self.builder.build_int_sub(lhs_i, rhs_i, &format!("v{}", result.0))?.into(),
                            BinOp::Mul => self.builder.build_int_mul(lhs_i, rhs_i, &format!("v{}", result.0))?.into(),
                            BinOp::Div | BinOp::FloorDiv => {
                                self.builder.build_int_signed_div(lhs_i, rhs_i, &format!("v{}", result.0))?.into()
                            }
                            BinOp::Mod => self.builder.build_int_signed_rem(lhs_i, rhs_i, &format!("v{}", result.0))?.into(),
                            BinOp::BitAnd => self.builder.build_and(lhs_i, rhs_i, &format!("v{}", result.0))?.into(),
                            BinOp::BitOr => self.builder.build_or(lhs_i, rhs_i, &format!("v{}", result.0))?.into(),
                            BinOp::BitXor => self.builder.build_xor(lhs_i, rhs_i, &format!("v{}", result.0))?.into(),
                            BinOp::Shl => self.builder.build_left_shift(lhs_i, rhs_i, &format!("v{}", result.0))?.into(),
                            BinOp::Shr => self.builder.build_right_shift(lhs_i, rhs_i, true, &format!("v{}", result.0))?.into(),
                            _ => {
                                return Err(CodegenError::Unsupported {
                                    backend: "llvm_native".into(),
                                    detail: format!("unsupported int op: {:?}", op),
                                });
                            }
                        }
                    };
                    self.values.insert(*result, result_val);
                }

                // Unary operations
                IrInstr::UnaryOp {
                    result,
                    op,
                    operand,
                    ty,
                } => {
                    let ov = self.get_value(*operand)?;
                    let is_float = matches!(ty, IrType::Scalar(DType::F32 | DType::F64));
                    let result_val = if is_float {
                        let fv = ov.into_float_value();
                        match op {
                            ScalarUnaryOp::Neg => self.builder.build_float_neg(fv, &format!("v{}", result.0))?.into(),
                            ScalarUnaryOp::Abs => {
                                // abs(x) = x < 0 ? -x : x
                                let zero = fv.get_type().const_float(0.0);
                                let neg = self.builder.build_float_neg(fv, "abs_neg")?;
                                let cmp = self.builder.build_float_compare(
                                    inkwell::FloatPredicate::OLT,
                                    fv,
                                    zero,
                                    "abs_cmp",
                                )?;
                                self.builder.build_select(cmp, neg, fv, &format!("v{}", result.0))?.into()
                            }
                            _ => {
                                return Err(CodegenError::Unsupported {
                                    backend: "llvm_native".into(),
                                    detail: format!("unsupported float unary: {:?}", op),
                                });
                            }
                        }
                    } else {
                        let iv = ov.into_int_value();
                        match op {
                            ScalarUnaryOp::Neg => self.builder.build_int_neg(iv, &format!("v{}", result.0))?.into(),
                            ScalarUnaryOp::Not => self.builder.build_not(iv, &format!("v{}", result.0))?.into(),
                            ScalarUnaryOp::BitNot => self.builder.build_not(iv, &format!("v{}", result.0))?.into(),
                            _ => {
                                return Err(CodegenError::Unsupported {
                                    backend: "llvm_native".into(),
                                    detail: format!("unsupported int unary: {:?}", op),
                                });
                            }
                        }
                    };
                    self.values.insert(*result, result_val);
                }

                // Control flow
                IrInstr::Return { values } => {
                    if values.is_empty() {
                        self.builder.build_return(None)?;
                    } else {
                        let v = self.get_value(values[0])?;
                        self.builder.build_return(Some(&v))?;
                    }
                }
                IrInstr::Br { target, .. } => {
                    if let Some(&bb) = self.block_labels.get(target) {
                        self.builder.build_unconditional_branch(bb)?;
                    }
                }
                IrInstr::CondBr {
                    cond,
                    then_block,
                    else_block,
                    ..
                } => {
                    let cv = self.get_value(*cond)?.into_int_value();
                    let then_bb = self.block_labels.get(then_block).copied();
                    let else_bb = self.block_labels.get(else_block).copied();
                    if let (Some(then_bb), Some(else_bb)) = (then_bb, else_bb) {
                        self.builder.build_conditional_branch(cv, then_bb, else_bb)?;
                    }
                }

                // Function calls
                IrInstr::Call {
                    result,
                    callee,
                    args,
                    result_ty,
                } => {
                    let callee_fn = self.module.get_function(callee).ok_or_else(|| CodegenError::Unsupported {
                        backend: "llvm_native".into(),
                        detail: format!("function '{}' not found", callee),
                    })?;
                    let mut arg_vals = Vec::new();
                    for arg in args {
                        arg_vals.push(self.get_value(*arg)?);
                    }
                    let call_result = self.builder.build_call(callee_fn, &arg_vals, "call_tmp")?;
                    if let Some(r) = result {
                        if let Some(val) = call_result.try_as_basic_value().left() {
                            self.values.insert(*r, val);
                        }
                    }
                }

                // Struct operations
                IrInstr::MakeStruct {
                    result,
                    fields,
                    result_ty,
                } => {
                    // For structs, we allocate on heap and return pointer
                    let ptr_ty = self.context.i8_type().ptr_type(AddressSpace::from(0));
                    let malloc_fn = self.get_or_declare_malloc()?;
                    let size = self.calculate_struct_size(result_ty)?;
                    let size_val = self.context.i64_type().const_int(size as u64, false);
                    let malloc_call = self.builder.build_call(malloc_fn, &[size_val.into()], "struct_alloc")?;
                    let ptr = malloc_call.try_as_basic_value().left().unwrap().into_pointer_value();
                    self.builder.build_bitcast(ptr, ptr_ty, "struct_ptr")?;
                    // Store fields
                    for (i, field_val) in fields.iter().enumerate() {
                        let field_ptr = unsafe {
                            self.builder.build_gep(
                                ptr_ty,
                                ptr,
                                &[self.context.i64_type().const_int(i as u64, false)],
                                &format!("field_{}_ptr", i),
                            )?
                        };
                        let field_val = self.get_value(*field_val)?;
                        self.builder.build_store(field_ptr, field_val)?;
                    }
                    self.values.insert(*result, ptr.into());
                }

                IrInstr::MakeTraitObject {
                    result,
                    value,
                    target_trait,
                    concrete_ty,
                    ..
                } => {
                    // Allocate {ptr, ptr} struct (16 bytes), store data + vtable ptr.
                    let ptr_ty = self.context.i8_type().ptr_type(AddressSpace::from(0));
                    let malloc_fn = self.get_or_declare_malloc()?;
                    let size_val = self.context.i64_type().const_int(16, false);
                    let malloc_call = self
                        .builder
                        .build_call(malloc_fn, &[size_val.into()], "trait_obj_alloc")?;
                    let obj_ptr = malloc_call
                        .try_as_basic_value()
                        .left()
                        .unwrap()
                        .into_pointer_value();
                    // Field 0: data pointer.
                    let f0 = unsafe {
                        self.builder.build_gep(
                            ptr_ty,
                            obj_ptr,
                            &[self.context.i64_type().const_int(0, false)],
                            "to_f0",
                        )?
                    };
                    let data_val = self.get_value(*value)?;
                    self.builder.build_store(f0, data_val)?;
                    // Field 1: vtable pointer (decayed from the global).
                    let vtable_name = format!("vtable_{}__{}", target_trait, concrete_ty);
                    let vtable_global = self.module.get_global(&vtable_name).ok_or_else(|| {
                        CodegenError::Unsupported {
                            backend: "llvm_native".into(),
                            detail: format!("vtable global '{}' not found", vtable_name),
                        }
                    })?;
                    let f1 = unsafe {
                        self.builder.build_gep(
                            ptr_ty,
                            obj_ptr,
                            &[self.context.i64_type().const_int(1, false)],
                            "to_f1",
                        )?
                    };
                    let vtable_ptr = vtable_global.as_pointer_value();
                    self.builder.build_store(f1, vtable_ptr)?;
                    self.values.insert(*result, obj_ptr.into());
                }

                // Tuple operations
                IrInstr::MakeTuple {
                    result,
                    elements,
                    result_ty,
                } => {
                    // Similar to struct - heap allocate and store elements
                    let ptr_ty = self.context.i8_type().ptr_type(AddressSpace::from(0));
                    let malloc_fn = self.get_or_declare_malloc()?;
                    let size = elements.len() * 8; // 8 bytes per element (i64/pointer)
                    let size_val = self.context.i64_type().const_int(size as u64, false);
                    let malloc_call = self.builder.build_call(malloc_fn, &[size_val.into()], "tuple_alloc")?;
                    let ptr = malloc_call.try_as_basic_value().left().unwrap().into_pointer_value();
                    self.builder.build_bitcast(ptr, ptr_ty, "tuple_ptr")?;
                    for (i, elem_val) in elements.iter().enumerate() {
                        let elem_ptr = unsafe {
                            self.builder.build_gep(
                                ptr_ty,
                                ptr,
                                &[self.context.i64_type().const_int(i as u64, false)],
                                &format!("elem_{}_ptr", i),
                            )?
                        };
                        let v = self.get_value(*elem_val)?;
                        self.builder.build_store(elem_ptr, v)?;
                    }
                    self.values.insert(*result, ptr.into());
                }

                // Field access
                IrInstr::GetField {
                    result,
                    base,
                    field_index,
                    result_ty,
                } => {
                    let base_ptr = self.get_value(*base)?.into_pointer_value();
                    let field_ptr = unsafe {
                        self.builder.build_gep(
                            self.context.i8_type().ptr_type(AddressSpace::from(0)),
                            base_ptr,
                            &[self.context.i64_type().const_int(*field_index as u64, false)],
                            &format!("field_{}_ptr", field_index),
                        )?
                    };
                    let loaded = self.builder.build_load(self.ir_type_to_basic_type(result_ty)?.unwrap(), field_ptr, &format!("field_{}", field_index))?;
                    self.values.insert(*result, loaded);
                }

                // Tuple element access
                IrInstr::GetElement {
                    result,
                    base,
                    index,
                    result_ty,
                } => {
                    let base_ptr = self.get_value(*base)?.into_pointer_value();
                    let elem_ptr = unsafe {
                        self.builder.build_gep(
                            self.context.i8_type().ptr_type(AddressSpace::from(0)),
                            base_ptr,
                            &[self.context.i64_type().const_int(*index as u64, false)],
                            &format!("elem_{}_ptr", index),
                        )?
                    };
                    let loaded = self.builder.build_load(self.ir_type_to_basic_type(result_ty)?.unwrap(), elem_ptr, &format!("elem_{}", index))?;
                    self.values.insert(*result, loaded);
                }

                // Array allocation
                IrInstr::AllocArray {
                    result,
                    elem_ty,
                    len,
                } => {
                    let ptr_ty = self.context.i8_type().ptr_type(AddressSpace::from(0));
                    let malloc_fn = self.get_or_declare_malloc()?;
                    let elem_size = self.type_size(elem_ty)?;
                    let total_size = elem_size * (*len as u64);
                    let size_val = self.context.i64_type().const_int(total_size, false);
                    let malloc_call = self.builder.build_call(malloc_fn, &[size_val.into()], "array_alloc")?;
                    let ptr = malloc_call.try_as_basic_value().left().unwrap().into_pointer_value();
                    self.builder.build_bitcast(ptr, ptr_ty, "array_ptr")?;
                    self.values.insert(*result, ptr.into());
                }

                // Array load
                IrInstr::ArrayLoad {
                    result,
                    array,
                    index,
                    elem_ty,
                } => {
                    let array_ptr = self.get_value(*array)?.into_pointer_value();
                    let idx_val = self.get_value(*index)?.into_int_value();
                    let elem_ptr = unsafe {
                        self.builder.build_gep(
                            self.context.i8_type().ptr_type(AddressSpace::from(0)),
                            array_ptr,
                            &[idx_val],
                            &format!("array_elem_{}_ptr", result.0),
                        )?
                    };
                    let loaded = self.builder.build_load(self.ir_type_to_basic_type(elem_ty)?.unwrap(), elem_ptr, &format!("array_load_{}", result.0))?;
                    self.values.insert(*result, loaded);
                }

                // Array store
                IrInstr::ArrayStore {
                    array,
                    index,
                    value,
                } => {
                    let array_ptr = self.get_value(*array)?.into_pointer_value();
                    let idx_val = self.get_value(*index)?.into_int_value();
                    let value_val = self.get_value(*value)?;
                    let elem_ptr = unsafe {
                        self.builder.build_gep(
                            self.context.i8_type().ptr_type(AddressSpace::from(0)),
                            array_ptr,
                            &[idx_val],
                            "array_store_ptr",
                        )?
                    };
                    self.builder.build_store(elem_ptr, value_val)?;
                }

                // Cast operation
                IrInstr::Cast {
                    result,
                    operand,
                    to_ty,
                } => {
                    let op_val = self.get_value(*operand)?;
                    let from_ty = func.value_type(*operand).cloned().unwrap_or(IrType::Unit);
                    let result_val = self.build_cast(op_val, &from_ty, to_ty, &format!("cast_{}", result.0))?;
                    self.values.insert(*result, result_val);
                }

                // Channel operations — delegate to C runtime
                IrInstr::ChanNew { result, elem_ty, capacity } => {
                    let cap_val = self.get_value(*capacity)?;
                    let chan_new_fn = self.get_or_declare_runtime_fn_ret("iris_chan_new", &[self.context.i64_type().into()], self.ptr_ty().into())?;
                    let call_result = self.builder.build_call(chan_new_fn, &[cap_val.into()], "chan_new")?;
                    if let Some(val) = call_result.try_as_basic_value().left() {
                        self.values.insert(*result, val);
                    }
                }
                IrInstr::ChanSend { chan, value } => {
                    let chan_ptr = self.get_value(*chan)?;
                    let value_val = self.get_value(*value)?;
                    let send_fn = self.get_or_declare_runtime_fn_void("iris_chan_send", &[self.ptr_ty().into(), self.ptr_ty().into()])?;
                    self.builder.build_call(send_fn, &[chan_ptr.into(), value_val.into()], "")?;
                }
                IrInstr::ChanRecv { result, chan, elem_ty } => {
                    let chan_ptr = self.get_value(*chan)?;
                    let recv_fn = self.get_or_declare_runtime_fn_ret("iris_chan_recv", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let call_result = self.builder.build_call(recv_fn, &[chan_ptr.into()], "chan_recv")?;
                    if let Some(val) = call_result.try_as_basic_value().left() {
                        self.values.insert(*result, val);
                    }
                }

                // Spawn — delegate to runtime
                IrInstr::Spawn { body_fn, args } => {
                    // The spawn trampoline is handled by the runtime; we call
                    // iris_spawn_fn(fn_ptr, packed_args_ptr). For the native
                    // backend we currently emit a stub; the real trampoline
                    // generation matches the text backend's pattern.
                    let spawn_fn = self.get_or_declare_runtime_fn_void(
                        "iris_spawn_fn",
                        &[self.ptr_ty().into(), self.ptr_ty().into()],
                    )?;
                    // Get the body function pointer
                    if let Some(fn_val) = self.module.get_function(body_fn) {
                        let fn_ptr = fn_val.as_global_value().as_pointer_value();
                        // Pack args into a malloc'd buffer
                        let n_args = args.len();
                        let buf_size = self.context.i64_type().const_int((n_args * 8) as u64, false);
                        let malloc_fn = self.get_or_declare_malloc()?;
                        let buf = self.builder.build_call(malloc_fn, &[buf_size.into()], "spawn_buf")?
                            .try_as_basic_value().left().unwrap().into_pointer_value();
                        for (i, arg) in args.iter().enumerate() {
                            let av = self.get_value(*arg)?;
                            let slot = unsafe {
                                self.builder.build_gep(
                                    self.ptr_ty(),
                                    buf,
                                    &[self.context.i64_type().const_int(i as u64, false)],
                                    &format!("spawn_arg_{}", i),
                                )?
                            };
                            self.builder.build_store(slot, av)?;
                        }
                        self.builder.build_call(spawn_fn, &[fn_ptr.into(), buf.into()], "")?;
                    }
                }

                // TaskGroup operations
                IrInstr::TaskGroupNew { result } => {
                    let tg_new_fn = self.get_or_declare_runtime_fn_ret("iris_task_group_new", &[], self.ptr_ty().into())?;
                    let call_result = self.builder.build_call(tg_new_fn, &[], "tg_new")?;
                    if let Some(val) = call_result.try_as_basic_value().left() {
                        self.values.insert(*result, val);
                    }
                }
                IrInstr::TaskGroupSpawn { group, body_fn, args } => {
                    let tg_spawn_fn = self.get_or_declare_runtime_fn_void(
                        "iris_task_group_spawn",
                        &[self.ptr_ty().into(), self.ptr_ty().into(), self.ptr_ty().into()],
                    )?;
                    let tg_ptr = self.get_value(*group)?;
                    if let Some(fn_val) = self.module.get_function(body_fn) {
                        let fn_ptr = fn_val.as_global_value().as_pointer_value();
                        let n_args = args.len();
                        let buf_size = self.context.i64_type().const_int((n_args * 8) as u64, false);
                        let malloc_fn = self.get_or_declare_malloc()?;
                        let buf = if n_args > 0 {
                            let b = self.builder.build_call(malloc_fn, &[buf_size.into()], "tg_spawn_buf")?
                                .try_as_basic_value().left().unwrap().into_pointer_value();
                            for (i, arg) in args.iter().enumerate() {
                                let av = self.get_value(*arg)?;
                                let slot = unsafe {
                                    self.builder.build_gep(
                                        self.ptr_ty(),
                                        b,
                                        &[self.context.i64_type().const_int(i as u64, false)],
                                        &format!("tg_arg_{}", i),
                                    )?
                                };
                                self.builder.build_store(slot, av)?;
                            }
                            b.into()
                        } else {
                            self.ptr_ty().const_null().into()
                        };
                        self.builder.build_call(tg_spawn_fn, &[tg_ptr.into(), fn_ptr.into(), buf.into()], "")?;
                    }
                }
                IrInstr::TaskGroupJoin { group } => {
                    let tg_join_fn = self.get_or_declare_runtime_fn_void("iris_task_group_join", &[self.ptr_ty().into()])?;
                    let tg_ptr = self.get_value(*group)?;
                    self.builder.build_call(tg_join_fn, &[tg_ptr.into()], "")?;
                }
                IrInstr::TaskGroupCancel { group } => {
                    let tg_cancel_fn = self.get_or_declare_runtime_fn_void("iris_task_group_cancel", &[self.ptr_ty().into()])?;
                    let tg_ptr = self.get_value(*group)?;
                    self.builder.build_call(tg_cancel_fn, &[tg_ptr.into()], "")?;
                }

                // List operations — delegate to C runtime
                IrInstr::ListNew { result, .. } => {
                    let list_new_fn = self.get_or_declare_runtime_fn_ret("iris_list_new", &[], self.ptr_ty().into())?;
                    let call_result = self.builder.build_call(list_new_fn, &[], "list_new")?;
                    if let Some(val) = call_result.try_as_basic_value().left() {
                        self.values.insert(*result, val);
                    }
                }
                IrInstr::ListPush { list, value } => {
                    let list_ptr = self.get_value(*list)?;
                    let value_val = self.get_value(*value)?;
                    let push_fn = self.get_or_declare_runtime_fn_void("iris_list_push", &[self.ptr_ty().into(), self.ptr_ty().into()])?;
                    self.builder.build_call(push_fn, &[list_ptr.into(), value_val.into()], "")?;
                }
                IrInstr::ListGet { result, list, index, .. } => {
                    let list_ptr = self.get_value(*list)?;
                    let idx_val = self.get_value(*index)?;
                    let get_fn = self.get_or_declare_runtime_fn_ret("iris_list_get", &[self.ptr_ty().into(), self.context.i64_type().into()], self.ptr_ty().into())?;
                    let call_result = self.builder.build_call(get_fn, &[list_ptr.into(), idx_val.into()], "list_get")?;
                    if let Some(val) = call_result.try_as_basic_value().left() {
                        self.values.insert(*result, val);
                    }
                }
                IrInstr::ListSet { list, index, value } => {
                    let list_ptr = self.get_value(*list)?;
                    let idx_val = self.get_value(*index)?;
                    let value_val = self.get_value(*value)?;
                    let set_fn = self.get_or_declare_runtime_fn_void("iris_list_set", &[self.ptr_ty().into(), self.context.i64_type().into(), self.ptr_ty().into()])?;
                    self.builder.build_call(set_fn, &[list_ptr.into(), idx_val.into(), value_val.into()], "")?;
                }
                IrInstr::ListPop { result, list, .. } => {
                    let list_ptr = self.get_value(*list)?;
                    let pop_fn = self.get_or_declare_runtime_fn_ret("iris_list_pop", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let call_result = self.builder.build_call(pop_fn, &[list_ptr.into()], "list_pop")?;
                    if let Some(val) = call_result.try_as_basic_value().left() {
                        self.values.insert(*result, val);
                    }
                }
                IrInstr::ListLen { result, list } => {
                    let list_ptr = self.get_value(*list)?;
                    let len_fn = self.get_or_declare_runtime_fn_ret("iris_list_len", &[self.ptr_ty().into()], self.context.i64_type().into())?;
                    let call_result = self.builder.build_call(len_fn, &[list_ptr.into()], "list_len")?;
                    if let Some(val) = call_result.try_as_basic_value().left() {
                        self.values.insert(*result, val);
                    }
                }
                IrInstr::ListContains { result, list, value } => {
                    let lp = self.get_value(*list)?;
                    let vp = self.get_value(*value)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_list_contains", &[self.ptr_ty().into(), self.ptr_ty().into()], self.context.bool_type().into())?;
                    let r = self.builder.build_call(f, &[lp.into(), vp.into()], "list_contains")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }
                IrInstr::ListSort { list } => {
                    let lp = self.get_value(*list)?;
                    let f = self.get_or_declare_runtime_fn_void("iris_list_sort", &[self.ptr_ty().into()])?;
                    self.builder.build_call(f, &[lp.into()], "")?;
                }
                IrInstr::ListConcat { result, lhs, rhs } => {
                    let a = self.get_value(*lhs)?;
                    let b = self.get_value(*rhs)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_list_concat", &[self.ptr_ty().into(), self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[a.into(), b.into()], "list_concat")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }
                IrInstr::ListSlice { result, list, start, end } => {
                    let lp = self.get_value(*list)?;
                    let s = self.get_value(*start)?;
                    let e = self.get_value(*end)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_list_slice", &[self.ptr_ty().into(), self.context.i64_type().into(), self.context.i64_type().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[lp.into(), s.into(), e.into()], "list_slice")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }

                // Map operations — delegate to C runtime
                IrInstr::MapNew { result, .. } => {
                    let f = self.get_or_declare_runtime_fn_ret("iris_map_new", &[], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[], "map_new")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }
                IrInstr::MapSet { map, key, value } => {
                    let mp = self.get_value(*map)?;
                    let kv = self.get_value(*key)?;
                    let vv = self.get_value(*value)?;
                    let f = self.get_or_declare_runtime_fn_void("iris_map_set", &[self.ptr_ty().into(), self.ptr_ty().into(), self.ptr_ty().into()])?;
                    self.builder.build_call(f, &[mp.into(), kv.into(), vv.into()], "")?;
                }
                IrInstr::MapGet { result, map, key, .. } => {
                    let mp = self.get_value(*map)?;
                    let kv = self.get_value(*key)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_map_get", &[self.ptr_ty().into(), self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[mp.into(), kv.into()], "map_get")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }
                IrInstr::MapContains { result, map, key } => {
                    let mp = self.get_value(*map)?;
                    let kv = self.get_value(*key)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_map_contains", &[self.ptr_ty().into(), self.ptr_ty().into()], self.context.bool_type().into())?;
                    let r = self.builder.build_call(f, &[mp.into(), kv.into()], "map_contains")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }
                IrInstr::MapRemove { map, key } => {
                    let mp = self.get_value(*map)?;
                    let kv = self.get_value(*key)?;
                    let f = self.get_or_declare_runtime_fn_void("iris_map_remove", &[self.ptr_ty().into(), self.ptr_ty().into()])?;
                    self.builder.build_call(f, &[mp.into(), kv.into()], "")?;
                }
                IrInstr::MapLen { result, map } => {
                    let mp = self.get_value(*map)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_map_len", &[self.ptr_ty().into()], self.context.i64_type().into())?;
                    let r = self.builder.build_call(f, &[mp.into()], "map_len")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }
                IrInstr::MapKeys { result, map } => {
                    let mp = self.get_value(*map)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_map_keys", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[mp.into()], "map_keys")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }
                IrInstr::MapValues { result, map } => {
                    let mp = self.get_value(*map)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_map_values", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[mp.into()], "map_values")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }

                // Option/Result operations
                IrInstr::MakeSome { result, value, .. } => {
                    let val = self.get_value(*value)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_make_some", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[val.into()], "make_some")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::MakeNone { result, .. } => {
                    let f = self.get_or_declare_runtime_fn_ret("iris_make_none", &[], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[], "make_none")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::IsSome { result, operand } => {
                    let val = self.get_value(*operand)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_is_some", &[self.ptr_ty().into()], self.context.bool_type().into())?;
                    let r = self.builder.build_call(f, &[val.into()], "is_some")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::OptionUnwrap { result, operand, .. } => {
                    let val = self.get_value(*operand)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_option_unwrap", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[val.into()], "option_unwrap")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::MakeOk { result, value, .. } => {
                    let val = self.get_value(*value)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_make_ok", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[val.into()], "make_ok")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::MakeErr { result, value, .. } => {
                    let val = self.get_value(*value)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_make_err", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[val.into()], "make_err")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::IsOk { result, operand } => {
                    let val = self.get_value(*operand)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_is_ok", &[self.ptr_ty().into()], self.context.bool_type().into())?;
                    let r = self.builder.build_call(f, &[val.into()], "is_ok")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::ResultUnwrap { result, operand, .. } => {
                    let val = self.get_value(*operand)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_result_unwrap", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[val.into()], "result_unwrap")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::ResultUnwrapErr { result, operand, .. } => {
                    let val = self.get_value(*operand)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_result_unwrap_err", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[val.into()], "result_unwrap_err")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }

                // String operations — delegate to C runtime
                IrInstr::ConstStr { result, value } => {
                    // Create a global string constant and get a pointer to it
                    let str_val = self.builder.build_global_string_ptr(value, &format!("str_{}", result.0))?;
                    self.values.insert(*result, str_val.as_pointer_value().into());
                }
                IrInstr::StrLen { result, operand } => {
                    let s = self.get_value(*operand)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_str_len", &[self.ptr_ty().into()], self.context.i64_type().into())?;
                    let r = self.builder.build_call(f, &[s.into()], "str_len")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::StrConcat { result, lhs, rhs } => {
                    let a = self.get_value(*lhs)?;
                    let b = self.get_value(*rhs)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_str_concat", &[self.ptr_ty().into(), self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[a.into(), b.into()], "str_concat")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::StrEq { result, lhs, rhs } => {
                    let a = self.get_value(*lhs)?;
                    let b = self.get_value(*rhs)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_str_eq", &[self.ptr_ty().into(), self.ptr_ty().into()], self.context.bool_type().into())?;
                    let r = self.builder.build_call(f, &[a.into(), b.into()], "str_eq")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::StrContains { result, haystack, needle } => {
                    let h = self.get_value(*haystack)?;
                    let n = self.get_value(*needle)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_str_contains", &[self.ptr_ty().into(), self.ptr_ty().into()], self.context.bool_type().into())?;
                    let r = self.builder.build_call(f, &[h.into(), n.into()], "str_contains")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::StrStartsWith { result, haystack, prefix } => {
                    let h = self.get_value(*haystack)?;
                    let p = self.get_value(*prefix)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_str_starts_with", &[self.ptr_ty().into(), self.ptr_ty().into()], self.context.bool_type().into())?;
                    let r = self.builder.build_call(f, &[h.into(), p.into()], "str_starts_with")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::StrEndsWith { result, haystack, suffix } => {
                    let h = self.get_value(*haystack)?;
                    let s = self.get_value(*suffix)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_str_ends_with", &[self.ptr_ty().into(), self.ptr_ty().into()], self.context.bool_type().into())?;
                    let r = self.builder.build_call(f, &[h.into(), s.into()], "str_ends_with")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::StrToUpper { result, operand } => {
                    let s = self.get_value(*operand)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_str_to_upper", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[s.into()], "str_to_upper")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::StrToLower { result, operand } => {
                    let s = self.get_value(*operand)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_str_to_lower", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[s.into()], "str_to_lower")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::StrTrim { result, operand } => {
                    let s = self.get_value(*operand)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_str_trim", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[s.into()], "str_trim")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::StrRepeat { result, operand, count } => {
                    let s = self.get_value(*operand)?;
                    let n = self.get_value(*count)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_str_repeat", &[self.ptr_ty().into(), self.context.i64_type().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[s.into(), n.into()], "str_repeat")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::StrIndex { result, string, index } => {
                    let s = self.get_value(*string)?;
                    let i = self.get_value(*index)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_str_index", &[self.ptr_ty().into(), self.context.i64_type().into()], self.context.i64_type().into())?;
                    let r = self.builder.build_call(f, &[s.into(), i.into()], "str_index")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::StrSlice { result, string, start, end } => {
                    let s = self.get_value(*string)?;
                    let st = self.get_value(*start)?;
                    let en = self.get_value(*end)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_str_slice", &[self.ptr_ty().into(), self.context.i64_type().into(), self.context.i64_type().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[s.into(), st.into(), en.into()], "str_slice")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::StrFind { result, haystack, needle } => {
                    let h = self.get_value(*haystack)?;
                    let n = self.get_value(*needle)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_str_find", &[self.ptr_ty().into(), self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[h.into(), n.into()], "str_find")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::StrReplace { result, string, from, to } => {
                    let s = self.get_value(*string)?;
                    let fr = self.get_value(*from)?;
                    let t = self.get_value(*to)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_str_replace", &[self.ptr_ty().into(), self.ptr_ty().into(), self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[s.into(), fr.into(), t.into()], "str_replace")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::ValueToStr { result, operand } => {
                    let v = self.get_value(*operand)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_value_to_str", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[v.into()], "to_str")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }

                // Print/Panic
                IrInstr::Print { operand } => {
                    let v = self.get_value(*operand)?;
                    let f = self.get_or_declare_runtime_fn_void("iris_print", &[self.ptr_ty().into()])?;
                    self.builder.build_call(f, &[v.into()], "")?;
                }
                IrInstr::Panic { msg } => {
                    let v = self.get_value(*msg)?;
                    let f = self.get_or_declare_runtime_fn_void("iris_panic", &[self.ptr_ty().into()])?;
                    self.builder.build_call(f, &[v.into()], "")?;
                    self.builder.build_unreachable()?;
                }

                // User input
                IrInstr::ReadLine { result } => {
                    let f = self.get_or_declare_runtime_fn_ret("iris_read_line", &[], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[], "read_line")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::ReadI64 { result } => {
                    let f = self.get_or_declare_runtime_fn_ret("iris_read_i64", &[], self.context.i64_type().into())?;
                    let r = self.builder.build_call(f, &[], "read_i64")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::ReadF64 { result } => {
                    let f = self.get_or_declare_runtime_fn_ret("iris_read_f64", &[], self.context.f64_type().into())?;
                    let r = self.builder.build_call(f, &[], "read_f64")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::ParseI64 { result, operand } => {
                    let s = self.get_value(*operand)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_parse_i64", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[s.into()], "parse_i64")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::ParseF64 { result, operand } => {
                    let s = self.get_value(*operand)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_parse_f64", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[s.into()], "parse_f64")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }

                // Variant (enum) operations — delegate to C runtime
                IrInstr::MakeVariant { result, variant_idx, fields, .. } => {
                    // For now, use runtime call with tag + field count
                    let tag = self.context.i64_type().const_int(*variant_idx as u64, false);
                    let nfields = self.context.i32_type().const_int(fields.len() as u64, false);
                    // Pack into iris_make_variant(tag, nfields, field0, field1, ...)
                    // Since variadic, just pass tag for now and handle in runtime
                    let f = self.get_or_declare_runtime_fn_ret("iris_make_variant", &[self.context.i64_type().into(), self.context.i32_type().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[tag.into(), nfields.into()], "make_variant")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::GetVariantTag { result, operand } => {
                    let v = self.get_value(*operand)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_get_variant_tag", &[self.ptr_ty().into()], self.context.i64_type().into())?;
                    let r = self.builder.build_call(f, &[v.into()], "get_tag")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }
                IrInstr::ExtractVariantField { result, operand, field_idx, .. } => {
                    let v = self.get_value(*operand)?;
                    let idx = self.context.i64_type().const_int(*field_idx as u64, false);
                    let f = self.get_or_declare_runtime_fn_ret("iris_extract_variant_field", &[self.ptr_ty().into(), self.context.i64_type().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[v.into(), idx.into()], "extract_field")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }
                IrInstr::SwitchVariant { scrutinee, arms, default_block } => {
                    let tag_val = self.get_value(*scrutinee)?;
                    // Get the tag: call iris_get_variant_tag
                    let tag_fn = self.get_or_declare_runtime_fn_ret("iris_get_variant_tag", &[self.ptr_ty().into()], self.context.i64_type().into())?;
                    let tag = self.builder.build_call(tag_fn, &[tag_val.into()], "switch_tag")?
                        .try_as_basic_value().left().unwrap().into_int_value();
                    let default_bb = default_block
                        .and_then(|b| self.block_labels.get(&b).copied())
                        .or_else(|| arms.first().and_then(|(_, b)| self.block_labels.get(b).copied()));
                    if let Some(def_bb) = default_bb {
                        let switch = self.builder.build_switch(tag, def_bb, &[])?;
                        for (idx, target) in arms {
                            if let Some(&bb) = self.block_labels.get(target) {
                                let case_val = self.context.i64_type().const_int(*idx as u64, false);
                                switch.add_case(case_val, bb);
                            }
                        }
                    }
                }

                // Closure operations
                IrInstr::MakeClosure { result, fn_name, captures, .. } => {
                    // Get the function pointer
                    if let Some(fn_val) = self.module.get_function(fn_name) {
                        let fn_ptr = fn_val.as_global_value().as_pointer_value();
                        let n_caps = self.context.i32_type().const_int(captures.len() as u64, false);
                        let make_fn = self.get_or_declare_runtime_fn_ret("iris_make_closure", &[self.ptr_ty().into(), self.context.i32_type().into()], self.ptr_ty().into())?;
                        let r = self.builder.build_call(make_fn, &[fn_ptr.into(), n_caps.into()], "make_closure")?;
                        if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                    } else {
                        // Function not in module — create null pointer as fallback
                        let null = self.ptr_ty().const_null();
                        self.values.insert(*result, null.into());
                    }
                }
                IrInstr::CallClosure { result, closure, args, result_ty, pass_env } => {
                    let clo = self.get_value(*closure)?;
                    // Get function pointer from closure
                    let fn_ptr_fn = self.get_or_declare_runtime_fn_ret("iris_closure_fn", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let fn_ptr = self.builder.build_call(fn_ptr_fn, &[clo.into()], "closure_fn")?
                        .try_as_basic_value().left().unwrap();
                    // For simplicity, call through runtime
                    let call_fn = self.get_or_declare_runtime_fn_ret("iris_call_closure", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(call_fn, &[clo.into()], "call_closure")?;
                    if let Some(res) = result {
                        if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*res, v); }
                    }
                }

                IrInstr::DynCall {
                    result,
                    obj,
                    method_name,
                    args,
                    ..
                } => {
                    // Inline vtable dispatch via inkwell API.
                    let ptr_ty = self.context.i8_type().ptr_type(AddressSpace::from(0));
                    let obj_val = self.get_value(*obj)?.into_pointer_value();
                    // Determine the trait name from the value type.
                    let obj_ty = func.value_type(*obj);
                    let trait_name = match obj_ty {
                        Some(IrType::TraitObject { name, .. }) => name.clone(),
                        _ => "_unknown".to_owned(),
                    };
                    // Resolve method slot index from trait_def.
                    let slot = module
                        .trait_def(&trait_name)
                        .and_then(|methods| methods.iter().position(|m| m.name == *method_name));
                    let Some(slot_idx) = slot else {
                        return Err(CodegenError::Unsupported {
                            backend: "llvm_native".into(),
                            detail: format!(
                                "DynCall: unknown method '{}' on trait '{}'",
                                method_name, trait_name
                            ),
                        });
                    };
                    // Load data ptr from offset 0 of the trait object.
                    let f0 = unsafe {
                        self.builder.build_gep(
                            ptr_ty,
                            obj_val,
                            &[self.context.i64_type().const_int(0, false)],
                            "dyn_f0",
                        )?
                    };
                    let data_ptr = self
                        .builder
                        .build_load(ptr_ty, f0, "dyn_data")?
                        .into_pointer_value();
                    // Load vtable ptr from offset 1.
                    let f1 = unsafe {
                        self.builder.build_gep(
                            ptr_ty,
                            obj_val,
                            &[self.context.i64_type().const_int(1, false)],
                            "dyn_f1",
                        )?
                    };
                    let vtable_ptr = self
                        .builder
                        .build_load(ptr_ty, f1, "dyn_vtable")?
                        .into_pointer_value();
                    // GEP into vtable at slot index.
                    let slot_gep = unsafe {
                        self.builder.build_gep(
                            ptr_ty,
                            vtable_ptr,
                            &[
                                self.context.i64_type().const_int(0, false),
                                self.context.i64_type().const_int(slot_idx as u64, false),
                            ],
                            "vt_slot",
                        )?
                    };
                    let fn_ptr = self
                        .builder
                        .build_load(ptr_ty, slot_gep, "vt_fn")?
                        .into_pointer_value();
                    // Build argument list: data ptr first, then remaining args.
                    let mut call_args = vec![data_ptr.into()];
                    for a in args {
                        call_args.push(self.get_value(*a)?);
                    }
                    // Determine return type.
                    let ret_basic = ir_type_to_basic_type(self.context, result_ty)?
                        .unwrap_or(self.ptr_ty().into());
                    // Create function type for indirect call.
                    let param_types: Vec<BasicTypeEnum> =
                        std::iter::once(self.ptr_ty().into())
                            .chain(args.iter().map(|_| self.ptr_ty().into()))
                            .collect();
                    let fn_type = match ret_basic {
                        BasicTypeEnum::VoidType(v) => v.fn_type(&param_types, false).into(),
                        _ => ret_basic.fn_type(&param_types, false).into(),
                    };
                    let fn_val = unsafe {
                        self.builder
                            .build_indirect_call(fn_type, fn_ptr, &call_args, "dyn_call")?
                    };
                    if let Some(r) = result {
                        if let Some(val) = fn_val.try_as_basic_value().left() {
                            self.values.insert(*r, val);
                        }
                    }
                }

                // File I/O — delegate to C runtime
                IrInstr::FileReadAll { result, path } => {
                    let p = self.get_value(*path)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_file_read_all", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[p.into()], "file_read")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::FileWriteAll { result, path, content } => {
                    let p = self.get_value(*path)?;
                    let c = self.get_value(*content)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_file_write_all", &[self.ptr_ty().into(), self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[p.into(), c.into()], "file_write")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::FileExists { result, path } => {
                    let p = self.get_value(*path)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_file_exists", &[self.ptr_ty().into()], self.context.bool_type().into())?;
                    let r = self.builder.build_call(f, &[p.into()], "file_exists")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::FileLines { result, path } => {
                    let p = self.get_value(*path)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_file_lines", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[p.into()], "file_lines")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }

                // Database operations — delegate to C runtime
                IrInstr::DbOpen { result, path } => {
                    let p = self.get_value(*path)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_db_open", &[self.ptr_ty().into()], self.context.i64_type().into())?;
                    let r = self.builder.build_call(f, &[p.into()], "db_open")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::DbExec { result, db, sql } => {
                    let d = self.get_value(*db)?;
                    let s = self.get_value(*sql)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_db_exec", &[self.context.i64_type().into(), self.ptr_ty().into()], self.context.i64_type().into())?;
                    let r = self.builder.build_call(f, &[d.into(), s.into()], "db_exec")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::DbExecParams { result, db, sql, params } => {
                    let d = self.get_value(*db)?;
                    let s = self.get_value(*sql)?;
                    let p = self.get_value(*params)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_db_exec_params", &[self.context.i64_type().into(), self.ptr_ty().into(), self.ptr_ty().into()], self.context.i64_type().into())?;
                    let r = self.builder.build_call(f, &[d.into(), s.into(), p.into()], "db_exec_params")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::DbQuery { result, db, sql } => {
                    let d = self.get_value(*db)?;
                    let s = self.get_value(*sql)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_db_query", &[self.context.i64_type().into(), self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[d.into(), s.into()], "db_query")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::DbQueryParams { result, db, sql, params } => {
                    let d = self.get_value(*db)?;
                    let s = self.get_value(*sql)?;
                    let p = self.get_value(*params)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_db_query_params", &[self.context.i64_type().into(), self.ptr_ty().into(), self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[d.into(), s.into(), p.into()], "db_query_params")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::DbClose { result, db } => {
                    let d = self.get_value(*db)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_db_close", &[self.context.i64_type().into()], self.context.i64_type().into())?;
                    let r = self.builder.build_call(f, &[d.into()], "db_close")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }

                // Process / environment
                IrInstr::ProcessExit { code } => {
                    let c = self.get_value(*code)?;
                    let exit_fn = self.get_or_declare_runtime_fn_void("exit", &[self.context.i32_type().into()])?;
                    let code_i32 = self.builder.build_int_truncate(c.into_int_value(), self.context.i32_type(), "exit_code")?;
                    self.builder.build_call(exit_fn, &[code_i32.into()], "")?;
                    self.builder.build_unreachable()?;
                }
                IrInstr::ProcessArgs { result } => {
                    let f = self.get_or_declare_runtime_fn_ret("iris_process_args", &[], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[], "process_args")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::EnvVar { result, name } => {
                    let n = self.get_value(*name)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_env_var", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[n.into()], "env_var")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }

                // AD/Tape operations — delegate to C runtime
                IrInstr::MakeGrad { result, value, tangent, .. } => {
                    let v = self.get_value(*value)?;
                    let t = self.get_value(*tangent)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_make_grad", &[self.context.f64_type().into(), self.context.f64_type().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[v.into(), t.into()], "make_grad")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }
                IrInstr::GradValue { result, operand, .. } => {
                    let v = self.get_value(*operand)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_grad_value", &[self.ptr_ty().into()], self.context.f64_type().into())?;
                    let r = self.builder.build_call(f, &[v.into()], "grad_value")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }
                IrInstr::GradTangent { result, operand, .. } => {
                    let v = self.get_value(*operand)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_grad_tangent", &[self.ptr_ty().into()], self.context.f64_type().into())?;
                    let r = self.builder.build_call(f, &[v.into()], "grad_tangent")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }
                IrInstr::TapeRecord { result, value, op, parents } => {
                    // Simplified: just pass value through for now
                    let v = self.get_value(*value)?;
                    self.values.insert(*result, v);
                }
                IrInstr::Backward { result, loss } => {
                    let l = self.get_value(*loss)?;
                    let f = self.get_or_declare_runtime_fn_void("iris_backward", &[self.ptr_ty().into()])?;
                    self.builder.build_call(f, &[l.into()], "")?;
                    // Result is unit
                    self.values.insert(*result, self.context.i64_type().const_int(0, false).into());
                }
                IrInstr::TapeGrad { result, tape_node } => {
                    let n = self.get_value(*tape_node)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_tape_grad", &[self.ptr_ty().into()], self.context.f64_type().into())?;
                    let r = self.builder.build_call(f, &[n.into()], "tape_grad")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }

                // Atomic / Mutex — delegate to C runtime
                IrInstr::AtomicNew { result, value, .. } => {
                    let v = self.get_value(*value)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_atomic_new", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[v.into()], "atomic_new")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }
                IrInstr::AtomicLoad { result, atomic, .. } => {
                    let a = self.get_value(*atomic)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_atomic_load", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[a.into()], "atomic_load")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }
                IrInstr::AtomicStore { atomic, value } => {
                    let a = self.get_value(*atomic)?;
                    let v = self.get_value(*value)?;
                    let f = self.get_or_declare_runtime_fn_void("iris_atomic_store", &[self.ptr_ty().into(), self.ptr_ty().into()])?;
                    self.builder.build_call(f, &[a.into(), v.into()], "")?;
                }
                IrInstr::AtomicAdd { result, atomic, value, .. } => {
                    let a = self.get_value(*atomic)?;
                    let v = self.get_value(*value)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_atomic_add", &[self.ptr_ty().into(), self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[a.into(), v.into()], "atomic_add")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }
                IrInstr::MutexNew { result, value, .. } => {
                    let v = self.get_value(*value)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_mutex_new", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[v.into()], "mutex_new")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }
                IrInstr::MutexLock { result, mutex, .. } => {
                    let m = self.get_value(*mutex)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_mutex_lock", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[m.into()], "mutex_lock")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }
                IrInstr::MutexUnlock { mutex } => {
                    let m = self.get_value(*mutex)?;
                    let f = self.get_or_declare_runtime_fn_void("iris_mutex_unlock", &[self.ptr_ty().into()])?;
                    self.builder.build_call(f, &[m.into()], "")?;
                }

                // GC reference counting
                IrInstr::Retain { ptr } => {
                    let p = self.get_value(*ptr)?;
                    let f = self.get_or_declare_runtime_fn_void("iris_retain", &[self.ptr_ty().into()])?;
                    self.builder.build_call(f, &[p.into()], "")?;
                }
                IrInstr::Release { ptr, .. } => {
                    let p = self.get_value(*ptr)?;
                    let f = self.get_or_declare_runtime_fn_void("iris_release", &[self.ptr_ty().into()])?;
                    self.builder.build_call(f, &[p.into()], "")?;
                }

                // Extern function calls
                IrInstr::CallExtern { result, name, args, ret_ty } => {
                    let mut arg_vals: Vec<BasicValueEnum> = Vec::new();
                    for a in args { arg_vals.push(self.get_value(*a)?); }
                    // Declare or get the extern function
                    let param_types: Vec<BasicTypeEnum> = arg_vals.iter().map(|v| v.get_type()).collect();
                    let ret = ir_type_to_basic_type(self.context, ret_ty)?;
                    let fn_type = match ret {
                        Some(bt) => bt.fn_type(&param_types, false),
                        None => self.context.void_type().fn_type(&param_types, false),
                    };
                    let ext_fn = if let Some(f) = self.module.get_function(name) {
                        f
                    } else {
                        self.module.add_function(name, fn_type, None)
                    };
                    let call_args: Vec<_> = arg_vals.iter().map(|v| (*v).into()).collect();
                    let call_result = self.builder.build_call(ext_fn, &call_args, &format!("ext_{}", name))?;
                    if let Some(r) = result {
                        if let Some(val) = call_result.try_as_basic_value().left() {
                            self.values.insert(*r, val);
                        }
                    }
                }

                // Barrier — no-op in sequential backends
                IrInstr::Barrier => {}

                // ParFor — delegate to runtime or expand as sequential loop
                IrInstr::ParFor { var, start, end, body_fn, args, .. } => {
                    // In the native backend, par_for is executed as a regular for loop
                    let start_val = self.get_value(*start)?.into_int_value();
                    let end_val = self.get_value(*end)?.into_int_value();
                    // For now, call runtime par_for which does sequential execution
                    if let Some(fn_val) = self.module.get_function(body_fn) {
                        let fn_ptr = fn_val.as_global_value().as_pointer_value();
                        let par_fn = self.get_or_declare_runtime_fn_void(
                            "iris_par_for",
                            &[self.ptr_ty().into(), self.context.i64_type().into(), self.context.i64_type().into()],
                        )?;
                        self.builder.build_call(par_fn, &[fn_ptr.into(), start_val.into(), end_val.into()], "")?;
                    }
                }

                // Sparse tensor ops — delegate to runtime
                IrInstr::Sparsify { result, operand, .. } => {
                    let v = self.get_value(*operand)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_sparsify", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[v.into()], "sparsify")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }
                IrInstr::Densify { result, operand, .. } => {
                    let v = self.get_value(*operand)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_densify", &[self.ptr_ty().into()], self.ptr_ty().into())?;
                    let r = self.builder.build_call(f, &[v.into()], "densify")?;
                    if let Some(val) = r.try_as_basic_value().left() { self.values.insert(*result, val); }
                }

                // TCP networking — delegate to runtime
                IrInstr::TcpConnect { result, host, port } => {
                    let h = self.get_value(*host)?;
                    let p = self.get_value(*port)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_tcp_connect", &[self.ptr_ty().into(), self.context.i64_type().into()], self.context.i64_type().into())?;
                    let r = self.builder.build_call(f, &[h.into(), p.into()], "tcp_connect")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::TcpListen { result, port } => {
                    let p = self.get_value(*port)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_tcp_listen", &[self.context.i64_type().into()], self.context.i64_type().into())?;
                    let r = self.builder.build_call(f, &[p.into()], "tcp_listen")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }
                IrInstr::TcpAccept { result, listener } => {
                    let l = self.get_value(*listener)?;
                    let f = self.get_or_declare_runtime_fn_ret("iris_tcp_accept", &[self.context.i64_type().into()], self.context.i64_type().into())?;
                    let r = self.builder.build_call(f, &[l.into()], "tcp_accept")?;
                    if let Some(v) = r.try_as_basic_value().left() { self.values.insert(*result, v); }
                }

                // Tensor and Load/Store operations — delegate to runtime
                IrInstr::TensorOp { result, inputs, .. } | IrInstr::Load { result, .. } | IrInstr::Store { .. } => {
                    // Tensor ops and memory load/store delegate to the runtime;
                    // provide a ptr-returning stub for now.
                    match instr {
                        IrInstr::Store { .. } => { /* side-effecting, no result */ }
                        _ => {
                            let null = self.ptr_ty().const_null();
                            if let IrInstr::TensorOp { result, .. } | IrInstr::Load { result, .. } = instr {
                                self.values.insert(*result, null.into());
                            }
                        }
                    }
                }

                // Catch-all: any remaining ops that aren't yet implemented
                // produce an error with details for debugging.
                _ => {
                    return Err(CodegenError::Unsupported {
                        backend: "llvm_native".into(),
                        detail: format!(
                            "instruction not yet implemented in native backend: {:?}",
                            std::mem::discriminant(instr)
                        ),
                    });
                }
            }
            Ok(())
        }

        fn build_cast(
            &self,
            val: BasicValueEnum<'ctx>,
            from_ty: &IrType,
            to_ty: &IrType,
            name: &str,
        ) -> Result<BasicValueEnum<'ctx>, CodegenError> {
            use crate::ir::types::{IrType, DType};
            match (from_ty, to_ty) {
                // Int to int (with potential trunc/ext)
                (IrType::Scalar(from_dt), IrType::Scalar(to_dt)) if from_dt.is_integer() && to_dt.is_integer() => {
                    let int_val = val.into_int_value();
                    let to_width = match to_dt {
                        DType::I8 | DType::U8 => 8,
                        DType::I32 | DType::U32 => 32,
                        DType::I64 | DType::U64 | DType::USize => 64,
                        _ => return Ok(val),
                    };
                    let current_width = match from_dt {
                        DType::I8 | DType::U8 => 8,
                        DType::I32 | DType::U32 => 32,
                        DType::I64 | DType::U64 | DType::USize => 64,
                        _ => return Ok(val),
                    };
                    let result = if current_width == to_width {
                        int_val.as_basic_value_enum()
                    } else if current_width > to_width {
                        self.builder.build_int_truncate(int_val, self.context.custom_width_int_type(to_width), name)?.as_basic_value_enum()
                    } else {
                        self.builder.build_int_zext(int_val, self.context.custom_width_int_type(to_width), name)?.as_basic_value_enum()
                    };
                    Ok(result)
                }
                // Float to float
                (IrType::Scalar(from_dt), IrType::Scalar(to_dt)) if from_dt.is_float() && to_dt.is_float() => {
                    let float_val = val.into_float_value();
                    let to_type = match to_dt {
                        DType::F32 => self.context.f32_type(),
                        DType::F64 => self.context.f64_type(),
                        _ => return Ok(val),
                    };
                    let current_type = match from_dt {
                        DType::F32 => self.context.f32_type(),
                        DType::F64 => self.context.f64_type(),
                        _ => return Ok(val),
                    };
                    let result = if current_type == to_type {
                        float_val.as_basic_value_enum()
                    } else if matches!(from_dt, DType::F64) && matches!(to_dt, DType::F32) {
                        self.builder.build_float_trunc(float_val, to_type, name)?.as_basic_value_enum()
                    } else {
                        self.builder.build_float_ext(float_val, to_type, name)?.as_basic_value_enum()
                    };
                    Ok(result)
                }
                // Int to float
                (IrType::Scalar(from_dt), IrType::Scalar(to_dt)) if from_dt.is_integer() && to_dt.is_float() => {
                    let int_val = val.into_int_value();
                    let to_type = match to_dt {
                        DType::F32 => self.context.f32_type(),
                        DType::F64 => self.context.f64_type(),
                        _ => return Ok(val),
                    };
                    let result = if from_dt.is_signed() {
                        self.builder.build_signed_int_to_float(int_val, to_type, name)?.as_basic_value_enum()
                    } else {
                        self.builder.build_unsigned_int_to_float(int_val, to_type, name)?.as_basic_value_enum()
                    };
                    Ok(result)
                }
                // Float to int
                (IrType::Scalar(from_dt), IrType::Scalar(to_dt)) if from_dt.is_float() && to_dt.is_integer() => {
                    let float_val = val.into_float_value();
                    let to_type = match to_dt {
                        DType::I8 | DType::U8 => self.context.i8_type(),
                        DType::I32 | DType::U32 => self.context.i32_type(),
                        DType::I64 | DType::U64 | DType::USize => self.context.i64_type(),
                        _ => return Ok(val),
                    };
                    let result = if to_dt.is_signed() {
                        self.builder.build_float_to_signed_int(float_val, to_type, name)?.as_basic_value_enum()
                    } else {
                        self.builder.build_float_to_unsigned_int(float_val, to_type, name)?.as_basic_value_enum()
                    };
                    Ok(result)
                }
                _ => Ok(val),
            }
        }

        fn get_or_declare_malloc(&self) -> Result<FunctionValue<'ctx>, CodegenError> {
            if let Some(fn_val) = self.module.get_function("malloc") {
                return Ok(fn_val);
            }
            let i64_type = self.context.i64_type();
            let ptr_type = self.context.i8_type().ptr_type(AddressSpace::from(0));
            let fn_type = ptr_type.fn_type(&[i64_type.into()], false);
            let malloc_fn = self.module.add_function("malloc", fn_type, None);
            Ok(malloc_fn)
        }

        fn calculate_struct_size(&self, ty: &IrType) -> Result<u64, CodegenError> {
            match ty {
                IrType::Struct { fields, .. } => {
                    let mut size = 0u64;
                    for (_, field_ty) in fields {
                        size += self.type_size(field_ty)?;
                    }
                    // Align to 8 bytes
                    size = (size + 7) & !7;
                    Ok(size)
                }
                IrType::Tuple(elems) => {
                    let mut size = 0u64;
                    for elem_ty in elems {
                        size += self.type_size(elem_ty)?;
                    }
                    // Align to 8 bytes
                    size = (size + 7) & !7;
                    Ok(size)
                }
                _ => Ok(8), // Default pointer size
            }
        }

        fn type_size(&self, ty: &IrType) -> Result<u64, CodegenError> {
            match ty {
                IrType::Scalar(dtype) => Ok(match dtype {
                    DType::Bool | DType::I8 | DType::U8 => 1,
                    DType::I32 | DType::U32 | DType::F32 => 4,
                    DType::I64 | DType::U64 | DType::USize | DType::F64 => 8,
                }),
                IrType::Str => Ok(8), // pointer
                IrType::Struct { .. } => self.calculate_struct_size(ty),
                IrType::Tuple(elems) => {
                    let mut size = 0;
                    for e in elems { size += self.type_size(e)?; }
                    Ok((size + 7) & !7)
                }
                _ => Ok(8), // pointer size for other compound types
            }
        }

        fn get_value(&self, id: ValueId) -> Result<BasicValueEnum<'ctx>, CodegenError> {
            self.values.get(&id).copied().ok_or_else(|| {
                CodegenError::Unsupported {
                    backend: "llvm_native".into(),
                    detail: format!("value v{} not found in SSA map", id.0),
                }
            })
        }

        fn ptr_ty(&self) -> PointerType<'ctx> {
            self.context.i8_type().ptr_type(AddressSpace::from(0))
        }

        fn get_or_declare_runtime_fn(
            &self,
            name: &str,
            param_types: &[BasicTypeEnum<'ctx>],
        ) -> Result<FunctionValue<'ctx>, CodegenError> {
            if let Some(fn_val) = self.module.get_function(name) {
                return Ok(fn_val);
            }
            let ret_ty = self.context.i8_type().ptr_type(AddressSpace::from(0));
            let fn_type = ret_ty.fn_type(param_types, false);
            let fn_val = self.module.add_function(name, fn_type, None);
            Ok(fn_val)
        }
    }

    // ------------------------------------------------------------------
    // Helper: IRIS IR type → LLVM BasicType
    // ------------------------------------------------------------------

    fn ir_type_to_basic_type<'ctx>(
        context: &'ctx Context,
        ir_type: &IrType,
    ) -> Result<Option<BasicTypeEnum<'ctx>>, CodegenError> {
        match ir_type {
            IrType::Unit | IrType::Void => Ok(None),
            IrType::Scalar(DType::Bool) => Ok(Some(context.bool_type().into())),
            IrType::Scalar(DType::I8) => Ok(Some(context.i8_type().into())),
            IrType::Scalar(DType::U8) => Ok(Some(context.i8_type().into())),
            IrType::Scalar(DType::I32) => Ok(Some(context.i32_type().into())),
            IrType::Scalar(DType::U32) => Ok(Some(context.i32_type().into())),
            IrType::Scalar(DType::I64) => Ok(Some(context.i64_type().into())),
            IrType::Scalar(DType::U64) => Ok(Some(context.i64_type().into())),
            IrType::Scalar(DType::USize) => Ok(Some(context.ptr_sized_int_type(&inkwell::targets::TargetData::new(&inkwell::targets::TargetMachine::get_default_triple(), "")).into())),
            IrType::Scalar(DType::F32) => Ok(Some(context.f32_type().into())),
            IrType::Scalar(DType::F64) => Ok(Some(context.f64_type().into())),
            IrType::Str => Ok(Some(
                context.i8_type().ptr_type(AddressSpace::from(0)).into(),
            )),
            // Compound types fall back to opaque ptr.
            _ => Ok(Some(
                context.i8_type().ptr_type(AddressSpace::from(0)).into(),
            )),
        }
    }

    fn block_label_name(name: Option<&str>, id: BlockId) -> String {
        match name {
            Some(n) if !n.is_empty() => format!("{}_{}", n, id.0),
            _ => format!("block_{}", id.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Public API (available even without the native-llvm feature)
// ---------------------------------------------------------------------------

/// Returns true if native LLVM (inkwell) support is compiled in.
pub const fn native_llvm_available() -> bool {
    cfg!(feature = "native-llvm")
}

#[cfg(feature = "native-llvm")]
pub use backend::{LlvmNativeCompiler, LlvmNativeConfig};

#[cfg(feature = "native-llvm")]
use std::path::Path;

#[cfg(feature = "native-llvm")]
use crate::error::CodegenError;
#[cfg(feature = "native-llvm")]
use crate::ir::module::IrModule;

/// Convenience function: compile an `IrModule` to a native object file
/// using the native LLVM backend.
///
/// Returns `Err` if the `native-llvm` feature is not enabled.
#[cfg(feature = "native-llvm")]
pub fn compile_to_object(
    ir_module: &IrModule,
    output_path: &Path,
    target: Option<&str>,
) -> Result<(), CodegenError> {
    use inkwell::context::Context;
    let context = Context::create();
    let config = LlvmNativeConfig {
        target_triple: target.map(|t| t.to_string()),
        ..Default::default()
    };
    let mut compiler = LlvmNativeCompiler::new(&context, &ir_module.name, config);
    compiler.compile_module(ir_module)?;
    compiler.verify()?;
    compiler.emit_object(output_path, target)?;
    Ok(())
}
