//! AST → IR lowering.
//!
//! The lowerer walks the AST and constructs an `IrModule` using
//! `IrFunctionBuilder`. Each function is lowered independently. Variable
//! bindings are tracked in a lexical scope map (name → ValueId).
//!
//! Type propagation: for scalar operations where operand types are fully known
//! at construction time, the concrete type is used immediately. This avoids
//! leaving `IrType::Infer` placeholders that would fail `ValidatePass`.

pub mod graph;
pub mod ir_from_graph;
pub use graph::lower_model;
pub use ir_from_graph::lower_graph_to_ir;

use std::collections::HashMap;

use crate::error::LowerError;
use crate::ir::block::BlockId;
use crate::ir::function::Param;
use crate::ir::instr::{BinOp, IrInstr, ScalarUnaryOp, TensorOp};
use crate::ir::module::{IrFunctionBuilder, IrModule};
use crate::ir::types::{DType, Dim, IrType, Shape};
use crate::ir::value::ValueId;
use crate::parser::ast::{
    AstBinOp, AstBlock, AstDim, AstExpr, AstFunction, AstModule, AstScalarKind, AstStmt, AstType,
    AstUnaryOp, AstWhenArm, Ident,
};
use crate::parser::lexer::Span;

/// Lower an `AstModule` to an `IrModule`.
pub fn lower(ast: &AstModule, module_name: &str) -> Result<IrModule, LowerError> {
    let mut module = IrModule::new(module_name);

    // 1. Register enum definitions so functions can reference them.
    for e in &ast.enums {
        let variants: Vec<String> = e.variants.iter().map(|v| v.name.clone()).collect();
        module
            .add_enum_def(e.name.name.clone(), variants)
            .map_err(|_| LowerError::DuplicateFunction {
                name: e.name.name.clone(),
                span: e.name.span,
            })?;
    }

    // 2. Register struct definitions so functions can reference them.
    for s in &ast.structs {
        let fields: Vec<(String, IrType)> = s
            .fields
            .iter()
            .map(|f| (f.name.name.clone(), lower_type_with_structs(&f.ty, &module)))
            .collect();
        module
            .add_struct_def(s.name.name.clone(), fields)
            .map_err(|_| LowerError::DuplicateFunction {
                name: s.name.name.clone(),
                span: s.name.span,
            })?;
    }

    // 3. Pre-collect function return types so call sites get concrete types.
    let mut fn_sigs: HashMap<String, IrType> = HashMap::new();
    for func in &ast.functions {
        let ret_ty = lower_type_with_structs(&func.return_ty, &module);
        fn_sigs.insert(func.name.name.clone(), ret_ty);
    }

    // 4. Lower all function definitions.
    let mut all_lifted: Vec<crate::ir::function::IrFunction> = Vec::new();
    for func in &ast.functions {
        let (ir_func, lifted) = lower_function(func, &module, &fn_sigs)?;
        module
            .add_function(ir_func)
            .map_err(|_| LowerError::DuplicateFunction {
                name: func.name.name.clone(),
                span: func.name.span,
            })?;
        all_lifted.extend(lifted);
    }
    // Add all lambda-lifted functions.
    for lf in all_lifted {
        // Skip if already added (duplicate lambda name guard).
        if module.function_by_name(&lf.name).is_none() {
            let _ = module.add_function(lf);
        }
    }
    Ok(module)
}

struct Lowerer<'m> {
    builder: IrFunctionBuilder,
    /// Current lexical scope: name → (ValueId, IrType).
    scope: HashMap<String, (ValueId, IrType)>,
    /// Stack of (header_block, merge_block, loop_var_names) for nested loops.
    loop_stack: Vec<(BlockId, BlockId, Vec<String>)>,
    /// Reference to the module for struct/enum type lookups.
    module: &'m IrModule,
    /// Pre-collected function return types for resolving call result types.
    fn_sigs: &'m HashMap<String, IrType>,
    /// Counter for unique lambda function names.
    lambda_counter: std::rc::Rc<std::cell::Cell<u32>>,
    /// Lambda functions to be added to the module after this function is lowered.
    lifted_fns: std::rc::Rc<std::cell::RefCell<Vec<crate::ir::function::IrFunction>>>,
}

impl<'m> Lowerer<'m> {
    fn new(
        builder: IrFunctionBuilder,
        module: &'m IrModule,
        fn_sigs: &'m HashMap<String, IrType>,
    ) -> Self {
        Self::new_with_lambda_state(builder, module, fn_sigs,
            std::rc::Rc::new(std::cell::Cell::new(0)),
            std::rc::Rc::new(std::cell::RefCell::new(Vec::new())),
        )
    }

    fn new_with_lambda_state(
        builder: IrFunctionBuilder,
        module: &'m IrModule,
        fn_sigs: &'m HashMap<String, IrType>,
        lambda_counter: std::rc::Rc<std::cell::Cell<u32>>,
        lifted_fns: std::rc::Rc<std::cell::RefCell<Vec<crate::ir::function::IrFunction>>>,
    ) -> Self {
        Self {
            builder,
            scope: HashMap::new(),
            loop_stack: Vec::new(),
            module,
            fn_sigs,
            lambda_counter,
            lifted_fns,
        }
    }

    /// Looks up a variable and returns its `ValueId` and type.
    fn lookup(&self, ident: &Ident) -> Result<(ValueId, IrType), LowerError> {
        self.scope
            .get(&ident.name)
            .cloned()
            .ok_or_else(|| LowerError::UndefinedVariable {
                name: ident.name.clone(),
                span: ident.span,
            })
    }

    fn lower_expr(&mut self, expr: &AstExpr) -> Result<(ValueId, IrType), LowerError> {
        match expr {
            AstExpr::Ident(ident) => self.lookup(ident),

            AstExpr::FloatLit { value, .. } => {
                let result = self.builder.fresh_value();
                let ty = IrType::Scalar(DType::F32);
                self.builder.push_instr(
                    IrInstr::ConstFloat {
                        result,
                        value: *value,
                        ty: ty.clone(),
                    },
                    Some(ty.clone()),
                );
                Ok((result, ty))
            }

            AstExpr::IntLit { value, .. } => {
                let result = self.builder.fresh_value();
                let ty = IrType::Scalar(DType::I64);
                self.builder.push_instr(
                    IrInstr::ConstInt {
                        result,
                        value: *value,
                        ty: ty.clone(),
                    },
                    Some(ty.clone()),
                );
                Ok((result, ty))
            }

            AstExpr::BoolLit { value, .. } => {
                let result = self.builder.fresh_value();
                let ty = IrType::Scalar(DType::Bool);
                self.builder.push_instr(
                    IrInstr::ConstBool {
                        result,
                        value: *value,
                    },
                    Some(ty.clone()),
                );
                Ok((result, ty))
            }

            // String literals are emitted as ConstStr instructions.
            AstExpr::StringLit { value, .. } => {
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstStr {
                        result,
                        value: value.clone(),
                    },
                    Some(IrType::Str),
                );
                Ok((result, IrType::Str))
            }

            AstExpr::BinOp { op, lhs, rhs, span } => {
                // Short-circuit logical operators get their own control flow.
                if matches!(op, AstBinOp::And | AstBinOp::Or) {
                    return self.lower_short_circuit(*op, lhs, rhs, *span);
                }

                let (lhs_val, lhs_ty) = self.lower_expr(lhs)?;
                let (rhs_val, rhs_ty) = self.lower_expr(rhs)?;

                // Require operand types to match for scalar binops.
                if lhs_ty != rhs_ty {
                    return Err(LowerError::TypeMismatch {
                        expected: format!("{}", lhs_ty),
                        found: format!("{}", rhs_ty),
                        span: *span,
                    });
                }

                let ir_op = lower_binop(*op);
                let result_ty = match op {
                    // Comparison ops yield bool regardless of operand type.
                    AstBinOp::CmpEq
                    | AstBinOp::CmpNe
                    | AstBinOp::CmpLt
                    | AstBinOp::CmpLe
                    | AstBinOp::CmpGt
                    | AstBinOp::CmpGe => IrType::Scalar(DType::Bool),
                    _ => lhs_ty.clone(),
                };

                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::BinOp {
                        result,
                        op: ir_op,
                        lhs: lhs_val,
                        rhs: rhs_val,
                        ty: result_ty.clone(),
                    },
                    Some(result_ty.clone()),
                );
                Ok((result, result_ty))
            }

            AstExpr::UnaryOp { op, expr, .. } => {
                let (val, ty) = self.lower_expr(expr)?;
                let result = self.builder.fresh_value();
                let ir_op = match op {
                    AstUnaryOp::Neg => ScalarUnaryOp::Neg,
                    AstUnaryOp::Not => ScalarUnaryOp::Not,
                };
                self.builder.push_instr(
                    IrInstr::UnaryOp {
                        result,
                        op: ir_op,
                        operand: val,
                        ty: ty.clone(),
                    },
                    Some(ty.clone()),
                );
                Ok((result, ty))
            }

            AstExpr::Call { callee, args, span } => self.lower_call(callee, args, *span),

            AstExpr::If {
                cond,
                then_block,
                else_block,
                span,
            } => self.lower_if_expr(cond, then_block, else_block.as_ref(), *span),

            AstExpr::Block(block) => {
                let result = self.lower_block(block)?;
                result.ok_or_else(|| LowerError::Unsupported {
                    detail: "block expression with no tail value".into(),
                    span: block.span,
                })
            }

            AstExpr::Index {
                base,
                indices,
                span,
            } => {
                let (base_val, base_ty) = self.lower_expr(base)?;
                // Array index: arr[i]
                if let IrType::Array { elem, .. } = &base_ty {
                    let elem_ty = (**elem).clone();
                    if indices.len() != 1 {
                        return Err(LowerError::Unsupported {
                            detail: "array index requires exactly 1 index".into(),
                            span: *span,
                        });
                    }
                    let (idx_val, _) = self.lower_expr(&indices[0])?;
                    let result = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::ArrayLoad {
                            result,
                            array: base_val,
                            index: idx_val,
                            elem_ty: elem_ty.clone(),
                        },
                        Some(elem_ty.clone()),
                    );
                    return Ok((result, elem_ty));
                }
                // Tensor index: tensor[i, j, ...]
                let mut idx_vals = Vec::new();
                for idx in indices {
                    let (iv, _) = self.lower_expr(idx)?;
                    idx_vals.push(iv);
                }
                // Extract element type from tensor type.
                let elem_ty = match &base_ty {
                    IrType::Tensor { dtype, .. } => IrType::Scalar(*dtype),
                    other => other.clone(), // fallback
                };
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::Load {
                        result,
                        tensor: base_val,
                        indices: idx_vals,
                        result_ty: elem_ty.clone(),
                    },
                    Some(elem_ty.clone()),
                );
                Ok((result, elem_ty))
            }

            AstExpr::StructLit { name, fields, span } => {
                // Look up the struct definition.
                let struct_fields = self
                    .module
                    .struct_def(name)
                    .ok_or_else(|| LowerError::UndefinedVariable {
                        name: name.clone(),
                        span: *span,
                    })?
                    .clone();

                // Lower each field expression in declaration order.
                let mut field_vals = Vec::with_capacity(struct_fields.len());
                for (field_name, _field_ty) in &struct_fields {
                    let provided =
                        fields
                            .iter()
                            .find(|(n, _)| n == field_name)
                            .ok_or_else(|| LowerError::Unsupported {
                                detail: format!("missing field '{}' in struct literal", field_name),
                                span: *span,
                            })?;
                    let (val, _) = self.lower_expr(&provided.1)?;
                    field_vals.push(val);
                }

                let result_ty = IrType::Struct {
                    name: name.clone(),
                    fields: struct_fields,
                };
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::MakeStruct {
                        result,
                        fields: field_vals,
                        result_ty: result_ty.clone(),
                    },
                    Some(result_ty.clone()),
                );
                Ok((result, result_ty))
            }

            AstExpr::FieldAccess { base, field, span } => {
                // Check if base is a bare identifier naming an enum → variant construction.
                if let AstExpr::Ident(base_ident) = base.as_ref() {
                    if let Some(variants) = self.module.enum_def(&base_ident.name) {
                        let variants = variants.clone();
                        let variant_idx =
                            variants.iter().position(|v| v == field).ok_or_else(|| {
                                LowerError::Unsupported {
                                    detail: format!(
                                        "no variant '{}' in enum '{}'",
                                        field, base_ident.name
                                    ),
                                    span: *span,
                                }
                            })?;
                        let result_ty = IrType::Enum {
                            name: base_ident.name.clone(),
                            variants,
                        };
                        let result = self.builder.fresh_value();
                        self.builder.push_instr(
                            IrInstr::MakeVariant {
                                result,
                                variant_idx,
                                result_ty: result_ty.clone(),
                            },
                            Some(result_ty.clone()),
                        );
                        return Ok((result, result_ty));
                    }
                }
                // Normal struct field access.
                let (base_val, base_ty) = self.lower_expr(base)?;
                let struct_fields = match &base_ty {
                    IrType::Struct { fields, .. } => fields.clone(),
                    _ => {
                        return Err(LowerError::Unsupported {
                            detail: format!("field access on non-struct type {}", base_ty),
                            span: *span,
                        });
                    }
                };
                let field_index = struct_fields
                    .iter()
                    .position(|(n, _)| n == field)
                    .ok_or_else(|| LowerError::Unsupported {
                        detail: format!("no field '{}' in struct", field),
                        span: *span,
                    })?;
                let result_ty = struct_fields[field_index].1.clone();
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::GetField {
                        result,
                        base: base_val,
                        field_index,
                        result_ty: result_ty.clone(),
                    },
                    Some(result_ty.clone()),
                );
                Ok((result, result_ty))
            }

            AstExpr::When {
                scrutinee,
                arms,
                span,
            } => self.lower_when_expr(scrutinee, arms, *span),

            AstExpr::Tuple { elements, span } => {
                let mut elem_vals = Vec::with_capacity(elements.len());
                let mut elem_tys = Vec::with_capacity(elements.len());
                for e in elements {
                    let (v, t) = self.lower_expr(e)?;
                    elem_vals.push(v);
                    elem_tys.push(t);
                }
                let result_ty = IrType::Tuple(elem_tys);
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::MakeTuple {
                        result,
                        elements: elem_vals,
                        result_ty: result_ty.clone(),
                    },
                    Some(result_ty.clone()),
                );
                let _ = span;
                Ok((result, result_ty))
            }

            AstExpr::TupleIndex { base, index, span } => {
                let (base_val, base_ty) = self.lower_expr(base)?;
                let elem_types = match &base_ty {
                    IrType::Tuple(elems) => elems.clone(),
                    _ => {
                        return Err(LowerError::Unsupported {
                            detail: format!("tuple index on non-tuple type {}", base_ty),
                            span: *span,
                        });
                    }
                };
                if *index >= elem_types.len() {
                    return Err(LowerError::Unsupported {
                        detail: format!(
                            "tuple index {} out of bounds for {} elements",
                            index,
                            elem_types.len()
                        ),
                        span: *span,
                    });
                }
                let result_ty = elem_types[*index].clone();
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::GetElement {
                        result,
                        base: base_val,
                        index: *index,
                        result_ty: result_ty.clone(),
                    },
                    Some(result_ty.clone()),
                );
                Ok((result, result_ty))
            }

            AstExpr::Lambda { params, body, span } => {
                self.lower_lambda(params, body, *span)
            }

            AstExpr::ArrayLit { elems, span } => {
                if elems.is_empty() {
                    return Err(LowerError::Unsupported {
                        detail: "empty array literal not supported".into(),
                        span: *span,
                    });
                }
                let mut elem_vals = Vec::with_capacity(elems.len());
                let mut elem_ty = IrType::Infer;
                for e in elems {
                    let (v, ty) = self.lower_expr(e)?;
                    elem_vals.push(v);
                    elem_ty = ty;
                }
                let size = elem_vals.len();
                let result_ty = IrType::Array {
                    elem: Box::new(elem_ty.clone()),
                    len: size,
                };
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::AllocArray {
                        result,
                        elem_ty: elem_ty.clone(),
                        size,
                        init: elem_vals,
                    },
                    Some(result_ty.clone()),
                );
                let _ = span;
                Ok((result, result_ty))
            }

            AstExpr::Cast { expr, ty, span } => {
                let (operand_val, from_ty) = self.lower_expr(expr)?;
                let to_ty = lower_type(ty);
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::Cast {
                        result,
                        operand: operand_val,
                        from_ty: from_ty.clone(),
                        to_ty: to_ty.clone(),
                    },
                    Some(to_ty.clone()),
                );
                let _ = span;
                Ok((result, to_ty))
            }
        }
    }

    /// Lowers a lambda expression using lambda-lifting.
    ///
    /// Finds free variables (scope entries not covered by lambda params),
    /// generates a unique name `__lambda_N`, builds an `IrFunction` with
    /// `(captures..., params...)` parameter list, then emits `MakeClosure`.
    fn lower_lambda(
        &mut self,
        params: &[crate::parser::ast::AstParam],
        body: &AstExpr,
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        let counter = self.lambda_counter.get();
        self.lambda_counter.set(counter + 1);
        let fn_name = format!("__lambda_{}", counter);

        // Collect parameter names to exclude from free-variable search.
        let param_names: std::collections::HashSet<String> =
            params.iter().map(|p| p.name.name.clone()).collect();

        // Free variables: everything in scope that isn't a lambda param.
        let captures: Vec<(String, ValueId, IrType)> = self
            .scope
            .iter()
            .filter(|(name, _)| !param_names.contains(*name))
            .map(|(name, (vid, ty))| (name.clone(), *vid, ty.clone()))
            .collect();

        // Build the lifted function: params = captures + lambda_params.
        let mut lifted_params: Vec<Param> = captures
            .iter()
            .map(|(name, _, ty)| Param { name: name.clone(), ty: ty.clone() })
            .collect();
        for p in params {
            lifted_params.push(Param {
                name: p.name.name.clone(),
                ty: lower_type_with_structs(&p.ty, self.module),
            });
        }

        // Infer return type by building a temporary lowerer for the lambda body.
        // We need to lower the body to know the return type.
        // Use IrType::Infer as a placeholder if we can't determine it statically.
        // For now we lower into a temporary builder.
        let temp_ret_ty = IrType::Infer; // will be fixed up after lowering
        let temp_builder = IrFunctionBuilder::new(&fn_name, lifted_params.clone(), temp_ret_ty);
        let mut lambda_lowerer = Lowerer::new_with_lambda_state(
            temp_builder,
            self.module,
            self.fn_sigs,
            self.lambda_counter.clone(),
            self.lifted_fns.clone(),
        );

        let entry = lambda_lowerer.builder.create_block(Some("entry"));
        lambda_lowerer.builder.set_current_block(entry);

        // Populate the lambda scope with captured + param values.
        for (name, _, ty) in &captures {
            let val = lambda_lowerer
                .builder
                .add_block_param(entry, Some(name), ty.clone());
            lambda_lowerer.scope.insert(name.clone(), (val, ty.clone()));
        }
        for p in params {
            let ty = lower_type_with_structs(&p.ty, self.module);
            let val = lambda_lowerer
                .builder
                .add_block_param(entry, Some(&p.name.name), ty.clone());
            lambda_lowerer.scope.insert(p.name.name.clone(), (val, ty));
        }

        let (ret_val, ret_ty) = lambda_lowerer.lower_expr(body)?;
        lambda_lowerer
            .builder
            .push_instr(IrInstr::Return { values: vec![ret_val] }, None);
        lambda_lowerer.builder.seal_unterminated_blocks();

        // Patch the return type.
        let mut ir_func = lambda_lowerer.builder.build();
        ir_func.return_ty = ret_ty.clone();

        // Register the lifted function.
        self.lifted_fns.borrow_mut().push(ir_func);

        // Also register in fn_sigs-equivalent for the current lowering context
        // (no direct mutation possible; closures are called via CallClosure).

        // Emit MakeClosure in the current context.
        let capture_vals: Vec<ValueId> = captures.iter().map(|(_, v, _)| *v).collect();
        let closure_ty = IrType::Fn {
            params: lifted_params.iter().map(|p| p.ty.clone()).collect(),
            ret: Box::new(ret_ty),
        };
        let result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::MakeClosure {
                result,
                fn_name: fn_name.clone(),
                captures: capture_vals,
                result_ty: closure_ty.clone(),
            },
            Some(closure_ty.clone()),
        );
        let _ = span;
        Ok((result, closure_ty))
    }

    /// Lowers a function call. Handles the built-in `einsum` intrinsic specially.
    fn lower_call(
        &mut self,
        callee: &Ident,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        // Built-in intrinsic: einsum("notation", inputs...)
        if callee.name == "einsum" {
            return self.lower_einsum(args, span);
        }

        // Check if the callee is a closure variable in scope.
        if let Some((closure_val, IrType::Fn { ret, .. })) = self.scope.get(&callee.name).cloned() {
            let ret_ty = *ret;
            let mut arg_vals = Vec::with_capacity(args.len());
            for arg in args {
                let (v, _) = self.lower_expr(arg)?;
                arg_vals.push(v);
            }
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::CallClosure {
                    result: Some(result),
                    closure: closure_val,
                    args: arg_vals,
                    result_ty: ret_ty.clone(),
                },
                Some(ret_ty.clone()),
            );
            return Ok((result, ret_ty));
        }

        // Built-in: len(s) → StrLen
        if callee.name == "len" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "len() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (operand, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::StrLen { result, operand },
                Some(ty.clone()),
            );
            return Ok((result, ty));
        }

        // Built-in: concat(s, t) → StrConcat
        if callee.name == "concat" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "concat() requires exactly 2 arguments".into(),
                    span,
                });
            }
            let (lhs, _) = self.lower_expr(&args[0])?;
            let (rhs, _) = self.lower_expr(&args[1])?;
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::StrConcat { result, lhs, rhs },
                Some(IrType::Str),
            );
            return Ok((result, IrType::Str));
        }

        // Built-in: print(v) → Print (returns unit, we return a dummy i64 zero for now)
        if callee.name == "print" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "print() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (operand, _) = self.lower_expr(&args[0])?;
            self.builder.push_instr(
                IrInstr::Print { operand },
                None,
            );
            // Return a dummy i64 zero as the "unit" value.
            let dummy = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::ConstInt { result: dummy, value: 0, ty: ty.clone() },
                Some(ty.clone()),
            );
            return Ok((dummy, ty));
        }

        // General function call — look up the callee's return type from
        // pre-collected signatures so the result has a concrete type.
        let ret_ty = self
            .fn_sigs
            .get(&callee.name)
            .cloned()
            .unwrap_or(IrType::Infer);

        let mut arg_vals = Vec::with_capacity(args.len());
        for arg in args {
            let (v, _) = self.lower_expr(arg)?;
            arg_vals.push(v);
        }
        let result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::Call {
                result: Some(result),
                callee: callee.name.clone(),
                args: arg_vals,
                result_ty: Some(ret_ty.clone()),
            },
            Some(ret_ty.clone()),
        );
        Ok((result, ret_ty))
    }

    fn lower_einsum(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        if args.is_empty() {
            return Err(LowerError::Unsupported {
                detail: "einsum requires at least one argument (the notation string)".into(),
                span,
            });
        }

        // First arg must be a string literal (the einsum notation).
        let notation = match &args[0] {
            AstExpr::StringLit { value, .. } => value.clone(),
            other => {
                return Err(LowerError::Unsupported {
                    detail: "first argument to einsum must be a string literal".into(),
                    span: other.span(),
                });
            }
        };

        // Remaining args are tensor inputs.
        let mut input_vals = Vec::new();
        let mut input_tys = Vec::new();
        for arg in &args[1..] {
            let (v, ty) = self.lower_expr(arg)?;
            input_vals.push(v);
            input_tys.push(ty);
        }

        // Derive result type from the einsum notation and input shapes.
        // For bootstrap: use Infer if we can't resolve, or derive from notation.
        let result_ty = derive_einsum_result_type(&notation, &input_tys);

        let result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::TensorOp {
                result,
                op: TensorOp::Einsum {
                    notation: notation.clone(),
                },
                inputs: input_vals,
                result_ty: result_ty.clone(),
            },
            Some(result_ty.clone()),
        );
        Ok((result, result_ty))
    }

    /// Lowers `if cond { then_blk } else { else_blk }` to SSA control flow.
    ///
    /// Creates three new blocks (then / else / merge) and emits a `CondBr`
    /// in the current block. Each branch is lowered independently with a
    /// saved/restored scope to prevent variable leakage across branches.
    /// The merge block receives the result via a block parameter.
    ///
    /// If a branch terminates early (e.g. via `return`), no `Br` to merge is
    /// emitted for that branch. If both branches terminate early, the merge
    /// block is unreachable but still created for well-formedness.
    fn lower_if_expr(
        &mut self,
        cond: &AstExpr,
        then_blk: &AstBlock,
        else_blk: Option<&AstBlock>,
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        let else_blk = else_blk.ok_or_else(|| LowerError::Unsupported {
            detail: "if-without-else is not supported as a value expression".into(),
            span,
        })?;

        // 1. Evaluate condition in the current block.
        let (cond_val, _) = self.lower_expr(cond)?;

        // 2. Allocate the three new blocks.
        let then_bb = self.builder.create_block(Some("then"));
        let else_bb = self.builder.create_block(Some("else"));
        let merge_bb = self.builder.create_block(Some("merge"));

        // 3. Terminate the current block with a conditional branch.
        self.builder.push_instr(
            IrInstr::CondBr {
                cond: cond_val,
                then_block: then_bb,
                then_args: vec![],
                else_block: else_bb,
                else_args: vec![],
            },
            None,
        );

        // 4. Lower the THEN branch.
        //    Save the outer scope so inner let-bindings don't leak out.
        let outer_scope = self.scope.clone();
        self.builder.set_current_block(then_bb);
        let then_result = self.lower_block(then_blk)?;
        // Only emit Br to merge if the branch produced a value (didn't return early).
        if let Some((then_val, _)) = &then_result {
            self.builder.push_instr(
                IrInstr::Br {
                    target: merge_bb,
                    args: vec![*then_val],
                },
                None,
            );
        }
        self.scope = outer_scope.clone();

        // 5. Lower the ELSE branch.
        self.builder.set_current_block(else_bb);
        let else_result = self.lower_block(else_blk)?;
        if let Some((else_val, _)) = &else_result {
            self.builder.push_instr(
                IrInstr::Br {
                    target: merge_bb,
                    args: vec![*else_val],
                },
                None,
            );
        }
        self.scope = outer_scope;

        // 6. Determine the result type for the merge block parameter.
        let result_ty = match (&then_result, &else_result) {
            (Some((_, ty)), _) => ty.clone(),
            (_, Some((_, ty))) => ty.clone(),
            // Both branches terminated early — merge is unreachable.
            (None, None) => IrType::Scalar(DType::I64),
        };

        let result = self
            .builder
            .add_block_param(merge_bb, Some("if_result"), result_ty.clone());
        self.builder.set_current_block(merge_bb);

        Ok((result, result_ty))
    }

    /// Lowers short-circuit `&&` / `||` to SSA control flow.
    ///
    /// `a && b`:
    ///   eval a → cond
    ///   CondBr cond → rhs_bb, merge_bb(false)
    ///   rhs_bb: eval b → rhs_val, Br merge_bb(rhs_val)
    ///   merge_bb(result: bool): …
    ///
    /// `a || b`:
    ///   eval a → cond
    ///   CondBr cond → merge_bb(true), rhs_bb
    ///   rhs_bb: eval b → rhs_val, Br merge_bb(rhs_val)
    ///   merge_bb(result: bool): …
    fn lower_short_circuit(
        &mut self,
        op: AstBinOp,
        lhs: &AstExpr,
        rhs: &AstExpr,
        _span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        let bool_ty = IrType::Scalar(DType::Bool);

        // 1. Evaluate LHS.
        let (lhs_val, _) = self.lower_expr(lhs)?;

        // 2. Create blocks.
        let rhs_bb = self.builder.create_block(Some("sc_rhs"));
        let merge_bb = self.builder.create_block(Some("sc_merge"));

        // 3. Emit the short-circuit constant for the skipped case.
        let short_val = self.builder.fresh_value();
        let short_bool = matches!(op, AstBinOp::Or); // ||: true, &&: false
        self.builder.push_instr(
            IrInstr::ConstBool {
                result: short_val,
                value: short_bool,
            },
            Some(bool_ty.clone()),
        );

        // 4. Emit CondBr.
        match op {
            AstBinOp::And => {
                // If LHS is true, eval RHS; if false, short-circuit to merge with false.
                self.builder.push_instr(
                    IrInstr::CondBr {
                        cond: lhs_val,
                        then_block: rhs_bb,
                        then_args: vec![],
                        else_block: merge_bb,
                        else_args: vec![short_val],
                    },
                    None,
                );
            }
            AstBinOp::Or => {
                // If LHS is true, short-circuit to merge with true; else eval RHS.
                self.builder.push_instr(
                    IrInstr::CondBr {
                        cond: lhs_val,
                        then_block: merge_bb,
                        then_args: vec![short_val],
                        else_block: rhs_bb,
                        else_args: vec![],
                    },
                    None,
                );
            }
            _ => unreachable!(),
        }

        // 5. RHS block: evaluate rhs, branch to merge.
        self.builder.set_current_block(rhs_bb);
        let (rhs_val, _) = self.lower_expr(rhs)?;
        self.builder.push_instr(
            IrInstr::Br {
                target: merge_bb,
                args: vec![rhs_val],
            },
            None,
        );

        // 6. Merge block with block parameter carrying the result.
        let result = self
            .builder
            .add_block_param(merge_bb, Some("sc_result"), bool_ty.clone());
        self.builder.set_current_block(merge_bb);

        Ok((result, bool_ty))
    }

    /// Lowers `when scrutinee { EnumName.Variant => expr, ... }` to SSA.
    ///
    /// Emits a `SwitchVariant` terminator that dispatches to one block per arm,
    /// each of which produces a value and jumps to a merge block.
    fn lower_when_expr(
        &mut self,
        scrutinee: &AstExpr,
        arms: &[AstWhenArm],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        if arms.is_empty() {
            return Err(LowerError::Unsupported {
                detail: "when expression must have at least one arm".into(),
                span,
            });
        }

        // 1. Evaluate the scrutinee.
        let (scrut_val, scrut_ty) = self.lower_expr(scrutinee)?;

        // 2. Verify it is an enum type and extract variants.
        let (enum_name, variants) = match &scrut_ty {
            IrType::Enum { name, variants } => (name.clone(), variants.clone()),
            _ => {
                return Err(LowerError::Unsupported {
                    detail: format!("when scrutinee must be an enum type, got {}", scrut_ty),
                    span,
                });
            }
        };

        // 3. Allocate one block per arm and a merge block.
        let mut arm_blocks: Vec<BlockId> = Vec::new();
        for arm in arms {
            arm_blocks.push(
                self.builder
                    .create_block(Some(&format!("when_{}_{}", enum_name, arm.variant_name))),
            );
        }
        let merge_bb = self.builder.create_block(Some("when_merge"));

        // 4. Build the arms list for SwitchVariant.
        let mut switch_arms: Vec<(usize, BlockId)> = Vec::new();
        for (arm_idx, arm) in arms.iter().enumerate() {
            let variant_idx = variants
                .iter()
                .position(|v| v == &arm.variant_name)
                .ok_or_else(|| LowerError::Unsupported {
                    detail: format!("no variant '{}' in enum '{}'", arm.variant_name, enum_name),
                    span: arm.span,
                })?;
            switch_arms.push((variant_idx, arm_blocks[arm_idx]));
        }

        // 5. Emit SwitchVariant terminator in the current block.
        self.builder.push_instr(
            IrInstr::SwitchVariant {
                scrutinee: scrut_val,
                arms: switch_arms,
                default_block: None,
            },
            None,
        );

        // 6. Lower each arm body.
        let outer_scope = self.scope.clone();
        let mut result_ty: Option<IrType> = None;
        for (arm, &arm_bb) in arms.iter().zip(arm_blocks.iter()) {
            self.scope = outer_scope.clone();
            self.builder.set_current_block(arm_bb);
            let (arm_val, arm_ty) = self.lower_expr(&arm.body)?;
            if result_ty.is_none() {
                result_ty = Some(arm_ty);
            }
            self.builder.push_instr(
                IrInstr::Br {
                    target: merge_bb,
                    args: vec![arm_val],
                },
                None,
            );
        }
        self.scope = outer_scope;

        let result_ty = result_ty.unwrap();

        // 7. Merge block receives the result.
        let result = self
            .builder
            .add_block_param(merge_bb, Some("when_result"), result_ty.clone());
        self.builder.set_current_block(merge_bb);

        Ok((result, result_ty))
    }

    /// Lowers a `while cond { body }` loop using SSA block parameters.
    fn lower_while(
        &mut self,
        cond: &AstExpr,
        body: &AstBlock,
        span: Span,
    ) -> Result<(), LowerError> {
        // Pre-scan body to find which variables get rebound.
        let rebound = find_rebound_vars(body);

        // Collect the loop variables that exist in the current scope.
        let mut loop_vars: Vec<(String, ValueId, IrType)> = Vec::new();
        for name in &rebound {
            if let Some((val, ty)) = self.scope.get(name).cloned() {
                loop_vars.push((name.clone(), val, ty));
            }
        }

        let initial_vals: Vec<ValueId> = loop_vars.iter().map(|(_, v, _)| *v).collect();

        // Create the three blocks.
        let header_bb = self.builder.create_block(Some("while_header"));
        let body_bb = self.builder.create_block(Some("while_body"));
        let merge_bb = self.builder.create_block(Some("while_merge"));

        // Add block params to header (one per loop variable).
        let mut header_params: Vec<ValueId> = Vec::new();
        for (name, _, ty) in &loop_vars {
            let p = self
                .builder
                .add_block_param(header_bb, Some(name), ty.clone());
            header_params.push(p);
        }

        // Add block params to merge (receive exit values from header's else path).
        let mut merge_params: Vec<ValueId> = Vec::new();
        for (name, _, ty) in &loop_vars {
            let p = self
                .builder
                .add_block_param(merge_bb, Some(name), ty.clone());
            merge_params.push(p);
        }

        // From the current block, branch to header with initial values.
        self.builder.push_instr(
            IrInstr::Br {
                target: header_bb,
                args: initial_vals,
            },
            None,
        );

        // Lower condition in header block.
        self.builder.set_current_block(header_bb);
        for ((name, _, ty), &param_val) in loop_vars.iter().zip(header_params.iter()) {
            self.scope.insert(name.clone(), (param_val, ty.clone()));
        }

        let (cond_val, _) = self.lower_expr(cond)?;

        // Emit CondBr: true → body (no args), false → merge (current header params).
        self.builder.push_instr(
            IrInstr::CondBr {
                cond: cond_val,
                then_block: body_bb,
                then_args: vec![],
                else_block: merge_bb,
                else_args: header_params.clone(),
            },
            None,
        );

        // Lower body block.
        self.builder.set_current_block(body_bb);
        let loop_var_names: Vec<String> = loop_vars.iter().map(|(n, _, _)| n.clone()).collect();
        self.loop_stack
            .push((header_bb, merge_bb, loop_var_names.clone()));
        self.lower_block_stmts(body)?;
        self.loop_stack.pop();

        // Emit back-edge Br if the body wasn't terminated by break/continue.
        if !self.builder.is_current_block_terminated() {
            let updated_vals: Vec<ValueId> = loop_vars
                .iter()
                .map(|(name, original_val, _)| {
                    self.scope
                        .get(name)
                        .map(|(v, _)| *v)
                        .unwrap_or(*original_val)
                })
                .collect();
            self.builder.push_instr(
                IrInstr::Br {
                    target: header_bb,
                    args: updated_vals,
                },
                None,
            );
        }

        // Move to merge block and update scope with loop var final values.
        self.builder.set_current_block(merge_bb);
        for ((name, _, ty), &merge_val) in loop_vars.iter().zip(merge_params.iter()) {
            self.scope.insert(name.clone(), (merge_val, ty.clone()));
        }

        let _ = span;
        Ok(())
    }

    /// Lowers `for <var> in <start>..<end> { body }` to SSA block-param loop.
    ///
    /// The loop variable is incremented by 1 after each body execution.
    /// Semantics: `start` and `end` are evaluated once before the loop.
    fn lower_for_range(
        &mut self,
        var: &crate::parser::ast::Ident,
        start: &AstExpr,
        end: &AstExpr,
        body: &AstBlock,
        span: Span,
    ) -> Result<(), LowerError> {
        // 1. Evaluate start and end once in the current (pre-loop) block.
        let (start_val, loop_var_ty) = self.lower_expr(start)?;
        let (end_val, _) = self.lower_expr(end)?;

        // 2. Pre-scan body for rebounded variables; loop var is always rebound.
        let mut rebound = find_rebound_vars(body);
        if !rebound.contains(&var.name) {
            rebound.push(var.name.clone());
        }

        // 3. Collect loop variables: loop var first, then other rebound outer vars.
        let mut loop_vars: Vec<(String, ValueId, IrType)> = Vec::new();
        loop_vars.push((var.name.clone(), start_val, loop_var_ty.clone()));
        for name in &rebound {
            if name == &var.name {
                continue;
            }
            if let Some((val, ty)) = self.scope.get(name).cloned() {
                loop_vars.push((name.clone(), val, ty));
            }
        }

        let initial_vals: Vec<ValueId> = loop_vars.iter().map(|(_, v, _)| *v).collect();

        // 4. Create blocks.
        let header_bb = self.builder.create_block(Some("for_header"));
        let body_bb = self.builder.create_block(Some("for_body"));
        let merge_bb = self.builder.create_block(Some("for_merge"));

        // 5. Header block params (one per loop variable).
        let mut header_params: Vec<ValueId> = Vec::new();
        for (name, _, ty) in &loop_vars {
            let p = self
                .builder
                .add_block_param(header_bb, Some(name), ty.clone());
            header_params.push(p);
        }

        // 6. Merge block params (receive final values on loop exit).
        let mut merge_params: Vec<ValueId> = Vec::new();
        for (name, _, ty) in &loop_vars {
            let p = self
                .builder
                .add_block_param(merge_bb, Some(name), ty.clone());
            merge_params.push(p);
        }

        // 7. Branch from current block to header with initial values.
        self.builder.push_instr(
            IrInstr::Br {
                target: header_bb,
                args: initial_vals,
            },
            None,
        );

        // 8. Header: update scope with params, emit `loop_var < end` condition.
        self.builder.set_current_block(header_bb);
        for ((name, _, ty), &param_val) in loop_vars.iter().zip(header_params.iter()) {
            self.scope.insert(name.clone(), (param_val, ty.clone()));
        }
        let loop_var_param = header_params[0]; // first param is always the loop var
        let cond_result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond_result,
                op: BinOp::CmpLt,
                lhs: loop_var_param,
                rhs: end_val,
                ty: IrType::Scalar(DType::Bool),
            },
            Some(IrType::Scalar(DType::Bool)),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond: cond_result,
                then_block: body_bb,
                then_args: vec![],
                else_block: merge_bb,
                else_args: header_params.clone(),
            },
            None,
        );

        // 9. Body block.
        self.builder.set_current_block(body_bb);
        let loop_var_names: Vec<String> = loop_vars.iter().map(|(n, _, _)| n.clone()).collect();
        self.loop_stack.push((header_bb, merge_bb, loop_var_names));
        self.lower_block_stmts(body)?;
        self.loop_stack.pop();

        // 10. Emit increment and back-edge (if body not terminated by break/continue).
        if !self.builder.is_current_block_terminated() {
            let cur_loop_var = self
                .scope
                .get(&var.name)
                .map(|(v, _)| *v)
                .unwrap_or(loop_var_param);
            let one = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: one,
                    value: 1,
                    ty: loop_var_ty.clone(),
                },
                Some(loop_var_ty.clone()),
            );
            let incremented = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BinOp {
                    result: incremented,
                    op: BinOp::Add,
                    lhs: cur_loop_var,
                    rhs: one,
                    ty: loop_var_ty.clone(),
                },
                Some(loop_var_ty.clone()),
            );
            self.scope
                .insert(var.name.clone(), (incremented, loop_var_ty));

            let updated_vals: Vec<ValueId> = loop_vars
                .iter()
                .map(|(name, original_val, _)| {
                    self.scope
                        .get(name)
                        .map(|(v, _)| *v)
                        .unwrap_or(*original_val)
                })
                .collect();
            self.builder.push_instr(
                IrInstr::Br {
                    target: header_bb,
                    args: updated_vals,
                },
                None,
            );
        }

        // 11. Move to merge block; update scope with final loop-var values.
        self.builder.set_current_block(merge_bb);
        for ((name, _, ty), &merge_val) in loop_vars.iter().zip(merge_params.iter()) {
            self.scope.insert(name.clone(), (merge_val, ty.clone()));
        }

        let _ = span;
        Ok(())
    }

    /// Lowers a `loop { body }` (infinite loop). `break` exits to merge_bb.
    fn lower_loop(&mut self, body: &AstBlock, span: Span) -> Result<(), LowerError> {
        let loop_bb = self.builder.create_block(Some("loop_body"));
        let merge_bb = self.builder.create_block(Some("loop_merge"));

        self.builder.push_instr(
            IrInstr::Br {
                target: loop_bb,
                args: vec![],
            },
            None,
        );

        self.builder.set_current_block(loop_bb);
        self.loop_stack.push((loop_bb, merge_bb, vec![]));
        self.lower_block_stmts(body)?;
        self.loop_stack.pop();

        if !self.builder.is_current_block_terminated() {
            self.builder.push_instr(
                IrInstr::Br {
                    target: loop_bb,
                    args: vec![],
                },
                None,
            );
        }

        self.builder.set_current_block(merge_bb);
        let _ = span;
        Ok(())
    }

    /// Lowers `break` — jumps to the merge block of the innermost loop.
    fn lower_break(&mut self, span: Span) -> Result<(), LowerError> {
        let (_, merge_bb, loop_var_names) =
            self.loop_stack
                .last()
                .cloned()
                .ok_or_else(|| LowerError::Unsupported {
                    detail: "break outside of loop".into(),
                    span,
                })?;

        let args: Vec<ValueId> = loop_var_names
            .iter()
            .filter_map(|name| self.scope.get(name).map(|(v, _)| *v))
            .collect();

        self.builder.push_instr(
            IrInstr::Br {
                target: merge_bb,
                args,
            },
            None,
        );
        Ok(())
    }

    /// Lowers `continue` — jumps to the header block of the innermost loop.
    fn lower_continue(&mut self, span: Span) -> Result<(), LowerError> {
        let (header_bb, _, loop_var_names) =
            self.loop_stack
                .last()
                .cloned()
                .ok_or_else(|| LowerError::Unsupported {
                    detail: "continue outside of loop".into(),
                    span,
                })?;

        let args: Vec<ValueId> = loop_var_names
            .iter()
            .filter_map(|name| self.scope.get(name).map(|(v, _)| *v))
            .collect();

        self.builder.push_instr(
            IrInstr::Br {
                target: header_bb,
                args,
            },
            None,
        );
        Ok(())
    }

    fn lower_block(&mut self, block: &AstBlock) -> Result<Option<(ValueId, IrType)>, LowerError> {
        self.lower_block_stmts(block)?;
        if let Some(tail) = &block.tail {
            if self.builder.is_current_block_terminated() {
                // Block was terminated early (e.g. break in body) — skip tail.
                Ok(None)
            } else {
                Ok(Some(self.lower_expr(tail)?))
            }
        } else {
            Ok(None)
        }
    }

    /// Lowers just the statements of a block (no tail expression).
    fn lower_block_stmts(&mut self, block: &AstBlock) -> Result<(), LowerError> {
        for stmt in &block.stmts {
            if self.builder.is_current_block_terminated() {
                break;
            }
            self.lower_stmt(stmt)?;
        }
        Ok(())
    }

    fn lower_stmt(&mut self, stmt: &AstStmt) -> Result<(), LowerError> {
        match stmt {
            AstStmt::Let { name, init, .. } => {
                let (val, ty) = self.lower_expr(init)?;
                self.scope.insert(name.name.clone(), (val, ty));
                Ok(())
            }
            AstStmt::LetTuple { names, init, span } => {
                let (tuple_val, tuple_ty) = self.lower_expr(init)?;
                let elem_types = match &tuple_ty {
                    IrType::Tuple(elems) => elems.clone(),
                    _ => {
                        return Err(LowerError::Unsupported {
                            detail: format!("destructuring requires a tuple, got {}", tuple_ty),
                            span: *span,
                        });
                    }
                };
                if names.len() != elem_types.len() {
                    return Err(LowerError::Unsupported {
                        detail: format!(
                            "tuple has {} elements but destructuring binds {}",
                            elem_types.len(),
                            names.len()
                        ),
                        span: *span,
                    });
                }
                for (i, name) in names.iter().enumerate() {
                    let elem_ty = elem_types[i].clone();
                    let result = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::GetElement {
                            result,
                            base: tuple_val,
                            index: i,
                            result_ty: elem_ty.clone(),
                        },
                        Some(elem_ty.clone()),
                    );
                    self.scope.insert(name.name.clone(), (result, elem_ty));
                }
                Ok(())
            }
            AstStmt::Expr(expr) => {
                self.lower_expr(expr)?;
                Ok(())
            }
            AstStmt::While { cond, body, span } => self.lower_while(cond, body, *span),
            AstStmt::ForRange {
                var,
                start,
                end,
                body,
                span,
            } => self.lower_for_range(var, start, end, body, *span),
            AstStmt::Loop { body, span } => self.lower_loop(body, *span),
            AstStmt::Break { span } => self.lower_break(*span),
            AstStmt::Continue { span } => self.lower_continue(*span),
            AstStmt::Assign {
                target,
                value,
                span,
            } => {
                match target.as_ref() {
                    // Plain identifier assignment: rebind the name in scope (SSA-style).
                    AstExpr::Ident(ident) => {
                        let (new_val, new_ty) = self.lower_expr(value)?;
                        self.scope.insert(ident.name.clone(), (new_val, new_ty));
                        Ok(())
                    }
                    // Array element store: `arr[i] = value`  or  tensor store
                    AstExpr::Index { base, indices, span } => {
                        let (base_val, base_ty) = self.lower_expr(base)?;
                        if let IrType::Array { .. } = &base_ty {
                            // Array store
                            if indices.len() != 1 {
                                return Err(LowerError::Unsupported {
                                    detail: "array store requires exactly 1 index".into(),
                                    span: *span,
                                });
                            }
                            let (idx_val, _) = self.lower_expr(&indices[0])?;
                            let (value_val, _) = self.lower_expr(value)?;
                            self.builder.push_instr(
                                IrInstr::ArrayStore {
                                    array: base_val,
                                    index: idx_val,
                                    value: value_val,
                                },
                                None,
                            );
                            // Update the binding so the new array version is in scope
                            if let AstExpr::Ident(arr_ident) = base.as_ref() {
                                // Re-use the same ValueId (mutable array in place)
                                // The interpreter handles this by mutating the vector
                                let _ = arr_ident;
                            }
                            Ok(())
                        } else {
                            // Tensor element store
                            let mut idx_vals = Vec::new();
                            for idx in indices {
                                let (iv, _) = self.lower_expr(idx)?;
                                idx_vals.push(iv);
                            }
                            let (value_val, _) = self.lower_expr(value)?;
                            self.builder.push_instr(
                                IrInstr::Store {
                                    tensor: base_val,
                                    indices: idx_vals,
                                    value: value_val,
                                },
                                None,
                            );
                            Ok(())
                        }
                    }
                    _ => Err(LowerError::Unsupported {
                        detail: "assignment target must be an identifier or tensor index".into(),
                        span: *span,
                    }),
                }
            }
            AstStmt::Return { value, .. } => {
                let ret_values = if let Some(expr) = value {
                    let (val, _ty) = self.lower_expr(expr)?;
                    vec![val]
                } else {
                    vec![]
                };
                self.builder
                    .push_instr(IrInstr::Return { values: ret_values }, None);
                // Create a new unreachable block so any subsequent instructions
                // (from following statements) don't pollute the terminated block.
                let unreachable_bb = self.builder.create_block(Some("post_return"));
                self.builder.set_current_block(unreachable_bb);
                Ok(())
            }
        }
    }
}

fn lower_function(
    func: &AstFunction,
    module: &IrModule,
    fn_sigs: &HashMap<String, IrType>,
) -> Result<(crate::ir::function::IrFunction, Vec<crate::ir::function::IrFunction>), LowerError> {
    let return_ty = lower_type_with_structs(&func.return_ty, module);
    let params: Vec<Param> = func
        .params
        .iter()
        .map(|p| Param {
            name: p.name.name.clone(),
            ty: lower_type_with_structs(&p.ty, module),
        })
        .collect();

    let mut builder = IrFunctionBuilder::new(&func.name.name, params.clone(), return_ty.clone());
    let entry = builder.create_block(Some("entry"));
    builder.set_current_block(entry);

    let lambda_counter = std::rc::Rc::new(std::cell::Cell::new(0u32));
    let lifted_fns = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let mut lowerer = Lowerer::new_with_lambda_state(builder, module, fn_sigs, lambda_counter, lifted_fns.clone());

    // Register function parameters as entry block params.
    for param in &func.params {
        let ty = lower_type_with_structs(&param.ty, module);
        let val = lowerer
            .builder
            .add_block_param(entry, Some(&param.name.name), ty.clone());
        lowerer.scope.insert(param.name.name.clone(), (val, ty));
    }

    // Lower the body.
    let tail_val = lowerer.lower_block(&func.body)?;

    // Emit return only if the current block isn't already terminated
    // (e.g. the body ended with an early `return` statement).
    if !lowerer.builder.is_current_block_terminated() {
        let ret_values: Vec<ValueId> = match tail_val {
            Some((v, _)) => vec![v],
            None => vec![],
        };
        lowerer
            .builder
            .push_instr(IrInstr::Return { values: ret_values }, None);
    }

    // Seal any unterminated blocks (e.g. post_return dead-code blocks).
    lowerer.builder.seal_unterminated_blocks();

    let ir_func = lowerer.builder.build();
    let lifted = std::rc::Rc::try_unwrap(lifted_fns)
        .unwrap_or_else(|rc| std::rc::Rc::new(std::cell::RefCell::new(rc.borrow().clone())))
        .into_inner();
    Ok((ir_func, lifted))
}

// ---------------------------------------------------------------------------
// Type lowering helpers
// ---------------------------------------------------------------------------

pub fn lower_type(ty: &AstType) -> IrType {
    match ty {
        AstType::Scalar(kind, _) => IrType::Scalar(lower_dtype(*kind)),
        AstType::Tensor { dtype, dims, .. } => {
            let shape = Shape(dims.iter().map(lower_dim).collect());
            IrType::Tensor {
                dtype: lower_dtype(*dtype),
                shape,
            }
        }
        AstType::Named(name, _) => {
            if name == "str" {
                IrType::Str
            } else {
                IrType::Struct {
                    name: name.clone(),
                    fields: Vec::new(), // fields resolved at use-site
                }
            }
        }
        AstType::Tuple(elems, _) => IrType::Tuple(elems.iter().map(lower_type).collect()),
        AstType::Array { elem, len, .. } => IrType::Array {
            elem: Box::new(lower_type(elem)),
            len: *len,
        },
    }
}

/// Type lowering with struct/enum definition lookup from the module.
pub fn lower_type_with_structs(ty: &AstType, module: &IrModule) -> IrType {
    match ty {
        AstType::Array { elem, len, .. } => {
            return IrType::Array {
                elem: Box::new(lower_type_with_structs(elem, module)),
                len: *len,
            };
        }
        AstType::Named(name, _) => {
            if name == "str" {
                return IrType::Str;
            }
            if let Some(fields) = module.struct_def(name) {
                IrType::Struct {
                    name: name.clone(),
                    fields: fields.clone(),
                }
            } else if let Some(variants) = module.enum_def(name) {
                IrType::Enum {
                    name: name.clone(),
                    variants: variants.clone(),
                }
            } else {
                IrType::Struct {
                    name: name.clone(),
                    fields: Vec::new(),
                }
            }
        }
        AstType::Tuple(elems, _) => IrType::Tuple(
            elems
                .iter()
                .map(|e| lower_type_with_structs(e, module))
                .collect(),
        ),
        other => lower_type(other),
    }
}

fn lower_dtype(kind: AstScalarKind) -> DType {
    match kind {
        AstScalarKind::F32 => DType::F32,
        AstScalarKind::F64 => DType::F64,
        AstScalarKind::I32 => DType::I32,
        AstScalarKind::I64 => DType::I64,
        AstScalarKind::Bool => DType::Bool,
    }
}

fn lower_dim(dim: &AstDim) -> Dim {
    match dim {
        AstDim::Literal(n) => Dim::Literal(*n),
        AstDim::Symbol(sym) => Dim::Symbolic(sym.name.clone()),
    }
}

fn lower_binop(op: AstBinOp) -> BinOp {
    match op {
        AstBinOp::Add => BinOp::Add,
        AstBinOp::Sub => BinOp::Sub,
        AstBinOp::Mul => BinOp::Mul,
        AstBinOp::Div => BinOp::Div,
        AstBinOp::Mod => BinOp::Mod,
        AstBinOp::CmpEq => BinOp::CmpEq,
        AstBinOp::CmpNe => BinOp::CmpNe,
        AstBinOp::CmpLt => BinOp::CmpLt,
        AstBinOp::CmpLe => BinOp::CmpLe,
        AstBinOp::CmpGt => BinOp::CmpGt,
        AstBinOp::CmpGe => BinOp::CmpGe,
        // And/Or are handled via short-circuit lowering, never reach here.
        AstBinOp::And | AstBinOp::Or => {
            unreachable!("logical operators use short-circuit lowering")
        }
    }
}

/// Derives the result type of an einsum operation from the notation string and
/// input tensor types.
///
/// For bootstrap: parses the output index string from the notation (the part
/// after "->") and infers the result shape by matching symbolic dim names.
/// Falls back to `IrType::Infer` if the notation cannot be parsed.
fn derive_einsum_result_type(notation: &str, input_tys: &[IrType]) -> IrType {
    // Extract output indices: "mk,kn->mn" → "mn"
    let output_indices = match notation.find("->") {
        Some(pos) => &notation[pos + 2..],
        None => return IrType::Infer,
    };

    // Build a map from index character → symbolic Dim, using input shapes.
    let input_part = &notation[..notation.find("->").unwrap()];
    let input_index_strs: Vec<&str> = input_part.split(',').collect();

    let mut char_to_dim: HashMap<char, Dim> = HashMap::new();
    let mut result_dtype: Option<DType> = None;

    for (idx_str, ty) in input_index_strs.iter().zip(input_tys.iter()) {
        if let IrType::Tensor { dtype, shape } = ty {
            if result_dtype.is_none() {
                result_dtype = Some(*dtype);
            }
            for (ch, dim) in idx_str.chars().zip(shape.0.iter()) {
                char_to_dim.entry(ch).or_insert_with(|| dim.clone());
            }
        }
    }

    let dtype = match result_dtype {
        Some(d) => d,
        None => return IrType::Infer,
    };

    let result_dims: Vec<Dim> = output_indices
        .chars()
        .map(|ch| {
            char_to_dim
                .get(&ch)
                .cloned()
                .unwrap_or_else(|| Dim::Symbolic(ch.to_string()))
        })
        .collect();

    IrType::Tensor {
        dtype,
        shape: Shape(result_dims),
    }
}

/// Scans a block for variables that get rebound, returning unique names.
///
/// At the direct level: includes `val`/`var` binding names, `x = expr`
/// targets, and `for`-loop variables.
/// In nested blocks: recursively collects `x = expr` mutations so that outer
/// variables modified inside inner loops are threaded through as SSA params.
fn find_rebound_vars(block: &AstBlock) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for stmt in &block.stmts {
        match stmt {
            AstStmt::Let { name, .. } => {
                if !names.contains(&name.name) {
                    names.push(name.name.clone());
                }
            }
            AstStmt::Assign { target, .. } => {
                if let AstExpr::Ident(ident) = target.as_ref() {
                    if !names.contains(&ident.name) {
                        names.push(ident.name.clone());
                    }
                }
            }
            AstStmt::ForRange { var, body, .. } => {
                if !names.contains(&var.name) {
                    names.push(var.name.clone());
                }
                // Recurse into the for body to collect mutations of outer vars.
                collect_nested_mutations(body, &mut names);
            }
            AstStmt::While { body, .. } | AstStmt::Loop { body, .. } => {
                collect_nested_mutations(body, &mut names);
            }
            _ => {}
        }
    }
    names
}

/// Recursively collects `x = expr` assignment targets from nested blocks.
/// Does NOT add `Let`/`var` names (new local bindings, not outer mutations).
fn collect_nested_mutations(block: &AstBlock, names: &mut Vec<String>) {
    for stmt in &block.stmts {
        match stmt {
            AstStmt::Assign { target, .. } => {
                if let AstExpr::Ident(ident) = target.as_ref() {
                    if !names.contains(&ident.name) {
                        names.push(ident.name.clone());
                    }
                }
            }
            AstStmt::ForRange { body, .. }
            | AstStmt::While { body, .. }
            | AstStmt::Loop { body, .. } => {
                collect_nested_mutations(body, names);
            }
            _ => {}
        }
    }
}
