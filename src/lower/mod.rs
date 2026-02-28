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
    AstUnaryOp, AstWhenArm, AstWhenPattern, Ident,
};
use crate::parser::lexer::Span;

/// Returns true if a pattern extracts variable bindings (needs safe evaluation order with guards).
fn pattern_has_bindings(pattern: &AstWhenPattern) -> bool {
    match pattern {
        AstWhenPattern::OptionSome { binding: Some(_) } => true,
        AstWhenPattern::ResultOk { binding: Some(_) } => true,
        AstWhenPattern::ResultErr { binding: Some(_) } => true,
        AstWhenPattern::EnumVariant { bindings, .. } if !bindings.is_empty() => true,
        // A tuple pattern has bindings if ANY sub-pattern is an ident binding
        AstWhenPattern::Tuple(subs) => subs.iter().any(|s| pattern_has_bindings(s)
            || matches!(s, AstWhenPattern::EnumVariant { enum_name, .. } if enum_name.is_empty())),
        _ => false,
    }
}

/// Lower an `AstModule` to an `IrModule`.
pub fn lower(ast: &AstModule, module_name: &str) -> Result<IrModule, LowerError> {
    let mut module = IrModule::new(module_name);

    // 0. Register type aliases so structs/functions can reference them.
    for alias in &ast.type_aliases {
        let ir_ty = lower_type(&alias.ty);
        module
            .add_type_alias(alias.name.clone(), ir_ty)
            .map_err(|_| LowerError::DuplicateFunction {
                name: alias.name.clone(),
                span: alias.span,
            })?;
    }

    // 1. Register enum definitions so functions can reference them.
    for e in &ast.enums {
        let variants: Vec<String> = e.variants.iter().map(|v| v.name.name.clone()).collect();
        let variant_fields: Vec<Vec<IrType>> = e.variants.iter()
            .map(|v| v.fields.iter().map(|ty| lower_type(ty)).collect())
            .collect();
        module
            .add_enum_def(e.name.name.clone(), variants, variant_fields)
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
    // Generic functions (with type_params) are excluded from fn_sigs; they're
    // monomorphized on demand during lower_call.
    let mut fn_sigs: HashMap<String, IrType> = HashMap::new();
    let mut generic_fn_map: HashMap<String, crate::parser::ast::AstFunction> = HashMap::new();
    let mut fn_defaults_map: HashMap<String, Vec<Option<crate::parser::ast::AstExpr>>> = HashMap::new();
    for func in &ast.functions {
        if func.type_params.is_empty() {
            let ret_ty = lower_type_with_structs(&func.return_ty, &module);
            fn_sigs.insert(func.name.name.clone(), ret_ty);
        } else {
            generic_fn_map.insert(func.name.name.clone(), func.clone());
        }
        if func.params.iter().any(|p| p.default.is_some()) {
            fn_defaults_map.insert(
                func.name.name.clone(),
                func.params.iter().map(|p| p.default.clone()).collect(),
            );
        }
    }
    let generic_fns = std::rc::Rc::new(generic_fn_map);
    let fn_defaults = std::rc::Rc::new(fn_defaults_map);

    // 3b. Collect global const declarations as named expressions.
    let const_defs_map: HashMap<String, AstExpr> = ast.consts.iter()
        .map(|c| (c.name.name.clone(), c.value.clone()))
        .collect();
    let const_defs = std::rc::Rc::new(const_defs_map);

    // 3c. Process impl blocks — register mangled method names in fn_sigs and build
    // the trait dispatch table (method_name → [(dispatch_type, mangled_fn_name)]).
    // Mangling:
    //   - `impl Trait for Type { def method }` → `Trait__Type__method`
    //   - `impl Type { def method }` (trait_name == "") → `Type__method`
    let mut trait_dispatch_map: HashMap<String, Vec<(IrType, String)>> = HashMap::new();
    let mut struct_method_map: HashMap<String, HashMap<String, String>> = HashMap::new();
    let mut impl_fns: Vec<crate::parser::ast::AstFunction> = Vec::new();
    for impl_def in &ast.impls {
        let dispatch_ty = type_name_to_ir_type(&impl_def.type_name, &module);
        for method in &impl_def.methods {
            let mangled = if impl_def.trait_name.is_empty() {
                // Standalone struct method: `TypeName__method`
                format!("{}__{}", impl_def.type_name, method.name.name)
            } else {
                // Trait impl: `TraitName__TypeName__method`
                format!("{}__{}__{}", impl_def.trait_name, impl_def.type_name, method.name.name)
            };
            let ret_ty = lower_type_with_structs(&method.return_ty, &module);
            fn_sigs.insert(mangled.clone(), ret_ty);
            if impl_def.trait_name.is_empty() {
                // Register in struct_method_map for obj.method() dispatch.
                struct_method_map
                    .entry(impl_def.type_name.clone())
                    .or_default()
                    .insert(method.name.name.clone(), mangled.clone());
            } else {
                trait_dispatch_map
                    .entry(method.name.name.clone())
                    .or_default()
                    .push((dispatch_ty.clone(), mangled.clone()));
            }
            // Build a renamed copy of the method for lowering.
            let mut renamed = method.clone();
            renamed.name.name = mangled;
            impl_fns.push(renamed);
        }
    }
    let trait_dispatch = std::rc::Rc::new(trait_dispatch_map);
    // struct_method_map is only used for mangling; the mangled names are already in fn_sigs.
    let _ = struct_method_map;

    // Shared monomorphization state across all top-level function lowerings.
    let mono_cache = std::rc::Rc::new(std::cell::RefCell::new(std::collections::HashSet::new()));
    let mono_sigs = std::rc::Rc::new(std::cell::RefCell::new(HashMap::new()));

    // 4. Lower all non-generic function definitions (including impl methods).
    let mut all_lifted: Vec<crate::ir::function::IrFunction> = Vec::new();
    for func in ast.functions.iter().chain(impl_fns.iter()) {
        if !func.type_params.is_empty() {
            continue; // generic: lowered on demand at call sites
        }
        let (ir_func, lifted) = lower_function_with_generics(
            func, &module, &fn_sigs, &const_defs,
            generic_fns.clone(), mono_cache.clone(), mono_sigs.clone(),
            trait_dispatch.clone(), fn_defaults.clone(),
        )?;
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
    /// Tracks the concrete element type of channels (channel ValueId → elem IrType).
    /// Populated when `send(ch, val)` is first called; used by `recv(ch)` to avoid Infer.
    chan_elem_types: HashMap<ValueId, IrType>,
    /// Active type-parameter substitutions for monomorphized generic functions.
    /// Maps type param name (e.g. "T") → concrete IrType.
    type_param_subs: HashMap<String, IrType>,
    /// Generic function AST templates: function name → AstFunction.
    generic_fns: std::rc::Rc<HashMap<String, crate::parser::ast::AstFunction>>,
    /// Tracks already-monomorphized specializations (mangled names) to avoid duplication.
    mono_cache: std::rc::Rc<std::cell::RefCell<std::collections::HashSet<String>>>,
    /// Return types of monomorphized specializations (mangled name → IrType).
    mono_sigs: std::rc::Rc<std::cell::RefCell<HashMap<String, IrType>>>,
    /// Global constants available for inlining.
    const_defs: std::rc::Rc<HashMap<String, crate::parser::ast::AstExpr>>,
    /// Trait method dispatch table: method_name → [(dispatch_type, mangled_fn_name)].
    /// The dispatch_type is the IrType of the first argument used to select the impl.
    trait_dispatch: std::rc::Rc<HashMap<String, Vec<(IrType, String)>>>,
    /// Default parameter expressions: fn_name → [Option<AstExpr>] per param.
    fn_defaults: std::rc::Rc<HashMap<String, Vec<Option<crate::parser::ast::AstExpr>>>>,
}

impl<'m> Lowerer<'m> {
    fn new_with_lambda_state(
        builder: IrFunctionBuilder,
        module: &'m IrModule,
        fn_sigs: &'m HashMap<String, IrType>,
        lambda_counter: std::rc::Rc<std::cell::Cell<u32>>,
        lifted_fns: std::rc::Rc<std::cell::RefCell<Vec<crate::ir::function::IrFunction>>>,
    ) -> Self {
        Self::new_generic(builder, module, fn_sigs, lambda_counter, lifted_fns,
            HashMap::new(),
            std::rc::Rc::new(HashMap::new()),
            std::rc::Rc::new(std::cell::RefCell::new(std::collections::HashSet::new())),
            std::rc::Rc::new(std::cell::RefCell::new(HashMap::new())),
            std::rc::Rc::new(HashMap::new()),
            std::rc::Rc::new(HashMap::new()),
            std::rc::Rc::new(HashMap::new()),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_generic(
        builder: IrFunctionBuilder,
        module: &'m IrModule,
        fn_sigs: &'m HashMap<String, IrType>,
        lambda_counter: std::rc::Rc<std::cell::Cell<u32>>,
        lifted_fns: std::rc::Rc<std::cell::RefCell<Vec<crate::ir::function::IrFunction>>>,
        type_param_subs: HashMap<String, IrType>,
        generic_fns: std::rc::Rc<HashMap<String, crate::parser::ast::AstFunction>>,
        mono_cache: std::rc::Rc<std::cell::RefCell<std::collections::HashSet<String>>>,
        mono_sigs: std::rc::Rc<std::cell::RefCell<HashMap<String, IrType>>>,
        const_defs: std::rc::Rc<HashMap<String, crate::parser::ast::AstExpr>>,
        trait_dispatch: std::rc::Rc<HashMap<String, Vec<(IrType, String)>>>,
        fn_defaults: std::rc::Rc<HashMap<String, Vec<Option<crate::parser::ast::AstExpr>>>>,
    ) -> Self {
        Self {
            builder,
            scope: HashMap::new(),
            loop_stack: Vec::new(),
            module,
            fn_sigs,
            lambda_counter,
            lifted_fns,
            chan_elem_types: HashMap::new(),
            type_param_subs,
            generic_fns,
            mono_cache,
            mono_sigs,
            const_defs,
            trait_dispatch,
            fn_defaults,
        }
    }

    /// Resolves an AstType, applying type-parameter substitutions first.
    fn resolve_ty(&self, ty: &AstType) -> IrType {
        if let AstType::Named(name, _) = ty {
            if let Some(concrete) = self.type_param_subs.get(name) {
                return concrete.clone();
            }
        }
        lower_type_with_structs(ty, self.module)
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
            AstExpr::Ident(ident) => {
                // Special built-in identifiers
                if ident.name == "none" {
                    let result_ty = IrType::Option(Box::new(IrType::Infer));
                    let result = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::MakeNone { result, result_ty: result_ty.clone() },
                        Some(result_ty.clone()),
                    );
                    return Ok((result, result_ty));
                }
                // If the ident is not in scope, check if it's a named function —
                // create a first-class function reference via MakeClosure.
                if !self.scope.contains_key(&ident.name) {
                    if let Some(ret_ty) = self.fn_sigs.get(&ident.name).cloned() {
                        let fn_ty = IrType::Fn {
                            params: vec![],           // param types not tracked in fn_sigs
                            ret: Box::new(ret_ty.clone()),
                        };
                        let result = self.builder.fresh_value();
                        self.builder.push_instr(
                            IrInstr::MakeClosure {
                                result,
                                fn_name: ident.name.clone(),
                                captures: vec![],
                                result_ty: fn_ty.clone(),
                            },
                            Some(fn_ty.clone()),
                        );
                        return Ok((result, fn_ty));
                    }
                }
                self.lookup(ident)
            }

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

                // Auto-promote f32 <-> f64: widen the narrower operand so that
                // float literals (always f32) work transparently with f64 params.
                let (lhs_val, rhs_val, lhs_ty) = match (&lhs_ty, &rhs_ty) {
                    (IrType::Scalar(DType::F32), IrType::Scalar(DType::F64)) => {
                        let cast = self.builder.fresh_value();
                        self.builder.push_instr(
                            IrInstr::Cast {
                                result: cast,
                                operand: lhs_val,
                                from_ty: lhs_ty.clone(),
                                to_ty: IrType::Scalar(DType::F64),
                            },
                            Some(IrType::Scalar(DType::F64)),
                        );
                        (cast, rhs_val, IrType::Scalar(DType::F64))
                    }
                    (IrType::Scalar(DType::F64), IrType::Scalar(DType::F32)) => {
                        let cast = self.builder.fresh_value();
                        self.builder.push_instr(
                            IrInstr::Cast {
                                result: cast,
                                operand: rhs_val,
                                from_ty: rhs_ty.clone(),
                                to_ty: IrType::Scalar(DType::F64),
                            },
                            Some(IrType::Scalar(DType::F64)),
                        );
                        (lhs_val, cast, lhs_ty)
                    }
                    _ => {
                        // Require operand types to match for all other scalar binops.
                        if lhs_ty != rhs_ty {
                            return Err(LowerError::TypeMismatch {
                                expected: format!("{}", lhs_ty),
                                found: format!("{}", rhs_ty),
                                span: *span,
                            });
                        }
                        (lhs_val, rhs_val, lhs_ty)
                    }
                };

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
                                fields: vec![],
                                result_ty: result_ty.clone(),
                            },
                            Some(result_ty.clone()),
                        );
                        return Ok((result, result_ty));
                    }
                }
                // Normal struct field access — also handles grad<T>.value / grad<T>.grad
                let (base_val, base_ty) = self.lower_expr(base)?;
                // grad<T> pseudo-fields: .value → GradValue, .grad / .tangent → GradTangent
                if let IrType::Grad(inner) = &base_ty {
                    let inner_ty = *inner.clone();
                    let result = self.builder.fresh_value();
                    let (instr, ret_ty) = if field == "value" {
                        (IrInstr::GradValue { result, operand: base_val, ty: inner_ty.clone() }, inner_ty)
                    } else if field == "grad" || field == "tangent" {
                        (IrInstr::GradTangent { result, operand: base_val, ty: inner_ty.clone() }, inner_ty)
                    } else {
                        return Err(LowerError::Unsupported {
                            detail: format!("grad<T> has no field '{}'; use .value or .grad", field),
                            span: *span,
                        });
                    };
                    self.builder.push_instr(instr, Some(ret_ty.clone()));
                    return Ok((result, ret_ty));
                }
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

            // await expr: just lower the inner expression (async is a no-op at IR level)
            AstExpr::Await { expr, .. } => {
                self.lower_expr(expr)
            }

            AstExpr::Try { expr, span } => {
                let (val, res_ty) = self.lower_expr(expr)?;

                // Extract Ok/Err inner types from the result type.
                let (ok_ty, err_ty) = if let IrType::ResultType(ok, err) = &res_ty {
                    ((**ok).clone(), (**err).clone())
                } else {
                    (IrType::Infer, IrType::Infer)
                };

                // Emit IsOk test.
                let is_ok_result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::IsOk { result: is_ok_result, operand: val },
                    Some(IrType::Scalar(DType::Bool)),
                );

                let ok_bb = self.builder.create_block(Some("try_ok"));
                let err_bb = self.builder.create_block(Some("try_err"));
                let cont_bb = self.builder.create_block(Some("try_cont"));

                self.builder.push_instr(
                    IrInstr::CondBr {
                        cond: is_ok_result,
                        then_block: ok_bb,
                        then_args: vec![],
                        else_block: err_bb,
                        else_args: vec![],
                    },
                    None,
                );

                // Ok branch: unwrap and continue.
                self.builder.set_current_block(ok_bb);
                let ok_unwrapped = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ResultUnwrap { result: ok_unwrapped, operand: val, result_ty: ok_ty.clone() },
                    Some(ok_ty.clone()),
                );
                self.builder.push_instr(
                    IrInstr::Br { target: cont_bb, args: vec![ok_unwrapped] },
                    None,
                );

                // Err branch: early return.
                self.builder.set_current_block(err_bb);
                let err_unwrapped = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ResultUnwrapErr { result: err_unwrapped, operand: val, result_ty: err_ty.clone() },
                    Some(err_ty.clone()),
                );
                // Wrap the error in a result and return early.
                let err_result = self.builder.fresh_value();
                let err_ret_ty = IrType::ResultType(Box::new(IrType::Infer), Box::new(err_ty.clone()));
                self.builder.push_instr(
                    IrInstr::MakeErr { result: err_result, value: err_unwrapped, result_ty: err_ret_ty.clone() },
                    Some(err_ret_ty.clone()),
                );
                self.builder.push_instr(
                    IrInstr::Return { values: vec![err_result] },
                    None,
                );

                // Continuation block: receives the Ok value.
                self.builder.set_current_block(cont_bb);
                let ok_result = self.builder.add_block_param(cont_bb, Some("try_result"), ok_ty.clone());
                let _ = span;
                Ok((ok_result, ok_ty))
            }

            AstExpr::MethodCall { base, method, args, span } => {
                // Check if base is a bare identifier naming an enum → variant construction with data.
                // e.g. `Shape.Circle(3.14)` is parsed as MethodCall(base=Ident("Shape"), method="Circle", args=[3.14])
                if let AstExpr::Ident(base_ident) = base.as_ref() {
                    if let Some(variants) = self.module.enum_def(&base_ident.name) {
                        let variants = variants.clone();
                        if let Some(variant_idx) = variants.iter().position(|v| v == method) {
                            // This is an enum variant constructor with data.
                            let mut field_vals = Vec::with_capacity(args.len());
                            for arg in args {
                                let (v, _) = self.lower_expr(arg)?;
                                field_vals.push(v);
                            }
                            let result_ty = IrType::Enum {
                                name: base_ident.name.clone(),
                                variants,
                            };
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::MakeVariant {
                                    result,
                                    variant_idx,
                                    fields: field_vals,
                                    result_ty: result_ty.clone(),
                                },
                                Some(result_ty.clone()),
                            );
                            return Ok((result, result_ty));
                        }
                    }
                }

                // Lower the receiver.
                let (base_val, base_ty) = self.lower_expr(base)?;

                // List functional method dispatch.
                if let IrType::List(inner_elem_ty) = &base_ty.clone() {
                    let elem_ty = *inner_elem_ty.clone();
                    match method.as_str() {
                        "map"    => return self.lower_list_map(base_val, elem_ty, args, *span),
                        "filter" => return self.lower_list_filter(base_val, elem_ty, args, *span),
                        "fold"   => return self.lower_list_fold(base_val, elem_ty, args, *span),
                        "any"    => return self.lower_list_any(base_val, elem_ty, args, *span),
                        "all"    => return self.lower_list_all(base_val, elem_ty, args, *span),
                        _ => {} // fall through to struct method dispatch
                    }
                }

                // Determine the struct type name.
                let type_name = match &base_ty {
                    IrType::Struct { name, .. } => name.clone(),
                    other => {
                        return Err(LowerError::Unsupported {
                            detail: format!(
                                "method call '.{}' on non-struct type {}",
                                method, other
                            ),
                            span: *span,
                        });
                    }
                };

                // Build the mangled function name `TypeName__method`.
                let mangled = format!("{}__{}", type_name, method);

                // Look up return type.
                let ret_ty = self.fn_sigs.get(&mangled).cloned().unwrap_or(IrType::Infer);

                // Lower remaining arguments.
                let mut arg_vals = vec![base_val];
                for arg in args {
                    let (v, _) = self.lower_expr(arg)?;
                    arg_vals.push(v);
                }

                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::Call {
                        result: Some(result),
                        callee: mangled,
                        args: arg_vals,
                        result_ty: Some(ret_ty.clone()),
                    },
                    Some(ret_ty.clone()),
                );
                Ok((result, ret_ty))
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
                ty: self.resolve_ty(&p.ty),
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
            let ty = self.resolve_ty(&p.ty);
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
        // Built-in: channel() → ChanNew
        if callee.name == "channel" {
            let elem_ty = IrType::Infer;
            let chan_ty = IrType::Chan(Box::new(elem_ty.clone()));
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ChanNew { result, elem_ty },
                Some(chan_ty.clone()),
            );
            return Ok((result, chan_ty));
        }

        // Built-in: send(ch, v) → ChanSend (returns unit, use dummy i64 0)
        if callee.name == "send" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "send() requires exactly 2 arguments (channel, value)".into(),
                    span,
                });
            }
            let (chan_val, _) = self.lower_expr(&args[0])?;
            let (val, val_ty) = self.lower_expr(&args[1])?;
            // Record the concrete element type so recv() can use it.
            self.chan_elem_types.entry(chan_val).or_insert_with(|| val_ty.clone());
            self.builder.push_instr(
                IrInstr::ChanSend { chan: chan_val, value: val },
                None,
            );
            // Return a dummy i64 0 as the "unit" value.
            let dummy = self.builder.fresh_value();
            let dummy_ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::ConstInt { result: dummy, value: 0, ty: dummy_ty.clone() },
                Some(dummy_ty.clone()),
            );
            return Ok((dummy, dummy_ty));
        }

        // Built-in: recv(ch) → ChanRecv
        if callee.name == "recv" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "recv() requires exactly 1 argument (channel)".into(),
                    span,
                });
            }
            let (chan_val, chan_ty) = self.lower_expr(&args[0])?;
            // Prefer the concrete element type recorded when send() was called.
            let elem_ty = self.chan_elem_types.get(&chan_val).cloned()
                .unwrap_or_else(|| {
                    if let IrType::Chan(elem) = chan_ty { *elem } else { IrType::Infer }
                });
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ChanRecv { result, chan: chan_val, elem_ty: elem_ty.clone() },
                Some(elem_ty.clone()),
            );
            return Ok((result, elem_ty));
        }

        // Built-in: atomic_new(v) → AtomicNew
        if callee.name == "atomic_new" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "atomic_new() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, inner_ty) = self.lower_expr(&args[0])?;
            let result_ty = IrType::Atomic(Box::new(inner_ty));
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::AtomicNew { result, value: val, result_ty: result_ty.clone() },
                Some(result_ty.clone()),
            );
            return Ok((result, result_ty));
        }

        // Built-in: atomic_load(a) → AtomicLoad
        if callee.name == "atomic_load" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "atomic_load() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, atomic_ty) = self.lower_expr(&args[0])?;
            let inner_ty = if let IrType::Atomic(inner) = atomic_ty { *inner } else { IrType::Infer };
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::AtomicLoad { result, atomic: val, result_ty: inner_ty.clone() },
                Some(inner_ty.clone()),
            );
            return Ok((result, inner_ty));
        }

        // Built-in: atomic_store(a, v) → AtomicStore
        if callee.name == "atomic_store" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "atomic_store() requires exactly 2 arguments".into(),
                    span,
                });
            }
            let (a, _) = self.lower_expr(&args[0])?;
            let (v, _) = self.lower_expr(&args[1])?;
            self.builder.push_instr(IrInstr::AtomicStore { atomic: a, value: v }, None);
            let dummy = self.builder.fresh_value();
            let dummy_ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(IrInstr::ConstInt { result: dummy, value: 0, ty: dummy_ty.clone() }, Some(dummy_ty.clone()));
            return Ok((dummy, dummy_ty));
        }

        // Built-in: mutex_new(v) → MutexNew
        if callee.name == "mutex_new" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "mutex_new() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, inner_ty) = self.lower_expr(&args[0])?;
            let result_ty = IrType::Mutex(Box::new(inner_ty));
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::MutexNew { result, value: val, result_ty: result_ty.clone() },
                Some(result_ty.clone()),
            );
            return Ok((result, result_ty));
        }

        // Built-in: mutex_lock(m) → MutexLock
        if callee.name == "mutex_lock" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "mutex_lock() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, mutex_ty) = self.lower_expr(&args[0])?;
            let inner_ty = if let IrType::Mutex(inner) = mutex_ty { *inner } else { IrType::Infer };
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::MutexLock { result, mutex: val, result_ty: inner_ty.clone() },
                Some(inner_ty.clone()),
            );
            return Ok((result, inner_ty));
        }

        // Built-in: barrier() → Barrier (sync point, no-op in interpreter)
        if callee.name == "barrier" {
            self.builder.push_instr(IrInstr::Barrier, None);
            let dummy = self.builder.fresh_value();
            let dummy_ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::ConstInt { result: dummy, value: 0, ty: dummy_ty.clone() },
                Some(dummy_ty.clone()),
            );
            return Ok((dummy, dummy_ty));
        }

        // Built-in: mutex_unlock(m) → MutexUnlock (no-op in interpreter, returns unit)
        if callee.name == "mutex_unlock" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "mutex_unlock() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, _) = self.lower_expr(&args[0])?;
            self.builder.push_instr(IrInstr::MutexUnlock { mutex: val }, None);
            let dummy = self.builder.fresh_value();
            let dummy_ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::ConstInt { result: dummy, value: 0, ty: dummy_ty.clone() },
                Some(dummy_ty.clone()),
            );
            return Ok((dummy, dummy_ty));
        }

        // Built-in: atomic_add(a, v) → AtomicAdd (returns new value)
        if callee.name == "atomic_add" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "atomic_add() requires exactly 2 arguments".into(),
                    span,
                });
            }
            let (a, atomic_ty) = self.lower_expr(&args[0])?;
            let (v, _) = self.lower_expr(&args[1])?;
            let inner_ty = if let IrType::Atomic(inner) = atomic_ty { *inner } else { IrType::Scalar(DType::I64) };
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::AtomicAdd { result, atomic: a, value: v, result_ty: inner_ty.clone() },
                Some(inner_ty.clone()),
            );
            return Ok((result, inner_ty));
        }

        // Built-in: some(v) → MakeSome
        if callee.name == "some" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "some() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, inner_ty) = self.lower_expr(&args[0])?;
            let result_ty = IrType::Option(Box::new(inner_ty));
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::MakeSome { result, value: val, result_ty: result_ty.clone() },
                Some(result_ty.clone()),
            );
            return Ok((result, result_ty));
        }

        // Built-in: none() → MakeNone (also handled as identifier)
        if callee.name == "none" {
            let result_ty = IrType::Option(Box::new(IrType::Infer));
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::MakeNone { result, result_ty: result_ty.clone() },
                Some(result_ty.clone()),
            );
            return Ok((result, result_ty));
        }

        // Built-in: is_some(v) → IsSome
        if callee.name == "is_some" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "is_some() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::Bool);
            self.builder.push_instr(
                IrInstr::IsSome { result, operand: val },
                Some(ty.clone()),
            );
            return Ok((result, ty));
        }

        // Built-in: unwrap(v) → OptionUnwrap (option<T>) or ResultUnwrap (result<T,E>)
        if callee.name == "unwrap" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "unwrap() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, val_ty) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            match &val_ty {
                IrType::ResultType(ok_ty, _) => {
                    let inner_ty = (**ok_ty).clone();
                    self.builder.push_instr(
                        IrInstr::ResultUnwrap { result, operand: val, result_ty: inner_ty.clone() },
                        Some(inner_ty.clone()),
                    );
                    return Ok((result, inner_ty));
                }
                IrType::Option(inner) => {
                    let inner_ty = (**inner).clone();
                    self.builder.push_instr(
                        IrInstr::OptionUnwrap { result, operand: val, result_ty: inner_ty.clone() },
                        Some(inner_ty.clone()),
                    );
                    return Ok((result, inner_ty));
                }
                _ => {
                    // Fallback — ValidatePass will catch remaining Infer.
                    self.builder.push_instr(
                        IrInstr::OptionUnwrap { result, operand: val, result_ty: IrType::Infer },
                        Some(IrType::Infer),
                    );
                    return Ok((result, IrType::Infer));
                }
            }
        }

        // Built-in: ok(v) → MakeOk
        if callee.name == "ok" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "ok() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, inner_ty) = self.lower_expr(&args[0])?;
            let result_ty = IrType::ResultType(Box::new(inner_ty), Box::new(IrType::Infer));
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::MakeOk { result, value: val, result_ty: result_ty.clone() },
                Some(result_ty.clone()),
            );
            return Ok((result, result_ty));
        }

        // Built-in: err(v) → MakeErr
        if callee.name == "err" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "err() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, inner_ty) = self.lower_expr(&args[0])?;
            let result_ty = IrType::ResultType(Box::new(IrType::Infer), Box::new(inner_ty));
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::MakeErr { result, value: val, result_ty: result_ty.clone() },
                Some(result_ty.clone()),
            );
            return Ok((result, result_ty));
        }

        // Built-in: is_ok(v) → IsOk
        if callee.name == "is_ok" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "is_ok() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::Bool);
            self.builder.push_instr(
                IrInstr::IsOk { result, operand: val },
                Some(ty.clone()),
            );
            return Ok((result, ty));
        }

        // Built-in: is_none(v) → !IsSome
        if callee.name == "is_none" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "is_none() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, _) = self.lower_expr(&args[0])?;
            let bool_ty = IrType::Scalar(DType::Bool);
            let is_some = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::IsSome { result: is_some, operand: val },
                Some(bool_ty.clone()),
            );
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::UnaryOp { result, op: ScalarUnaryOp::Not, operand: is_some, ty: bool_ty.clone() },
                Some(bool_ty.clone()),
            );
            return Ok((result, bool_ty));
        }

        // Built-in: is_err(v) → !IsOk
        if callee.name == "is_err" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "is_err() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, _) = self.lower_expr(&args[0])?;
            let bool_ty = IrType::Scalar(DType::Bool);
            let is_ok = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::IsOk { result: is_ok, operand: val },
                Some(bool_ty.clone()),
            );
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::UnaryOp { result, op: ScalarUnaryOp::Not, operand: is_ok, ty: bool_ty.clone() },
                Some(bool_ty.clone()),
            );
            return Ok((result, bool_ty));
        }

        // Built-in: unwrap_err(v) → ResultUnwrapErr
        if callee.name == "unwrap_err" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "unwrap_err() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, val_ty) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let err_ty = match &val_ty {
                IrType::ResultType(_, err_ty) => (**err_ty).clone(),
                _ => IrType::Infer,
            };
            self.builder.push_instr(
                IrInstr::ResultUnwrapErr { result, operand: val, result_ty: err_ty.clone() },
                Some(err_ty.clone()),
            );
            return Ok((result, err_ty));
        }

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

        // Built-in: to_str(v) → ValueToStr
        if callee.name == "to_str" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "to_str() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (operand, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ValueToStr { result, operand },
                Some(IrType::Str),
            );
            return Ok((result, IrType::Str));
        }

        // Built-in: format("...", args...) — split on "{}" and concat with args
        if callee.name == "format" {
            if args.is_empty() {
                return Err(LowerError::Unsupported {
                    detail: "format() requires at least 1 argument (the format string)".into(),
                    span,
                });
            }
            // First arg must be a string literal.
            let fmt_str = match &args[0] {
                AstExpr::StringLit { value, .. } => value.clone(),
                _ => return Err(LowerError::Unsupported {
                    detail: "format() first argument must be a string literal".into(),
                    span,
                }),
            };
            // Split the format string on "{}" to get pieces.
            let pieces: Vec<&str> = fmt_str.split("{}").collect();
            let n_holes = pieces.len().saturating_sub(1);
            if n_holes != args.len() - 1 {
                return Err(LowerError::Unsupported {
                    detail: format!(
                        "format() has {} holes but {} arguments",
                        n_holes,
                        args.len() - 1
                    ),
                    span,
                });
            }
            // Lower each argument (skip index 0, the format string).
            let mut arg_vals: Vec<ValueId> = Vec::new();
            for arg in &args[1..] {
                let (v, _) = self.lower_expr(arg)?;
                // Convert to string representation.
                let s = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ValueToStr { result: s, operand: v },
                    Some(IrType::Str),
                );
                arg_vals.push(s);
            }
            // Build the concatenated string: piece[0] + arg[0] + piece[1] + arg[1] + ...
            // Start with the first piece as a ConstStr.
            let mut acc = {
                let r = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstStr { result: r, value: pieces[0].to_owned() },
                    Some(IrType::Str),
                );
                r
            };
            for i in 0..n_holes {
                // Concat with the argument.
                let after_arg = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::StrConcat { result: after_arg, lhs: acc, rhs: arg_vals[i] },
                    Some(IrType::Str),
                );
                // Concat with the next piece.
                let next_piece = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstStr { result: next_piece, value: pieces[i + 1].to_owned() },
                    Some(IrType::Str),
                );
                acc = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::StrConcat { result: acc, lhs: after_arg, rhs: next_piece },
                    Some(IrType::Str),
                );
            }
            return Ok((acc, IrType::Str));
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

        // Built-in: read_line() → ReadLine
        if callee.name == "read_line" {
            if !args.is_empty() {
                return Err(LowerError::Unsupported {
                    detail: "read_line() takes no arguments".into(),
                    span,
                });
            }
            let result = self.builder.fresh_value();
            self.builder.push_instr(IrInstr::ReadLine { result }, Some(IrType::Str));
            return Ok((result, IrType::Str));
        }

        // Built-in: read_i64() → ReadI64
        if callee.name == "read_i64" {
            if !args.is_empty() {
                return Err(LowerError::Unsupported {
                    detail: "read_i64() takes no arguments".into(),
                    span,
                });
            }
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(IrInstr::ReadI64 { result }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: read_f64() → ReadF64
        if callee.name == "read_f64" {
            if !args.is_empty() {
                return Err(LowerError::Unsupported {
                    detail: "read_f64() takes no arguments".into(),
                    span,
                });
            }
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::F64);
            self.builder.push_instr(IrInstr::ReadF64 { result }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: parse_i64(s) → ParseI64 → option<i64>
        if callee.name == "parse_i64" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "parse_i64() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (operand, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Option(Box::new(IrType::Scalar(DType::I64)));
            self.builder.push_instr(IrInstr::ParseI64 { result, operand }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: parse_f64(s) → ParseF64 → option<f64>
        if callee.name == "parse_f64" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "parse_f64() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (operand, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Option(Box::new(IrType::Scalar(DType::F64)));
            self.builder.push_instr(IrInstr::ParseF64 { result, operand }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: str_index(s, i) → StrIndex → i64
        if callee.name == "str_index" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "str_index() requires 2 arguments: (str, i64)".into(),
                    span,
                });
            }
            let (string, _) = self.lower_expr(&args[0])?;
            let (index, _) = self.lower_expr(&args[1])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(IrInstr::StrIndex { result, string, index }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: slice(s, start, end) → StrSlice → str
        if callee.name == "slice" {
            if args.len() != 3 {
                return Err(LowerError::Unsupported {
                    detail: "slice() requires 3 arguments: (str, i64, i64)".into(),
                    span,
                });
            }
            let (string, _) = self.lower_expr(&args[0])?;
            let (start, _) = self.lower_expr(&args[1])?;
            let (end, _) = self.lower_expr(&args[2])?;
            let result = self.builder.fresh_value();
            self.builder.push_instr(IrInstr::StrSlice { result, string, start, end }, Some(IrType::Str));
            return Ok((result, IrType::Str));
        }

        // Built-in: find(s, sub) → StrFind → option<i64>
        if callee.name == "find" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "find() requires 2 arguments: (str, str)".into(),
                    span,
                });
            }
            let (haystack, _) = self.lower_expr(&args[0])?;
            let (needle, _) = self.lower_expr(&args[1])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Option(Box::new(IrType::Scalar(DType::I64)));
            self.builder.push_instr(IrInstr::StrFind { result, haystack, needle }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: str_replace(s, old, new) → StrReplace → str
        if callee.name == "str_replace" {
            if args.len() != 3 {
                return Err(LowerError::Unsupported {
                    detail: "str_replace() requires 3 arguments: (str, str, str)".into(),
                    span,
                });
            }
            let (string, _) = self.lower_expr(&args[0])?;
            let (from, _) = self.lower_expr(&args[1])?;
            let (to, _) = self.lower_expr(&args[2])?;
            let result = self.builder.fresh_value();
            self.builder.push_instr(IrInstr::StrReplace { result, string, from, to }, Some(IrType::Str));
            return Ok((result, IrType::Str));
        }

        // Built-in: list(elem_ty) → ListNew — create an empty list
        // We infer the element type from the first push, or default to i64.
        // Usage: list() creates list<i64> by default; type annotation determines actual type.
        if callee.name == "list" {
            if !args.is_empty() {
                return Err(LowerError::Unsupported {
                    detail: "list() takes no arguments — it creates an empty dynamic list".into(),
                    span,
                });
            }
            let elem_ty = IrType::Scalar(DType::I64); // default; type inference may refine
            let result = self.builder.fresh_value();
            let list_ty = IrType::List(Box::new(elem_ty.clone()));
            self.builder.push_instr(IrInstr::ListNew { result, elem_ty }, Some(list_ty.clone()));
            return Ok((result, list_ty));
        }

        // Built-in: push(lst, val) → ListPush — append to list
        if callee.name == "push" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "push() requires 2 arguments: (list, value)".into(),
                    span,
                });
            }
            let (list, _) = self.lower_expr(&args[0])?;
            let (value, _) = self.lower_expr(&args[1])?;
            self.builder.push_instr(IrInstr::ListPush { list, value }, None);
            let dummy = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(IrInstr::ConstInt { result: dummy, value: 0, ty: ty.clone() }, Some(ty.clone()));
            return Ok((dummy, ty));
        }

        // Built-in: list_len(lst) → ListLen → i64
        if callee.name == "list_len" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "list_len() requires 1 argument".into(),
                    span,
                });
            }
            let (list, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(IrInstr::ListLen { result, list }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: list_get(lst, i) → ListGet → elem
        if callee.name == "list_get" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "list_get() requires 2 arguments: (list, index)".into(),
                    span,
                });
            }
            let (list, list_ty) = self.lower_expr(&args[0])?;
            let (index, _) = self.lower_expr(&args[1])?;
            let elem_ty = if let IrType::List(inner) = &list_ty { *inner.clone() } else { IrType::Scalar(DType::I64) };
            let result = self.builder.fresh_value();
            self.builder.push_instr(IrInstr::ListGet { result, list, index, elem_ty: elem_ty.clone() }, Some(elem_ty.clone()));
            return Ok((result, elem_ty));
        }

        // Built-in: list_set(lst, i, val) → ListSet
        if callee.name == "list_set" {
            if args.len() != 3 {
                return Err(LowerError::Unsupported {
                    detail: "list_set() requires 3 arguments: (list, index, value)".into(),
                    span,
                });
            }
            let (list, _) = self.lower_expr(&args[0])?;
            let (index, _) = self.lower_expr(&args[1])?;
            let (value, _) = self.lower_expr(&args[2])?;
            self.builder.push_instr(IrInstr::ListSet { list, index, value }, None);
            let dummy = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(IrInstr::ConstInt { result: dummy, value: 0, ty: ty.clone() }, Some(ty.clone()));
            return Ok((dummy, ty));
        }

        // Built-in: list_pop(lst) → ListPop → elem
        if callee.name == "list_pop" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "list_pop() requires 1 argument".into(),
                    span,
                });
            }
            let (list, list_ty) = self.lower_expr(&args[0])?;
            let elem_ty = if let IrType::List(inner) = &list_ty { *inner.clone() } else { IrType::Scalar(DType::I64) };
            let result = self.builder.fresh_value();
            self.builder.push_instr(IrInstr::ListPop { result, list, elem_ty: elem_ty.clone() }, Some(elem_ty.clone()));
            return Ok((result, elem_ty));
        }

        // Built-in: map() → MapNew — create an empty hash map (keys: str, values: i64 default)
        if callee.name == "map" {
            if !args.is_empty() {
                return Err(LowerError::Unsupported {
                    detail: "map() takes no arguments — it creates an empty hash map".into(),
                    span,
                });
            }
            let key_ty = IrType::Str;
            let val_ty = IrType::Scalar(DType::I64);
            let result = self.builder.fresh_value();
            let map_ty = IrType::Map(Box::new(key_ty.clone()), Box::new(val_ty.clone()));
            self.builder.push_instr(IrInstr::MapNew { result, key_ty, val_ty }, Some(map_ty.clone()));
            return Ok((result, map_ty));
        }

        // Built-in: map_set(m, k, v) → MapSet
        if callee.name == "map_set" {
            if args.len() != 3 {
                return Err(LowerError::Unsupported {
                    detail: "map_set() requires 3 arguments: (map, key, value)".into(),
                    span,
                });
            }
            let (map, _) = self.lower_expr(&args[0])?;
            let (key, _) = self.lower_expr(&args[1])?;
            let (value, _) = self.lower_expr(&args[2])?;
            self.builder.push_instr(IrInstr::MapSet { map, key, value }, None);
            let dummy = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(IrInstr::ConstInt { result: dummy, value: 0, ty: ty.clone() }, Some(ty.clone()));
            return Ok((dummy, ty));
        }

        // Built-in: map_get(m, k) → MapGet → option<val_ty>
        if callee.name == "map_get" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "map_get() requires 2 arguments: (map, key)".into(),
                    span,
                });
            }
            let (map, map_ty) = self.lower_expr(&args[0])?;
            let (key, _) = self.lower_expr(&args[1])?;
            let val_ty = if let IrType::Map(_, v) = &map_ty { *v.clone() } else { IrType::Scalar(DType::I64) };
            let opt_ty = IrType::Option(Box::new(val_ty.clone()));
            let result = self.builder.fresh_value();
            self.builder.push_instr(IrInstr::MapGet { result, map, key, val_ty }, Some(opt_ty.clone()));
            return Ok((result, opt_ty));
        }

        // Built-in: map_contains(m, k) → MapContains → bool
        if callee.name == "map_contains" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "map_contains() requires 2 arguments: (map, key)".into(),
                    span,
                });
            }
            let (map, _) = self.lower_expr(&args[0])?;
            let (key, _) = self.lower_expr(&args[1])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::Bool);
            self.builder.push_instr(IrInstr::MapContains { result, map, key }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: map_remove(m, k) → MapRemove
        if callee.name == "map_remove" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "map_remove() requires 2 arguments: (map, key)".into(),
                    span,
                });
            }
            let (map, _) = self.lower_expr(&args[0])?;
            let (key, _) = self.lower_expr(&args[1])?;
            self.builder.push_instr(IrInstr::MapRemove { map, key }, None);
            let dummy = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(IrInstr::ConstInt { result: dummy, value: 0, ty: ty.clone() }, Some(ty.clone()));
            return Ok((dummy, ty));
        }

        // Built-in: map_len(m) → MapLen → i64
        if callee.name == "map_len" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "map_len() requires 1 argument".into(),
                    span,
                });
            }
            let (map, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(IrInstr::MapLen { result, map }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // ── Phase 56: File I/O builtins ──────────────────────────────────────

        // Built-in: file_read_all(path) → FileReadAll → result<str, str>
        if callee.name == "file_read_all" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported { detail: "file_read_all() requires 1 argument".into(), span });
            }
            let (path, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::ResultType(Box::new(IrType::Str), Box::new(IrType::Str));
            self.builder.push_instr(IrInstr::FileReadAll { result, path }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: file_write_all(path, content) → FileWriteAll → result<i64, str>
        if callee.name == "file_write_all" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported { detail: "file_write_all() requires 2 arguments".into(), span });
            }
            let (path, _) = self.lower_expr(&args[0])?;
            let (content, _) = self.lower_expr(&args[1])?;
            let result = self.builder.fresh_value();
            let ty = IrType::ResultType(Box::new(IrType::Scalar(DType::I64)), Box::new(IrType::Str));
            self.builder.push_instr(IrInstr::FileWriteAll { result, path, content }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: file_exists(path) → FileExists → bool
        if callee.name == "file_exists" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported { detail: "file_exists() requires 1 argument".into(), span });
            }
            let (path, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::Bool);
            self.builder.push_instr(IrInstr::FileExists { result, path }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: file_lines(path) → FileLines → list<str>
        if callee.name == "file_lines" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported { detail: "file_lines() requires 1 argument".into(), span });
            }
            let (path, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::List(Box::new(IrType::Str));
            self.builder.push_instr(IrInstr::FileLines { result, path }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // ── Phase 58: Extended collection builtins ───────────────────────────

        // Built-in: list_contains(list, val) → ListContains → bool
        if callee.name == "list_contains" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported { detail: "list_contains() requires 2 arguments".into(), span });
            }
            let (list, _) = self.lower_expr(&args[0])?;
            let (value, _) = self.lower_expr(&args[1])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::Bool);
            self.builder.push_instr(IrInstr::ListContains { result, list, value }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: list_sort(list) → ListSort (side-effecting, returns unit-like dummy)
        if callee.name == "list_sort" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported { detail: "list_sort() requires 1 argument".into(), span });
            }
            let (list, _) = self.lower_expr(&args[0])?;
            self.builder.push_instr(IrInstr::ListSort { list }, None);
            let dummy = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(IrInstr::ConstInt { result: dummy, value: 0, ty: ty.clone() }, Some(ty.clone()));
            return Ok((dummy, ty));
        }

        // Built-in: map_keys(map) → MapKeys → list<str>
        if callee.name == "map_keys" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported { detail: "map_keys() requires 1 argument".into(), span });
            }
            let (map, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::List(Box::new(IrType::Str));
            self.builder.push_instr(IrInstr::MapKeys { result, map }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: map_values(map) → MapValues → list<?>
        if callee.name == "map_values" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported { detail: "map_values() requires 1 argument".into(), span });
            }
            let (map, map_ty) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let val_ty = if let IrType::Map(_, v) = &map_ty { *v.clone() } else { IrType::Scalar(DType::I64) };
            let ty = IrType::List(Box::new(val_ty));
            self.builder.push_instr(IrInstr::MapValues { result, map }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: list_concat(a, b) → ListConcat → list
        if callee.name == "list_concat" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported { detail: "list_concat() requires 2 arguments".into(), span });
            }
            let (lhs, lhs_ty) = self.lower_expr(&args[0])?;
            let (rhs, _) = self.lower_expr(&args[1])?;
            let result = self.builder.fresh_value();
            self.builder.push_instr(IrInstr::ListConcat { result, lhs, rhs }, Some(lhs_ty.clone()));
            return Ok((result, lhs_ty));
        }

        // Built-in: list_slice(list, start, end) → ListSlice → list
        if callee.name == "list_slice" {
            if args.len() != 3 {
                return Err(LowerError::Unsupported { detail: "list_slice() requires 3 arguments".into(), span });
            }
            let (list, list_ty) = self.lower_expr(&args[0])?;
            let (start, _) = self.lower_expr(&args[1])?;
            let (end, _) = self.lower_expr(&args[2])?;
            let result = self.builder.fresh_value();
            self.builder.push_instr(IrInstr::ListSlice { result, list, start, end }, Some(list_ty.clone()));
            return Ok((result, list_ty));
        }

        // ── Phase 59: Process / environment builtins ─────────────────────────

        // Built-in: exit(code) → ProcessExit (does not return)
        if callee.name == "exit" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported { detail: "exit() requires 1 argument".into(), span });
            }
            let (code, _) = self.lower_expr(&args[0])?;
            self.builder.push_instr(IrInstr::ProcessExit { code }, None);
            let dummy = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(IrInstr::ConstInt { result: dummy, value: 0, ty: ty.clone() }, Some(ty.clone()));
            return Ok((dummy, ty));
        }

        // Built-in: args() → ProcessArgs → list<str>
        if callee.name == "args" {
            if !args.is_empty() {
                return Err(LowerError::Unsupported { detail: "args() takes no arguments".into(), span });
            }
            let result = self.builder.fresh_value();
            let ty = IrType::List(Box::new(IrType::Str));
            self.builder.push_instr(IrInstr::ProcessArgs { result }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: env_var(name) → EnvVar → option<str>
        if callee.name == "env_var" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported { detail: "env_var() requires 1 argument".into(), span });
            }
            let (name, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Option(Box::new(IrType::Str));
            self.builder.push_instr(IrInstr::EnvVar { result, name }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: panic(msg) → Panic (terminator; does not return)
        if callee.name == "panic" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "panic() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (msg, _) = self.lower_expr(&args[0])?;
            self.builder.push_instr(IrInstr::Panic { msg }, None);
            // Return a dummy value so the type-checker is happy.
            let dummy = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::ConstInt { result: dummy, value: 0, ty: ty.clone() },
                Some(ty.clone()),
            );
            return Ok((dummy, ty));
        }

        // Built-in: assert(cond) — lowers to: if cond { continue } else { panic("assertion failed") }
        if callee.name == "assert" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "assert() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (cond, _) = self.lower_expr(&args[0])?;
            let then_block  = self.builder.create_block(Some("assert_ok"));
            let panic_block = self.builder.create_block(Some("assert_fail"));
            let merge_block = self.builder.create_block(Some("assert_merge"));
            // CondBr: if cond → then_block, else → panic_block
            self.builder.push_instr(
                IrInstr::CondBr {
                    cond,
                    then_block,
                    then_args: vec![],
                    else_block: panic_block,
                    else_args: vec![],
                },
                None,
            );
            // panic_block: emit panic message + unreachable return (ValidatePass needs a terminator)
            self.builder.set_current_block(panic_block);
            let msg_val = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ConstStr { result: msg_val, value: "assertion failed".into() },
                Some(IrType::Str),
            );
            self.builder.push_instr(IrInstr::Panic { msg: msg_val }, None);
            self.builder.push_instr(IrInstr::Return { values: vec![] }, None);
            // then_block: jump to merge
            self.builder.set_current_block(then_block);
            self.builder.push_instr(
                IrInstr::Br { target: merge_block, args: vec![] },
                None,
            );
            // merge_block: continue with dummy zero
            self.builder.set_current_block(merge_block);
            let dummy = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::ConstInt { result: dummy, value: 0, ty: ty.clone() },
                Some(ty.clone()),
            );
            return Ok((dummy, ty));
        }

        // Built-in: grad(v) → MakeGrad(value=v, tangent=1.0)
        if callee.name == "grad" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "grad() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, inner_ty) = self.lower_expr(&args[0])?;
            let result_ty = IrType::Grad(Box::new(inner_ty));
            // tangent = 1.0 (seeding the derivative)
            let one = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ConstFloat { result: one, value: 1.0, ty: IrType::Scalar(DType::F64) },
                Some(IrType::Scalar(DType::F64)),
            );
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::MakeGrad { result, value: val, tangent: one, ty: result_ty.clone() },
                Some(result_ty.clone()),
            );
            return Ok((result, result_ty));
        }

        // Built-in: grad_of(closure, x) → numerical derivative via central finite differences
        // Returns (f(x+h) - f(x-h)) / (2*h)  where h = 1e-7
        if callee.name == "grad_of" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "grad_of() requires exactly 2 arguments: grad_of(closure, x)".into(),
                    span,
                });
            }
            let (closure_val, _closure_ty) = self.lower_expr(&args[0])?;
            let (x_val, x_ty) = self.lower_expr(&args[1])?;
            // Use x's type for all arithmetic so types stay consistent
            // h = 1e-3 (step for central finite difference; large enough for f32 precision)
            let h_val = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ConstFloat { result: h_val, value: 1e-3, ty: x_ty.clone() },
                Some(x_ty.clone()),
            );
            // x_plus = x + h
            let x_plus = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BinOp { result: x_plus, op: BinOp::Add, lhs: x_val, rhs: h_val, ty: x_ty.clone() },
                Some(x_ty.clone()),
            );
            // x_minus = x - h
            let x_minus = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BinOp { result: x_minus, op: BinOp::Sub, lhs: x_val, rhs: h_val, ty: x_ty.clone() },
                Some(x_ty.clone()),
            );
            // f_plus = closure(x_plus)
            let f_plus = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::CallClosure { result: Some(f_plus), closure: closure_val, args: vec![x_plus], result_ty: x_ty.clone() },
                Some(x_ty.clone()),
            );
            // f_minus = closure(x_minus)
            let f_minus = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::CallClosure { result: Some(f_minus), closure: closure_val, args: vec![x_minus], result_ty: x_ty.clone() },
                Some(x_ty.clone()),
            );
            // diff = f_plus - f_minus
            let diff = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BinOp { result: diff, op: BinOp::Sub, lhs: f_plus, rhs: f_minus, ty: x_ty.clone() },
                Some(x_ty.clone()),
            );
            // two_h = 2.0 * h
            let two = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ConstFloat { result: two, value: 2.0, ty: x_ty.clone() },
                Some(x_ty.clone()),
            );
            let two_h = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BinOp { result: two_h, op: BinOp::Mul, lhs: two, rhs: h_val, ty: x_ty.clone() },
                Some(x_ty.clone()),
            );
            // result = diff / two_h
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BinOp { result, op: BinOp::Div, lhs: diff, rhs: two_h, ty: x_ty.clone() },
                Some(x_ty.clone()),
            );
            return Ok((result, x_ty));
        }

        // Built-in: sparsify(arr) → Sparsify (convert dense array to sparse representation)
        if callee.name == "sparsify" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "sparsify() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, inner_ty) = self.lower_expr(&args[0])?;
            let result_ty = IrType::Sparse(Box::new(inner_ty));
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::Sparsify { result, operand: val, ty: result_ty.clone() },
                Some(result_ty.clone()),
            );
            return Ok((result, result_ty));
        }

        // Built-in: densify(sparse) → Densify (convert sparse back; returns nnz count as i64)
        if callee.name == "densify" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "densify() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, _) = self.lower_expr(&args[0])?;
            let result_ty = IrType::Scalar(DType::I64);
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::Densify { result, operand: val, ty: result_ty.clone() },
                Some(result_ty.clone()),
            );
            return Ok((result, result_ty));
        }

        // Built-in string predicates: contains(s, sub), starts_with(s, p), ends_with(s, p)
        {
            let str_pred: Option<fn(ValueId, ValueId, ValueId) -> IrInstr> = match callee.name.as_str() {
                "contains"    => Some(|result, haystack, needle| IrInstr::StrContains { result, haystack, needle }),
                "starts_with" => Some(|result, haystack, prefix| IrInstr::StrStartsWith { result, haystack, prefix }),
                "ends_with"   => Some(|result, haystack, suffix| IrInstr::StrEndsWith { result, haystack, suffix }),
                _ => None,
            };
            if let Some(mk) = str_pred {
                if args.len() != 2 {
                    return Err(LowerError::Unsupported {
                        detail: format!("{}() requires exactly 2 arguments", callee.name),
                        span,
                    });
                }
                let (haystack, _) = self.lower_expr(&args[0])?;
                let (second, _) = self.lower_expr(&args[1])?;
                let result = self.builder.fresh_value();
                let ret_ty = IrType::Scalar(DType::Bool);
                self.builder.push_instr(mk(result, haystack, second), Some(ret_ty.clone()));
                return Ok((result, ret_ty));
            }
        }

        // Built-in string transforms: to_upper(s), to_lower(s), trim(s)
        {
            let str_xform: Option<fn(ValueId, ValueId) -> IrInstr> = match callee.name.as_str() {
                "to_upper" => Some(|result, operand| IrInstr::StrToUpper { result, operand }),
                "to_lower" => Some(|result, operand| IrInstr::StrToLower { result, operand }),
                "trim"     => Some(|result, operand| IrInstr::StrTrim { result, operand }),
                _ => None,
            };
            if let Some(mk) = str_xform {
                if args.len() != 1 {
                    return Err(LowerError::Unsupported {
                        detail: format!("{}() requires exactly 1 argument", callee.name),
                        span,
                    });
                }
                let (operand, _) = self.lower_expr(&args[0])?;
                let result = self.builder.fresh_value();
                let ret_ty = IrType::Str;
                self.builder.push_instr(mk(result, operand), Some(ret_ty.clone()));
                return Ok((result, ret_ty));
            }
        }

        // Built-in: repeat(s, n) → StrRepeat
        if callee.name == "repeat" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "repeat() requires exactly 2 arguments".into(),
                    span,
                });
            }
            let (operand, _) = self.lower_expr(&args[0])?;
            let (count, _) = self.lower_expr(&args[1])?;
            let result = self.builder.fresh_value();
            let ret_ty = IrType::Str;
            self.builder.push_instr(
                IrInstr::StrRepeat { result, operand, count },
                Some(ret_ty.clone()),
            );
            return Ok((result, ret_ty));
        }

        // Built-in bitwise binary: band(a,b), bor(a,b), bxor(a,b), shl(a,b), shr(a,b)
        {
            let bitbin: Option<BinOp> = match callee.name.as_str() {
                "band" => Some(BinOp::BitAnd),
                "bor"  => Some(BinOp::BitOr),
                "bxor" => Some(BinOp::BitXor),
                "shl"  => Some(BinOp::Shl),
                "shr"  => Some(BinOp::Shr),
                _ => None,
            };
            if let Some(op) = bitbin {
                if args.len() != 2 {
                    return Err(LowerError::Unsupported {
                        detail: format!("{}() requires exactly 2 arguments", callee.name),
                        span,
                    });
                }
                let (lhs, lhs_ty) = self.lower_expr(&args[0])?;
                let (rhs, _) = self.lower_expr(&args[1])?;
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::BinOp { result, op, lhs, rhs, ty: lhs_ty.clone() },
                    Some(lhs_ty.clone()),
                );
                return Ok((result, lhs_ty));
            }
        }

        // Built-in bitwise unary: bitnot(x)
        if callee.name == "bitnot" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "bitnot() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (operand, op_ty) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::UnaryOp { result, op: ScalarUnaryOp::BitNot, operand, ty: op_ty.clone() },
                Some(op_ty.clone()),
            );
            return Ok((result, op_ty));
        }

        // Built-in math unary: sqrt, abs, floor, ceil, sin, cos, tan, exp, log, log2, round, sign
        {
            let math_unary: Option<ScalarUnaryOp> = match callee.name.as_str() {
                "sqrt"  => Some(ScalarUnaryOp::Sqrt),
                "abs"   => Some(ScalarUnaryOp::Abs),
                "floor" => Some(ScalarUnaryOp::Floor),
                "ceil"  => Some(ScalarUnaryOp::Ceil),
                "sin"   => Some(ScalarUnaryOp::Sin),
                "cos"   => Some(ScalarUnaryOp::Cos),
                "tan"   => Some(ScalarUnaryOp::Tan),
                "exp"   => Some(ScalarUnaryOp::Exp),
                "log"   => Some(ScalarUnaryOp::Log),
                "log2"  => Some(ScalarUnaryOp::Log2),
                "round" => Some(ScalarUnaryOp::Round),
                "sign"  => Some(ScalarUnaryOp::Sign),
                _ => None,
            };
            if let Some(op) = math_unary {
                if args.len() != 1 {
                    return Err(LowerError::Unsupported {
                        detail: format!("{}() requires exactly 1 argument", callee.name),
                        span,
                    });
                }
                let (operand, op_ty) = self.lower_expr(&args[0])?;
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::UnaryOp { result, op, operand, ty: op_ty.clone() },
                    Some(op_ty.clone()),
                );
                return Ok((result, op_ty));
            }
        }

        // clamp(x, lo, hi) → min(max(x, lo), hi)
        if callee.name == "clamp" {
            if args.len() != 3 {
                return Err(LowerError::Unsupported {
                    detail: "clamp() requires exactly 3 arguments".into(),
                    span,
                });
            }
            let (x,  x_ty) = self.lower_expr(&args[0])?;
            let (lo, _)    = self.lower_expr(&args[1])?;
            let (hi, _)    = self.lower_expr(&args[2])?;
            let inner = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BinOp { result: inner, op: BinOp::Max, lhs: x, rhs: lo, ty: x_ty.clone() },
                Some(x_ty.clone()),
            );
            let outer = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BinOp { result: outer, op: BinOp::Min, lhs: inner, rhs: hi, ty: x_ty.clone() },
                Some(x_ty.clone()),
            );
            return Ok((outer, x_ty));
        }

        // Built-in math binary: pow(base, exp), min(a, b), max(a, b)
        {
            let math_bin: Option<BinOp> = match callee.name.as_str() {
                "pow" => Some(BinOp::Pow),
                "min" => Some(BinOp::Min),
                "max" => Some(BinOp::Max),
                _ => None,
            };
            if let Some(op) = math_bin {
                if args.len() != 2 {
                    return Err(LowerError::Unsupported {
                        detail: format!("{}() requires exactly 2 arguments", callee.name),
                        span,
                    });
                }
                let (lhs, lhs_ty) = self.lower_expr(&args[0])?;
                let (rhs, _) = self.lower_expr(&args[1])?;
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::BinOp { result, op, lhs, rhs, ty: lhs_ty.clone() },
                    Some(lhs_ty.clone()),
                );
                return Ok((result, lhs_ty));
            }
        }

        // Generic function call — monomorphize on demand.
        if let Some(generic_fn) = self.generic_fns.get(&callee.name).cloned() {
            // Lower each argument and collect concrete types.
            let mut arg_vals = Vec::with_capacity(args.len());
            let mut arg_tys = Vec::with_capacity(args.len());
            for arg in args {
                let (v, ty) = self.lower_expr(arg)?;
                arg_vals.push(v);
                arg_tys.push(ty);
            }

            // Build type substitution by matching type_params against arg types.
            let mut subs: HashMap<String, IrType> = HashMap::new();
            for (tp_name, arg_ty) in generic_fn.type_params.iter().zip(arg_tys.iter()) {
                subs.insert(tp_name.clone(), arg_ty.clone());
            }

            // Resolve the concrete return type.
            let resolve = |ty: &AstType| -> IrType {
                if let AstType::Named(n, _) = ty {
                    if let Some(c) = subs.get(n) { return c.clone(); }
                }
                lower_type_with_structs(ty, self.module)
            };
            let concrete_ret = resolve(&generic_fn.return_ty);

            // Generate mangled name: e.g. `max_val__i64` for T=i64.
            let mangle = subs.iter()
                .map(|(_, ty)| format!("{}", ty).replace(['<', '>', ',', ' '], "_"))
                .collect::<Vec<_>>()
                .join("_");
            let mangled = format!("{}__{}", callee.name, mangle);

            // Register the return type for the mangled name.
            self.mono_sigs.borrow_mut().insert(mangled.clone(), concrete_ret.clone());

            // Monomorphize if not already done.
            if !self.mono_cache.borrow().contains(&mangled) {
                self.mono_cache.borrow_mut().insert(mangled.clone());

                // Build a renamed copy of the generic function.
                let mut mono_fn = generic_fn.clone();
                mono_fn.name.name = mangled.clone();
                mono_fn.type_params = Vec::new(); // no longer generic

                // Lower the specialized function.
                let fn_sigs_ref = self.fn_sigs;
                let (ir_func, extra_lifted) = lower_function_with_generics_and_subs(
                    &mono_fn, self.module, fn_sigs_ref,
                    &self.const_defs,
                    self.generic_fns.clone(),
                    self.mono_cache.clone(),
                    self.mono_sigs.clone(),
                    subs,
                    self.trait_dispatch.clone(),
                    self.fn_defaults.clone(),
                ).map_err(|e| e)?;

                self.lifted_fns.borrow_mut().push(ir_func);
                self.lifted_fns.borrow_mut().extend(extra_lifted);
            }

            // Emit the call to the specialized function.
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::Call {
                    result: Some(result),
                    callee: mangled,
                    args: arg_vals,
                    result_ty: Some(concrete_ret.clone()),
                },
                Some(concrete_ret.clone()),
            );
            return Ok((result, concrete_ret));
        }

        // Trait method dispatch — static dispatch based on first arg's concrete type.
        if let Some(impls) = self.trait_dispatch.get(&callee.name).cloned() {
            if !args.is_empty() {
                let (first_val, first_ty) = self.lower_expr(&args[0])?;
                let type_key = ir_type_dispatch_name(&first_ty);
                if let Some((_, mangled)) = impls.iter().find(|(dispatch_ty, _)| {
                    ir_type_dispatch_name(dispatch_ty) == type_key
                }) {
                    let mangled = mangled.clone();
                    let ret_ty = self.fn_sigs.get(&mangled)
                        .cloned()
                        .unwrap_or(IrType::Infer);
                    let mut arg_vals = vec![first_val];
                    for arg in &args[1..] {
                        let (v, _) = self.lower_expr(arg)?;
                        arg_vals.push(v);
                    }
                    let result = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::Call {
                            result: Some(result),
                            callee: mangled,
                            args: arg_vals,
                            result_ty: Some(ret_ty.clone()),
                        },
                        Some(ret_ty.clone()),
                    );
                    return Ok((result, ret_ty));
                }
            }
        }

        // General function call — look up the callee's return type from
        // pre-collected signatures so the result has a concrete type.
        let ret_ty = self
            .fn_sigs
            .get(&callee.name)
            .cloned()
            .or_else(|| self.mono_sigs.borrow().get(&callee.name).cloned())
            .unwrap_or(IrType::Infer);

        // Build argument list, filling in defaults for omitted trailing args.
        let defaults = self.fn_defaults.get(&callee.name).cloned();
        let mut arg_vals = Vec::with_capacity(args.len());
        for arg in args {
            let (v, _) = self.lower_expr(arg)?;
            arg_vals.push(v);
        }
        if let Some(ref defs) = defaults {
            for def_opt in defs.iter().skip(arg_vals.len()) {
                if let Some(default_expr) = def_opt {
                    let (v, _) = self.lower_expr(default_expr)?;
                    arg_vals.push(v);
                }
            }
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

    /// Lowers `if cond { then_blk } [else { else_blk }]` to SSA control flow.
    ///
    /// **With else**: Creates three blocks (then / else / merge) with a `CondBr`.
    /// Each branch is lowered independently; the merge block receives the result
    /// via a block parameter.
    ///
    /// **Without else**: Creates two blocks (then / merge). The expression always
    /// evaluates to unit (`i64 0`) — the then branch runs for its side effects.
    ///
    /// If a branch terminates early (e.g. via `return`), no `Br` to merge is
    /// emitted for that branch.
    fn lower_if_expr(
        &mut self,
        cond: &AstExpr,
        then_blk: &AstBlock,
        else_blk: Option<&AstBlock>,
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        let _ = span; // span used only for error messages which no longer apply

        // 1. Evaluate condition in the current block.
        let (cond_val, _) = self.lower_expr(cond)?;

        if let Some(else_blk) = else_blk {
            // Full if/else: three-block CFG (then / else / merge).
            let then_bb = self.builder.create_block(Some("then"));
            let else_bb = self.builder.create_block(Some("else"));
            let merge_bb = self.builder.create_block(Some("merge"));

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

            // Lower THEN branch.
            let outer_scope = self.scope.clone();
            self.builder.set_current_block(then_bb);
            let then_result = self.lower_block(then_blk)?;
            if let Some((then_val, _)) = &then_result {
                self.builder.push_instr(
                    IrInstr::Br { target: merge_bb, args: vec![*then_val] },
                    None,
                );
            }
            self.scope = outer_scope.clone();

            // Lower ELSE branch.
            self.builder.set_current_block(else_bb);
            let else_result = self.lower_block(else_blk)?;
            if let Some((else_val, _)) = &else_result {
                self.builder.push_instr(
                    IrInstr::Br { target: merge_bb, args: vec![*else_val] },
                    None,
                );
            }
            self.scope = outer_scope;

            // Merge block parameter type = type of whichever branch produced a value.
            let result_ty = match (&then_result, &else_result) {
                (Some((_, ty)), _) => ty.clone(),
                (_, Some((_, ty))) => ty.clone(),
                (None, None) => IrType::Scalar(DType::I64),
            };

            let result = self
                .builder
                .add_block_param(merge_bb, Some("if_result"), result_ty.clone());
            self.builder.set_current_block(merge_bb);
            Ok((result, result_ty))
        } else {
            // if-without-else: two-block CFG (then / merge).
            // The whole expression evaluates to unit (i64 0).
            let unit_ty = IrType::Scalar(DType::I64);
            let unit_val = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ConstInt { result: unit_val, value: 0, ty: unit_ty.clone() },
                Some(unit_ty.clone()),
            );

            let then_bb = self.builder.create_block(Some("then"));
            let merge_bb = self.builder.create_block(Some("merge"));

            // False branch jumps directly to merge with the unit value.
            self.builder.push_instr(
                IrInstr::CondBr {
                    cond: cond_val,
                    then_block: then_bb,
                    then_args: vec![],
                    else_block: merge_bb,
                    else_args: vec![unit_val],
                },
                None,
            );

            // Lower THEN branch (side effects only; result is discarded).
            let outer_scope = self.scope.clone();
            self.builder.set_current_block(then_bb);
            let _then_result = self.lower_block(then_blk)?;
            if !self.builder.is_current_block_terminated() {
                // Branch didn't return early: jump to merge with unit.
                self.builder.push_instr(
                    IrInstr::Br { target: merge_bb, args: vec![unit_val] },
                    None,
                );
            }
            self.scope = outer_scope;

            let merge_param = self
                .builder
                .add_block_param(merge_bb, Some("if_result"), unit_ty.clone());
            self.builder.set_current_block(merge_bb);
            Ok((merge_param, unit_ty))
        }
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

        // Check if this is an option or result pattern match.
        let is_option_when = arms.iter().any(|a| {
            matches!(a.pattern, AstWhenPattern::OptionSome { .. } | AstWhenPattern::OptionNone)
        });
        let is_result_when = arms.iter().any(|a| {
            matches!(a.pattern, AstWhenPattern::ResultOk { .. } | AstWhenPattern::ResultErr { .. })
        });

        // If any option/result arm has a guard, use chain lowering so that
        // guards are evaluated correctly with bindings in scope.
        let option_has_guards = is_option_when && arms.iter().any(|a| a.guard.is_some());
        let result_has_guards = is_result_when && arms.iter().any(|a| a.guard.is_some());

        if is_option_when && !option_has_guards {
            return self.lower_option_when(scrut_val, &scrut_ty, arms, span);
        }
        if is_result_when && !result_has_guards {
            return self.lower_result_when(scrut_val, &scrut_ty, arms, span);
        }

        // Check if any arm has guards or non-enum patterns (wildcard, literal).
        // If so, lower as an if-else chain instead of SwitchVariant.
        let needs_chain = arms.iter().any(|a| {
            a.guard.is_some()
                || matches!(
                    a.pattern,
                    AstWhenPattern::Wildcard
                        | AstWhenPattern::IntLit(_)
                        | AstWhenPattern::BoolLit(_)
                        | AstWhenPattern::StringLit(_)
                )
        });
        // Also use chain when the scrutinee is not an enum (e.g. matching i64/bool/str).
        let is_enum_scrut = matches!(&scrut_ty, IrType::Enum { .. });
        if needs_chain || !is_enum_scrut {
            return self.lower_when_as_chain(scrut_val, &scrut_ty, arms, span);
        }

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
        // Get the variant field types for this enum so we can emit ExtractVariantField.
        let variant_field_types: Vec<Vec<IrType>> = self.module
            .enum_variant_fields(&enum_name)
            .cloned()
            .unwrap_or_default();

        let outer_scope = self.scope.clone();
        let mut result_ty: Option<IrType> = None;
        for (arm, &arm_bb) in arms.iter().zip(arm_blocks.iter()) {
            self.scope = outer_scope.clone();
            self.builder.set_current_block(arm_bb);

            // Emit ExtractVariantField instructions for pattern bindings.
            if let AstWhenPattern::EnumVariant { variant_name, bindings, .. } = &arm.pattern {
                if !bindings.is_empty() {
                    // Find the variant index for field type lookup.
                    let vidx = variants.iter().position(|v| v == variant_name);
                    let empty_fields: Vec<IrType> = Vec::new();
                    let field_types: &Vec<IrType> = vidx
                        .and_then(|i| variant_field_types.get(i))
                        .unwrap_or(&empty_fields);
                    let variant_idx = vidx.unwrap_or(0);
                    for (field_idx, binding_name) in bindings.iter().enumerate() {
                        let field_ty = field_types.get(field_idx).cloned().unwrap_or(IrType::Infer);
                        let result = self.builder.fresh_value();
                        self.builder.push_instr(
                            IrInstr::ExtractVariantField {
                                result,
                                operand: scrut_val,
                                variant_idx,
                                field_idx,
                                result_ty: field_ty.clone(),
                            },
                            Some(field_ty.clone()),
                        );
                        self.scope.insert(binding_name.clone(), (result, field_ty));
                    }
                }
            }

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

    /// Lowers `when opt_val { some(x) => body, none => body }` for option types.
    fn lower_option_when(
        &mut self,
        scrut_val: ValueId,
        scrut_ty: &IrType,
        arms: &[AstWhenArm],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        // Extract inner type from option type.
        let inner_ty = if let IrType::Option(inner) = scrut_ty {
            (**inner).clone()
        } else {
            IrType::Infer
        };
        // Find the some and none arms.
        let some_arm = arms.iter().find(|a| matches!(a.pattern, AstWhenPattern::OptionSome { .. }));
        let none_arm = arms.iter().find(|a| matches!(a.pattern, AstWhenPattern::OptionNone));

        if some_arm.is_none() && none_arm.is_none() {
            return Err(LowerError::Unsupported {
                detail: "option when expression needs some/none arms".into(),
                span,
            });
        }

        // Emit IsSome test.
        let is_some_result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::IsSome { result: is_some_result, operand: scrut_val },
            Some(IrType::Scalar(DType::Bool)),
        );

        let some_bb = self.builder.create_block(Some("option_some"));
        let none_bb = self.builder.create_block(Some("option_none"));
        let merge_bb = self.builder.create_block(Some("option_merge"));

        let outer_scope = self.scope.clone();
        let unit_ty = IrType::Scalar(DType::I64);

        // When only one arm is present the whole expression evaluates to unit (i64 0).
        // When both arms are present the result type comes from the arm bodies.
        let partial = some_arm.is_none() || none_arm.is_none();

        // Pre-compute a unit value in the current (pre-branch) block BEFORE the CondBr
        // terminates the block, so it's accessible from both successor arms.
        let unit_val = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt { result: unit_val, value: 0, ty: unit_ty.clone() },
            Some(unit_ty.clone()),
        );

        self.builder.push_instr(
            IrInstr::CondBr {
                cond: is_some_result,
                then_block: some_bb,
                then_args: vec![],
                else_block: none_bb,
                else_args: vec![],
            },
            None,
        );

        // Some branch.
        self.builder.set_current_block(some_bb);
        self.scope = outer_scope.clone();
        let (some_val, mut result_ty): (ValueId, Option<IrType>) = if let Some(arm) = some_arm {
            // Bind the inner value if a name was given.
            if let AstWhenPattern::OptionSome { binding: Some(ref bind_name) } = arm.pattern {
                // Unwrap the option to get the inner value.
                let unwrapped = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::OptionUnwrap { result: unwrapped, operand: scrut_val, result_ty: inner_ty.clone() },
                    Some(inner_ty.clone()),
                );
                self.scope.insert(bind_name.clone(), (unwrapped, inner_ty.clone()));
            }
            let (v, ty) = self.lower_expr(&arm.body)?;
            if partial { (unit_val, Some(unit_ty.clone())) } else { (v, Some(ty)) }
        } else {
            (unit_val, Some(unit_ty.clone()))
        };
        self.builder.push_instr(IrInstr::Br { target: merge_bb, args: vec![some_val] }, None);

        // None branch.
        self.builder.set_current_block(none_bb);
        self.scope = outer_scope.clone();
        let none_val = if let Some(arm) = none_arm {
            let (v, ty) = self.lower_expr(&arm.body)?;
            if result_ty.is_none() { result_ty = Some(ty.clone()); }
            if partial { unit_val } else { v }
        } else {
            unit_val
        };
        self.builder.push_instr(IrInstr::Br { target: merge_bb, args: vec![none_val] }, None);

        self.scope = outer_scope;
        let result_ty = result_ty.unwrap_or(unit_ty);
        let result = self.builder.add_block_param(merge_bb, Some("option_result"), result_ty.clone());
        self.builder.set_current_block(merge_bb);
        Ok((result, result_ty))
    }

    /// Lowers `when res_val { ok(x) => body, err(e) => body }` for result types.
    fn lower_result_when(
        &mut self,
        scrut_val: ValueId,
        scrut_ty: &IrType,
        arms: &[AstWhenArm],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        let (ok_inner_ty, err_inner_ty) = if let IrType::ResultType(ok, err) = scrut_ty {
            ((**ok).clone(), (**err).clone())
        } else {
            (IrType::Infer, IrType::Infer)
        };
        let ok_arm = arms.iter().find(|a| matches!(a.pattern, AstWhenPattern::ResultOk { .. }));
        let err_arm = arms.iter().find(|a| matches!(a.pattern, AstWhenPattern::ResultErr { .. }));

        if ok_arm.is_none() && err_arm.is_none() {
            return Err(LowerError::Unsupported {
                detail: "result when expression needs ok/err arms".into(),
                span,
            });
        }

        // Emit IsOk test.
        let is_ok_result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::IsOk { result: is_ok_result, operand: scrut_val },
            Some(IrType::Scalar(DType::Bool)),
        );

        let ok_bb = self.builder.create_block(Some("result_ok"));
        let err_bb = self.builder.create_block(Some("result_err"));
        let merge_bb = self.builder.create_block(Some("result_merge"));

        let outer_scope = self.scope.clone();
        let unit_ty = IrType::Scalar(DType::I64);

        // When only one arm is present the whole expression evaluates to unit (i64 0).
        let partial = ok_arm.is_none() || err_arm.is_none();

        // Pre-compute unit value BEFORE the CondBr terminates the block,
        // so it's accessible from both successor arms.
        let unit_val = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt { result: unit_val, value: 0, ty: unit_ty.clone() },
            Some(unit_ty.clone()),
        );

        self.builder.push_instr(
            IrInstr::CondBr {
                cond: is_ok_result,
                then_block: ok_bb,
                then_args: vec![],
                else_block: err_bb,
                else_args: vec![],
            },
            None,
        );

        // Ok branch.
        self.builder.set_current_block(ok_bb);
        self.scope = outer_scope.clone();
        let (ok_val, mut result_ty): (ValueId, Option<IrType>) = if let Some(arm) = ok_arm {
            if let AstWhenPattern::ResultOk { binding: Some(ref bind_name) } = arm.pattern {
                let unwrapped = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ResultUnwrap { result: unwrapped, operand: scrut_val, result_ty: ok_inner_ty.clone() },
                    Some(ok_inner_ty.clone()),
                );
                self.scope.insert(bind_name.clone(), (unwrapped, ok_inner_ty.clone()));
            }
            let (v, ty) = self.lower_expr(&arm.body)?;
            if partial { (unit_val, Some(unit_ty.clone())) } else { (v, Some(ty)) }
        } else {
            (unit_val, Some(unit_ty.clone()))
        };
        self.builder.push_instr(IrInstr::Br { target: merge_bb, args: vec![ok_val] }, None);

        // Err branch.
        self.builder.set_current_block(err_bb);
        self.scope = outer_scope.clone();
        let err_val = if let Some(arm) = err_arm {
            if let AstWhenPattern::ResultErr { binding: Some(ref bind_name) } = arm.pattern {
                let unwrapped = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ResultUnwrapErr { result: unwrapped, operand: scrut_val, result_ty: err_inner_ty.clone() },
                    Some(err_inner_ty.clone()),
                );
                self.scope.insert(bind_name.clone(), (unwrapped, err_inner_ty.clone()));
            }
            let (v, ty) = self.lower_expr(&arm.body)?;
            if result_ty.is_none() { result_ty = Some(ty.clone()); }
            if partial { unit_val } else { v }
        } else {
            unit_val
        };
        self.builder.push_instr(IrInstr::Br { target: merge_bb, args: vec![err_val] }, None);

        self.scope = outer_scope;
        let result_ty = result_ty.unwrap_or(unit_ty);
        let result = self.builder.add_block_param(merge_bb, Some("result_result"), result_ty.clone());
        self.builder.set_current_block(merge_bb);
        Ok((result, result_ty))
    }

    /// Lowers a `when` expression as an if-else chain.
    ///
    /// Used for:
    /// - Arms with guards (`pattern if cond =>`)
    /// - Wildcard patterns (`_`)
    /// - Literal patterns (integer, bool, string)
    /// - Enum patterns when guards or wildcards are mixed in
    fn lower_when_as_chain(
        &mut self,
        scrut_val: ValueId,
        scrut_ty: &IrType,
        arms: &[AstWhenArm],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        // Create a merge block that all arms jump to with their result value.
        let merge_bb = self.builder.create_block(Some("when_merge"));

        // Pre-allocate: we'll build the chain from first arm to last.
        // We need a "no-match" fallback block (panic or unreachable) for non-exhaustive matches.
        // But we emit a runtime panic for safety.
        let no_match_bb = self.builder.create_block(Some("when_no_match"));

        let outer_scope = self.scope.clone();
        let mut result_ty: Option<IrType> = None;

        // Extract enum variant info if scrutinee is an enum.
        let (enum_name_opt, enum_variants_opt): (Option<String>, Option<Vec<String>>) =
            if let IrType::Enum { name, variants } = scrut_ty {
                (Some(name.clone()), Some(variants.clone()))
            } else {
                (None, None)
            };
        let enum_variant_field_types: Vec<Vec<IrType>> = if let Some(ref ename) = enum_name_opt {
            self.module.enum_variant_fields(ename).cloned().unwrap_or_default()
        } else {
            Vec::new()
        };

        // We chain arms: for each arm, emit:
        //   current_bb: cond = (pattern_matches && guard?)
        //               condBr cond -> arm_body_bb, next_check_bb
        //   arm_body_bb: bind vars, lower body, br merge_bb
        //   next_check_bb: (next iteration's current_bb)
        let mut current_check_bb = self.builder.current_block();

        for (arm_idx, arm) in arms.iter().enumerate() {
            let is_last = arm_idx == arms.len() - 1;
            let has_guard_with_bindings = arm.guard.is_some() && pattern_has_bindings(&arm.pattern);

            // IMPORTANT: create bind_guard_bb BEFORE arm_body_bb so the validator's
            // linear block scan sees value definitions before their uses.
            let bind_guard_bb_opt = if has_guard_with_bindings {
                Some(self.builder.create_block(Some(&format!("when_guard_{}", arm_idx))))
            } else {
                None
            };

            // Create the arm body block.
            let arm_body_bb = self.builder.create_block(Some(&format!("when_arm_{}", arm_idx)));
            // Create the next-check block (reuse no_match_bb for last arm).
            let next_check_bb = if is_last {
                no_match_bb
            } else {
                self.builder.create_block(Some(&format!("when_check_{}", arm_idx + 1)))
            };

            // Emit the pattern match condition into current_check_bb.
            self.builder.set_current_block(current_check_bb);
            self.scope = outer_scope.clone();

            // Compute the pattern condition (tag check only, no extraction).
            let pat_cond = self.emit_pattern_condition(
                scrut_val,
                scrut_ty,
                &arm.pattern,
                &enum_name_opt,
                &enum_variants_opt,
                span,
            )?;

            // If guard is present AND pattern has extractable bindings, use a 3-block approach:
            //   check_bb → (pat matches?) → bind_guard_bb → (guard?) → arm_body_bb
            // This ensures bindings are available to the guard expression.
            let body_scope = if let Some(bind_guard_bb) = bind_guard_bb_opt {
                // In check_bb: branch on pattern condition.
                self.builder.push_instr(
                    IrInstr::CondBr {
                        cond: pat_cond,
                        then_block: bind_guard_bb,
                        then_args: vec![],
                        else_block: next_check_bb,
                        else_args: vec![],
                    },
                    None,
                );

                // In bind_guard_bb: emit bindings, evaluate guard.
                self.builder.set_current_block(bind_guard_bb);
                self.scope = outer_scope.clone();
                self.bind_pattern_vars(
                    scrut_val, scrut_ty, &arm.pattern,
                    &enum_variants_opt, &enum_variant_field_types,
                )?;
                let guard_expr = arm.guard.as_ref().unwrap();
                let (guard_val, _) = self.lower_expr(guard_expr)?;
                self.builder.push_instr(
                    IrInstr::CondBr {
                        cond: guard_val,
                        then_block: arm_body_bb,
                        then_args: vec![],
                        else_block: next_check_bb,
                        else_args: vec![],
                    },
                    None,
                );

                // Body scope carries the bindings from bind_guard_bb.
                self.scope.clone()
            } else {
                // Simple case: combine pat_cond + optional guard in single block.
                let final_cond = if let Some(ref guard_expr) = arm.guard {
                    let (guard_val, _) = self.lower_expr(guard_expr)?;
                    let and_result = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::BinOp {
                            result: and_result,
                            op: BinOp::BitAnd,
                            lhs: pat_cond,
                            rhs: guard_val,
                            ty: IrType::Scalar(DType::Bool),
                        },
                        Some(IrType::Scalar(DType::Bool)),
                    );
                    and_result
                } else {
                    pat_cond
                };

                self.builder.push_instr(
                    IrInstr::CondBr {
                        cond: final_cond,
                        then_block: arm_body_bb,
                        then_args: vec![],
                        else_block: next_check_bb,
                        else_args: vec![],
                    },
                    None,
                );

                outer_scope.clone()
            };

            // Emit arm body block.
            self.builder.set_current_block(arm_body_bb);
            self.scope = body_scope;

            // Bind pattern variables (no-op if already bound by bind_guard_bb path).
            if arm.guard.is_none() || !pattern_has_bindings(&arm.pattern) {
                self.bind_pattern_vars(
                    scrut_val, scrut_ty, &arm.pattern,
                    &enum_variants_opt, &enum_variant_field_types,
                )?;
            }

            let (arm_val, arm_ty) = self.lower_expr(&arm.body)?;
            if result_ty.is_none() {
                result_ty = Some(arm_ty);
            }
            self.builder.push_instr(
                IrInstr::Br { target: merge_bb, args: vec![arm_val] },
                None,
            );

            current_check_bb = next_check_bb;
        }

        // Emit the no-match block (runtime panic).
        self.builder.set_current_block(no_match_bb);
        self.scope = outer_scope.clone();
        let panic_msg = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstStr { result: panic_msg, value: "when: no pattern matched".to_string() },
            Some(IrType::Str),
        );
        self.builder.push_instr(IrInstr::Panic { msg: panic_msg }, None);
        // Panic is not a terminator; we still need a Return to satisfy IR structure.
        let unit_val = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt { result: unit_val, value: 0, ty: IrType::Scalar(DType::I64) },
            Some(IrType::Scalar(DType::I64)),
        );
        let rty = result_ty.clone().unwrap_or(IrType::Scalar(DType::I64));
        self.builder.push_instr(
            IrInstr::Br { target: merge_bb, args: vec![unit_val] },
            None,
        );

        self.scope = outer_scope;
        let result_ty = result_ty.unwrap_or(IrType::Scalar(DType::I64));
        // Sanity check: rty is used to suppress unused variable warning.
        let _ = rty;
        let result = self.builder.add_block_param(merge_bb, Some("when_result"), result_ty.clone());
        self.builder.set_current_block(merge_bb);
        Ok((result, result_ty))
    }

    /// Emits instructions computing a bool condition for whether `scrut_val` matches `pattern`.
    fn emit_pattern_condition(
        &mut self,
        scrut_val: ValueId,
        scrut_ty: &IrType,
        pattern: &AstWhenPattern,
        enum_name_opt: &Option<String>,
        enum_variants_opt: &Option<Vec<String>>,
        span: Span,
    ) -> Result<ValueId, LowerError> {
        match pattern {
            AstWhenPattern::Wildcard => {
                // Always matches: emit `true`.
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstBool { result, value: true },
                    Some(IrType::Scalar(DType::Bool)),
                );
                Ok(result)
            }
            AstWhenPattern::IntLit(n) => {
                // scrutinee == n
                let lit_val = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstInt { result: lit_val, value: *n, ty: scrut_ty.clone() },
                    Some(scrut_ty.clone()),
                );
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::BinOp {
                        result,
                        op: BinOp::CmpEq,
                        lhs: scrut_val,
                        rhs: lit_val,
                        ty: scrut_ty.clone(),
                    },
                    Some(IrType::Scalar(DType::Bool)),
                );
                Ok(result)
            }
            AstWhenPattern::BoolLit(b) => {
                // scrutinee == b
                let lit_val = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstBool { result: lit_val, value: *b },
                    Some(IrType::Scalar(DType::Bool)),
                );
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::BinOp {
                        result,
                        op: BinOp::CmpEq,
                        lhs: scrut_val,
                        rhs: lit_val,
                        ty: IrType::Scalar(DType::Bool),
                    },
                    Some(IrType::Scalar(DType::Bool)),
                );
                Ok(result)
            }
            AstWhenPattern::StringLit(s) => {
                // StrEq(scrutinee, s)
                let str_val = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstStr { result: str_val, value: s.clone() },
                    Some(IrType::Str),
                );
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::StrEq { result, lhs: scrut_val, rhs: str_val },
                    Some(IrType::Scalar(DType::Bool)),
                );
                Ok(result)
            }
            AstWhenPattern::Tuple(subs) => {
                // For a tuple pattern (a, b, ...): extract each element and check each sub-pattern.
                // Pure bindings (EnumVariant with empty enum_name) always succeed.
                // Literal sub-patterns (IntLit, BoolLit, StringLit) emit a check.
                // All checks are AND-ed together.
                let bool_ty = IrType::Scalar(DType::Bool);
                let tuple_elems = match scrut_ty {
                    IrType::Tuple(ref elems) => elems.clone(),
                    _ => return Err(LowerError::Unsupported {
                        detail: format!("tuple pattern on non-tuple type {}", scrut_ty),
                        span,
                    }),
                };
                let mut all_ok = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstBool { result: all_ok, value: true },
                    Some(bool_ty.clone()),
                );
                for (i, sub) in subs.iter().enumerate() {
                    let elem_ty = tuple_elems.get(i).cloned().unwrap_or(IrType::Scalar(DType::I64));
                    let elem_val = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::GetElement { result: elem_val, base: scrut_val, index: i, result_ty: elem_ty.clone() },
                        Some(elem_ty.clone()),
                    );
                    // Check sub-pattern
                    let sub_ok = match sub {
                        // Binding or wildcard: always true
                        AstWhenPattern::EnumVariant { enum_name, variant_name, .. } if enum_name.is_empty() => {
                            // Bind this element under variant_name (handled separately in bindings)
                            let _ = variant_name;
                            let t = self.builder.fresh_value();
                            self.builder.push_instr(IrInstr::ConstBool { result: t, value: true }, Some(bool_ty.clone()));
                            t
                        }
                        AstWhenPattern::Wildcard => {
                            let t = self.builder.fresh_value();
                            self.builder.push_instr(IrInstr::ConstBool { result: t, value: true }, Some(bool_ty.clone()));
                            t
                        }
                        other => self.emit_pattern_condition(elem_val, &elem_ty, other, &None, &None, span)?,
                    };
                    let new_all = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::BinOp { result: new_all, op: BinOp::BitAnd, lhs: all_ok, rhs: sub_ok, ty: bool_ty.clone() },
                        Some(bool_ty.clone()),
                    );
                    all_ok = new_all;
                }
                Ok(all_ok)
            }
            AstWhenPattern::OptionNone => {
                // !IsSome(scrutinee)
                let is_some_result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::IsSome { result: is_some_result, operand: scrut_val },
                    Some(IrType::Scalar(DType::Bool)),
                );
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::UnaryOp {
                        result,
                        op: ScalarUnaryOp::Not,
                        operand: is_some_result,
                        ty: IrType::Scalar(DType::Bool),
                    },
                    Some(IrType::Scalar(DType::Bool)),
                );
                Ok(result)
            }
            AstWhenPattern::OptionSome { .. } => {
                // IsSome(scrutinee)
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::IsSome { result, operand: scrut_val },
                    Some(IrType::Scalar(DType::Bool)),
                );
                Ok(result)
            }
            AstWhenPattern::ResultOk { .. } => {
                // IsOk(scrutinee)
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::IsOk { result, operand: scrut_val },
                    Some(IrType::Scalar(DType::Bool)),
                );
                Ok(result)
            }
            AstWhenPattern::ResultErr { .. } => {
                // !IsOk(scrutinee)
                let is_ok_result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::IsOk { result: is_ok_result, operand: scrut_val },
                    Some(IrType::Scalar(DType::Bool)),
                );
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::UnaryOp {
                        result,
                        op: ScalarUnaryOp::Not,
                        operand: is_ok_result,
                        ty: IrType::Scalar(DType::Bool),
                    },
                    Some(IrType::Scalar(DType::Bool)),
                );
                Ok(result)
            }
            AstWhenPattern::EnumVariant { variant_name, .. } => {
                // GetVariantTag(scrutinee) == variant_idx_const
                let variants = enum_variants_opt.as_ref().ok_or_else(|| LowerError::Unsupported {
                    detail: "EnumVariant pattern used with non-enum scrutinee".into(),
                    span,
                })?;
                let variant_idx = variants
                    .iter()
                    .position(|v| v == variant_name)
                    .ok_or_else(|| LowerError::Unsupported {
                        detail: format!(
                            "no variant '{}' in enum '{}'",
                            variant_name,
                            enum_name_opt.as_deref().unwrap_or("?")
                        ),
                        span,
                    })?;
                let tag_val = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::GetVariantTag { result: tag_val, operand: scrut_val },
                    Some(IrType::Scalar(DType::I64)),
                );
                let idx_val = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstInt {
                        result: idx_val,
                        value: variant_idx as i64,
                        ty: IrType::Scalar(DType::I64),
                    },
                    Some(IrType::Scalar(DType::I64)),
                );
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::BinOp {
                        result,
                        op: BinOp::CmpEq,
                        lhs: tag_val,
                        rhs: idx_val,
                        ty: IrType::Scalar(DType::I64),
                    },
                    Some(IrType::Scalar(DType::Bool)),
                );
                Ok(result)
            }
            AstWhenPattern::Range { lo, hi } => {
                // lo <= scrutinee && scrutinee <= hi
                let lo_val = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstInt { result: lo_val, value: *lo, ty: scrut_ty.clone() },
                    Some(scrut_ty.clone()),
                );
                let hi_val = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstInt { result: hi_val, value: *hi, ty: scrut_ty.clone() },
                    Some(scrut_ty.clone()),
                );
                let lo_ok = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::BinOp { result: lo_ok, op: BinOp::CmpLe, lhs: lo_val, rhs: scrut_val, ty: scrut_ty.clone() },
                    Some(IrType::Scalar(DType::Bool)),
                );
                let hi_ok = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::BinOp { result: hi_ok, op: BinOp::CmpLe, lhs: scrut_val, rhs: hi_val, ty: scrut_ty.clone() },
                    Some(IrType::Scalar(DType::Bool)),
                );
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::BinOp { result, op: BinOp::BitAnd, lhs: lo_ok, rhs: hi_ok, ty: IrType::Scalar(DType::Bool) },
                    Some(IrType::Scalar(DType::Bool)),
                );
                Ok(result)
            }
        }
    }

    /// Binds pattern variable names into the current scope.
    fn bind_pattern_vars(
        &mut self,
        scrut_val: ValueId,
        scrut_ty: &IrType,
        pattern: &AstWhenPattern,
        enum_variants_opt: &Option<Vec<String>>,
        enum_variant_field_types: &[Vec<IrType>],
    ) -> Result<(), LowerError> {
        match pattern {
            AstWhenPattern::OptionSome { binding: Some(bind_name) } => {
                let inner_ty = if let IrType::Option(inner) = scrut_ty {
                    (**inner).clone()
                } else {
                    IrType::Infer
                };
                let unwrapped = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::OptionUnwrap { result: unwrapped, operand: scrut_val, result_ty: inner_ty.clone() },
                    Some(inner_ty.clone()),
                );
                self.scope.insert(bind_name.clone(), (unwrapped, inner_ty));
            }
            AstWhenPattern::ResultOk { binding: Some(bind_name) } => {
                let ok_ty = if let IrType::ResultType(ok, _) = scrut_ty {
                    (**ok).clone()
                } else {
                    IrType::Infer
                };
                let unwrapped = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ResultUnwrap { result: unwrapped, operand: scrut_val, result_ty: ok_ty.clone() },
                    Some(ok_ty.clone()),
                );
                self.scope.insert(bind_name.clone(), (unwrapped, ok_ty));
            }
            AstWhenPattern::ResultErr { binding: Some(bind_name) } => {
                let err_ty = if let IrType::ResultType(_, err) = scrut_ty {
                    (**err).clone()
                } else {
                    IrType::Infer
                };
                let unwrapped = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ResultUnwrapErr { result: unwrapped, operand: scrut_val, result_ty: err_ty.clone() },
                    Some(err_ty.clone()),
                );
                self.scope.insert(bind_name.clone(), (unwrapped, err_ty));
            }
            AstWhenPattern::EnumVariant { variant_name, bindings, .. } => {
                if !bindings.is_empty() {
                    if let Some(variants) = enum_variants_opt {
                        let vidx = variants.iter().position(|v| v == variant_name);
                        let empty_fields: Vec<IrType> = Vec::new();
                        let field_types: &Vec<IrType> = vidx
                            .and_then(|i| enum_variant_field_types.get(i))
                            .unwrap_or(&empty_fields);
                        let variant_idx = vidx.unwrap_or(0);
                        for (field_idx, binding_name) in bindings.iter().enumerate() {
                            let field_ty = field_types.get(field_idx).cloned().unwrap_or(IrType::Infer);
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::ExtractVariantField {
                                    result,
                                    operand: scrut_val,
                                    variant_idx,
                                    field_idx,
                                    result_ty: field_ty.clone(),
                                },
                                Some(field_ty.clone()),
                            );
                            self.scope.insert(binding_name.clone(), (result, field_ty));
                        }
                    }
                }
            }
            // Tuple pattern: bind each element to its name (sub-patterns that are ident bindings).
            AstWhenPattern::Tuple(subs) => {
                let tuple_elems = if let IrType::Tuple(ref elems) = scrut_ty {
                    elems.clone()
                } else {
                    vec![]
                };
                for (i, sub) in subs.iter().enumerate() {
                    if let AstWhenPattern::EnumVariant { enum_name, variant_name, .. } = sub {
                        if enum_name.is_empty() {
                            // This is an ident binding
                            let elem_ty = tuple_elems.get(i).cloned().unwrap_or(IrType::Scalar(DType::I64));
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::GetElement { result, base: scrut_val, index: i, result_ty: elem_ty.clone() },
                                Some(elem_ty.clone()),
                            );
                            self.scope.insert(variant_name.clone(), (result, elem_ty));
                        }
                    }
                }
            }
            // No bindings for wildcard, literals, none.
            _ => {}
        }
        Ok(())
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

        // 11. Move to merge block; update scope with final values of rebound outer
        //     variables, but remove the loop variable (it's no longer in scope).
        self.builder.set_current_block(merge_bb);
        for ((name, _, ty), &merge_val) in loop_vars.iter().zip(merge_params.iter()) {
            if name == &var.name {
                // Loop variable goes out of scope at the end of the for loop.
                self.scope.remove(name);
            } else {
                self.scope.insert(name.clone(), (merge_val, ty.clone()));
            }
        }

        let _ = span;
        Ok(())
    }

    /// Lowers `for <var> in <list_expr> { body }` to SSA block-param loop.
    ///
    /// Desugars to:
    /// ```text
    /// val __iter_N = lower(iter_expr)
    /// var __idx_N  = 0
    /// val __len_N  = list_len(__iter_N)
    /// while __idx_N < __len_N {
    ///     val <var> = list_get(__iter_N, __idx_N)
    ///     lower(body)
    ///     __idx_N = __idx_N + 1
    /// }
    /// ```
    fn lower_foreach(
        &mut self,
        var: &crate::parser::ast::Ident,
        iter: &AstExpr,
        body: &AstBlock,
        span: Span,
    ) -> Result<(), LowerError> {
        let i64_ty = IrType::Scalar(DType::I64);

        // Evaluate the list expression once.
        let (iter_val, iter_ty) = self.lower_expr(iter)?;

        // Compute length once before loop.
        let len_val = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ListLen { result: len_val, list: iter_val }, Some(i64_ty.clone()));

        // Initial index = 0.
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt { result: idx_init, value: 0, ty: i64_ty.clone() },
            Some(i64_ty.clone()),
        );

        // Pre-scan body for rebound outer vars (the loop index is always rebound).
        let mut rebound = find_rebound_vars(body);
        let idx_name = format!("__foreach_idx_{}", var.span.start.0);
        if !rebound.contains(&idx_name) {
            rebound.push(idx_name.clone());
        }

        // Collect loop variables: index first, then other rebound outer vars.
        let mut loop_vars: Vec<(String, ValueId, IrType)> = Vec::new();
        loop_vars.push((idx_name.clone(), idx_init, i64_ty.clone()));
        for name in &rebound {
            if name == &idx_name {
                continue;
            }
            if let Some((val, ty)) = self.scope.get(name).cloned() {
                loop_vars.push((name.clone(), val, ty));
            }
        }

        let initial_vals: Vec<ValueId> = loop_vars.iter().map(|(_, v, _)| *v).collect();

        // Create blocks.
        let header_bb = self.builder.create_block(Some("foreach_header"));
        let body_bb = self.builder.create_block(Some("foreach_body"));
        let merge_bb = self.builder.create_block(Some("foreach_merge"));

        // Header block params.
        let mut header_params: Vec<ValueId> = Vec::new();
        for (name, _, ty) in &loop_vars {
            let p = self.builder.add_block_param(header_bb, Some(name), ty.clone());
            header_params.push(p);
        }

        // Merge block params.
        let mut merge_params: Vec<ValueId> = Vec::new();
        for (name, _, ty) in &loop_vars {
            let p = self.builder.add_block_param(merge_bb, Some(name), ty.clone());
            merge_params.push(p);
        }

        // Branch from current block to header.
        self.builder.push_instr(IrInstr::Br { target: header_bb, args: initial_vals }, None);

        // Header block: check index < len.
        self.builder.set_current_block(header_bb);
        let idx_param = header_params[0];
        for ((name, _, ty), &param_val) in loop_vars.iter().zip(header_params.iter()) {
            self.scope.insert(name.clone(), (param_val, ty.clone()));
        }

        let cond_val = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp { result: cond_val, op: BinOp::CmpLt, lhs: idx_param, rhs: len_val, ty: i64_ty.clone() },
            Some(IrType::Scalar(DType::Bool)),
        );
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

        // Body block.
        self.builder.set_current_block(body_bb);

        // Bind loop variable: list_get(iter_val, idx_param).
        let elem_ty = match &iter_ty {
            IrType::List(inner) => *inner.clone(),
            _ => IrType::Infer,
        };
        let elem_val = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet { result: elem_val, list: iter_val, index: idx_param, elem_ty: elem_ty.clone() },
            Some(elem_ty.clone()),
        );
        self.scope.insert(var.name.clone(), (elem_val, elem_ty));

        let loop_var_names: Vec<String> = loop_vars.iter().map(|(n, _, _)| n.clone()).collect();
        self.loop_stack.push((header_bb, merge_bb, loop_var_names.clone()));
        self.lower_block_stmts(body)?;
        self.loop_stack.pop();

        // Emit back-edge Br if body was not terminated.
        if !self.builder.is_current_block_terminated() {
            // Increment index.
            let cur_idx = self.scope.get(&idx_name).map(|(v, _)| *v).unwrap_or(idx_param);
            let one = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ConstInt { result: one, value: 1, ty: i64_ty.clone() },
                Some(i64_ty.clone()),
            );
            let next_idx = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BinOp { result: next_idx, op: BinOp::Add, lhs: cur_idx, rhs: one, ty: i64_ty.clone() },
                Some(i64_ty.clone()),
            );
            self.scope.insert(idx_name.clone(), (next_idx, i64_ty.clone()));

            let updated_vals: Vec<ValueId> = loop_vars
                .iter()
                .map(|(name, original_val, _)| {
                    self.scope.get(name).map(|(v, _)| *v).unwrap_or(*original_val)
                })
                .collect();
            self.builder.push_instr(IrInstr::Br { target: header_bb, args: updated_vals }, None);
        }

        // Move to merge block; restore outer rebound vars.
        self.builder.set_current_block(merge_bb);
        for ((name, _, ty), &merge_val) in loop_vars.iter().zip(merge_params.iter()) {
            if name != &idx_name {
                self.scope.insert(name.clone(), (merge_val, ty.clone()));
            }
        }
        // Remove the synthetic index name from scope if it leaked in.
        self.scope.remove(&idx_name);
        // Loop iteration variable is not in scope after the loop.
        self.scope.remove(&var.name);

        let _ = span;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // List functional operations: map, filter, fold, any, all
    // These are desugared to SSA loops at lowering time (no new IrInstr needed).
    // -----------------------------------------------------------------------

    fn lower_list_map(
        &mut self,
        base_val: ValueId,
        elem_ty: IrType,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        let i64_ty = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);
        if args.len() != 1 {
            return Err(LowerError::Unsupported {
                detail: "list.map expects 1 argument (closure)".into(),
                span,
            });
        }
        let (closure_val, closure_ty) = self.lower_expr(&args[0])?;
        // Extract the closure's return type to use as the mapped element type.
        let mapped_elem_ty = match &closure_ty {
            IrType::Fn { ret, .. } => *ret.clone(),
            _ => elem_ty.clone(),
        };
        let len_val = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ListLen { result: len_val, list: base_val }, Some(i64_ty.clone()));
        let out_list = self.builder.fresh_value();
        let out_list_ty = IrType::List(Box::new(mapped_elem_ty.clone()));
        self.builder.push_instr(IrInstr::ListNew { result: out_list, elem_ty: mapped_elem_ty.clone() }, Some(out_list_ty.clone()));
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ConstInt { result: idx_init, value: 0, ty: i64_ty.clone() }, Some(i64_ty.clone()));

        let header_bb = self.builder.create_block(Some("map_header"));
        let body_bb   = self.builder.create_block(Some("map_body"));
        let merge_bb  = self.builder.create_block(Some("map_merge"));

        let idx_param = self.builder.add_block_param(header_bb, Some("map_idx"), i64_ty.clone());
        let _idx_fin  = self.builder.add_block_param(merge_bb,  Some("map_idx_fin"), i64_ty.clone());

        self.builder.push_instr(IrInstr::Br { target: header_bb, args: vec![idx_init] }, None);

        self.builder.set_current_block(header_bb);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp { result: cond, op: BinOp::CmpLt, lhs: idx_param, rhs: len_val, ty: i64_ty.clone() },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(IrInstr::CondBr { cond, then_block: body_bb, then_args: vec![], else_block: merge_bb, else_args: vec![idx_param] }, None);

        self.builder.set_current_block(body_bb);
        let elem = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ListGet { result: elem, list: base_val, index: idx_param, elem_ty: elem_ty.clone() }, Some(elem_ty.clone()));
        let mapped = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::CallClosure { result: Some(mapped), closure: closure_val, args: vec![elem], result_ty: mapped_elem_ty.clone() }, Some(mapped_elem_ty.clone()));
        self.builder.push_instr(IrInstr::ListPush { list: out_list, value: mapped }, None);
        let one = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ConstInt { result: one, value: 1, ty: i64_ty.clone() }, Some(i64_ty.clone()));
        let next_idx = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::BinOp { result: next_idx, op: BinOp::Add, lhs: idx_param, rhs: one, ty: i64_ty.clone() }, Some(i64_ty.clone()));
        self.builder.push_instr(IrInstr::Br { target: header_bb, args: vec![next_idx] }, None);

        self.builder.set_current_block(merge_bb);
        let _ = span;
        Ok((out_list, out_list_ty))
    }

    fn lower_list_filter(
        &mut self,
        base_val: ValueId,
        elem_ty: IrType,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        let i64_ty = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);
        if args.len() != 1 {
            return Err(LowerError::Unsupported {
                detail: "list.filter expects 1 argument (closure)".into(),
                span,
            });
        }
        let (closure_val, _) = self.lower_expr(&args[0])?;
        let len_val = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ListLen { result: len_val, list: base_val }, Some(i64_ty.clone()));
        let out_list = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ListNew { result: out_list, elem_ty: elem_ty.clone() }, Some(IrType::List(Box::new(elem_ty.clone()))));
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ConstInt { result: idx_init, value: 0, ty: i64_ty.clone() }, Some(i64_ty.clone()));

        let header_bb  = self.builder.create_block(Some("filter_header"));
        let body_bb    = self.builder.create_block(Some("filter_body"));
        let push_bb    = self.builder.create_block(Some("filter_push"));
        let inc_bb     = self.builder.create_block(Some("filter_inc"));
        let merge_bb   = self.builder.create_block(Some("filter_merge"));

        let idx_param  = self.builder.add_block_param(header_bb, Some("filter_idx"), i64_ty.clone());
        let _idx_fin   = self.builder.add_block_param(merge_bb,  Some("filter_idx_fin"), i64_ty.clone());

        self.builder.push_instr(IrInstr::Br { target: header_bb, args: vec![idx_init] }, None);

        self.builder.set_current_block(header_bb);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp { result: cond, op: BinOp::CmpLt, lhs: idx_param, rhs: len_val, ty: i64_ty.clone() },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(IrInstr::CondBr { cond, then_block: body_bb, then_args: vec![], else_block: merge_bb, else_args: vec![idx_param] }, None);

        self.builder.set_current_block(body_bb);
        let elem = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ListGet { result: elem, list: base_val, index: idx_param, elem_ty: elem_ty.clone() }, Some(elem_ty.clone()));
        let keep = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::CallClosure { result: Some(keep), closure: closure_val, args: vec![elem], result_ty: bool_ty.clone() }, Some(bool_ty.clone()));
        self.builder.push_instr(IrInstr::CondBr { cond: keep, then_block: push_bb, then_args: vec![], else_block: inc_bb, else_args: vec![] }, None);

        self.builder.set_current_block(push_bb);
        self.builder.push_instr(IrInstr::ListPush { list: out_list, value: elem }, None);
        self.builder.push_instr(IrInstr::Br { target: inc_bb, args: vec![] }, None);

        self.builder.set_current_block(inc_bb);
        let one = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ConstInt { result: one, value: 1, ty: i64_ty.clone() }, Some(i64_ty.clone()));
        let next_idx = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::BinOp { result: next_idx, op: BinOp::Add, lhs: idx_param, rhs: one, ty: i64_ty.clone() }, Some(i64_ty.clone()));
        self.builder.push_instr(IrInstr::Br { target: header_bb, args: vec![next_idx] }, None);

        self.builder.set_current_block(merge_bb);
        let _ = span;
        Ok((out_list, IrType::List(Box::new(elem_ty))))
    }

    fn lower_list_fold(
        &mut self,
        base_val: ValueId,
        elem_ty: IrType,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        let i64_ty = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);
        if args.len() != 2 {
            return Err(LowerError::Unsupported {
                detail: "list.fold expects 2 arguments (init, closure)".into(),
                span,
            });
        }
        let (init_val, init_ty) = self.lower_expr(&args[0])?;
        let (closure_val, _) = self.lower_expr(&args[1])?;
        let len_val = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ListLen { result: len_val, list: base_val }, Some(i64_ty.clone()));
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ConstInt { result: idx_init, value: 0, ty: i64_ty.clone() }, Some(i64_ty.clone()));

        let header_bb = self.builder.create_block(Some("fold_header"));
        let body_bb   = self.builder.create_block(Some("fold_body"));
        let merge_bb  = self.builder.create_block(Some("fold_merge"));

        let idx_param = self.builder.add_block_param(header_bb, Some("fold_idx"), i64_ty.clone());
        let acc_param = self.builder.add_block_param(header_bb, Some("fold_acc"), init_ty.clone());
        let _idx_fin  = self.builder.add_block_param(merge_bb,  Some("fold_idx_fin"), i64_ty.clone());
        let acc_fin   = self.builder.add_block_param(merge_bb,  Some("fold_acc_fin"), init_ty.clone());

        self.builder.push_instr(IrInstr::Br { target: header_bb, args: vec![idx_init, init_val] }, None);

        self.builder.set_current_block(header_bb);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp { result: cond, op: BinOp::CmpLt, lhs: idx_param, rhs: len_val, ty: i64_ty.clone() },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(IrInstr::CondBr { cond, then_block: body_bb, then_args: vec![], else_block: merge_bb, else_args: vec![idx_param, acc_param] }, None);

        self.builder.set_current_block(body_bb);
        let elem = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ListGet { result: elem, list: base_val, index: idx_param, elem_ty: elem_ty.clone() }, Some(elem_ty.clone()));
        let new_acc = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::CallClosure { result: Some(new_acc), closure: closure_val, args: vec![acc_param, elem], result_ty: init_ty.clone() }, Some(init_ty.clone()));
        let one = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ConstInt { result: one, value: 1, ty: i64_ty.clone() }, Some(i64_ty.clone()));
        let next_idx = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::BinOp { result: next_idx, op: BinOp::Add, lhs: idx_param, rhs: one, ty: i64_ty.clone() }, Some(i64_ty.clone()));
        self.builder.push_instr(IrInstr::Br { target: header_bb, args: vec![next_idx, new_acc] }, None);

        self.builder.set_current_block(merge_bb);
        let _ = span;
        Ok((acc_fin, init_ty))
    }

    fn lower_list_any(
        &mut self,
        base_val: ValueId,
        elem_ty: IrType,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        let i64_ty  = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);
        if args.len() != 1 {
            return Err(LowerError::Unsupported {
                detail: "list.any expects 1 argument (closure)".into(),
                span,
            });
        }
        let (closure_val, _) = self.lower_expr(&args[0])?;
        let len_val = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ListLen { result: len_val, list: base_val }, Some(i64_ty.clone()));
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ConstInt { result: idx_init, value: 0, ty: i64_ty.clone() }, Some(i64_ty.clone()));
        let acc_init = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ConstBool { result: acc_init, value: false }, Some(bool_ty.clone()));

        let header_bb = self.builder.create_block(Some("any_header"));
        let body_bb   = self.builder.create_block(Some("any_body"));
        let merge_bb  = self.builder.create_block(Some("any_merge"));

        let idx_param = self.builder.add_block_param(header_bb, Some("any_idx"), i64_ty.clone());
        let acc_param = self.builder.add_block_param(header_bb, Some("any_acc"), bool_ty.clone());
        let _idx_fin  = self.builder.add_block_param(merge_bb,  Some("any_idx_fin"), i64_ty.clone());
        let acc_fin   = self.builder.add_block_param(merge_bb,  Some("any_acc_fin"), bool_ty.clone());

        self.builder.push_instr(IrInstr::Br { target: header_bb, args: vec![idx_init, acc_init] }, None);

        self.builder.set_current_block(header_bb);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp { result: cond, op: BinOp::CmpLt, lhs: idx_param, rhs: len_val, ty: i64_ty.clone() },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(IrInstr::CondBr { cond, then_block: body_bb, then_args: vec![], else_block: merge_bb, else_args: vec![idx_param, acc_param] }, None);

        self.builder.set_current_block(body_bb);
        let elem = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ListGet { result: elem, list: base_val, index: idx_param, elem_ty: elem_ty.clone() }, Some(elem_ty.clone()));
        let val = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::CallClosure { result: Some(val), closure: closure_val, args: vec![elem], result_ty: bool_ty.clone() }, Some(bool_ty.clone()));
        let new_acc = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::BinOp { result: new_acc, op: BinOp::BitOr, lhs: acc_param, rhs: val, ty: bool_ty.clone() }, Some(bool_ty.clone()));
        let one = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ConstInt { result: one, value: 1, ty: i64_ty.clone() }, Some(i64_ty.clone()));
        let next_idx = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::BinOp { result: next_idx, op: BinOp::Add, lhs: idx_param, rhs: one, ty: i64_ty.clone() }, Some(i64_ty.clone()));
        self.builder.push_instr(IrInstr::Br { target: header_bb, args: vec![next_idx, new_acc] }, None);

        self.builder.set_current_block(merge_bb);
        let _ = span;
        Ok((acc_fin, bool_ty))
    }

    fn lower_list_all(
        &mut self,
        base_val: ValueId,
        elem_ty: IrType,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        let i64_ty  = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);
        if args.len() != 1 {
            return Err(LowerError::Unsupported {
                detail: "list.all expects 1 argument (closure)".into(),
                span,
            });
        }
        let (closure_val, _) = self.lower_expr(&args[0])?;
        let len_val = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ListLen { result: len_val, list: base_val }, Some(i64_ty.clone()));
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ConstInt { result: idx_init, value: 0, ty: i64_ty.clone() }, Some(i64_ty.clone()));
        let acc_init = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ConstBool { result: acc_init, value: true }, Some(bool_ty.clone()));

        let header_bb = self.builder.create_block(Some("all_header"));
        let body_bb   = self.builder.create_block(Some("all_body"));
        let merge_bb  = self.builder.create_block(Some("all_merge"));

        let idx_param = self.builder.add_block_param(header_bb, Some("all_idx"), i64_ty.clone());
        let acc_param = self.builder.add_block_param(header_bb, Some("all_acc"), bool_ty.clone());
        let _idx_fin  = self.builder.add_block_param(merge_bb,  Some("all_idx_fin"), i64_ty.clone());
        let acc_fin   = self.builder.add_block_param(merge_bb,  Some("all_acc_fin"), bool_ty.clone());

        self.builder.push_instr(IrInstr::Br { target: header_bb, args: vec![idx_init, acc_init] }, None);

        self.builder.set_current_block(header_bb);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp { result: cond, op: BinOp::CmpLt, lhs: idx_param, rhs: len_val, ty: i64_ty.clone() },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(IrInstr::CondBr { cond, then_block: body_bb, then_args: vec![], else_block: merge_bb, else_args: vec![idx_param, acc_param] }, None);

        self.builder.set_current_block(body_bb);
        let elem = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ListGet { result: elem, list: base_val, index: idx_param, elem_ty: elem_ty.clone() }, Some(elem_ty.clone()));
        let val = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::CallClosure { result: Some(val), closure: closure_val, args: vec![elem], result_ty: bool_ty.clone() }, Some(bool_ty.clone()));
        let new_acc = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::BinOp { result: new_acc, op: BinOp::BitAnd, lhs: acc_param, rhs: val, ty: bool_ty.clone() }, Some(bool_ty.clone()));
        let one = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::ConstInt { result: one, value: 1, ty: i64_ty.clone() }, Some(i64_ty.clone()));
        let next_idx = self.builder.fresh_value();
        self.builder.push_instr(IrInstr::BinOp { result: next_idx, op: BinOp::Add, lhs: idx_param, rhs: one, ty: i64_ty.clone() }, Some(i64_ty.clone()));
        self.builder.push_instr(IrInstr::Br { target: header_bb, args: vec![next_idx, new_acc] }, None);

        self.builder.set_current_block(merge_bb);
        let _ = span;
        Ok((acc_fin, bool_ty))
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

            AstStmt::Spawn { body, span } => {
                // Lambda-lift the spawn body into a function __spawn_N().
                let counter = self.lambda_counter.get();
                self.lambda_counter.set(counter + 1);
                let fn_name = format!("__spawn_{}", counter);

                // Collect captures (all in-scope variables).
                let captures: Vec<(String, ValueId, IrType)> = self
                    .scope
                    .iter()
                    .map(|(name, (vid, ty))| (name.clone(), *vid, ty.clone()))
                    .collect();

                let lifted_params: Vec<crate::ir::function::Param> = captures
                    .iter()
                    .map(|(name, _, ty)| crate::ir::function::Param { name: name.clone(), ty: ty.clone() })
                    .collect();

                // Build the lifted function with a synthetic AstBlock.
                let ast_block = AstBlock {
                    stmts: body.clone(),
                    tail: None,
                    span: *span,
                };
                let temp_builder = IrFunctionBuilder::new(&fn_name, lifted_params.clone(), IrType::Scalar(DType::I64));
                let mut spawn_lowerer = Lowerer::new_with_lambda_state(
                    temp_builder,
                    self.module,
                    self.fn_sigs,
                    self.lambda_counter.clone(),
                    self.lifted_fns.clone(),
                );
                let entry = spawn_lowerer.builder.create_block(Some("entry"));
                spawn_lowerer.builder.set_current_block(entry);
                // Track outer_val → inner_val mapping to propagate chan_elem_types back.
                let mut capture_val_map: Vec<(ValueId, ValueId)> = Vec::new();
                for (name, outer_val, ty) in &captures {
                    let inner_val = spawn_lowerer.builder.add_block_param(entry, Some(name), ty.clone());
                    spawn_lowerer.scope.insert(name.clone(), (inner_val, ty.clone()));
                    capture_val_map.push((*outer_val, inner_val));
                }
                // Pre-populate spawn_lowerer's chan_elem_types from parent (inner val → elem ty).
                for (outer_val, inner_val) in &capture_val_map {
                    if let Some(elem_ty) = self.chan_elem_types.get(outer_val) {
                        spawn_lowerer.chan_elem_types.insert(*inner_val, elem_ty.clone());
                    }
                }
                spawn_lowerer.lower_block(&ast_block)?;
                // Propagate any new chan_elem_types discovered in spawn back to parent.
                for (outer_val, inner_val) in &capture_val_map {
                    if let Some(elem_ty) = spawn_lowerer.chan_elem_types.get(inner_val) {
                        self.chan_elem_types.entry(*outer_val).or_insert_with(|| elem_ty.clone());
                    }
                }
                // Emit a return of 0 if not already terminated.
                let dummy_ret = spawn_lowerer.builder.fresh_value();
                spawn_lowerer.builder.push_instr(IrInstr::ConstInt { result: dummy_ret, value: 0, ty: IrType::Scalar(DType::I64) }, Some(IrType::Scalar(DType::I64)));
                spawn_lowerer.builder.push_instr(IrInstr::Return { values: vec![dummy_ret] }, None);
                spawn_lowerer.builder.seal_unterminated_blocks();
                let ir_func = spawn_lowerer.builder.build();
                self.lifted_fns.borrow_mut().push(ir_func);

                let capture_vals: Vec<ValueId> = captures.iter().map(|(_, v, _)| *v).collect();
                self.builder.push_instr(IrInstr::Spawn { body_fn: fn_name, args: capture_vals }, None);
                let _ = span;
                Ok(())
            }

            AstStmt::ForEach { var, iter, body, span } => {
                self.lower_foreach(var, iter, body, *span)
            }

            AstStmt::ParFor { var, start, end, body, span } => {
                // Lambda-lift body into __par_body_N(var: i64, captures...) { body }.
                let counter = self.lambda_counter.get();
                self.lambda_counter.set(counter + 1);
                let fn_name = format!("__par_body_{}", counter);

                // Collect outer-scope captures (all in-scope variables except the loop var).
                let captures: Vec<(String, ValueId, IrType)> = self
                    .scope
                    .iter()
                    .filter(|(name, _)| *name != &var.name)
                    .map(|(name, (vid, ty))| (name.clone(), *vid, ty.clone()))
                    .collect();

                // Build params: loop var first, then captures.
                let mut params = vec![crate::ir::function::Param { name: var.name.clone(), ty: IrType::Scalar(DType::I64) }];
                for (name, _, ty) in &captures {
                    params.push(crate::ir::function::Param { name: name.clone(), ty: ty.clone() });
                }

                let temp_builder = IrFunctionBuilder::new(&fn_name, params, IrType::Scalar(DType::I64));
                let mut body_lowerer = Lowerer::new_with_lambda_state(
                    temp_builder,
                    self.module,
                    self.fn_sigs,
                    self.lambda_counter.clone(),
                    self.lifted_fns.clone(),
                );
                let entry = body_lowerer.builder.create_block(Some("entry"));
                body_lowerer.builder.set_current_block(entry);
                // Add loop var as first block param.
                let var_val = body_lowerer.builder.add_block_param(entry, Some(&var.name), IrType::Scalar(DType::I64));
                body_lowerer.scope.insert(var.name.clone(), (var_val, IrType::Scalar(DType::I64)));
                // Add capture params.
                for (name, _, ty) in &captures {
                    let inner_val = body_lowerer.builder.add_block_param(entry, Some(name), ty.clone());
                    body_lowerer.scope.insert(name.clone(), (inner_val, ty.clone()));
                }
                body_lowerer.lower_block(body)?;
                let dummy_ret = body_lowerer.builder.fresh_value();
                body_lowerer.builder.push_instr(IrInstr::ConstInt { result: dummy_ret, value: 0, ty: IrType::Scalar(DType::I64) }, Some(IrType::Scalar(DType::I64)));
                body_lowerer.builder.push_instr(IrInstr::Return { values: vec![dummy_ret] }, None);
                body_lowerer.builder.seal_unterminated_blocks();
                let ir_func = body_lowerer.builder.build();
                self.lifted_fns.borrow_mut().push(ir_func);

                let (start_val, _) = self.lower_expr(start)?;
                let (end_val, _) = self.lower_expr(end)?;
                let var_id = self.builder.fresh_value();
                let capture_vals: Vec<ValueId> = captures.iter().map(|(_, v, _)| *v).collect();
                self.builder.push_instr(
                    IrInstr::ParFor { var: var_id, start: start_val, end: end_val, body_fn: fn_name, args: capture_vals },
                    None,
                );
                let _ = span;
                Ok(())
            }
        }
    }
}

/// Lower a function with full generic/monomorphization state.
#[allow(clippy::too_many_arguments)]
fn lower_function_with_generics(
    func: &AstFunction,
    module: &IrModule,
    fn_sigs: &HashMap<String, IrType>,
    const_defs: &std::rc::Rc<HashMap<String, AstExpr>>,
    generic_fns: std::rc::Rc<HashMap<String, AstFunction>>,
    mono_cache: std::rc::Rc<std::cell::RefCell<std::collections::HashSet<String>>>,
    mono_sigs: std::rc::Rc<std::cell::RefCell<HashMap<String, IrType>>>,
    trait_dispatch: std::rc::Rc<HashMap<String, Vec<(IrType, String)>>>,
    fn_defaults: std::rc::Rc<HashMap<String, Vec<Option<AstExpr>>>>,
) -> Result<(crate::ir::function::IrFunction, Vec<crate::ir::function::IrFunction>), LowerError> {
    lower_function_with_generics_and_subs(
        func, module, fn_sigs, const_defs, generic_fns, mono_cache, mono_sigs,
        HashMap::new(), // no type param subs for top-level functions
        trait_dispatch, fn_defaults,
    )
}

#[allow(clippy::too_many_arguments)]
fn lower_function_with_generics_and_subs(
    func: &AstFunction,
    module: &IrModule,
    fn_sigs: &HashMap<String, IrType>,
    const_defs: &std::rc::Rc<HashMap<String, AstExpr>>,
    generic_fns: std::rc::Rc<HashMap<String, AstFunction>>,
    mono_cache: std::rc::Rc<std::cell::RefCell<std::collections::HashSet<String>>>,
    mono_sigs: std::rc::Rc<std::cell::RefCell<HashMap<String, IrType>>>,
    type_param_subs: HashMap<String, IrType>,
    trait_dispatch: std::rc::Rc<HashMap<String, Vec<(IrType, String)>>>,
    fn_defaults: std::rc::Rc<HashMap<String, Vec<Option<AstExpr>>>>,
) -> Result<(crate::ir::function::IrFunction, Vec<crate::ir::function::IrFunction>), LowerError> {
    // Resolve param and return types with substitution applied.
    let resolve = |ty: &AstType| -> IrType {
        if let AstType::Named(name, _) = ty {
            if let Some(concrete) = type_param_subs.get(name) {
                return concrete.clone();
            }
        }
        lower_type_with_structs(ty, module)
    };

    let return_ty = resolve(&func.return_ty);
    let params: Vec<Param> = func.params.iter()
        .map(|p| Param { name: p.name.name.clone(), ty: resolve(&p.ty) })
        .collect();

    let mut builder = IrFunctionBuilder::new(&func.name.name, params.clone(), return_ty.clone());
    let entry = builder.create_block(Some("entry"));
    builder.set_current_block(entry);

    let lambda_counter = std::rc::Rc::new(std::cell::Cell::new(0u32));
    let lifted_fns = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let mut lowerer = Lowerer::new_generic(
        builder, module, fn_sigs, lambda_counter, lifted_fns.clone(),
        type_param_subs, generic_fns, mono_cache, mono_sigs, const_defs.clone(),
        trait_dispatch, fn_defaults,
    );

    // Register function parameters as entry block params.
    for (param, ir_param) in func.params.iter().zip(params.iter()) {
        let val = lowerer.builder.add_block_param(entry, Some(&param.name.name), ir_param.ty.clone());
        lowerer.scope.insert(param.name.name.clone(), (val, ir_param.ty.clone()));
    }

    // Inject global constants into scope.
    for (name, expr) in lowerer.const_defs.clone().iter() {
        let (val, ty) = lowerer.lower_expr(expr)?;
        lowerer.scope.insert(name.clone(), (val, ty));
    }

    let tail_val = lowerer.lower_block(&func.body)?;

    if !lowerer.builder.is_current_block_terminated() {
        let ret_values: Vec<ValueId> = match tail_val {
            Some((v, _)) => vec![v],
            None => vec![],
        };
        lowerer.builder.push_instr(IrInstr::Return { values: ret_values }, None);
    }

    lowerer.builder.seal_unterminated_blocks();

    let ir_func = lowerer.builder.build();
    let lifted = match std::rc::Rc::try_unwrap(lifted_fns) {
        Ok(cell) => cell.into_inner(),
        Err(rc) => rc.borrow().clone(),
    };
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
        AstType::Option(inner, _) => IrType::Option(Box::new(lower_type(inner))),
        AstType::Result(ok_ty, err_ty, _) => {
            IrType::ResultType(Box::new(lower_type(ok_ty)), Box::new(lower_type(err_ty)))
        }
        AstType::Chan(elem, _) => IrType::Chan(Box::new(lower_type(elem))),
        AstType::Atomic(inner, _) => IrType::Atomic(Box::new(lower_type(inner))),
        AstType::Mutex(inner, _) => IrType::Mutex(Box::new(lower_type(inner))),
        AstType::Grad(inner, _) => IrType::Grad(Box::new(lower_type(inner))),
        AstType::Sparse(inner, _) => IrType::Sparse(Box::new(lower_type(inner))),
        AstType::Fn { params, ret, .. } => IrType::Fn {
            params: params.iter().map(lower_type).collect(),
            ret: Box::new(lower_type(ret)),
        },
    }
}

/// Converts a type name string (as written in `impl Trait for TypeName`) to an `IrType`.
fn type_name_to_ir_type(name: &str, module: &IrModule) -> IrType {
    match name {
        "i64"  => IrType::Scalar(DType::I64),
        "i32"  => IrType::Scalar(DType::I32),
        "f64"  => IrType::Scalar(DType::F64),
        "f32"  => IrType::Scalar(DType::F32),
        "bool" => IrType::Scalar(DType::Bool),
        "str"  => IrType::Str,
        _ => {
            if let Some(fields) = module.struct_def(name) {
                IrType::Struct { name: name.to_owned(), fields: fields.clone() }
            } else if let Some(variants) = module.enum_def(name) {
                IrType::Enum { name: name.to_owned(), variants: variants.clone() }
            } else {
                IrType::Infer
            }
        }
    }
}

/// Returns a short string key for `ty` used to look up trait dispatch entries.
fn ir_type_dispatch_name(ty: &IrType) -> String {
    match ty {
        IrType::Scalar(DType::I64)  => "i64".to_owned(),
        IrType::Scalar(DType::I32)  => "i32".to_owned(),
        IrType::Scalar(DType::F64)  => "f64".to_owned(),
        IrType::Scalar(DType::F32)  => "f32".to_owned(),
        IrType::Scalar(DType::Bool) => "bool".to_owned(),
        IrType::Str => "str".to_owned(),
        IrType::Struct { name, .. } => name.clone(),
        IrType::Enum { name, .. } => name.clone(),
        other => format!("{}", other),
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
            // Check type aliases first.
            if let Some(aliased) = module.type_alias(name) {
                return aliased.clone();
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
        AstType::Option(inner, _) => {
            IrType::Option(Box::new(lower_type_with_structs(inner, module)))
        }
        AstType::Result(ok_ty, err_ty, _) => IrType::ResultType(
            Box::new(lower_type_with_structs(ok_ty, module)),
            Box::new(lower_type_with_structs(err_ty, module)),
        ),
        AstType::Chan(elem, _) => IrType::Chan(Box::new(lower_type_with_structs(elem, module))),
        AstType::Atomic(inner, _) => IrType::Atomic(Box::new(lower_type_with_structs(inner, module))),
        AstType::Mutex(inner, _) => IrType::Mutex(Box::new(lower_type_with_structs(inner, module))),
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
        AstScalarKind::U8 => DType::U8,
        AstScalarKind::I8 => DType::I8,
        AstScalarKind::U32 => DType::U32,
        AstScalarKind::U64 => DType::U64,
        AstScalarKind::USize => DType::USize,
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
            AstStmt::ForEach { var, body, .. } => {
                if !names.contains(&var.name) {
                    names.push(var.name.clone());
                }
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
            | AstStmt::ForEach { body, .. }
            | AstStmt::While { body, .. }
            | AstStmt::Loop { body, .. } => {
                collect_nested_mutations(body, names);
            }
            _ => {}
        }
    }
}
