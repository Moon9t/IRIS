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

/// Simple Levenshtein distance between two strings (caps at 4 for speed).
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let m = a.len();
    let n = b.len();
    if m.abs_diff(n) > 4 {
        return 5;
    }
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for (i, row) in dp.iter_mut().enumerate().take(m + 1) {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate().take(n + 1) {
        *cell = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            dp[i][j] = if a[i - 1] == b[j - 1] {
                dp[i - 1][j - 1]
            } else {
                1 + dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1])
            };
        }
    }
    dp[m][n]
}

/// Find the closest name in `candidates` to `name`, if within edit distance 2.
fn did_you_mean<'a>(name: &str, candidates: impl Iterator<Item = &'a str>) -> Option<String> {
    let mut best: Option<(usize, &str)> = None;
    for c in candidates {
        let d = levenshtein(name, c);
        if d <= 2 && best.map(|(bd, _)| d < bd).unwrap_or(true) {
            best = Some((d, c));
        }
    }
    best.map(|(_, s)| s.to_owned())
}

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

thread_local! {
    /// Function names defined by the source currently being lowered.
    ///
    /// `fn_sigs` cannot answer "did the user write this?" because it is
    /// pre-populated with builtin return types (println, print, sleep_ms, ...)
    /// so call sites get concrete types. Deciding whether a definition shadows
    /// a builtin needs the set the *source* actually declares. See #15.
    static CURRENT_USER_FNS: RefCell<std::collections::HashSet<String>> =
        RefCell::new(std::collections::HashSet::new());
    /// Every non-generic function's AST, so a call carrying a reverse-mode tape
    /// value can be lowered inline and keep its handle. A tape handle cannot be
    /// passed as an ordinary argument -- it has no place in an `f64` parameter --
    /// and multi-value returns are not supported, so the handle cannot come back
    /// out either. Inlining at lowering time sidesteps both: `tape_nodes` is
    /// keyed by `ValueId`, so binding the callee's parameters to the caller's
    /// argument values carries the mapping across for free. See #49.
    static CURRENT_FN_ASTS: RefCell<HashMap<String, crate::parser::ast::AstFunction>> =
        RefCell::new(HashMap::new());
    static CURRENT_BRING_PREFIXES: RefCell<Vec<Vec<String>>> = const { RefCell::new(Vec::new()) };
    static CURRENT_BRING_MAPPINGS: RefCell<Vec<HashMap<String, String>>> = const { RefCell::new(Vec::new()) };
}

fn set_current_brings(brings: &[crate::parser::ast::AstBring]) {
    let mut prefixes = Vec::new();
    let mut mappings = HashMap::new();
    for bring in brings {
        match &bring.path {
            crate::parser::ast::BringPath::File(rel_path) => {
                let path_obj = std::path::Path::new(rel_path);
                let stem = path_obj
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("module")
                    .replace(['.', '-'], "_");
                if !prefixes.contains(&stem) {
                    prefixes.push(stem.clone());
                }
                mappings.insert(stem.clone(), stem);
            }
            crate::parser::ast::BringPath::Stdlib(name) => {
                let mangled_prefix = name.replace(['.', '-'], "_");
                if !prefixes.contains(&mangled_prefix) {
                    prefixes.push(mangled_prefix.clone());
                }
                let qualifier = if let Some(last_dot) = name.rfind('.') {
                    &name[last_dot + 1..]
                } else {
                    name.as_str()
                };
                mappings.insert(qualifier.to_string(), mangled_prefix);
            }
        }
    }
    CURRENT_BRING_PREFIXES.with(|p| {
        p.borrow_mut().push(prefixes);
    });
    CURRENT_BRING_MAPPINGS.with(|m| {
        m.borrow_mut().push(mappings);
    });
}

fn clear_current_brings() {
    CURRENT_BRING_PREFIXES.with(|p| {
        p.borrow_mut().pop();
    });
    CURRENT_BRING_MAPPINGS.with(|m| {
        m.borrow_mut().pop();
    });
}

fn resolve_qualifier(qualifier: &str) -> String {
    CURRENT_BRING_MAPPINGS.with(|m| {
        m.borrow()
            .last()
            .and_then(|map| map.get(qualifier))
            .cloned()
            .unwrap_or_else(|| qualifier.to_string())
    })
}

pub(crate) fn resolve_brought_name(name: &str, module: &IrModule) -> String {
    let resolved = CURRENT_BRING_PREFIXES.with(|prefixes_stack| {
        if let Some(prefixes) = prefixes_stack.borrow().last() {
            for prefix in prefixes.iter() {
                let candidate = format!("{}__{}", prefix, name);
                if module.struct_def(&candidate).is_some()
                    || module.enum_def(&candidate).is_some()
                    || module.type_alias(&candidate).is_some()
                {
                    return Some(candidate);
                }
            }
        }
        None
    });
    if let Some(res) = resolved {
        return res;
    }
    if module.struct_def(name).is_some()
        || module.enum_def(name).is_some()
        || module.type_alias(name).is_some()
    {
        return name.to_string();
    }
    // Fallback: scan all type registries for any mangled candidate matching `*__name`.
    let suffix = format!("__{}", name);
    for key in module.struct_defs.keys() {
        if key.ends_with(&suffix) {
            return key.clone();
        }
    }
    for key in module.enum_defs.keys() {
        if key.ends_with(&suffix) {
            return key.clone();
        }
    }
    for key in module.type_aliases.keys() {
        if key.ends_with(&suffix) {
            return key.clone();
        }
    }
    name.to_string()
}

use crate::error::LowerError;
use crate::ir::block::BlockId;
use crate::ir::function::Param;
use crate::ir::instr::{BinOp, IrInstr, ScalarUnaryOp, TensorOp};
use crate::ir::module::{IrFunctionBuilder, IrModule};
use crate::ir::types::{DType, Dim, IrType, Shape, TraitMethodSig};
use crate::ir::value::ValueId;
use crate::parser::ast::{
    AstBinOp, AstBlock, AstDim, AstExpr, AstFunction, AstHandlerArm, AstModule, AstScalarKind,
    AstStmt, AstType,
    AstUnaryOp, AstWhenArm, AstWhenPattern, Ident,
};
use crate::parser::lexer::Span;

/// Returns true if a pattern extracts variable bindings (needs safe evaluation order with guards).
fn pattern_has_bindings(pattern: &AstWhenPattern) -> bool {
    match pattern {
        AstWhenPattern::Binding { .. } => true,
        AstWhenPattern::OptionSome { binding: Some(_) } => true,
        AstWhenPattern::ResultOk { binding: Some(_) } => true,
        AstWhenPattern::ResultErr { binding: Some(_) } => true,
        AstWhenPattern::EnumVariant { bindings, .. } if !bindings.is_empty() => true,
        // A tuple pattern has bindings if ANY sub-pattern is an ident binding
        AstWhenPattern::Tuple(subs) => subs.iter().any(|s| pattern_has_bindings(s)
            || matches!(s, AstWhenPattern::EnumVariant { enum_name, .. } if enum_name.is_empty())),
        // A struct pattern has bindings if any field sub-pattern has bindings or is a bare ident
        AstWhenPattern::Struct { fields, .. } => fields.iter().any(|(_, f)| pattern_has_bindings(f)
            || matches!(f, AstWhenPattern::EnumVariant { enum_name, .. } if enum_name.is_empty())
            || matches!(f, AstWhenPattern::Wildcard)),
        _ => false,
    }
}

/// Lower an `AstModule` to an `IrModule`.
pub fn lower(ast: &AstModule, module_name: &str) -> Result<IrModule, LowerError> {
    set_current_brings(&ast.brings);
    struct ScopeGuard;
    impl Drop for ScopeGuard {
        fn drop(&mut self) {
            clear_current_brings();
        }
    }
    let _guard = ScopeGuard;

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
        let variant_fields: Vec<Vec<IrType>> = e
            .variants
            .iter()
            .map(|v| v.fields.iter().map(lower_type).collect())
            .collect();
        module
            .add_enum_def(e.name.name.clone(), variants, variant_fields)
            .map_err(|_| LowerError::DuplicateFunction {
                name: e.name.name.clone(),
                span: e.name.span,
            })?;
    }

    // 2. Register struct definitions so functions can reference them.
    // Also, pre-collect and monomorphize generic struct applications on demand.
    let mut generic_apps = std::collections::HashSet::new();
    for s in &ast.structs {
        for f in &s.fields {
            collect_generic_apps_in_type(&f.ty, &mut generic_apps);
        }
    }
    for f in &ast.functions {
        for p in &f.params {
            collect_generic_apps_in_type(&p.ty, &mut generic_apps);
        }
        collect_generic_apps_in_type(&f.return_ty, &mut generic_apps);
        collect_generic_apps_in_block(&f.body, &mut generic_apps);
    }
    for i in &ast.impls {
        for m in &i.methods {
            for p in &m.params {
                collect_generic_apps_in_type(&p.ty, &mut generic_apps);
            }
            collect_generic_apps_in_type(&m.return_ty, &mut generic_apps);
            collect_generic_apps_in_block(&m.body, &mut generic_apps);
        }
    }

    let mut generic_struct_templates = std::collections::HashMap::new();
    for s in &ast.structs {
        if !s.type_params.is_empty() {
            generic_struct_templates.insert(s.name.name.clone(), s.clone());
        }
    }

    // Register non-generic struct definitions first.
    for s in &ast.structs {
        if s.type_params.is_empty() {
            let fields: Vec<(String, IrType)> = s
                .fields
                .iter()
                .map(|f| (f.name.name.clone(), lower_type_with_structs(&f.ty, &module)))
                .collect();
            let defaults: Vec<Option<AstExpr>> = s
                .fields
                .iter()
                .map(|f| f.default.clone())
                .collect();
            module
                .add_struct_def(s.name.name.clone(), fields)
                .map_err(|_| LowerError::DuplicateFunction {
                    name: s.name.name.clone(),
                    span: s.name.span,
                })?;
            if defaults.iter().any(|d| d.is_some()) {
                module.struct_defaults.insert(s.name.name.clone(), defaults);
            }
        }
    }

    // Register generic struct templates so field access works inside generic
    // functions.  We use `lower_type_with_structs` (without Infer substitution)
    // so type params appear as `Struct { name: "T", fields: [] }` — a marker
    // that can be substituted with the concrete type at monomorphization time.
    // Sorted, because `generic_struct_templates` is a `HashMap` and the order
    // in which templates are registered decides the order their monomorphised
    // instantiations are created. With a nested generic that changed the
    // resulting IR between runs of the same program. See known-issues #21.
    let mut template_names: Vec<&String> = generic_struct_templates.keys().collect();
    template_names.sort();
    for name in template_names {
        let template = &generic_struct_templates[name];
        let fields: Vec<(String, IrType)> = template
            .fields
            .iter()
            .map(|f| (f.name.name.clone(), lower_type_with_structs(&f.ty, &module)))
            .collect();
        let defaults: Vec<Option<AstExpr>> = template
            .fields
            .iter()
            .map(|f| f.default.clone())
            .collect();
        let _ = module.add_struct_def(name.clone(), fields);
        if defaults.iter().any(|d| d.is_some()) {
            module.struct_defaults.insert(name.clone(), defaults);
        }
    }

    let mut worklist: Vec<(String, Vec<AstType>)> = generic_apps.into_iter().collect();
    let mut processed_apps = std::collections::HashSet::new();

    while let Some((base_name, type_args)) = worklist.pop() {
        if !processed_apps.insert((base_name.clone(), type_args.clone())) {
            continue;
        }
        if let Some(template) = generic_struct_templates.get(&base_name).cloned() {
            let mut type_subs = std::collections::HashMap::new();
            let mut constructor_subs = std::collections::HashMap::new();
            for (p, arg) in template.type_params.iter().zip(type_args.iter()) {
                match p {
                    crate::parser::ast::AstGenericParam::Type(p_name, _, _) => {
                        type_subs.insert(p_name.clone(), arg.clone());
                    }
                    crate::parser::ast::AstGenericParam::Hkt(p_name, _, _, _) => {
                        let target_constructor = match arg {
                            AstType::Named(n, _) => n.clone(),
                            AstType::Generic { name: n, .. } => n.clone(),
                            _ => "".to_string(),
                        };
                        constructor_subs.insert(p_name.clone(), target_constructor);
                    }
                    _ => {}
                }
            }

            let mut substituted_fields = Vec::new();
            let mut local_apps = std::collections::HashSet::new();
            for f in &template.fields {
                let sub_ty = substitute_ast_type(&f.ty, &type_subs, &constructor_subs);
                collect_generic_apps_in_type(&sub_ty, &mut local_apps);
                substituted_fields.push((f.name.name.clone(), sub_ty));
            }

            for app in local_apps {
                if !processed_apps.contains(&app) {
                    worklist.push(app.clone());
                }
            }

            // Lower the fields of the monomorphized struct.
            let mut lowered_fields = Vec::new();
            for (f_name, f_ty) in substituted_fields {
                lowered_fields.push((f_name, lower_type_with_structs(&f_ty, &module)));
            }

            // Register the monomorphized struct definition.
            let mangled_name = format!("{}__{}", base_name, type_args.iter().map(|arg| mangle_ir_type(&lower_type_with_structs(arg, &module))).collect::<Vec<_>>().join("_"));
            module.add_struct_def(mangled_name.clone(), lowered_fields).ok();
            // Carry over defaults from the template (defaults are typically literals).
            if let Some(template_defaults) = module.struct_defaults.get(&base_name).cloned() {
                module.struct_defaults.insert(mangled_name, template_defaults);
            }
        }
    }

    // Perform field population sweep to resolve any empty forward-referenced struct fields.
    let mut updated_defs = module.struct_defs.clone();
    for fields in updated_defs.values_mut() {
        for (_, f_ty) in fields {
            populate_struct_fields(f_ty, &module);
        }
    }
    module.struct_defs = updated_defs;

    // 2b. Register trait definitions so `dyn Trait` IR types can be resolved
    // and vtables can be emitted at codegen.
    for t in &ast.traits {
        let methods: Vec<TraitMethodSig> = t
            .methods
            .iter()
            .map(|m| TraitMethodSig {
                name: m.name.name.clone(),
                params: m.params.iter().map(|p| lower_type_with_structs(&p.ty, &module)).collect(),
                ret: Box::new(lower_type_with_structs(&m.return_ty, &module)),
            })
            .collect();
        module.add_trait_def(t.name.name.clone(), methods);
    }
    // 2c. Register impl methods so trait_object codegen can find the concrete
    // mangled function name for each `(trait, concrete_struct, method)` triple.
    for impl_def in &ast.impls {
        if impl_def.trait_name.is_empty() {
            continue;
        }
        // Skip blanket impls — they're monomorphized later based on known concrete types.
        if !impl_def.generic_params.is_empty() {
            continue;
        }
        let trait_name = impl_def.trait_name.clone();
        for method in &impl_def.methods {
            let mangled = format!(
                "{}__{}__{}",
                impl_def.trait_name, impl_def.type_name, method.name.name
            );
            module.add_trait_impl_method(
                trait_name.clone(),
                impl_def.type_name.clone(),
                method.name.name.clone(),
                mangled,
            );
        }
    }

    // 3. Pre-collect function return types so call sites get concrete types.
    // Generic functions (with type_params) are excluded from fn_sigs; they're
    // monomorphized on demand during lower_call.
    let mut fn_sigs: HashMap<String, IrType> = HashMap::new();
    // Names this program actually defines.
    //
    // Distinct from `fn_sigs`, which is pre-populated below with builtin return
    // types (println, print, sleep_ms, ...) so call sites get concrete types.
    // That makes `fn_sigs` useless for answering "did the user write this?",
    // which is what deciding a builtin shadow requires.
    let mut user_fn_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut generic_fn_map: HashMap<String, crate::parser::ast::AstFunction> = HashMap::new();
    let mut fn_defaults_map: HashMap<String, Vec<Option<crate::parser::ast::AstExpr>>> =
        HashMap::new();
    // Declared parameter names, in order, for every function in the program.
    // Used to resolve named arguments `f(w = 3, h = 4)` to positional slots.
    // Kept separate from `fn_defaults`, which is only populated for functions
    // that actually have defaults.
    let mut fn_param_names_map: HashMap<String, Vec<String>> = HashMap::new();
    // Declared parameter types, used to coerce a concrete value to a trait
    // object at a call site. Built from the AST rather than read back from
    // `IrModule`, because a callee defined later in the file has not been
    // lowered yet when its caller is.
    let mut fn_param_types_map: HashMap<String, Vec<IrType>> = HashMap::new();
    for func in &ast.functions {
        user_fn_names.insert(func.name.name.clone());
        if func.type_params.is_empty() {
            let mut ret_ty = lower_type_with_structs(&func.return_ty, &module);
            if func.is_async {
                ret_ty = IrType::Chan(Box::new(ret_ty));
            }
            fn_sigs.insert(func.name.name.clone(), ret_ty);
        } else {
            generic_fn_map.insert(func.name.name.clone(), func.clone());
        }
        // Also store const fn bodies so we can evaluate calls at compile time.
        if func.is_const {
            generic_fn_map.insert(func.name.name.clone(), func.clone());
        }
        fn_param_names_map.insert(
            func.name.name.clone(),
            func.params.iter().map(|p| p.name.name.clone()).collect(),
        );
        fn_param_types_map.insert(
            func.name.name.clone(),
            func.params
                .iter()
                .map(|p| lower_type_with_structs(&p.ty, &module))
                .collect(),
        );
        if func.params.iter().any(|p| p.default.is_some()) {
            fn_defaults_map.insert(
                func.name.name.clone(),
                func.params.iter().map(|p| p.default.clone()).collect(),
            );
        }
    }
    let generic_fns = std::rc::Rc::new(generic_fn_map);
    CURRENT_USER_FNS.with(|u| {
        *u.borrow_mut() = user_fn_names;
    });
    CURRENT_FN_ASTS.with(|m| {
        let mut map = m.borrow_mut();
        map.clear();
        for func in &ast.functions {
            if func.type_params.is_empty() {
                map.insert(func.name.name.clone(), func.clone());
            }
        }
    });
    let fn_defaults = std::rc::Rc::new(fn_defaults_map);
    let fn_param_names = std::rc::Rc::new(fn_param_names_map);
    let fn_param_types = std::rc::Rc::new(fn_param_types_map);

    // Pre-populate built-in / runtime function return types so call sites
    // get concrete types instead of Infer.
    fn_sigs
        .entry("println".into())
        .or_insert(IrType::Scalar(DType::I64));
    fn_sigs
        .entry("print".into())
        .or_insert(IrType::Scalar(DType::I64));
    fn_sigs
        .entry("eprintln".into())
        .or_insert(IrType::Scalar(DType::I64));

    fn_sigs
        .entry("eprint".into())
        .or_insert(IrType::Scalar(DType::I64));
    fn_sigs
        .entry("sleep_ms".into())
        .or_insert(IrType::Scalar(DType::I64));
    fn_sigs
        .entry("random_i64".into())
        .or_insert(IrType::Scalar(DType::I64));
    fn_sigs
        .entry("random_f64".into())
        .or_insert(IrType::Scalar(DType::F64));
    fn_sigs
        .entry("time_ms".into())
        .or_insert(IrType::Scalar(DType::I64));
    fn_sigs
        .entry("exit".into())
        .or_insert(IrType::Scalar(DType::I64));
    fn_sigs
        .entry("len".into())
        .or_insert(IrType::Scalar(DType::I64));
    fn_sigs
        .entry("str_len".into())
        .or_insert(IrType::Scalar(DType::I64));
    fn_sigs
        .entry("assert".into())
        .or_insert(IrType::Scalar(DType::I64));
    fn_sigs
        .entry("assert_eq".into())
        .or_insert(IrType::Scalar(DType::I64));

    // 3b. Collect global const declarations as named expressions.
    let const_defs_map: HashMap<String, AstExpr> = ast
        .consts
        .iter()
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
        // A generic *target* -- `impl[T] Show for list<T>` -- is not a blanket
        // impl. A blanket impl enumerates the concrete types satisfying a bound
        // and emits one copy per type; here the parameter belongs to the target
        // container, and the method body is usually indifferent to it
        // (`list_len(self)` works for any element type). Registering it once
        // against `list<_>` and letting dispatch match on the constructor is
        // both correct and far cheaper than an instantiation per element type.
        //
        // Without this the blanket path caught it, searched for concrete types
        // satisfying a bound it does not have, found none, and emitted nothing.
        // See known-issues #38.
        let target_is_generic = impl_def.target_ty.is_some()
            && !impl_def.generic_params.is_empty();

        // Handle blanket impls: monomorphize for each known concrete type.
        if !impl_def.generic_params.is_empty()
            && !impl_def.trait_name.is_empty()
            && !target_is_generic
        {
            // Find the type param name (e.g., "T" in `impl[T where T: Show]`).
            if let crate::parser::ast::AstGenericParam::Type(ref tp_name, ref bounds, _) =
                &impl_def.generic_params[0]
            {
                // Find all concrete types that satisfy the required trait bounds.
                // We use trait_impl_methods to find types implementing each bound.
                let mut candidate_types: Option<Vec<String>> = None;
                for bound_trait in bounds {
                    let types_for_bound: Vec<String> = module
                        .trait_impl_methods()
                        .get(bound_trait)
                        .map(|entries| {
                            entries.iter().map(|(cty, _, _)| cty.clone()).collect::<std::collections::HashSet<_>>().into_iter().collect()
                        })
                        .unwrap_or_default();
                    candidate_types = Some(match candidate_types {
                        Some(prev) => prev.into_iter().filter(|t| types_for_bound.contains(t)).collect(),
                        None => types_for_bound,
                    });
                }
                // If no bounds specified, skip (can't determine concrete types).
                let concrete_types = candidate_types.unwrap_or_default();
                // If bounds exist but no concrete types found yet, also add all types in the module.
                let concrete_types = if concrete_types.is_empty() && !bounds.is_empty() {
                    // Fallback: try all known struct/enum types.
                    module.struct_defs.keys().cloned().collect::<Vec<_>>()
                } else {
                    concrete_types
                };

                for concrete_ty_name in &concrete_types {
                    let concrete_ir = type_name_to_ir_type(concrete_ty_name, &module);
                    // Also register in trait_impl_methods for vtable/dyn dispatch.
                    for method in &impl_def.methods {
                        let mangled = format!(
                            "{}__{}__{}",
                            impl_def.trait_name, concrete_ty_name, method.name.name
                        );
                        module.add_trait_impl_method(
                            impl_def.trait_name.clone(),
                            concrete_ty_name.clone(),
                            method.name.name.clone(),
                            mangled.clone(),
                        );
                        // Resolve return type with substitutions.
                        let mut ret_ty = lower_type_with_structs(&method.return_ty, &module);
                        if method.is_async { ret_ty = IrType::Chan(Box::new(ret_ty)); }
                        fn_sigs.insert(mangled.clone(), ret_ty);
                        trait_dispatch_map
                            .entry(method.name.name.clone())
                            .or_default()
                            .push((concrete_ir.clone(), mangled.clone()));
                        // Build a renamed copy with concrete type substitutions.
                        let mut renamed = method.clone();
                        renamed.name.name = mangled;
                        renamed.type_params.clear(); // No longer generic.
                        for param in &mut renamed.params {
                            if param.name.name == "self" {
                                if let crate::parser::ast::AstType::Named(ref n, _) = param.ty {
                                    if n == tp_name || n == "self" || n == "Self" {
                                        param.ty = crate::parser::ast::AstType::Named(
                                            concrete_ty_name.to_string(),
                                            param.ty.span(),
                                        );
                                    }
                                }
                            } else {
                                // Substitute type params in other param types.
                                let ast_subs: HashMap<String, AstType> = vec![
                                    (tp_name.clone(), crate::parser::ast::AstType::Named(concrete_ty_name.to_string(), method.span)),
                                ].into_iter().collect();
                                param.ty = substitute_ast_type(&param.ty, &ast_subs, &HashMap::new());
                            }
                        }
                        let ast_subs: HashMap<String, AstType> = vec![
                            (tp_name.clone(), crate::parser::ast::AstType::Named(concrete_ty_name.to_string(), method.span)),
                        ].into_iter().collect();
                        renamed.return_ty = substitute_ast_type(&renamed.return_ty, &ast_subs, &HashMap::new());
                        impl_fns.push(renamed);
                    }
                }
                continue; // Skip the normal (non-generic) path below.
            }
        }
        // Normal (non-blanket) impl processing.
        //
        // A target carrying type arguments dispatches on the whole type, so
        // `impl Show for list<i64>` registers `IrType::List(I64)` rather than
        // resolving the bare name `list` to `Infer` and matching nothing.
        let dispatch_ty = match &impl_def.target_ty {
            Some(t) => lower_type_with_structs(t, &module),
            None => type_name_to_ir_type(&impl_def.type_name, &module),
        };
        // The mangled name has to include the arguments, or two impls of one
        // trait at different element types would collide on a single symbol --
        // `Show__list__fmt` for both `list<i64>` and `list<str>`.
        let impl_type_key = match &impl_def.target_ty {
            Some(_) => format!("{}_{}", impl_def.type_name, mangle_ir_type(&dispatch_ty)),
            None => impl_def.type_name.clone(),
        };
        for method in &impl_def.methods {
            let mangled = if impl_def.trait_name.is_empty() {
                // Standalone struct method: `TypeName__method`
                format!("{}__{}", impl_type_key, method.name.name)
            } else {
                // Trait impl: `TraitName__TypeName__method`
                format!(
                    "{}__{}__{}",
                    impl_def.trait_name, impl_type_key, method.name.name
                )
            };
            let mut ret_ty = lower_type_with_structs(&method.return_ty, &module);
            if method.is_async {
                ret_ty = IrType::Chan(Box::new(ret_ty));
            }
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
            // Replace bare `self` param type with the concrete struct type.
            let mut renamed = method.clone();
            renamed.name.name = mangled;
            for param in &mut renamed.params {
                if param.name.name == "self" {
                    // A generic target gives `self` the full type, so
                    // `list_len(self)` inside the impl sees a list rather than
                    // an unresolved name.
                    if let Some(ref t) = impl_def.target_ty {
                        param.ty = t.clone();
                        continue;
                    }
                    if let crate::parser::ast::AstType::Named(ref n, _) = param.ty {
                        if n == "self" {
                            param.ty = crate::parser::ast::AstType::Named(
                                impl_def.type_name.clone(),
                                param.ty.span(),
                            );
                        }
                    }
                }
            }
            impl_fns.push(renamed);
        }
    }
    let trait_dispatch = std::rc::Rc::new(trait_dispatch_map);
    // struct_method_map is only used for mangling; the mangled names are already in fn_sigs.
    let _ = struct_method_map;

    // 3d. Collect extern function declarations so call sites can emit CallExtern.
    for ext in &ast.extern_fns {
        if module.extern_fns.iter().any(|e| e.name == ext.name.name) {
            continue;
        }
        let param_types: Vec<IrType> = ext
            .params
            .iter()
            .map(|p| lower_type_with_structs(&p.ty, &module))
            .collect();
        let ret_ty = lower_type_with_structs(&ext.ret_ty, &module);
        // Register in fn_sigs so lower_call resolves the correct return type.
        fn_sigs.insert(ext.name.name.clone(), ret_ty.clone());
        module.extern_fns.push(crate::ir::module::IrExternFn {
            name: ext.name.name.clone(),
            param_types,
            ret_ty,
            abi: ext.abi.clone(),
            link_lib: ext.link_lib.clone(),
        });
    }

    // 3e. Collect effect declarations and register their operations into fn_sigs and extern_fns.
    for eff in &ast.effects {
        for op in &eff.operations {
            let param_types: Vec<IrType> = op
                .params
                .iter()
                .map(|p| lower_type_with_structs(&p.ty, &module))
                .collect();
            let ret_ty = lower_type_with_structs(&op.ret_ty, &module);
            fn_sigs.insert(op.name.name.clone(), ret_ty.clone());
            if !module.extern_fns.iter().any(|e| e.name == op.name.name) {
                module.extern_fns.push(crate::ir::module::IrExternFn {
                    name: op.name.name.clone(),
                    param_types,
                    ret_ty,
                    abi: Some("C".to_string()),
                    link_lib: None,
                });
            }
        }
    }

    // Shared monomorphization state across all top-level function lowerings.
    let mono_cache = std::rc::Rc::new(std::cell::RefCell::new(std::collections::HashSet::new()));
    let mono_sigs = std::rc::Rc::new(std::cell::RefCell::new(HashMap::new()));
    // Shared lambda counter across all top-level function lowerings ensures unique names.
    let lambda_counter = std::rc::Rc::new(std::cell::Cell::new(0u32));

    // 4. Lower all non-generic function definitions (including impl methods).
    let mut all_lifted: Vec<crate::ir::function::IrFunction> = Vec::new();
    for func in ast.functions.iter().chain(impl_fns.iter()) {
        if !func.type_params.is_empty() {
            continue; // generic: lowered on demand at call sites
        }
        let (ir_func, lifted) = lower_function_with_generics(
            func,
            &module,
            &fn_sigs,
            &const_defs,
            generic_fns.clone(),
            mono_cache.clone(),
            mono_sigs.clone(),
            trait_dispatch.clone(),
            fn_defaults.clone(),
            fn_param_names.clone(),
            fn_param_types.clone(),
            lambda_counter.clone(),
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
        let _ = module.add_function(lf);
    }

    // Post-lowering: scan all functions for struct types used in MakeStruct
    // that aren't in module.struct_defs (e.g. monomorphized generics like
    // Box__i64 created during struct literal lowering). Register them.
    let mut extra_struct_defs: HashMap<String, Vec<(String, IrType)>> = HashMap::new();
    for func in module.functions() {
        for block in func.blocks() {
            for instr in &block.instrs {
                if let IrInstr::MakeStruct { result_ty, .. } = instr {
                    if let IrType::Struct { name, fields } = result_ty {
                        if module.struct_def(name).is_none() && !extra_struct_defs.contains_key(name) {
                            extra_struct_defs.insert(name.clone(), fields.clone());
                        }
                    }
                }
            }
        }
    }
    for (name, fields) in extra_struct_defs {
        let _ = module.add_struct_def(name, fields);
    }

    Ok(module)
}

struct Lowerer<'m> {
    builder: IrFunctionBuilder,
    /// Current lexical scope: name → (ValueId, IrType).
    scope: HashMap<String, (ValueId, IrType)>,
    /// Functions currently being inlined for a taped call, innermost last.
    /// Guards against a recursive taped call expanding forever (#49).
    taped_inline_stack: Vec<String>,
    /// Stack of (header_block, merge_block, loop_var_names, label) for nested loops.
    loop_stack: Vec<(BlockId, BlockId, BlockId, Vec<String>, Option<String>, bool)>,
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
    /// Declared parameter names per function, for resolving named arguments.
    fn_param_names: std::rc::Rc<HashMap<String, Vec<String>>>,
    fn_param_types: std::rc::Rc<HashMap<String, Vec<IrType>>>,
    /// Expected type from a `val x: T = expr` annotation — used by collection
    /// constructors (e.g. `list()`, `map()`) to infer the element/key/value type.
    binding_ty: Option<IrType>,
    /// Expected return type from the enclosing function (set during
    /// `Return` lowering so we can coerce returned struct values into
    /// `dyn Trait` trait objects).
    current_return_ty: Option<IrType>,
    /// Expected type for the current expression being lowered. Set by `val`
    /// and `return` handlers before calling `lower_expr` so that if/else
    /// branches can coerce their results to `dyn Trait` before the merge.
    expected_expr_ty: Option<IrType>,
    /// Primal SSA values that participate in source-level reverse-mode AD.
    taped_values: HashSet<ValueId>,
    /// Mapping from primal SSA values to their tape-node SSA values.
    tape_nodes: HashMap<ValueId, ValueId>,
    /// When lowering inside a handler arm, the name of the resume continuation
    /// binding (e.g. `v` in `k(p) -> resume(v) => v(x)`). When `lower_call`
    /// sees a call to this name, it emits `ResumeCont` instead of a regular
    /// call instruction.
    resume_param_name: Option<String>,
    /// Monomorphized struct defs computed on-the-fly during generic function
    /// lowering.  Checked before `module.struct_def()` in field access.
    local_struct_defs: HashMap<String, Vec<(String, IrType)>>,
    /// Default values for monomorphized structs computed on-the-fly.
    /// Checked alongside `module.struct_defaults()` during struct literal lowering.
    local_struct_defaults: HashMap<String, Vec<Option<crate::parser::ast::AstExpr>>>,
    /// Deferred expressions to emit before each return or at function exit.
    /// Entries are in declaration order; emitted in reverse (LIFO).
    defer_stack: Vec<crate::parser::ast::AstExpr>,
    /// Try-catch state: the catch block to branch to on `?` unwrap failure.
    try_catch_catch_bb: Option<BlockId>,
    /// Try-catch state: the continuation block after the try/catch.
    try_catch_cont_bb: Option<BlockId>,
    /// Try-catch state: the catch parameter name (e.g. `e` in `catch e { ... }`).
    try_catch_param: Option<String>,
}

fn get_qualified_enum_name(base: &AstExpr) -> Option<String> {
    if let AstExpr::FieldAccess {
        base: inner_base,
        field: inner_field,
        ..
    } = base
    {
        if let AstExpr::Ident(inner_ident) = inner_base.as_ref() {
            let resolved_q = resolve_qualifier(&inner_ident.name);
            return Some(format!("{}__{}", resolved_q, inner_field));
        }
    }
    None
}

impl<'m> Lowerer<'m> {
    fn resolve_unqualified_name(&self, name: &str) -> String {
        if self.scope.contains_key(name) {
            return name.to_string();
        }
        let resolved = CURRENT_BRING_PREFIXES.with(|prefixes_stack| {
            if let Some(prefixes) = prefixes_stack.borrow().last() {
                for prefix in prefixes.iter() {
                    let candidate = format!("{}__{}", prefix, name);
                    if self.fn_sigs.contains_key(&candidate)
                        || self.mono_sigs.borrow().contains_key(&candidate)
                        || self.generic_fns.contains_key(&candidate)
                        || self.const_defs.contains_key(&candidate)
                        || self.module.struct_def(&candidate).is_some()
                        || self.module.enum_def(&candidate).is_some()
                        || self.module.type_alias(&candidate).is_some()
                    {
                        return Some(candidate);
                    }
                }
            }
            None
        });
        if let Some(res) = resolved {
            return res;
        }
        if self.fn_sigs.contains_key(name)
            || self.mono_sigs.borrow().contains_key(name)
            || self.generic_fns.contains_key(name)
            || self.const_defs.contains_key(name)
            || self.module.struct_def(name).is_some()
            || self.module.enum_def(name).is_some()
            || self.module.type_alias(name).is_some()
        {
            return name.to_string();
        }
        // Fallback: scan all fn_sigs / const_defs / module registries for any
        // mangled candidate matching `*__name`. This handles transitive brings
        // where a dependency module internally brings another module (e.g.,
        // nn.iris does `bring std.ml` and calls `xavier_init` which becomes
        // `ml__xavier_init` after mangling).
        let suffix = format!("__{}", name);

        // Gather every candidate before choosing, rather than returning the
        // first key a `HashMap` happens to yield.
        //
        // Returning the first match made this resolution depend on the
        // per-process hash seed. With a nested generic, both `Box__i64` and
        // `Box__Box__i64` end with the same suffix, so the winner changed
        // between runs: `wrap(wrap(5))` compiled and ran correctly in about
        // half of attempts and failed type validation in the rest -- 6 of 12
        // when measured. See known-issues #21, and #17 for the same defect
        // class in LICM and capture ordering.
        //
        // Ties are broken by taking the *longest* candidate, then
        // lexicographically. Longest wins because a longer mangled name is the
        // more specific instantiation: resolving `Box__Box__i64` to `Box__i64`
        // loses a level of nesting, which is precisely the mismatch that was
        // being reported.
        let mut candidates: Vec<String> = Vec::new();
        for key in self.fn_sigs.keys() {
            if key.ends_with(&suffix) {
                candidates.push(key.clone());
            }
        }
        for key in self.mono_sigs.borrow().keys() {
            if key.ends_with(&suffix) {
                candidates.push(key.clone());
            }
        }
        for key in self.generic_fns.keys() {
            if key.ends_with(&suffix) {
                candidates.push(key.clone());
            }
        }
        for key in self.const_defs.keys() {
            if key.ends_with(&suffix) {
                candidates.push(key.clone());
            }
        }
        if !candidates.is_empty() {
            candidates.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
            return candidates.remove(0);
        }
        name.to_string()
    }

    fn new_with_lambda_state(
        builder: IrFunctionBuilder,
        module: &'m IrModule,
        fn_sigs: &'m HashMap<String, IrType>,
        lambda_counter: std::rc::Rc<std::cell::Cell<u32>>,
        lifted_fns: std::rc::Rc<std::cell::RefCell<Vec<crate::ir::function::IrFunction>>>,
    ) -> Self {
        Self::new_generic(
            builder,
            module,
            fn_sigs,
            lambda_counter,
            lifted_fns,
            HashMap::new(),
            std::rc::Rc::new(HashMap::new()),
            std::rc::Rc::new(std::cell::RefCell::new(std::collections::HashSet::new())),
            std::rc::Rc::new(std::cell::RefCell::new(HashMap::new())),
            std::rc::Rc::new(HashMap::new()),
            std::rc::Rc::new(HashMap::new()),
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
        fn_param_names: std::rc::Rc<HashMap<String, Vec<String>>>,
        fn_param_types: std::rc::Rc<HashMap<String, Vec<IrType>>>,
    ) -> Self {
        Self {
            builder,
            scope: HashMap::new(),
            taped_inline_stack: Vec::new(),
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
            fn_param_names,
            fn_param_types,
            binding_ty: None,
            current_return_ty: None,
            expected_expr_ty: None,
            taped_values: HashSet::new(),
            tape_nodes: HashMap::new(),
            resume_param_name: None,
            local_struct_defs: HashMap::new(),
            local_struct_defaults: HashMap::new(),
            defer_stack: Vec::new(),
            try_catch_catch_bb: None,
            try_catch_cont_bb: None,
            try_catch_param: None,
        }
    }

    fn is_reverse_diff_scalar(ty: &IrType) -> bool {
        matches!(
            ty,
            IrType::Scalar(
                DType::F32
                    | DType::F64
                    | DType::I32
                    | DType::I64
                    | DType::U8
                    | DType::I8
                    | DType::U32
                    | DType::U64
                    | DType::USize
            ) | IrType::Tensor { .. }
        )
    }

    fn tape_ref_for(&self, value: ValueId) -> ValueId {
        self.tape_nodes.get(&value).copied().unwrap_or(value)
    }

    // `_ty` is the primal's type, no longer used now that the handle is typed
    // `IrType::TapeRef` rather than as its primal. Kept in the signature because
    // callers have it to hand and a future typed-tape (`tape<f32>`) would need it.
    fn ensure_taped_leaf(&mut self, value: ValueId, _ty: &IrType) {
        if self.tape_nodes.contains_key(&value) {
            self.taped_values.insert(value);
            return;
        }

        let tape_result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::TapeRecord {
                result: tape_result,
                value,
                op: "leaf".to_owned(),
                parents: vec![],
            },
            // The handle is a pointer into the runtime tape, not the primal.
            // Typing it as the primal made it emit as `double`, so a taped value
            // crossing a block boundary stopped being a handle (#49).
            Some(IrType::TapeRef),
        );
        self.taped_values.insert(value);
        self.tape_nodes.insert(value, tape_result);
    }

    fn maybe_record_tape_result(
        &mut self,
        primal_result: ValueId,
        primal_ty: &IrType,
        op: &str,
        parents: &[ValueId],
    ) {
        if !Self::is_reverse_diff_scalar(primal_ty)
            || !parents
                .iter()
                .any(|parent| self.taped_values.contains(parent))
        {
            return;
        }

        let tape_result = self.builder.fresh_value();
        let tape_parents = parents
            .iter()
            .copied()
            .map(|parent| self.tape_ref_for(parent))
            .collect();
        self.builder.push_instr(
            IrInstr::TapeRecord {
                result: tape_result,
                value: primal_result,
                op: op.to_owned(),
                parents: tape_parents,
            },
            Some(IrType::TapeRef),
        );
        self.taped_values.insert(primal_result);
        self.tape_nodes.insert(primal_result, tape_result);
    }

    fn resolve_ty(&self, ty: &AstType) -> IrType {
        resolve_ast_type_with_subs(ty, &self.type_param_subs, self.module)
    }

    fn infer_ast_expr_type_simple(&self, expr: &AstExpr) -> IrType {
        match expr {
            AstExpr::IntLit { .. } => IrType::Scalar(DType::I64),
            AstExpr::FloatLit { .. } => IrType::Scalar(DType::F64),
            AstExpr::BoolLit { .. } => IrType::Scalar(DType::Bool),
            AstExpr::StringLit { .. } => IrType::Str,
            AstExpr::Ident(ident) => {
                let resolved_name = self.resolve_unqualified_name(&ident.name);
                if let Some((_, ty)) = self.scope.get(&resolved_name) {
                    ty.clone()
                } else {
                    IrType::Infer
                }
            }
            AstExpr::Tuple { elements, .. } => {
                IrType::Tuple(elements.iter().map(|e| self.infer_ast_expr_type_simple(e)).collect())
            }
            AstExpr::MapLiteral { entries, .. } => {
                if let Some((k, _)) = entries.first() {
                    let k_ty = self.infer_ast_expr_type_simple(k);
                    let v_ty = self.infer_ast_expr_type_simple(&entries[0].1);
                    IrType::Map(Box::new(k_ty), Box::new(v_ty))
                } else {
                    IrType::Map(Box::new(IrType::Str), Box::new(IrType::Scalar(DType::I64)))
                }
            }
            _ => IrType::Infer,
        }
    }

    fn find_matching_monomorphized_struct(&self, base_name: &str, fields: &[(String, IrType)]) -> Option<String> {
        let prefix = format!("{}__", base_name);
        for (struct_name, struct_fields) in &self.module.struct_defs {
            if struct_name.starts_with(&prefix) {
                if struct_fields.len() == fields.len() {
                    let mut matched = true;
                    for (f_name, f_ty) in fields {
                        if let Some((_, def_ty)) = struct_fields.iter().find(|(n, _)| n == f_name) {
                            if *f_ty != IrType::Infer && f_ty != def_ty {
                                matched = false;
                                break;
                            }
                        } else {
                            matched = false;
                            break;
                        }
                    }
                    if matched {
                        return Some(struct_name.clone());
                    }
                }
            }
        }
        None
    }

    /// Looks up a variable and returns its `ValueId` and type.
    fn lookup(&self, ident: &Ident) -> Result<(ValueId, IrType), LowerError> {
        let resolved_name = self.resolve_unqualified_name(&ident.name);
        self.scope.get(&resolved_name).cloned().ok_or_else(|| {
            // Build a combined candidate list: scope names + known function names.
            let scope_names: Vec<&str> = self.scope.keys().map(|s| s.as_str()).collect();
            let fn_names: Vec<&str> = self.fn_sigs.keys().map(|s| s.as_str()).collect();
            let all = scope_names.iter().chain(fn_names.iter()).copied();
            let suggestion = did_you_mean(&ident.name, all);
            LowerError::UndefinedVariable {
                name: ident.name.clone(),
                span: ident.span,
                suggestion,
            }
        })
    }

    /// If `value` is a concrete struct and `expected` is `dyn Trait`, insert a
    /// `MakeTraitObject` and return the new trait-object value. Otherwise
    /// pass through unchanged. Used at return sites where the function
    /// signature dictates the expected trait-object type. The span is used
    /// only for error reporting.
    fn coerce_to_trait_object(
        &mut self,
        value: ValueId,
        ty: IrType,
        expected: &IrType,
        span: crate::parser::lexer::Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        if let IrType::TraitObject { name, methods: exp_methods } = expected {
            // Already a matching trait object: no-op.
            if let IrType::TraitObject { name: existing_name, .. } = &ty {
                if existing_name == name {
                    return Ok((value, ty));
                }
            }
            if let IrType::Struct { name: concrete, .. } = &ty {
                let obj_val = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::MakeTraitObject {
                        result: obj_val,
                        value,
                        target_trait: name.clone(),
                        concrete_ty: concrete.clone(),
                        result_ty: expected.clone(),
                    },
                    Some(expected.clone()),
                );
                return Ok((
                    obj_val,
                    IrType::TraitObject {
                        name: name.clone(),
                        methods: exp_methods.clone(),
                    },
                ));
            } else if matches!(ty, IrType::Infer) {
                // Type inference will resolve the init later.
                return Ok((value, ty));
            } else {
                return Err(LowerError::Unsupported {
                    detail: format!("cannot coerce {} to dyn {}", ty, name),
                    span,
                });
            }
        }
        Ok((value, ty))
    }

    fn lower_expr(&mut self, expr: &AstExpr) -> Result<(ValueId, IrType), LowerError> {
        match expr {
            AstExpr::Ident(ident) => {
                // Special built-in identifiers
                if ident.name == "none" {
                    let result_ty = IrType::Option(Box::new(IrType::Infer));
                    let result = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::MakeNone {
                            result,
                            result_ty: result_ty.clone(),
                        },
                        Some(result_ty.clone()),
                    );
                    return Ok((result, result_ty));
                }
                let resolved_name = self.resolve_unqualified_name(&ident.name);
                // If the ident is not in scope, check if it's a named function —
                // create a first-class function reference via MakeClosure.
                if !self.scope.contains_key(&resolved_name) {
                    if let Some(ret_ty) = self.fn_sigs.get(&resolved_name).cloned() {
                        let fn_ty = IrType::Fn {
                            params: vec![], // param types not tracked in fn_sigs
                            ret: Box::new(ret_ty.clone()),
                        };
                        let result = self.builder.fresh_value();
                        self.builder.push_instr(
                            IrInstr::MakeClosure {
                                result,
                                fn_name: resolved_name,
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
                let ty = IrType::Scalar(DType::F64);
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

                if lhs_ty == IrType::Str && rhs_ty == IrType::Str && matches!(op, AstBinOp::Add) {
                    let result = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::StrConcat { result, lhs: lhs_val, rhs: rhs_val },
                        Some(IrType::Str),
                    );
                    return Ok((result, IrType::Str));
                }

                if lhs_ty == IrType::Str && matches!(op, AstBinOp::Mul) {
                    let count_val = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::Cast { result: count_val, operand: rhs_val, from_ty: rhs_ty.clone(), to_ty: IrType::Scalar(DType::I64) },
                        Some(IrType::Scalar(DType::I64)),
                    );
                    let result = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::StrRepeat { result, operand: lhs_val, count: count_val },
                        Some(IrType::Str),
                    );
                    return Ok((result, IrType::Str));
                }

                if rhs_ty == IrType::Str && matches!(op, AstBinOp::Mul) {
                    let count_val = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::Cast { result: count_val, operand: lhs_val, from_ty: lhs_ty.clone(), to_ty: IrType::Scalar(DType::I64) },
                        Some(IrType::Scalar(DType::I64)),
                    );
                    let result = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::StrRepeat { result, operand: rhs_val, count: count_val },
                        Some(IrType::Str),
                    );
                    return Ok((result, IrType::Str));
                }

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
                    // Exactly one operand is still `Infer`: adopt the other's
                    // concrete type.
                    //
                    // This is *unification* — propagating a known type onto an
                    // unknown one — and deliberately NOT the `i64` defaulting that
                    // docs/architecture-vs-rustc.md §4.1 warns against. Nothing is
                    // invented here: if both sides are unknown the arm does not
                    // match and the pair stays `Infer` for HmTypeInferPass to
                    // solve.
                    //
                    // The operand keeps its own inference slot; setting the BinOp's
                    // type merely records the constraint. HmTypeInferPass already
                    // unifies BinOp lhs/rhs (`type_infer_hm.rs`) and already threads
                    // list element types through ListNew/ListPush/ListGet, so a
                    // loop variable from
                    //     val xs = list(); push(xs, 1); for x in xs { sum + x }
                    // resolves to the *pushed element type* rather than a default.
                    // Rejecting it during lowering discarded a program the very
                    // next pass would have typed correctly.
                    (IrType::Infer, concrete) if !matches!(concrete, IrType::Infer) => {
                        let adopted = rhs_ty.clone();
                        (lhs_val, rhs_val, adopted)
                    }
                    (concrete, IrType::Infer) if !matches!(concrete, IrType::Infer) => {
                        let adopted = lhs_ty.clone();
                        (lhs_val, rhs_val, adopted)
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

                // Phase 86: operator overloading for struct types.
                // Check if lhs is a struct and there's a matching operator impl.
                if let IrType::Struct {
                    name: struct_name, ..
                } = &lhs_ty
                {
                    let trait_method = op_trait_method(*op);
                    if let Some((trait_name, method_name)) = trait_method {
                        let mangled = format!("{}__{}__{}", trait_name, struct_name, method_name);
                        if let Some(ret_ty) = self.fn_sigs.get(&mangled).cloned() {
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::Call {
                                    result: Some(result),
                                    callee: mangled,
                                    args: vec![lhs_val, rhs_val],
                                    result_ty: Some(ret_ty.clone()),
                                },
                                Some(ret_ty.clone()),
                            );
                            return Ok((result, ret_ty));
                        }
                    }
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
                let tape_op = match ir_op {
                    BinOp::Add => Some("add"),
                    BinOp::Sub => Some("sub"),
                    BinOp::Mul => Some("mul"),
                    BinOp::Div => Some("div"),
                    _ => None,
                };
                if let Some(tape_op) = tape_op {
                    self.maybe_record_tape_result(result, &result_ty, tape_op, &[lhs_val, rhs_val]);
                }
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
                if matches!(ir_op, ScalarUnaryOp::Neg) {
                    self.maybe_record_tape_result(result, &ty, "neg", &[val]);
                }
                Ok((result, ty))
            }

            AstExpr::Call { callee, args, named_args, span } => {
                self.lower_call(callee, args, named_args, *span)
            }

            AstExpr::If {
                cond,
                then_block,
                else_block,
                span,
            } => self.lower_if_expr(cond, then_block, else_block.as_ref(), *span),

            AstExpr::Block(block) => {
                if let Some(v) = self.lower_block(block)? {
                    Ok(v)
                } else if let Some((param_id, param_ty)) = self.builder.current_block_first_param() {
                    // Block has no tail but the current block has a block param
                    // (e.g. from `?` unwrapping). Use it as the block's value.
                    Ok((param_id, param_ty))
                } else {
                    // Block was terminated early (e.g. break/return/continue).
                    // The value is only used to satisfy type requirements; the
                    // current block is already sealed, so we cannot emit more
                    // instructions.
                    Ok((self.builder.fresh_value(), IrType::Scalar(DType::I64)))
                }
            }

            AstExpr::Mask { body, .. } => {
                // Effect mask: lower the body as if it were a normal block.
                if let Some(v) = self.lower_block(body)? {
                    Ok(v)
                } else if let Some((param_id, param_ty)) = self.builder.current_block_first_param() {
                    Ok((param_id, param_ty))
                } else {
                    Ok((self.builder.fresh_value(), IrType::Scalar(DType::I64)))
                }
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

            AstExpr::StructLit { name, fields, spread, span } => {
                let mut resolved_name = resolve_brought_name(name, self.module);
                // Always check if binding_ty provides a more specific (mangled) name
                // like "Box__i64" even when the base template "Box" exists.
                if let Some(IrType::Struct { name: expected_name, .. }) = &self.binding_ty {
                    if expected_name.starts_with(&format!("{}__", name)) {
                        resolved_name = expected_name.clone();
                    }
                }
                if self.module.struct_def(&resolved_name).is_none() {
                    let mut found = false;
                    if let Some(IrType::Struct { name: expected_name, .. }) = &self.binding_ty {
                        if expected_name == name || expected_name.starts_with(&format!("{}__", name)) {
                            resolved_name = expected_name.clone();
                            found = true;
                        }
                    }
                    if !found {
                        let mut lowered_fields = Vec::new();
                        for (f_name, f_expr) in fields {
                            let f_ty = self.infer_ast_expr_type_simple(f_expr);
                            lowered_fields.push((f_name.clone(), f_ty));
                        }
                        if let Some(matched_name) = self.find_matching_monomorphized_struct(name, &lowered_fields) {
                            resolved_name = matched_name;
                        }
                    }
                }

                // Look up the struct definition.
                let mut struct_fields = self
                    .module
                    .struct_def(&resolved_name)
                    .ok_or_else(|| LowerError::UndefinedVariable {
                        name: name.clone(),
                        span: *span,
                        suggestion: None,
                    })?
                    .clone();

                // When inside a monomorphized generic function, resolve template
                // field types using type_param_subs (e.g. Struct { name: "T" } → Scalar(I64)).
                // Also compute the mangled struct name.
                if !self.type_param_subs.is_empty() {
                    for (_fname, fty) in &mut struct_fields {
                        *fty = resolve_concrete_field(fty, &self.type_param_subs, self.module);
                    }
                    // Compute mangled name: e.g. "MinHeap" → "MinHeap__i64"
                    //
                    // Ordered by type-parameter name. `type_param_subs` is a
                    // `HashMap`, and reading `.values()` directly built the
                    // mangled name in hash order, so the same program produced
                    // `unwrapw__i64_Box` on one run and `unwrapw__Box_i64` on
                    // the next -- and the definition and use sites then
                    // disagreed about the name. Measured 5 of 10 runs passing.
                    // See known-issues #21.
                    let mut mangle_keys: Vec<&String> = self.type_param_subs.keys().collect();
                    mangle_keys.sort();
                    let mangle = mangle_keys
                        .iter()
                        .filter_map(|k| self.type_param_subs.get(*k))
                        .map(mangle_ir_type)
                        .collect::<Vec<_>>()
                        .join("_");
                    if !mangle.is_empty() {
                        resolved_name = format!("{}__{}", resolved_name, mangle);
                    }
                }

                // If spread is present, evaluate it to get the source struct.
                let spread_val = if let Some(spread_expr) = spread {
                    let (sv, st) = self.lower_expr(spread_expr)?;
                    Some((sv, st))
                } else {
                    None
                };

                // Lower each field expression in declaration order.
                let mut field_vals = Vec::with_capacity(struct_fields.len());
                let mut field_actual_tys = Vec::with_capacity(struct_fields.len());
                // Look up field defaults for this struct (by resolved name).
                let field_defaults = self.local_struct_defaults.get(&resolved_name)
                    .or_else(|| self.module.struct_defaults.get(&resolved_name))
                    .cloned();
                for (field_idx, (field_name, field_ty)) in struct_fields.iter().enumerate() {
                    if let Some(provided) =
                        fields
                            .iter()
                            .find(|(n, _)| n == field_name)
                    {
                        // Propagate the struct field type down as binding_ty so `list()` can infer its type.
                        let prev_binding_ty = self.binding_ty.take();
                        self.binding_ty = Some(field_ty.clone());
                        let (val, actual_ty) = self.lower_expr(&provided.1)?;
                        self.binding_ty = prev_binding_ty;
                        field_vals.push(val);
                        field_actual_tys.push(actual_ty);
                    } else if let Some((spread_val_id, spread_ty)) = &spread_val {
                        // Copy field from spread source struct via GetField.
                        let spread_struct_fields = match spread_ty {
                            IrType::Struct { fields, .. } => fields.clone(),
                            _ => {
                                return Err(LowerError::Unsupported {
                                    detail: format!("struct update source is not a struct type"),
                                    span: *span,
                                });
                            }
                        };
                        let field_index = spread_struct_fields
                            .iter()
                            .position(|(n, _)| n == field_name)
                            .ok_or_else(|| LowerError::Unsupported {
                                detail: format!("spread source struct has no field '{}'", field_name),
                                span: *span,
                            })?;
                        let result = self.builder.fresh_value();
                        self.builder.push_instr(
                            IrInstr::GetField {
                                result,
                                base: *spread_val_id,
                                field_index,
                                result_ty: field_ty.clone(),
                            },
                            Some(field_ty.clone()),
                        );
                        field_vals.push(result);
                        field_actual_tys.push(field_ty.clone());
                    } else if let Some(defaults) = &field_defaults {
                        if let Some(Some(default_expr)) = defaults.get(field_idx) {
                            // Use the default value for this field.
                            let prev_binding_ty = self.binding_ty.take();
                            self.binding_ty = Some(field_ty.clone());
                            let (val, actual_ty) = self.lower_expr(default_expr)?;
                            self.binding_ty = prev_binding_ty;
                            field_vals.push(val);
                            field_actual_tys.push(actual_ty);
                        } else {
                            return Err(LowerError::Unsupported {
                                detail: format!("missing field '{}' in struct literal", field_name),
                                span: *span,
                            });
                        }
                    } else {
                        return Err(LowerError::Unsupported {
                            detail: format!("missing field '{}' in struct literal", field_name),
                            span: *span,
                        });
                    }
                }

                // When not inside a monomorphized function, try to infer type params
                // from template fields vs actual field value types. E.g. Box { value: 42 }
                // has template field ("value", Struct{name:"T"}) and actual type Scalar(I64),
                // so we infer T=i64, resolve the fields, and mangle the name to Box__i64.
                if self.type_param_subs.is_empty() && resolved_name == *name && spread_val.is_none() {
                    let mut inferred_subs: std::collections::HashMap<String, IrType> = std::collections::HashMap::new();
                    for ((_, template_ty), actual_ty) in struct_fields.iter().zip(field_actual_tys.iter()) {
                        if let IrType::Struct { name: pname, fields: pfields } = template_ty {
                            if pfields.is_empty() {
                                inferred_subs.entry(pname.clone()).or_insert_with(|| actual_ty.clone());
                            }
                        }
                    }
                    if !inferred_subs.is_empty() {
                        for (_, fty) in &mut struct_fields {
                            *fty = resolve_concrete_field(fty, &inferred_subs, self.module);
                        }
                        // Same ordering rule as above -- sorted by type-parameter
                        // name, so the name built here matches the one the use
                        // site computes. See known-issues #21.
                        let mut inf_keys: Vec<&String> = inferred_subs.keys().collect();
                        inf_keys.sort();
                        let mangle = inf_keys
                            .iter()
                            .filter_map(|k| inferred_subs.get(*k))
                            .map(mangle_ir_type)
                            .collect::<Vec<_>>()
                            .join("_");
                        if !mangle.is_empty() {
                            resolved_name = format!("{}__{}", name, mangle);
                            self.local_struct_defs.insert(resolved_name.clone(), struct_fields.clone());
                            // Carry over defaults from the template.
                            if let Some(template_defaults) = self.module.struct_defaults.get(name).cloned() {
                                self.local_struct_defaults.insert(resolved_name.clone(), template_defaults);
                            }
                        }
                    }
                }

                let result_ty = IrType::Struct {
                    name: resolved_name,
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
                    // Check if this is a qualified constant lookup
                    if !self.scope.contains_key(&base_ident.name) {
                        let resolved_q = resolve_qualifier(&base_ident.name);
                        let mangled_const = format!("{}__{}", resolved_q, field);
                        if let Some(val_ty) = self.scope.get(&mangled_const).cloned() {
                            return Ok(val_ty);
                        }
                    }

                    let resolved_enum = self.resolve_unqualified_name(&base_ident.name);
                    if let Some(variants) = self.module.enum_def(&resolved_enum) {
                        let variants = variants.clone();
                        let variant_idx =
                            variants.iter().position(|v| v == field).ok_or_else(|| {
                                LowerError::Unsupported {
                                    detail: format!(
                                        "no variant '{}' in enum '{}'",
                                        field, resolved_enum
                                    ),
                                    span: *span,
                                }
                            })?;
                        let result_ty = IrType::Enum {
                            name: resolved_enum,
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

                // Check if base is a qualified enum name (e.g. `utils.Shape.Circle`)
                if let Some(enum_name) = get_qualified_enum_name(base.as_ref()) {
                    if let Some(variants) = self.module.enum_def(&enum_name) {
                        let variants = variants.clone();
                        let variant_idx =
                            variants.iter().position(|v| v == field).ok_or_else(|| {
                                LowerError::Unsupported {
                                    detail: format!(
                                        "no variant '{}' in enum '{}'",
                                        field, enum_name
                                    ),
                                    span: *span,
                                }
                            })?;
                        let result_ty = IrType::Enum {
                            name: enum_name,
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
                        (
                            IrInstr::GradValue {
                                result,
                                operand: base_val,
                                ty: inner_ty.clone(),
                            },
                            inner_ty,
                        )
                    } else if field == "grad" || field == "tangent" {
                        (
                            IrInstr::GradTangent {
                                result,
                                operand: base_val,
                                ty: inner_ty.clone(),
                            },
                            inner_ty,
                        )
                    } else {
                        return Err(LowerError::Unsupported {
                            detail: format!(
                                "grad<T> has no field '{}'; use .value or .grad",
                                field
                            ),
                            span: *span,
                        });
                    };
                    self.builder.push_instr(instr, Some(ret_ty.clone()));
                    return Ok((result, ret_ty));
                }
                let struct_fields = match &base_ty {
                    IrType::Struct { name: s_name, fields } => {
                        if let Some(def_fields) = self.local_struct_defs.get(s_name) {
                            def_fields.clone()
                        } else if let Some(def_fields) = self.module.struct_def(s_name) {
                            def_fields.clone()
                        } else {
                            fields.clone()
                        }
                    }
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

            AstExpr::Lambda { params, body, span } => self.lower_lambda(params, body, *span),

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

            // await expr: expect a channel and recv its value
            AstExpr::Await { expr, span } => {
                let (val, ty) = self.lower_expr(expr)?;
                match ty {
                    IrType::Chan(elem) => {
                        let elem_ty = (*elem).clone();
                        let result = self.builder.fresh_value();
                        self.builder.push_instr(
                            IrInstr::ChanRecv {
                                result,
                                chan: val,
                                elem_ty: elem_ty.clone(),
                            },
                            Some(elem_ty.clone()),
                        );
                        Ok((result, elem_ty))
                    }
                    other => Err(LowerError::TypeMismatch {
                        expected: "chan<T>".to_owned(),
                        found: format!("{}", other),
                        span: *span,
                    }),
                }
            }

            AstExpr::Try { expr, span } => {
                let (val, res_ty) = self.lower_expr(expr)?;

                // Determine if this is a result or option type and branch accordingly.
                if let IrType::ResultType(ok, err) = &res_ty {
                    let ok_ty = (**ok).clone();
                    let err_ty = (**err).clone();

                    // Emit IsOk test.
                    let is_ok_result = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::IsOk {
                            result: is_ok_result,
                            operand: val,
                        },
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
                        IrInstr::ResultUnwrap {
                            result: ok_unwrapped,
                            operand: val,
                            result_ty: ok_ty.clone(),
                        },
                        Some(ok_ty.clone()),
                    );
                    self.builder.push_instr(
                        IrInstr::Br {
                            target: cont_bb,
                            args: vec![ok_unwrapped],
                        },
                        None,
                    );

                    // Err branch: early return or jump to catch block.
                    self.builder.set_current_block(err_bb);
                    let err_unwrapped = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::ResultUnwrapErr {
                            result: err_unwrapped,
                            operand: val,
                            result_ty: err_ty.clone(),
                        },
                        Some(err_ty.clone()),
                    );
                    if let (Some(catch_bb), Some(_cont_bb), Some(ref _catch_param)) =
                        (self.try_catch_catch_bb, self.try_catch_cont_bb, self.try_catch_param.clone())
                    {
                        // Inside try/catch: jump to catch block with error value.
                        self.builder.push_instr(
                            IrInstr::Br {
                                target: catch_bb,
                                args: vec![err_unwrapped],
                            },
                            None,
                        );
                    } else {
                        // Outside try/catch: early return with MakeErr.
                        let err_result = self.builder.fresh_value();
                        let err_ret_ty =
                            IrType::ResultType(Box::new(IrType::Infer), Box::new(err_ty.clone()));
                        self.builder.push_instr(
                            IrInstr::MakeErr {
                                result: err_result,
                                value: err_unwrapped,
                                result_ty: err_ret_ty.clone(),
                            },
                            Some(err_ret_ty.clone()),
                        );
                        self.builder.push_instr(
                            IrInstr::Return {
                                values: vec![err_result],
                            },
                            None,
                        );
                    }

                    // Continuation block: receives the Ok value.
                    self.builder.set_current_block(cont_bb);
                    let ok_result =
                        self.builder
                            .add_block_param(cont_bb, Some("try_result"), ok_ty.clone());
                    let _ = span;
                    Ok((ok_result, ok_ty))
                } else if let IrType::Option(inner) = &res_ty {
                    let ok_ty = (**inner).clone();

                    // Emit IsSome test.
                    let is_some_result = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::IsSome {
                            result: is_some_result,
                            operand: val,
                        },
                        Some(IrType::Scalar(DType::Bool)),
                    );

                    let some_bb = self.builder.create_block(Some("try_some"));
                    let none_bb = self.builder.create_block(Some("try_none"));
                    let cont_bb = self.builder.create_block(Some("try_cont"));

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

                    // Some branch: unwrap and continue.
                    self.builder.set_current_block(some_bb);
                    let some_unwrapped = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::OptionUnwrap {
                            result: some_unwrapped,
                            operand: val,
                            result_ty: ok_ty.clone(),
                        },
                        Some(ok_ty.clone()),
                    );
                    self.builder.push_instr(
                        IrInstr::Br {
                            target: cont_bb,
                            args: vec![some_unwrapped],
                        },
                        None,
                    );

                    // None branch: early return with MakeNone.
                    self.builder.set_current_block(none_bb);
                    let none_result = self.builder.fresh_value();
                    let none_ret_ty =
                        IrType::Option(Box::new(ok_ty.clone()));
                    self.builder.push_instr(
                        IrInstr::MakeNone {
                            result: none_result,
                            result_ty: none_ret_ty.clone(),
                        },
                        Some(none_ret_ty.clone()),
                    );
                    self.builder.push_instr(
                        IrInstr::Return {
                            values: vec![none_result],
                        },
                        None,
                    );

                    // Continuation block: receives the Some value.
                    self.builder.set_current_block(cont_bb);
                    let ok_result =
                        self.builder
                            .add_block_param(cont_bb, Some("try_result"), ok_ty.clone());
                    let _ = span;
                    Ok((ok_result, ok_ty))
                } else {
                    let _ = span;
                    Ok((val, res_ty))
                }
            }

            AstExpr::MethodCall {
                base,
                method,
                args,
                span,
            } => {
                // Check if base is a bare identifier naming an enum → variant construction with data.
                // e.g. `Shape.Circle(3.14)` is parsed as MethodCall(base=Ident("Shape"), method="Circle", args=[3.14])
                // Also handle module-qualified calls: `utils.normalize(x)` where `utils` is not a
                // local variable but `normalize` is a known function (imported via `bring`).
                if let AstExpr::Ident(base_ident) = base.as_ref() {
                    let resolved_q = resolve_qualifier(&base_ident.name);
                    let base_resolved_enum = self.resolve_unqualified_name(&base_ident.name);
                    // Module-qualified call: base not in scope, but method is a known function.
                    if !self.scope.contains_key(&base_ident.name)
                        && self.module.enum_def(&base_resolved_enum).is_none()
                        && self.module.struct_def(&base_resolved_enum).is_none()
                    {
                        let mangled_fn = format!("{}__{}", resolved_q, method);
                        // For generic functions, delegate to lower_call which handles monomorphization.
                        if self.generic_fns.contains_key(mangled_fn.as_str()) {
                            let synthetic_callee = Ident {
                                name: mangled_fn,
                                span: base_ident.span,
                            };
                            return self.lower_call(&synthetic_callee, args, &[], *span);
                        }
                        let ret_ty =
                            self.fn_sigs.get(mangled_fn.as_str()).cloned().or_else(|| {
                                self.mono_sigs.borrow().get(mangled_fn.as_str()).cloned()
                            });
                        if let Some(ret_ty) = ret_ty {
                            let mut arg_vals = Vec::with_capacity(args.len());
                            for arg in args {
                                let (v, _) = self.lower_expr(arg)?;
                                arg_vals.push(v);
                            }
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::Call {
                                    result: Some(result),
                                    callee: mangled_fn,
                                    args: arg_vals,
                                    result_ty: Some(ret_ty.clone()),
                                },
                                Some(ret_ty.clone()),
                            );
                            return Ok((result, ret_ty));
                        }

                        let unqualified_mangled_fn = self.resolve_unqualified_name(method);
                        // For generic functions, delegate to lower_call which handles monomorphization.
                        if self.generic_fns.contains_key(unqualified_mangled_fn.as_str()) {
                            let synthetic_callee = Ident {
                                name: unqualified_mangled_fn,
                                span: base_ident.span,
                            };
                            return self.lower_call(&synthetic_callee, args, &[], *span);
                        }
                        let ret_ty = self
                            .fn_sigs
                            .get(unqualified_mangled_fn.as_str())
                            .cloned()
                            .or_else(|| {
                                self.mono_sigs
                                    .borrow()
                                    .get(unqualified_mangled_fn.as_str())
                                    .cloned()
                            });
                        if let Some(ret_ty) = ret_ty {
                            let mut arg_vals = Vec::with_capacity(args.len());
                            for arg in args {
                                let (v, _) = self.lower_expr(arg)?;
                                arg_vals.push(v);
                            }
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::Call {
                                    result: Some(result),
                                    callee: unqualified_mangled_fn,
                                    args: arg_vals,
                                    result_ty: Some(ret_ty.clone()),
                                },
                                Some(ret_ty.clone()),
                            );
                            return Ok((result, ret_ty));
                        }
                    }
                    if let Some(variants) = self.module.enum_def(&base_resolved_enum) {
                        let variants = variants.clone();
                        if let Some(variant_idx) = variants.iter().position(|v| v == method) {
                            // This is an enum variant constructor with data.
                            let mut field_vals = Vec::with_capacity(args.len());
                            for arg in args {
                                let (v, _) = self.lower_expr(arg)?;
                                field_vals.push(v);
                            }
                            let result_ty = IrType::Enum {
                                name: base_resolved_enum,
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

                // Check if base is a qualified enum name (e.g. `utils.Shape.Circle(x)`)
                if let Some(enum_name) = get_qualified_enum_name(base.as_ref()) {
                    if let Some(variants) = self.module.enum_def(&enum_name) {
                        let variants = variants.clone();
                        if let Some(variant_idx) = variants.iter().position(|v| v == method) {
                            let mut field_vals = Vec::with_capacity(args.len());
                            for arg in args {
                                let (v, _) = self.lower_expr(arg)?;
                                field_vals.push(v);
                            }
                            let result_ty = IrType::Enum {
                                name: enum_name,
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
                        "map" => return self.lower_list_map(base_val, elem_ty, args, *span),
                        "filter" => return self.lower_list_filter(base_val, elem_ty, args, *span),
                        "fold" => return self.lower_list_fold(base_val, elem_ty, args, *span),
                        "any" => return self.lower_list_any(base_val, elem_ty, args, *span),
                        "all" => return self.lower_list_all(base_val, elem_ty, args, *span),
                        "len" => {
                            // lst.len() → ListLen
                            let result = self.builder.fresh_value();
                            let ty = IrType::Scalar(DType::I64);
                            self.builder.push_instr(
                                IrInstr::ListLen { result, list: base_val },
                                Some(ty.clone()),
                            );
                            return Ok((result, ty));
                        }
                        "push" => {
                            // lst.push(v) → list_push(lst, v) → returns unit
                            if args.len() != 1 {
                                return Err(LowerError::Unsupported {
                                    detail: "push() requires exactly 1 argument".into(),
                                    span: *span,
                                });
                            }
                            let (v, _) = self.lower_expr(&args[0])?;
                            self.builder.push_instr(
                                IrInstr::ListPush { list: base_val, value: v },
                                None,
                            );
                            // Return a dummy i64 zero as unit value
                            let zero = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::ConstInt { result: zero, value: 0, ty: IrType::Scalar(DType::I64) },
                                Some(IrType::Scalar(DType::I64)),
                            );
                            return Ok((zero, IrType::Scalar(DType::I64)));
                        }
                        "pop" => {
                            // lst.pop() → list_pop(lst) → returns option<elem>
                            let result = self.builder.fresh_value();
                            let ret_ty = IrType::Option(Box::new(elem_ty.clone()));
                            self.builder.push_instr(
                                IrInstr::ListPop { result, list: base_val, elem_ty: elem_ty.clone() },
                                Some(ret_ty.clone()),
                            );
                            return Ok((result, ret_ty));
                        }
                        "get" => {
                            // lst.get(i) → list_get(lst, i) → returns elem
                            if args.len() != 1 {
                                return Err(LowerError::Unsupported {
                                    detail: "get() requires exactly 1 argument".into(),
                                    span: *span,
                                });
                            }
                            let (idx, _) = self.lower_expr(&args[0])?;
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::ListGet { result, list: base_val, index: idx, elem_ty: elem_ty.clone() },
                                Some(elem_ty.clone()),
                            );
                            return Ok((result, elem_ty));
                        }
                        "set" => {
                            // lst.set(i, v) → list_set(lst, i, v) → returns unit
                            if args.len() != 2 {
                                return Err(LowerError::Unsupported {
                                    detail: "set() requires exactly 2 arguments".into(),
                                    span: *span,
                                });
                            }
                            let (idx, _) = self.lower_expr(&args[0])?;
                            let (v, _) = self.lower_expr(&args[1])?;
                            self.builder.push_instr(
                                IrInstr::ListSet { list: base_val, index: idx, value: v },
                                None,
                            );
                            // Return a dummy i64 zero as unit value
                            let zero = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::ConstInt { result: zero, value: 0, ty: IrType::Scalar(DType::I64) },
                                Some(IrType::Scalar(DType::I64)),
                            );
                            return Ok((zero, IrType::Scalar(DType::I64)));
                        }
                        _ => {} // fall through to struct method dispatch
                    }
                }

                // Option<T> method dispatch.
                if let IrType::Option(inner_ty) = &base_ty {
                    let inner = (**inner_ty).clone();
                    match method.as_str() {
                        "is_some" => {
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::IsSome { result, operand: base_val },
                                Some(IrType::Scalar(DType::Bool)),
                            );
                            return Ok((result, IrType::Scalar(DType::Bool)));
                        }
                        "is_none" => {
                            let is_some_result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::IsSome { result: is_some_result, operand: base_val },
                                Some(IrType::Scalar(DType::Bool)),
                            );
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::UnaryOp { result, op: ScalarUnaryOp::Not, operand: is_some_result, ty: IrType::Scalar(DType::Bool) },
                                Some(IrType::Scalar(DType::Bool)),
                            );
                            return Ok((result, IrType::Scalar(DType::Bool)));
                        }
                        "unwrap" => {
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::OptionUnwrap { result, operand: base_val, result_ty: inner.clone() },
                                Some(inner.clone()),
                            );
                            return Ok((result, inner));
                        }
                        "unwrap_or" => {
                            // opt.unwrap_or(default) → emulate with conditional via call to unwrap_or builtin
                            if args.len() != 1 {
                                return Err(LowerError::Unsupported {
                                    detail: "unwrap_or() requires exactly 1 argument".into(),
                                    span: *span,
                                });
                            }
                            // Build inline conditional: if is_some(opt) { opt.unwrap() } else { default }
                            let (default_val, _default_ty) = self.lower_expr(&args[0])?;
                            let is_some_result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::IsSome { result: is_some_result, operand: base_val },
                                Some(IrType::Scalar(DType::Bool)),
                            );
                            let then_bb = self.builder.create_block(Some("unwrap_or_then"));
                            let else_bb = self.builder.create_block(Some("unwrap_or_else"));
                            let merge_bb = self.builder.create_block(Some("unwrap_or_merge"));
                            self.builder.push_instr(
                                IrInstr::CondBr {
                                    cond: is_some_result,
                                    then_block: then_bb,
                                    then_args: vec![],
                                    else_block: else_bb,
                                    else_args: vec![],
                                },
                                None,
                            );
                            // Then branch: opt.unwrap()
                            self.builder.set_current_block(then_bb);
                            let unwrapped = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::OptionUnwrap { result: unwrapped, operand: base_val, result_ty: inner.clone() },
                                Some(inner.clone()),
                            );
                            let then_result = unwrapped;
                            self.builder.push_instr(
                                IrInstr::Br { target: merge_bb, args: vec![then_result] },
                                None,
                            );
                            // Else branch: default
                            self.builder.set_current_block(else_bb);
                            let else_result = default_val;
                            self.builder.push_instr(
                                IrInstr::Br { target: merge_bb, args: vec![else_result] },
                                None,
                            );
                            // Merge: phi
                            let merge_result = self.builder.add_block_param(merge_bb, Some("unwrap_or_res"), inner.clone());
                            self.builder.set_current_block(merge_bb);
                            return Ok((merge_result, inner));
                        }
                        "map" => {
                            // opt.map(f) → emulate: if is_some(opt) { some(f(opt.unwrap())) } else { none() }
                            if args.len() != 1 {
                                return Err(LowerError::Unsupported {
                                    detail: "map() requires exactly 1 argument".into(),
                                    span: *span,
                                });
                            }
                            let (f_val, f_ty) = self.lower_expr(&args[0])?;
                            // Get the function return type from the closure type
                            let ret_ty = if let IrType::Fn { ret, .. } = &f_ty {
                                (**ret).clone()
                            } else {
                                IrType::Infer
                            };
                            let is_some_result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::IsSome { result: is_some_result, operand: base_val },
                                Some(IrType::Scalar(DType::Bool)),
                            );
                            let then_bb = self.builder.create_block(Some("opt_map_then"));
                            let else_bb = self.builder.create_block(Some("opt_map_else"));
                            let merge_bb = self.builder.create_block(Some("opt_map_merge"));
                            self.builder.push_instr(
                                IrInstr::CondBr {
                                    cond: is_some_result,
                                    then_block: then_bb,
                                    then_args: vec![],
                                    else_block: else_bb,
                                    else_args: vec![],
                                },
                                None,
                            );
                            // Then: call f(opt.unwrap())
                            self.builder.set_current_block(then_bb);
                            let unwrapped = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::OptionUnwrap { result: unwrapped, operand: base_val, result_ty: inner.clone() },
                                Some(inner.clone()),
                            );
                            // Call the closure
                            let call_result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::CallClosure {
                                    result: Some(call_result),
                                    closure: f_val,
                                    args: vec![unwrapped],
                                    result_ty: ret_ty.clone(),
                                    pass_env: true,
                                },
                                Some(ret_ty.clone()),
                            );
                            // Wrap in Some(...)
                            let some_result = self.builder.fresh_value();
                            let some_ty = IrType::Option(Box::new(ret_ty.clone()));
                            self.builder.push_instr(
                                IrInstr::MakeSome { result: some_result, value: call_result, result_ty: some_ty.clone() },
                                Some(some_ty.clone()),
                            );
                            self.builder.push_instr(
                                IrInstr::Br { target: merge_bb, args: vec![some_result] },
                                None,
                            );
                            // Else: None
                            self.builder.set_current_block(else_bb);
                            let none_result = self.builder.fresh_value();
                            let none_ty = IrType::Option(Box::new(ret_ty.clone()));
                            self.builder.push_instr(
                                IrInstr::MakeNone { result: none_result, result_ty: none_ty.clone() },
                                Some(none_ty.clone()),
                            );
                            self.builder.push_instr(
                                IrInstr::Br { target: merge_bb, args: vec![none_result] },
                                None,
                            );
                            // Merge
                            let merge_result = self.builder.add_block_param(merge_bb, Some("opt_map_res"), none_ty.clone());
                            self.builder.set_current_block(merge_bb);
                            return Ok((merge_result, none_ty));
                        }
                        _ => {} // fall through
                    }
                }

                // Result<T, E> method dispatch.
                if let IrType::ResultType(ok_ty, err_ty) = &base_ty {
                    let ok = (**ok_ty).clone();
                    let err = (**err_ty).clone();
                    match method.as_str() {
                        "is_ok" => {
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::IsOk { result, operand: base_val },
                                Some(IrType::Scalar(DType::Bool)),
                            );
                            return Ok((result, IrType::Scalar(DType::Bool)));
                        }
                        "is_err" => {
                            let is_ok_result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::IsOk { result: is_ok_result, operand: base_val },
                                Some(IrType::Scalar(DType::Bool)),
                            );
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::UnaryOp { result, op: ScalarUnaryOp::Not, operand: is_ok_result, ty: IrType::Scalar(DType::Bool) },
                                Some(IrType::Scalar(DType::Bool)),
                            );
                            return Ok((result, IrType::Scalar(DType::Bool)));
                        }
                        "unwrap" => {
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::ResultUnwrap { result, operand: base_val, result_ty: ok.clone() },
                                Some(ok.clone()),
                            );
                            return Ok((result, ok));
                        }
                        "unwrap_err" => {
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::ResultUnwrapErr { result, operand: base_val, result_ty: err.clone() },
                                Some(err.clone()),
                            );
                            return Ok((result, err));
                        }
                        "unwrap_or" => {
                            if args.len() != 1 {
                                return Err(LowerError::Unsupported {
                                    detail: "unwrap_or() requires exactly 1 argument".into(),
                                    span: *span,
                                });
                            }
                            // Build inline conditional: if is_ok(res) { res.unwrap() } else { default }
                            let (default_val, _) = self.lower_expr(&args[0])?;
                            let is_ok_result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::IsOk { result: is_ok_result, operand: base_val },
                                Some(IrType::Scalar(DType::Bool)),
                            );
                            let then_bb = self.builder.create_block(Some("res_unwrap_or_then"));
                            let else_bb = self.builder.create_block(Some("res_unwrap_or_else"));
                            let merge_bb = self.builder.create_block(Some("res_unwrap_or_merge"));
                            self.builder.push_instr(
                                IrInstr::CondBr {
                                    cond: is_ok_result,
                                    then_block: then_bb,
                                    then_args: vec![],
                                    else_block: else_bb,
                                    else_args: vec![],
                                },
                                None,
                            );
                            // Then branch: res.unwrap()
                            self.builder.set_current_block(then_bb);
                            let unwrapped = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::ResultUnwrap { result: unwrapped, operand: base_val, result_ty: ok.clone() },
                                Some(ok.clone()),
                            );
                            self.builder.push_instr(
                                IrInstr::Br { target: merge_bb, args: vec![unwrapped] },
                                None,
                            );
                            // Else branch: default
                            self.builder.set_current_block(else_bb);
                            self.builder.push_instr(
                                IrInstr::Br { target: merge_bb, args: vec![default_val] },
                                None,
                            );
                            // Merge: phi
                            let merge_result = self.builder.add_block_param(merge_bb, Some("res_unwrap_or_res"), ok.clone());
                            self.builder.set_current_block(merge_bb);
                            return Ok((merge_result, ok));
                        }
                        _ => {} // fall through
                    }
                }

                // str method dispatch.
                if matches!(base_ty, IrType::Str) {
                    match method.as_str() {
                        "len" => {
                            let result = self.builder.fresh_value();
                            let ty = IrType::Scalar(DType::I64);
                            self.builder.push_instr(
                                IrInstr::StrLen { result, operand: base_val },
                                Some(ty.clone()),
                            );
                            return Ok((result, ty));
                        }
                        "to_upper" => {
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::StrToUpper { result, operand: base_val },
                                Some(IrType::Str),
                            );
                            return Ok((result, IrType::Str));
                        }
                        "to_lower" => {
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::StrToLower { result, operand: base_val },
                                Some(IrType::Str),
                            );
                            return Ok((result, IrType::Str));
                        }
                        "trim" => {
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::StrTrim { result, operand: base_val },
                                Some(IrType::Str),
                            );
                            return Ok((result, IrType::Str));
                        }
                        "contains" => {
                            if args.len() != 1 {
                                return Err(LowerError::Unsupported {
                                    detail: "contains() requires exactly 1 argument".into(),
                                    span: *span,
                                });
                            }
                            let (needle, _) = self.lower_expr(&args[0])?;
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::StrContains { result, haystack: base_val, needle },
                                Some(IrType::Scalar(DType::Bool)),
                            );
                            return Ok((result, IrType::Scalar(DType::Bool)));
                        }
                        "starts_with" => {
                            if args.len() != 1 {
                                return Err(LowerError::Unsupported {
                                    detail: "starts_with() requires exactly 1 argument".into(),
                                    span: *span,
                                });
                            }
                            let (prefix, _) = self.lower_expr(&args[0])?;
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::StrStartsWith { result, haystack: base_val, prefix },
                                Some(IrType::Scalar(DType::Bool)),
                            );
                            return Ok((result, IrType::Scalar(DType::Bool)));
                        }
                        "ends_with" => {
                            if args.len() != 1 {
                                return Err(LowerError::Unsupported {
                                    detail: "ends_with() requires exactly 1 argument".into(),
                                    span: *span,
                                });
                            }
                            let (suffix, _) = self.lower_expr(&args[0])?;
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::StrEndsWith { result, haystack: base_val, suffix },
                                Some(IrType::Scalar(DType::Bool)),
                            );
                            return Ok((result, IrType::Scalar(DType::Bool)));
                        }
                        "index" => {
                            // s.index(i) → StrIndex
                            if args.len() != 1 {
                                return Err(LowerError::Unsupported {
                                    detail: "index() requires exactly 1 argument".into(),
                                    span: *span,
                                });
                            }
                            let (idx, _) = self.lower_expr(&args[0])?;
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::StrIndex { result, string: base_val, index: idx },
                                Some(IrType::Scalar(DType::I64)),
                            );
                            return Ok((result, IrType::Scalar(DType::I64)));
                        }
                        "slice" => {
                            // s.slice(start, end) → StrSlice
                            if args.len() != 2 {
                                return Err(LowerError::Unsupported {
                                    detail: "slice() requires exactly 2 arguments".into(),
                                    span: *span,
                                });
                            }
                            let (start, _) = self.lower_expr(&args[0])?;
                            let (end, _) = self.lower_expr(&args[1])?;
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::StrSlice { result, string: base_val, start, end },
                                Some(IrType::Str),
                            );
                            return Ok((result, IrType::Str));
                        }
                        "find" => {
                            if args.len() != 1 {
                                return Err(LowerError::Unsupported {
                                    detail: "find() requires exactly 1 argument".into(),
                                    span: *span,
                                });
                            }
                            let (needle, _) = self.lower_expr(&args[0])?;
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::StrFind { result, haystack: base_val, needle },
                                Some(IrType::Option(Box::new(IrType::Scalar(DType::I64)))),
                            );
                            return Ok((result, IrType::Option(Box::new(IrType::Scalar(DType::I64)))));
                        }
                        "replace" => {
                            if args.len() != 2 {
                                return Err(LowerError::Unsupported {
                                    detail: "replace() requires exactly 2 arguments".into(),
                                    span: *span,
                                });
                            }
                            let (old_s, _) = self.lower_expr(&args[0])?;
                            let (new_s, _) = self.lower_expr(&args[1])?;
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::StrReplace { result, string: base_val, from: old_s, to: new_s },
                                Some(IrType::Str),
                            );
                            return Ok((result, IrType::Str));
                        }
                        "is_empty" => {
                            let len_result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::StrLen { result: len_result, operand: base_val },
                                Some(IrType::Scalar(DType::I64)),
                            );
                            let zero = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::ConstInt { result: zero, value: 0, ty: IrType::Scalar(DType::I64) },
                                Some(IrType::Scalar(DType::I64)),
                            );
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::BinOp { result, op: BinOp::CmpEq, lhs: len_result, rhs: zero, ty: IrType::Scalar(DType::I64) },
                                Some(IrType::Scalar(DType::Bool)),
                            );
                            return Ok((result, IrType::Scalar(DType::Bool)));
                        }
                        "split" => {
                            if args.len() != 1 {
                                return Err(LowerError::Unsupported {
                                    detail: "split() requires exactly 1 argument".into(),
                                    span: *span,
                                });
                            }
                            let (delim, _) = self.lower_expr(&args[0])?;
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::StrSplit { result, str_val: base_val, delim },
                                Some(IrType::List(Box::new(IrType::Str))),
                            );
                            return Ok((result, IrType::List(Box::new(IrType::Str))));
                        }
                        "repeat" => {
                            if args.len() != 1 {
                                return Err(LowerError::Unsupported {
                                    detail: "repeat() requires exactly 1 argument".into(),
                                    span: *span,
                                });
                            }
                            let (count, _) = self.lower_expr(&args[0])?;
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::StrRepeat { result, operand: base_val, count },
                                Some(IrType::Str),
                            );
                            return Ok((result, IrType::Str));
                        }
                        _ => {} // fall through
                    }
                }

                // i64/f64 method dispatch: to_str()
                if let IrType::Scalar(_dt) = &base_ty {
                    if method == "to_str" && args.is_empty() {
                        let result = self.builder.fresh_value();
                        self.builder.push_instr(
                            IrInstr::ValueToStr { result, operand: base_val },
                            Some(IrType::Str),
                        );
                        return Ok((result, IrType::Str));
                    }
                    // Extension methods on scalars — e.g. 21.double()
                    // Try the extension method dispatch below.
                }

                // map<K,V> method dispatch.
                if let IrType::Map(_k_ty, v_ty) = &base_ty {
                    let v = (**v_ty).clone();
                    match method.as_str() {
                        "len" => {
                            let result = self.builder.fresh_value();
                            let ty = IrType::Scalar(DType::I64);
                            self.builder.push_instr(
                                IrInstr::MapLen { result, map: base_val },
                                Some(ty.clone()),
                            );
                            return Ok((result, ty));
                        }
                        "get" => {
                            if args.len() != 1 {
                                return Err(LowerError::Unsupported {
                                    detail: "get() requires exactly 1 argument".into(),
                                    span: *span,
                                });
                            }
                            let (k, _) = self.lower_expr(&args[0])?;
                            let result = self.builder.fresh_value();
                            let ret_ty = IrType::Option(Box::new(v.clone()));
                            self.builder.push_instr(
                                IrInstr::MapGet { result, map: base_val, key: k, val_ty: v.clone() },
                                Some(ret_ty.clone()),
                            );
                            return Ok((result, ret_ty));
                        }
                        "set" => {
                            if args.len() != 2 {
                                return Err(LowerError::Unsupported {
                                    detail: "set() requires exactly 2 arguments".into(),
                                    span: *span,
                                });
                            }
                            let (k, _) = self.lower_expr(&args[0])?;
                            let (v_val, _) = self.lower_expr(&args[1])?;
                            self.builder.push_instr(
                                IrInstr::MapSet { map: base_val, key: k, value: v_val },
                                None,
                            );
                            // Return a dummy i64 zero as unit value
                            let zero = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::ConstInt { result: zero, value: 0, ty: IrType::Scalar(DType::I64) },
                                Some(IrType::Scalar(DType::I64)),
                            );
                            return Ok((zero, IrType::Scalar(DType::I64)));
                        }
                        "contains" => {
                            if args.len() != 1 {
                                return Err(LowerError::Unsupported {
                                    detail: "contains() requires exactly 1 argument".into(),
                                    span: *span,
                                });
                            }
                            let (k, _) = self.lower_expr(&args[0])?;
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::MapContains { result, map: base_val, key: k },
                                Some(IrType::Scalar(DType::Bool)),
                            );
                            return Ok((result, IrType::Scalar(DType::Bool)));
                        }
                        _ => {} // fall through
                    }
                }

                // channel<T> method dispatch.
                if let IrType::Chan(inner) = &base_ty {
                    let elem = (**inner).clone();
                    match method.as_str() {
                        "send" => {
                            if args.len() != 1 {
                                return Err(LowerError::Unsupported {
                                    detail: "send() requires exactly 1 argument".into(),
                                    span: *span,
                                });
                            }
                            let (v, _) = self.lower_expr(&args[0])?;
                            self.builder.push_instr(
                                IrInstr::ChanSend { chan: base_val, value: v },
                                None,
                            );
                            // Return a dummy i64 zero as unit value
                            let zero = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::ConstInt { result: zero, value: 0, ty: IrType::Scalar(DType::I64) },
                                Some(IrType::Scalar(DType::I64)),
                            );
                            return Ok((zero, IrType::Scalar(DType::I64)));
                        }
                        "recv" => {
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::ChanRecv { result, chan: base_val, elem_ty: elem.clone() },
                                Some(elem.clone()),
                            );
                            return Ok((result, elem));
                        }
                        _ => {} // fall through
                    }
                }

                // Extension method dispatch: try to find a user-defined function
                // named `method` whose first param matches the receiver type.
                // This enables `21.double()` → `double(21)` for any function `double(i64)`.
                if self.fn_sigs.contains_key(method) || self.generic_fns.contains_key(method) {
                    // Build a Call expression: method(base, args...)
                    let mut call_args = vec![*base.clone()];
                    call_args.extend(args.iter().cloned());
                    let call_expr = AstExpr::Call {
                        callee: Ident { name: method.clone(), span: *span },
                        args: call_args,
                        named_args: vec![],
                        span: *span,
                    };
                    return self.lower_expr(&call_expr);
                }

                // Trait-object dispatch: `obj.method(...)` where obj : dyn Trait.
                if let IrType::TraitObject { name: trait_name, methods: trait_methods } = &base_ty {
                    let method_sig = trait_methods
                        .iter()
                        .find(|m| m.name == *method)
                        .ok_or_else(|| LowerError::Unsupported {
                            detail: format!(
                                "trait '{}' has no method '{}'",
                                trait_name, method
                            ),
                            span: *span,
                        })?;
                    let mut arg_vals = Vec::with_capacity(args.len());
                    for arg in args {
                        let (v, _) = self.lower_expr(arg)?;
                        arg_vals.push(v);
                    }
                    let ret_ty = *method_sig.ret.clone();
                    let result = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::DynCall {
                            result,
                            obj: base_val,
                            method_name: method.clone(),
                            args: arg_vals,
                            result_ty: ret_ty.clone(),
                        },
                        Some(ret_ty.clone()),
                    );
                    return Ok((result, ret_ty));
                }

                // A trait implemented for a non-struct receiver, e.g.
                // `impl Show for list<i64>`.
                //
                // Method dispatch required the receiver to be a struct, so an
                // impl on a container was registered in `trait_dispatch` and
                // then never consulted -- the call was rejected before reaching
                // it. That is what stopped `list` and `option` from sharing an
                // interface. See known-issues #38.
                if !matches!(base_ty, IrType::Struct { .. }) {
                    if let Some(cands) = self.trait_dispatch.get(method) {
                        // Exact match first; then, if the receiver's element
                        // type is unresolved, fall back to the constructor
                        // alone -- but only when that is unambiguous.
                        //
                        // `val n: option<i64> = none()` produces `option<_>`:
                        // the annotation does not reach the `none()` call, so
                        // an exact comparison against `option<i64>` fails even
                        // though the program named the type. Matching on the
                        // constructor recovers it; requiring uniqueness stops
                        // `option<i64>` and `option<str>` impls from silently
                        // resolving to whichever was registered first.
                        let exact = cands.iter().find(|(recv_ty, _)| recv_ty == &base_ty);
                        let chosen = match exact {
                            Some(hit) => Some(hit),
                            None => {
                                let same_ctor: Vec<&(IrType, String)> = cands
                                    .iter()
                                    .filter(|(recv_ty, _)| {
                                        same_constructor_with_infer(recv_ty, &base_ty)
                                    })
                                    .collect();
                                if same_ctor.len() == 1 {
                                    Some(same_ctor[0])
                                } else {
                                    None
                                }
                            }
                        };
                        if let Some((_, fname)) = chosen {
                            let fname = fname.clone();
                            let mut arg_vals = Vec::with_capacity(args.len() + 1);
                            arg_vals.push(base_val);
                            for arg in args {
                                let (v, _) = self.lower_expr(arg)?;
                                arg_vals.push(v);
                            }
                            let ret_ty = self
                                .fn_sigs
                                .get(&fname)
                                .cloned()
                                .unwrap_or(IrType::Infer);
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::Call {
                                    result: Some(result),
                                    callee: fname,
                                    args: arg_vals,
                                    result_ty: Some(ret_ty.clone()),
                                },
                                Some(ret_ty.clone()),
                            );
                            return Ok((result, ret_ty));
                        }
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

                // Check if `method` is a struct field with a function type.
                // If so, extract the field and call it as a closure.
                let struct_mangled = format!("{}__{}", type_name, method);
                if !self.fn_sigs.contains_key(&struct_mangled) {
                    if let Some(fields) = self.module.struct_def(&type_name) {
                        if let Some((_, field_ty)) = fields.iter().find(|(name, _)| name == method) {
                            if let IrType::Fn { params: _, ret } = field_ty {
                                let field_val = self.builder.fresh_value();
                                self.builder.push_instr(
                                    IrInstr::GetField {
                                        result: field_val,
                                        base: base_val,
                                        field_index: fields.iter().position(|(n, _)| n == method).unwrap(),
                                        result_ty: field_ty.clone(),
                                    },
                                    Some(field_ty.clone()),
                                );
                                let mut arg_vals = Vec::with_capacity(args.len());
                                for arg in args {
                                    let (v, _) = self.lower_expr(arg)?;
                                    arg_vals.push(v);
                                }
                                let ret_ty = *ret.clone();
                                let result = self.builder.fresh_value();
                                // Regular function via closure field: no env pointer.
                                self.builder.push_instr(
                                    IrInstr::CallClosure {
                                        result: Some(result),
                                        closure: field_val,
                                        args: arg_vals,
                                        result_ty: ret_ty.clone(),
                                        pass_env: false,
                                    },
                                    Some(ret_ty.clone()),
                                );
                                return Ok((result, ret_ty));
                            }
                        }
                    }
                }

                // Normal method dispatch: find the mangled function `TypeName__method`.
                let mangled = if self.fn_sigs.contains_key(&struct_mangled) {
                    struct_mangled
                } else if let Some(impls) = self.trait_dispatch.get(method) {
                    // Find the impl for this concrete type.
                    let dispatch_ty = IrType::Struct {
                        name: type_name.clone(),
                        fields: Vec::new(),
                    };
                    impls
                        .iter()
                        .find(|(ty, _)| {
                            if let (
                                IrType::Struct { name: n1, .. },
                                IrType::Struct { name: n2, .. },
                            ) = (ty, &dispatch_ty)
                            {
                                n1 == n2
                            } else {
                                ty == &dispatch_ty
                            }
                        })
                        .map(|(_, name)| name.clone())
                        .unwrap_or(struct_mangled)
                } else {
                    struct_mangled
                };

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

            AstExpr::Handle {
                expr, arms, return_ty, ..
            } => self.lower_handle(expr, arms, return_ty),

            AstExpr::NullCoal { expr, default, .. } => {
                let (val, val_ty) = self.lower_expr(expr)?;
                let (default_val, default_ty) = self.lower_expr(default)?;
                match &val_ty {
                    IrType::Option(inner) => {
                        let result_type = if matches!(**inner, IrType::Infer) {
                            default_ty.clone()
                        } else {
                            (**inner).clone()
                        };
                        let is_some_result = self.builder.fresh_value();
                        self.builder.push_instr(
                            IrInstr::IsSome { result: is_some_result, operand: val },
                            Some(IrType::Scalar(DType::Bool)),
                        );
                        let then_bb = self.builder.create_block(Some("nullcoal_some"));
                        let else_bb = self.builder.create_block(Some("nullcoal_none"));
                        let merge_bb = self.builder.create_block(Some("nullcoal_merge"));
                        self.builder.push_instr(
                            IrInstr::CondBr {
                                cond: is_some_result,
                                then_block: then_bb,
                                then_args: vec![],
                                else_block: else_bb,
                                else_args: vec![],
                            },
                            None,
                        );
                        self.builder.set_current_block(then_bb);
                        let unwrapped = self.builder.fresh_value();
                        self.builder.push_instr(
                            IrInstr::OptionUnwrap { result: unwrapped, operand: val, result_ty: result_type.clone() },
                            Some(result_type.clone()),
                        );
                        self.builder.push_instr(IrInstr::Br { target: merge_bb, args: vec![unwrapped] }, None);
                        self.builder.set_current_block(else_bb);
                        self.builder.push_instr(IrInstr::Br { target: merge_bb, args: vec![default_val] }, None);
                        let merge_result = self.builder.add_block_param(merge_bb, Some("nullcoal_res"), result_type.clone());
                        self.builder.set_current_block(merge_bb);
                        Ok((merge_result, result_type))
                    }
                    IrType::ResultType(ok, _) => {
                        let result_type = if matches!(**ok, IrType::Infer) {
                            default_ty.clone()
                        } else {
                            (**ok).clone()
                        };
                        let is_ok_result = self.builder.fresh_value();
                        self.builder.push_instr(
                            IrInstr::IsOk { result: is_ok_result, operand: val },
                            Some(IrType::Scalar(DType::Bool)),
                        );
                        let then_bb = self.builder.create_block(Some("nullcoal_ok"));
                        let else_bb = self.builder.create_block(Some("nullcoal_err"));
                        let merge_bb = self.builder.create_block(Some("nullcoal_merge"));
                        self.builder.push_instr(
                            IrInstr::CondBr {
                                cond: is_ok_result,
                                then_block: then_bb,
                                then_args: vec![],
                                else_block: else_bb,
                                else_args: vec![],
                            },
                            None,
                        );
                        self.builder.set_current_block(then_bb);
                        let unwrapped = self.builder.fresh_value();
                        self.builder.push_instr(
                            IrInstr::ResultUnwrap { result: unwrapped, operand: val, result_ty: result_type.clone() },
                            Some(result_type.clone()),
                        );
                        self.builder.push_instr(IrInstr::Br { target: merge_bb, args: vec![unwrapped] }, None);
                        self.builder.set_current_block(else_bb);
                        self.builder.push_instr(IrInstr::Br { target: merge_bb, args: vec![default_val] }, None);
                        let merge_result = self.builder.add_block_param(merge_bb, Some("nullcoal_res"), result_type.clone());
                        self.builder.set_current_block(merge_bb);
                        Ok((merge_result, result_type))
                    }
                    _ => {
                        Ok((val, val_ty))
                    }
                }
            }

            AstExpr::MapLiteral { entries, span: _ } => {
                // Create empty map, then call map_set for each pair.
                let (key_ty, val_ty) = if let Some(IrType::Map(k, v)) = &self.binding_ty {
                    (*k.clone(), *v.clone())
                } else {
                    // Infer from first entry if possible
                    if let Some((k_expr, _v_expr)) = entries.first() {
                        let k_ty = self.infer_ast_expr_type_simple(k_expr);
                        let v_ty = self.infer_ast_expr_type_simple(&_v_expr);
                        (k_ty, v_ty)
                    } else {
                        (IrType::Str, IrType::Scalar(DType::I64))
                    }
                };
                let map_ty = IrType::Map(Box::new(key_ty.clone()), Box::new(val_ty.clone()));
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::MapNew { result, key_ty: key_ty.clone(), val_ty: val_ty.clone() },
                    Some(map_ty.clone()),
                );
                for (k, v) in entries {
                    let (key_id, _) = self.lower_expr(&k)?;
                    let (val_id, _) = self.lower_expr(&v)?;
                    self.builder.push_instr(
                        IrInstr::MapSet { map: result, key: key_id, value: val_id },
                        None,
                    );
                }
                Ok((result, map_ty))
            }

            AstExpr::Ref { expr, .. } => self.lower_expr(expr),
            AstExpr::RefMut { expr, .. } => self.lower_expr(expr),
            AstExpr::Deref { expr, .. } => self.lower_expr(expr),
            AstExpr::Move { expr, .. } => self.lower_expr(expr),
            AstExpr::Unsafe { body, .. } => self.lower_expr(body),
            AstExpr::Splat { expr, .. } => self.lower_expr(expr),
            AstExpr::MacroCall { name, .. } => {
                return Err(crate::error::LowerError::Unsupported {
                    detail: format!("unexpanded macro call '{}'", name.name),
                    span: name.span,
                });
            }

            AstExpr::TryCatch { body, catch_param, catch_body, span: _ } => {
                let prev_catch = self.try_catch_catch_bb.take();
                let prev_cont = self.try_catch_cont_bb.take();
                let prev_param = self.try_catch_param.take();

                let catch_bb = self.builder.create_block(Some("catch"));
                let cont_bb = self.builder.create_block(Some("try_cont"));

                self.try_catch_catch_bb = Some(catch_bb);
                self.try_catch_cont_bb = Some(cont_bb);
                self.try_catch_param = Some(catch_param.clone());

                let (body_val, body_ty) = self.lower_expr(body)?;
                if !self.builder.is_current_block_terminated() {
                    self.builder.push_instr(IrInstr::Br { target: cont_bb, args: vec![body_val] }, None);
                }

                self.builder.set_current_block(catch_bb);
                let catch_param_val = self.builder.add_block_param(catch_bb, Some(catch_param), IrType::Infer);
                self.scope.insert(catch_param.clone(), (catch_param_val, IrType::Infer));
                let (catch_val, _catch_ty) = self.lower_expr(catch_body)?;
                if !self.builder.is_current_block_terminated() {
                    self.builder.push_instr(IrInstr::Br { target: cont_bb, args: vec![catch_val] }, None);
                }

                self.builder.set_current_block(cont_bb);
                let result = self.builder.add_block_param(cont_bb, Some("try_result"), body_ty.clone());

                self.try_catch_catch_bb = prev_catch;
                self.try_catch_cont_bb = prev_cont;
                self.try_catch_param = prev_param;

                Ok((result, body_ty))
            }

            AstExpr::Raise { effect_name, args, span: _ } => {
                let mut lowered_args = Vec::new();
                let mut arg_tys = Vec::new();
                for a in args {
                    let (v, ty) = self.lower_expr(a)?;
                    lowered_args.push(v);
                    arg_tys.push(ty);
                }
                let ret_ty = self.module.extern_fns.iter()
                    .find(|ef| ef.name == *effect_name)
                    .map(|ef| ef.ret_ty.clone())
                    .unwrap_or(IrType::Infer);
                let result = self.builder.fresh_value();
                self.builder.push_instr(IrInstr::CallExtern {
                    result: Some(result),
                    name: effect_name.clone(),
                    args: lowered_args,
                    ret_ty: ret_ty.clone(),
                }, Some(ret_ty.clone()));
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
        //
        // Sorted by name because `scope` is a `HashMap`, whose iteration order
        // changes with the per-process hash seed. The order is self-consistent
        // within one run — the lifted parameter list and the `MakeClosure`
        // argument list are both built from this vector — so programs were
        // correct, but the same source compiled to different IR on every run.
        // Reproducible builds matter on their own, and non-reproducibility also
        // hid a real miscompilation for as long as it lasted (known-issues #17).
        let mut captures: Vec<(String, ValueId, IrType)> = self
            .scope
            .iter()
            .filter(|(name, _)| !param_names.contains(*name))
            .map(|(name, (vid, ty))| (name.clone(), *vid, ty.clone()))
            .collect();
        captures.sort_by(|a, b| a.0.cmp(&b.0));
        let captures = captures;

        // Build the lifted function: params = captures + lambda_params.
        let mut lifted_params: Vec<Param> = captures
            .iter()
            .map(|(name, _, ty)| Param {
                name: name.clone(),
                ty: ty.clone(),
            })
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
        lambda_lowerer.builder.push_instr(
            IrInstr::Return {
                values: vec![ret_val],
            },
            None,
        );
        lambda_lowerer.builder.seal_unterminated_blocks();

        // Patch the return type and capture count.
        let mut ir_func = lambda_lowerer.builder.build();
        ir_func.return_ty = ret_ty.clone();
        ir_func.capture_count = captures.len();

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

    /// Lowers a `handle <expr> with { <arms> }` expression.
    /// Emits `PushHandler`, body, `PopHandler`.
    fn lower_handle(
        &mut self,
        expr: &AstExpr,
        arms: &[AstHandlerArm],
        return_ty: &AstType,
    ) -> Result<(ValueId, IrType), LowerError> {
        // Lower each handler arm into an IR function + HandlerArm descriptor.
        let mut handler_arms = Vec::new();
        for arm in arms {
            let handler_arm = self.lower_handler_arm(arm)?;
            handler_arms.push(handler_arm);
        }

        // Emit PushHandler.
        self.builder.push_instr(
            IrInstr::PushHandler {
                arms: handler_arms,
            },
            None,
        );

        // Lower the body expression.
        let (body_val, body_ty) = self.lower_expr(expr)?;

        // Emit PopHandler.
        self.builder.push_instr(IrInstr::PopHandler, None);

        // The result type of the handle expression.
        // If the parsed return type is Infer, use the body's inferred type.
        let ret_ty = self.resolve_ty(return_ty);
        let result_ty = if ret_ty == IrType::Infer { body_ty } else { ret_ty };
        Ok((body_val, result_ty))
    }

    /// Lowers a single handler arm to a HandlerArm descriptor + lifted function.
    /// The handler body only receives the effect payload args (not captures),
    /// because the args are already evaluated before the extern call.
    fn lower_handler_arm(
        &mut self,
        arm: &AstHandlerArm,
    ) -> Result<crate::ir::instr::HandlerArm, LowerError> {
        let counter = self.lambda_counter.get();
        self.lambda_counter.set(counter + 1);
        let fn_name = format!("__handler_{}", counter);

        // Build the lifted function: params = handler_params only (no captures).
        // Use the extern function's signature for param types.
        let extern_fn = self.module.extern_fns.iter().find(|ef| ef.name == arm.effect_name);
        let mut lifted_params: Vec<Param> = Vec::new();
        for (i, p) in arm.params.iter().enumerate() {
            let param_ty = extern_fn.and_then(|ef| ef.param_types.get(i).cloned()).unwrap_or(IrType::Infer);
            lifted_params.push(Param {
                name: p.name.clone(),
                ty: param_ty,
            });
        }
        // Always add a leading block param for the continuation pointer.
        // For has_resume arms, this is named after the resume param.
        // For non-resume arms, this is an unnamed "__cont" param (ignored).
        let cont_param_name = if let Some(ref rp) = arm.resume_param {
            rp.name.clone()
        } else {
            "__cont".to_string()
        };
        lifted_params.insert(
            0,
            Param {
                name: cont_param_name.clone(),
                ty: IrType::WeakRef(Box::new(IrType::Infer)),
            },
        );

        let temp_ret_ty = IrType::Infer;
        let temp_builder = IrFunctionBuilder::new(&fn_name, lifted_params.clone(), temp_ret_ty);
        let mut handler_lowerer = Lowerer::new_with_lambda_state(
            temp_builder,
            self.module,
            self.fn_sigs,
            self.lambda_counter.clone(),
            self.lifted_fns.clone(),
        );
        handler_lowerer.resume_param_name = arm.resume_param.as_ref().map(|r| r.name.clone());

        let entry = handler_lowerer.builder.create_block(Some("entry"));
        handler_lowerer.builder.set_current_block(entry);

        // Populate the handler scope with param values only.
        // The continuation is the first block param when has_resume is true.
        if let Some(ref rp) = arm.resume_param {
            let cont_ty = IrType::WeakRef(Box::new(IrType::Infer));
            let val = handler_lowerer
                .builder
                .add_block_param(entry, Some(&rp.name), cont_ty.clone());
            handler_lowerer.scope.insert(rp.name.clone(), (val, cont_ty));
        }
        // Use the types already recovered from the extern signature above.
        //
        // These block params were previously added as `IrType::Infer`, which
        // discarded the very types `lifted_params` had just looked up. Inference
        // then defaulted them to `i64`, so a handler that *used* its parameter
        // produced a `str` bound as an integer: interpreted, `"s:" + p`
        // evaluated to 0; natively, codegen emitted
        // `call ptr @iris_str_concat(ptr %v2, ptr %p)` with `%p` defined as
        // i64, which fails LLVM verification.
        //
        // `test_resume_handler.iris` passed on both backends throughout,
        // because its handler ignores its argument. See known-issues #31.
        // Read the types straight from the extern signature rather than from
        // `lifted_params`, which has the continuation inserted at index 0 and
        // is therefore off by one against `arm.params`. Indexing it by the arm
        // position gave every first parameter the continuation's
        // `weak_ref<_>`, which is why `"s:" + p` reported
        // `expected 'str' but found 'weak_ref<_>'`.
        for (i, p) in arm.params.iter().enumerate() {
            let ty = extern_fn
                .and_then(|ef| ef.param_types.get(i).cloned())
                .unwrap_or(IrType::Infer);
            let val = handler_lowerer
                .builder
                .add_block_param(entry, Some(&p.name), ty.clone());
            handler_lowerer.scope.insert(p.name.clone(), (val, ty));
        }

        let (ret_val, ret_ty) = handler_lowerer.lower_expr(&arm.body)?;
        handler_lowerer.builder.push_instr(
            IrInstr::Return {
                values: vec![ret_val],
            },
            None,
        );
        handler_lowerer.builder.seal_unterminated_blocks();

        let mut ir_func = handler_lowerer.builder.build();
        ir_func.return_ty = ret_ty.clone();
        ir_func.capture_count = 0;

        self.lifted_fns.borrow_mut().push(ir_func);

        Ok(crate::ir::instr::HandlerArm {
            effect_name: arm.effect_name.clone(),
            func_name: fn_name,
            num_args: arm.params.len(),
            has_resume: arm.resume_param.is_some(),
        })
    }

    /// Try to evaluate a `const def` function at compile time.
    /// Returns `Some(result)` if all arguments are integer/float/bool literals
    /// and the body is a simple expression (arithmetic, if/else, etc.).
    /// Returns `None` if evaluation fails (non-literal args, unsupported body, etc.).
    fn try_eval_const_fn(
        &self,
        func: &crate::parser::ast::AstFunction,
        args: &[AstExpr],
    ) -> Option<i64> {
        // All arguments must be compile-time known literals.
        let mut params: HashMap<String, i64> = HashMap::new();
        for (param, arg) in func.params.iter().zip(args.iter()) {
            let val = match arg {
                AstExpr::IntLit { value, .. } => *value,
                AstExpr::FloatLit { value, .. } => value.to_bits() as i64,
                AstExpr::BoolLit { value, .. } => if *value { 1 } else { 0 },
                _ => return None,
            };
            params.insert(param.name.name.clone(), val);
        }

        self.eval_const_block(&func.body.stmts, &params)
    }

    /// Evaluate a block of statements at compile time, returning the value of
    /// the final expression statement.
    fn eval_const_block(
        &self,
        stmts: &[crate::parser::ast::AstStmt],
        params: &HashMap<String, i64>,
    ) -> Option<i64> {
        let mut locals: HashMap<String, i64> = params.clone();
        for stmt in stmts {
            match stmt {
                crate::parser::ast::AstStmt::Let { name, init, .. } => {
                    let val = self.eval_const_expr(init, &locals)?;
                    locals.insert(name.name.clone(), val);
                }
                crate::parser::ast::AstStmt::Expr(expr) => {
                    return self.eval_const_expr(expr, &locals);
                }
                _ => return None,
            }
        }
        None
    }

    /// Evaluate a simple expression at compile time.
    fn eval_const_expr(
        &self,
        expr: &AstExpr,
        locals: &HashMap<String, i64>,
    ) -> Option<i64> {
        match expr {
            AstExpr::IntLit { value, .. } => Some(*value),
            AstExpr::FloatLit { value, .. } => Some(value.to_bits() as i64),
            AstExpr::BoolLit { value, .. } => Some(if *value { 1 } else { 0 }),
            AstExpr::Ident(name) => locals.get(name.name.as_str()).copied(),
            AstExpr::BinOp { op, lhs, rhs, .. } => {
                let l = self.eval_const_expr(lhs, locals)?;
                let r = self.eval_const_expr(rhs, locals)?;
                match op {
                    crate::parser::ast::AstBinOp::Add => Some(l.wrapping_add(r)),
                    crate::parser::ast::AstBinOp::Sub => Some(l.wrapping_sub(r)),
                    crate::parser::ast::AstBinOp::Mul => Some(l.wrapping_mul(r)),
                    crate::parser::ast::AstBinOp::Div => {
                        if r == 0 { return None; }
                        Some(l.wrapping_div(r))
                    }
                    crate::parser::ast::AstBinOp::Mod => {
                        if r == 0 { return None; }
                        Some(l.wrapping_rem(r))
                    }
                    crate::parser::ast::AstBinOp::CmpEq => Some(if l == r { 1 } else { 0 }),
                    crate::parser::ast::AstBinOp::CmpNe => Some(if l != r { 1 } else { 0 }),
                    crate::parser::ast::AstBinOp::CmpLt => Some(if l < r { 1 } else { 0 }),
                    crate::parser::ast::AstBinOp::CmpLe => Some(if l <= r { 1 } else { 0 }),
                    crate::parser::ast::AstBinOp::CmpGt => Some(if l > r { 1 } else { 0 }),
                    crate::parser::ast::AstBinOp::CmpGe => Some(if l >= r { 1 } else { 0 }),
                    crate::parser::ast::AstBinOp::And => Some(if l != 0 && r != 0 { 1 } else { 0 }),
                    crate::parser::ast::AstBinOp::Or => Some(if l != 0 || r != 0 { 1 } else { 0 }),
                }
            }
            AstExpr::UnaryOp { op, expr, .. } => {
                let v = self.eval_const_expr(expr, locals)?;
                match op {
                    crate::parser::ast::AstUnaryOp::Neg => Some(v.wrapping_neg()),
                    crate::parser::ast::AstUnaryOp::Not => Some(if v == 0 { 1 } else { 0 }),
                }
            }
            AstExpr::If { cond, then_block, else_block, .. } => {
                let c = self.eval_const_expr(cond, locals)?;
                if c != 0 {
                    self.eval_const_block(&then_block.stmts, locals)
                } else if let Some(eb) = else_block {
                    self.eval_const_block(&eb.stmts, locals)
                } else {
                    None
                }
            }
            AstExpr::Block(block) => {
                let mut block_locals = locals.clone();
                for stmt in &block.stmts {
                    match stmt {
                        crate::parser::ast::AstStmt::Let { name, init, .. } => {
                            let val = self.eval_const_expr(init, &block_locals)?;
                            block_locals.insert(name.name.clone(), val);
                        }
                        _ => return None,
                    }
                }
                if let Some(ref result_expr) = block.tail {
                    self.eval_const_expr(result_expr, &block_locals)
                } else {
                    Some(0)
                }
            }
            _ => None,
        }
    }

    /// Lowers a function call. Handles the built-in `einsum` intrinsic specially.
    /// Resolve named arguments `f(w = 3, h = 4)` into positional order.
    ///
    /// Named arguments were parsed into `AstExpr::Call::named_args` but never
    /// read by the lowerer, so the call was lowered with only its *positional*
    /// arguments -- usually none at all. The callee then read whatever happened
    /// to be in the argument registers. No stage produced a diagnostic:
    /// `tests/test_named_args.iris` printed "All named arg tests passed!" while
    /// every value it computed was wrong. See known-issues #1.
    ///
    /// Only reachable when `named_args` is non-empty, so a purely positional
    /// call takes exactly the path it always did.
    fn resolve_named_args(
        &self,
        callee_name: &str,
        args: &[AstExpr],
        named_args: &[(String, AstExpr)],
        span: Span,
    ) -> Result<Vec<AstExpr>, LowerError> {
        let params = self
            .fn_param_names
            .get(callee_name)
            .ok_or_else(|| LowerError::Rejected {
                detail: format!(
                    "named arguments are not supported when calling `{}`. They are matched \
                     against declared parameter names, which are only known for functions \
                     defined in this program -- not builtins, externs or closures. Pass the \
                     arguments positionally instead.",
                    callee_name
                ),
                span,
            })?;

        if args.len() > params.len() {
            return Err(LowerError::Rejected {
                detail: format!(
                    "`{}` declares {} parameter(s) but {} positional argument(s) were given \
                     before the named ones",
                    callee_name,
                    params.len(),
                    args.len()
                ),
                span,
            });
        }

        // Positional arguments fill the leading slots; named ones fill by name.
        let mut slots: Vec<Option<AstExpr>> = params.iter().map(|_| None).collect();
        for (i, a) in args.iter().enumerate() {
            slots[i] = Some(a.clone());
        }

        for (name, value) in named_args {
            let idx = params.iter().position(|p| p == name).ok_or_else(|| {
                // Nearest declared parameter by shared prefix, for a "did you mean".
                let suggestion = params
                    .iter()
                    .find(|p| {
                        let n = name.as_bytes();
                        let q = p.as_bytes();
                        !n.is_empty() && !q.is_empty() && n[0] == q[0]
                    })
                    .cloned();
                LowerError::Rejected {
                    detail: match suggestion {
                        Some(near) => format!(
                            "`{}` has no parameter named `{}` -- did you mean `{}`? It declares: {}",
                            callee_name,
                            name,
                            near,
                            params.join(", ")
                        ),
                        None => format!(
                            "`{}` has no parameter named `{}`. It declares: {}",
                            callee_name,
                            name,
                            params.join(", ")
                        ),
                    },
                    span: value.span(),
                }
            })?;
            if slots[idx].is_some() {
                return Err(LowerError::Rejected {
                    detail: format!(
                        "parameter `{}` of `{}` is supplied more than once",
                        name, callee_name
                    ),
                    span: value.span(),
                });
            }
            slots[idx] = Some(value.clone());
        }

        // Fill any gap from that parameter's default. A gap with no default is
        // an error here rather than a silently short argument list.
        let defaults = self.fn_defaults.get(callee_name);
        let mut out = Vec::with_capacity(slots.len());
        for (i, slot) in slots.into_iter().enumerate() {
            match slot {
                Some(e) => out.push(e),
                None => match defaults.and_then(|d| d.get(i)).and_then(|d| d.clone()) {
                    Some(d) => out.push(d),
                    None => {
                        return Err(LowerError::Rejected {
                            detail: format!(
                                "parameter `{}` of `{}` has no argument and no default value",
                                params[i], callee_name
                            ),
                            span,
                        });
                    }
                },
            }
        }
        Ok(out)
    }

    fn lower_call(
        &mut self,
        callee: &Ident,
        args: &[AstExpr],
        named_args: &[(String, AstExpr)],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        let callee_name = self.resolve_unqualified_name(&callee.name);

        // Named arguments are folded into positional order before anything
        // else inspects `args`, so every path below sees a normal call.
        let reordered: Vec<AstExpr>;
        let args: &[AstExpr] = if named_args.is_empty() {
            args
        } else {
            reordered = self.resolve_named_args(&callee_name, args, named_args, span)?;
            &reordered
        };

        // A user (or stdlib) definition of this name shadows the builtin.
        //
        // The builtin used to win unconditionally and silently:
        //
        //     def band(a: i64, b: i64) -> i64 { 999 }
        //     band(12, 10)   ->  8      // the builtin ran; the user's fn never did
        //
        // No error, no warning. 203 builtin names are effectively reserved this
        // way — among them len, min, max, map, filter, get, set, push, pop —
        // and none is a keyword, so nothing stops a program choosing one. If the
        // arity differs the failure is at least loud, but it names the builtin
        // the programmer never intended to call.
        //
        // Thirteen *stdlib* functions collide with a builtin too, so their IRIS
        // implementations were dead code as well.
        //
        // Blanking the dispatch name routes the call to the general
        // user-function path, which is the one the programmer wrote.
        // See known-issues #15.
        let shadows_builtin = CURRENT_USER_FNS.with(|u| {
            let set = u.borrow();
            set.contains(&callee.name) || set.contains(&callee_name)
        }) || self.generic_fns.contains_key(&callee.name)
            || self.generic_fns.contains_key(&callee_name);
        let dispatch_name: &str = if shadows_builtin { "" } else { callee.name.as_str() };

        // Built-in: resume continuation call.
        // When lowering inside a handler arm with `has_resume`, a call to the
        // resume_param name (e.g. `v(x)`) lowers to a `ResumeCont` instruction
        // that signals the parent interpreter to return `x` from the effect
        // perform site.
        if let Some(ref rp_name) = self.resume_param_name {
            let rp_name = rp_name.clone();
            if callee.name == rp_name && args.len() == 1 {
                let (val, val_ty) = self.lower_expr(&args[0])?;
                let cont_val = self
                    .scope
                    .get(&rp_name)
                    .map(|(v, _)| *v)
                    .ok_or_else(|| LowerError::Unsupported {
                        detail: format!("resume continuation '{}' not in scope", rp_name),
                        span,
                    })?;
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ResumeCont {
                        cont: cont_val,
                        value: val,
                        result,
                    },
                    Some(val_ty.clone()),
                );
                return Ok((result, val_ty));
            }
        }

        // Built-in: println(x) / print(x) → Print instruction
        if dispatch_name == "println" || callee.name == "print" || callee.name == "eprintln" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: format!("{}() requires exactly 1 argument", callee.name),
                    span,
                });
            }
            let (operand, _) = self.lower_expr(&args[0])?;
            self.builder.push_instr(IrInstr::Print { operand }, None);
            // Return a dummy i64 0 as the "unit" value.
            let dummy = self.builder.fresh_value();
            let dummy_ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: dummy,
                    value: 0,
                    ty: dummy_ty.clone(),
                },
                Some(dummy_ty.clone()),
            );
            return Ok((dummy, dummy_ty));
        }

        // Built-in: task_group() → TaskGroupNew
        if dispatch_name == "task_group" {
            let result = self.builder.fresh_value();
            self.builder
                .push_instr(IrInstr::TaskGroupNew { result }, Some(IrType::TaskGroup));
            return Ok((result, IrType::TaskGroup));
        }

        // Built-in: task_group_join(tg) → TaskGroupJoin
        if dispatch_name == "task_group_join" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "task_group_join() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (group_val, _) = self.lower_expr(&args[0])?;
            self.builder
                .push_instr(IrInstr::TaskGroupJoin { group: group_val }, None);
            return Ok((self.builder.fresh_value(), IrType::Scalar(DType::I64)));
        }

        // Built-in: task_group_cancel(tg) → TaskGroupCancel
        if dispatch_name == "task_group_cancel" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "task_group_cancel() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (group_val, _) = self.lower_expr(&args[0])?;
            self.builder
                .push_instr(IrInstr::TaskGroupCancel { group: group_val }, None);
            return Ok((self.builder.fresh_value(), IrType::Scalar(DType::I64)));
        }

        // Built-in: channel() → ChanNew
        if dispatch_name == "channel" {
            let elem_ty = if let Some(IrType::Chan(inner)) = &self.binding_ty {
                (**inner).clone()
            } else {
                IrType::Infer
            };
            let chan_ty = IrType::Chan(Box::new(elem_ty.clone()));
            let capacity_val = if args.len() == 1 {
                self.lower_expr(&args[0])?.0
            } else {
                let dummy = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstInt {
                        result: dummy,
                        value: -1,
                        ty: IrType::Scalar(DType::I64),
                    },
                    Some(IrType::Scalar(DType::I64)),
                );
                dummy
            };
            let result = self.builder.fresh_value();
            self.builder
                .push_instr(IrInstr::ChanNew { result, elem_ty, capacity: capacity_val }, Some(chan_ty.clone()));
            return Ok((result, chan_ty));
        }

        // Built-in: send(ch, v) / chan_send(ch, v) → ChanSend (returns unit, use dummy i64 0)
        if dispatch_name == "send" || callee.name == "chan_send" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "send() requires exactly 2 arguments (channel, value)".into(),
                    span,
                });
            }
            let (chan_val, _) = self.lower_expr(&args[0])?;
            let (val, val_ty) = self.lower_expr(&args[1])?;
            // Record the concrete element type so recv() can use it.
            self.chan_elem_types
                .entry(chan_val)
                .or_insert_with(|| val_ty.clone());
            self.builder.push_instr(
                IrInstr::ChanSend {
                    chan: chan_val,
                    value: val,
                },
                None,
            );
            // Return a dummy i64 0 as the "unit" value.
            let dummy = self.builder.fresh_value();
            let dummy_ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: dummy,
                    value: 0,
                    ty: dummy_ty.clone(),
                },
                Some(dummy_ty.clone()),
            );
            return Ok((dummy, dummy_ty));
        }

        // Built-in: recv(ch) → ChanRecv
        if dispatch_name == "recv" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "recv() requires exactly 1 argument (channel)".into(),
                    span,
                });
            }
            let (chan_val, chan_ty) = self.lower_expr(&args[0])?;
            // Prefer the concrete element type recorded when send() was called.
            let elem_ty = self.chan_elem_types.get(&chan_val).cloned().unwrap_or({
                if let IrType::Chan(elem) = chan_ty {
                    *elem
                } else {
                    IrType::Infer
                }
            });
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ChanRecv {
                    result,
                    chan: chan_val,
                    elem_ty: elem_ty.clone(),
                },
                Some(elem_ty.clone()),
            );
            return Ok((result, elem_ty));
        }

        // Built-in: tape(x) → mark scalar value as a reverse-mode leaf.
        if dispatch_name == "tape" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "tape() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (value, ty) = self.lower_expr(&args[0])?;
            if !Self::is_reverse_diff_scalar(&ty) {
                return Err(LowerError::Unsupported {
                    detail: format!("tape() only supports numeric scalar values, got {}", ty),
                    span,
                });
            }
            self.ensure_taped_leaf(value, &ty);
            return Ok((value, ty));
        }

        // Built-in: backward(loss) → run reverse-mode backprop from a taped scalar.
        if dispatch_name == "backward" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "backward() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (loss, loss_ty) = self.lower_expr(&args[0])?;
            if !Self::is_reverse_diff_scalar(&loss_ty) {
                return Err(LowerError::Unsupported {
                    detail: format!("backward() requires a numeric scalar loss, got {}", loss_ty),
                    span,
                });
            }
            let internal = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::Backward {
                    result: internal,
                    loss: self.tape_ref_for(loss),
                },
                Some(IrType::Scalar(DType::Bool)),
            );
            let dummy = self.builder.fresh_value();
            let dummy_ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: dummy,
                    value: 0,
                    ty: dummy_ty.clone(),
                },
                Some(dummy_ty.clone()),
            );
            return Ok((dummy, dummy_ty));
        }

        // Built-in: grad(x) → extract d(loss)/d(x) after backward(loss).
        // Only activates when `x` is a taped value (reverse-mode AD context).
        // For forward-mode dual numbers `grad(literal)` falls through to MakeGrad.
        if dispatch_name == "grad" && args.len() == 1 {
            let (value, _) = self.lower_expr(&args[0])?;
            if self.taped_values.contains(&value) {
                let result = self.builder.fresh_value();
                let result_ty = IrType::Scalar(DType::F64);
                self.builder.push_instr(
                    IrInstr::TapeGrad {
                        result,
                        tape_node: self.tape_ref_for(value),
                    },
                    Some(result_ty.clone()),
                );
                return Ok((result, result_ty));
            }
            // Not a taped value → forward-mode MakeGrad path below handles it.
            // Fall through (the MakeGrad handler will re-lower args[0], which is fine
            // for literals/simple vars with no side effects).
        }

        // Built-in: atomic(v) / atomic_new(v) → AtomicNew
        if dispatch_name == "atomic" || callee.name == "atomic_new" {
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
                IrInstr::AtomicNew {
                    result,
                    value: val,
                    result_ty: result_ty.clone(),
                },
                Some(result_ty.clone()),
            );
            return Ok((result, result_ty));
        }

        // Built-in: atomic_load(a) → AtomicLoad
        if dispatch_name == "atomic_load" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "atomic_load() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, atomic_ty) = self.lower_expr(&args[0])?;
            let inner_ty = if let IrType::Atomic(inner) = atomic_ty {
                *inner
            } else {
                IrType::Infer
            };
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::AtomicLoad {
                    result,
                    atomic: val,
                    result_ty: inner_ty.clone(),
                },
                Some(inner_ty.clone()),
            );
            return Ok((result, inner_ty));
        }

        // Built-in: atomic_store(a, v) → AtomicStore
        if dispatch_name == "atomic_store" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "atomic_store() requires exactly 2 arguments".into(),
                    span,
                });
            }
            let (a, _) = self.lower_expr(&args[0])?;
            let (v, _) = self.lower_expr(&args[1])?;
            self.builder.push_instr(
                IrInstr::AtomicStore {
                    atomic: a,
                    value: v,
                },
                None,
            );
            let dummy = self.builder.fresh_value();
            let dummy_ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: dummy,
                    value: 0,
                    ty: dummy_ty.clone(),
                },
                Some(dummy_ty.clone()),
            );
            return Ok((dummy, dummy_ty));
        }

        // Built-in: mutex_new(v) → MutexNew
        if dispatch_name == "mutex_new" {
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
                IrInstr::MutexNew {
                    result,
                    value: val,
                    result_ty: result_ty.clone(),
                },
                Some(result_ty.clone()),
            );
            return Ok((result, result_ty));
        }

        // Built-in: mutex_lock(m) → MutexLock
        if dispatch_name == "mutex_lock" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "mutex_lock() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, mutex_ty) = self.lower_expr(&args[0])?;
            let inner_ty = if let IrType::Mutex(inner) = mutex_ty {
                *inner
            } else {
                IrType::Infer
            };
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::MutexLock {
                    result,
                    mutex: val,
                    result_ty: inner_ty.clone(),
                },
                Some(inner_ty.clone()),
            );
            return Ok((result, inner_ty));
        }

        // Built-in: barrier() → Barrier (sync point, no-op in interpreter)
        if dispatch_name == "barrier" {
            self.builder.push_instr(IrInstr::Barrier, None);
            let dummy = self.builder.fresh_value();
            let dummy_ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: dummy,
                    value: 0,
                    ty: dummy_ty.clone(),
                },
                Some(dummy_ty.clone()),
            );
            return Ok((dummy, dummy_ty));
        }

        // Built-in: mutex_unlock(m) → MutexUnlock (no-op in interpreter, returns unit)
        if dispatch_name == "mutex_unlock" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "mutex_unlock() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, _) = self.lower_expr(&args[0])?;
            self.builder
                .push_instr(IrInstr::MutexUnlock { mutex: val }, None);
            let dummy = self.builder.fresh_value();
            let dummy_ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: dummy,
                    value: 0,
                    ty: dummy_ty.clone(),
                },
                Some(dummy_ty.clone()),
            );
            return Ok((dummy, dummy_ty));
        }

        // Built-in: atomic_add(a, v) → AtomicAdd (returns new value)
        if dispatch_name == "atomic_add" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "atomic_add() requires exactly 2 arguments".into(),
                    span,
                });
            }
            let (a, atomic_ty) = self.lower_expr(&args[0])?;
            let (v, _) = self.lower_expr(&args[1])?;
            let inner_ty = if let IrType::Atomic(inner) = atomic_ty {
                *inner
            } else {
                IrType::Scalar(DType::I64)
            };
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::AtomicAdd {
                    result,
                    atomic: a,
                    value: v,
                    result_ty: inner_ty.clone(),
                },
                Some(inner_ty.clone()),
            );
            return Ok((result, inner_ty));
        }

        // Built-in: some(v) → MakeSome
        if dispatch_name == "some" {
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
                IrInstr::MakeSome {
                    result,
                    value: val,
                    result_ty: result_ty.clone(),
                },
                Some(result_ty.clone()),
            );
            return Ok((result, result_ty));
        }

        // Built-in: none() → MakeNone (also handled as identifier)
        if dispatch_name == "none" {
            let result_ty = IrType::Option(Box::new(IrType::Infer));
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::MakeNone {
                    result,
                    result_ty: result_ty.clone(),
                },
                Some(result_ty.clone()),
            );
            return Ok((result, result_ty));
        }

        // Built-in: is_some(v) → IsSome
        if dispatch_name == "is_some" {
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
                IrInstr::IsSome {
                    result,
                    operand: val,
                },
                Some(ty.clone()),
            );
            return Ok((result, ty));
        }

        // Built-in: unwrap(v) → OptionUnwrap (option<T>) or ResultUnwrap (result<T,E>)
        if dispatch_name == "unwrap" {
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
                        IrInstr::ResultUnwrap {
                            result,
                            operand: val,
                            result_ty: inner_ty.clone(),
                        },
                        Some(inner_ty.clone()),
                    );
                    return Ok((result, inner_ty));
                }
                IrType::Option(inner) => {
                    let inner_ty = (**inner).clone();
                    self.builder.push_instr(
                        IrInstr::OptionUnwrap {
                            result,
                            operand: val,
                            result_ty: inner_ty.clone(),
                        },
                        Some(inner_ty.clone()),
                    );
                    return Ok((result, inner_ty));
                }
                _ => {
                    // Fallback — ValidatePass will catch remaining Infer.
                    self.builder.push_instr(
                        IrInstr::OptionUnwrap {
                            result,
                            operand: val,
                            result_ty: IrType::Infer,
                        },
                        Some(IrType::Infer),
                    );
                    return Ok((result, IrType::Infer));
                }
            }
        }

        // Built-in: ok(v) → MakeOk
        if dispatch_name == "ok" {
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
                IrInstr::MakeOk {
                    result,
                    value: val,
                    result_ty: result_ty.clone(),
                },
                Some(result_ty.clone()),
            );
            return Ok((result, result_ty));
        }

        // Built-in: err(v) → MakeErr
        if dispatch_name == "err" {
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
                IrInstr::MakeErr {
                    result,
                    value: val,
                    result_ty: result_ty.clone(),
                },
                Some(result_ty.clone()),
            );
            return Ok((result, result_ty));
        }

        // Built-in: is_ok(v) → IsOk
        if dispatch_name == "is_ok" {
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
                IrInstr::IsOk {
                    result,
                    operand: val,
                },
                Some(ty.clone()),
            );
            return Ok((result, ty));
        }

        // Built-in: is_none(v) → !IsSome
        if dispatch_name == "is_none" {
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
                IrInstr::IsSome {
                    result: is_some,
                    operand: val,
                },
                Some(bool_ty.clone()),
            );
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::UnaryOp {
                    result,
                    op: ScalarUnaryOp::Not,
                    operand: is_some,
                    ty: bool_ty.clone(),
                },
                Some(bool_ty.clone()),
            );
            return Ok((result, bool_ty));
        }

        // Built-in: is_err(v) → !IsOk
        if dispatch_name == "is_err" {
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
                IrInstr::IsOk {
                    result: is_ok,
                    operand: val,
                },
                Some(bool_ty.clone()),
            );
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::UnaryOp {
                    result,
                    op: ScalarUnaryOp::Not,
                    operand: is_ok,
                    ty: bool_ty.clone(),
                },
                Some(bool_ty.clone()),
            );
            return Ok((result, bool_ty));
        }

        // Built-in: unwrap_err(v) → ResultUnwrapErr
        if dispatch_name == "unwrap_err" {
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
                IrInstr::ResultUnwrapErr {
                    result,
                    operand: val,
                    result_ty: err_ty.clone(),
                },
                Some(err_ty.clone()),
            );
            return Ok((result, err_ty));
        }

        // Built-in intrinsic: einsum("notation", inputs...)
        if dispatch_name == "einsum" {
            return self.lower_einsum(args, span);
        }

        // Check if the callee is a closure variable in scope.
        if let Some((closure_val, IrType::Fn { ret, .. })) = self.scope.get(&callee_name).cloned() {
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
                    pass_env: true,
                },
                Some(ret_ty.clone()),
            );
            return Ok((result, ret_ty));
        }

        // Built-in: len(s) → StrLen or ListLen depending on argument type
        if dispatch_name == "len" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "len() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (operand, operand_ty) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            match &operand_ty {
                IrType::List(_) => {
                    self.builder.push_instr(
                        IrInstr::ListLen {
                            result,
                            list: operand,
                        },
                        Some(ty.clone()),
                    );
                }
                _ => {
                    self.builder
                        .push_instr(IrInstr::StrLen { result, operand }, Some(ty.clone()));
                }
            }
            return Ok((result, ty));
        }

        // Built-in: concat(s, t) → StrConcat
        if dispatch_name == "concat" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "concat() requires exactly 2 arguments".into(),
                    span,
                });
            }
            let (lhs, _) = self.lower_expr(&args[0])?;
            let (rhs, _) = self.lower_expr(&args[1])?;
            let result = self.builder.fresh_value();
            self.builder
                .push_instr(IrInstr::StrConcat { result, lhs, rhs }, Some(IrType::Str));
            return Ok((result, IrType::Str));
        }

        // Built-in: to_str(v) → ValueToStr
        if dispatch_name == "to_str" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "to_str() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (operand, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            self.builder
                .push_instr(IrInstr::ValueToStr { result, operand }, Some(IrType::Str));
            return Ok((result, IrType::Str));
        }

        // Built-in: to_f64(v) → Cast to f64
        if dispatch_name == "to_f64" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "to_f64() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (operand, from_ty) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::Cast {
                    result,
                    operand,
                    from_ty,
                    to_ty: IrType::Scalar(DType::F64),
                },
                Some(IrType::Scalar(DType::F64)),
            );
            return Ok((result, IrType::Scalar(DType::F64)));
        }

        // Built-in: format("...", args...) — split on "{}" and concat with args
        if dispatch_name == "format" {
            if args.is_empty() {
                return Err(LowerError::Unsupported {
                    detail: "format() requires at least 1 argument (the format string)".into(),
                    span,
                });
            }
            // First arg must be a string literal.
            let fmt_str = match &args[0] {
                AstExpr::StringLit { value, .. } => value.clone(),
                _ => {
                    return Err(LowerError::Unsupported {
                        detail: "format() first argument must be a string literal".into(),
                        span,
                    })
                }
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
                    IrInstr::ValueToStr {
                        result: s,
                        operand: v,
                    },
                    Some(IrType::Str),
                );
                arg_vals.push(s);
            }
            // Build the concatenated string: piece[0] + arg[0] + piece[1] + arg[1] + ...
            // Start with the first piece as a ConstStr.
            let mut acc = {
                let r = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstStr {
                        result: r,
                        value: pieces[0].to_owned(),
                    },
                    Some(IrType::Str),
                );
                r
            };
            for i in 0..n_holes {
                // Concat with the argument.
                let after_arg = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::StrConcat {
                        result: after_arg,
                        lhs: acc,
                        rhs: arg_vals[i],
                    },
                    Some(IrType::Str),
                );
                // Concat with the next piece.
                let next_piece = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstStr {
                        result: next_piece,
                        value: pieces[i + 1].to_owned(),
                    },
                    Some(IrType::Str),
                );
                acc = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::StrConcat {
                        result: acc,
                        lhs: after_arg,
                        rhs: next_piece,
                    },
                    Some(IrType::Str),
                );
            }
            return Ok((acc, IrType::Str));
        }

        // Built-in: print(v) → Print (returns unit, we return a dummy i64 zero for now)
        if dispatch_name == "print" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "print() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (operand, _) = self.lower_expr(&args[0])?;
            self.builder.push_instr(IrInstr::Print { operand }, None);
            // Return a dummy i64 zero as the "unit" value.
            let dummy = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: dummy,
                    value: 0,
                    ty: ty.clone(),
                },
                Some(ty.clone()),
            );
            return Ok((dummy, ty));
        }

        // Built-in: read_line() → ReadLine
        if dispatch_name == "read_line" {
            if !args.is_empty() {
                return Err(LowerError::Unsupported {
                    detail: "read_line() takes no arguments".into(),
                    span,
                });
            }
            let result = self.builder.fresh_value();
            self.builder
                .push_instr(IrInstr::ReadLine { result }, Some(IrType::Str));
            return Ok((result, IrType::Str));
        }

        // Built-in: read_i64() → ReadI64
        if dispatch_name == "read_i64" {
            if !args.is_empty() {
                return Err(LowerError::Unsupported {
                    detail: "read_i64() takes no arguments".into(),
                    span,
                });
            }
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder
                .push_instr(IrInstr::ReadI64 { result }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: read_f64() → ReadF64
        if dispatch_name == "read_f64" {
            if !args.is_empty() {
                return Err(LowerError::Unsupported {
                    detail: "read_f64() takes no arguments".into(),
                    span,
                });
            }
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::F64);
            self.builder
                .push_instr(IrInstr::ReadF64 { result }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: parse_i64(s) → ParseI64 → option<i64>
        if dispatch_name == "parse_i64" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "parse_i64() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (operand, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Option(Box::new(IrType::Scalar(DType::I64)));
            self.builder
                .push_instr(IrInstr::ParseI64 { result, operand }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: parse_f64(s) → ParseF64 → option<f64>
        if dispatch_name == "parse_f64" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "parse_f64() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (operand, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Option(Box::new(IrType::Scalar(DType::F64)));
            self.builder
                .push_instr(IrInstr::ParseF64 { result, operand }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: str_index(s, i) → StrIndex → i64
        if dispatch_name == "str_index" {
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
            self.builder.push_instr(
                IrInstr::StrIndex {
                    result,
                    string,
                    index,
                },
                Some(ty.clone()),
            );
            return Ok((result, ty));
        }

        // Built-in: slice(s, start, end) → StrSlice → str
        if dispatch_name == "slice" {
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
            self.builder.push_instr(
                IrInstr::StrSlice {
                    result,
                    string,
                    start,
                    end,
                },
                Some(IrType::Str),
            );
            return Ok((result, IrType::Str));
        }

        // Built-in: find(s, sub) → StrFind → option<i64>
        if dispatch_name == "find" {
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
            self.builder.push_instr(
                IrInstr::StrFind {
                    result,
                    haystack,
                    needle,
                },
                Some(ty.clone()),
            );
            return Ok((result, ty));
        }

        // Built-in: str_replace(s, old, new) → StrReplace → str
        if dispatch_name == "str_replace" {
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
            self.builder.push_instr(
                IrInstr::StrReplace {
                    result,
                    string,
                    from,
                    to,
                },
                Some(IrType::Str),
            );
            return Ok((result, IrType::Str));
        }

        // Built-in: list() or list(a, b, c) → ListNew + optional ListPush
        if dispatch_name == "list" {
            // Determine element type from binding_ty annotation or first arg
            let elem_ty = if let Some(IrType::List(inner)) = &self.binding_ty {
                *inner.clone()
            } else if !args.is_empty() {
                // Infer from first argument
                let (_, first_ty) = self.lower_expr(&args[0])?;
                first_ty
            } else {
                IrType::Infer
            };

            let list_ty = IrType::List(Box::new(elem_ty.clone()));

            if args.is_empty() {
                // Empty list
                let result = self.builder.fresh_value();
                self.builder.push_instr(IrInstr::ListNew { result, elem_ty }, Some(list_ty.clone()));
                return Ok((result, list_ty));
            }

            // list(a, b, c) — create list and push each element
            let result = self.builder.fresh_value();
            self.builder.push_instr(IrInstr::ListNew { result, elem_ty: elem_ty.clone() }, Some(list_ty.clone()));

            for arg in args {
                let (val, _) = self.lower_expr(arg)?;
                self.builder.push_instr(IrInstr::ListPush { list: result, value: val }, None);
            }

            return Ok((result, list_ty));
        }

        // Built-in: push(lst, val) / list_push(lst, val) → ListPush — append to list
        if dispatch_name == "push" || callee.name == "list_push" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "push() requires 2 arguments: (list, value)".into(),
                    span,
                });
            }
            let (list, list_ty) = self.lower_expr(&args[0])?;
            let (value, value_ty) = self.lower_expr(&args[1])?;
            // Pushing a concrete struct into a `list<dyn Trait>` coerces it
            // rather than failing validation with "type mismatch: dyn Trait vs
            // Struct". A heterogeneous collection is the main reason trait
            // objects exist, so refusing this made the feature close to
            // useless. See known-issues #18.
            let elem_expected = match &list_ty {
                IrType::List(elem) if matches!(**elem, IrType::TraitObject { .. }) => {
                    Some((**elem).clone())
                }
                _ => None,
            };
            let value = match elem_expected {
                Some(exp) => {
                    self.coerce_to_trait_object(value, value_ty, &exp, args[1].span())?.0
                }
                None => value,
            };
            self.builder
                .push_instr(IrInstr::ListPush { list, value }, None);
            let dummy = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: dummy,
                    value: 0,
                    ty: ty.clone(),
                },
                Some(ty.clone()),
            );
            return Ok((dummy, ty));
        }

        // Built-in: pop(lst) → ListPop → elem  (alias for list_pop)
        if dispatch_name == "pop" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "pop() requires 1 argument: (list)".into(),
                    span,
                });
            }
            let (list, list_ty) = self.lower_expr(&args[0])?;
            let elem_ty = if let IrType::List(inner) = &list_ty {
                *inner.clone()
            } else {
                IrType::Scalar(DType::I64)
            };
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ListPop {
                    result,
                    list,
                    elem_ty: elem_ty.clone(),
                },
                Some(elem_ty.clone()),
            );
            return Ok((result, elem_ty));
        }

        // Built-in: list_len(lst) → ListLen → i64
        if dispatch_name == "list_len" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "list_len() requires 1 argument".into(),
                    span,
                });
            }
            let (list, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder
                .push_instr(IrInstr::ListLen { result, list }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: list_get(lst, i) → ListGet → elem
        if dispatch_name == "list_get" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "list_get() requires 2 arguments: (list, index)".into(),
                    span,
                });
            }
            let (list, list_ty) = self.lower_expr(&args[0])?;
            let (index, _) = self.lower_expr(&args[1])?;
            let elem_ty = if let IrType::List(inner) = &list_ty {
                *inner.clone()
            } else {
                IrType::Scalar(DType::I64)
            };
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ListGet {
                    result,
                    list,
                    index,
                    elem_ty: elem_ty.clone(),
                },
                Some(elem_ty.clone()),
            );
            return Ok((result, elem_ty));
        }

        // Built-in: list_set(lst, i, val) → ListSet
        if dispatch_name == "list_set" {
            if args.len() != 3 {
                return Err(LowerError::Unsupported {
                    detail: "list_set() requires 3 arguments: (list, index, value)".into(),
                    span,
                });
            }
            let (list, _) = self.lower_expr(&args[0])?;
            let (index, _) = self.lower_expr(&args[1])?;
            let (value, _) = self.lower_expr(&args[2])?;
            self.builder
                .push_instr(IrInstr::ListSet { list, index, value }, None);
            let dummy = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: dummy,
                    value: 0,
                    ty: ty.clone(),
                },
                Some(ty.clone()),
            );
            return Ok((dummy, ty));
        }

        // Built-in: list_pop(lst) → ListPop → elem
        if dispatch_name == "list_pop" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "list_pop() requires 1 argument".into(),
                    span,
                });
            }
            let (list, list_ty) = self.lower_expr(&args[0])?;
            let elem_ty = if let IrType::List(inner) = &list_ty {
                *inner.clone()
            } else {
                IrType::Scalar(DType::I64)
            };
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ListPop {
                    result,
                    list,
                    elem_ty: elem_ty.clone(),
                },
                Some(elem_ty.clone()),
            );
            return Ok((result, elem_ty));
        }

        // Built-in: list_remove(lst, idx) → ListRemove → elem
        if dispatch_name == "list_remove" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "list_remove() requires 2 arguments (list, index)".into(),
                    span,
                });
            }
            let (list, list_ty) = self.lower_expr(&args[0])?;
            let (idx, _) = self.lower_expr(&args[1])?;
            let elem_ty = if let IrType::List(inner) = &list_ty {
                *inner.clone()
            } else {
                IrType::Scalar(DType::I64)
            };
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BuiltinCall {
                    result,
                    name: "list_remove".to_string(),
                    args: vec![list, idx],
                    result_ty: elem_ty.clone(),
                },
                Some(elem_ty.clone()),
            );
            return Ok((result, elem_ty));
        }

        // Built-in: list_insert(lst, idx, val) → ListInsert → i64 0 (side-effecting)
        if dispatch_name == "list_insert" {
            if args.len() != 3 {
                return Err(LowerError::Unsupported {
                    detail: "list_insert() requires 3 arguments (list, index, value)".into(),
                    span,
                });
            }
            let (list, _) = self.lower_expr(&args[0])?;
            let (idx, _) = self.lower_expr(&args[1])?;
            let (val, _) = self.lower_expr(&args[2])?;
            let dummy_ty = IrType::Scalar(DType::I64);
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BuiltinCall {
                    result,
                    name: "list_insert".to_string(),
                    args: vec![list, idx, val],
                    result_ty: dummy_ty.clone(),
                },
                Some(dummy_ty.clone()),
            );
            return Ok((result, dummy_ty));
        }

        // Built-in: array_to_list(arr) → convert [T; N] to list<T>
        if dispatch_name == "array_to_list" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "array_to_list() requires exactly 1 argument (an array)".into(),
                    span,
                });
            }
            let (arr_val, arr_ty) = self.lower_expr(&args[0])?;
            let (elem_ty, arr_len) = match &arr_ty {
                IrType::Array { elem, len } => ((**elem).clone(), *len),
                _ => {
                    return Err(LowerError::Unsupported {
                        detail: "array_to_list() requires an array argument".into(),
                        span,
                    });
                }
            };
            let list_ty = IrType::List(Box::new(elem_ty.clone()));
            let list = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ListNew { result: list, elem_ty: elem_ty.clone() },
                Some(list_ty.clone()),
            );
            for i in 0..arr_len {
                let idx = self.builder.fresh_value();
                let idx_ty = IrType::Scalar(DType::I64);
                self.builder.push_instr(
                    IrInstr::ConstInt { result: idx, value: i as i64, ty: idx_ty.clone() },
                    Some(idx_ty.clone()),
                );
                let elem = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ArrayLoad {
                        result: elem,
                        array: arr_val,
                        index: idx,
                        elem_ty: elem_ty.clone(),
                    },
                    Some(elem_ty.clone()),
                );
                self.builder.push_instr(
                    IrInstr::ListPush { list, value: elem },
                    None,
                );
            }
            return Ok((list, list_ty));
        }

        // Built-in: map() → MapNew — create an empty hash map (keys: str, values: i64 default)
        if dispatch_name == "map" {
            if !args.is_empty() {
                return Err(LowerError::Unsupported {
                    detail: "map() takes no arguments — it creates an empty hash map".into(),
                    span,
                });
            }
            // Use binding_ty from `val m: map<K, V> = map()` annotation if available.
            let (key_ty, val_ty) = if let Some(IrType::Map(k, v)) = &self.binding_ty {
                (*k.clone(), *v.clone())
            } else {
                (IrType::Str, IrType::Scalar(DType::I64))
            };
            let result = self.builder.fresh_value();
            let map_ty = IrType::Map(Box::new(key_ty.clone()), Box::new(val_ty.clone()));
            self.builder.push_instr(
                IrInstr::MapNew {
                    result,
                    key_ty,
                    val_ty,
                },
                Some(map_ty.clone()),
            );
            return Ok((result, map_ty));
        }

        // Built-in: map_set(m, k, v) → MapSet
        if dispatch_name == "map_set" {
            if args.len() != 3 {
                return Err(LowerError::Unsupported {
                    detail: "map_set() requires 3 arguments: (map, key, value)".into(),
                    span,
                });
            }
            let (map, _) = self.lower_expr(&args[0])?;
            let (key, _) = self.lower_expr(&args[1])?;
            let (value, _) = self.lower_expr(&args[2])?;
            self.builder
                .push_instr(IrInstr::MapSet { map, key, value }, None);
            let dummy = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: dummy,
                    value: 0,
                    ty: ty.clone(),
                },
                Some(ty.clone()),
            );
            return Ok((dummy, ty));
        }

        // Built-in: map_get(m, k) → MapGet → option<val_ty>
        if dispatch_name == "map_get" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "map_get() requires 2 arguments: (map, key)".into(),
                    span,
                });
            }
            let (map, map_ty) = self.lower_expr(&args[0])?;
            let (key, _) = self.lower_expr(&args[1])?;
            let val_ty = if let IrType::Map(_, v) = &map_ty {
                *v.clone()
            } else {
                IrType::Scalar(DType::I64)
            };
            let opt_ty = IrType::Option(Box::new(val_ty.clone()));
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::MapGet {
                    result,
                    map,
                    key,
                    val_ty,
                },
                Some(opt_ty.clone()),
            );
            return Ok((result, opt_ty));
        }

        // Built-in: map_contains(m, k) → MapContains → bool
        if dispatch_name == "map_contains" {
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
            self.builder
                .push_instr(IrInstr::MapContains { result, map, key }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: map_remove(m, k) → MapRemove
        if dispatch_name == "map_remove" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "map_remove() requires 2 arguments: (map, key)".into(),
                    span,
                });
            }
            let (map, _) = self.lower_expr(&args[0])?;
            let (key, _) = self.lower_expr(&args[1])?;
            self.builder
                .push_instr(IrInstr::MapRemove { map, key }, None);
            let dummy = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: dummy,
                    value: 0,
                    ty: ty.clone(),
                },
                Some(ty.clone()),
            );
            return Ok((dummy, ty));
        }

        // Built-in: map_len(m) → MapLen → i64
        if dispatch_name == "map_len" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "map_len() requires 1 argument".into(),
                    span,
                });
            }
            let (map, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder
                .push_instr(IrInstr::MapLen { result, map }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // ── Phase 56: File I/O builtins ──────────────────────────────────────

        // Built-in: file_read_all(path) → FileReadAll → result<str, str>
        if dispatch_name == "file_read_all" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "file_read_all() requires 1 argument".into(),
                    span,
                });
            }
            let (path, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::ResultType(Box::new(IrType::Str), Box::new(IrType::Str));
            self.builder
                .push_instr(IrInstr::FileReadAll { result, path }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: file_write_all(path, content) → FileWriteAll → result<i64, str>
        if dispatch_name == "file_write_all" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "file_write_all() requires 2 arguments".into(),
                    span,
                });
            }
            let (path, _) = self.lower_expr(&args[0])?;
            let (content, _) = self.lower_expr(&args[1])?;
            let result = self.builder.fresh_value();
            let ty =
                IrType::ResultType(Box::new(IrType::Scalar(DType::I64)), Box::new(IrType::Str));
            self.builder.push_instr(
                IrInstr::FileWriteAll {
                    result,
                    path,
                    content,
                },
                Some(ty.clone()),
            );
            return Ok((result, ty));
        }

        // Built-in: file_exists(path) → FileExists → bool
        if dispatch_name == "file_exists" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "file_exists() requires 1 argument".into(),
                    span,
                });
            }
            let (path, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::Bool);
            self.builder
                .push_instr(IrInstr::FileExists { result, path }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: file_lines(path) → FileLines → list<str>
        if dispatch_name == "file_lines" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "file_lines() requires 1 argument".into(),
                    span,
                });
            }
            let (path, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::List(Box::new(IrType::Str));
            self.builder
                .push_instr(IrInstr::FileLines { result, path }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // ── Database operations ─────────────────────────────────────────────
        if dispatch_name == "db_open" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "db_open(path) requires 1 argument".into(),
                    span,
                });
            }
            let (path, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder
                .push_instr(IrInstr::DbOpen { result, path }, Some(ty.clone()));
            return Ok((result, ty));
        }
        if dispatch_name == "db_exec" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "db_exec(db, sql) requires 2 arguments".into(),
                    span,
                });
            }
            let (db, _) = self.lower_expr(&args[0])?;
            let (sql, _) = self.lower_expr(&args[1])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder
                .push_instr(IrInstr::DbExec { result, db, sql }, Some(ty.clone()));
            return Ok((result, ty));
        }
        if dispatch_name == "db_exec_params" {
            if args.len() != 3 {
                return Err(LowerError::Unsupported {
                    detail: "db_exec_params(db, sql, params) requires 3 arguments".into(),
                    span,
                });
            }
            let (db, _) = self.lower_expr(&args[0])?;
            let (sql, _) = self.lower_expr(&args[1])?;
            let (params, _) = self.lower_expr(&args[2])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::DbExecParams {
                    result,
                    db,
                    sql,
                    params,
                },
                Some(ty.clone()),
            );
            return Ok((result, ty));
        }
        if dispatch_name == "db_query" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "db_query(db, sql) requires 2 arguments".into(),
                    span,
                });
            }
            let (db, _) = self.lower_expr(&args[0])?;
            let (sql, _) = self.lower_expr(&args[1])?;
            let result = self.builder.fresh_value();
            let ty = IrType::List(Box::new(IrType::List(Box::new(IrType::Str))));
            self.builder
                .push_instr(IrInstr::DbQuery { result, db, sql }, Some(ty.clone()));
            return Ok((result, ty));
        }
        if dispatch_name == "db_query_params" {
            if args.len() != 3 {
                return Err(LowerError::Unsupported {
                    detail: "db_query_params(db, sql, params) requires 3 arguments".into(),
                    span,
                });
            }
            let (db, _) = self.lower_expr(&args[0])?;
            let (sql, _) = self.lower_expr(&args[1])?;
            let (params, _) = self.lower_expr(&args[2])?;
            let result = self.builder.fresh_value();
            let ty = IrType::List(Box::new(IrType::List(Box::new(IrType::Str))));
            self.builder.push_instr(
                IrInstr::DbQueryParams {
                    result,
                    db,
                    sql,
                    params,
                },
                Some(ty.clone()),
            );
            return Ok((result, ty));
        }
        if dispatch_name == "db_close" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "db_close(db) requires 1 argument".into(),
                    span,
                });
            }
            let (db, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder
                .push_instr(IrInstr::DbClose { result, db }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // ── Phase 89: Mutable cell (for closure captures) ───────────────────
        // cell(v) → list containing one element (shared via Rc)
        // cell_get(c) → read element 0
        // cell_set(c, v) → write element 0

        if dispatch_name == "cell" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "cell(v) requires 1 argument".into(),
                    span,
                });
            }
            let (val_v, val_ty) = self.lower_expr(&args[0])?;
            let list_ty = IrType::List(Box::new(val_ty.clone()));
            let list = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ListNew {
                    result: list,
                    elem_ty: val_ty.clone(),
                },
                Some(list_ty.clone()),
            );
            self.builder
                .push_instr(IrInstr::ListPush { list, value: val_v }, None);
            return Ok((list, list_ty));
        }
        if dispatch_name == "cell_get" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "cell_get(c) requires 1 argument".into(),
                    span,
                });
            }
            let (cell, cell_ty) = self.lower_expr(&args[0])?;
            let elem_ty = if let IrType::List(inner) = &cell_ty {
                *inner.clone()
            } else {
                IrType::Infer
            };
            let zero = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: zero,
                    value: 0,
                    ty: IrType::Scalar(DType::I64),
                },
                Some(IrType::Scalar(DType::I64)),
            );
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ListGet {
                    result,
                    list: cell,
                    index: zero,
                    elem_ty: elem_ty.clone(),
                },
                Some(elem_ty.clone()),
            );
            return Ok((result, elem_ty));
        }
        if dispatch_name == "cell_set" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "cell_set(c, v) requires 2 arguments".into(),
                    span,
                });
            }
            let (cell, _) = self.lower_expr(&args[0])?;
            let (new_val, _) = self.lower_expr(&args[1])?;
            let zero = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: zero,
                    value: 0,
                    ty: IrType::Scalar(DType::I64),
                },
                Some(IrType::Scalar(DType::I64)),
            );
            self.builder.push_instr(
                IrInstr::ListSet {
                    list: cell,
                    index: zero,
                    value: new_val,
                },
                None,
            );
            let unit = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: unit,
                    value: 0,
                    ty: IrType::Scalar(DType::I64),
                },
                Some(IrType::Scalar(DType::I64)),
            );
            return Ok((unit, IrType::Scalar(DType::I64)));
        }

        // ── Phase 88: TCP network I/O ────────────────────────────────────────

        if dispatch_name == "tcp_connect" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "tcp_connect(host, port) requires 2 args".into(),
                    span,
                });
            }
            let (host, _) = self.lower_expr(&args[0])?;
            let (port, _) = self.lower_expr(&args[1])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder
                .push_instr(IrInstr::TcpConnect { result, host, port }, Some(ty.clone()));
            return Ok((result, ty));
        }
        if dispatch_name == "tcp_listen" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "tcp_listen(port) requires 1 arg".into(),
                    span,
                });
            }
            let (port, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder
                .push_instr(IrInstr::TcpListen { result, port }, Some(ty.clone()));
            return Ok((result, ty));
        }
        if dispatch_name == "tcp_accept" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "tcp_accept(listener) requires 1 arg".into(),
                    span,
                });
            }
            let (listener, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder
                .push_instr(IrInstr::TcpAccept { result, listener }, Some(ty.clone()));
            return Ok((result, ty));
        }
        if dispatch_name == "tcp_read" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "tcp_read(conn) requires 1 arg".into(),
                    span,
                });
            }
            let (conn, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Str;
            self.builder
                .push_instr(IrInstr::TcpRead { result, conn }, Some(ty.clone()));
            return Ok((result, ty));
        }
        if dispatch_name == "tcp_write" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "tcp_write(conn, data) requires 2 args".into(),
                    span,
                });
            }
            let (conn, _) = self.lower_expr(&args[0])?;
            let (data, _) = self.lower_expr(&args[1])?;
            let unit = self.builder.fresh_value();
            self.builder
                .push_instr(IrInstr::TcpWrite { conn, data }, None);
            // Return a dummy unit value.
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: unit,
                    value: 0,
                    ty: IrType::Scalar(DType::I64),
                },
                Some(IrType::Scalar(DType::I64)),
            );
            return Ok((unit, IrType::Scalar(DType::I64)));
        }
        if dispatch_name == "tcp_close" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "tcp_close(conn) requires 1 arg".into(),
                    span,
                });
            }
            let (conn, _) = self.lower_expr(&args[0])?;
            let unit = self.builder.fresh_value();
            self.builder.push_instr(IrInstr::TcpClose { conn }, None);
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: unit,
                    value: 0,
                    ty: IrType::Scalar(DType::I64),
                },
                Some(IrType::Scalar(DType::I64)),
            );
            return Ok((unit, IrType::Scalar(DType::I64)));
        }

        // ── Phase 58: Extended collection builtins ───────────────────────────

        // Built-in: list_contains(list, val) → ListContains → bool
        if dispatch_name == "list_contains" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "list_contains() requires 2 arguments".into(),
                    span,
                });
            }
            let (list, _) = self.lower_expr(&args[0])?;
            let (value, _) = self.lower_expr(&args[1])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::Bool);
            self.builder.push_instr(
                IrInstr::ListContains {
                    result,
                    list,
                    value,
                },
                Some(ty.clone()),
            );
            return Ok((result, ty));
        }

        // Built-in: list_sort(list) → ListSort (side-effecting, returns unit-like dummy)
        if dispatch_name == "list_sort" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "list_sort() requires 1 argument".into(),
                    span,
                });
            }
            let (list, _) = self.lower_expr(&args[0])?;
            self.builder.push_instr(IrInstr::ListSort { list }, None);
            let dummy = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: dummy,
                    value: 0,
                    ty: ty.clone(),
                },
                Some(ty.clone()),
            );
            return Ok((dummy, ty));
        }

        // Built-in: map_keys(map) → MapKeys → list<str>
        if dispatch_name == "map_keys" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "map_keys() requires 1 argument".into(),
                    span,
                });
            }
            let (map, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::List(Box::new(IrType::Str));
            self.builder
                .push_instr(IrInstr::MapKeys { result, map }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: map_values(map) → MapValues → list<?>
        if dispatch_name == "map_values" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "map_values() requires 1 argument".into(),
                    span,
                });
            }
            let (map, map_ty) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let val_ty = if let IrType::Map(_, v) = &map_ty {
                *v.clone()
            } else {
                IrType::Scalar(DType::I64)
            };
            let ty = IrType::List(Box::new(val_ty));
            self.builder
                .push_instr(IrInstr::MapValues { result, map }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: list_concat(a, b) → ListConcat → list
        if dispatch_name == "list_concat" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "list_concat() requires 2 arguments".into(),
                    span,
                });
            }
            let (lhs, lhs_ty) = self.lower_expr(&args[0])?;
            let (rhs, _) = self.lower_expr(&args[1])?;
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ListConcat { result, lhs, rhs },
                Some(lhs_ty.clone()),
            );
            return Ok((result, lhs_ty));
        }

        // Built-in: list_slice(list, start, end) → ListSlice → list
        if dispatch_name == "list_slice" {
            if args.len() != 3 {
                return Err(LowerError::Unsupported {
                    detail: "list_slice() requires 3 arguments".into(),
                    span,
                });
            }
            let (list, list_ty) = self.lower_expr(&args[0])?;
            let (start, _) = self.lower_expr(&args[1])?;
            let (end, _) = self.lower_expr(&args[2])?;
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ListSlice {
                    result,
                    list,
                    start,
                    end,
                },
                Some(list_ty.clone()),
            );
            return Ok((result, list_ty));
        }

        // ── Phase 59: Process / environment builtins ─────────────────────────

        // Built-in: exit(code) → ProcessExit (does not return)
        if dispatch_name == "exit" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "exit() requires 1 argument".into(),
                    span,
                });
            }
            let (code, _) = self.lower_expr(&args[0])?;
            self.builder.push_instr(IrInstr::ProcessExit { code }, None);
            let dummy = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: dummy,
                    value: 0,
                    ty: ty.clone(),
                },
                Some(ty.clone()),
            );
            return Ok((dummy, ty));
        }

        // Built-in: args() → ProcessArgs → list<str>
        if dispatch_name == "args" {
            if !args.is_empty() {
                return Err(LowerError::Unsupported {
                    detail: "args() takes no arguments".into(),
                    span,
                });
            }
            let result = self.builder.fresh_value();
            let ty = IrType::List(Box::new(IrType::Str));
            self.builder
                .push_instr(IrInstr::ProcessArgs { result }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: env_var(name) → EnvVar → option<str>
        if dispatch_name == "env_var" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "env_var() requires 1 argument".into(),
                    span,
                });
            }
            let (name, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Option(Box::new(IrType::Str));
            self.builder
                .push_instr(IrInstr::EnvVar { result, name }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in: panic(msg) → Panic (terminator; does not return)
        if dispatch_name == "panic" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "panic() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (msg, _) = self.lower_expr(&args[0])?;
            if !self.builder.is_current_block_terminated() {
                self.builder.push_instr(
                    IrInstr::Panic { msg, span_byte: Some(span.start.0) },
                    None,
                );
            }
            // Return a dummy value so the type-checker is happy.
            let dummy = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            return Ok((dummy, ty));
        }

        // Built-in: assert(cond) — lowers to: if cond { continue } else { panic("assertion failed") }
        if dispatch_name == "assert" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "assert() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (cond, _) = self.lower_expr(&args[0])?;
            let then_block = self.builder.create_block(Some("assert_ok"));
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
                IrInstr::ConstStr {
                    result: msg_val,
                    value: "assertion failed".into(),
                },
                Some(IrType::Str),
            );
            // Attach the assert's own source position to the Panic.
            //
            // Without this the failure reported the *previous* statement — a line
            // that succeeded, which is worse than no location at all. The span
            // recorded by `lower_stmt` is consumed by the first instruction the
            // condition emits, and when the condition is const-foldable that
            // instruction is then deleted by DCE. `span_table` is keyed by
            // (block_id, instr_idx) and no pass maintains it, so the entry is
            // simply lost and the interpreter's sticky `last_byte` still holds
            // the previous statement's position.
            //
            // Recording it here is robust because `assert_fail` holds exactly two
            // instructions, both live: the message and the Panic that consumes it.
            // See known-issues #20 for the general span-table staleness, which
            // this does not fix.
            self.builder.push_instr(
                IrInstr::Panic { msg: msg_val, span_byte: Some(span.start.0) },
                None,
            );
            // then_block: jump to merge
            self.builder.set_current_block(then_block);
            self.builder.push_instr(
                IrInstr::Br {
                    target: merge_block,
                    args: vec![],
                },
                None,
            );
            // merge_block: continue with dummy zero
            self.builder.set_current_block(merge_block);
            let dummy = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: dummy,
                    value: 0,
                    ty: ty.clone(),
                },
                Some(ty.clone()),
            );
            return Ok((dummy, ty));
        }

        // Built-in: grad(v) → MakeGrad(value=v, tangent=1.0)
        if dispatch_name == "grad" {
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
                IrInstr::ConstFloat {
                    result: one,
                    value: 1.0,
                    ty: IrType::Scalar(DType::F64),
                },
                Some(IrType::Scalar(DType::F64)),
            );
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::MakeGrad {
                    result,
                    value: val,
                    tangent: one,
                    ty: result_ty.clone(),
                },
                Some(result_ty.clone()),
            );
            return Ok((result, result_ty));
        }

        // Built-in: grad_of(closure, x) → numerical derivative via central finite differences
        // Returns (f(x+h) - f(x-h)) / (2*h)  where h = 1e-7
        if dispatch_name == "grad_of" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "grad_of() requires exactly 2 arguments: grad_of(closure, x)".into(),
                    span,
                });
            }
            let (closure_val, closure_ty) = self.lower_expr(&args[0])?;
            let (mut x_val, mut x_ty) = self.lower_expr(&args[1])?;
            // Prefer the closure's param type for arithmetic and calls so
            // CallClosure args/results match the closure signature.
            let (param_ty, ret_ty): (IrType, IrType) = match &closure_ty {
                IrType::Fn { params, ret } if !params.is_empty() => {
                    (params[0].clone(), (*ret).as_ref().clone())
                }
                _ => (x_ty.clone(), x_ty.clone()),
            };
            // If the provided x has a different type, cast it to the closure param type.
            if x_ty != param_ty {
                let cast = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::Cast {
                        result: cast,
                        operand: x_val,
                        from_ty: x_ty.clone(),
                        to_ty: param_ty.clone(),
                    },
                    Some(param_ty.clone()),
                );
                x_val = cast;
                x_ty = param_ty.clone();
            }
            // h = 1e-3 (step for central finite difference; large enough for f32 precision)
            let h_val = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ConstFloat {
                    result: h_val,
                    value: 1e-3,
                    ty: x_ty.clone(),
                },
                Some(x_ty.clone()),
            );
            // x_plus = x + h
            let x_plus = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BinOp {
                    result: x_plus,
                    op: BinOp::Add,
                    lhs: x_val,
                    rhs: h_val,
                    ty: x_ty.clone(),
                },
                Some(x_ty.clone()),
            );
            // x_minus = x - h
            let x_minus = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BinOp {
                    result: x_minus,
                    op: BinOp::Sub,
                    lhs: x_val,
                    rhs: h_val,
                    ty: x_ty.clone(),
                },
                Some(x_ty.clone()),
            );
            // f_plus = closure(x_plus)
            let f_plus = self.builder.fresh_value();
            // Cast arg to closure param type if needed, call closure.
            let f_plus_call_arg = if x_ty != param_ty {
                let cast_arg = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::Cast {
                        result: cast_arg,
                        operand: x_plus,
                        from_ty: x_ty.clone(),
                        to_ty: param_ty.clone(),
                    },
                    Some(param_ty.clone()),
                );
                cast_arg
            } else {
                x_plus
            };
            self.builder.push_instr(
                IrInstr::CallClosure {
                    result: Some(f_plus),
                    closure: closure_val,
                    args: vec![f_plus_call_arg],
                    result_ty: ret_ty.clone(),
                    pass_env: true,
                },
                Some(ret_ty.clone()),
            );
            // f_minus = closure(x_minus)
            let f_minus = self.builder.fresh_value();
            let f_minus_call_arg = if x_ty != param_ty {
                let cast_arg = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::Cast {
                        result: cast_arg,
                        operand: x_minus,
                        from_ty: x_ty.clone(),
                        to_ty: param_ty.clone(),
                    },
                    Some(param_ty.clone()),
                );
                cast_arg
            } else {
                x_minus
            };
            self.builder.push_instr(
                IrInstr::CallClosure {
                    result: Some(f_minus),
                    closure: closure_val,
                    args: vec![f_minus_call_arg],
                    result_ty: ret_ty.clone(),
                    pass_env: true,
                },
                Some(ret_ty.clone()),
            );
            // Perform arithmetic in the closure's return type (conservative fix).
            // diff = f_plus - f_minus
            let diff = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BinOp {
                    result: diff,
                    op: BinOp::Sub,
                    lhs: f_plus,
                    rhs: f_minus,
                    ty: ret_ty.clone(),
                },
                Some(ret_ty.clone()),
            );
            // two_h = 2.0 * h (compute in return type)
            let two = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ConstFloat {
                    result: two,
                    value: 2.0,
                    ty: ret_ty.clone(),
                },
                Some(ret_ty.clone()),
            );
            // Cast h (which is in param type x_ty) to return type if needed
            // so multiplication happens in the same type.
            let h_val_ret = if x_ty != ret_ty {
                let cf = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::Cast {
                        result: cf,
                        operand: h_val,
                        from_ty: x_ty.clone(),
                        to_ty: ret_ty.clone(),
                    },
                    Some(ret_ty.clone()),
                );
                cf
            } else {
                h_val
            };
            let two_h = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BinOp {
                    result: two_h,
                    op: BinOp::Mul,
                    lhs: two,
                    rhs: h_val_ret,
                    ty: ret_ty.clone(),
                },
                Some(ret_ty.clone()),
            );
            // result = diff / two_h (in return type)
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BinOp {
                    result,
                    op: BinOp::Div,
                    lhs: diff,
                    rhs: two_h,
                    ty: ret_ty.clone(),
                },
                Some(ret_ty.clone()),
            );
            return Ok((result, ret_ty));
        }

        // Built-in: sparsify(arr) → Sparsify (convert dense array to sparse representation)
        if dispatch_name == "sparsify" {
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
                IrInstr::Sparsify {
                    result,
                    operand: val,
                    ty: result_ty.clone(),
                },
                Some(result_ty.clone()),
            );
            return Ok((result, result_ty));
        }

        // Built-in: densify(sparse) → Densify (reconstruct the dense collection)
        //
        // This returns dense data, as the name says and as the README, the LSP
        // hover signature and the runtime's `iris_densify` all describe. It used
        // to return the non-zero count as an i64 — a shortcut that contradicted
        // every one of those, and that the native backend did not implement the
        // same way, so a program's meaning depended on how it was run. The count
        // is now `nnz(s)` below.
        if dispatch_name == "densify" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "densify() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, sparse_ty) = self.lower_expr(&args[0])?;
            // The runtime reconstructs an IrisList, so the result is a list of the
            // sparse value's element type.
            let elem_ty = match &sparse_ty {
                IrType::Sparse(inner) => match &**inner {
                    IrType::Array { elem, .. } => (**elem).clone(),
                    IrType::List(elem) => (**elem).clone(),
                    other => other.clone(),
                },
                other => other.clone(),
            };
            let result_ty = IrType::List(Box::new(elem_ty));
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::Densify {
                    result,
                    operand: val,
                    ty: result_ty.clone(),
                },
                Some(result_ty.clone()),
            );
            return Ok((result, result_ty));
        }

        // Built-in: nnz(sparse) → count of stored non-zero elements.
        if dispatch_name == "nnz" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "nnz() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (val, _) = self.lower_expr(&args[0])?;
            let result_ty = IrType::Scalar(DType::I64);
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::SparseNnz {
                    result,
                    operand: val,
                },
                Some(result_ty.clone()),
            );
            return Ok((result, result_ty));
        }

        // Built-in: split(s, delim) → list<str>
        if dispatch_name == "split" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "split() requires exactly 2 arguments".to_owned(),
                    span,
                });
            }
            let (str_val, _) = self.lower_expr(&args[0])?;
            let (delim, _) = self.lower_expr(&args[1])?;
            let result = self.builder.fresh_value();
            let ret_ty = IrType::List(Box::new(IrType::Str));
            self.builder.push_instr(
                IrInstr::StrSplit {
                    result,
                    str_val,
                    delim,
                },
                Some(ret_ty.clone()),
            );
            return Ok((result, ret_ty));
        }

        // Built-in: join(lst, delim) → str
        if dispatch_name == "join" {
            if args.len() != 2 {
                return Err(LowerError::Unsupported {
                    detail: "join() requires exactly 2 arguments".to_owned(),
                    span,
                });
            }
            let (list_val, _) = self.lower_expr(&args[0])?;
            let (delim, _) = self.lower_expr(&args[1])?;
            let result = self.builder.fresh_value();
            let ret_ty = IrType::Str;
            self.builder.push_instr(
                IrInstr::StrJoin {
                    result,
                    list_val,
                    delim,
                },
                Some(ret_ty.clone()),
            );
            return Ok((result, ret_ty));
        }

        // Phase 97: time_now_ms() -> i64
        if dispatch_name == "time_now_ms" {
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder
                .push_instr(IrInstr::NowMs { result }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Phase 97: sleep_ms(n: i64) -> i64
        if dispatch_name == "sleep_ms" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "sleep_ms() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (ms, _) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            let ty = IrType::Scalar(DType::I64);
            self.builder
                .push_instr(IrInstr::SleepMs { result, ms }, Some(ty.clone()));
            return Ok((result, ty));
        }

        // Built-in string predicates: contains(s, sub), starts_with(s, p), ends_with(s, p)
        {
            let str_pred: Option<fn(ValueId, ValueId, ValueId) -> IrInstr> =
                match dispatch_name {
                    "contains" => Some(|result, haystack, needle| IrInstr::StrContains {
                        result,
                        haystack,
                        needle,
                    }),
                    "starts_with" => Some(|result, haystack, prefix| IrInstr::StrStartsWith {
                        result,
                        haystack,
                        prefix,
                    }),
                    "ends_with" => Some(|result, haystack, suffix| IrInstr::StrEndsWith {
                        result,
                        haystack,
                        suffix,
                    }),
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
                self.builder
                    .push_instr(mk(result, haystack, second), Some(ret_ty.clone()));
                return Ok((result, ret_ty));
            }
        }

        // Built-in string transforms: to_upper(s), to_lower(s), trim(s)
        {
            let str_xform: Option<fn(ValueId, ValueId) -> IrInstr> = match dispatch_name {
                "to_upper" => Some(|result, operand| IrInstr::StrToUpper { result, operand }),
                "to_lower" => Some(|result, operand| IrInstr::StrToLower { result, operand }),
                "trim" => Some(|result, operand| IrInstr::StrTrim { result, operand }),
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
                self.builder
                    .push_instr(mk(result, operand), Some(ret_ty.clone()));
                return Ok((result, ret_ty));
            }
        }

        // Built-in: repeat(s, n) → StrRepeat
        if dispatch_name == "repeat" {
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
                IrInstr::StrRepeat {
                    result,
                    operand,
                    count,
                },
                Some(ret_ty.clone()),
            );
            return Ok((result, ret_ty));
        }

        // Built-in bitwise binary: band(a,b), bor(a,b), bxor(a,b), shl(a,b), shr(a,b)
        {
            let bitbin: Option<BinOp> = match dispatch_name {
                "band" => Some(BinOp::BitAnd),
                "bor" => Some(BinOp::BitOr),
                "bxor" => Some(BinOp::BitXor),
                "shl" => Some(BinOp::Shl),
                "shr" => Some(BinOp::Shr),
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
                    IrInstr::BinOp {
                        result,
                        op,
                        lhs,
                        rhs,
                        ty: lhs_ty.clone(),
                    },
                    Some(lhs_ty.clone()),
                );
                return Ok((result, lhs_ty));
            }
        }

        // Built-in bitwise unary: bitnot(x)
        if dispatch_name == "bitnot" {
            if args.len() != 1 {
                return Err(LowerError::Unsupported {
                    detail: "bitnot() requires exactly 1 argument".into(),
                    span,
                });
            }
            let (operand, op_ty) = self.lower_expr(&args[0])?;
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::UnaryOp {
                    result,
                    op: ScalarUnaryOp::BitNot,
                    operand,
                    ty: op_ty.clone(),
                },
                Some(op_ty.clone()),
            );
            return Ok((result, op_ty));
        }

        // Built-in math unary: sqrt, abs, floor, ceil, sin, cos, tan, exp, log, log2, round, sign
        {
            let math_unary: Option<ScalarUnaryOp> = match dispatch_name {
                "sqrt" => Some(ScalarUnaryOp::Sqrt),
                "abs" => Some(ScalarUnaryOp::Abs),
                "floor" => Some(ScalarUnaryOp::Floor),
                "ceil" => Some(ScalarUnaryOp::Ceil),
                "sin" => Some(ScalarUnaryOp::Sin),
                "cos" => Some(ScalarUnaryOp::Cos),
                "tan" => Some(ScalarUnaryOp::Tan),
                "exp" => Some(ScalarUnaryOp::Exp),
                "log" => Some(ScalarUnaryOp::Log),
                "log2" => Some(ScalarUnaryOp::Log2),
                "round" => Some(ScalarUnaryOp::Round),
                "sign" => Some(ScalarUnaryOp::Sign),
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
                    IrInstr::UnaryOp {
                        result,
                        op,
                        operand,
                        ty: op_ty.clone(),
                    },
                    Some(op_ty.clone()),
                );
                let tape_op = match op {
                    ScalarUnaryOp::Sqrt => Some("sqrt"),
                    ScalarUnaryOp::Abs => Some("abs"),
                    ScalarUnaryOp::Sin => Some("sin"),
                    ScalarUnaryOp::Cos => Some("cos"),
                    ScalarUnaryOp::Exp => Some("exp"),
                    ScalarUnaryOp::Log => Some("log"),
                    _ => None,
                };
                if let Some(tape_op) = tape_op {
                    self.maybe_record_tape_result(result, &op_ty, tape_op, &[operand]);
                }
                return Ok((result, op_ty));
            }
        }

        // clamp(x, lo, hi) → min(max(x, lo), hi)
        if dispatch_name == "clamp" {
            if args.len() != 3 {
                return Err(LowerError::Unsupported {
                    detail: "clamp() requires exactly 3 arguments".into(),
                    span,
                });
            }
            let (x, x_ty) = self.lower_expr(&args[0])?;
            let (lo, _) = self.lower_expr(&args[1])?;
            let (hi, _) = self.lower_expr(&args[2])?;
            let inner = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BinOp {
                    result: inner,
                    op: BinOp::Max,
                    lhs: x,
                    rhs: lo,
                    ty: x_ty.clone(),
                },
                Some(x_ty.clone()),
            );
            let outer = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BinOp {
                    result: outer,
                    op: BinOp::Min,
                    lhs: inner,
                    rhs: hi,
                    ty: x_ty.clone(),
                },
                Some(x_ty.clone()),
            );
            return Ok((outer, x_ty));
        }

        // Built-in math binary: pow(base, exp), min(a, b), max(a, b)
        {
            let math_bin: Option<BinOp> = match dispatch_name {
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
                    IrInstr::BinOp {
                        result,
                        op,
                        lhs,
                        rhs,
                        ty: lhs_ty.clone(),
                    },
                    Some(lhs_ty.clone()),
                );
                if matches!(op, BinOp::Pow) {
                    self.maybe_record_tape_result(result, &lhs_ty, "pow", &[lhs, rhs]);
                }
                return Ok((result, lhs_ty));
            }
        }

        // ── Functional list operations: expand inline instead of BuiltinCall ──
        // These use CallClosure which the codegen handles natively, so they work
        // with both the interpreter and the binary/LLVM backend.
        if matches!(
            callee.name.as_str(),
            "list_map" | "list_filter" | "list_reduce" | "list_any" | "list_all"
        ) {
            match dispatch_name {
                "list_map" => {
                    // list_map(list, closure)
                    if args.len() != 2 {
                        return Err(LowerError::Unsupported {
                            detail: "list_map() requires 2 arguments: (list, closure)".into(),
                            span,
                        });
                    }
                    let (base_val, base_ty) = self.lower_expr(&args[0])?;
                    let elem_ty = match &base_ty {
                        IrType::List(inner) => *inner.clone(),
                        _ => IrType::Scalar(DType::I64),
                    };
                    return self.lower_list_map(base_val, elem_ty, &args[1..], span);
                }
                "list_filter" => {
                    // list_filter(list, closure)
                    if args.len() != 2 {
                        return Err(LowerError::Unsupported {
                            detail: "list_filter() requires 2 arguments: (list, closure)".into(),
                            span,
                        });
                    }
                    let (base_val, base_ty) = self.lower_expr(&args[0])?;
                    let elem_ty = match &base_ty {
                        IrType::List(inner) => *inner.clone(),
                        _ => IrType::Scalar(DType::I64),
                    };
                    return self.lower_list_filter(base_val, elem_ty, &args[1..], span);
                }
                "list_reduce" => {
                    // list_reduce(list, initial, closure)
                    if args.len() != 3 {
                        return Err(LowerError::Unsupported {
                            detail: "list_reduce() requires 3 arguments: (list, initial, closure)"
                                .into(),
                            span,
                        });
                    }
                    let (base_val, base_ty) = self.lower_expr(&args[0])?;
                    let elem_ty = match &base_ty {
                        IrType::List(inner) => *inner.clone(),
                        _ => IrType::Scalar(DType::I64),
                    };
                    return self.lower_list_fold(base_val, elem_ty, &args[1..], span);
                }
                "list_any" => {
                    // list_any(list, closure)
                    if args.len() != 2 {
                        return Err(LowerError::Unsupported {
                            detail: "list_any() requires 2 arguments: (list, closure)".into(),
                            span,
                        });
                    }
                    let (base_val, base_ty) = self.lower_expr(&args[0])?;
                    let elem_ty = match &base_ty {
                        IrType::List(inner) => *inner.clone(),
                        _ => IrType::Scalar(DType::I64),
                    };
                    return self.lower_list_any(base_val, elem_ty, &args[1..], span);
                }
                "list_all" => {
                    // list_all(list, closure)
                    if args.len() != 2 {
                        return Err(LowerError::Unsupported {
                            detail: "list_all() requires 2 arguments: (list, closure)".into(),
                            span,
                        });
                    }
                    let (base_val, base_ty) = self.lower_expr(&args[0])?;
                    let elem_ty = match &base_ty {
                        IrType::List(inner) => *inner.clone(),
                        _ => IrType::Scalar(DType::I64),
                    };
                    return self.lower_list_all(base_val, elem_ty, &args[1..], span);
                }
                _ => unreachable!(),
            }
        }

        // ── Phase 104: New runtime builtins (HTTP, JSON, Regex, DateTime, OS, etc.) ──
        // NOTE: set_*, json_parse, path_exists are NOT here — they are stdlib .iris functions.
        {
            let builtin_info: Option<(&str, IrType)> = match dispatch_name {
                // HTTP
                "http_get" => Some(("http_get", IrType::Str)),
                "http_post" => Some(("http_post", IrType::Str)),
                // JSON (json_parse is in stdlib; json_stringify is a new builtin)
                "json_stringify" => Some(("json_stringify", IrType::Str)),
                // Regex
                "regex_match" => Some(("regex_match", IrType::Scalar(DType::Bool))),
                "regex_find_all" => Some(("regex_find_all", IrType::List(Box::new(IrType::Str)))),
                "regex_replace" => Some(("regex_replace", IrType::Str)),
                "regex_replace_all" => Some(("regex_replace_all", IrType::Str)),
                // DateTime
                "datetime_now" => Some(("datetime_now", IrType::Str)),
                "datetime_timestamp" => Some(("datetime_timestamp", IrType::Scalar(DType::F64))),
                "datetime_format" => Some(("datetime_format", IrType::Str)),
                // OS / Path (path_exists is in stdlib fs.iris)
                "cwd" => Some(("cwd", IrType::Str)),
                "list_dir" => Some(("listdir", IrType::List(Box::new(IrType::Str)))),
                "path_join" => Some(("path_join", IrType::Str)),
                "mkdir" => Some(("mkdir", IrType::Scalar(DType::Bool))),
                "remove_file" => Some(("remove_file", IrType::Scalar(DType::Bool))),
                // Type introspection
                "type_of" => Some(("type_of", IrType::Str)),
                // Random
                "random" => Some(("random", IrType::Scalar(DType::F64))),
                "random_range" => Some(("random_range", IrType::Scalar(DType::I64))),
                // Seeding. `seed(n)` returns n so a run can log the seed it
                // used; `random_seed()` reports the seed in effect, generating
                // one if none was set. Together they make an evolved system
                // reproducible -- print the seed, feed it back, get the same
                // system. Without that, a self-evolving program cannot be
                // audited or replayed after a failure.
                "seed" => Some(("seed", IrType::Scalar(DType::I64))),
                "random_seed" => Some(("random_seed", IrType::Scalar(DType::I64))),
                // Hashing / Encoding
                "hash" => Some(("hash", IrType::Scalar(DType::I64))),
                "base64_encode" => Some(("base64_encode", IrType::Str)),
                "base64_decode" => Some(("base64_decode", IrType::Str)),
                // String extras
                "char_at" => Some(("char_at", IrType::Str)),
                "str_reverse" => Some(("str_reverse", IrType::Str)),

                // ── Phase 105: Async/Concurrency extensions ──
                "chan_len" => Some(("chan_len", IrType::Scalar(DType::I64))),
                "select" => Some(("select", IrType::Scalar(DType::I64))),
                "timeout" => Some(("timeout", IrType::Scalar(DType::Bool))),
                "thread_count" => Some(("thread_count", IrType::Scalar(DType::I64))),
                "par_map" => Some(("par_map", IrType::List(Box::new(IrType::Infer)))),

                // ── Phase 105: Deque (double-ended queue) ──
                "deque_new" => Some((
                    "deque_new",
                    IrType::List(Box::new(IrType::Scalar(DType::I64))),
                )),
                "deque_push_front" => Some((
                    "deque_push_front",
                    IrType::List(Box::new(IrType::Scalar(DType::I64))),
                )),
                "deque_push_back" => Some((
                    "deque_push_back",
                    IrType::List(Box::new(IrType::Scalar(DType::I64))),
                )),
                "deque_pop_front" => Some(("deque_pop_front", IrType::Scalar(DType::I64))),
                "deque_pop_back" => Some(("deque_pop_back", IrType::Scalar(DType::I64))),
                "deque_len" => Some(("deque_len", IrType::Scalar(DType::I64))),
                "deque_front" => Some(("deque_front", IrType::Scalar(DType::I64))),
                "deque_back" => Some(("deque_back", IrType::Scalar(DType::I64))),

                // ── Phase 105: Sorted collection helpers ──
                "sorted_keys" => Some(("sorted_keys", IrType::List(Box::new(IrType::Str)))),

                // ── Phase 105: BitSet ──
                "bitset_new" => Some((
                    "bitset_new",
                    IrType::List(Box::new(IrType::Scalar(DType::I64))),
                )),
                "bitset_set" => Some((
                    "bitset_set",
                    IrType::List(Box::new(IrType::Scalar(DType::I64))),
                )),
                "bitset_get" => Some(("bitset_get", IrType::Scalar(DType::Bool))),
                "bitset_count" => Some(("bitset_count", IrType::Scalar(DType::I64))),
                "bitset_clear" => Some((
                    "bitset_clear",
                    IrType::List(Box::new(IrType::Scalar(DType::I64))),
                )),

                // ── Phase 105: FFI (dynamic library loading) ──
                "ffi_open" => Some(("ffi_open", IrType::Scalar(DType::I64))),
                "ffi_call" => Some(("ffi_call", IrType::Scalar(DType::I64))),
                "ffi_close" => Some(("ffi_close", IrType::Scalar(DType::Bool))),

                // ── Phase 106: Expanded FFI — C / Python / Rust ──
                // C FFI with typed arguments
                "ffi_call_i64" => Some(("ffi_call_i64", IrType::Scalar(DType::I64))),
                "ffi_call_f64" => Some(("ffi_call_f64", IrType::Scalar(DType::F64))),
                "ffi_call_str" => Some(("ffi_call_str", IrType::Str)),
                "ffi_call_void" => Some(("ffi_call_void", IrType::Scalar(DType::I64))),
                "ffi_call_args" => Some(("ffi_call_args", IrType::Scalar(DType::I64))),
                // FFI out-parameter cells. A pointer fits in the int64 slot the
                // dispatcher already passes, so these need no new calling
                // convention — only a way for IRIS to own memory and name its
                // address. Without them every C function returning through a
                // pointer was unreachable.
                "ffi_out_new" => Some(("ffi_out_new", IrType::Scalar(DType::I64))),
                "ffi_out_free" => Some(("ffi_out_free", IrType::Scalar(DType::I64))),
                "ffi_out_get_f64" => Some(("ffi_out_get_f64", IrType::Scalar(DType::F64))),
                "ffi_out_get_i64" => Some(("ffi_out_get_i64", IrType::Scalar(DType::I64))),
                "ffi_out_get_str" => Some(("ffi_out_get_str", IrType::Str)),
                "ffi_out_set_f64" => Some(("ffi_out_set_f64", IrType::Scalar(DType::I64))),
                "ffi_out_set_i64" => Some(("ffi_out_set_i64", IrType::Scalar(DType::I64))),
                // Python FFI
                "python_eval" => Some(("python_eval", IrType::Str)),
                "python_exec" => Some(("python_exec", IrType::Scalar(DType::I64))),
                "python_call" => Some(("python_call", IrType::Str)),
                "python_version" => Some(("python_version", IrType::Str)),
                // Rust FFI (cdylib — uses same dlopen mechanism as C)
                "rust_lib_open" => Some(("rust_lib_open", IrType::Scalar(DType::I64))),
                "rust_call_i64" => Some(("rust_call_i64", IrType::Scalar(DType::I64))),
                "rust_call_f64" => Some(("rust_call_f64", IrType::Scalar(DType::F64))),
                "rust_call_void" => Some(("rust_call_void", IrType::Scalar(DType::I64))),

                // ── Phase 105: OS / System (env, exec, pid) ──
                "env_get" => Some(("env_get", IrType::Str)),
                "env_set" => Some(("env_set", IrType::Scalar(DType::Bool))),
                "exit_code" => Some(("exit_code", IrType::Scalar(DType::I64))),
                "exec_cmd" => Some(("exec_cmd", IrType::Str)),
                "pid" => Some(("pid", IrType::Scalar(DType::I64))),

                // ── Phase 105: Crypto / UUID ──
                "uuid" => Some(("uuid", IrType::Str)),
                "sha256" => Some(("sha256", IrType::Str)),
                "hex_encode" => Some(("hex_encode", IrType::Str)),

                // ── File Stream API ──
                "file_open" => Some(("file_open", IrType::Scalar(DType::I64))),
                "file_close" => Some(("file_close", IrType::Scalar(DType::Bool))),
                "file_read" => Some(("file_read", IrType::Str)),
                "file_write" => Some(("file_write", IrType::Scalar(DType::Bool))),
                "hex_decode" => Some(("hex_decode", IrType::Str)),

                // ── Phase 105: String extras ──
                "str_pad_left" => Some(("str_pad_left", IrType::Str)),
                "str_pad_right" => Some(("str_pad_right", IrType::Str)),
                "str_chars" => Some(("str_chars", IrType::List(Box::new(IrType::Str)))),
                "str_bytes" => Some((
                    "str_bytes",
                    IrType::List(Box::new(IrType::Scalar(DType::I64))),
                )),
                "str_count" => Some(("str_count", IrType::Scalar(DType::I64))),

                // ── Phase 105: Math constants and predicates ──
                "math_pi" => Some(("math_pi", IrType::Scalar(DType::F64))),
                "math_e" => Some(("math_e", IrType::Scalar(DType::F64))),
                "math_inf" => Some(("math_inf", IrType::Scalar(DType::F64))),
                "is_nan" => Some(("is_nan", IrType::Scalar(DType::Bool))),
                "is_inf" => Some(("is_inf", IrType::Scalar(DType::Bool))),

                // ── Phase 105: Functional list operations ──
                "list_map" => Some((
                    "list_map",
                    IrType::List(Box::new(IrType::Scalar(DType::I64))),
                )),
                "list_filter" => Some((
                    "list_filter",
                    IrType::List(Box::new(IrType::Scalar(DType::I64))),
                )),
                "list_reduce" => Some(("list_reduce", IrType::Scalar(DType::I64))),
                "list_any" => Some(("list_any", IrType::Scalar(DType::Bool))),
                "list_all" => Some(("list_all", IrType::Scalar(DType::Bool))),
                "list_zip" => Some((
                    "list_zip",
                    IrType::List(Box::new(IrType::Scalar(DType::I64))),
                )),
                "list_enumerate" => Some((
                    "list_enumerate",
                    IrType::List(Box::new(IrType::Scalar(DType::I64))),
                )),
                "list_flatten" => Some((
                    "list_flatten",
                    IrType::List(Box::new(IrType::Scalar(DType::I64))),
                )),
                "list_unique" => Some((
                    "list_unique",
                    IrType::List(Box::new(IrType::Scalar(DType::I64))),
                )),
                "list_reverse" => Some((
                    "list_reverse",
                    IrType::List(Box::new(IrType::Scalar(DType::I64))),
                )),
                "list_sorted" => Some((
                    "list_sorted",
                    IrType::List(Box::new(IrType::Scalar(DType::I64))),
                )),
                "list_sum" => Some(("list_sum", IrType::Scalar(DType::F64))),
                "list_min" => Some(("list_min", IrType::Scalar(DType::I64))),
                "list_max" => Some(("list_max", IrType::Scalar(DType::I64))),
                "list_index_of" => Some(("list_index_of", IrType::Scalar(DType::I64))),
                "list_count" => Some(("list_count", IrType::Scalar(DType::I64))),
                "list_take" => Some((
                    "list_take",
                    IrType::List(Box::new(IrType::Scalar(DType::I64))),
                )),
                "list_drop" => Some((
                    "list_drop",
                    IrType::List(Box::new(IrType::Scalar(DType::I64))),
                )),

                // ── Terminal / Interactive Input ──
                "read_key" => Some(("read_key", IrType::Scalar(DType::I64))),
                "read_password" => Some(("read_password", IrType::Str)),
                "term_clear" => Some(("term_clear", IrType::Scalar(DType::I64))),
                "term_cursor" => Some(("term_cursor", IrType::Scalar(DType::I64))),
                "term_show_cursor" => Some(("term_show_cursor", IrType::Scalar(DType::I64))),
                "term_set_color" => Some(("term_set_color", IrType::Scalar(DType::I64))),
                "term_reset" => Some(("term_reset", IrType::Scalar(DType::I64))),
                "term_rows" => Some(("term_rows", IrType::Scalar(DType::I64))),
                "term_cols" => Some(("term_cols", IrType::Scalar(DType::I64))),

                // ── UDP Networking ──
                "udp_open" => Some(("udp_open", IrType::Scalar(DType::I64))),
                "udp_send" => Some(("udp_send", IrType::Scalar(DType::I64))),
                "udp_recv" => Some(("udp_recv", IrType::Str)),
                "udp_close" => Some(("udp_close", IrType::Scalar(DType::I64))),

                // ── HTTP extended ──
                "http_request" => Some(("http_request", IrType::Str)),
                "http_post_json" => Some(("http_post_json", IrType::Str)),

                // ── Weak references ──
                "weak_ref" => Some(("weak_ref", IrType::WeakRef(Box::new(IrType::Infer)))),
                "weak_alive" => Some(("weak_alive", IrType::Scalar(DType::Bool))),

                // ── GC ──
                "gc_stats" => Some(("gc_stats_map", IrType::Map(Box::new(IrType::Str), Box::new(IrType::Scalar(DType::I64))))),
                "gc_collect" => Some(("gc_collect_call", IrType::Scalar(DType::Bool))),

                // ── Map extras ──
                "map_entries" => Some(("map_entries", IrType::List(Box::new(IrType::Str)))),

                _ => None,
            };

            // Special handling: chan_try_recv — use the channel's element type.
            if dispatch_name == "chan_try_recv" {
                if args.len() != 1 {
                    return Err(LowerError::Unsupported {
                        detail: "chan_try_recv() requires exactly 1 argument".into(),
                        span,
                    });
                }
                let (chan_val, chan_ty) = self.lower_expr(&args[0])?;
                let elem_ty = match &chan_ty {
                    IrType::Chan(elem) => (**elem).clone(),
                    other => return Err(LowerError::TypeMismatch {
                        expected: "channel".into(),
                        found: format!("{}", other),
                        span,
                    }),
                };
                let ret_ty = IrType::Option(Box::new(elem_ty));
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::BuiltinCall {
                        result,
                        name: "chan_try_recv".to_string(),
                        args: vec![chan_val],
                        result_ty: ret_ty.clone(),
                    },
                    Some(ret_ty.clone()),
                );
                return Ok((result, ret_ty));
            }

            // Special handling: recv_timeout(ch, ms) — returns option<elem_ty>
            if dispatch_name == "recv_timeout" {
                if args.len() != 2 {
                    return Err(LowerError::Unsupported {
                        detail: "recv_timeout() requires 2 arguments (channel, timeout_ms)".into(),
                        span,
                    });
                }
                let (chan_val, chan_ty) = self.lower_expr(&args[0])?;
                let (timeout_ms, _) = self.lower_expr(&args[1])?;
                let elem_ty = self.chan_elem_types.get(&chan_val).cloned().unwrap_or_else(|| {
                    match &chan_ty {
                        IrType::Chan(elem) => (**elem).clone(),
                        _ => IrType::Infer,
                    }
                });
                let ret_ty = IrType::Option(Box::new(elem_ty));
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::BuiltinCall {
                        result,
                        name: "recv_timeout".to_string(),
                        args: vec![chan_val, timeout_ms],
                        result_ty: ret_ty.clone(),
                    },
                    Some(ret_ty.clone()),
                );
                return Ok((result, ret_ty));
            }

            // Special handling: weak_upgrade(w) — returns option<Infer>
            if dispatch_name == "weak_upgrade" {
                if args.len() != 1 {
                    return Err(LowerError::Unsupported {
                        detail: "weak_upgrade() requires exactly 1 argument".into(),
                        span,
                    });
                }
                let (weak_val, _) = self.lower_expr(&args[0])?;
                let ret_ty = IrType::Option(Box::new(IrType::Infer));
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::BuiltinCall {
                        result,
                        name: "weak_upgrade".to_string(),
                        args: vec![weak_val],
                        result_ty: ret_ty.clone(),
                    },
                    Some(ret_ty.clone()),
                );
                return Ok((result, ret_ty));
            }

            if let Some((rt_name, ret_ty)) = builtin_info {
                let mut arg_vals = Vec::with_capacity(args.len());
                for arg in args {
                    let (v, _) = self.lower_expr(arg)?;
                    arg_vals.push(v);
                }
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::BuiltinCall {
                        result,
                        name: rt_name.to_string(),
                        args: arg_vals,
                        result_ty: ret_ty.clone(),
                    },
                    Some(ret_ty.clone()),
                );
                return Ok((result, ret_ty));
            }
        }

        // Const function call — evaluate at compile time if all args are literals.
        if let Some(const_fn) = self.generic_fns.get(&callee_name).cloned() {
            if const_fn.is_const {
                if let Some(result_val) = self.try_eval_const_fn(&const_fn, args) {
                    let ret_ty = self.resolve_ty(&const_fn.return_ty);
                    match &ret_ty {
                        IrType::Scalar(DType::I64) => {
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::ConstInt { result, value: result_val, ty: ret_ty.clone() },
                                Some(ret_ty.clone()),
                            );
                            return Ok((result, ret_ty));
                        }
                        IrType::Scalar(DType::F64) => {
                            let result = self.builder.fresh_value();
                            let bits = result_val as u64;
                            let float_val = f64::from_bits(bits);
                            self.builder.push_instr(
                                IrInstr::ConstFloat { result, value: float_val, ty: ret_ty.clone() },
                                Some(ret_ty.clone()),
                            );
                            return Ok((result, ret_ty));
                        }
                        IrType::Scalar(DType::Bool) => {
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::ConstBool { result, value: result_val != 0 },
                                Some(ret_ty.clone()),
                            );
                            return Ok((result, ret_ty));
                        }
                        _ => {} // Fall through to normal call for non-scalar return types
                    }
                }
            }
        }

        // Generic function call — monomorphize on demand.
        if let Some(generic_fn) = self.generic_fns.get(&callee_name).cloned() {
            // Lower each argument and collect concrete types.
            let mut arg_vals = Vec::with_capacity(args.len());
            let mut arg_tys = Vec::with_capacity(args.len());
            for arg in args {
                let (v, ty) = self.lower_expr(arg)?;
                arg_vals.push(v);
                arg_tys.push(ty);
            }

            // Build type substitution by matching each parameter's declared type
            // against the concrete argument type.  Only params whose type IS a
            // type parameter (e.g. `a: T`) contribute to the substitution;
            // params with concrete types (e.g. `cond: bool`) are skipped.
            // Two-pass: Named first (direct params), then Generic (nested type args).
            let mut subs: HashMap<String, IrType> = HashMap::new();
            // Pass 1: Direct Named type params (e.g. v: T).
            for (param, arg_ty) in generic_fn.params.iter().zip(arg_tys.iter()) {
                if let crate::parser::ast::AstType::Named(n, _) = &param.ty {
                    let is_generic_param = generic_fn
                        .type_params
                        .iter()
                        .any(|p| matches!(p, crate::parser::ast::AstGenericParam::Type(name, _, _) | crate::parser::ast::AstGenericParam::Hkt(name, _, _, _) if name == n));
                    if is_generic_param {
                        subs.entry(n.clone()).or_insert_with(|| arg_ty.clone());
                    }
                }
            }
            // Pass 2: Extract type params from Generic AST types (e.g. h: MinHeap<T>).
            // Only inserts if the type param wasn't already found in Pass 1.
            for (param, arg_ty) in generic_fn.params.iter().zip(arg_tys.iter()) {
                if let crate::parser::ast::AstType::Generic { name: gname, args: gargs, .. } = &param.ty {
                    // Try matching from monomorphized struct name: "MinHeap__i64" → ["i64"]
                    let concrete_name = match arg_ty {
                        IrType::Struct { name, .. } => Some(name.as_str()),
                        _ => None,
                    };
                    if let Some(s_name) = concrete_name {
                        let suffix = format!("{}__", gname);
                        if let Some(rest) = s_name.strip_prefix(&suffix) {
                            let concrete_parts: Vec<&str> = rest.split('_').collect();
                            for (ast_arg, concrete) in gargs.iter().zip(concrete_parts.iter()) {
                                if let crate::parser::ast::AstType::Named(n, _) = ast_arg {
                                    if subs.contains_key(n) { continue; }
                                    let is_generic_param = generic_fn
                                        .type_params
                                        .iter()
                                        .any(|p| matches!(p, crate::parser::ast::AstGenericParam::Type(name, _, _) | crate::parser::ast::AstGenericParam::Hkt(name, _, _, _) if name == n));
                                    if is_generic_param {
                                        let concrete_ir = lower_type_with_structs(
                                            &crate::parser::ast::AstType::Named(concrete.to_string(), crate::parser::lexer::Span::at(0)),
                                            self.module,
                                        );
                                        subs.insert(n.clone(), concrete_ir);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Fallback: walk AST param types recursively to extract type params.
            // This handles cases where the struct name doesn't have a mangled suffix
            // (e.g. arg is `Box { value: T_marker }` instead of `Box__i64`).
            if subs.values().count() < generic_fn.type_params.len() {
                // A type constructor is recorded as a nameless-field struct
                // marker, the same convention the lowerer already uses for
                // unsubstituted type parameters. `list`, `option` and `map` name
                // the builtin constructors; anything else names a user record.
                fn constructor_marker(name: &str) -> IrType {
                    IrType::Struct {
                        name: name.to_string(),
                        fields: Vec::new(),
                    }
                }

                // Decode one component of a mangled name such as the `i64` in
                // `Box__i64`. Anything unrecognised is a nominal type.
                fn ir_type_from_mangled_part(part: &str) -> IrType {
                    match part {
                        "i64" => IrType::Scalar(DType::I64),
                        "i32" => IrType::Scalar(DType::I32),
                        "f64" => IrType::Scalar(DType::F64),
                        "f32" => IrType::Scalar(DType::F32),
                        "bool" => IrType::Scalar(DType::Bool),
                        "str" => IrType::Str,
                        other => IrType::Struct {
                            name: other.to_string(),
                            fields: Vec::new(),
                        },
                    }
                }

                fn extract_from_ast_type(
                    ast_ty: &crate::parser::ast::AstType,
                    concrete_ty: &IrType,
                    type_params: &[String],
                    hkt_params: &[String],
                    subs: &mut HashMap<String, IrType>,
                ) {
                    // Higher-kinded: the AST says `F<T>` and `F` is a declared
                    // constructor parameter, so split the concrete type into
                    // its constructor and its element and bind both.
                    if let crate::parser::ast::AstType::Generic { name: gname, args: gargs, .. } = ast_ty {
                        if hkt_params.contains(gname) {
                            match concrete_ty {
                                IrType::List(inner) => {
                                    subs.entry(gname.clone())
                                        .or_insert_with(|| constructor_marker("list"));
                                    if let Some(a0) = gargs.first() {
                                        extract_from_ast_type(a0, inner, type_params, hkt_params, subs);
                                    }
                                    return;
                                }
                                IrType::Option(inner) => {
                                    subs.entry(gname.clone())
                                        .or_insert_with(|| constructor_marker("option"));
                                    if let Some(a0) = gargs.first() {
                                        extract_from_ast_type(a0, inner, type_params, hkt_params, subs);
                                    }
                                    return;
                                }
                                IrType::Map(k, v) => {
                                    subs.entry(gname.clone())
                                        .or_insert_with(|| constructor_marker("map"));
                                    if let Some(a0) = gargs.first() {
                                        extract_from_ast_type(a0, k, type_params, hkt_params, subs);
                                    }
                                    if let Some(a1) = gargs.get(1) {
                                        extract_from_ast_type(a1, v, type_params, hkt_params, subs);
                                    }
                                    return;
                                }
                                // A user record arrives already monomorphised as
                                // `Box__i64`; the constructor is the part before
                                // the separator and the element types follow it.
                                IrType::Struct { name: sname, fields } => {
                                    let (base, rest) = match sname.split_once("__") {
                                        Some((b, r)) => (b.to_string(), Some(r.to_string())),
                                        None => (sname.clone(), None),
                                    };
                                    subs.entry(gname.clone())
                                        .or_insert_with(|| constructor_marker(&base));
                                    if let Some(rest) = rest {
                                        for (garg, part) in gargs.iter().zip(rest.split('_')) {
                                            if let crate::parser::ast::AstType::Named(n, _) = garg {
                                                if type_params.contains(n) && !subs.contains_key(n) {
                                                    subs.insert(
                                                        n.clone(),
                                                        ir_type_from_mangled_part(part),
                                                    );
                                                }
                                            }
                                        }
                                    } else {
                                        // Not mangled: fall back to the field
                                        // layout, which is what the existing
                                        // non-HKT path does.
                                        for (garg, (_, fty)) in gargs.iter().zip(fields.iter()) {
                                            extract_from_ast_type(garg, fty, type_params, hkt_params, subs);
                                        }
                                    }
                                    return;
                                }
                                _ => {}
                            }
                        }
                    }
                    match ast_ty {
                        crate::parser::ast::AstType::Named(n, _) if type_params.contains(n) && !subs.contains_key(n) => {
                            subs.insert(n.clone(), concrete_ty.clone());
                        }
                        crate::parser::ast::AstType::Generic { args: gargs, .. } => {
                            if let IrType::Struct { fields, .. } = concrete_ty {
                                for (garg, (_, fty)) in gargs.iter().zip(fields.iter()) {
                                    extract_from_ast_type(garg, fty, type_params, hkt_params, subs);
                                }
                            }
                        }
                        crate::parser::ast::AstType::List(inner_ast, _) => {
                            if let IrType::List(inner_concrete) = concrete_ty {
                                extract_from_ast_type(inner_ast, inner_concrete, type_params, hkt_params, subs);
                            }
                        }
                        crate::parser::ast::AstType::Map(k_ast, v_ast, _) => {
                            if let IrType::Map(k_concrete, v_concrete) = concrete_ty {
                                extract_from_ast_type(k_ast, k_concrete, type_params, hkt_params, subs);
                                extract_from_ast_type(v_ast, v_concrete, type_params, hkt_params, subs);
                            }
                        }
                        crate::parser::ast::AstType::Option(inner_ast, _) => {
                            if let IrType::Option(inner_concrete) = concrete_ty {
                                extract_from_ast_type(inner_ast, inner_concrete, type_params, hkt_params, subs);
                            }
                        }
                        crate::parser::ast::AstType::Tuple(elems_ast, _) => {
                            if let IrType::Tuple(elems_concrete) = concrete_ty {
                                for (a, c) in elems_ast.iter().zip(elems_concrete.iter()) {
                                    extract_from_ast_type(a, c, type_params, hkt_params, subs);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                let type_param_names: Vec<String> = generic_fn.type_params.iter().filter_map(|p| {
                    if let crate::parser::ast::AstGenericParam::Type(n, _, _) = p { Some(n.clone()) } else { None }
                }).collect();
                // Constructor parameters are collected separately: they bind to a
                // constructor rather than to a type, so `extract_from_ast_type`
                // has to treat them differently. Leaving them out of both lists
                // is why `F` was never bound.
                let hkt_param_names: Vec<String> = generic_fn.type_params.iter().filter_map(|p| {
                    if let crate::parser::ast::AstGenericParam::Hkt(n, _, _, _) = p { Some(n.clone()) } else { None }
                }).collect();
                for (param2, arg_ty2) in generic_fn.params.iter().zip(arg_tys.iter()) {
                    extract_from_ast_type(&param2.ty, arg_ty2, &type_param_names, &hkt_param_names, &mut subs);
                }
            }

            // Pass 4 — expected-type propagation.
            //
            // The three passes above all read type parameters out of *argument*
            // types. A zero-argument constructor has none: `set_new[T]() -> Set<T>`
            // mentions `T` only in its return type, so `T` stayed unbound, the
            // signature resolved to `%set__Set__set__Set__T` — a struct defined
            // nowhere — and monomorphisation failed with
            //   type mismatch: %set__Set vs %set__Set__set__Set__T
            //
            // Unify the declared return type against the type expected at the
            // binding instead. `val s: Set<str> = set_new()` annotates as the
            // monomorphised struct `set__Set__str`, whose suffix carries the type
            // arguments — the same recovery Pass 2 performs for parameters like
            // `h: MinHeap<T>`, applied to the return position.
            //
            // This resolves the parameter from information the programmer actually
            // supplied. It does not invent one: a call with no argument and no
            // annotation still leaves `T` unbound, and is reported below.
            let all_type_params: Vec<String> = generic_fn
                .type_params
                .iter()
                .filter_map(|p| {
                    if let crate::parser::ast::AstGenericParam::Type(n, _, _) = p {
                        Some(n.clone())
                    } else {
                        None
                    }
                })
                .collect();
            if subs.len() < all_type_params.len() {
                if let Some(IrType::Struct { name: expected_name, .. }) = self.binding_ty.clone() {
                    if let crate::parser::ast::AstType::Generic {
                        name: gname,
                        args: gargs,
                        ..
                    } = &generic_fn.return_ty
                    {
                        // The return type of a brought generic carries the
                        // module-qualified name (`set__Set`), while the binding
                        // annotation resolves unqualified (`Set__str`). Try the
                        // qualified form, the written form, and the unqualified
                        // tail so either spelling matches.
                        let base = resolve_brought_name(gname, self.module);
                        let tail = gname.rsplit("__").next().unwrap_or(gname).to_string();
                        for prefix in [
                            format!("{}__", base),
                            format!("{}__", gname),
                            format!("{}__", tail),
                        ] {
                            if let Some(rest) = expected_name.strip_prefix(prefix.as_str()) {
                                for (ast_arg, concrete) in gargs.iter().zip(rest.split('_')) {
                                    if let crate::parser::ast::AstType::Named(n, _) = ast_arg {
                                        if all_type_params.contains(n) && !subs.contains_key(n) {
                                            let concrete_ir = lower_type_with_structs(
                                                &crate::parser::ast::AstType::Named(
                                                    concrete.to_string(),
                                                    crate::parser::lexer::Span::at(0),
                                                ),
                                                self.module,
                                            );
                                            subs.insert(n.clone(), concrete_ir);
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
            }

            // A type parameter that nothing constrains is an error, not a default.
            // Reporting it here replaces a bogus struct name that surfaced much
            // later as an opaque pass failure, and tells the caller what to do.
            if subs.len() < all_type_params.len() {
                let missing: Vec<&str> = all_type_params
                    .iter()
                    .filter(|n| !subs.contains_key(*n))
                    .map(|s| s.as_str())
                    .collect();
                return Err(LowerError::Unsupported {
                    detail: format!(
                        "cannot infer type parameter{} `{}` for `{}` — no argument mentions {}, \
                         so annotate the binding (e.g. `val x: {}<...> = {}(...)`)",
                        if missing.len() == 1 { "" } else { "s" },
                        missing.join(", "),
                        callee_name,
                        if missing.len() == 1 { "it" } else { "them" },
                        match &generic_fn.return_ty {
                            crate::parser::ast::AstType::Generic { name, .. } => name.clone(),
                            _ => "T".to_string(),
                        },
                        callee_name,
                    ),
                    span,
                });
            }

            // Resolve the concrete return type using resolve_ast_type_with_subs
            // which handles both Named("T") and Generic { name, args } with subs.
            let mut concrete_ret = resolve_ast_type_with_subs(&generic_fn.return_ty, &subs, self.module);
            if generic_fn.is_async {
                concrete_ret = IrType::Chan(Box::new(concrete_ret));
            }

            // Generate mangled name: e.g. `max_val__i64` for T=i64. Use a
            // sanitised form (no '%', '<', etc.) since the result is used
            // as an LLVM IR identifier.
            let mangle = subs
                .values()
                .map(mangle_ir_type)
                .collect::<Vec<_>>()
                .join("_");
            let mangled = format!("{}__{}", callee_name, mangle);

            // Register the return type for the mangled name.
            self.mono_sigs
                .borrow_mut()
                .insert(mangled.clone(), concrete_ret.clone());

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
                    &mono_fn,
                    self.module,
                    fn_sigs_ref,
                    &self.const_defs,
                    self.generic_fns.clone(),
                    self.mono_cache.clone(),
                    self.mono_sigs.clone(),
                    subs,
                    self.trait_dispatch.clone(),
                    self.fn_defaults.clone(),
                    self.fn_param_names.clone(),
                    self.fn_param_types.clone(),
                    self.lambda_counter.clone(),
                )?;

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
        if let Some(impls) = self.trait_dispatch.get(&callee_name).cloned() {
            if !args.is_empty() {
                let (first_val, first_ty) = self.lower_expr(&args[0])?;
                let type_key = ir_type_dispatch_name(&first_ty);
                if let Some((_, mangled)) = impls
                    .iter()
                    .find(|(dispatch_ty, _)| ir_type_dispatch_name(dispatch_ty) == type_key)
                {
                    let mangled = mangled.clone();
                    let ret_ty = self.fn_sigs.get(&mangled).cloned().unwrap_or(IrType::Infer);
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

        // ML/AI intrinsics (Phases 77–80)
        match dispatch_name {
            "zeros" => return self.lower_ml_zeros(args, span),
            "ones" => return self.lower_ml_ones(args, span),
            "fill" => return self.lower_ml_fill(args, span),
            "linspace" => return self.lower_ml_linspace(args, span),
            "arange" => return self.lower_ml_arange(args, span),
            "list_sum" => return self.lower_ml_list_sum(args, span),
            "list_mean" => return self.lower_ml_list_mean(args, span),
            "list_max_val" => return self.lower_ml_list_max_val(args, span),
            "list_min_val" => return self.lower_ml_list_min_val(args, span),
            "list_std" => return self.lower_ml_list_std(args, span),
            "list_norm" => return self.lower_ml_list_norm(args, span),
            "list_dot" => return self.lower_ml_list_dot(args, span),
            "list_add" => return self.lower_ml_list_binop(args, span, BinOp::Add),
            "list_sub" => return self.lower_ml_list_binop(args, span, BinOp::Sub),
            "list_mul_elem" => return self.lower_ml_list_binop(args, span, BinOp::Mul),
            "list_scale" => return self.lower_ml_list_scale(args, span),
            "list_relu" => return self.lower_ml_list_relu(args, span),
            "list_sigmoid" => return self.lower_ml_list_sigmoid(args, span),
            "list_softmax" => return self.lower_ml_list_softmax(args, span),
            "mse_loss" => return self.lower_ml_mse_loss(args, span),
            "cross_entropy" => return self.lower_ml_cross_entropy(args, span),
            "list_axpy" => return self.lower_ml_list_axpy(args, span),
            "sgd_step" => return self.lower_ml_sgd_step(args, span),
            // Phase 82: BLAS-named bindings
            "list_dot_blas" => return self.lower_ml_list_dot(args, span),
            "list_axpy_blas" => return self.lower_ml_list_axpy(args, span),
            "list_scale_blas" => return self.lower_ml_list_scale(args, span),
            "matmul" => return self.lower_ml_matmul(args, span),
            _ => {}
        }

        // General function call — look up the callee's return type from
        // pre-collected signatures so the result has a concrete type.
        let ret_ty = self
            .fn_sigs
            .get(&callee_name)
            .cloned()
            .or_else(|| self.mono_sigs.borrow().get(&callee_name).cloned())
            .unwrap_or(IrType::Infer);

        // If we still have Infer here, the callee is not defined anywhere.
        // Emit a compile-time error rather than silently producing bad IR.
        if ret_ty == IrType::Infer && !self.module.extern_fns.iter().any(|e| e.name == callee_name)
        {
            return Err(LowerError::UndefinedVariable {
                name: callee_name,
                span,
                suggestion: None,
            });
        }

        // Declared parameter types, so a concrete struct passed where a
        // `dyn Trait` is expected becomes a trait object here. Without this it
        // reached `DynCall` as a bare struct and failed at *runtime* with
        // "DynCall on non-trait-object" -- a mistake knowable at compile time,
        // deferred to execution. See known-issues #18.
        let callee_param_tys: Option<Vec<IrType>> =
            self.fn_param_types.get(&callee_name).cloned();

        // Build argument list, expanding splat args (`..expr`).
        let defaults = self.fn_defaults.get(&callee_name).cloned();
        let mut arg_vals = Vec::new();
        for arg in args {
            if let AstExpr::Splat { expr, .. } = arg {
                let (arr_val, arr_ty) = self.lower_expr(expr)?;
                let elem_ty;
                let arr_len;
                match &arr_ty {
                    IrType::Array { elem, len } => {
                        elem_ty = (**elem).clone();
                        arr_len = *len;
                    }
                    _ => {
                        return Err(LowerError::Unsupported {
                            detail: "splat (`..expr`) requires an array expression with known size".into(),
                            span: arg.span(),
                        });
                    }
                }
                for i in 0..arr_len {
                    let idx_val = self.builder.fresh_value();
                    let idx_ty = IrType::Scalar(DType::I64);
                    self.builder.push_instr(
                        IrInstr::ConstInt { result: idx_val, value: i as i64, ty: idx_ty.clone() },
                        Some(idx_ty.clone()),
                    );
                    let elem_val = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::ArrayLoad {
                            result: elem_val,
                            array: arr_val,
                            index: idx_val,
                            elem_ty: elem_ty.clone(),
                        },
                        Some(elem_ty.clone()),
                    );
                    arg_vals.push(elem_val);
                }
            } else {
                let (v, vty) = self.lower_expr(arg)?;
                let slot = arg_vals.len();
                let expected = callee_param_tys
                    .as_ref()
                    .and_then(|ts| ts.get(slot))
                    .filter(|t| matches!(t, IrType::TraitObject { .. }))
                    .cloned();
                let v = match expected {
                    Some(exp) => self.coerce_to_trait_object(v, vty, &exp, arg.span())?.0,
                    None => v,
                };
                arg_vals.push(v);
            }
        }
        if let Some(ref defs) = defaults {
            for default_expr in defs.iter().skip(arg_vals.len()).flatten() {
                let (v, _) = self.lower_expr(default_expr)?;
                arg_vals.push(v);
            }
        }

        // Check if this is an extern (C-linkage) function.
        let is_extern = self.module.extern_fns.iter().any(|e| e.name == callee_name);
        if is_extern {
            let result = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::CallExtern {
                    result: Some(result),
                    name: callee_name.clone(),
                    args: arg_vals,
                    ret_ty: ret_ty.clone(),
                },
                Some(ret_ty.clone()),
            );
            return Ok((result, ret_ty));
        }

        // A call carrying a taped argument is lowered inline, so the tape graph
        // continues through the callee's body instead of stopping at its
        // signature. Without this, `sq(x)` on a taped `x` produced an untaped
        // result and `backward` was rejected outright (#49).
        if arg_vals.iter().any(|v| self.tape_nodes.contains_key(v)) {
            if let Some(inlined) = self.try_lower_taped_call(&callee_name, &arg_vals)? {
                return Ok(inlined);
            }
        }

        let result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::Call {
                result: Some(result),
                callee: callee_name,
                args: arg_vals,
                result_ty: Some(ret_ty.clone()),
            },
            Some(ret_ty.clone()),
        );
        Ok((result, ret_ty))
    }

    /// Lowers `callee`'s body directly into the current function, with its
    /// parameters bound to `arg_vals`.
    ///
    /// Returns `Ok(None)` when the callee is not a candidate, in which case the
    /// caller emits an ordinary call and the handle is lost as before -- a
    /// rejected program rather than a wrong gradient.
    ///
    /// Declined deliberately when:
    /// * the body uses any form outside the inliner's whitelist -- notably
    ///   `return`, which would terminate the *caller*;
    /// * the function is already being inlined on this path -- a recursive taped
    ///   call would otherwise expand forever;
    /// * the arity does not match, so defaults and splats keep the normal path.
    fn try_lower_taped_call(
        &mut self,
        callee_name: &str,
        arg_vals: &[ValueId],
    ) -> Result<Option<(ValueId, IrType)>, LowerError> {
        if self.taped_inline_stack.iter().any(|n| n == callee_name) {
            return Ok(None);
        }
        let Some(func) = CURRENT_FN_ASTS.with(|m| m.borrow().get(callee_name).cloned()) else {
            return Ok(None);
        };
        if func.params.len() != arg_vals.len() || !taped_inline_ok_block(&func.body) {
            return Ok(None);
        }

        let saved_scope = self.scope.clone();
        self.taped_inline_stack.push(callee_name.to_owned());
        for (param, val) in func.params.iter().zip(arg_vals.iter()) {
            // Bind the parameter to the caller's *value*, not a copy. That is
            // what carries `tape_nodes` across: the body sees the very ValueId
            // whose tape node the caller recorded.
            let ty = self.resolve_ty(&param.ty);
            self.scope.insert(param.name.name.clone(), (*val, ty));
        }
        let lowered = self.lower_block(&func.body);
        self.taped_inline_stack.pop();
        self.scope = saved_scope;

        Ok(lowered?)
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
                inputs: input_vals.clone(),
                result_ty: result_ty.clone(),
            },
            Some(result_ty.clone()),
        );
        self.maybe_record_tape_result(result, &result_ty, "einsum", &input_vals);
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
        // Consume the expected expression type (set by `val` / `return` handlers)
        // so that if/else branches can coerce their results before the merge.
        let expected_ty = self.expected_expr_ty.take();
        let outer_scope = self.scope.clone();
        let mut rebound_names = find_rebound_vars(then_blk);
        if let Some(else_blk) = else_blk {
            for name in find_rebound_vars(else_blk) {
                if !rebound_names.contains(&name) {
                    rebound_names.push(name);
                }
            }
        }
        rebound_names.retain(|name| outer_scope.contains_key(name));

        if let Some(else_blk) = else_blk {
            // Full if/else: three-block CFG (then / else / merge).
            // If both branches are statement-only, the expression evaluates to
            // unit and both branches must still jump to the merge block.
            let unit_ty = IrType::Scalar(DType::I64);
            let unit_val = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: unit_val,
                    value: 0,
                    ty: unit_ty.clone(),
                },
                Some(unit_ty.clone()),
            );

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

            // Which block each arm's `Br` ended up in, so its arguments can be
            // extended once both arms are known (autodiff handles, #50).
            let mut then_br_block: Option<BlockId> = None;
            let mut else_br_block: Option<BlockId> = None;

            // Lower THEN branch.
            self.builder.set_current_block(then_bb);
            self.scope = outer_scope.clone();
            let mut then_result = self.lower_block(then_blk)?;
            // Coerce to expected dyn Trait if needed.
            if let (Some(ref expected), Some((tv, ref tt))) = (&expected_ty, &then_result) {
                let (cv, ct) = self.coerce_to_trait_object(*tv, tt.clone(), expected, span)?;
                then_result = Some((cv, ct));
            }
            let then_scope = self.scope.clone();
            if !self.builder.is_current_block_terminated() {
                let mut then_args = vec![then_result
                    .as_ref()
                    .map(|(then_val, _)| *then_val)
                    .unwrap_or(unit_val)];
                for name in &rebound_names {
                    let rebound_val = then_scope
                        .get(name)
                        .or_else(|| outer_scope.get(name))
                        .map(|(val, _)| *val)
                        .expect("rebound variable missing from then-scope");
                    then_args.push(rebound_val);
                }
                then_br_block = Some(self.builder.current_block());
                self.builder.push_instr(
                    IrInstr::Br {
                        target: merge_bb,
                        args: then_args,
                    },
                    None,
                );
            }
            self.scope = outer_scope.clone();

            // Lower ELSE branch.
            self.builder.set_current_block(else_bb);
            self.scope = outer_scope.clone();
            let mut else_result = self.lower_block(else_blk)?;
            // Coerce to expected dyn Trait if needed.
            if let (Some(ref expected), Some((ev, ref et))) = (&expected_ty, &else_result) {
                let (cv, ct) = self.coerce_to_trait_object(*ev, et.clone(), expected, span)?;
                else_result = Some((cv, ct));
            }
            let else_scope = self.scope.clone();
            if !self.builder.is_current_block_terminated() {
                let mut else_args = vec![else_result
                    .as_ref()
                    .map(|(else_val, _)| *else_val)
                    .unwrap_or(unit_val)];
                for name in &rebound_names {
                    let rebound_val = else_scope
                        .get(name)
                        .or_else(|| outer_scope.get(name))
                        .map(|(val, _)| *val)
                        .expect("rebound variable missing from else-scope");
                    else_args.push(rebound_val);
                }
                else_br_block = Some(self.builder.current_block());
                self.builder.push_instr(
                    IrInstr::Br {
                        target: merge_bb,
                        args: else_args,
                    },
                    None,
                );
            }
            self.scope = outer_scope.clone();

            // Merge block parameter type = type of whichever branch produced a value.
            let result_ty = match (&then_result, &else_result) {
                (Some((_, ty)), _) => ty.clone(),
                (_, Some((_, ty))) => ty.clone(),
                (None, None) => unit_ty,
            };

            let result =
                self.builder
                    .add_block_param(merge_bb, Some("if_result"), result_ty.clone());
            let mut rebound_params = Vec::new();
            for name in &rebound_names {
                let Some((_, ty)) = outer_scope.get(name) else {
                    continue;
                };
                let param = self
                    .builder
                    .add_block_param(merge_bb, Some(name), ty.clone());
                rebound_params.push((name.clone(), param, ty.clone()));
            }
            // If both arms produced a taped value, the merged result must carry a
            // handle too -- otherwise it arrives at `backward` as a bare f64 and
            // is rejected. Threaded as a trailing merge parameter, appended to
            // both branch arguments after the fact because the arms were emitted
            // before their sibling was known. Only when *both* arms are taped:
            // a one-sided graph would silently drop the other path's gradient.
            let then_handle = then_result
                .as_ref()
                .and_then(|(v, _)| self.tape_nodes.get(v).copied());
            let else_handle = else_result
                .as_ref()
                .and_then(|(v, _)| self.tape_nodes.get(v).copied());
            if let (Some(th), Some(eh), Some(tb), Some(eb)) =
                (then_handle, else_handle, then_br_block, else_br_block)
            {
                let handle_param =
                    self.builder
                        .add_block_param(merge_bb, Some("if_result$tape"), IrType::TapeRef);
                let patched_then = self.builder.append_br_arg(tb, merge_bb, th);
                let patched_else = self.builder.append_br_arg(eb, merge_bb, eh);
                if patched_then && patched_else {
                    self.taped_values.insert(result);
                    self.tape_nodes.insert(result, handle_param);
                }
            }

            self.builder.set_current_block(merge_bb);
            for (name, param, ty) in rebound_params {
                self.scope.insert(name, (param, ty));
            }
            Ok((result, result_ty))
        } else {
            // if-without-else: two-block CFG (then / merge).
            // The whole expression evaluates to unit (i64 0).
            let unit_ty = IrType::Scalar(DType::I64);
            let unit_val = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: unit_val,
                    value: 0,
                    ty: unit_ty.clone(),
                },
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
                    else_args: std::iter::once(unit_val)
                        .chain(
                            rebound_names
                                .iter()
                                .filter_map(|name| outer_scope.get(name).map(|(val, _)| *val)),
                        )
                        .collect(),
                },
                None,
            );

            // Lower THEN branch (side effects only; result is discarded).
            self.builder.set_current_block(then_bb);
            self.scope = outer_scope.clone();
            let _then_result = self.lower_block(then_blk)?;
            let then_scope = self.scope.clone();
            if !self.builder.is_current_block_terminated() {
                // Branch didn't return early: jump to merge with unit.
                let mut then_args = vec![unit_val];
                for name in &rebound_names {
                    let rebound_val = then_scope
                        .get(name)
                        .or_else(|| outer_scope.get(name))
                        .map(|(val, _)| *val)
                        .expect("rebound variable missing from then-scope");
                    then_args.push(rebound_val);
                }
                self.builder.push_instr(
                    IrInstr::Br {
                        target: merge_bb,
                        args: then_args,
                    },
                    None,
                );
            }
            self.scope = outer_scope.clone();

            let merge_param =
                self.builder
                    .add_block_param(merge_bb, Some("if_result"), unit_ty.clone());
            let mut rebound_params = Vec::new();
            for name in &rebound_names {
                let Some((_, ty)) = outer_scope.get(name) else {
                    continue;
                };
                let param = self
                    .builder
                    .add_block_param(merge_bb, Some(name), ty.clone());
                rebound_params.push((name.clone(), param, ty.clone()));
            }
            self.builder.set_current_block(merge_bb);
            for (name, param, ty) in rebound_params {
                self.scope.insert(name, (param, ty));
            }
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
        if !self.builder.is_current_block_terminated() {
            self.builder.push_instr(
                IrInstr::Br {
                    target: merge_bb,
                    args: vec![rhs_val],
                },
                None,
            );
        }

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
            matches!(
                a.pattern,
                AstWhenPattern::OptionSome { .. } | AstWhenPattern::OptionNone
            )
        });
        let is_result_when = arms.iter().any(|a| {
            matches!(
                a.pattern,
                AstWhenPattern::ResultOk { .. } | AstWhenPattern::ResultErr { .. }
            )
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
        // For enum matches with wildcards, use SwitchVariant with default_block.
        // Only use chain if there are guards or non-enum literal patterns.
        let is_enum_scrut = matches!(&scrut_ty, IrType::Enum { .. });
        let has_guard = arms.iter().any(|a| a.guard.is_some());

        // Use chain only if: has guard OR not enum type OR has literal patterns (non-wildcard)
        let needs_chain = has_guard || !is_enum_scrut;
        if needs_chain {
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
        let mut default_block_opt: Option<BlockId> = None;
        for (arm_idx, arm) in arms.iter().enumerate() {
            match &arm.pattern {
                AstWhenPattern::Wildcard => {
                    // Wildcard becomes the default block
                    let bb = self
                        .builder
                        .create_block(Some(&format!("when_{}_wildcard", enum_name)));
                    default_block_opt = Some(bb);
                    arm_blocks.push(bb);
                }
                AstWhenPattern::EnumVariant { .. } => {
                    // Regular variant arm
                    arm_blocks.push(
                        self.builder.create_block(Some(&format!(
                            "when_{}_{}",
                            enum_name, arm.variant_name
                        ))),
                    );
                }
                AstWhenPattern::Or(_) => {
                    // Or-pattern: allocate one shared body block
                    arm_blocks.push(
                        self.builder.create_block(Some(&format!(
                            "when_or_{}",
                            arm_idx
                        ))),
                    );
                }
                AstWhenPattern::Binding { .. } => {
                    // Binding pattern: delegate to chain lowering
                    return self.lower_when_as_chain(scrut_val, &scrut_ty, arms, span);
                }
                _ => {
                    return Err(LowerError::Unsupported {
                        detail: format!("unexpected pattern in enum match: {:?}", arm.pattern),
                        span: arm.span,
                    });
                }
            }
        }
        let merge_bb = self.builder.create_block(Some("when_merge"));

        // 4. Build the arms list for SwitchVariant (skip wildcard patterns).
        let mut switch_arms: Vec<(usize, BlockId)> = Vec::new();
        for (arm_idx, arm) in arms.iter().enumerate() {
            match &arm.pattern {
                AstWhenPattern::EnumVariant { variant_name, .. } => {
                    let variant_idx =
                        variants
                            .iter()
                            .position(|v| v == variant_name)
                            .ok_or_else(|| LowerError::Unsupported {
                                detail: format!(
                                    "no variant '{}' in enum '{}'",
                                    variant_name, enum_name
                                ),
                                span: arm.span,
                            })?;
                    switch_arms.push((variant_idx, arm_blocks[arm_idx]));
                }
                AstWhenPattern::Or(subs) => {
                    // Or-pattern: each sub-pattern maps to the shared body block
                    for sub in subs {
                        if let AstWhenPattern::EnumVariant { variant_name, .. } = sub {
                            let variant_idx =
                                variants
                                    .iter()
                                    .position(|v| v == variant_name)
                                    .ok_or_else(|| LowerError::Unsupported {
                                        detail: format!(
                                            "no variant '{}' in enum '{}'",
                                            variant_name, enum_name
                                        ),
                                        span: arm.span,
                                    })?;
                            switch_arms.push((variant_idx, arm_blocks[arm_idx]));
                        } else {
                            return Err(LowerError::Unsupported {
                                detail: format!("unsupported sub-pattern in or-pattern for enum match: {:?}", sub),
                                span: arm.span,
                            });
                        }
                    }
                }
                AstWhenPattern::Wildcard => {
                    // Skip - handled as default_block
                }
                _ => {
                    return Err(LowerError::Unsupported {
                        detail: format!("unexpected pattern in enum match: {:?}", arm.pattern),
                        span: arm.span,
                    });
                }
            }
        }

        // 5. Emit SwitchVariant terminator in the current block.
        self.builder.push_instr(
            IrInstr::SwitchVariant {
                scrutinee: scrut_val,
                arms: switch_arms,
                default_block: default_block_opt,
            },
            None,
        );

        // 6. Lower each arm body.
        // Get the variant field types for this enum so we can emit ExtractVariantField.
        let variant_field_types: Vec<Vec<IrType>> = self
            .module
            .enum_variant_fields(&enum_name)
            .cloned()
            .unwrap_or_default();

        let outer_scope = self.scope.clone();
        let mut result_ty: Option<IrType> = None;
        for (arm, &arm_bb) in arms.iter().zip(arm_blocks.iter()) {
            self.scope = outer_scope.clone();
            self.builder.set_current_block(arm_bb);

            // Emit ExtractVariantField instructions for pattern bindings.
            if let AstWhenPattern::EnumVariant {
                variant_name,
                bindings,
                ..
            } = &arm.pattern
            {
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
            if !self.builder.is_current_block_terminated() {
                self.builder.push_instr(
                    IrInstr::Br {
                        target: merge_bb,
                        args: vec![arm_val],
                    },
                    None,
                );
            }
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
        let some_arm = arms
            .iter()
            .find(|a| matches!(a.pattern, AstWhenPattern::OptionSome { .. }));
        let none_arm = arms
            .iter()
            .find(|a| matches!(a.pattern, AstWhenPattern::OptionNone))
            .or_else(|| arms.iter().find(|a| matches!(a.pattern, AstWhenPattern::Wildcard)));

        if some_arm.is_none() && none_arm.is_none() {
            return Err(LowerError::Unsupported {
                detail: "option when expression needs some/none arms".into(),
                span,
            });
        }

        // Emit IsSome test.
        let is_some_result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::IsSome {
                result: is_some_result,
                operand: scrut_val,
            },
            Some(IrType::Scalar(DType::Bool)),
        );

        let some_bb = self.builder.create_block(Some("option_some"));
        let none_bb = self.builder.create_block(Some("option_none"));
        let merge_bb = self.builder.create_block(Some("option_merge"));

        let outer_scope = self.scope.clone();
        let mut rebound_names = Vec::new();
        for arm in arms {
            collect_rebound_vars_in_expr(&arm.body, &mut rebound_names);
        }
        rebound_names.retain(|name| outer_scope.contains_key(name));

        let unit_ty = IrType::Scalar(DType::I64);

        // When only one arm is present the whole expression evaluates to unit (i64 0).
        // When both arms are present the result type comes from the arm bodies.
        let partial = some_arm.is_none() || none_arm.is_none();

        // Pre-compute a unit value in the current (pre-branch) block BEFORE the CondBr
        // terminates the block, so it's accessible from both successor arms.
        let unit_val = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: unit_val,
                value: 0,
                ty: unit_ty.clone(),
            },
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
            if let AstWhenPattern::OptionSome {
                binding: Some(ref bind_name),
            } = arm.pattern
            {
                // Unwrap the option to get the inner value.
                let unwrapped = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::OptionUnwrap {
                        result: unwrapped,
                        operand: scrut_val,
                        result_ty: inner_ty.clone(),
                    },
                    Some(inner_ty.clone()),
                );
                self.scope
                    .insert(bind_name.clone(), (unwrapped, inner_ty.clone()));
            }
            let (v, ty) = self.lower_expr(&arm.body)?;
            if partial {
                (unit_val, Some(unit_ty.clone()))
            } else {
                (v, Some(ty))
            }
        } else {
            (unit_val, Some(unit_ty.clone()))
        };
        if !self.builder.is_current_block_terminated() {
            let mut branch_args = vec![some_val];
            let some_scope = self.scope.clone();
            for name in &rebound_names {
                let val = some_scope.get(name).map(|(v, _)| *v).unwrap_or_else(|| {
                    outer_scope.get(name).map(|(v, _)| *v).expect("missing var in outer_scope")
                });
                branch_args.push(val);
            }
            self.builder.push_instr(
                IrInstr::Br {
                    target: merge_bb,
                    args: branch_args,
                },
                None,
            );
        }

        // None branch.
        self.builder.set_current_block(none_bb);
        self.scope = outer_scope.clone();
        let none_val = if let Some(arm) = none_arm {
            let (v, ty) = self.lower_expr(&arm.body)?;
            if result_ty.is_none() {
                result_ty = Some(ty.clone());
            }
            if partial {
                unit_val
            } else {
                v
            }
        } else {
            unit_val
        };
        if !self.builder.is_current_block_terminated() {
            let mut branch_args = vec![none_val];
            let none_scope = self.scope.clone();
            for name in &rebound_names {
                let val = none_scope.get(name).map(|(v, _)| *v).unwrap_or_else(|| {
                    outer_scope.get(name).map(|(v, _)| *v).expect("missing var in outer_scope")
                });
                branch_args.push(val);
            }
            self.builder.push_instr(
                IrInstr::Br {
                    target: merge_bb,
                    args: branch_args,
                },
                None,
            );
        }

        self.scope = outer_scope;
        let result_ty = result_ty.unwrap_or(unit_ty);
        let result =
            self.builder
                .add_block_param(merge_bb, Some("option_result"), result_ty.clone());
        let mut rebound_params = Vec::new();
        for name in &rebound_names {
            let Some((_, ty)) = self.scope.get(name) else {
                continue;
            };
            let param = self
                .builder
                .add_block_param(merge_bb, Some(name), ty.clone());
            rebound_params.push((name.clone(), param, ty.clone()));
        }
        self.builder.set_current_block(merge_bb);
        for (name, param, ty) in rebound_params {
            self.scope.insert(name, (param, ty));
        }
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
        let ok_arm = arms
            .iter()
            .find(|a| matches!(a.pattern, AstWhenPattern::ResultOk { .. }));
        let err_arm = arms
            .iter()
            .find(|a| matches!(a.pattern, AstWhenPattern::ResultErr { .. }))
            .or_else(|| arms.iter().find(|a| matches!(a.pattern, AstWhenPattern::Wildcard)));

        if ok_arm.is_none() && err_arm.is_none() {
            return Err(LowerError::Unsupported {
                detail: "result when expression needs ok/err arms".into(),
                span,
            });
        }

        // Emit IsOk test.
        let is_ok_result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::IsOk {
                result: is_ok_result,
                operand: scrut_val,
            },
            Some(IrType::Scalar(DType::Bool)),
        );

        let ok_bb = self.builder.create_block(Some("result_ok"));
        let err_bb = self.builder.create_block(Some("result_err"));
        let merge_bb = self.builder.create_block(Some("result_merge"));

        let outer_scope = self.scope.clone();
        let mut rebound_names = Vec::new();
        for arm in arms {
            collect_rebound_vars_in_expr(&arm.body, &mut rebound_names);
        }
        rebound_names.retain(|name| outer_scope.contains_key(name));

        let unit_ty = IrType::Scalar(DType::I64);

        // When only one arm is present the whole expression evaluates to unit (i64 0).
        let partial = ok_arm.is_none() || err_arm.is_none();

        // Pre-compute unit value BEFORE the CondBr terminates the block,
        // so it's accessible from both successor arms.
        let unit_val = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: unit_val,
                value: 0,
                ty: unit_ty.clone(),
            },
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
            if let AstWhenPattern::ResultOk {
                binding: Some(ref bind_name),
            } = arm.pattern
            {
                let unwrapped = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ResultUnwrap {
                        result: unwrapped,
                        operand: scrut_val,
                        result_ty: ok_inner_ty.clone(),
                    },
                    Some(ok_inner_ty.clone()),
                );
                self.scope
                    .insert(bind_name.clone(), (unwrapped, ok_inner_ty.clone()));
            }
            let (v, ty) = self.lower_expr(&arm.body)?;
            if partial {
                (unit_val, Some(unit_ty.clone()))
            } else {
                (v, Some(ty))
            }
        } else {
            (unit_val, Some(unit_ty.clone()))
        };
        if !self.builder.is_current_block_terminated() {
            let mut branch_args = vec![ok_val];
            let ok_scope = self.scope.clone();
            for name in &rebound_names {
                let val = ok_scope.get(name).map(|(v, _)| *v).unwrap_or_else(|| {
                    outer_scope.get(name).map(|(v, _)| *v).expect("missing var in outer_scope")
                });
                branch_args.push(val);
            }
            self.builder.push_instr(
                IrInstr::Br {
                    target: merge_bb,
                    args: branch_args,
                },
                None,
            );
        }

        // Err branch.
        self.builder.set_current_block(err_bb);
        self.scope = outer_scope.clone();
        let err_val = if let Some(arm) = err_arm {
            if let AstWhenPattern::ResultErr {
                binding: Some(ref bind_name),
            } = arm.pattern
            {
                let unwrapped = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ResultUnwrapErr {
                        result: unwrapped,
                        operand: scrut_val,
                        result_ty: err_inner_ty.clone(),
                    },
                    Some(err_inner_ty.clone()),
                );
                self.scope
                    .insert(bind_name.clone(), (unwrapped, err_inner_ty.clone()));
            }
            let (v, ty) = self.lower_expr(&arm.body)?;
            if result_ty.is_none() {
                result_ty = Some(ty.clone());
            }
            if partial {
                unit_val
            } else {
                v
            }
        } else {
            unit_val
        };
        if !self.builder.is_current_block_terminated() {
            let mut branch_args = vec![err_val];
            let err_scope = self.scope.clone();
            for name in &rebound_names {
                let val = err_scope.get(name).map(|(v, _)| *v).unwrap_or_else(|| {
                    outer_scope.get(name).map(|(v, _)| *v).expect("missing var in outer_scope")
                });
                branch_args.push(val);
            }
            self.builder.push_instr(
                IrInstr::Br {
                    target: merge_bb,
                    args: branch_args,
                },
                None,
            );
        }

        self.scope = outer_scope;
        let result_ty = result_ty.unwrap_or(unit_ty);
        let result =
            self.builder
                .add_block_param(merge_bb, Some("result_result"), result_ty.clone());
        let mut rebound_params = Vec::new();
        for name in &rebound_names {
            let Some((_, ty)) = self.scope.get(name) else {
                continue;
            };
            let param = self
                .builder
                .add_block_param(merge_bb, Some(name), ty.clone());
            rebound_params.push((name.clone(), param, ty.clone()));
        }
        self.builder.set_current_block(merge_bb);
        for (name, param, ty) in rebound_params {
            self.scope.insert(name, (param, ty));
        }
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
        let _no_match_scrut = self
            .builder
            .add_block_param(no_match_bb, None, scrut_ty.clone());

        let outer_scope = self.scope.clone();
        let mut rebound_names = Vec::new();
        for arm in arms {
            collect_rebound_vars_in_expr(&arm.body, &mut rebound_names);
        }
        rebound_names.retain(|name| outer_scope.contains_key(name));

        let mut result_ty: Option<IrType> = None;

        // Extract enum variant info if scrutinee is an enum.
        let (enum_name_opt, enum_variants_opt): (Option<String>, Option<Vec<String>>) =
            if let IrType::Enum { name, variants } = scrut_ty {
                (Some(name.clone()), Some(variants.clone()))
            } else {
                (None, None)
            };
        let enum_variant_field_types: Vec<Vec<IrType>> = if let Some(ref ename) = enum_name_opt {
            self.module
                .enum_variant_fields(ename)
                .cloned()
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // We chain arms: for each arm, emit:
        //   current_bb: cond = (pattern_matches && guard?)
        //               condBr cond -> arm_body_bb, next_check_bb
        //   arm_body_bb: bind vars, lower body, br merge_bb
        //   next_check_bb: (next iteration's current_bb)
        let mut current_check_bb = self.builder.current_block();
        // The initial scrutinee value (from entry block or expression result)
        let mut current_scrut_val = scrut_val;

        for (arm_idx, arm) in arms.iter().enumerate() {
            let is_last = arm_idx == arms.len() - 1;
            let has_guard_with_bindings = arm.guard.is_some() && pattern_has_bindings(&arm.pattern);

            // IMPORTANT: create bind_guard_bb BEFORE arm_body_bb so the validator's
            // linear block scan sees value definitions before their uses.
            let (bind_guard_bb_opt, guard_bb_scrut_opt) = if has_guard_with_bindings {
                let bb = self
                    .builder
                    .create_block(Some(&format!("when_guard_{}", arm_idx)));
                let scrut_param = self.builder.add_block_param(bb, None, scrut_ty.clone());
                (Some(bb), Some(scrut_param))
            } else {
                (None, None)
            };

            // Create the arm body block.
            let arm_body_bb = self
                .builder
                .create_block(Some(&format!("when_arm_{}", arm_idx)));
            let arm_bb_scrut = self
                .builder
                .add_block_param(arm_body_bb, None, scrut_ty.clone());

            // Create the next-check block (reuse no_match_bb for last arm).
            let (next_check_bb, next_bb_scrut_opt) = if is_last {
                (no_match_bb, None)
            } else {
                let bb = self
                    .builder
                    .create_block(Some(&format!("when_check_{}", arm_idx + 1)));
                let scrut_param = self.builder.add_block_param(bb, None, scrut_ty.clone());
                (bb, Some(scrut_param))
            };

            // Emit the pattern match condition into current_check_bb.
            self.builder.set_current_block(current_check_bb);
            self.scope = outer_scope.clone();

            // Compute the pattern condition (tag check only, no extraction).
            let pat_cond = self.emit_pattern_condition(
                current_scrut_val,
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
                let guard_bb_scrut = guard_bb_scrut_opt.unwrap_or(current_scrut_val);
                // In check_bb: branch on pattern condition.
                self.builder.push_instr(
                    IrInstr::CondBr {
                        cond: pat_cond,
                        then_block: bind_guard_bb,
                        then_args: vec![current_scrut_val],
                        else_block: next_check_bb,
                        else_args: vec![current_scrut_val],
                    },
                    None,
                );

                // In bind_guard_bb: emit bindings, evaluate guard.
                self.builder.set_current_block(bind_guard_bb);
                self.scope = outer_scope.clone();
                self.bind_pattern_vars(
                    guard_bb_scrut,
                    scrut_ty,
                    &arm.pattern,
                    &enum_variants_opt,
                    &enum_variant_field_types,
                )?;
                let guard_expr = arm.guard.as_ref().unwrap();
                let (guard_val, _) = self.lower_expr(guard_expr)?;
                self.builder.push_instr(
                    IrInstr::CondBr {
                        cond: guard_val,
                        then_block: arm_body_bb,
                        then_args: vec![guard_bb_scrut],
                        else_block: next_check_bb,
                        else_args: vec![current_scrut_val],
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
                        then_args: vec![current_scrut_val],
                        else_block: next_check_bb,
                        else_args: vec![current_scrut_val],
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
                    arm_bb_scrut,
                    scrut_ty,
                    &arm.pattern,
                    &enum_variants_opt,
                    &enum_variant_field_types,
                )?;
            }

            let (arm_val, arm_ty) = self.lower_expr(&arm.body)?;
            if result_ty.is_none() {
                result_ty = Some(arm_ty);
            }
            if !self.builder.is_current_block_terminated() {
                let mut branch_args = vec![arm_val];
                let arm_scope = self.scope.clone();
                for name in &rebound_names {
                    let val = arm_scope.get(name).map(|(v, _)| *v).unwrap_or_else(|| {
                        outer_scope.get(name).map(|(v, _)| *v).expect("missing var in outer_scope")
                    });
                    branch_args.push(val);
                }
                self.builder.push_instr(
                    IrInstr::Br {
                        target: merge_bb,
                        args: branch_args,
                    },
                    None,
                );
            }

            // Update current scrutinee value for next iteration (from next_check_bb's block parameter)
            current_scrut_val = next_bb_scrut_opt.unwrap_or(scrut_val);
            current_check_bb = next_check_bb;
        }

        // Emit the no-match block (runtime panic).
        self.builder.set_current_block(no_match_bb);
        self.scope = outer_scope.clone();
        let panic_msg = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstStr {
                result: panic_msg,
                value: "when: no pattern matched".to_string(),
            },
            Some(IrType::Str),
        );
        self.builder
            .push_instr(IrInstr::Panic { msg: panic_msg, span_byte: None }, None);
        // Panic is now a terminator; we do not need a dummy branch to merge_bb.

        self.scope = outer_scope;
        let result_ty = result_ty.unwrap_or(IrType::Scalar(DType::I64));
        let result = self
            .builder
            .add_block_param(merge_bb, Some("when_result"), result_ty.clone());
        let mut rebound_params = Vec::new();
        for name in &rebound_names {
            let Some((_, ty)) = self.scope.get(name) else {
                continue;
            };
            let param = self
                .builder
                .add_block_param(merge_bb, Some(name), ty.clone());
            rebound_params.push((name.clone(), param, ty.clone()));
        }
        self.builder.set_current_block(merge_bb);
        for (name, param, ty) in rebound_params {
            self.scope.insert(name, (param, ty));
        }
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
            AstWhenPattern::Binding { pattern: inner, .. } => {
                self.emit_pattern_condition(scrut_val, scrut_ty, inner, enum_name_opt, enum_variants_opt, span)
            }
            AstWhenPattern::Struct { struct_name, fields, .. } => {
                // Struct pattern: look up the struct def, extract field indices, check sub-patterns.
                let bool_ty = IrType::Scalar(DType::Bool);
                let def = self.module.struct_def(struct_name).ok_or_else(|| LowerError::Unsupported {
                    detail: format!("unknown struct '{}' in pattern", struct_name),
                    span,
                })?;
                let field_map: HashMap<String, (usize, IrType)> = def.iter().enumerate()
                    .map(|(i, (n, t))| (n.clone(), (i, t.clone()))).collect();
                let mut all_ok = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstBool { result: all_ok, value: true },
                    Some(bool_ty.clone()),
                );
                for (field_name, sub_pat) in fields {
                    let (field_idx, field_ty) = field_map.get(field_name).ok_or_else(|| LowerError::Unsupported {
                        detail: format!("no field '{}' in struct '{}'", field_name, struct_name),
                        span,
                    })?;
                    let field_val = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::GetField { result: field_val, base: scrut_val, field_index: *field_idx, result_ty: field_ty.clone() },
                        Some(field_ty.clone()),
                    );
                    let sub_cond = match sub_pat {
                        AstWhenPattern::EnumVariant { enum_name, .. } if enum_name.is_empty() => {
                            let t = self.builder.fresh_value();
                            self.builder.push_instr(IrInstr::ConstBool { result: t, value: true }, Some(bool_ty.clone()));
                            t
                        }
                        AstWhenPattern::Wildcard => {
                            let t = self.builder.fresh_value();
                            self.builder.push_instr(IrInstr::ConstBool { result: t, value: true }, Some(bool_ty.clone()));
                            t
                        }
                        other => self.emit_pattern_condition(field_val, field_ty, other, &None, &None, span)?,
                    };
                    let new_ok = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::BinOp { result: new_ok, op: BinOp::BitAnd, lhs: all_ok, rhs: sub_cond, ty: bool_ty.clone() },
                        Some(bool_ty.clone()),
                    );
                    all_ok = new_ok;
                }
                Ok(all_ok)
            }
            AstWhenPattern::Wildcard => {
                // Always matches: emit `true`.
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstBool {
                        result,
                        value: true,
                    },
                    Some(IrType::Scalar(DType::Bool)),
                );
                Ok(result)
            }
            AstWhenPattern::IntLit(n) => {
                // scrutinee == n
                let lit_val = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstInt {
                        result: lit_val,
                        value: *n,
                        ty: scrut_ty.clone(),
                    },
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
            AstWhenPattern::FloatLit(f) => {
                // scrutinee == f (float comparison)
                let f64_ty = IrType::Scalar(DType::F64);
                let lit_val = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstFloat {
                        result: lit_val,
                        value: *f,
                        ty: f64_ty.clone(),
                    },
                    Some(f64_ty.clone()),
                );
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::BinOp {
                        result,
                        op: BinOp::CmpEq,
                        lhs: scrut_val,
                        rhs: lit_val,
                        ty: f64_ty.clone(),
                    },
                    Some(IrType::Scalar(DType::Bool)),
                );
                Ok(result)
            }
            AstWhenPattern::BoolLit(b) => {
                // scrutinee == b
                let lit_val = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstBool {
                        result: lit_val,
                        value: *b,
                    },
                    Some(IrType::Scalar(DType::Bool)),
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
            AstWhenPattern::StringLit(s) => {
                // StrEq(scrutinee, s)
                let str_val = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstStr {
                        result: str_val,
                        value: s.clone(),
                    },
                    Some(IrType::Str),
                );
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::StrEq {
                        result,
                        lhs: scrut_val,
                        rhs: str_val,
                    },
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
                    _ => {
                        return Err(LowerError::Unsupported {
                            detail: format!("tuple pattern on non-tuple type {}", scrut_ty),
                            span,
                        })
                    }
                };
                let mut all_ok = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstBool {
                        result: all_ok,
                        value: true,
                    },
                    Some(bool_ty.clone()),
                );
                for (i, sub) in subs.iter().enumerate() {
                    let elem_ty = tuple_elems
                        .get(i)
                        .cloned()
                        .unwrap_or(IrType::Scalar(DType::I64));
                    let elem_val = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::GetElement {
                            result: elem_val,
                            base: scrut_val,
                            index: i,
                            result_ty: elem_ty.clone(),
                        },
                        Some(elem_ty.clone()),
                    );
                    // Check sub-pattern
                    let sub_ok = match sub {
                        // Binding or wildcard: always true
                        AstWhenPattern::EnumVariant {
                            enum_name,
                            variant_name,
                            ..
                        } if enum_name.is_empty() => {
                            // Bind this element under variant_name (handled separately in bindings)
                            let _ = variant_name;
                            let t = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::ConstBool {
                                    result: t,
                                    value: true,
                                },
                                Some(bool_ty.clone()),
                            );
                            t
                        }
                        AstWhenPattern::Wildcard => {
                            let t = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::ConstBool {
                                    result: t,
                                    value: true,
                                },
                                Some(bool_ty.clone()),
                            );
                            t
                        }
                        other => self.emit_pattern_condition(
                            elem_val, &elem_ty, other, &None, &None, span,
                        )?,
                    };
                    let new_all = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::BinOp {
                            result: new_all,
                            op: BinOp::BitAnd,
                            lhs: all_ok,
                            rhs: sub_ok,
                            ty: bool_ty.clone(),
                        },
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
                    IrInstr::IsSome {
                        result: is_some_result,
                        operand: scrut_val,
                    },
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
                    IrInstr::IsSome {
                        result,
                        operand: scrut_val,
                    },
                    Some(IrType::Scalar(DType::Bool)),
                );
                Ok(result)
            }
            AstWhenPattern::ResultOk { .. } => {
                // IsOk(scrutinee)
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::IsOk {
                        result,
                        operand: scrut_val,
                    },
                    Some(IrType::Scalar(DType::Bool)),
                );
                Ok(result)
            }
            AstWhenPattern::ResultErr { .. } => {
                // !IsOk(scrutinee)
                let is_ok_result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::IsOk {
                        result: is_ok_result,
                        operand: scrut_val,
                    },
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
            AstWhenPattern::EnumVariant { variant_name, enum_name, .. } => {
                // If enum_name is empty, this is a plain identifier binding (always matches).
                if enum_name.is_empty() {
                    let result = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::ConstBool { result, value: true },
                        Some(IrType::Scalar(DType::Bool)),
                    );
                    return Ok(result);
                }
                // GetVariantTag(scrutinee) == variant_idx_const
                let variants =
                    enum_variants_opt
                        .as_ref()
                        .ok_or_else(|| LowerError::Unsupported {
                            detail: "EnumVariant pattern used with non-enum scrutinee".into(),
                            span,
                        })?;
                let variant_idx =
                    variants
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
                    IrInstr::GetVariantTag {
                        result: tag_val,
                        operand: scrut_val,
                    },
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
                    IrInstr::ConstInt {
                        result: lo_val,
                        value: *lo,
                        ty: scrut_ty.clone(),
                    },
                    Some(scrut_ty.clone()),
                );
                let hi_val = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstInt {
                        result: hi_val,
                        value: *hi,
                        ty: scrut_ty.clone(),
                    },
                    Some(scrut_ty.clone()),
                );
                let lo_ok = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::BinOp {
                        result: lo_ok,
                        op: BinOp::CmpLe,
                        lhs: lo_val,
                        rhs: scrut_val,
                        ty: scrut_ty.clone(),
                    },
                    Some(IrType::Scalar(DType::Bool)),
                );
                let hi_ok = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::BinOp {
                        result: hi_ok,
                        op: BinOp::CmpLe,
                        lhs: scrut_val,
                        rhs: hi_val,
                        ty: scrut_ty.clone(),
                    },
                    Some(IrType::Scalar(DType::Bool)),
                );
                let result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::BinOp {
                        result,
                        op: BinOp::BitAnd,
                        lhs: lo_ok,
                        rhs: hi_ok,
                        ty: IrType::Scalar(DType::Bool),
                    },
                    Some(IrType::Scalar(DType::Bool)),
                );
                Ok(result)
            }
            AstWhenPattern::Or(sub_pats) => {
                let mut cond_val = None;
                for p in sub_pats {
                    let sub_cond = self.emit_pattern_condition(
                        scrut_val,
                        scrut_ty,
                        p,
                        enum_name_opt,
                        enum_variants_opt,
                        span,
                    )?;
                    if let Some(prev) = cond_val {
                        let combined = self.builder.fresh_value();
                        self.builder.push_instr(
                            IrInstr::BinOp {
                                result: combined,
                                op: BinOp::BitOr,
                                lhs: prev,
                                rhs: sub_cond,
                                ty: IrType::Scalar(DType::Bool),
                            },
                            Some(IrType::Scalar(DType::Bool)),
                        );
                        cond_val = Some(combined);
                    } else {
                        cond_val = Some(sub_cond);
                    }
                }
                Ok(cond_val.unwrap_or_else(|| {
                    let res = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::ConstBool {
                            result: res,
                            value: true,
                        },
                        Some(IrType::Scalar(DType::Bool)),
                    );
                    res
                }))
            }
            AstWhenPattern::Slice { prefix, rest } => {
                let bool_ty = IrType::Scalar(DType::Bool);
                let i64_ty = IrType::Scalar(DType::I64);
                let prefix_len = prefix.len() as i64;

                // Get list length
                let len_val = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ListLen { result: len_val, list: scrut_val },
                    Some(i64_ty.clone()),
                );

                // Check length: if no rest, len == prefix.len(); if rest, len >= prefix.len()
                let prefix_len_val = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstInt {
                        result: prefix_len_val,
                        value: prefix_len,
                        ty: i64_ty.clone(),
                    },
                    Some(i64_ty.clone()),
                );
                let len_check = if rest.is_some() {
                    let cmp = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::BinOp {
                            result: cmp,
                            op: BinOp::CmpGe,
                            lhs: len_val,
                            rhs: prefix_len_val,
                            ty: i64_ty.clone(),
                        },
                        Some(bool_ty.clone()),
                    );
                    cmp
                } else {
                    let cmp = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::BinOp {
                            result: cmp,
                            op: BinOp::CmpEq,
                            lhs: len_val,
                            rhs: prefix_len_val,
                            ty: i64_ty.clone(),
                        },
                        Some(bool_ty.clone()),
                    );
                    cmp
                };

                // If no prefix elements, length check is the only condition
                if prefix.is_empty() {
                    return Ok(len_check);
                }

                // Restructure with CondBr: check length first, only then check elements.
                // This avoids ListGet out-of-bounds panics when length is insufficient.
                let check_elems_bb = self.builder.create_block(Some("check_elems"));
                let no_match_bb = self.builder.create_block(Some("no_match"));
                let merge_bb = self.builder.create_block(Some("merge"));

                self.builder.push_instr(
                    IrInstr::CondBr {
                        cond: len_check,
                        then_block: check_elems_bb,
                        then_args: vec![],
                        else_block: no_match_bb,
                        else_args: vec![],
                    },
                    None,
                );

                // no_match: length was insufficient — false → merge
                self.builder.set_current_block(no_match_bb);
                let false_val = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstBool { result: false_val, value: false },
                    Some(bool_ty.clone()),
                );
                self.builder.push_instr(
                    IrInstr::Br { target: merge_bb, args: vec![false_val] },
                    None,
                );

                // check_elems: length is sufficient — check each prefix element via ListGet
                self.builder.set_current_block(check_elems_bb);
                let elem_ty = match scrut_ty {
                    IrType::List(inner) => (**inner).clone(),
                    _ => IrType::Infer,
                };

                let mut all_ok = len_check;
                for (i, sub) in prefix.iter().enumerate() {
                    let idx_val = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::ConstInt {
                            result: idx_val,
                            value: i as i64,
                            ty: i64_ty.clone(),
                        },
                        Some(i64_ty.clone()),
                    );
                    let elem_val = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::ListGet {
                            result: elem_val,
                            list: scrut_val,
                            index: idx_val,
                            elem_ty: elem_ty.clone(),
                        },
                        Some(elem_ty.clone()),
                    );
                    let sub_ok = self.emit_pattern_condition(
                        elem_val, &elem_ty, sub, &None, &None, span,
                    )?;
                    let new_all = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::BinOp {
                            result: new_all,
                            op: BinOp::BitAnd,
                            lhs: all_ok,
                            rhs: sub_ok,
                            ty: bool_ty.clone(),
                        },
                        Some(bool_ty.clone()),
                    );
                    all_ok = new_all;
                }
                self.builder.push_instr(
                    IrInstr::Br { target: merge_bb, args: vec![all_ok] },
                    None,
                );

                // merge: result = phi from no_match(false) or check_elems(all_ok)
                let result = self.builder.add_block_param(merge_bb, Some("slice_match"), bool_ty.clone());
                self.builder.set_current_block(merge_bb);
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
            AstWhenPattern::Binding { name, pattern: inner, .. } => {
                self.scope.insert(name.clone(), (scrut_val, scrut_ty.clone()));
                self.bind_pattern_vars(scrut_val, scrut_ty, inner, enum_variants_opt, enum_variant_field_types)?;
            }
            AstWhenPattern::OptionSome {
                binding: Some(bind_name),
            } => {
                let inner_ty = if let IrType::Option(inner) = scrut_ty {
                    (**inner).clone()
                } else {
                    IrType::Infer
                };
                let unwrapped = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::OptionUnwrap {
                        result: unwrapped,
                        operand: scrut_val,
                        result_ty: inner_ty.clone(),
                    },
                    Some(inner_ty.clone()),
                );
                self.scope.insert(bind_name.clone(), (unwrapped, inner_ty));
            }
            AstWhenPattern::ResultOk {
                binding: Some(bind_name),
            } => {
                let ok_ty = if let IrType::ResultType(ok, _) = scrut_ty {
                    (**ok).clone()
                } else {
                    IrType::Infer
                };
                let unwrapped = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ResultUnwrap {
                        result: unwrapped,
                        operand: scrut_val,
                        result_ty: ok_ty.clone(),
                    },
                    Some(ok_ty.clone()),
                );
                self.scope.insert(bind_name.clone(), (unwrapped, ok_ty));
            }
            AstWhenPattern::ResultErr {
                binding: Some(bind_name),
            } => {
                let err_ty = if let IrType::ResultType(_, err) = scrut_ty {
                    (**err).clone()
                } else {
                    IrType::Infer
                };
                let unwrapped = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ResultUnwrapErr {
                        result: unwrapped,
                        operand: scrut_val,
                        result_ty: err_ty.clone(),
                    },
                    Some(err_ty.clone()),
                );
                self.scope.insert(bind_name.clone(), (unwrapped, err_ty));
            }
            AstWhenPattern::EnumVariant {
                variant_name,
                bindings,
                ..
            } => {
                if !bindings.is_empty() {
                    if let Some(variants) = enum_variants_opt {
                        let vidx = variants.iter().position(|v| v == variant_name);
                        let empty_fields: Vec<IrType> = Vec::new();
                        let field_types: &Vec<IrType> = vidx
                            .and_then(|i| enum_variant_field_types.get(i))
                            .unwrap_or(&empty_fields);
                        let variant_idx = vidx.unwrap_or(0);
                        for (field_idx, binding_name) in bindings.iter().enumerate() {
                            let field_ty =
                                field_types.get(field_idx).cloned().unwrap_or(IrType::Infer);
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
            AstWhenPattern::Struct { struct_name, fields, .. } => {
                let def = self.module.struct_def(struct_name).cloned().unwrap_or_default();
                let field_map: HashMap<String, (usize, IrType)> = def.into_iter().enumerate()
                    .map(|(i, (n, t))| (n, (i, t))).collect();
                for (field_name, sub_pat) in fields {
                    if let Some((field_idx, field_ty)) = field_map.get(field_name) {
                        let field_val = self.builder.fresh_value();
                        self.builder.push_instr(
                            IrInstr::GetField { result: field_val, base: scrut_val, field_index: *field_idx, result_ty: field_ty.clone() },
                            Some(field_ty.clone()),
                        );
                        match sub_pat {
                            AstWhenPattern::EnumVariant { enum_name, variant_name, .. } if enum_name.is_empty() => {
                                self.scope.insert(variant_name.clone(), (field_val, field_ty.clone()));
                            }
                            AstWhenPattern::Binding { name, .. } => {
                                self.scope.insert(name.clone(), (field_val, field_ty.clone()));
                            }
                            _ => {
                                self.bind_pattern_vars(field_val, field_ty, sub_pat, enum_variants_opt, enum_variant_field_types)?;
                            }
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
                    if let AstWhenPattern::EnumVariant {
                        enum_name,
                        variant_name,
                        ..
                    } = sub
                    {
                        if enum_name.is_empty() {
                            // This is an ident binding
                            let elem_ty = tuple_elems
                                .get(i)
                                .cloned()
                                .unwrap_or(IrType::Scalar(DType::I64));
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::GetElement {
                                    result,
                                    base: scrut_val,
                                    index: i,
                                    result_ty: elem_ty.clone(),
                                },
                                Some(elem_ty.clone()),
                            );
                            self.scope.insert(variant_name.clone(), (result, elem_ty));
                        }
                    }
                }
            }
            // Slice pattern: bind prefix elements and optional rest sub-list.
            AstWhenPattern::Slice { prefix, rest } => {
                let i64_ty = IrType::Scalar(DType::I64);
                let elem_ty = match scrut_ty {
                    IrType::List(inner) => (**inner).clone(),
                    _ => IrType::Infer,
                };
                // Bind each prefix element
                for (i, sub) in prefix.iter().enumerate() {
                    if let AstWhenPattern::EnumVariant {
                        enum_name,
                        variant_name,
                        ..
                    } = sub
                    {
                        if enum_name.is_empty() {
                            let idx_val = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::ConstInt {
                                    result: idx_val,
                                    value: i as i64,
                                    ty: i64_ty.clone(),
                                },
                                Some(i64_ty.clone()),
                            );
                            let elem_val = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::ListGet {
                                    result: elem_val,
                                    list: scrut_val,
                                    index: idx_val,
                                    elem_ty: elem_ty.clone(),
                                },
                                Some(elem_ty.clone()),
                            );
                            self.scope.insert(variant_name.clone(), (elem_val, elem_ty.clone()));
                        }
                    }
                }
                // Bind rest variable as sub-list via ListSlice(prefix.len, len)
                if let Some(rest_name) = rest {
                    let prefix_len_val = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::ConstInt {
                            result: prefix_len_val,
                            value: prefix.len() as i64,
                            ty: i64_ty.clone(),
                        },
                        Some(i64_ty.clone()),
                    );
                    let len_val = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::ListLen { result: len_val, list: scrut_val },
                        Some(i64_ty.clone()),
                    );
                    let rest_val = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::ListSlice {
                            result: rest_val,
                            list: scrut_val,
                            start: prefix_len_val,
                            end: len_val,
                        },
                        Some(scrut_ty.clone()),
                    );
                    self.scope.insert(rest_name.clone(), (rest_val, scrut_ty.clone()));
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
        label: &Option<String>,
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

        let mut initial_vals: Vec<ValueId> = loop_vars.iter().map(|(_, v, _)| *v).collect();

        // A loop variable holding a taped value carries *two* things across the
        // back-edge: the primal and the handle to its tape node. Only the primal
        // used to be threaded, so on the second iteration the handle was gone and
        // `backward` was rejected -- a loop-accumulated loss could not be written
        // at all (#49). The handles ride along as extra block parameters,
        // appended after the primals so the existing indices are undisturbed.
        let taped_loop_vars: Vec<(usize, ValueId)> = loop_vars
            .iter()
            .enumerate()
            .filter_map(|(i, (_, v, _))| self.tape_nodes.get(v).map(|h| (i, *h)))
            .collect();
        for (_, handle) in &taped_loop_vars {
            initial_vals.push(*handle);
        }

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
        let header_handle_params: Vec<ValueId> = taped_loop_vars
            .iter()
            .map(|(i, _)| {
                let name = format!("{}$tape", loop_vars[*i].0);
                self.builder
                    .add_block_param(header_bb, Some(&name), IrType::TapeRef)
            })
            .collect();

        // Add block params to merge (receive exit values from header's else path).
        let mut merge_params: Vec<ValueId> = Vec::new();
        for (name, _, ty) in &loop_vars {
            let p = self
                .builder
                .add_block_param(merge_bb, Some(name), ty.clone());
            merge_params.push(p);
        }
        let merge_handle_params: Vec<ValueId> = taped_loop_vars
            .iter()
            .map(|(i, _)| {
                let name = format!("{}$tape", loop_vars[*i].0);
                self.builder
                    .add_block_param(merge_bb, Some(&name), IrType::TapeRef)
            })
            .collect();

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
        // Re-point the tape mapping at this iteration's parameters, so a taped
        // value read inside the body resolves to the handle that came round the
        // back-edge rather than to the one from before the loop.
        for ((i, _), &hp) in taped_loop_vars.iter().zip(header_handle_params.iter()) {
            self.taped_values.insert(header_params[*i]);
            self.tape_nodes.insert(header_params[*i], hp);
        }

        let (cond_val, _) = self.lower_expr(cond)?;

        // Emit CondBr: true → body (no args), false → merge (current header params).
        self.builder.push_instr(
            IrInstr::CondBr {
                cond: cond_val,
                then_block: body_bb,
                then_args: vec![],
                else_block: merge_bb,
                else_args: header_params
                    .iter()
                    .copied()
                    .chain(header_handle_params.iter().copied())
                    .collect(),
            },
            None,
        );

        // Lower body block.
        self.builder.set_current_block(body_bb);
        let loop_var_names: Vec<String> = loop_vars.iter().map(|(n, _, _)| n.clone()).collect();
        self.loop_stack
            .push((header_bb, merge_bb, header_bb, loop_var_names.clone(), label.clone(), false));
        let _ = self.lower_block(body)?;
        self.loop_stack.pop();

        // Emit back-edge Br if the body wasn't terminated by break/continue.
        if !self.builder.is_current_block_terminated() {
            let mut updated_vals: Vec<ValueId> = loop_vars
                .iter()
                .map(|(name, original_val, _)| {
                    self.scope
                        .get(name)
                        .map(|(v, _)| *v)
                        .unwrap_or(*original_val)
                })
                .collect();
            // Whatever the body last assigned to a taped loop variable has its
            // own tape node; that handle is what the next iteration must see.
            for (i, incoming) in &taped_loop_vars {
                let current = updated_vals[*i];
                let handle = self.tape_nodes.get(&current).copied().unwrap_or(*incoming);
                updated_vals.push(handle);
            }
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
        // The whole point: after the loop, the accumulated value still has a
        // handle, so `backward(acc)` sees a tape node rather than a bare f64.
        for ((i, _), &hp) in taped_loop_vars.iter().zip(merge_handle_params.iter()) {
            self.taped_values.insert(merge_params[*i]);
            self.tape_nodes.insert(merge_params[*i], hp);
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
        inclusive: bool,
        step: Option<&AstExpr>,
        label: &Option<String>,
        span: Span,
    ) -> Result<(), LowerError> {
        // 1. Evaluate start and end once in the current (pre-loop) block.
        let (start_val, loop_var_ty) = self.lower_expr(start)?;
        let (end_val, _) = self.lower_expr(end)?;

        // Defensive: the loop variable must be an integer scalar.  A non-integer
        // type here usually means the start expression was something unexpected
        // (e.g. a bare side-effecting call before the loop that the parser
        // attached to start).  Fail clearly rather than propagating a bad type
        // through all the block params.
        if !matches!(
            loop_var_ty,
            IrType::Scalar(
                DType::I64
                    | DType::I32
                    | DType::U64
                    | DType::U32
                    | DType::USize
                    | DType::I8
                    | DType::U8
            )
        ) {
            return Err(LowerError::TypeMismatch {
                expected: "integer scalar (for-loop range variable)".to_string(),
                found: format!("{}", loop_var_ty),
                span,
            });
        }

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

        let mut initial_vals: Vec<ValueId> = loop_vars.iter().map(|(_, v, _)| *v).collect();

        // A taped loop variable carries its tape handle across the back-edge as
        // well as its primal; see the same treatment in `lower_while` (#49).
        let taped_loop_vars: Vec<(usize, ValueId)> = loop_vars
            .iter()
            .enumerate()
            .filter_map(|(i, (_, v, _))| self.tape_nodes.get(v).map(|h| (i, *h)))
            .collect();
        for (_, handle) in &taped_loop_vars {
            initial_vals.push(*handle);
        }

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
        let header_handle_params: Vec<ValueId> = taped_loop_vars
            .iter()
            .map(|(i, _)| {
                let name = format!("{}$tape", loop_vars[*i].0);
                self.builder
                    .add_block_param(header_bb, Some(&name), IrType::TapeRef)
            })
            .collect();

        // 6. Merge block params (receive final values on loop exit).
        let mut merge_params: Vec<ValueId> = Vec::new();
        for (name, _, ty) in &loop_vars {
            let p = self
                .builder
                .add_block_param(merge_bb, Some(name), ty.clone());
            merge_params.push(p);
        }
        let merge_handle_params: Vec<ValueId> = taped_loop_vars
            .iter()
            .map(|(i, _)| {
                let name = format!("{}$tape", loop_vars[*i].0);
                self.builder
                    .add_block_param(merge_bb, Some(&name), IrType::TapeRef)
            })
            .collect();

        // 7. Branch from current block to header with initial values.
        self.builder.push_instr(
            IrInstr::Br {
                target: header_bb,
                args: initial_vals,
            },
            None,
        );

        // 8. Header: update scope with params, emit loop_var < end (or <=) condition.
        self.builder.set_current_block(header_bb);
        for ((name, _, ty), &param_val) in loop_vars.iter().zip(header_params.iter()) {
            self.scope.insert(name.clone(), (param_val, ty.clone()));
        }
        for ((i, _), &hp) in taped_loop_vars.iter().zip(header_handle_params.iter()) {
            self.taped_values.insert(header_params[*i]);
            self.tape_nodes.insert(header_params[*i], hp);
        }
        let loop_var_param = header_params[0]; // first param is always the loop var
        let cond_result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond_result,
                op: if inclusive { BinOp::CmpLe } else { BinOp::CmpLt },
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
                else_args: header_params
                    .iter()
                    .copied()
                    .chain(header_handle_params.iter().copied())
                    .collect(),
            },
            None,
        );

        // 9. Body block.
        self.builder.set_current_block(body_bb);
        let loop_var_names: Vec<String> = loop_vars.iter().map(|(n, _, _)| n.clone()).collect();
        self.loop_stack.push((header_bb, merge_bb, header_bb, loop_var_names, label.clone(), true));
        // Use lower_block (not lower_block_stmts) so tail expressions like `print(x)` without `;`
        // are also evaluated as side-effecting statements.
        self.lower_block(body)?;
        self.loop_stack.pop();

        // 10. Emit increment and back-edge (if body not terminated by break/continue).
        if !self.builder.is_current_block_terminated() {
            let cur_loop_var = self
                .scope
                .get(&var.name)
                .map(|(v, _)| *v)
                .unwrap_or(loop_var_param);
            let step_val = if let Some(step_expr) = step {
                self.lower_expr(step_expr)?.0
            } else {
                let one = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::ConstInt {
                        result: one,
                        value: 1,
                        ty: loop_var_ty.clone(),
                    },
                    Some(loop_var_ty.clone()),
                );
                one
            };
            let incremented = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BinOp {
                    result: incremented,
                    op: BinOp::Add,
                    lhs: cur_loop_var,
                    rhs: step_val,
                    ty: loop_var_ty.clone(),
                },
                Some(loop_var_ty.clone()),
            );
            self.scope
                .insert(var.name.clone(), (incremented, loop_var_ty));

            let mut updated_vals: Vec<ValueId> = loop_vars
                .iter()
                .map(|(name, original_val, _)| {
                    self.scope
                        .get(name)
                        .map(|(v, _)| *v)
                        .unwrap_or(*original_val)
                })
                .collect();
            for (i, incoming) in &taped_loop_vars {
                let current = updated_vals[*i];
                let handle = self.tape_nodes.get(&current).copied().unwrap_or(*incoming);
                updated_vals.push(handle);
            }
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
        for ((i, _), &hp) in taped_loop_vars.iter().zip(merge_handle_params.iter()) {
            if loop_vars[*i].0 != var.name {
                self.taped_values.insert(merge_params[*i]);
                self.tape_nodes.insert(merge_params[*i], hp);
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
        self.builder.push_instr(
            IrInstr::ListLen {
                result: len_val,
                list: iter_val,
            },
            Some(i64_ty.clone()),
        );

        // Initial index = 0.
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: idx_init,
                value: 0,
                ty: i64_ty.clone(),
            },
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
            let p = self
                .builder
                .add_block_param(header_bb, Some(name), ty.clone());
            header_params.push(p);
        }

        // Merge block params.
        let mut merge_params: Vec<ValueId> = Vec::new();
        for (name, _, ty) in &loop_vars {
            let p = self
                .builder
                .add_block_param(merge_bb, Some(name), ty.clone());
            merge_params.push(p);
        }

        // Branch from current block to header.
        self.builder.push_instr(
            IrInstr::Br {
                target: header_bb,
                args: initial_vals,
            },
            None,
        );

        // Header block: check index < len.
        self.builder.set_current_block(header_bb);
        let idx_param = header_params[0];
        for ((name, _, ty), &param_val) in loop_vars.iter().zip(header_params.iter()) {
            self.scope.insert(name.clone(), (param_val, ty.clone()));
        }

        let cond_val = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond_val,
                op: BinOp::CmpLt,
                lhs: idx_param,
                rhs: len_val,
                ty: i64_ty.clone(),
            },
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
            IrInstr::ListGet {
                result: elem_val,
                list: iter_val,
                index: idx_param,
                elem_ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );
        self.scope.insert(var.name.clone(), (elem_val, elem_ty));

        let loop_var_names: Vec<String> = loop_vars.iter().map(|(n, _, _)| n.clone()).collect();
        self.loop_stack
            .push((header_bb, merge_bb, header_bb, loop_var_names.clone(), None, false));
        let _ = self.lower_block(body)?;
        self.loop_stack.pop();

        // Emit back-edge Br if body was not terminated.
        if !self.builder.is_current_block_terminated() {
            // Increment index.
            let cur_idx = self
                .scope
                .get(&idx_name)
                .map(|(v, _)| *v)
                .unwrap_or(idx_param);
            let one = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ConstInt {
                    result: one,
                    value: 1,
                    ty: i64_ty.clone(),
                },
                Some(i64_ty.clone()),
            );
            let next_idx = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BinOp {
                    result: next_idx,
                    op: BinOp::Add,
                    lhs: cur_idx,
                    rhs: one,
                    ty: i64_ty.clone(),
                },
                Some(i64_ty.clone()),
            );
            self.scope
                .insert(idx_name.clone(), (next_idx, i64_ty.clone()));

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
        self.builder.push_instr(
            IrInstr::ListLen {
                result: len_val,
                list: base_val,
            },
            Some(i64_ty.clone()),
        );
        let out_list = self.builder.fresh_value();
        let out_list_ty = IrType::List(Box::new(mapped_elem_ty.clone()));
        self.builder.push_instr(
            IrInstr::ListNew {
                result: out_list,
                elem_ty: mapped_elem_ty.clone(),
            },
            Some(out_list_ty.clone()),
        );
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: idx_init,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );

        let header_bb = self.builder.create_block(Some("map_header"));
        let body_bb = self.builder.create_block(Some("map_body"));
        let merge_bb = self.builder.create_block(Some("map_merge"));

        let idx_param = self
            .builder
            .add_block_param(header_bb, Some("map_idx"), i64_ty.clone());
        let _idx_fin = self
            .builder
            .add_block_param(merge_bb, Some("map_idx_fin"), i64_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: header_bb,
                args: vec![idx_init],
            },
            None,
        );

        self.builder.set_current_block(header_bb);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond,
                op: BinOp::CmpLt,
                lhs: idx_param,
                rhs: len_val,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond,
                then_block: body_bb,
                then_args: vec![],
                else_block: merge_bb,
                else_args: vec![idx_param],
            },
            None,
        );

        self.builder.set_current_block(body_bb);
        let elem = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: elem,
                list: base_val,
                index: idx_param,
                elem_ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );
        let mapped = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::CallClosure {
                result: Some(mapped),
                closure: closure_val,
                args: vec![elem],
                result_ty: mapped_elem_ty.clone(),
                pass_env: true,
            },
            Some(mapped_elem_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::ListPush {
                list: out_list,
                value: mapped,
            },
            None,
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_idx = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_idx,
                op: BinOp::Add,
                lhs: idx_param,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: header_bb,
                args: vec![next_idx],
            },
            None,
        );

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
        self.builder.push_instr(
            IrInstr::ListLen {
                result: len_val,
                list: base_val,
            },
            Some(i64_ty.clone()),
        );
        let out_list = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListNew {
                result: out_list,
                elem_ty: elem_ty.clone(),
            },
            Some(IrType::List(Box::new(elem_ty.clone()))),
        );
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: idx_init,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );

        let header_bb = self.builder.create_block(Some("filter_header"));
        let body_bb = self.builder.create_block(Some("filter_body"));
        let push_bb = self.builder.create_block(Some("filter_push"));
        let inc_bb = self.builder.create_block(Some("filter_inc"));
        let merge_bb = self.builder.create_block(Some("filter_merge"));

        let idx_param = self
            .builder
            .add_block_param(header_bb, Some("filter_idx"), i64_ty.clone());
        let _idx_fin =
            self.builder
                .add_block_param(merge_bb, Some("filter_idx_fin"), i64_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: header_bb,
                args: vec![idx_init],
            },
            None,
        );

        self.builder.set_current_block(header_bb);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond,
                op: BinOp::CmpLt,
                lhs: idx_param,
                rhs: len_val,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond,
                then_block: body_bb,
                then_args: vec![],
                else_block: merge_bb,
                else_args: vec![idx_param],
            },
            None,
        );

        self.builder.set_current_block(body_bb);
        let elem = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: elem,
                list: base_val,
                index: idx_param,
                elem_ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );
        let keep = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::CallClosure {
                result: Some(keep),
                closure: closure_val,
                args: vec![elem],
                result_ty: bool_ty.clone(),
                pass_env: true,
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond: keep,
                then_block: push_bb,
                then_args: vec![],
                else_block: inc_bb,
                else_args: vec![],
            },
            None,
        );

        self.builder.set_current_block(push_bb);
        self.builder.push_instr(
            IrInstr::ListPush {
                list: out_list,
                value: elem,
            },
            None,
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: inc_bb,
                args: vec![],
            },
            None,
        );

        self.builder.set_current_block(inc_bb);
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_idx = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_idx,
                op: BinOp::Add,
                lhs: idx_param,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: header_bb,
                args: vec![next_idx],
            },
            None,
        );

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
        self.builder.push_instr(
            IrInstr::ListLen {
                result: len_val,
                list: base_val,
            },
            Some(i64_ty.clone()),
        );
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: idx_init,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );

        let header_bb = self.builder.create_block(Some("fold_header"));
        let body_bb = self.builder.create_block(Some("fold_body"));
        let merge_bb = self.builder.create_block(Some("fold_merge"));

        let idx_param = self
            .builder
            .add_block_param(header_bb, Some("fold_idx"), i64_ty.clone());
        let acc_param = self
            .builder
            .add_block_param(header_bb, Some("fold_acc"), init_ty.clone());
        let _idx_fin = self
            .builder
            .add_block_param(merge_bb, Some("fold_idx_fin"), i64_ty.clone());
        let acc_fin = self
            .builder
            .add_block_param(merge_bb, Some("fold_acc_fin"), init_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: header_bb,
                args: vec![idx_init, init_val],
            },
            None,
        );

        self.builder.set_current_block(header_bb);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond,
                op: BinOp::CmpLt,
                lhs: idx_param,
                rhs: len_val,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond,
                then_block: body_bb,
                then_args: vec![],
                else_block: merge_bb,
                else_args: vec![idx_param, acc_param],
            },
            None,
        );

        self.builder.set_current_block(body_bb);
        let elem = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: elem,
                list: base_val,
                index: idx_param,
                elem_ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );
        let new_acc = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::CallClosure {
                result: Some(new_acc),
                closure: closure_val,
                args: vec![acc_param, elem],
                result_ty: init_ty.clone(),
                pass_env: true,
            },
            Some(init_ty.clone()),
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_idx = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_idx,
                op: BinOp::Add,
                lhs: idx_param,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: header_bb,
                args: vec![next_idx, new_acc],
            },
            None,
        );

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
        let i64_ty = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);
        if args.len() != 1 {
            return Err(LowerError::Unsupported {
                detail: "list.any expects 1 argument (closure)".into(),
                span,
            });
        }
        let (closure_val, _) = self.lower_expr(&args[0])?;
        let len_val = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListLen {
                result: len_val,
                list: base_val,
            },
            Some(i64_ty.clone()),
        );
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: idx_init,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let acc_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstBool {
                result: acc_init,
                value: false,
            },
            Some(bool_ty.clone()),
        );

        let header_bb = self.builder.create_block(Some("any_header"));
        let body_bb = self.builder.create_block(Some("any_body"));
        let merge_bb = self.builder.create_block(Some("any_merge"));

        let idx_param = self
            .builder
            .add_block_param(header_bb, Some("any_idx"), i64_ty.clone());
        let acc_param = self
            .builder
            .add_block_param(header_bb, Some("any_acc"), bool_ty.clone());
        let _idx_fin = self
            .builder
            .add_block_param(merge_bb, Some("any_idx_fin"), i64_ty.clone());
        let acc_fin = self
            .builder
            .add_block_param(merge_bb, Some("any_acc_fin"), bool_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: header_bb,
                args: vec![idx_init, acc_init],
            },
            None,
        );

        self.builder.set_current_block(header_bb);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond,
                op: BinOp::CmpLt,
                lhs: idx_param,
                rhs: len_val,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond,
                then_block: body_bb,
                then_args: vec![],
                else_block: merge_bb,
                else_args: vec![idx_param, acc_param],
            },
            None,
        );

        self.builder.set_current_block(body_bb);
        let elem = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: elem,
                list: base_val,
                index: idx_param,
                elem_ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );
        let val = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::CallClosure {
                result: Some(val),
                closure: closure_val,
                args: vec![elem],
                result_ty: bool_ty.clone(),
                pass_env: true,
            },
            Some(bool_ty.clone()),
        );
        let new_acc = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: new_acc,
                op: BinOp::BitOr,
                lhs: acc_param,
                rhs: val,
                ty: bool_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_idx = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_idx,
                op: BinOp::Add,
                lhs: idx_param,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: header_bb,
                args: vec![next_idx, new_acc],
            },
            None,
        );

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
        let i64_ty = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);
        if args.len() != 1 {
            return Err(LowerError::Unsupported {
                detail: "list.all expects 1 argument (closure)".into(),
                span,
            });
        }
        let (closure_val, _) = self.lower_expr(&args[0])?;
        let len_val = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListLen {
                result: len_val,
                list: base_val,
            },
            Some(i64_ty.clone()),
        );
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: idx_init,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let acc_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstBool {
                result: acc_init,
                value: true,
            },
            Some(bool_ty.clone()),
        );

        let header_bb = self.builder.create_block(Some("all_header"));
        let body_bb = self.builder.create_block(Some("all_body"));
        let merge_bb = self.builder.create_block(Some("all_merge"));

        let idx_param = self
            .builder
            .add_block_param(header_bb, Some("all_idx"), i64_ty.clone());
        let acc_param = self
            .builder
            .add_block_param(header_bb, Some("all_acc"), bool_ty.clone());
        let _idx_fin = self
            .builder
            .add_block_param(merge_bb, Some("all_idx_fin"), i64_ty.clone());
        let acc_fin = self
            .builder
            .add_block_param(merge_bb, Some("all_acc_fin"), bool_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: header_bb,
                args: vec![idx_init, acc_init],
            },
            None,
        );

        self.builder.set_current_block(header_bb);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond,
                op: BinOp::CmpLt,
                lhs: idx_param,
                rhs: len_val,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond,
                then_block: body_bb,
                then_args: vec![],
                else_block: merge_bb,
                else_args: vec![idx_param, acc_param],
            },
            None,
        );

        self.builder.set_current_block(body_bb);
        let elem = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: elem,
                list: base_val,
                index: idx_param,
                elem_ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );
        let val = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::CallClosure {
                result: Some(val),
                closure: closure_val,
                args: vec![elem],
                result_ty: bool_ty.clone(),
                pass_env: true,
            },
            Some(bool_ty.clone()),
        );
        let new_acc = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: new_acc,
                op: BinOp::BitAnd,
                lhs: acc_param,
                rhs: val,
                ty: bool_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_idx = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_idx,
                op: BinOp::Add,
                lhs: idx_param,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: header_bb,
                args: vec![next_idx, new_acc],
            },
            None,
        );

        self.builder.set_current_block(merge_bb);
        let _ = span;
        Ok((acc_fin, bool_ty))
    }

    // -----------------------------------------------------------------------
    // ML/AI intrinsics (Phases 77–80)
    // All are macro-expanded to existing IR ops — no new IrInstr variants.
    // -----------------------------------------------------------------------

    // ── Phase 77: Array creation ────────────────────────────────────────────

    fn lower_ml_zeros(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        if args.len() != 1 {
            return Err(LowerError::Unsupported {
                detail: "zeros(n) expects 1 argument".into(),
                span,
            });
        }
        let (n_val, _) = self.lower_expr(&args[0])?;
        let f64_ty = IrType::Scalar(DType::F64);
        let i64_ty = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);
        let list_ty = IrType::List(Box::new(f64_ty.clone()));

        let result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListNew {
                result,
                elem_ty: f64_ty.clone(),
            },
            Some(list_ty.clone()),
        );
        let zero_f = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstFloat {
                result: zero_f,
                value: 0.0,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: idx_init,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );

        let hdr = self.builder.create_block(Some("zeros_hdr"));
        let body = self.builder.create_block(Some("zeros_body"));
        let merge = self.builder.create_block(Some("zeros_merge"));
        let idx = self
            .builder
            .add_block_param(hdr, Some("zi"), i64_ty.clone());
        let _ = self
            .builder
            .add_block_param(merge, Some("zi_fin"), i64_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![idx_init],
            },
            None,
        );
        self.builder.set_current_block(hdr);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond,
                op: BinOp::CmpLt,
                lhs: idx,
                rhs: n_val,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond,
                then_block: body,
                then_args: vec![],
                else_block: merge,
                else_args: vec![idx],
            },
            None,
        );

        self.builder.set_current_block(body);
        self.builder.push_instr(
            IrInstr::ListPush {
                list: result,
                value: zero_f,
            },
            None,
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_i,
                op: BinOp::Add,
                lhs: idx,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![next_i],
            },
            None,
        );

        self.builder.set_current_block(merge);
        let _ = span;
        Ok((result, list_ty))
    }

    fn lower_ml_ones(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        if args.len() != 1 {
            return Err(LowerError::Unsupported {
                detail: "ones(n) expects 1 argument".into(),
                span,
            });
        }
        let (n_val, _) = self.lower_expr(&args[0])?;
        let f64_ty = IrType::Scalar(DType::F64);
        let i64_ty = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);
        let list_ty = IrType::List(Box::new(f64_ty.clone()));

        let result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListNew {
                result,
                elem_ty: f64_ty.clone(),
            },
            Some(list_ty.clone()),
        );
        let one_f = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstFloat {
                result: one_f,
                value: 1.0,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: idx_init,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );

        let hdr = self.builder.create_block(Some("ones_hdr"));
        let body = self.builder.create_block(Some("ones_body"));
        let merge = self.builder.create_block(Some("ones_merge"));
        let idx = self
            .builder
            .add_block_param(hdr, Some("oi"), i64_ty.clone());
        let _ = self
            .builder
            .add_block_param(merge, Some("oi_fin"), i64_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![idx_init],
            },
            None,
        );
        self.builder.set_current_block(hdr);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond,
                op: BinOp::CmpLt,
                lhs: idx,
                rhs: n_val,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond,
                then_block: body,
                then_args: vec![],
                else_block: merge,
                else_args: vec![idx],
            },
            None,
        );

        self.builder.set_current_block(body);
        self.builder.push_instr(
            IrInstr::ListPush {
                list: result,
                value: one_f,
            },
            None,
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_i,
                op: BinOp::Add,
                lhs: idx,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![next_i],
            },
            None,
        );

        self.builder.set_current_block(merge);
        let _ = span;
        Ok((result, list_ty))
    }

    fn lower_ml_fill(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        if args.len() != 2 {
            return Err(LowerError::Unsupported {
                detail: "fill(n, v) expects 2 arguments".into(),
                span,
            });
        }
        let (n_val, _) = self.lower_expr(&args[0])?;
        let (val_raw, val_ty) = self.lower_expr(&args[1])?;
        let f64_ty = IrType::Scalar(DType::F64);
        // Coerce fill value to f64 if needed
        let val_v = if val_ty != f64_ty {
            let coerced = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::Cast {
                    result: coerced,
                    operand: val_raw,
                    from_ty: val_ty,
                    to_ty: f64_ty.clone(),
                },
                Some(f64_ty.clone()),
            );
            coerced
        } else {
            val_raw
        };
        let i64_ty = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);
        let list_ty = IrType::List(Box::new(f64_ty.clone()));

        let result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListNew {
                result,
                elem_ty: f64_ty.clone(),
            },
            Some(list_ty.clone()),
        );
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: idx_init,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );

        let hdr = self.builder.create_block(Some("fill_hdr"));
        let body = self.builder.create_block(Some("fill_body"));
        let merge = self.builder.create_block(Some("fill_merge"));
        let idx = self
            .builder
            .add_block_param(hdr, Some("fi"), i64_ty.clone());
        let _ = self
            .builder
            .add_block_param(merge, Some("fi_fin"), i64_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![idx_init],
            },
            None,
        );
        self.builder.set_current_block(hdr);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond,
                op: BinOp::CmpLt,
                lhs: idx,
                rhs: n_val,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond,
                then_block: body,
                then_args: vec![],
                else_block: merge,
                else_args: vec![idx],
            },
            None,
        );

        self.builder.set_current_block(body);
        self.builder.push_instr(
            IrInstr::ListPush {
                list: result,
                value: val_v,
            },
            None,
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_i,
                op: BinOp::Add,
                lhs: idx,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![next_i],
            },
            None,
        );

        self.builder.set_current_block(merge);
        let _ = span;
        Ok((result, list_ty))
    }

    fn lower_ml_linspace(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        // linspace(start: f64, end: f64, n: i64) -> list<f64>
        if args.len() != 3 {
            return Err(LowerError::Unsupported {
                detail: "linspace(start, end, n) expects 3 arguments".into(),
                span,
            });
        }
        let (start_raw, start_ty) = self.lower_expr(&args[0])?;
        let (end_raw, end_ty) = self.lower_expr(&args[1])?;
        let (n_val, _) = self.lower_expr(&args[2])?;
        let f64_ty = IrType::Scalar(DType::F64);
        let i64_ty = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);
        let list_ty = IrType::List(Box::new(f64_ty.clone()));
        // Coerce start/end to f64
        let start_v = if start_ty != f64_ty {
            let c = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::Cast {
                    result: c,
                    operand: start_raw,
                    from_ty: start_ty,
                    to_ty: f64_ty.clone(),
                },
                Some(f64_ty.clone()),
            );
            c
        } else {
            start_raw
        };
        let end_v = if end_ty != f64_ty {
            let c = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::Cast {
                    result: c,
                    operand: end_raw,
                    from_ty: end_ty,
                    to_ty: f64_ty.clone(),
                },
                Some(f64_ty.clone()),
            );
            c
        } else {
            end_raw
        };

        let result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListNew {
                result,
                elem_ty: f64_ty.clone(),
            },
            Some(list_ty.clone()),
        );

        // step = (end - start) / (n - 1)
        let range = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: range,
                op: BinOp::Sub,
                lhs: end_v,
                rhs: start_v,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        // n_f = cast(n, f64)
        let n_f = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::Cast {
                result: n_f,
                operand: n_val,
                from_ty: i64_ty.clone(),
                to_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let one_f = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstFloat {
                result: one_f,
                value: 1.0,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let n_m1_f = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: n_m1_f,
                op: BinOp::Sub,
                lhs: n_f,
                rhs: one_f,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let step = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: step,
                op: BinOp::Div,
                lhs: range,
                rhs: n_m1_f,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );

        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: idx_init,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );

        let hdr = self.builder.create_block(Some("lsp_hdr"));
        let body = self.builder.create_block(Some("lsp_body"));
        let merge = self.builder.create_block(Some("lsp_merge"));
        let idx = self
            .builder
            .add_block_param(hdr, Some("lsp_i"), i64_ty.clone());
        let _ = self
            .builder
            .add_block_param(merge, Some("lsp_i_fin"), i64_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![idx_init],
            },
            None,
        );
        self.builder.set_current_block(hdr);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond,
                op: BinOp::CmpLt,
                lhs: idx,
                rhs: n_val,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond,
                then_block: body,
                then_args: vec![],
                else_block: merge,
                else_args: vec![idx],
            },
            None,
        );

        self.builder.set_current_block(body);
        // val = start + i * step
        let idx_f = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::Cast {
                result: idx_f,
                operand: idx,
                from_ty: i64_ty.clone(),
                to_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let offset = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: offset,
                op: BinOp::Mul,
                lhs: idx_f,
                rhs: step,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let val = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: val,
                op: BinOp::Add,
                lhs: start_v,
                rhs: offset,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::ListPush {
                list: result,
                value: val,
            },
            None,
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_i,
                op: BinOp::Add,
                lhs: idx,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![next_i],
            },
            None,
        );

        self.builder.set_current_block(merge);
        let _ = span;
        Ok((result, list_ty))
    }

    fn lower_ml_arange(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        // arange(start: f64, end: f64, step: f64) -> list<f64>
        if args.len() != 3 {
            return Err(LowerError::Unsupported {
                detail: "arange(start, end, step) expects 3 arguments".into(),
                span,
            });
        }
        let (start_raw, start_ty) = self.lower_expr(&args[0])?;
        let (end_raw, end_ty) = self.lower_expr(&args[1])?;
        let (step_raw, step_ty) = self.lower_expr(&args[2])?;
        let f64_ty = IrType::Scalar(DType::F64);
        let bool_ty = IrType::Scalar(DType::Bool);
        let list_ty = IrType::List(Box::new(f64_ty.clone()));

        // Coerce all inputs to f64
        let coerce = |builder: &mut crate::ir::module::IrFunctionBuilder,
                      v: ValueId,
                      ty: IrType|
         -> ValueId {
            if ty == f64_ty {
                v
            } else {
                let c = builder.fresh_value();
                builder.push_instr(
                    IrInstr::Cast {
                        result: c,
                        operand: v,
                        from_ty: ty,
                        to_ty: f64_ty.clone(),
                    },
                    Some(f64_ty.clone()),
                );
                c
            }
        };
        let start_v = coerce(&mut self.builder, start_raw, start_ty);
        let end_v = coerce(&mut self.builder, end_raw, end_ty);
        let step_v = coerce(&mut self.builder, step_raw, step_ty);

        let result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListNew {
                result,
                elem_ty: f64_ty.clone(),
            },
            Some(list_ty.clone()),
        );

        // Loop: cur = start; while cur < end { push(cur); cur += step }
        let hdr = self.builder.create_block(Some("arange_hdr"));
        let body = self.builder.create_block(Some("arange_body"));
        let merge = self.builder.create_block(Some("arange_merge"));
        let cur = self
            .builder
            .add_block_param(hdr, Some("ar_cur"), f64_ty.clone());
        let _ = self
            .builder
            .add_block_param(merge, Some("ar_fin"), f64_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![start_v],
            },
            None,
        );
        self.builder.set_current_block(hdr);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond,
                op: BinOp::CmpLt,
                lhs: cur,
                rhs: end_v,
                ty: f64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond,
                then_block: body,
                then_args: vec![],
                else_block: merge,
                else_args: vec![cur],
            },
            None,
        );

        self.builder.set_current_block(body);
        self.builder.push_instr(
            IrInstr::ListPush {
                list: result,
                value: cur,
            },
            None,
        );
        let next_cur = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_cur,
                op: BinOp::Add,
                lhs: cur,
                rhs: step_v,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![next_cur],
            },
            None,
        );

        self.builder.set_current_block(merge);
        let _ = span;
        Ok((result, list_ty))
    }

    // ── Phase 78: Array reductions ──────────────────────────────────────────

    /// Shared loop body for f64 accumulator reductions.
    /// Returns (acc_fin, f64_ty) — caller emits the loop with custom acc update.
    fn ml_reduce_loop(
        &mut self,
        prefix: &str,
        list_val: ValueId,
        elem_ty: IrType,
        acc_init: ValueId,
    ) -> (
        crate::ir::block::BlockId,
        crate::ir::block::BlockId,
        crate::ir::block::BlockId,
        ValueId,
        ValueId,
        ValueId,
        ValueId,
    ) {
        let i64_ty = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);
        let f64_ty = elem_ty.clone();

        let len = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListLen {
                result: len,
                list: list_val,
            },
            Some(i64_ty.clone()),
        );
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: idx_init,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );

        let hdr = self.builder.create_block(Some(&format!("{}_hdr", prefix)));
        let body = self.builder.create_block(Some(&format!("{}_body", prefix)));
        let merge = self
            .builder
            .create_block(Some(&format!("{}_merge", prefix)));

        let idx = self
            .builder
            .add_block_param(hdr, Some(&format!("{}_i", prefix)), i64_ty.clone());
        let acc =
            self.builder
                .add_block_param(hdr, Some(&format!("{}_acc", prefix)), f64_ty.clone());
        let _idx_fin =
            self.builder
                .add_block_param(merge, Some(&format!("{}_if", prefix)), i64_ty.clone());
        let acc_fin =
            self.builder
                .add_block_param(merge, Some(&format!("{}_af", prefix)), f64_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![idx_init, acc_init],
            },
            None,
        );
        self.builder.set_current_block(hdr);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond,
                op: BinOp::CmpLt,
                lhs: idx,
                rhs: len,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond,
                then_block: body,
                then_args: vec![],
                else_block: merge,
                else_args: vec![idx, acc],
            },
            None,
        );

        // Return values needed by caller to populate body
        (hdr, body, merge, idx, acc, acc_fin, len)
    }

    fn lower_ml_list_sum(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        if args.len() != 1 {
            return Err(LowerError::Unsupported {
                detail: "list_sum(v) expects 1 argument".into(),
                span,
            });
        }
        let (v_val, v_ty) = self.lower_expr(&args[0])?;
        let elem_ty = match &v_ty {
            IrType::List(e) => *e.clone(),
            _ => IrType::Scalar(DType::F64),
        };
        let i64_ty = IrType::Scalar(DType::I64);

        let acc_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstFloat {
                result: acc_init,
                value: 0.0,
                ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );

        let (hdr, body, merge, idx, acc, acc_fin, _len) =
            self.ml_reduce_loop("sum", v_val, elem_ty.clone(), acc_init);

        self.builder.set_current_block(body);
        let elem = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: elem,
                list: v_val,
                index: idx,
                elem_ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );
        let new_acc = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: new_acc,
                op: BinOp::Add,
                lhs: acc,
                rhs: elem,
                ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_i,
                op: BinOp::Add,
                lhs: idx,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![next_i, new_acc],
            },
            None,
        );

        self.builder.set_current_block(merge);
        let _ = span;
        Ok((acc_fin, elem_ty))
    }

    fn lower_ml_list_mean(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        if args.len() != 1 {
            return Err(LowerError::Unsupported {
                detail: "list_mean(v) expects 1 argument".into(),
                span,
            });
        }
        let f64_ty = IrType::Scalar(DType::F64);
        let i64_ty = IrType::Scalar(DType::I64);
        // Reuse list_sum then divide by len
        let (sum_v, _) = self.lower_ml_list_sum(args, span)?;
        let v_val = self.lower_expr(&args[0])?.0; // re-lower (already computed, but we need len)
        let len_v = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListLen {
                result: len_v,
                list: v_val,
            },
            Some(i64_ty.clone()),
        );
        let len_f = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::Cast {
                result: len_f,
                operand: len_v,
                from_ty: i64_ty.clone(),
                to_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let mean_v = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: mean_v,
                op: BinOp::Div,
                lhs: sum_v,
                rhs: len_f,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        Ok((mean_v, f64_ty))
    }

    fn lower_ml_list_max_val(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        if args.len() != 1 {
            return Err(LowerError::Unsupported {
                detail: "list_max_val(v) expects 1 argument".into(),
                span,
            });
        }
        let (v_val, v_ty) = self.lower_expr(&args[0])?;
        let elem_ty = match &v_ty {
            IrType::List(e) => *e.clone(),
            _ => IrType::Scalar(DType::F64),
        };
        let i64_ty = IrType::Scalar(DType::I64);

        // Initialize with first element
        let zero_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: zero_i,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let acc_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: acc_init,
                list: v_val,
                index: zero_i,
                elem_ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );

        let (hdr, body, merge, idx, acc, acc_fin, _len) =
            self.ml_reduce_loop("max", v_val, elem_ty.clone(), acc_init);

        self.builder.set_current_block(body);
        let elem = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: elem,
                list: v_val,
                index: idx,
                elem_ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );
        let new_acc = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: new_acc,
                op: BinOp::Max,
                lhs: acc,
                rhs: elem,
                ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_i,
                op: BinOp::Add,
                lhs: idx,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![next_i, new_acc],
            },
            None,
        );

        self.builder.set_current_block(merge);
        let _ = span;
        Ok((acc_fin, elem_ty))
    }

    fn lower_ml_list_min_val(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        if args.len() != 1 {
            return Err(LowerError::Unsupported {
                detail: "list_min_val(v) expects 1 argument".into(),
                span,
            });
        }
        let (v_val, v_ty) = self.lower_expr(&args[0])?;
        let elem_ty = match &v_ty {
            IrType::List(e) => *e.clone(),
            _ => IrType::Scalar(DType::F64),
        };
        let i64_ty = IrType::Scalar(DType::I64);

        let zero_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: zero_i,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let acc_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: acc_init,
                list: v_val,
                index: zero_i,
                elem_ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );

        let (hdr, body, merge, idx, acc, acc_fin, _len) =
            self.ml_reduce_loop("min", v_val, elem_ty.clone(), acc_init);

        self.builder.set_current_block(body);
        let elem = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: elem,
                list: v_val,
                index: idx,
                elem_ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );
        let new_acc = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: new_acc,
                op: BinOp::Min,
                lhs: acc,
                rhs: elem,
                ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_i,
                op: BinOp::Add,
                lhs: idx,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![next_i, new_acc],
            },
            None,
        );

        self.builder.set_current_block(merge);
        let _ = span;
        Ok((acc_fin, elem_ty))
    }

    fn lower_ml_list_std(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        // std(v) = sqrt(mean((v - mean(v))^2))
        // Computed in two passes via two calls to ml_list_sum and arithmetic
        if args.len() != 1 {
            return Err(LowerError::Unsupported {
                detail: "list_std(v) expects 1 argument".into(),
                span,
            });
        }
        let f64_ty = IrType::Scalar(DType::F64);
        let i64_ty = IrType::Scalar(DType::I64);
        let elem_ty = f64_ty.clone();

        let (v_val, _v_ty) = self.lower_expr(&args[0])?;

        // mean
        let len_v = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListLen {
                result: len_v,
                list: v_val,
            },
            Some(i64_ty.clone()),
        );
        let len_f = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::Cast {
                result: len_f,
                operand: len_v,
                from_ty: i64_ty.clone(),
                to_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );

        // sum for mean
        let acc0 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstFloat {
                result: acc0,
                value: 0.0,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let (h1, b1, m1, i1, a1, sum_v, _) =
            self.ml_reduce_loop("std_sum", v_val, elem_ty.clone(), acc0);
        self.builder.set_current_block(b1);
        let e1 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: e1,
                list: v_val,
                index: i1,
                elem_ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );
        let na1 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: na1,
                op: BinOp::Add,
                lhs: a1,
                rhs: e1,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let one1 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one1,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let ni1 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: ni1,
                op: BinOp::Add,
                lhs: i1,
                rhs: one1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: h1,
                args: vec![ni1, na1],
            },
            None,
        );
        self.builder.set_current_block(m1);

        let mean_v = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: mean_v,
                op: BinOp::Div,
                lhs: sum_v,
                rhs: len_f,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );

        // sum of squared deviations
        let acc2 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstFloat {
                result: acc2,
                value: 0.0,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let (h2, b2, m2, i2, a2, var_sum, _) =
            self.ml_reduce_loop("std_var", v_val, elem_ty.clone(), acc2);
        self.builder.set_current_block(b2);
        let e2 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: e2,
                list: v_val,
                index: i2,
                elem_ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );
        let diff = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: diff,
                op: BinOp::Sub,
                lhs: e2,
                rhs: mean_v,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let sq = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: sq,
                op: BinOp::Mul,
                lhs: diff,
                rhs: diff,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let na2 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: na2,
                op: BinOp::Add,
                lhs: a2,
                rhs: sq,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let one2 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one2,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let ni2 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: ni2,
                op: BinOp::Add,
                lhs: i2,
                rhs: one2,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: h2,
                args: vec![ni2, na2],
            },
            None,
        );
        self.builder.set_current_block(m2);

        let var = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: var,
                op: BinOp::Div,
                lhs: var_sum,
                rhs: len_f,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let std_v = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::UnaryOp {
                result: std_v,
                op: ScalarUnaryOp::Sqrt,
                operand: var,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );

        let _ = span;
        Ok((std_v, f64_ty))
    }

    fn lower_ml_list_norm(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        // norm(v) = sqrt(sum(v[i]^2))
        if args.len() != 1 {
            return Err(LowerError::Unsupported {
                detail: "list_norm(v) expects 1 argument".into(),
                span,
            });
        }
        let f64_ty = IrType::Scalar(DType::F64);
        let i64_ty = IrType::Scalar(DType::I64);
        let elem_ty = f64_ty.clone();
        let (v_val, _) = self.lower_expr(&args[0])?;

        let acc0 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstFloat {
                result: acc0,
                value: 0.0,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let (hdr, body, merge, idx, acc, sum_sq, _) =
            self.ml_reduce_loop("norm", v_val, elem_ty.clone(), acc0);

        self.builder.set_current_block(body);
        let elem = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: elem,
                list: v_val,
                index: idx,
                elem_ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );
        let sq = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: sq,
                op: BinOp::Mul,
                lhs: elem,
                rhs: elem,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let new_acc = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: new_acc,
                op: BinOp::Add,
                lhs: acc,
                rhs: sq,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_i,
                op: BinOp::Add,
                lhs: idx,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![next_i, new_acc],
            },
            None,
        );

        self.builder.set_current_block(merge);
        let norm_v = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::UnaryOp {
                result: norm_v,
                op: ScalarUnaryOp::Sqrt,
                operand: sum_sq,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let _ = span;
        Ok((norm_v, f64_ty))
    }

    fn lower_ml_list_dot(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        // dot(a, b) = sum(a[i]*b[i])
        if args.len() != 2 {
            return Err(LowerError::Unsupported {
                detail: "list_dot(a, b) expects 2 arguments".into(),
                span,
            });
        }
        let f64_ty = IrType::Scalar(DType::F64);
        let i64_ty = IrType::Scalar(DType::I64);
        let elem_ty = f64_ty.clone();
        let (a_val, _) = self.lower_expr(&args[0])?;
        let (b_val, _) = self.lower_expr(&args[1])?;

        let acc0 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstFloat {
                result: acc0,
                value: 0.0,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let (hdr, body, merge, idx, acc, dot_v, _) =
            self.ml_reduce_loop("dot", a_val, elem_ty.clone(), acc0);

        self.builder.set_current_block(body);
        let ea = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: ea,
                list: a_val,
                index: idx,
                elem_ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );
        let eb = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: eb,
                list: b_val,
                index: idx,
                elem_ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );
        let prod = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: prod,
                op: BinOp::Mul,
                lhs: ea,
                rhs: eb,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let new_acc = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: new_acc,
                op: BinOp::Add,
                lhs: acc,
                rhs: prod,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_i,
                op: BinOp::Add,
                lhs: idx,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![next_i, new_acc],
            },
            None,
        );

        self.builder.set_current_block(merge);
        let _ = span;
        Ok((dot_v, f64_ty))
    }

    // ── Phase 79: Elementwise ops ───────────────────────────────────────────

    /// Generic elementwise binary op on two lists: result[i] = op(a[i], b[i]).
    fn lower_ml_list_binop(
        &mut self,
        args: &[AstExpr],
        span: Span,
        op: BinOp,
    ) -> Result<(ValueId, IrType), LowerError> {
        if args.len() != 2 {
            return Err(LowerError::Unsupported {
                detail: "elementwise list op expects 2 arguments".into(),
                span,
            });
        }
        let f64_ty = IrType::Scalar(DType::F64);
        let i64_ty = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);
        let elem_ty = f64_ty.clone();
        let list_ty = IrType::List(Box::new(f64_ty.clone()));

        let (a_val, _) = self.lower_expr(&args[0])?;
        let (b_val, _) = self.lower_expr(&args[1])?;

        let result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListNew {
                result,
                elem_ty: elem_ty.clone(),
            },
            Some(list_ty.clone()),
        );
        let len_v = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListLen {
                result: len_v,
                list: a_val,
            },
            Some(i64_ty.clone()),
        );
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: idx_init,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );

        let hdr = self.builder.create_block(Some("lbop_hdr"));
        let body = self.builder.create_block(Some("lbop_body"));
        let merge = self.builder.create_block(Some("lbop_merge"));
        let idx = self
            .builder
            .add_block_param(hdr, Some("lbop_i"), i64_ty.clone());
        let _ = self
            .builder
            .add_block_param(merge, Some("lbop_if"), i64_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![idx_init],
            },
            None,
        );
        self.builder.set_current_block(hdr);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond,
                op: BinOp::CmpLt,
                lhs: idx,
                rhs: len_v,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond,
                then_block: body,
                then_args: vec![],
                else_block: merge,
                else_args: vec![idx],
            },
            None,
        );

        self.builder.set_current_block(body);
        let ea = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: ea,
                list: a_val,
                index: idx,
                elem_ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );
        let eb = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: eb,
                list: b_val,
                index: idx,
                elem_ty: elem_ty.clone(),
            },
            Some(elem_ty.clone()),
        );
        let out = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: out,
                op,
                lhs: ea,
                rhs: eb,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::ListPush {
                list: result,
                value: out,
            },
            None,
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_i,
                op: BinOp::Add,
                lhs: idx,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![next_i],
            },
            None,
        );

        self.builder.set_current_block(merge);
        let _ = span;
        Ok((result, list_ty))
    }

    fn lower_ml_list_scale(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        // list_scale(v, s) = [v[i] * s for i in 0..len(v)]
        if args.len() != 2 {
            return Err(LowerError::Unsupported {
                detail: "list_scale(v, s) expects 2 arguments".into(),
                span,
            });
        }
        let f64_ty = IrType::Scalar(DType::F64);
        let i64_ty = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);
        let list_ty = IrType::List(Box::new(f64_ty.clone()));

        let (v_val, _) = self.lower_expr(&args[0])?;
        let (s_raw, s_ty) = self.lower_expr(&args[1])?;
        // Coerce scalar to f64
        let s_val = if s_ty != f64_ty {
            let c = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::Cast {
                    result: c,
                    operand: s_raw,
                    from_ty: s_ty,
                    to_ty: f64_ty.clone(),
                },
                Some(f64_ty.clone()),
            );
            c
        } else {
            s_raw
        };

        let result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListNew {
                result,
                elem_ty: f64_ty.clone(),
            },
            Some(list_ty.clone()),
        );
        let len_v = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListLen {
                result: len_v,
                list: v_val,
            },
            Some(i64_ty.clone()),
        );
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: idx_init,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );

        let hdr = self.builder.create_block(Some("scl_hdr"));
        let body = self.builder.create_block(Some("scl_body"));
        let merge = self.builder.create_block(Some("scl_merge"));
        let idx = self
            .builder
            .add_block_param(hdr, Some("scl_i"), i64_ty.clone());
        let _ = self
            .builder
            .add_block_param(merge, Some("scl_if"), i64_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![idx_init],
            },
            None,
        );
        self.builder.set_current_block(hdr);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond,
                op: BinOp::CmpLt,
                lhs: idx,
                rhs: len_v,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond,
                then_block: body,
                then_args: vec![],
                else_block: merge,
                else_args: vec![idx],
            },
            None,
        );

        self.builder.set_current_block(body);
        let elem = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: elem,
                list: v_val,
                index: idx,
                elem_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let out = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: out,
                op: BinOp::Mul,
                lhs: elem,
                rhs: s_val,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::ListPush {
                list: result,
                value: out,
            },
            None,
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_i,
                op: BinOp::Add,
                lhs: idx,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![next_i],
            },
            None,
        );

        self.builder.set_current_block(merge);
        let _ = span;
        Ok((result, list_ty))
    }

    fn lower_ml_list_relu(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        // relu(v) = [max(0, v[i]) for i in 0..len(v)]
        if args.len() != 1 {
            return Err(LowerError::Unsupported {
                detail: "list_relu(v) expects 1 argument".into(),
                span,
            });
        }
        let f64_ty = IrType::Scalar(DType::F64);
        let i64_ty = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);
        let list_ty = IrType::List(Box::new(f64_ty.clone()));

        let (v_val, _) = self.lower_expr(&args[0])?;
        let zero_f = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstFloat {
                result: zero_f,
                value: 0.0,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );

        let result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListNew {
                result,
                elem_ty: f64_ty.clone(),
            },
            Some(list_ty.clone()),
        );
        let len_v = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListLen {
                result: len_v,
                list: v_val,
            },
            Some(i64_ty.clone()),
        );
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: idx_init,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );

        let hdr = self.builder.create_block(Some("relu_hdr"));
        let body = self.builder.create_block(Some("relu_body"));
        let merge = self.builder.create_block(Some("relu_merge"));
        let idx = self
            .builder
            .add_block_param(hdr, Some("relu_i"), i64_ty.clone());
        let _ = self
            .builder
            .add_block_param(merge, Some("relu_if"), i64_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![idx_init],
            },
            None,
        );
        self.builder.set_current_block(hdr);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond,
                op: BinOp::CmpLt,
                lhs: idx,
                rhs: len_v,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond,
                then_block: body,
                then_args: vec![],
                else_block: merge,
                else_args: vec![idx],
            },
            None,
        );

        self.builder.set_current_block(body);
        let elem = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: elem,
                list: v_val,
                index: idx,
                elem_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let out = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: out,
                op: BinOp::Max,
                lhs: elem,
                rhs: zero_f,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::ListPush {
                list: result,
                value: out,
            },
            None,
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_i,
                op: BinOp::Add,
                lhs: idx,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![next_i],
            },
            None,
        );

        self.builder.set_current_block(merge);
        let _ = span;
        Ok((result, list_ty))
    }

    fn lower_ml_list_sigmoid(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        // sigmoid(v) = [1 / (1 + exp(-v[i])) for i]
        if args.len() != 1 {
            return Err(LowerError::Unsupported {
                detail: "list_sigmoid(v) expects 1 argument".into(),
                span,
            });
        }
        let f64_ty = IrType::Scalar(DType::F64);
        let i64_ty = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);
        let list_ty = IrType::List(Box::new(f64_ty.clone()));

        let (v_val, _) = self.lower_expr(&args[0])?;
        let one_f = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstFloat {
                result: one_f,
                value: 1.0,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );

        let result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListNew {
                result,
                elem_ty: f64_ty.clone(),
            },
            Some(list_ty.clone()),
        );
        let len_v = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListLen {
                result: len_v,
                list: v_val,
            },
            Some(i64_ty.clone()),
        );
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: idx_init,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );

        let hdr = self.builder.create_block(Some("sig_hdr"));
        let body = self.builder.create_block(Some("sig_body"));
        let merge = self.builder.create_block(Some("sig_merge"));
        let idx = self
            .builder
            .add_block_param(hdr, Some("sig_i"), i64_ty.clone());
        let _ = self
            .builder
            .add_block_param(merge, Some("sig_if"), i64_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![idx_init],
            },
            None,
        );
        self.builder.set_current_block(hdr);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond,
                op: BinOp::CmpLt,
                lhs: idx,
                rhs: len_v,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond,
                then_block: body,
                then_args: vec![],
                else_block: merge,
                else_args: vec![idx],
            },
            None,
        );

        self.builder.set_current_block(body);
        let elem = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: elem,
                list: v_val,
                index: idx,
                elem_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let neg_e = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::UnaryOp {
                result: neg_e,
                op: ScalarUnaryOp::Neg,
                operand: elem,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let exp_e = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::UnaryOp {
                result: exp_e,
                op: ScalarUnaryOp::Exp,
                operand: neg_e,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let denom = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: denom,
                op: BinOp::Add,
                lhs: one_f,
                rhs: exp_e,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let out = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: out,
                op: BinOp::Div,
                lhs: one_f,
                rhs: denom,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::ListPush {
                list: result,
                value: out,
            },
            None,
        );
        let one_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one_i,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_i,
                op: BinOp::Add,
                lhs: idx,
                rhs: one_i,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![next_i],
            },
            None,
        );

        self.builder.set_current_block(merge);
        let _ = span;
        Ok((result, list_ty))
    }

    fn lower_ml_list_softmax(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        // softmax(v): exp-shift stable softmax
        //   max_v = max(v)
        //   exp_v = [exp(v[i] - max_v) for i]
        //   sum_e = sum(exp_v)
        //   result = [e / sum_e for e in exp_v]
        if args.len() != 1 {
            return Err(LowerError::Unsupported {
                detail: "list_softmax(v) expects 1 argument".into(),
                span,
            });
        }
        let f64_ty = IrType::Scalar(DType::F64);
        let i64_ty = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);
        let list_ty = IrType::List(Box::new(f64_ty.clone()));

        let (v_val, _) = self.lower_expr(&args[0])?;

        // max_v using list_max_val pattern (inline to avoid re-lowering args)
        let zero_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: zero_i,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let max_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: max_init,
                list: v_val,
                index: zero_i,
                elem_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let acc0 = max_init;
        let (h1, b1, m1, i1, a1, max_v, len_v) =
            self.ml_reduce_loop("smx_max", v_val, f64_ty.clone(), acc0);
        self.builder.set_current_block(b1);
        let e1 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: e1,
                list: v_val,
                index: i1,
                elem_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let na1 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: na1,
                op: BinOp::Max,
                lhs: a1,
                rhs: e1,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let one1 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one1,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let ni1 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: ni1,
                op: BinOp::Add,
                lhs: i1,
                rhs: one1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: h1,
                args: vec![ni1, na1],
            },
            None,
        );
        self.builder.set_current_block(m1);

        // exp_v: new list of exp(v[i] - max_v)
        let exp_list = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListNew {
                result: exp_list,
                elem_ty: f64_ty.clone(),
            },
            Some(list_ty.clone()),
        );
        let idx_init2 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: idx_init2,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );

        let h2 = self.builder.create_block(Some("smx_exp_hdr"));
        let b2 = self.builder.create_block(Some("smx_exp_body"));
        let m2 = self.builder.create_block(Some("smx_exp_merge"));
        let i2 = self
            .builder
            .add_block_param(h2, Some("smx_ei"), i64_ty.clone());
        let _ = self
            .builder
            .add_block_param(m2, Some("smx_eif"), i64_ty.clone());
        self.builder.push_instr(
            IrInstr::Br {
                target: h2,
                args: vec![idx_init2],
            },
            None,
        );
        self.builder.set_current_block(h2);
        let c2 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: c2,
                op: BinOp::CmpLt,
                lhs: i2,
                rhs: len_v,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond: c2,
                then_block: b2,
                then_args: vec![],
                else_block: m2,
                else_args: vec![i2],
            },
            None,
        );
        self.builder.set_current_block(b2);
        let ve2 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: ve2,
                list: v_val,
                index: i2,
                elem_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let shifted = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: shifted,
                op: BinOp::Sub,
                lhs: ve2,
                rhs: max_v,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let expv = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::UnaryOp {
                result: expv,
                op: ScalarUnaryOp::Exp,
                operand: shifted,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::ListPush {
                list: exp_list,
                value: expv,
            },
            None,
        );
        let one2 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one2,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let ni2 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: ni2,
                op: BinOp::Add,
                lhs: i2,
                rhs: one2,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: h2,
                args: vec![ni2],
            },
            None,
        );
        self.builder.set_current_block(m2);

        // sum_exp: sum of exp_list
        let acc_s = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstFloat {
                result: acc_s,
                value: 0.0,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let (h3, b3, m3, i3, a3, sum_exp, _) =
            self.ml_reduce_loop("smx_sum", exp_list, f64_ty.clone(), acc_s);
        self.builder.set_current_block(b3);
        let e3 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: e3,
                list: exp_list,
                index: i3,
                elem_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let na3 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: na3,
                op: BinOp::Add,
                lhs: a3,
                rhs: e3,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let one3 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one3,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let ni3 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: ni3,
                op: BinOp::Add,
                lhs: i3,
                rhs: one3,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: h3,
                args: vec![ni3, na3],
            },
            None,
        );
        self.builder.set_current_block(m3);

        // normalize: result[i] = exp_list[i] / sum_exp
        let out_list = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListNew {
                result: out_list,
                elem_ty: f64_ty.clone(),
            },
            Some(list_ty.clone()),
        );
        let idx_init4 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: idx_init4,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let h4 = self.builder.create_block(Some("smx_norm_hdr"));
        let b4 = self.builder.create_block(Some("smx_norm_body"));
        let m4 = self.builder.create_block(Some("smx_norm_merge"));
        let i4 = self
            .builder
            .add_block_param(h4, Some("smx_ni"), i64_ty.clone());
        let _ = self
            .builder
            .add_block_param(m4, Some("smx_nif"), i64_ty.clone());
        self.builder.push_instr(
            IrInstr::Br {
                target: h4,
                args: vec![idx_init4],
            },
            None,
        );
        self.builder.set_current_block(h4);
        let c4 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: c4,
                op: BinOp::CmpLt,
                lhs: i4,
                rhs: len_v,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond: c4,
                then_block: b4,
                then_args: vec![],
                else_block: m4,
                else_args: vec![i4],
            },
            None,
        );
        self.builder.set_current_block(b4);
        let ev4 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: ev4,
                list: exp_list,
                index: i4,
                elem_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let norm_v = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: norm_v,
                op: BinOp::Div,
                lhs: ev4,
                rhs: sum_exp,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::ListPush {
                list: out_list,
                value: norm_v,
            },
            None,
        );
        let one4 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one4,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let ni4 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: ni4,
                op: BinOp::Add,
                lhs: i4,
                rhs: one4,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: h4,
                args: vec![ni4],
            },
            None,
        );
        self.builder.set_current_block(m4);

        let _ = span;
        Ok((out_list, list_ty))
    }

    // ── Phase 80: Loss functions and training ───────────────────────────────

    fn lower_ml_mse_loss(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        // mse_loss(pred, target) = mean((pred[i] - target[i])^2)
        if args.len() != 2 {
            return Err(LowerError::Unsupported {
                detail: "mse_loss(pred, target) expects 2 arguments".into(),
                span,
            });
        }
        let f64_ty = IrType::Scalar(DType::F64);
        let i64_ty = IrType::Scalar(DType::I64);

        let (pred_v, _) = self.lower_expr(&args[0])?;
        let (target_v, _) = self.lower_expr(&args[1])?;

        let len_v = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListLen {
                result: len_v,
                list: pred_v,
            },
            Some(i64_ty.clone()),
        );
        let acc0 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstFloat {
                result: acc0,
                value: 0.0,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );

        let (hdr, body, merge, idx, acc, sum_sq, _) =
            self.ml_reduce_loop("mse", pred_v, f64_ty.clone(), acc0);

        self.builder.set_current_block(body);
        let p_e = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: p_e,
                list: pred_v,
                index: idx,
                elem_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let t_e = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: t_e,
                list: target_v,
                index: idx,
                elem_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let diff = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: diff,
                op: BinOp::Sub,
                lhs: p_e,
                rhs: t_e,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let sq = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: sq,
                op: BinOp::Mul,
                lhs: diff,
                rhs: diff,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let new_acc = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: new_acc,
                op: BinOp::Add,
                lhs: acc,
                rhs: sq,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_i,
                op: BinOp::Add,
                lhs: idx,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![next_i, new_acc],
            },
            None,
        );

        self.builder.set_current_block(merge);
        let len_f = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::Cast {
                result: len_f,
                operand: len_v,
                from_ty: i64_ty.clone(),
                to_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let mse = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: mse,
                op: BinOp::Div,
                lhs: sum_sq,
                rhs: len_f,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );

        let _ = span;
        Ok((mse, f64_ty))
    }

    fn lower_ml_cross_entropy(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        // cross_entropy(probs: list<f64>, targets: list<f64>) = -mean(targets * log(probs + eps))
        if args.len() != 2 {
            return Err(LowerError::Unsupported {
                detail: "cross_entropy(probs, targets) expects 2 arguments".into(),
                span,
            });
        }
        let f64_ty = IrType::Scalar(DType::F64);
        let i64_ty = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);

        let (probs_v, _) = self.lower_expr(&args[0])?;
        let (target_v, _) = self.lower_expr(&args[1])?;

        let len_v = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListLen {
                result: len_v,
                list: probs_v,
            },
            Some(i64_ty.clone()),
        );

        // eps = 1e-9 to avoid log(0)
        let eps = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstFloat {
                result: eps,
                value: 1e-9,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );

        let acc_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstFloat {
                result: acc_init,
                value: 0.0,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: idx_init,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );

        let hdr = self.builder.create_block(Some("ce_hdr"));
        let body = self.builder.create_block(Some("ce_body"));
        let merge = self.builder.create_block(Some("ce_merge"));
        let idx = self
            .builder
            .add_block_param(hdr, Some("ce_i"), i64_ty.clone());
        let acc = self
            .builder
            .add_block_param(hdr, Some("ce_acc"), f64_ty.clone());
        let _ = self
            .builder
            .add_block_param(merge, Some("ce_if"), i64_ty.clone());
        let sum_v = self
            .builder
            .add_block_param(merge, Some("ce_s"), f64_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![idx_init, acc_init],
            },
            None,
        );
        self.builder.set_current_block(hdr);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond,
                op: BinOp::CmpLt,
                lhs: idx,
                rhs: len_v,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond,
                then_block: body,
                then_args: vec![],
                else_block: merge,
                else_args: vec![idx, acc],
            },
            None,
        );

        self.builder.set_current_block(body);
        let p_e = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: p_e,
                list: probs_v,
                index: idx,
                elem_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let t_e = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: t_e,
                list: target_v,
                index: idx,
                elem_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let p_eps = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: p_eps,
                op: BinOp::Add,
                lhs: p_e,
                rhs: eps,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let log_p = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::UnaryOp {
                result: log_p,
                op: ScalarUnaryOp::Log,
                operand: p_eps,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let t_log = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: t_log,
                op: BinOp::Mul,
                lhs: t_e,
                rhs: log_p,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let new_acc = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: new_acc,
                op: BinOp::Add,
                lhs: acc,
                rhs: t_log,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_i,
                op: BinOp::Add,
                lhs: idx,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![next_i, new_acc],
            },
            None,
        );

        // merge: result = -sum / len
        self.builder.set_current_block(merge);
        let len_f = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::Cast {
                result: len_f,
                operand: len_v,
                from_ty: i64_ty.clone(),
                to_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let mean_sum = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: mean_sum,
                op: BinOp::Div,
                lhs: sum_v,
                rhs: len_f,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let neg_ce = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::UnaryOp {
                result: neg_ce,
                op: ScalarUnaryOp::Neg,
                operand: mean_sum,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );

        let _ = span;
        Ok((neg_ce, f64_ty))
    }

    fn lower_ml_list_axpy(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        // axpy(alpha, x, y) = [alpha*x[i] + y[i] for i]  (BLAS: y = alpha*x + y)
        if args.len() != 3 {
            return Err(LowerError::Unsupported {
                detail: "list_axpy(alpha, x, y) expects 3 arguments".into(),
                span,
            });
        }
        let f64_ty = IrType::Scalar(DType::F64);
        let i64_ty = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);
        let list_ty = IrType::List(Box::new(f64_ty.clone()));

        let (alpha_raw, alpha_ty) = self.lower_expr(&args[0])?;
        let (a_val, _) = self.lower_expr(&args[1])?;
        let (b_val, _) = self.lower_expr(&args[2])?;
        // Coerce alpha to f64
        let s_val = if alpha_ty != f64_ty {
            let c = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::Cast {
                    result: c,
                    operand: alpha_raw,
                    from_ty: alpha_ty,
                    to_ty: f64_ty.clone(),
                },
                Some(f64_ty.clone()),
            );
            c
        } else {
            alpha_raw
        };

        let result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListNew {
                result,
                elem_ty: f64_ty.clone(),
            },
            Some(list_ty.clone()),
        );
        let len_v = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListLen {
                result: len_v,
                list: a_val,
            },
            Some(i64_ty.clone()),
        );
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: idx_init,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );

        let hdr = self.builder.create_block(Some("axpy_hdr"));
        let body = self.builder.create_block(Some("axpy_body"));
        let merge = self.builder.create_block(Some("axpy_merge"));
        let idx = self
            .builder
            .add_block_param(hdr, Some("axpy_i"), i64_ty.clone());
        let _ = self
            .builder
            .add_block_param(merge, Some("axpy_if"), i64_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![idx_init],
            },
            None,
        );
        self.builder.set_current_block(hdr);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond,
                op: BinOp::CmpLt,
                lhs: idx,
                rhs: len_v,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond,
                then_block: body,
                then_args: vec![],
                else_block: merge,
                else_args: vec![idx],
            },
            None,
        );

        self.builder.set_current_block(body);
        let ea = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: ea,
                list: a_val,
                index: idx,
                elem_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let eb = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: eb,
                list: b_val,
                index: idx,
                elem_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        // out = alpha*x[i] + y[i]
        let sa = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: sa,
                op: BinOp::Mul,
                lhs: s_val,
                rhs: ea,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let out = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: out,
                op: BinOp::Add,
                lhs: sa,
                rhs: eb,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::ListPush {
                list: result,
                value: out,
            },
            None,
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_i,
                op: BinOp::Add,
                lhs: idx,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![next_i],
            },
            None,
        );

        self.builder.set_current_block(merge);
        let _ = span;
        Ok((result, list_ty))
    }

    fn lower_ml_sgd_step(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        // sgd_step(params, grads, lr): in-place params[i] -= lr * grads[i]. Returns unit (i64 0).
        if args.len() != 3 {
            return Err(LowerError::Unsupported {
                detail: "sgd_step(params, grads, lr) expects 3 arguments".into(),
                span,
            });
        }
        let f64_ty = IrType::Scalar(DType::F64);
        let i64_ty = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);

        let (params_v, _) = self.lower_expr(&args[0])?;
        let (grads_v, _) = self.lower_expr(&args[1])?;
        let (lr_raw, lr_ty) = self.lower_expr(&args[2])?;
        // Coerce lr to f64
        let lr_val = if lr_ty != f64_ty {
            let c = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::Cast {
                    result: c,
                    operand: lr_raw,
                    from_ty: lr_ty,
                    to_ty: f64_ty.clone(),
                },
                Some(f64_ty.clone()),
            );
            c
        } else {
            lr_raw
        };

        let len_v = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListLen {
                result: len_v,
                list: params_v,
            },
            Some(i64_ty.clone()),
        );
        let idx_init = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: idx_init,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );

        let hdr = self.builder.create_block(Some("sgd_hdr"));
        let body = self.builder.create_block(Some("sgd_body"));
        let merge = self.builder.create_block(Some("sgd_merge"));
        let idx = self
            .builder
            .add_block_param(hdr, Some("sgd_i"), i64_ty.clone());
        let _ = self
            .builder
            .add_block_param(merge, Some("sgd_if"), i64_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![idx_init],
            },
            None,
        );
        self.builder.set_current_block(hdr);
        let cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: cond,
                op: BinOp::CmpLt,
                lhs: idx,
                rhs: len_v,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond,
                then_block: body,
                then_args: vec![],
                else_block: merge,
                else_args: vec![idx],
            },
            None,
        );

        self.builder.set_current_block(body);
        let p_e = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: p_e,
                list: params_v,
                index: idx,
                elem_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let g_e = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: g_e,
                list: grads_v,
                index: idx,
                elem_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let lr_g = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: lr_g,
                op: BinOp::Mul,
                lhs: lr_val,
                rhs: g_e,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let new_p = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: new_p,
                op: BinOp::Sub,
                lhs: p_e,
                rhs: lr_g,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::ListSet {
                list: params_v,
                index: idx,
                value: new_p,
            },
            None,
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_i,
                op: BinOp::Add,
                lhs: idx,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: hdr,
                args: vec![next_i],
            },
            None,
        );

        self.builder.set_current_block(merge);
        // Return unit (i64 0)
        let unit = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: unit,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let _ = span;
        Ok((unit, i64_ty))
    }

    // ── Phase 82: BLAS matmul ────────────────────────────────────────────────

    /// matmul(a, m, k, b, n) → list<f64> of length m*n
    /// A is m×k (row-major flat list), B is k×n → C is m×n.
    fn lower_ml_matmul(
        &mut self,
        args: &[AstExpr],
        span: Span,
    ) -> Result<(ValueId, IrType), LowerError> {
        if args.len() != 5 {
            return Err(LowerError::Unsupported {
                detail: "matmul(a, m, k, b, n) expects 5 arguments".into(),
                span,
            });
        }
        let f64_ty = IrType::Scalar(DType::F64);
        let i64_ty = IrType::Scalar(DType::I64);
        let bool_ty = IrType::Scalar(DType::Bool);
        let list_ty = IrType::List(Box::new(f64_ty.clone()));

        let (a_v, _) = self.lower_expr(&args[0])?;
        let (m_raw, m_ty) = self.lower_expr(&args[1])?;
        let (k_raw, k_ty) = self.lower_expr(&args[2])?;
        let (b_v, _) = self.lower_expr(&args[3])?;
        let (n_raw, n_ty) = self.lower_expr(&args[4])?;

        // Coerce m, k, n to i64 if needed
        let m_v = if m_ty != i64_ty {
            let c = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::Cast {
                    result: c,
                    operand: m_raw,
                    from_ty: m_ty,
                    to_ty: i64_ty.clone(),
                },
                Some(i64_ty.clone()),
            );
            c
        } else {
            m_raw
        };
        let k_v = if k_ty != i64_ty {
            let c = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::Cast {
                    result: c,
                    operand: k_raw,
                    from_ty: k_ty,
                    to_ty: i64_ty.clone(),
                },
                Some(i64_ty.clone()),
            );
            c
        } else {
            k_raw
        };
        let n_v = if n_ty != i64_ty {
            let c = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::Cast {
                    result: c,
                    operand: n_raw,
                    from_ty: n_ty,
                    to_ty: i64_ty.clone(),
                },
                Some(i64_ty.clone()),
            );
            c
        } else {
            n_raw
        };

        // Allocate result list C
        let c_v = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListNew {
                result: c_v,
                elem_ty: f64_ty.clone(),
            },
            Some(list_ty.clone()),
        );

        // Outer loop: i in 0..m
        let zero = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: zero,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );

        let i_hdr = self.builder.create_block(Some("mm_i_hdr"));
        let i_body = self.builder.create_block(Some("mm_i_body"));
        let i_merge = self.builder.create_block(Some("mm_i_merge"));
        let i_param = self
            .builder
            .add_block_param(i_hdr, Some("mm_i"), i64_ty.clone());
        let _ = self
            .builder
            .add_block_param(i_merge, Some("mm_if"), i64_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: i_hdr,
                args: vec![zero],
            },
            None,
        );

        // i loop header
        self.builder.set_current_block(i_hdr);
        let i_cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: i_cond,
                op: BinOp::CmpLt,
                lhs: i_param,
                rhs: m_v,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond: i_cond,
                then_block: i_body,
                then_args: vec![],
                else_block: i_merge,
                else_args: vec![i_param],
            },
            None,
        );

        // i loop body — inner loop: j in 0..n
        self.builder.set_current_block(i_body);
        let zero2 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: zero2,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );

        let j_hdr = self.builder.create_block(Some("mm_j_hdr"));
        let j_body = self.builder.create_block(Some("mm_j_body"));
        let j_merge = self.builder.create_block(Some("mm_j_merge"));
        let j_param = self
            .builder
            .add_block_param(j_hdr, Some("mm_j"), i64_ty.clone());
        let _ = self
            .builder
            .add_block_param(j_merge, Some("mm_jf"), i64_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: j_hdr,
                args: vec![zero2],
            },
            None,
        );

        // j loop header
        self.builder.set_current_block(j_hdr);
        let j_cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: j_cond,
                op: BinOp::CmpLt,
                lhs: j_param,
                rhs: n_v,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond: j_cond,
                then_block: j_body,
                then_args: vec![],
                else_block: j_merge,
                else_args: vec![j_param],
            },
            None,
        );

        // j loop body — innermost loop: kk in 0..k
        self.builder.set_current_block(j_body);
        let zero3 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: zero3,
                value: 0,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let acc0 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstFloat {
                result: acc0,
                value: 0.0,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );

        let k_hdr = self.builder.create_block(Some("mm_k_hdr"));
        let k_body = self.builder.create_block(Some("mm_k_body"));
        let k_merge = self.builder.create_block(Some("mm_k_merge"));
        let k_param = self
            .builder
            .add_block_param(k_hdr, Some("mm_kk"), i64_ty.clone());
        let k_acc = self
            .builder
            .add_block_param(k_hdr, Some("mm_acc"), f64_ty.clone());
        let _ = self
            .builder
            .add_block_param(k_merge, Some("mm_kf"), i64_ty.clone());
        let sum_v = self
            .builder
            .add_block_param(k_merge, Some("mm_sum"), f64_ty.clone());

        self.builder.push_instr(
            IrInstr::Br {
                target: k_hdr,
                args: vec![zero3, acc0],
            },
            None,
        );

        // k loop header
        self.builder.set_current_block(k_hdr);
        let k_cond = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: k_cond,
                op: BinOp::CmpLt,
                lhs: k_param,
                rhs: k_v,
                ty: i64_ty.clone(),
            },
            Some(bool_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::CondBr {
                cond: k_cond,
                then_block: k_body,
                then_args: vec![],
                else_block: k_merge,
                else_args: vec![k_param, k_acc],
            },
            None,
        );

        // k loop body: acc += a[i*k + kk] * b[kk*n + j]
        self.builder.set_current_block(k_body);
        let i_times_k = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: i_times_k,
                op: BinOp::Mul,
                lhs: i_param,
                rhs: k_v,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let a_idx = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: a_idx,
                op: BinOp::Add,
                lhs: i_times_k,
                rhs: k_param,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let a_elem = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: a_elem,
                list: a_v,
                index: a_idx,
                elem_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let kk_times_n = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: kk_times_n,
                op: BinOp::Mul,
                lhs: k_param,
                rhs: n_v,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let b_idx = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: b_idx,
                op: BinOp::Add,
                lhs: kk_times_n,
                rhs: j_param,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let b_elem = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ListGet {
                result: b_elem,
                list: b_v,
                index: b_idx,
                elem_ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let prod = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: prod,
                op: BinOp::Mul,
                lhs: a_elem,
                rhs: b_elem,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let new_acc = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: new_acc,
                op: BinOp::Add,
                lhs: k_acc,
                rhs: prod,
                ty: f64_ty.clone(),
            },
            Some(f64_ty.clone()),
        );
        let one = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_k = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_k,
                op: BinOp::Add,
                lhs: k_param,
                rhs: one,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: k_hdr,
                args: vec![next_k, new_acc],
            },
            None,
        );

        // k merge: push sum_v to C
        self.builder.set_current_block(k_merge);
        self.builder.push_instr(
            IrInstr::ListPush {
                list: c_v,
                value: sum_v,
            },
            None,
        );
        // advance j
        let one2 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one2,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_j = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_j,
                op: BinOp::Add,
                lhs: j_param,
                rhs: one2,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: j_hdr,
                args: vec![next_j],
            },
            None,
        );

        // j merge: advance i
        self.builder.set_current_block(j_merge);
        let one3 = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::ConstInt {
                result: one3,
                value: 1,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        let next_i = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::BinOp {
                result: next_i,
                op: BinOp::Add,
                lhs: i_param,
                rhs: one3,
                ty: i64_ty.clone(),
            },
            Some(i64_ty.clone()),
        );
        self.builder.push_instr(
            IrInstr::Br {
                target: i_hdr,
                args: vec![next_i],
            },
            None,
        );

        // i merge
        self.builder.set_current_block(i_merge);
        let _ = span;
        Ok((c_v, list_ty))
    }

    /// Lowers a `loop { body }` (infinite loop). `break` exits to merge_bb.
    fn lower_loop(&mut self, body: &AstBlock, label: &Option<String>, span: Span) -> Result<(), LowerError> {
        // Pre-scan body to find which variables get rebound inside the loop.
        let rebound = find_rebound_vars(body);

        // Collect the loop variables that exist in the current scope.
        let mut loop_vars: Vec<(String, ValueId, IrType)> = Vec::new();
        for name in &rebound {
            if let Some((val, ty)) = self.scope.get(name).cloned() {
                loop_vars.push((name.clone(), val, ty));
            }
        }

        let initial_vals: Vec<ValueId> = loop_vars.iter().map(|(_, v, _)| *v).collect();

        // Create header/body/merge blocks with SSA block params for loop vars.
        let header_bb = self.builder.create_block(Some("loop_header"));
        let body_bb = self.builder.create_block(Some("loop_body"));
        let merge_bb = self.builder.create_block(Some("loop_merge"));

        // Header block params (one per loop variable).
        let mut header_params: Vec<ValueId> = Vec::new();
        for (name, _, ty) in &loop_vars {
            let p = self
                .builder
                .add_block_param(header_bb, Some(name), ty.clone());
            header_params.push(p);
        }

        // Merge block params (receive final values on loop exit).
        let mut merge_params: Vec<ValueId> = Vec::new();
        for (name, _, ty) in &loop_vars {
            let p = self
                .builder
                .add_block_param(merge_bb, Some(name), ty.clone());
            merge_params.push(p);
        }

        // Branch from current block to header with initial values.
        self.builder.push_instr(
            IrInstr::Br {
                target: header_bb,
                args: initial_vals,
            },
            None,
        );

        // Header: update scope with params and jump to body.
        self.builder.set_current_block(header_bb);
        for ((name, _, ty), &param_val) in loop_vars.iter().zip(header_params.iter()) {
            self.scope.insert(name.clone(), (param_val, ty.clone()));
        }
        self.builder.push_instr(
                IrInstr::Br {
                    target: body_bb,
                    args: vec![],
                },
                None,
            );

        // Lower body block.
        self.builder.set_current_block(body_bb);
        let loop_var_names: Vec<String> = loop_vars.iter().map(|(n, _, _)| n.clone()).collect();
        self.loop_stack
            .push((header_bb, merge_bb, header_bb, loop_var_names.clone(), label.clone(), false));
        let _ = self.lower_block(body)?;
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

        // Move to merge block and update scope with final values.
        self.builder.set_current_block(merge_bb);
        for ((name, _, ty), &merge_val) in loop_vars.iter().zip(merge_params.iter()) {
            self.scope.insert(name.clone(), (merge_val, ty.clone()));
        }

        let _ = span;
        Ok(())
    }

    /// Lowers `break [label]` — jumps to the merge block of the (optionally labeled) loop.
    fn lower_break(&mut self, label: &Option<String>, span: Span) -> Result<(), LowerError> {
        let (_, merge_bb, _, loop_var_names, _, _) = if let Some(target_label) = label {
            self.loop_stack
                .iter()
                .rev()
                .find(|(_, _, _, _, lbl, _)| lbl.as_ref() == Some(target_label))
                .cloned()
                .ok_or_else(|| LowerError::Unsupported {
                    detail: format!("break with unknown label '{}'", target_label),
                    span,
                })?
        } else {
            self.loop_stack
                .last()
                .cloned()
                .ok_or_else(|| LowerError::Unsupported {
                    detail: "break outside of loop".into(),
                    span,
                })?
        };

        let mut args = Vec::with_capacity(loop_var_names.len());
        for name in &loop_var_names {
            let Some((value, _)) = self.scope.get(name) else {
                return Err(LowerError::Unsupported {
                    detail: format!("missing loop variable '{}' in break", name),
                    span,
                });
            };
            args.push(*value);
        }

        self.builder.push_instr(
            IrInstr::Br {
                target: merge_bb,
                args,
            },
            None,
        );
        Ok(())
    }

    /// Lowers `continue [label]` — jumps to the continue target of the (optionally labeled) loop.
    fn lower_continue(&mut self, label: &Option<String>, span: Span) -> Result<(), LowerError> {
        let (header_bb, _, _, loop_var_names, _, is_for_range) = if let Some(target_label) = label {
            self.loop_stack
                .iter()
                .rev()
                .find(|(_, _, _, _, lbl, _)| lbl.as_ref() == Some(target_label))
                .cloned()
                .ok_or_else(|| LowerError::Unsupported {
                    detail: format!("continue with unknown label '{}'", target_label),
                    span,
                })?
        } else {
            self.loop_stack
                .last()
                .cloned()
                .ok_or_else(|| LowerError::Unsupported {
                    detail: "continue outside of loop".into(),
                    span,
                })?
        };

        if is_for_range {
            // For-range: increment the loop var inline, then branch to header with all
            // updated values. Save and restore scope so the increment doesn't leak.
            let saved_scope: Vec<_> = self.scope.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

            let (cur_loop_var, loop_var_ty) = self
                .scope
                .get(&loop_var_names[0])
                .cloned()
                .unwrap_or((ValueId(0), IrType::Scalar(DType::I64)));
            let one = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::ConstInt { result: one, value: 1, ty: loop_var_ty.clone() },
                Some(loop_var_ty.clone()),
            );
            let incremented = self.builder.fresh_value();
            self.builder.push_instr(
                IrInstr::BinOp { result: incremented, op: BinOp::Add, lhs: cur_loop_var, rhs: one, ty: loop_var_ty.clone() },
                Some(loop_var_ty.clone()),
            );
            self.scope.insert(loop_var_names[0].clone(), (incremented, loop_var_ty));

            let mut args = Vec::with_capacity(loop_var_names.len());
            for name in &loop_var_names {
                let Some((value, _)) = self.scope.get(name) else {
                    return Err(LowerError::Unsupported {
                        detail: format!("missing loop variable '{}' in continue", name),
                        span,
                    });
                };
                args.push(*value);
            }

            // Restore scope so subsequent code isn't affected.
            self.scope.clear();
            for (k, v) in saved_scope {
                self.scope.insert(k, v);
            }

            self.builder.push_instr(IrInstr::Br { target: header_bb, args }, None);
        } else {
            // While/loop: just pass current values of loop vars to header params.
            let mut args = Vec::with_capacity(loop_var_names.len());
            for name in &loop_var_names {
                let Some((value, _)) = self.scope.get(name) else {
                    return Err(LowerError::Unsupported {
                        detail: format!("missing loop variable '{}' in continue", name),
                        span,
                    });
                };
                args.push(*value);
            }
            self.builder.push_instr(IrInstr::Br { target: header_bb, args }, None);
        }
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
        // Record source position for the debugger span table.
        let span_byte = match stmt {
            AstStmt::Let { span, .. } => Some(span.start.0),
            AstStmt::Expr(expr) => Some(expr.span().start.0),
            AstStmt::While { span, .. } => Some(span.start.0),
            AstStmt::Loop { span, .. } => Some(span.start.0),
            AstStmt::Break { span, .. } => Some(span.start.0),
            AstStmt::Continue { span, .. } => Some(span.start.0),
            AstStmt::ForRange { span, .. } => Some(span.start.0),
            AstStmt::Assign { span, .. } => Some(span.start.0),
            AstStmt::LetTuple { span, .. } => Some(span.start.0),
            AstStmt::Return { span, .. } => Some(span.start.0),
            AstStmt::Spawn { span, .. } => Some(span.start.0),
            AstStmt::ParFor { span, .. } => Some(span.start.0),
            AstStmt::ForEach { span, .. } => Some(span.start.0),
            AstStmt::MaskStmt { span, .. } => Some(span.start.0),
            AstStmt::HandleStmt { span, .. } => Some(span.start.0),
            AstStmt::Defer { span, .. } => Some(span.start.0),
            AstStmt::Select { span, .. } => Some(span.start.0),
            AstStmt::Yield { span, .. } => Some(span.start.0),
        };
        if let Some(byte) = span_byte {
            self.builder.set_span_byte(byte);
        }
        match stmt {
            AstStmt::Let {
                name,
                ty: ann_ty,
                init,
                ..
            } => {
                // Set binding_ty from the annotation so constructors like list() can
                // infer their element type (e.g. `val xs: list<f64> = list()`).
                if let Some(ast_ty) = ann_ty {
                    self.binding_ty = Some(self.resolve_ty(ast_ty));
                    // Set expected_expr_ty so lower_if_expr can coerce branches
                    // before the merge (avoids branch-type mismatch).
                    self.expected_expr_ty = Some(self.resolve_ty(ast_ty));
                }
                let (mut val, mut ty) = self.lower_expr(init)?;
                self.binding_ty = None;
                self.expected_expr_ty = None;

                // Cohersion: if the binding is annotated `dyn Trait` and the
                // initializer has a concrete struct type, materialize a
                // trait-object fat pointer via MakeTraitObject.
                if let Some(ast_ty_for_box) = ann_ty {
                    let ann = self.resolve_ty(ast_ty_for_box);
                    let (nv, nt) = self.coerce_to_trait_object(
                        val,
                        ty.clone(),
                        &ann,
                        ast_ty_for_box.span(),
                    )?;
                    val = nv;
                    ty = nt;
                }

                self.scope.insert(name.name.clone(), (val, ty));
                Ok(())
            }
            AstStmt::LetTuple { names, init, span, .. } => {
                let (tuple_val, tuple_ty) = self.lower_expr(init)?;
                let elem_types = match &tuple_ty {
                    IrType::Tuple(elems) => elems.clone(),
                    IrType::Struct { fields, .. } => {
                        // Struct destructuring by position: (a, b) = struct_val
                        if names.len() != fields.len() {
                            return Err(LowerError::Unsupported {
                                detail: format!(
                                    "struct has {} fields but destructuring binds {}",
                                    fields.len(),
                                    names.len()
                                ),
                                span: *span,
                            });
                        }
                        for (i, name) in names.iter().enumerate() {
                            let (_, elem_ty) = &fields[i];
                            let elem_ty = elem_ty.clone();
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::GetField {
                                    result,
                                    base: tuple_val,
                                    field_index: i,
                                    result_ty: elem_ty.clone(),
                                },
                                Some(elem_ty.clone()),
                            );
                            self.scope.insert(name.name.clone(), (result, elem_ty));
                        }
                        return Ok(());
                    }
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
                // If the expression is a block, lower it as a statement block (allow no tail).
                // Otherwise, lower it as a normal expression.
                if let AstExpr::Block(block) = expr.as_ref() {
                    self.lower_block(block)?;
                } else {
                    self.lower_expr(expr)?;
                }
                Ok(())
            }
            AstStmt::While { cond, body, span, label } => self.lower_while(cond, body, label, *span),
            AstStmt::ForRange {
                var,
                start,
                end,
                body,
                inclusive,
                step,
                label,
                span,
            } => self.lower_for_range(var, start, end, body, *inclusive, step.as_deref(), label, *span),
            AstStmt::Loop { body, span, label } => self.lower_loop(body, label, *span),
            AstStmt::Break { label, span, .. } => self.lower_break(label, *span),
            AstStmt::Continue { label, span, .. } => self.lower_continue(label, *span),
            AstStmt::Assign {
                target,
                op,
                value,
                span,
            } => {
                match target.as_ref() {
                    // Plain identifier assignment: rebind the name in scope (SSA-style).
                    AstExpr::Ident(ident) => {
                        let new_val = if let Some(bin_op) = op {
                            let (lhs_val, lhs_ty) = self.scope.get(&ident.name).cloned()
                                .ok_or_else(|| LowerError::Unsupported {
                                    detail: format!("variable '{}' not in scope for compound assignment", ident.name),
                                    span: *span,
                                })?;
                            let (rhs_val, _) = self.lower_expr(value)?;
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::BinOp {
                                    result,
                                    op: bin_op.clone(),
                                    lhs: lhs_val,
                                    rhs: rhs_val,
                                    ty: lhs_ty.clone(),
                                },
                                Some(lhs_ty.clone()),
                            );
                            (result, lhs_ty)
                        } else {
                            self.lower_expr(value)?
                        };
                        self.scope.insert(ident.name.clone(), (new_val.0, new_val.1));
                        Ok(())
                    }
                    // Array element store: `arr[i] = value`  or  tensor store
                    AstExpr::Index {
                        base,
                        indices,
                        span,
                    } => {
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
                    AstExpr::FieldAccess { base, field, span } => {
                        let (value_val, _) = self.lower_expr(value)?;
                        self.lower_field_assignment(base, field, value_val, *span)
                    }
                    AstExpr::TupleIndex { base, index, span } => {
                        if let AstExpr::Ident(base_ident) = base.as_ref() {
                            let (base_val, base_ty) = self.lower_expr(base)?;
                            let elem_types = match &base_ty {
                                IrType::Tuple(elems) => elems.clone(),
                                _ => {
                                    return Err(LowerError::Unsupported {
                                        detail: format!("tuple index assignment on non-tuple type {}", base_ty),
                                        span: *span,
                                    });
                                }
                            };
                            if *index >= elem_types.len() {
                                return Err(LowerError::Unsupported {
                                    detail: format!("tuple index {} out of bounds for {} elements", index, elem_types.len()),
                                    span: *span,
                                });
                            }
                            let (value_val, _) = self.lower_expr(value)?;

                            let mut new_elements = Vec::with_capacity(elem_types.len());
                            for (i, elem_ty) in elem_types.iter().enumerate() {
                                if i == *index {
                                    new_elements.push(value_val);
                                } else {
                                    let f_ty = elem_ty.clone();
                                    let f_val = self.builder.fresh_value();
                                    self.builder.push_instr(
                                        IrInstr::GetElement {
                                            result: f_val,
                                            base: base_val,
                                            index: i,
                                            result_ty: f_ty.clone(),
                                        },
                                        Some(f_ty.clone()),
                                    );
                                    new_elements.push(f_val);
                                }
                            }
                            let result_ty = base_ty.clone();
                            let result = self.builder.fresh_value();
                            self.builder.push_instr(
                                IrInstr::MakeTuple {
                                    result,
                                    elements: new_elements,
                                    result_ty: result_ty.clone(),
                                },
                                Some(result_ty.clone()),
                            );
                            self.scope.insert(base_ident.name.clone(), (result, result_ty));
                            Ok(())
                        } else {
                            Err(LowerError::Unsupported {
                                detail: "tuple index assignment target base must be an identifier".into(),
                                span: *span,
                            })
                        }
                    }
                    _ => Err(LowerError::Unsupported {
                        detail: "assignment target must be an identifier, tensor index, struct field, or tuple index".into(),
                        span: *span,
                    }),
                }
            }
            AstStmt::Return { value, span, .. } => {
                let ret_values = if let Some(expr) = value {
                    // Seed expected_expr_ty so that if/else tail expressions in
                    // the return value get coerced before branch merge.
                    if self.expected_expr_ty.is_none() {
                        self.expected_expr_ty = self.current_return_ty.clone();
                    }
                    let (val, ty) = self.lower_expr(expr)?;
                    self.expected_expr_ty = None;
                    // If the function returns `dyn Trait` and we're returning a
                    // concrete struct, coerce it via MakeTraitObject.
                    let (val, _ty) = if let Some(expected) = self.current_return_ty.clone() {
                        self.coerce_to_trait_object(val, ty, &expected, *span)?
                    } else {
                        (val, ty)
                    };
                    vec![val]
                } else {
                    vec![]
                };
                // Emit deferred expressions in reverse order (LIFO) before return.
                let defers: Vec<_> = self.defer_stack.clone();
                for expr in defers.iter().rev() {
                    let _ = self.lower_expr(expr);
                }
                self.builder
                    .push_instr(IrInstr::Return { values: ret_values }, None);
                // Create a new unreachable block so any subsequent instructions
                // (from following statements) don't pollute the terminated block.
                let unreachable_bb = self.builder.create_block(Some("post_return"));
                self.builder.set_current_block(unreachable_bb);
                Ok(())
            }

            AstStmt::Spawn { body, span, group } => {
                // Lambda-lift the spawn body into a function __spawn_N().
                let counter = self.lambda_counter.get();
                self.lambda_counter.set(counter + 1);
                let fn_name = format!("__spawn_{}", counter);

                // Collect captures (all in-scope variables), sorted by name so
                // the spawned body's parameter list does not depend on
                // `HashMap` iteration order. See the note in `lower_lambda`.
                let mut captures: Vec<(String, ValueId, IrType)> = self
                    .scope
                    .iter()
                    .map(|(name, (vid, ty))| (name.clone(), *vid, ty.clone()))
                    .collect();
                captures.sort_by(|a, b| a.0.cmp(&b.0));
                let captures = captures;

                let lifted_params: Vec<crate::ir::function::Param> = captures
                    .iter()
                    .map(|(name, _, ty)| crate::ir::function::Param {
                        name: name.clone(),
                        ty: ty.clone(),
                    })
                    .collect();

                // Build the lifted function with a synthetic AstBlock.
                let ast_block = AstBlock {
                    stmts: body.clone(),
                    tail: None,
                    span: *span,
                };
                let temp_builder = IrFunctionBuilder::new(
                    &fn_name,
                    lifted_params.clone(),
                    IrType::Scalar(DType::I64),
                );
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
                    let inner_val =
                        spawn_lowerer
                            .builder
                            .add_block_param(entry, Some(name), ty.clone());
                    spawn_lowerer
                        .scope
                        .insert(name.clone(), (inner_val, ty.clone()));
                    capture_val_map.push((*outer_val, inner_val));
                }
                // Pre-populate spawn_lowerer's chan_elem_types from parent (inner val → elem ty).
                for (outer_val, inner_val) in &capture_val_map {
                    if let Some(elem_ty) = self.chan_elem_types.get(outer_val) {
                        spawn_lowerer
                            .chan_elem_types
                            .insert(*inner_val, elem_ty.clone());
                    }
                }
                spawn_lowerer.lower_block(&ast_block)?;
                // Propagate any new chan_elem_types discovered in spawn back to parent.
                for (outer_val, inner_val) in &capture_val_map {
                    if let Some(elem_ty) = spawn_lowerer.chan_elem_types.get(inner_val) {
                        self.chan_elem_types
                            .entry(*outer_val)
                            .or_insert_with(|| elem_ty.clone());
                    }
                }
                // Emit a return of 0 if not already terminated.
                let dummy_ret = spawn_lowerer.builder.fresh_value();
                spawn_lowerer.builder.push_instr(
                    IrInstr::ConstInt {
                        result: dummy_ret,
                        value: 0,
                        ty: IrType::Scalar(DType::I64),
                    },
                    Some(IrType::Scalar(DType::I64)),
                );
                spawn_lowerer.builder.push_instr(
                    IrInstr::Return {
                        values: vec![dummy_ret],
                    },
                    None,
                );
                spawn_lowerer.builder.seal_unterminated_blocks();
                let ir_func = spawn_lowerer.builder.build();
                self.lifted_fns.borrow_mut().push(ir_func);

                let capture_vals: Vec<ValueId> = captures.iter().map(|(_, v, _)| *v).collect();
                if let Some(group_expr) = group {
                    let (group_val, _) = self.lower_expr(group_expr)?;
                    self.builder.push_instr(
                        IrInstr::TaskGroupSpawn {
                            group: group_val,
                            body_fn: fn_name,
                            args: capture_vals,
                        },
                        None,
                    );
                } else {
                    self.builder.push_instr(
                        IrInstr::Spawn {
                            body_fn: fn_name,
                            args: capture_vals,
                        },
                        None,
                    );
                }
                let _ = span;
                Ok(())
            }

            AstStmt::ForEach {
                var,
                iter,
                body,
                span,
                ..
            } => self.lower_foreach(var, iter, body, *span),

            AstStmt::ParFor {
                var,
                start,
                end,
                body,
                inclusive,
                span,
                ..
            } => {
                // Lambda-lift body into __par_body_N(var: i64, captures...) { body }.
                let counter = self.lambda_counter.get();
                self.lambda_counter.set(counter + 1);
                let fn_name = format!("__par_body_{}", counter);

                // Collect outer-scope captures (all in-scope variables except
                // the loop var), sorted by name so the generated
                // `__par_body_N` signature does not depend on `HashMap`
                // iteration order. See the note in `lower_lambda`.
                let mut captures: Vec<(String, ValueId, IrType)> = self
                    .scope
                    .iter()
                    .filter(|(name, _)| *name != &var.name)
                    .map(|(name, (vid, ty))| (name.clone(), *vid, ty.clone()))
                    .collect();
                captures.sort_by(|a, b| a.0.cmp(&b.0));
                let captures = captures;

                // Reject mutation of a captured collection from the loop body.
                //
                // `par for` runs its body on several OS threads. A captured
                // `list` or `map` mutated from inside is a data race: the
                // runtime now locks the collection primitives, so it will not
                // corrupt memory, but the *result* is still order-dependent and
                // the program is not deterministic. Locking makes it survivable;
                // it does not make it correct.
                //
                // A language whose selling point is a statically verifiable
                // autonomy layer should not admit a race through the front door,
                // and the captures are already enumerated here, so the check
                // costs nothing.
                //
                // Limit, stated plainly: this catches a mutating builtin applied
                // directly to a captured name. Mutation reached through a
                // user-defined function is not detected — that needs
                // interprocedural analysis. False negatives, never false
                // positives.
                {
                    let captured_names: std::collections::HashSet<&str> =
                        captures.iter().map(|(n, _, _)| n.as_str()).collect();
                    let mut offender: Option<(String, String)> = None;
                    find_captured_mutation(body, &captured_names, &mut offender);
                    if let Some((coll, op)) = offender {
                        return Err(LowerError::Rejected {
                            detail: format!("`par for` body mutates the captured collection `{}` via `{}`.\nIterations run concurrently, so the result depends on thread scheduling.\nUse `atomic` for a shared counter, or build a per-iteration value and combine after the loop.", coll, op),
                            span: *span,
                        });
                    }
                }

                // Build params: loop var first, then captures.
                let mut params = vec![crate::ir::function::Param {
                    name: var.name.clone(),
                    ty: IrType::Scalar(DType::I64),
                }];
                for (name, _, ty) in &captures {
                    params.push(crate::ir::function::Param {
                        name: name.clone(),
                        ty: ty.clone(),
                    });
                }

                let temp_builder =
                    IrFunctionBuilder::new(&fn_name, params, IrType::Scalar(DType::I64));
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
                let var_val = body_lowerer.builder.add_block_param(
                    entry,
                    Some(&var.name),
                    IrType::Scalar(DType::I64),
                );
                body_lowerer
                    .scope
                    .insert(var.name.clone(), (var_val, IrType::Scalar(DType::I64)));
                // Add capture params.
                for (name, _, ty) in &captures {
                    let inner_val =
                        body_lowerer
                            .builder
                            .add_block_param(entry, Some(name), ty.clone());
                    body_lowerer
                        .scope
                        .insert(name.clone(), (inner_val, ty.clone()));
                }
                body_lowerer.lower_block(body)?;
                let dummy_ret = body_lowerer.builder.fresh_value();
                body_lowerer.builder.push_instr(
                    IrInstr::ConstInt {
                        result: dummy_ret,
                        value: 0,
                        ty: IrType::Scalar(DType::I64),
                    },
                    Some(IrType::Scalar(DType::I64)),
                );
                body_lowerer.builder.push_instr(
                    IrInstr::Return {
                        values: vec![dummy_ret],
                    },
                    None,
                );
                body_lowerer.builder.seal_unterminated_blocks();
                let ir_func = body_lowerer.builder.build();
                self.lifted_fns.borrow_mut().push(ir_func);

                let (start_val, _) = self.lower_expr(start)?;
                let (end_val, _) = self.lower_expr(end)?;
                let var_id = self.builder.fresh_value();
                let capture_vals: Vec<ValueId> = captures.iter().map(|(_, v, _)| *v).collect();
                self.builder.push_instr(
                    IrInstr::ParFor {
                        var: var_id,
                        start: start_val,
                        end: end_val,
                        inclusive: *inclusive,
                        body_fn: fn_name,
                        args: capture_vals,
                    },
                    None,
                );
                let _ = span;
                Ok(())
            }
            AstStmt::MaskStmt { body, .. } => {
                self.lower_block(body)?;
                Ok(())
            }
            AstStmt::HandleStmt {
                expr, arms, return_ty, ..
            } => {
                // Lower each handler arm into an IR function + HandlerArm descriptor.
                let mut handler_arms = Vec::new();
                for arm in arms {
                    let ha = self.lower_handler_arm(arm)?;
                    handler_arms.push(ha);
                }

                // Emit PushHandler.
                self.builder.push_instr(
                    IrInstr::PushHandler {
                        arms: handler_arms,
                    },
                    None,
                );

                // Lower body.
                self.lower_expr(expr)?;

                // Emit PopHandler.
                self.builder.push_instr(IrInstr::PopHandler, None);

                let _ = return_ty;
                Ok(())
            }
            AstStmt::Defer { expr, .. } => {
                self.defer_stack.push(expr.as_ref().clone());
                Ok(())
            }
            AstStmt::Yield { expr, .. } => {
                self.lower_expr(&*expr)?;
                Ok(())
            }
            AstStmt::Select { arms, default, .. } => {
                let outer_scope = self.scope.clone();

                // Collect rebound variable names from all arm bodies and default body.
                let mut rebound_names: Vec<String> = Vec::new();
                for arm in arms {
                    collect_rebound_vars_in_block(&arm.body, &mut rebound_names, false);
                }
                if let Some(d) = default {
                    collect_rebound_vars_in_block(d, &mut rebound_names, false);
                }
                rebound_names.retain(|name| outer_scope.contains_key(name));

                let loop_bb = self.builder.create_block(Some("select_loop"));
                let merge_bb = self.builder.create_block(Some("select_merge"));

                // Add block params to merge_bb for rebound variables (phi merge).
                let mut rebound_params: Vec<(String, ValueId, IrType)> = Vec::new();
                for name in &rebound_names {
                    let Some((_, ty)) = outer_scope.get(name) else { continue };
                    let param = self.builder.add_block_param(merge_bb, Some(name), ty.clone());
                    rebound_params.push((name.clone(), param, ty.clone()));
                }

                // Branch to the loop
                self.builder.push_instr(
                    IrInstr::Br { target: loop_bb, args: vec![] },
                    None,
                );

                // -- loop_bb: call select(ch0, ch1, ...)
                self.builder.set_current_block(loop_bb);
                let channels: Vec<ValueId> = arms.iter().map(|arm| {
                    let (v, _) = self.lower_expr(&arm.channel)?;
                    Ok(v)
                }).collect::<Result<Vec<_>, LowerError>>()?;

                // Emit select(ch0, ch1, ...) builtin call
                let select_result = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::BuiltinCall {
                        result: select_result,
                        name: "select".to_string(),
                        args: channels.clone(),
                        result_ty: IrType::Scalar(DType::I64),
                    },
                    Some(IrType::Scalar(DType::I64)),
                );

                // For each arm: check if idx == i, if so recv and execute
                let next_check_bb = self.builder.create_block(Some("select_no_match"));
                for (i, arm) in arms.iter().enumerate() {
                    let idx_val = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::ConstInt {
                            result: idx_val,
                            value: i as i64,
                            ty: IrType::Scalar(DType::I64),
                        },
                        Some(IrType::Scalar(DType::I64)),
                    );
                    let cmp_val = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::BinOp {
                            result: cmp_val,
                            op: BinOp::CmpEq,
                            lhs: select_result,
                            rhs: idx_val,
                            ty: IrType::Scalar(DType::Bool),
                        },
                        Some(IrType::Scalar(DType::Bool)),
                    );
                    let arm_body_bb = self.builder.create_block(Some(&format!("select_arm_{}", i)));
                    let fall_bb = if i < arms.len() - 1 {
                        self.builder.create_block(Some(&format!("select_check_{}", i + 1)))
                    } else {
                        next_check_bb
                    };
                    self.builder.push_instr(
                        IrInstr::CondBr {
                            cond: cmp_val,
                            then_block: arm_body_bb,
                            else_block: fall_bb,
                            then_args: vec![],
                            else_args: vec![],
                        },
                        None,
                    );

                    // arm_body_bb: recv from channel and execute body
                    self.builder.set_current_block(arm_body_bb);
                    self.scope = outer_scope.clone();
                    let recv_result = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::ChanRecv {
                            result: recv_result,
                            chan: channels[i],
                            elem_ty: IrType::Infer,
                        },
                        Some(IrType::Infer),
                    );
                    // Bind the received value
                    self.scope.insert(arm.binding.clone(), (recv_result, IrType::Infer));
                    for stmt in &arm.body.stmts {
                        self.lower_stmt(stmt)?;
                    }
                    if let Some(tail) = &arm.body.tail {
                        self.lower_expr(tail)?;
                    }
                    if !self.builder.is_current_block_terminated() {
                        let mut merge_args: Vec<ValueId> = Vec::new();
                        for name in &rebound_names {
                            let val = self.scope.get(name)
                                .or_else(|| outer_scope.get(name))
                                .map(|(v, _)| *v)
                                .expect("rebound variable missing from select arm scope");
                            merge_args.push(val);
                        }
                        self.builder.push_instr(
                            IrInstr::Br { target: merge_bb, args: merge_args },
                            None,
                        );
                    }

                    // Set next check block
                    if i < arms.len() - 1 {
                        self.builder.set_current_block(fall_bb);
                    }
                }

                // next_check_bb: no arm matched — check default
                self.builder.set_current_block(next_check_bb);
                self.scope = outer_scope.clone();
                if let Some(default_block) = default {
                    for stmt in &default_block.stmts {
                        self.lower_stmt(stmt)?;
                    }
                    if let Some(tail) = &default_block.tail {
                        self.lower_expr(tail)?;
                    }
                    if !self.builder.is_current_block_terminated() {
                        let mut merge_args: Vec<ValueId> = Vec::new();
                        for name in &rebound_names {
                            let val = self.scope.get(name)
                                .or_else(|| outer_scope.get(name))
                                .map(|(v, _)| *v)
                                .expect("rebound variable missing from select default scope");
                            merge_args.push(val);
                        }
                        self.builder.push_instr(
                            IrInstr::Br { target: merge_bb, args: merge_args },
                            None,
                        );
                    }
                } else {
                    // No default: sleep(1) and loop back
                    let ms_val = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::ConstInt {
                            result: ms_val,
                            value: 1,
                            ty: IrType::Scalar(DType::I64),
                        },
                        Some(IrType::Scalar(DType::I64)),
                    );
                    let sleep_result = self.builder.fresh_value();
                    self.builder.push_instr(
                        IrInstr::SleepMs { result: sleep_result, ms: ms_val },
                        Some(IrType::Scalar(DType::I64)),
                    );
                    self.builder.push_instr(
                        IrInstr::Br { target: loop_bb, args: vec![] },
                        None,
                    );
                }

                // merge_bb: insert block params into scope for rebound variables
                self.scope = outer_scope;
                for (name, param, ty) in rebound_params {
                    self.scope.insert(name, (param, ty));
                }
                self.builder.set_current_block(merge_bb);
                Ok(())
            }
        }
    }

    fn lower_field_assignment(
        &mut self,
        base: &AstExpr,
        field: &str,
        value_val: ValueId,
        span: Span,
    ) -> Result<(), LowerError> {
        let (base_val, base_ty) = self.lower_expr(base)?;
        let struct_fields = match &base_ty {
            IrType::Struct { fields, .. } => fields.clone(),
            _ => {
                return Err(LowerError::Unsupported {
                    detail: format!("field assignment on non-struct type {}", base_ty),
                    span,
                });
            }
        };
        let field_index = struct_fields
            .iter()
            .position(|(n, _)| n == field)
            .ok_or_else(|| LowerError::Unsupported {
                detail: format!("no field '{}' in struct", field),
                span,
            })?;

        let mut new_fields = Vec::with_capacity(struct_fields.len());
        for (i, field_info) in struct_fields.iter().enumerate() {
            if i == field_index {
                new_fields.push(value_val);
            } else {
                let f_ty = field_info.1.clone();
                let f_val = self.builder.fresh_value();
                self.builder.push_instr(
                    IrInstr::GetField {
                        result: f_val,
                        base: base_val,
                        field_index: i,
                        result_ty: f_ty.clone(),
                    },
                    Some(f_ty.clone()),
                );
                new_fields.push(f_val);
            }
        }
        let result_ty = base_ty.clone();
        let result = self.builder.fresh_value();
        self.builder.push_instr(
            IrInstr::MakeStruct {
                result,
                fields: new_fields,
                result_ty: result_ty.clone(),
            },
            Some(result_ty.clone()),
        );

        match base {
            AstExpr::Ident(base_ident) => {
                self.scope.insert(base_ident.name.clone(), (result, result_ty));
                Ok(())
            }
            AstExpr::FieldAccess { base: parent_base, field: parent_field, span: parent_span } => {
                self.lower_field_assignment(parent_base, parent_field, result, *parent_span)
            }
            _ => {
                Err(LowerError::Unsupported {
                    detail: "field assignment target base must be an identifier or field access".into(),
                    span,
                })
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
    fn_param_names: std::rc::Rc<HashMap<String, Vec<String>>>,
    fn_param_types: std::rc::Rc<HashMap<String, Vec<IrType>>>,
    lambda_counter: std::rc::Rc<std::cell::Cell<u32>>,
) -> Result<
    (
        crate::ir::function::IrFunction,
        Vec<crate::ir::function::IrFunction>,
    ),
    LowerError,
> {
    lower_function_with_generics_and_subs(
        func,
        module,
        fn_sigs,
        const_defs,
        generic_fns,
        mono_cache,
        mono_sigs,
        HashMap::new(), // no type param subs for top-level functions
        trait_dispatch,
        fn_defaults,
        fn_param_names,
        fn_param_types,
        lambda_counter,
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
    fn_param_names: std::rc::Rc<HashMap<String, Vec<String>>>,
    fn_param_types: std::rc::Rc<HashMap<String, Vec<IrType>>>,
    lambda_counter: std::rc::Rc<std::cell::Cell<u32>>,
) -> Result<
    (
        crate::ir::function::IrFunction,
        Vec<crate::ir::function::IrFunction>,
    ),
    LowerError,
> {
    let resolve = |ty: &AstType| -> IrType {
        resolve_ast_type_with_subs(ty, &type_param_subs, module)
    };

    let return_ty = resolve(&func.return_ty);
    let params: Vec<Param> = func
        .params
        .iter()
        .map(|p| Param {
            name: p.name.name.clone(),
            ty: resolve(&p.ty),
        })
        .collect();

    if func.is_async {
        let chan_ty = IrType::Chan(Box::new(return_ty.clone()));
        let inner_name = format!("__async_inner_{}", func.name.name);
        let spawn_name = format!("__async_spawn_{}", func.name.name);

        // Build the inner (sync) function that computes the result.
        let mut inner_func = func.clone();
        inner_func.name.name = inner_name.clone();
        inner_func.is_async = false;
        let (mut inner_ir, mut inner_lifted) = lower_function_with_generics_and_subs(
            &inner_func,
            module,
            fn_sigs,
            const_defs,
            generic_fns.clone(),
            mono_cache.clone(),
            mono_sigs.clone(),
            type_param_subs.clone(),
            trait_dispatch.clone(),
            fn_defaults.clone(),
            fn_param_names.clone(),
            fn_param_types.clone(),
            lambda_counter.clone(),
        )?;
        // Preserve attributes on the inner implementation only.
        inner_ir.attrs = func.attrs.iter().map(|a| ast_attr_to_ir_attr(a)).collect();

        // Build the spawn body function: call inner, send on channel, return 0.
        let mut spawn_params: Vec<Param> = Vec::with_capacity(params.len() + 1);
        spawn_params.push(Param {
            name: "__ch".to_owned(),
            ty: chan_ty.clone(),
        });
        spawn_params.extend(params.iter().cloned());
        let mut spawn_builder = IrFunctionBuilder::new(
            &spawn_name,
            spawn_params.clone(),
            IrType::Scalar(DType::I64),
        );
        let spawn_entry = spawn_builder.create_block(Some("entry"));
        spawn_builder.set_current_block(spawn_entry);
        let chan_val = spawn_builder.add_block_param(spawn_entry, Some("__ch"), chan_ty.clone());
        let mut arg_vals: Vec<ValueId> = Vec::with_capacity(params.len());
        for p in &params {
            let v = spawn_builder.add_block_param(spawn_entry, Some(&p.name), p.ty.clone());
            arg_vals.push(v);
        }
        let call_result = spawn_builder.fresh_value();
        spawn_builder.push_instr(
            IrInstr::Call {
                result: Some(call_result),
                callee: inner_name.clone(),
                args: arg_vals,
                result_ty: Some(return_ty.clone()),
            },
            Some(return_ty.clone()),
        );
        spawn_builder.push_instr(
            IrInstr::ChanSend {
                chan: chan_val,
                value: call_result,
            },
            None,
        );
        let dummy = spawn_builder.fresh_value();
        let dummy_ty = IrType::Scalar(DType::I64);
        spawn_builder.push_instr(
            IrInstr::ConstInt {
                result: dummy,
                value: 0,
                ty: dummy_ty.clone(),
            },
            Some(dummy_ty.clone()),
        );
        spawn_builder.push_instr(
            IrInstr::Return {
                values: vec![dummy],
            },
            None,
        );
        spawn_builder.seal_unterminated_blocks();
        let spawn_ir = spawn_builder.build();

        // Build the async wrapper: create channel, spawn worker, return channel.
        let mut builder = IrFunctionBuilder::new(&func.name.name, params.clone(), chan_ty.clone());
        let entry = builder.create_block(Some("entry"));
        builder.set_current_block(entry);
        let mut wrapper_args: Vec<ValueId> = Vec::with_capacity(params.len());
        for p in &params {
            let v = builder.add_block_param(entry, Some(&p.name), p.ty.clone());
            wrapper_args.push(v);
        }
        let dummy_cap = builder.fresh_value();
        builder.push_instr(
            IrInstr::ConstInt {
                result: dummy_cap,
                value: -1,
                ty: IrType::Scalar(DType::I64),
            },
            Some(IrType::Scalar(DType::I64)),
        );
        let chan_val = builder.fresh_value();
        builder.push_instr(
            IrInstr::ChanNew {
                result: chan_val,
                elem_ty: return_ty.clone(),
                capacity: dummy_cap,
            },
            Some(chan_ty.clone()),
        );
        let mut spawn_args = Vec::with_capacity(wrapper_args.len() + 1);
        spawn_args.push(chan_val);
        spawn_args.extend(wrapper_args);
        builder.push_instr(
            IrInstr::Spawn {
                body_fn: spawn_name.clone(),
                args: spawn_args,
            },
            None,
        );
        builder.push_instr(
            IrInstr::Return {
                values: vec![chan_val],
            },
            None,
        );
        builder.seal_unterminated_blocks();
        let mut wrapper_ir = builder.build();
        wrapper_ir.attrs = Vec::new();

        inner_lifted.push(inner_ir);
        inner_lifted.push(spawn_ir);
        return Ok((wrapper_ir, inner_lifted));
    }

    let mut builder = IrFunctionBuilder::new(&func.name.name, params.clone(), return_ty.clone());
    let entry = builder.create_block(Some("entry"));
    builder.set_current_block(entry);

    let lifted_fns = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let mut lowerer = Lowerer::new_generic(
        builder,
        module,
        fn_sigs,
        lambda_counter,
        lifted_fns.clone(),
        type_param_subs.clone(),
        generic_fns,
        mono_cache,
        mono_sigs,
        const_defs.clone(),
        trait_dispatch,
        fn_defaults,
        fn_param_names,
        fn_param_types,
    );
    lowerer.current_return_ty = Some(return_ty.clone());

    // Pre-register monomorphized struct defs for Generic types used in this function.
    // Walk AST param types + return type, find Generic types, resolve with subs,
    // and register the concrete struct def so field access works.
    {
        let mut local_defs = std::collections::HashMap::new();
        // Helper: walk an AST type and register monomorphized struct defs.
        fn register_generic_structs(
            ast_ty: &crate::parser::ast::AstType,
            subs: &HashMap<String, IrType>,
            module: &IrModule,
            local_defs: &mut HashMap<String, Vec<(String, IrType)>>,
        ) {
            if let crate::parser::ast::AstType::Generic { name, args, .. } = ast_ty {
                let resolved_args: Vec<IrType> = args
                    .iter()
                    .map(|a| resolve_ast_type_with_subs(a, subs, module))
                    .collect();
                let mangled = format!("{}__{}", name, resolved_args.iter().map(mangle_ir_type).collect::<Vec<_>>().join("_"));
                let resolved = resolve_brought_name(&mangled, module);
                if module.struct_def(&resolved).is_none() && !local_defs.contains_key(&resolved) {
                    // Compute concrete fields from the template.
                    if let Some(template_fields) = module.struct_def(&resolve_brought_name(name, module)) {
                        let concrete_fields: Vec<(String, IrType)> = template_fields
                            .iter()
                            .map(|(fn_, ft)| {
                                let concrete_ft = resolve_concrete_field(ft, subs, module);
                                (fn_.clone(), concrete_ft)
                            })
                            .collect();
                        local_defs.insert(resolved, concrete_fields);
                    }
                }
            }
        }
        for p in &func.params {
            register_generic_structs(&p.ty, &type_param_subs, module, &mut local_defs);
        }
        register_generic_structs(&func.return_ty, &type_param_subs, module, &mut local_defs);
        lowerer.local_struct_defs = local_defs;
        // Also carry over defaults for any monomorphized structs registered above.
        let mut local_defaults: HashMap<String, Vec<Option<crate::parser::ast::AstExpr>>> = HashMap::new();
        for mangled_name in lowerer.local_struct_defs.keys() {
            if module.struct_defaults.get(mangled_name).is_none() {
                // Find the base name by matching known base names with __ suffix.
                for (base_name, base_defaults) in &module.struct_defaults {
                    if mangled_name == base_name || mangled_name.starts_with(&format!("{}__", base_name)) {
                        local_defaults.insert(mangled_name.clone(), base_defaults.clone());
                        break;
                    }
                }
            }
        }
        lowerer.local_struct_defaults = local_defaults;
    }

    // Register function parameters as entry block params.
    for (param, ir_param) in func.params.iter().zip(params.iter()) {
        let val =
            lowerer
                .builder
                .add_block_param(entry, Some(&param.name.name), ir_param.ty.clone());
        lowerer
            .scope
            .insert(param.name.name.clone(), (val, ir_param.ty.clone()));
    }

    // Inject global constants into scope.
    for (name, expr) in lowerer.const_defs.clone().iter() {
        let (val, ty) = lowerer.lower_expr(expr)?;
        lowerer.scope.insert(name.clone(), (val, ty));
    }

    let tail_val = lowerer.lower_block(&func.body)?;

    if !lowerer.builder.is_current_block_terminated() {
        // Emit deferred expressions in reverse order (LIFO) before implicit return.
        let defers: Vec<_> = lowerer.defer_stack.clone();
        for expr in defers.iter().rev() {
            let _ = lowerer.lower_expr(expr);
        }
        let ret_values: Vec<ValueId> = match tail_val {
            Some((v, _)) => vec![v],
            None => vec![],
        };
        lowerer
            .builder
            .push_instr(IrInstr::Return { values: ret_values }, None);
    }

    lowerer.builder.seal_unterminated_blocks();

    let mut ir_func = lowerer.builder.build();
    // Propagate AST function attributes (e.g., "kernel", "differentiable") to IR.
    ir_func.attrs = func.attrs.iter().map(|a| ast_attr_to_ir_attr(a)).collect();
    ir_func.is_const = func.is_const;
    let lifted = match std::rc::Rc::try_unwrap(lifted_fns) {
        Ok(cell) => cell.into_inner(),
        Err(rc) => rc.borrow().clone(),
    };
    Ok((ir_func, lifted))
}

/// Convert an AST attribute to an IR attribute (args omitted for simplicity).
fn ast_attr_to_ir_attr(attr: &crate::parser::ast::AstAttribute) -> crate::ir::function::IrAttribute {
    crate::ir::function::IrAttribute {
        name: attr.name.clone(),
        args: Vec::new(),
    }
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
        AstType::List(elem, _) => IrType::List(Box::new(lower_type(elem))),
        AstType::Map(k, v, _) => IrType::Map(Box::new(lower_type(k)), Box::new(lower_type(v))),
        AstType::Fn { params, ret, .. } => IrType::Fn {
            params: params.iter().map(lower_type).collect(),
            ret: Box::new(lower_type(ret)),
        },
        AstType::WeakRef(inner, _) => IrType::WeakRef(Box::new(lower_type(inner))),
        AstType::Generic { name, args, .. } => IrType::Struct {
            name: format!("{}__{}", name, args.iter().map(|a| format!("{:?}", a)).collect::<Vec<_>>().join("_")),
            fields: Vec::new(),
        },
        AstType::ConstInt(_, _) => IrType::Scalar(DType::I64),
        AstType::AssocType { .. } => IrType::Infer,
        // When lower_type (no module) sees `dyn Trait`, fall back to an empty
        // TraitObject; the full lower_type_with_structs path fills it in.
        AstType::DynTrait { trait_name, .. } => IrType::TraitObject {
            name: trait_name.clone(),
            methods: Vec::new(),
        },
        AstType::MaskEffectType { .. } => IrType::Infer,
        AstType::Ref(inner, _) => lower_type(inner),
        AstType::RefMut(inner, _) => lower_type(inner),
    }
}

/// Converts a type name string (as written in `impl Trait for TypeName`) to an `IrType`.
/// True when two types share a constructor and differ only where one side is
/// still unresolved -- `option<_>` against `option<i64>`.
///
/// Used only to pick a trait impl for a receiver whose element type inference
/// did not reach, and only when exactly one candidate matches.
fn same_constructor_with_infer(a: &IrType, b: &IrType) -> bool {
    // A struct with no fields is the lowerer's marker for an unsubstituted type
    // parameter -- `impl[T] Show for list<T>` lowers its target to
    // `list<%T>`. Treating it as a wildcard is what lets one generic impl serve
    // every element type instead of needing one instantiation each.
    //
    // A user's genuinely empty record would also match here. That is tolerable
    // because this helper is consulted only after an exact match has failed,
    // and only when exactly one candidate matches -- so it can widen a lookup
    // that would otherwise fail outright, never silently redirect one that
    // would have succeeded.
    fn is_param_marker(t: &IrType) -> bool {
        matches!(t, IrType::Struct { fields, .. } if fields.is_empty())
    }
    if is_param_marker(a) || is_param_marker(b) {
        return true;
    }
    match (a, b) {
        (IrType::Infer, _) | (_, IrType::Infer) => true,
        (IrType::List(x), IrType::List(y)) => same_constructor_with_infer(x, y),
        (IrType::Option(x), IrType::Option(y)) => same_constructor_with_infer(x, y),
        (IrType::Map(k1, v1), IrType::Map(k2, v2)) => {
            same_constructor_with_infer(k1, k2) && same_constructor_with_infer(v1, v2)
        }
        (IrType::Struct { name: n1, .. }, IrType::Struct { name: n2, .. }) => n1 == n2,
        _ => a == b,
    }
}

fn type_name_to_ir_type(name: &str, module: &IrModule) -> IrType {
    match name {
        "i64" => IrType::Scalar(DType::I64),
        "i32" => IrType::Scalar(DType::I32),
        "f64" => IrType::Scalar(DType::F64),
        "f32" => IrType::Scalar(DType::F32),
        "bool" => IrType::Scalar(DType::Bool),
        "str" => IrType::Str,
        _ => {
            if let Some(fields) = module.struct_def(name) {
                IrType::Struct {
                    name: name.to_owned(),
                    fields: fields.clone(),
                }
            } else if let Some(variants) = module.enum_def(name) {
                IrType::Enum {
                    name: name.to_owned(),
                    variants: variants.clone(),
                }
            } else {
                IrType::Infer
            }
        }
    }
}

/// Returns a short string key for `ty` used to look up trait dispatch entries.
fn ir_type_dispatch_name(ty: &IrType) -> String {
    match ty {
        IrType::Scalar(DType::I64) => "i64".to_owned(),
        IrType::Scalar(DType::I32) => "i32".to_owned(),
        IrType::Scalar(DType::F64) => "f64".to_owned(),
        IrType::Scalar(DType::F32) => "f32".to_owned(),
        IrType::Scalar(DType::Bool) => "bool".to_owned(),
        IrType::Str => "str".to_owned(),
        IrType::Struct { name, .. } => name.clone(),
        IrType::Enum { name, .. } => name.clone(),
        other => format!("{}", other),
    }
}

/// Returns an LLVM-IR-safe identifier fragment for `ty`. Used to build
/// monomorphised function names (`fn__T` -> `fn__MyStruct`) where the result
/// must be a valid C identifier. Strips the `%` prefix from struct Display
/// output and replaces whitespace/commas/brackets with `_`.
pub(crate) fn mangle_ir_type(ty: &IrType) -> String {
    let s = match ty {
        IrType::TapeRef => "taperef".to_owned(),
        IrType::Scalar(d) => format!("{}", d),
        IrType::Str => "str".to_owned(),
        IrType::Struct { name, .. } => name.clone(),
        IrType::Enum { name, .. } => name.clone(),
        IrType::TraitObject { name, .. } => format!("dyn{}", name),
        IrType::TaskGroup => "taskgroup".to_owned(),
        IrType::WeakRef(inner) => format!("weakref_{}", mangle_ir_type(inner)),
        IrType::Infer => "infer".to_owned(),
        IrType::Tuple(elems) => {
            let inner: Vec<String> = elems.iter().map(mangle_ir_type).collect();
            format!("tuple_{}", inner.join("_"))
        }
        IrType::Array { elem, len } => {
            format!("arr{}_{}", len, mangle_ir_type(elem))
        }
        IrType::Option(inner) => format!("opt_{}", mangle_ir_type(inner)),
        IrType::ResultType(ok, err) => format!(
            "res_{}_{}",
            mangle_ir_type(ok),
            mangle_ir_type(err)
        ),
        IrType::Chan(inner) => format!("chan_{}", mangle_ir_type(inner)),
        IrType::Atomic(inner) => format!("atomic_{}", mangle_ir_type(inner)),
        IrType::Mutex(inner) => format!("mutex_{}", mangle_ir_type(inner)),
        IrType::Grad(inner) => format!("grad_{}", mangle_ir_type(inner)),
        IrType::Sparse(inner) => format!("sparse_{}", mangle_ir_type(inner)),
        IrType::List(inner) => format!("list_{}", mangle_ir_type(inner)),
        IrType::Map(k, v) => format!(
            "map_{}_{}",
            mangle_ir_type(k),
            mangle_ir_type(v)
        ),
        IrType::Tensor { dtype, shape } => format!("tensor_{}_{}", dtype, shape),
        IrType::Fn { params, ret } => {
            let ps: Vec<String> = params.iter().map(mangle_ir_type).collect();
            format!("fn_{}_to_{}", ps.join("_"), mangle_ir_type(ret))
        }
    };
    s.replace(|c: char| !c.is_alphanumeric() && c != '_', "_")
}

/// Type lowering with struct/enum definition lookup from the module.
pub fn lower_type_with_structs(ty: &AstType, module: &IrModule) -> IrType {
    match ty {
        AstType::Array { elem, len, .. } => IrType::Array {
            elem: Box::new(lower_type_with_structs(elem, module)),
            len: *len,
        },
        AstType::Named(name, _) => {
            if name == "str" {
                return IrType::Str;
            }
            if name == "Infer" {
                return IrType::Infer;
            }
            // Map scalar type names.
            if name == "i64" {
                return IrType::Scalar(DType::I64);
            }
            if name == "i32" {
                return IrType::Scalar(DType::I32);
            }
            if name == "f64" {
                return IrType::Scalar(DType::F64);
            }
            if name == "f32" {
                return IrType::Scalar(DType::F32);
            }
            if name == "bool" {
                return IrType::Scalar(DType::Bool);
            }
            let resolved_name = resolve_brought_name(name, module);
            // Check type aliases first.
            if let Some(aliased) = module.type_alias(&resolved_name) {
                return aliased.clone();
            }
            if let Some(fields) = module.struct_def(&resolved_name) {
                IrType::Struct {
                    name: resolved_name,
                    fields: fields.clone(),
                }
            } else if let Some(variants) = module.enum_def(&resolved_name) {
                IrType::Enum {
                    name: resolved_name,
                    variants: variants.clone(),
                }
            } else {
                IrType::Struct {
                    name: resolved_name,
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
        AstType::Atomic(inner, _) => {
            IrType::Atomic(Box::new(lower_type_with_structs(inner, module)))
        }
        AstType::Mutex(inner, _) => IrType::Mutex(Box::new(lower_type_with_structs(inner, module))),
        AstType::List(elem, _) => IrType::List(Box::new(lower_type_with_structs(elem, module))),
        AstType::Map(k, v, _) => IrType::Map(
            Box::new(lower_type_with_structs(k, module)),
            Box::new(lower_type_with_structs(v, module)),
        ),
        AstType::WeakRef(inner, _) => IrType::WeakRef(Box::new(lower_type_with_structs(inner, module))),
        AstType::Fn { params, ret, .. } => IrType::Fn {
            params: params.iter().map(|p| lower_type_with_structs(p, module)).collect(),
            ret: Box::new(lower_type_with_structs(ret, module)),
        },
        AstType::Grad(inner, _) => IrType::Grad(Box::new(lower_type_with_structs(inner, module))),
        AstType::Sparse(inner, _) => IrType::Sparse(Box::new(lower_type_with_structs(inner, module))),
        AstType::Generic { name, args, .. } => {
            let resolved_args: Vec<IrType> = args
                .iter()
                .map(|arg| lower_type_with_structs(arg, module))
                .collect();
            let mangled_name = format!("{}__{}", name, resolved_args.iter().map(mangle_ir_type).collect::<Vec<_>>().join("_"));
            let resolved_name = resolve_brought_name(&mangled_name, module);
            if let Some(fields) = module.struct_def(&resolved_name) {
                IrType::Struct {
                    name: resolved_name,
                    fields: fields.clone(),
                }
            } else {
                // Fallback: use the generic struct template's registered field layout.
                let resolved_base = resolve_brought_name(name, module);
                if let Some(template_fields) = module.struct_def(&resolved_base) {
                    IrType::Struct {
                        name: resolved_name,
                        fields: template_fields.clone(),
                    }
                } else {
                    IrType::Struct {
                        name: resolved_name,
                        fields: Vec::new(),
                    }
                }
            }
        }
        AstType::ConstInt(_, _) => IrType::Scalar(DType::I64),
        AstType::AssocType { .. } => IrType::Infer,
        AstType::DynTrait { trait_name, .. } => {
            let methods = module
                .trait_def(trait_name)
                .cloned()
                .unwrap_or_default();
            IrType::TraitObject {
                name: trait_name.clone(),
                methods,
            }
        }
        AstType::MaskEffectType { .. } => IrType::Infer,
        AstType::Ref(inner, _) => lower_type_with_structs(inner, module),
        AstType::RefMut(inner, _) => lower_type_with_structs(inner, module),
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

/// Returns (trait_name, method_name) for an operator that can be overloaded,
/// or None for ops that cannot be overloaded (comparisons, logical).
fn op_trait_method(op: AstBinOp) -> Option<(&'static str, &'static str)> {
    match op {
        AstBinOp::Add => Some(("Add", "add")),
        AstBinOp::Sub => Some(("Sub", "sub")),
        AstBinOp::Mul => Some(("Mul", "mul")),
        AstBinOp::Div => Some(("Div", "div")),
        AstBinOp::Mod => Some(("Rem", "rem")),
        _ => None,
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
/// True when `block` is built only from forms this inliner fully traverses.
///
/// A whitelist, not a `return` detector, and deliberately so. A `return` can
/// only be reached through a block, and blocks appear in exactly two expression
/// forms (`If` and `Mask`) -- but reaching those means recursing through every
/// expression variant, and the nearest existing walker
/// (`collect_rebound_vars_in_expr`) covers 33 of roughly a hundred. Missing one
/// branch would mean inlining a body whose `return` terminates the *caller*: a
/// miscompile, not a missed optimisation. Whitelisting inverts the failure --
/// an unrecognised form declines the inline, and the program merely gets the
/// same error it did before.
fn taped_inline_ok_block(block: &AstBlock) -> bool {
    let stmts_ok = block.stmts.iter().all(|st| match st {
        AstStmt::Let { init, .. } => taped_inline_ok_expr(init),
        AstStmt::LetTuple { init, .. } => taped_inline_ok_expr(init),
        AstStmt::Expr(e) => taped_inline_ok_expr(e),
        AstStmt::Assign { target, value, .. } => {
            taped_inline_ok_expr(target) && taped_inline_ok_expr(value)
        }
        // `return` is the case this exists to reject. `break`/`continue` are
        // rejected because in an inlined body they would bind to whatever loop
        // encloses the *call site*. Everything else is simply not traversed.
        _ => false,
    });
    stmts_ok && block.tail.as_deref().is_some_and(taped_inline_ok_expr)
}

fn taped_inline_ok_expr(expr: &AstExpr) -> bool {
    match expr {
        AstExpr::Ident(_)
        | AstExpr::IntLit { .. }
        | AstExpr::FloatLit { .. }
        | AstExpr::BoolLit { .. }
        | AstExpr::StringLit { .. } => true,
        AstExpr::BinOp { lhs, rhs, .. } => {
            taped_inline_ok_expr(lhs) && taped_inline_ok_expr(rhs)
        }
        AstExpr::UnaryOp { expr, .. } => taped_inline_ok_expr(expr),
        AstExpr::Cast { expr, .. } => taped_inline_ok_expr(expr),
        AstExpr::FieldAccess { base, .. } => taped_inline_ok_expr(base),
        AstExpr::Tuple { elements, .. } => elements.iter().all(taped_inline_ok_expr),
        AstExpr::Index { base, indices, .. } => {
            taped_inline_ok_expr(base) && indices.iter().all(taped_inline_ok_expr)
        }
        AstExpr::Call { args, named_args, .. } => {
            args.iter().all(taped_inline_ok_expr)
                && named_args.iter().all(|(_, e)| taped_inline_ok_expr(e))
        }
        AstExpr::MethodCall { base, args, .. } => {
            taped_inline_ok_expr(base) && args.iter().all(taped_inline_ok_expr)
        }
        // The one block-bearing form worth supporting: a helper whose body is
        // `if cond { a } else { b }`. Both arms are checked by the same rule.
        AstExpr::If {
            cond,
            then_block,
            else_block,
            ..
        } => {
            taped_inline_ok_expr(cond)
                && taped_inline_ok_block(then_block)
                && else_block.as_ref().is_none_or(taped_inline_ok_block)
        }
        _ => false,
    }
}

fn find_rebound_vars(block: &AstBlock) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    collect_rebound_vars_in_block(block, &mut names, true);
    names
}

fn collect_rebound_vars_in_block(block: &AstBlock, names: &mut Vec<String>, include_lets: bool) {
    for stmt in &block.stmts {
        collect_rebound_vars_in_stmt(stmt, names, include_lets);
    }
    if let Some(tail) = &block.tail {
        collect_rebound_vars_in_expr(tail, names);
    }
}

fn collect_rebound_vars_in_stmt(stmt: &AstStmt, names: &mut Vec<String>, include_lets: bool) {
    match stmt {
        AstStmt::Let { name, init, .. } => {
            if include_lets && !names.contains(&name.name) {
                names.push(name.name.clone());
            }
            collect_rebound_vars_in_expr(init, names);
        }
        AstStmt::Expr(expr) => collect_rebound_vars_in_expr(expr, names),
        AstStmt::While { cond, body, .. } => {
            collect_rebound_vars_in_expr(cond, names);
            collect_rebound_vars_in_block(body, names, false);
        }
        AstStmt::Loop { body, .. } => {
            collect_rebound_vars_in_block(body, names, false);
        }
        AstStmt::Break { .. } | AstStmt::Continue { .. } => {}
        AstStmt::ForRange {
            var,
            start,
            end,
            body,
            ..
        }
        | AstStmt::ParFor {
            var,
            start,
            end,
            body,
            ..
        } => {
            if include_lets && !names.contains(&var.name) {
                names.push(var.name.clone());
            }
            collect_rebound_vars_in_expr(start, names);
            collect_rebound_vars_in_expr(end, names);
            collect_rebound_vars_in_block(body, names, false);
        }
        AstStmt::Assign { target, value, .. } => {
            if let AstExpr::Ident(ident) = target.as_ref() {
                if !names.contains(&ident.name) {
                    names.push(ident.name.clone());
                }
            }
            collect_rebound_vars_in_expr(target, names);
            collect_rebound_vars_in_expr(value, names);
        }
        AstStmt::LetTuple {
            names: tuple_names,
            init,
            ..
        } => {
            if include_lets {
                for name in tuple_names {
                    if !names.contains(&name.name) {
                        names.push(name.name.clone());
                    }
                }
            }
            collect_rebound_vars_in_expr(init, names);
        }
        AstStmt::Return { value, .. } => {
            if let Some(value) = value {
                collect_rebound_vars_in_expr(value, names);
            }
        }
        AstStmt::Spawn { body, .. } => {
            for stmt in body {
                collect_rebound_vars_in_stmt(stmt, names, false);
            }
        }
        AstStmt::ForEach {
            var, iter, body, ..
        } => {
            if include_lets && !names.contains(&var.name) {
                names.push(var.name.clone());
            }
            collect_rebound_vars_in_expr(iter, names);
            collect_rebound_vars_in_block(body, names, false);
        }
        AstStmt::MaskStmt { body, .. } => collect_rebound_vars_in_block(body, names, false),
        AstStmt::HandleStmt { expr, .. } => collect_rebound_vars_in_expr(expr, names),
        AstStmt::Defer { expr, .. } => collect_rebound_vars_in_expr(expr, names),
        AstStmt::Yield { expr, .. } => {
            collect_rebound_vars_in_expr(expr, names);
        }
        AstStmt::Select { arms, default, .. } => {
            for arm in arms {
                collect_rebound_vars_in_expr(&arm.channel, names);
                collect_rebound_vars_in_block(&arm.body, names, false);
            }
            if let Some(d) = default {
                collect_rebound_vars_in_block(d, names, false);
            }
        }
    }
}

/// Builtins that mutate a collection in place. First argument is the target.
const MUTATING_COLLECTION_BUILTINS: &[&str] = &[
    "push", "pop", "list_push", "list_pop", "list_set", "list_insert",
    "list_remove", "list_sort", "map_set", "map_insert", "map_remove", "set",
];

/// Find the first call of a mutating collection builtin whose target is one of
/// `captured`. Records `(collection name, builtin name)`.
///
/// Deliberately conservative: it only inspects statement and expression forms
/// that can contain a call, so an unhandled form yields a missed violation
/// rather than a spurious one.
fn find_captured_mutation(
    block: &AstBlock,
    captured: &std::collections::HashSet<&str>,
    out: &mut Option<(String, String)>,
) {
    for stmt in &block.stmts {
        find_captured_mutation_stmt(stmt, captured, out);
    }
    if let Some(tail) = &block.tail {
        find_captured_mutation_expr(tail, captured, out);
    }
}

fn find_captured_mutation_stmt(
    stmt: &AstStmt,
    captured: &std::collections::HashSet<&str>,
    out: &mut Option<(String, String)>,
) {
    if out.is_some() {
        return;
    }
    match stmt {
        AstStmt::Expr(e) => find_captured_mutation_expr(e, captured, out),
        AstStmt::Let { init, .. } => find_captured_mutation_expr(init, captured, out),
        AstStmt::Assign { value, .. } => find_captured_mutation_expr(value, captured, out),
        AstStmt::Return { value: Some(e), .. } => find_captured_mutation_expr(e, captured, out),
        AstStmt::While { body, cond, .. } => {
            find_captured_mutation_expr(cond, captured, out);
            find_captured_mutation(body, captured, out);
        }
        AstStmt::Loop { body, .. } => find_captured_mutation(body, captured, out),
        AstStmt::ForRange { body, .. } => find_captured_mutation(body, captured, out),
        AstStmt::ForEach { body, .. } => find_captured_mutation(body, captured, out),
        AstStmt::ParFor { body, .. } => find_captured_mutation(body, captured, out),
        _ => {}
    }
}

fn find_captured_mutation_expr(
    expr: &AstExpr,
    captured: &std::collections::HashSet<&str>,
    out: &mut Option<(String, String)>,
) {
    if out.is_some() {
        return;
    }
    match expr {
        AstExpr::Call { callee, args, .. } => {
            if MUTATING_COLLECTION_BUILTINS.contains(&callee.name.as_str()) {
                if let Some(AstExpr::Ident(target)) = args.first() {
                    if captured.contains(target.name.as_str()) {
                        *out = Some((target.name.clone(), callee.name.clone()));
                        return;
                    }
                }
            }
            for a in args {
                find_captured_mutation_expr(a, captured, out);
            }
        }
        AstExpr::BinOp { lhs, rhs, .. } => {
            find_captured_mutation_expr(lhs, captured, out);
            find_captured_mutation_expr(rhs, captured, out);
        }
        AstExpr::UnaryOp { expr, .. } => find_captured_mutation_expr(expr, captured, out),
        AstExpr::If {
            cond,
            then_block,
            else_block,
            ..
        } => {
            find_captured_mutation_expr(cond, captured, out);
            find_captured_mutation(then_block, captured, out);
            if let Some(eb) = else_block {
                find_captured_mutation(eb, captured, out);
            }
        }
        AstExpr::Block(b) => find_captured_mutation(b, captured, out),
        _ => {}
    }
}

fn collect_rebound_vars_in_expr(expr: &AstExpr, names: &mut Vec<String>) {
    match expr {
        AstExpr::Ident(_)
        | AstExpr::IntLit { .. }
        | AstExpr::FloatLit { .. }
        | AstExpr::BoolLit { .. }
        | AstExpr::StringLit { .. } => {}
        AstExpr::BinOp { lhs, rhs, .. } => {
            collect_rebound_vars_in_expr(lhs, names);
            collect_rebound_vars_in_expr(rhs, names);
        }
        AstExpr::Call { args, .. } | AstExpr::Tuple { elements: args, .. } => {
            for arg in args {
                collect_rebound_vars_in_expr(arg, names);
            }
        }
        AstExpr::UnaryOp { expr, .. }
        | AstExpr::Cast { expr, .. }
        | AstExpr::Await { expr, .. }
        | AstExpr::Try { expr, .. }
        | AstExpr::Lambda { body: expr, .. } => {
            collect_rebound_vars_in_expr(expr, names);
        }
        AstExpr::If {
            cond,
            then_block,
            else_block,
            ..
        } => {
            collect_rebound_vars_in_expr(cond, names);
            collect_rebound_vars_in_block(then_block, names, false);
            if let Some(else_block) = else_block {
                collect_rebound_vars_in_block(else_block, names, false);
            }
        }
        AstExpr::Block(block) => collect_rebound_vars_in_block(block, names, false),
        AstExpr::Index { base, indices, .. } => {
            collect_rebound_vars_in_expr(base, names);
            for index in indices {
                collect_rebound_vars_in_expr(index, names);
            }
        }
        AstExpr::StructLit { fields, spread, .. } => {
            for (_, value) in fields {
                collect_rebound_vars_in_expr(value, names);
            }
            if let Some(s) = spread {
                collect_rebound_vars_in_expr(s, names);
            }
        }
        AstExpr::FieldAccess { base, .. } | AstExpr::TupleIndex { base, .. } => {
            collect_rebound_vars_in_expr(base, names);
        }
        AstExpr::When {
            scrutinee, arms, ..
        } => {
            collect_rebound_vars_in_expr(scrutinee, names);
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    collect_rebound_vars_in_expr(guard, names);
                }
                collect_rebound_vars_in_expr(&arm.body, names);
            }
        }
        AstExpr::ArrayLit { elems, .. } => {
            for elem in elems {
                collect_rebound_vars_in_expr(elem, names);
            }
        }
        AstExpr::MethodCall { base, args, .. } => {
            collect_rebound_vars_in_expr(base, names);
            for arg in args {
                collect_rebound_vars_in_expr(arg, names);
            }
        }
        AstExpr::Mask { body, .. } => collect_rebound_vars_in_block(body, names, false),
        AstExpr::Handle { expr, .. } => collect_rebound_vars_in_expr(expr, names),
        AstExpr::NullCoal { expr, default, .. } => {
            collect_rebound_vars_in_expr(expr, names);
            collect_rebound_vars_in_expr(default, names);
        }
        AstExpr::MapLiteral { entries, .. } => {
            for (k, v) in entries {
                collect_rebound_vars_in_expr(k, names);
                collect_rebound_vars_in_expr(v, names);
            }
        }
        AstExpr::Ref { expr, .. }
        | AstExpr::RefMut { expr, .. }
        | AstExpr::Deref { expr, .. }
        | AstExpr::Move { expr, .. } => {
            collect_rebound_vars_in_expr(expr, names);
        }
        AstExpr::Unsafe { body, .. } => collect_rebound_vars_in_expr(body, names),
        AstExpr::Splat { expr, .. } => collect_rebound_vars_in_expr(expr, names),
        AstExpr::MacroCall { args, .. } => {
            for a in args {
                collect_rebound_vars_in_expr(a, names);
            }
        }
        AstExpr::TryCatch { body, catch_body, .. } => {
            collect_rebound_vars_in_expr(body, names);
            collect_rebound_vars_in_expr(catch_body, names);
        }
        AstExpr::Raise { args, .. } => {
            for a in args {
                collect_rebound_vars_in_expr(a, names);
            }
        }
    }
}

pub fn substitute_ast_type(
    ty: &AstType,
    type_subs: &std::collections::HashMap<String, AstType>,
    constructor_subs: &std::collections::HashMap<String, String>,
) -> AstType {
    match ty {
        AstType::Scalar(k, span) => AstType::Scalar(*k, *span),
        AstType::Named(name, span) => {
            if let Some(concrete) = type_subs.get(name) {
                concrete.clone()
            } else {
                AstType::Named(name.clone(), *span)
            }
        }
        AstType::Tuple(elems, span) => AstType::Tuple(
            elems
                .iter()
                .map(|e| substitute_ast_type(e, type_subs, constructor_subs))
                .collect(),
            *span,
        ),
        AstType::Array { elem, len, len_expr, span } => AstType::Array {
            elem: Box::new(substitute_ast_type(elem, type_subs, constructor_subs)),
            len: *len,
            len_expr: len_expr.clone(),
            span: *span,
        },
        AstType::Option(inner, span) => AstType::Option(
            Box::new(substitute_ast_type(inner, type_subs, constructor_subs)),
            *span,
        ),
        AstType::Result(ok_ty, err_ty, span) => AstType::Result(
            Box::new(substitute_ast_type(ok_ty, type_subs, constructor_subs)),
            Box::new(substitute_ast_type(err_ty, type_subs, constructor_subs)),
            *span,
        ),
        AstType::Chan(elem, span) => AstType::Chan(
            Box::new(substitute_ast_type(elem, type_subs, constructor_subs)),
            *span,
        ),
        AstType::Atomic(inner, span) => AstType::Atomic(
            Box::new(substitute_ast_type(inner, type_subs, constructor_subs)),
            *span,
        ),
        AstType::Mutex(inner, span) => AstType::Mutex(
            Box::new(substitute_ast_type(inner, type_subs, constructor_subs)),
            *span,
        ),
        AstType::Grad(inner, span) => AstType::Grad(
            Box::new(substitute_ast_type(inner, type_subs, constructor_subs)),
            *span,
        ),
        AstType::Sparse(inner, span) => AstType::Sparse(
            Box::new(substitute_ast_type(inner, type_subs, constructor_subs)),
            *span,
        ),
        AstType::List(elem, span) => AstType::List(
            Box::new(substitute_ast_type(elem, type_subs, constructor_subs)),
            *span,
        ),
        AstType::Map(k, v, span) => AstType::Map(
            Box::new(substitute_ast_type(k, type_subs, constructor_subs)),
            Box::new(substitute_ast_type(v, type_subs, constructor_subs)),
            *span,
        ),
        AstType::WeakRef(inner, span) => AstType::WeakRef(
            Box::new(substitute_ast_type(inner, type_subs, constructor_subs)),
            *span,
        ),
        AstType::Fn { params, ret, span } => AstType::Fn {
            params: params
                .iter()
                .map(|p| substitute_ast_type(p, type_subs, constructor_subs))
                .collect(),
            ret: Box::new(substitute_ast_type(ret, type_subs, constructor_subs)),
            span: *span,
        },
        AstType::Generic { name, args, span } => {
            let new_args: Vec<AstType> = args
                .iter()
                .map(|a| substitute_ast_type(a, type_subs, constructor_subs))
                .collect();
            let base_name = if let Some(target) = constructor_subs.get(name) {
                target.clone()
            } else if let Some(AstType::Named(target, _)) = type_subs.get(name) {
                target.clone()
            } else {
                name.clone()
            };
            AstType::Generic {
                name: base_name,
                args: new_args,
                span: *span,
            }
        }
        AstType::ConstInt(v, span) => AstType::ConstInt(*v, *span),
        AstType::AssocType { base, assoc_name, span } => AstType::AssocType {
            base: base.clone(),
            assoc_name: assoc_name.clone(),
            span: *span,
        },
        AstType::DynTrait { trait_name, span } => AstType::DynTrait {
            trait_name: trait_name.clone(),
            span: *span,
        },
        AstType::MaskEffectType { effects, span } => AstType::MaskEffectType {
            effects: effects.clone(),
            span: *span,
        },
        other => other.clone(),
    }
}

pub fn resolve_ast_type_with_subs(
    ty: &AstType,
    subs: &std::collections::HashMap<String, IrType>,
    module: &IrModule,
) -> IrType {
    match ty {
        AstType::Scalar(k, _) => IrType::Scalar(lower_dtype(*k)),
        AstType::Named(name, _) => {
            if name == "str" {
                return IrType::Str;
            }
            if name == "Infer" {
                return IrType::Infer;
            }
            if name == "i64" {
                return IrType::Scalar(DType::I64);
            }
            if name == "i32" {
                return IrType::Scalar(DType::I32);
            }
            if name == "f64" {
                return IrType::Scalar(DType::F64);
            }
            if name == "f32" {
                return IrType::Scalar(DType::F32);
            }
            if name == "bool" {
                return IrType::Scalar(DType::Bool);
            }
            if let Some(concrete) = subs.get(name) {
                return concrete.clone();
            }
            let resolved_name = resolve_brought_name(name, module);
            if let Some(aliased) = module.type_alias(&resolved_name) {
                return aliased.clone();
            }
            if let Some(fields) = module.struct_def(&resolved_name) {
                IrType::Struct {
                    name: resolved_name,
                    fields: fields.clone(),
                }
            } else if let Some(variants) = module.enum_def(&resolved_name) {
                IrType::Enum {
                    name: resolved_name,
                    variants: variants.clone(),
                }
            } else {
                IrType::Struct {
                    name: resolved_name,
                    fields: Vec::new(),
                }
            }
        }
        AstType::Tuple(elems, _) => IrType::Tuple(
            elems
                .iter()
                .map(|e| resolve_ast_type_with_subs(e, subs, module))
                .collect(),
        ),
        AstType::Array { elem, len, .. } => IrType::Array {
            elem: Box::new(resolve_ast_type_with_subs(elem, subs, module)),
            len: *len,
        },
        AstType::Option(inner, _) => {
            IrType::Option(Box::new(resolve_ast_type_with_subs(inner, subs, module)))
        }
        AstType::Result(ok_ty, err_ty, _) => IrType::ResultType(
            Box::new(resolve_ast_type_with_subs(ok_ty, subs, module)),
            Box::new(resolve_ast_type_with_subs(err_ty, subs, module)),
        ),
        AstType::Chan(elem, _) => IrType::Chan(Box::new(resolve_ast_type_with_subs(elem, subs, module))),
        AstType::Atomic(inner, _) => {
            IrType::Atomic(Box::new(resolve_ast_type_with_subs(inner, subs, module)))
        }
        AstType::Mutex(inner, _) => IrType::Mutex(Box::new(resolve_ast_type_with_subs(inner, subs, module))),
        AstType::List(elem, _) => IrType::List(Box::new(resolve_ast_type_with_subs(elem, subs, module))),
        AstType::Map(k, v, _) => IrType::Map(
            Box::new(resolve_ast_type_with_subs(k, subs, module)),
            Box::new(resolve_ast_type_with_subs(v, subs, module)),
        ),
        AstType::WeakRef(inner, _) => IrType::WeakRef(Box::new(resolve_ast_type_with_subs(inner, subs, module))),
        AstType::Fn { params, ret, .. } => IrType::Fn {
            params: params.iter().map(|p| resolve_ast_type_with_subs(p, subs, module)).collect(),
            ret: Box::new(resolve_ast_type_with_subs(ret, subs, module)),
        },
        AstType::Grad(inner, _) => IrType::Grad(Box::new(resolve_ast_type_with_subs(inner, subs, module))),
        AstType::Sparse(inner, _) => IrType::Sparse(Box::new(resolve_ast_type_with_subs(inner, subs, module))),
        AstType::Generic { name, args, .. } => {
            let resolved_args: Vec<IrType> = args
                .iter()
                .map(|arg| resolve_ast_type_with_subs(arg, subs, module))
                .collect();

            // `F` bound to a builtin constructor rebuilds the builtin type
            // rather than looking for a record called `list__i64`. Without this
            // the substitution silently produced a struct name that is defined
            // nowhere, which is the same failure mode as #14.
            if let Some(IrType::Struct { name: ctor, fields }) = subs.get(name) {
                if fields.is_empty() {
                    match ctor.as_str() {
                        "list" => {
                            return IrType::List(Box::new(
                                resolved_args.first().cloned().unwrap_or(IrType::Infer),
                            ))
                        }
                        "option" => {
                            return IrType::Option(Box::new(
                                resolved_args.first().cloned().unwrap_or(IrType::Infer),
                            ))
                        }
                        "map" => {
                            return IrType::Map(
                                Box::new(resolved_args.first().cloned().unwrap_or(IrType::Infer)),
                                Box::new(resolved_args.get(1).cloned().unwrap_or(IrType::Infer)),
                            )
                        }
                        _ => {}
                    }
                }
            }

            let base_name = if let Some(concrete_ty) = subs.get(name) {
                match concrete_ty {
                    IrType::Struct { name: s_name, .. } => s_name.clone(),
                    _ => name.clone(),
                }
            } else {
                name.clone()
            };
            let mangled_name = format!("{}__{}", base_name, resolved_args.iter().map(mangle_ir_type).collect::<Vec<_>>().join("_"));
            let resolved_name = resolve_brought_name(&mangled_name, module);
            if let Some(fields) = module.struct_def(&resolved_name) {
                let resolved_fields: Vec<(String, IrType)> = fields.iter()
                    .map(|(n, t)| (n.clone(), resolve_concrete_field(t, subs, module)))
                    .collect();
                IrType::Struct {
                    name: resolved_name,
                    fields: resolved_fields,
                }
            } else {
                // Fallback: use the generic struct template's registered field layout.
                let resolved_base = resolve_brought_name(name, module);
                if let Some(template_fields) = module.struct_def(&resolved_base) {
                    let resolved_fields: Vec<(String, IrType)> = template_fields.iter()
                        .map(|(n, t)| (n.clone(), resolve_concrete_field(t, subs, module)))
                        .collect();
                    IrType::Struct {
                        name: resolved_name,
                        fields: resolved_fields,
                    }
                } else {
                    IrType::Struct {
                        name: resolved_name,
                        fields: Vec::new(),
                    }
                }
            }
        }
        AstType::ConstInt(_, _) => IrType::Scalar(DType::I64),
        AstType::AssocType { .. } => IrType::Infer,
        AstType::DynTrait { trait_name, .. } => {
            let methods = module
                .trait_def(trait_name)
                .cloned()
                .unwrap_or_default();
            IrType::TraitObject {
                name: trait_name.clone(),
                methods,
            }
        }
        AstType::MaskEffectType { .. } => IrType::Infer,
        other => lower_type(other),
    }
}

pub(crate) fn resolve_concrete_field(ft: &IrType, subs: &HashMap<String, IrType>, module: &IrModule) -> IrType {
    match ft {
        IrType::List(inner) => IrType::List(Box::new(resolve_concrete_field(inner, subs, module))),
        IrType::Map(k, v) => IrType::Map(
            Box::new(resolve_concrete_field(k, subs, module)),
            Box::new(resolve_concrete_field(v, subs, module)),
        ),
        IrType::Option(inner) => IrType::Option(Box::new(resolve_concrete_field(inner, subs, module))),
        IrType::Chan(inner) => IrType::Chan(Box::new(resolve_concrete_field(inner, subs, module))),
        IrType::Tuple(elems) => IrType::Tuple(
            elems.iter().map(|e| resolve_concrete_field(e, subs, module)).collect(),
        ),
        IrType::Struct { name, fields } => {
            if let Some(concrete) = subs.get(name) {
                return concrete.clone();
            }
            let new_fields: Vec<(String, IrType)> = fields
                .iter()
                .map(|(n, t)| (n.clone(), resolve_concrete_field(t, subs, module)))
                .collect();
            IrType::Struct { name: name.clone(), fields: new_fields }
        }
        other => other.clone(),
    }
}

pub fn collect_generic_apps_in_type(ty: &AstType, apps: &mut std::collections::HashSet<(String, Vec<AstType>)>) {
    match ty {
        AstType::Tuple(elems, _) => {
            for e in elems {
                collect_generic_apps_in_type(e, apps);
            }
        }
        AstType::Array { elem, .. } => collect_generic_apps_in_type(elem, apps),
        AstType::Option(inner, _) => collect_generic_apps_in_type(inner, apps),
        AstType::Result(ok, err, _) => {
            collect_generic_apps_in_type(ok, apps);
            collect_generic_apps_in_type(err, apps);
        }
        AstType::Chan(elem, _) => collect_generic_apps_in_type(elem, apps),
        AstType::Atomic(inner, _) => collect_generic_apps_in_type(inner, apps),
        AstType::Mutex(inner, _) => collect_generic_apps_in_type(inner, apps),
        AstType::Grad(inner, _) => collect_generic_apps_in_type(inner, apps),
        AstType::Sparse(inner, _) => collect_generic_apps_in_type(inner, apps),
        AstType::List(elem, _) => collect_generic_apps_in_type(elem, apps),
        AstType::Map(k, v, _) => {
            collect_generic_apps_in_type(k, apps);
            collect_generic_apps_in_type(v, apps);
        }
        AstType::WeakRef(inner, _) => collect_generic_apps_in_type(inner, apps),
        AstType::Fn { params, ret, .. } => {
            for p in params {
                collect_generic_apps_in_type(p, apps);
            }
            collect_generic_apps_in_type(ret, apps);
        }
        AstType::Generic { name, args, .. } => {
            apps.insert((name.clone(), args.clone()));
            for a in args {
                collect_generic_apps_in_type(a, apps);
            }
        }
        _ => {}
    }
}

pub fn collect_generic_apps_in_expr(expr: &AstExpr, apps: &mut std::collections::HashSet<(String, Vec<AstType>)>) {
    match expr {
        AstExpr::Ident(_) => {}
        AstExpr::IntLit { .. } | AstExpr::FloatLit { .. } | AstExpr::BoolLit { .. } | AstExpr::StringLit { .. } => {}
        AstExpr::BinOp { lhs, rhs, .. } => {
            collect_generic_apps_in_expr(lhs, apps);
            collect_generic_apps_in_expr(rhs, apps);
        }
        AstExpr::Call { args, .. } => {
            for a in args {
                collect_generic_apps_in_expr(a, apps);
            }
        }
        AstExpr::UnaryOp { expr: inner, .. } => {
            collect_generic_apps_in_expr(inner, apps);
        }
        AstExpr::If { cond, then_block, else_block, .. } => {
            collect_generic_apps_in_expr(cond, apps);
            collect_generic_apps_in_block(then_block, apps);
            if let Some(ref eb) = else_block {
                collect_generic_apps_in_block(eb, apps);
            }
        }
        AstExpr::Block(block) => collect_generic_apps_in_block(block, apps),
        AstExpr::Index { base, indices, .. } => {
            collect_generic_apps_in_expr(base, apps);
            for i in indices {
                collect_generic_apps_in_expr(i, apps);
            }
        }
        AstExpr::Cast { expr: inner, ty, .. } => {
            collect_generic_apps_in_expr(inner, apps);
            collect_generic_apps_in_type(ty, apps);
        }
        AstExpr::StructLit { fields, spread, .. } => {
            for (_, f_expr) in fields {
                collect_generic_apps_in_expr(f_expr, apps);
            }
            if let Some(s) = spread {
                collect_generic_apps_in_expr(s, apps);
            }
        }
        AstExpr::FieldAccess { base, .. } => {
            collect_generic_apps_in_expr(base, apps);
        }
        AstExpr::When { scrutinee, arms, .. } => {
            collect_generic_apps_in_expr(scrutinee, apps);
            for arm in arms {
                collect_generic_apps_in_expr(&arm.body, apps);
            }
        }
        AstExpr::Tuple { elements, .. } => {
            for e in elements {
                collect_generic_apps_in_expr(e, apps);
            }
        }
        AstExpr::TupleIndex { base, .. } => {
            collect_generic_apps_in_expr(base, apps);
        }
        AstExpr::ArrayLit { elems, .. } => {
            for e in elems {
                collect_generic_apps_in_expr(e, apps);
            }
        }
        AstExpr::Lambda { params, body, .. } => {
            for p in params {
                collect_generic_apps_in_type(&p.ty, apps);
            }
            collect_generic_apps_in_expr(body, apps);
        }
        AstExpr::Await { expr: inner, .. } => {
            collect_generic_apps_in_expr(inner, apps);
        }
        AstExpr::Try { expr: inner, .. } => {
            collect_generic_apps_in_expr(inner, apps);
        }
        AstExpr::MethodCall { base, args, .. } => {
            collect_generic_apps_in_expr(base, apps);
            for a in args {
                collect_generic_apps_in_expr(a, apps);
            }
        }
        AstExpr::Mask { body, .. } => {
            collect_generic_apps_in_block(body, apps);
        }
        AstExpr::Handle { expr: inner, arms, return_ty, .. } => {
            collect_generic_apps_in_expr(inner, apps);
            collect_generic_apps_in_type(return_ty, apps);
            for arm in arms {
                collect_generic_apps_in_expr(&arm.body, apps);
            }
        }
        AstExpr::NullCoal { expr, default, .. } => {
            collect_generic_apps_in_expr(expr, apps);
            collect_generic_apps_in_expr(default, apps);
        }
        AstExpr::MapLiteral { entries, .. } => {
            for (k, v) in entries {
                collect_generic_apps_in_expr(k, apps);
                collect_generic_apps_in_expr(v, apps);
            }
        }
        AstExpr::Ref { expr, .. }
        | AstExpr::RefMut { expr, .. }
        | AstExpr::Deref { expr, .. }
        | AstExpr::Move { expr, .. } => {
            collect_generic_apps_in_expr(expr, apps);
        }
        AstExpr::Unsafe { body, .. } => collect_generic_apps_in_expr(body, apps),
        AstExpr::Splat { expr, .. } => collect_generic_apps_in_expr(expr, apps),
        AstExpr::MacroCall { args, .. } => {
            for a in args {
                collect_generic_apps_in_expr(a, apps);
            }
        }
        AstExpr::TryCatch { body, catch_body, .. } => {
            collect_generic_apps_in_expr(body, apps);
            collect_generic_apps_in_expr(catch_body, apps);
        }
        AstExpr::Raise { args, .. } => {
            for a in args {
                collect_generic_apps_in_expr(a, apps);
            }
        }
    }
}

pub fn collect_generic_apps_in_stmt(stmt: &AstStmt, apps: &mut std::collections::HashSet<(String, Vec<AstType>)>) {
    match stmt {
        AstStmt::Let { ty, init, .. } => {
            if let Some(t) = ty {
                collect_generic_apps_in_type(t, apps);
            }
            collect_generic_apps_in_expr(init, apps);
        }
        AstStmt::Expr(expr) => collect_generic_apps_in_expr(expr, apps),
        AstStmt::While { cond, body, .. } => {
            collect_generic_apps_in_expr(cond, apps);
            collect_generic_apps_in_block(body, apps);
        }
        AstStmt::Loop { body, .. } => {
            collect_generic_apps_in_block(body, apps);
        }
        AstStmt::Break { .. } | AstStmt::Continue { .. } => {}
        AstStmt::ForRange { start, end, body, .. } => {
            collect_generic_apps_in_expr(start, apps);
            collect_generic_apps_in_expr(end, apps);
            collect_generic_apps_in_block(body, apps);
        }
        AstStmt::Assign { target, value, .. } => {
            collect_generic_apps_in_expr(target, apps);
            collect_generic_apps_in_expr(value, apps);
        }
        AstStmt::LetTuple { init, .. } => {
            collect_generic_apps_in_expr(init, apps);
        }
        AstStmt::Return { value, .. } => {
            if let Some(v) = value {
                collect_generic_apps_in_expr(v, apps);
            }
        }
        AstStmt::Spawn { body, group, .. } => {
            for s in body {
                collect_generic_apps_in_stmt(s, apps);
            }
            if let Some(g) = group {
                collect_generic_apps_in_expr(g, apps);
            }
        }
        AstStmt::ParFor { start, end, body, .. } => {
            collect_generic_apps_in_expr(start, apps);
            collect_generic_apps_in_expr(end, apps);
            collect_generic_apps_in_block(body, apps);
        }
        AstStmt::ForEach { iter, body, .. } => {
            collect_generic_apps_in_expr(iter, apps);
            collect_generic_apps_in_block(body, apps);
        }
        AstStmt::MaskStmt { body, .. } => {
            collect_generic_apps_in_block(body, apps);
        }
        AstStmt::HandleStmt { expr, arms, return_ty, .. } => {
            collect_generic_apps_in_expr(expr, apps);
            collect_generic_apps_in_type(return_ty, apps);
            for arm in arms {
                collect_generic_apps_in_expr(&arm.body, apps);
            }
        }
        AstStmt::Defer { expr, .. } => {
            collect_generic_apps_in_expr(expr, apps);
        }
        AstStmt::Yield { expr, .. } => {
            collect_generic_apps_in_expr(expr, apps);
        }
        AstStmt::Select { arms, default, .. } => {
            for arm in arms {
                collect_generic_apps_in_expr(&arm.channel, apps);
                collect_generic_apps_in_block(&arm.body, apps);
            }
            if let Some(d) = default {
                collect_generic_apps_in_block(d, apps);
            }
        }
    }
}

pub fn collect_generic_apps_in_block(block: &AstBlock, apps: &mut std::collections::HashSet<(String, Vec<AstType>)>) {
    for stmt in &block.stmts {
        collect_generic_apps_in_stmt(stmt, apps);
    }
}

pub fn populate_struct_fields(ty: &mut IrType, module: &IrModule) {
    match ty {
        IrType::Struct { name, fields } => {
            if fields.is_empty() {
                if let Some(def_fields) = module.struct_def(name) {
                    *fields = def_fields.clone();
                }
            }
            for (_, f_ty) in fields {
                populate_struct_fields(f_ty, module);
            }
        }
        IrType::Tuple(elems) => {
            for e in elems {
                populate_struct_fields(e, module);
            }
        }
        IrType::Array { elem, .. } => populate_struct_fields(elem, module),
        IrType::Option(inner) => populate_struct_fields(inner, module),
        IrType::ResultType(ok, err) => {
            populate_struct_fields(ok, module);
            populate_struct_fields(err, module);
        }
        IrType::Chan(inner) => populate_struct_fields(inner, module),
        IrType::Atomic(inner) => populate_struct_fields(inner, module),
        IrType::Mutex(inner) => populate_struct_fields(inner, module),
        IrType::Grad(inner) => populate_struct_fields(inner, module),
        IrType::Sparse(inner) => populate_struct_fields(inner, module),
        IrType::List(inner) => populate_struct_fields(inner, module),
        IrType::Map(k, v) => {
            populate_struct_fields(k, module);
            populate_struct_fields(v, module);
        }
        IrType::WeakRef(inner) => populate_struct_fields(inner, module),
        IrType::Fn { params, ret } => {
            for p in params {
                populate_struct_fields(p, module);
            }
            populate_struct_fields(ret, module);
        }
        _ => {}
    }
}
