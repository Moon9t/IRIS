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
use crate::ir::instr::{BinOp, IrInstr, ScalarUnaryOp, TensorOp};
use crate::ir::module::IrModule;
use crate::ir::types::{DType, IrType};
use crate::ir::value::ValueId;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Phase 101: Cross-platform target support
// ---------------------------------------------------------------------------

/// Maps a user-friendly preset name to an LLVM target triple.
/// Returns `None` for unknown presets.
pub fn target_preset_to_triple(preset: &str) -> Option<&'static str> {
    match preset {
        "linux-x64" => Some("x86_64-unknown-linux-gnu"),
        "linux-arm64" => Some("aarch64-unknown-linux-gnu"),
        "macos-x64" => Some("x86_64-apple-macosx14.0"),
        "macos-arm64" => Some("aarch64-apple-macosx14.0"),
        "windows-x64" => Some("x86_64-pc-windows-gnu"),
        "windows-arm64" => Some("aarch64-pc-windows-gnu"),
        "riscv64-linux" => Some("riscv64gc-unknown-linux-gnu"),
        _ => None,
    }
}

/// Returns the LLVM data layout string for a given target triple.
pub fn target_data_layout(triple: &str) -> &'static str {
    if triple.starts_with("aarch64-apple") {
        "e-m:o-i64:64-i128:128-n32:64-S128"
    } else if triple.starts_with("aarch64") {
        "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
    } else if triple.starts_with("riscv64") {
        "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
    } else {
        // x86_64 (linux, windows, macos)
        "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
    }
}

/// Returns the default LLVM target triple for the host toolchain used by IRIS.
pub fn native_target_triple() -> &'static str {
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        "x86_64-pc-windows-gnu"
    }
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        "aarch64-pc-windows-gnu"
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        "x86_64-apple-macosx14.0"
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        "aarch64-apple-macosx14.0"
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        "x86_64-unknown-linux-gnu"
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        "aarch64-unknown-linux-gnu"
    }
    #[cfg(all(target_os = "linux", target_arch = "riscv64"))]
    {
        "riscv64gc-unknown-linux-gnu"
    }
    #[cfg(not(any(
        all(
            target_os = "windows",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        all(
            target_os = "macos",
            any(target_arch = "x86_64", target_arch = "aarch64")
        ),
        all(
            target_os = "linux",
            any(
                target_arch = "x86_64",
                target_arch = "aarch64",
                target_arch = "riscv64"
            )
        )
    )))]
    {
        "x86_64-unknown-linux-gnu"
    }
}

/// Emits complete LLVM IR for all functions in the module.
///
/// Improvements over `emit_llvm_stub`:
/// 1. Named struct type declarations.
/// 2. Fixed scalar arrays use `[N x T]` + `alloca`.
/// 3. User function calls use real typed signatures.
/// 4. GEP-based inline struct/array access.
pub fn emit_llvm_ir(module: &IrModule) -> Result<String, CodegenError> {
    emit_llvm_ir_impl(module, None, None)
}

/// Like `emit_llvm_ir` but overrides the target triple (and deduces data layout).
/// `target` may be a preset name (e.g. `"macos-arm64"`) or a raw LLVM triple.
pub fn emit_llvm_ir_with_target(
    module: &IrModule,
    target: Option<&str>,
) -> Result<String, CodegenError> {
    emit_llvm_ir_impl(module, None, target)
}

/// Like `emit_llvm_ir` but for native binary: renames the entry to `iris_main`
/// and appends a C-compatible `main(i32, ptr)` wrapper.
pub fn emit_llvm_ir_for_binary(module: &IrModule) -> Result<String, CodegenError> {
    emit_llvm_ir_for_binary_with_target(module, None)
}

/// Like `emit_llvm_ir_for_binary` but overrides the target triple.
pub fn emit_llvm_ir_for_binary_with_target(
    module: &IrModule,
    target: Option<&str>,
) -> Result<String, CodegenError> {
    emit_llvm_ir_impl(module, Some(()), target)
}

/// Like `emit_llvm_ir_for_binary` but for `EmitKind::Eval`: the `@main` wrapper
/// prints the return value of the entry function to stdout (instead of using it
/// as an exit code), then returns 0.  This makes the captured stdout identical
/// to what the interpreter would return.
pub fn emit_llvm_ir_for_eval(module: &IrModule) -> Result<String, CodegenError> {
    emit_llvm_ir_for_eval_with_target(module, None)
}

/// Like `emit_llvm_ir_for_eval` but overrides the target triple.
pub fn emit_llvm_ir_for_eval_with_target(
    module: &IrModule,
    target: Option<&str>,
) -> Result<String, CodegenError> {
    emit_llvm_ir_for_named_eval_with_target(module, None, target)
}

pub(crate) fn emit_llvm_ir_for_named_eval_with_target(
    module: &IrModule,
    entry_name: Option<&str>,
    target: Option<&str>,
) -> Result<String, CodegenError> {
    // Get the base LLVM IR (no binary wrapper — we'll add our own).
    let mut base = emit_llvm_ir_impl(module, None, target)?;

    // Find the entry function.
    let entry = match entry_name {
        Some(name) => module
            .functions()
            .iter()
            .find(|f| f.name == name && f.params.is_empty()),
        None => module
            .functions()
            .iter()
            .find(|f| f.name == "main" && f.params.is_empty())
            .or_else(|| module.functions().iter().find(|f| f.params.is_empty())),
    };

    let Some(entry_fn) = entry else {
        return if let Some(name) = entry_name {
            Err(CodegenError::Unsupported {
                backend: "llvm".into(),
                detail: format!(
                    "cannot build native wrapper: zero-argument function '{}' not found",
                    name
                ),
            })
        } else {
            Ok(base)
        };
    };

    // Rename @entry_name to @iris_eval_main in the emitted IR.
    let orig = format!("@{}(", entry_fn.name);
    base = base.replace(&orig, "@iris_eval_main(");

    // Determine the LLVM return type of the entry function.
    let ret_llvm = llvm_type_complete(&entry_fn.return_ty).unwrap_or_else(|_| "i64".to_owned());

    // Build the call + print based on return type.
    let (call_line, print_line) = if ret_llvm == "void" {
        ("  call void @iris_eval_main()".to_owned(), String::new())
    } else {
        let call = format!("  %eval_ret = call {} @iris_eval_main()", ret_llvm);
        let print = match ret_llvm.as_str() {
            "i64" => "  call void @iris_print_i64(i64 %eval_ret)".to_owned(),
            "i32" => "  call void @iris_print_i32(i32 %eval_ret)".to_owned(),
            "double" => "  call void @iris_print_f64(double %eval_ret)".to_owned(),
            "float" => "  call void @iris_print_f32(float %eval_ret)".to_owned(),
            "i1" => "  %eval_bool = zext i1 %eval_ret to i32\n  call void @iris_print_bool(i32 %eval_bool)".to_owned(),
            _ => "  call void @iris_print_str(ptr %eval_ret)".to_owned(),
        };
        (call, print)
    };

    // Append the eval wrapper.
    use std::fmt::Write as _;
    let _ = writeln!(
        base,
        "\ndefine i32 @main(i32 %argc, ptr %argv) {{\nentry:\n  call void @iris_set_argv(i32 %argc, ptr %argv)\n{call_line}\n{print_line}\n  ret i32 0\n}}\n",
    );

    Ok(base)
}

pub(crate) fn emit_llvm_ir_for_test_entry_with_target(
    module: &IrModule,
    entry_name: &str,
    target: Option<&str>,
) -> Result<String, CodegenError> {
    let mut base = emit_llvm_ir_impl(module, None, target)?;
    let entry_fn = module
        .functions()
        .iter()
        .find(|f| f.name == entry_name && f.params.is_empty())
        .ok_or_else(|| CodegenError::Unsupported {
            backend: "llvm".into(),
            detail: format!(
                "cannot build native test wrapper: zero-argument function '{}' not found",
                entry_name
            ),
        })?;

    let orig = format!("@{}(", entry_fn.name);
    base = base.replace(&orig, "@iris_test_entry(");

    let ret_llvm = llvm_type_complete(&entry_fn.return_ty).unwrap_or_else(|_| "void".to_owned());
    let wrapper = match ret_llvm.as_str() {
        "void" => {
            "\ndefine i32 @main(i32 %argc, ptr %argv) {\nentry:\n  call void @iris_set_argv(i32 %argc, ptr %argv)\n  call void @iris_test_entry()\n  ret i32 0\n}\n".to_owned()
        }
        "i1" => {
            "\ndefine i32 @main(i32 %argc, ptr %argv) {\nentry:\n  call void @iris_set_argv(i32 %argc, ptr %argv)\n  %test_ret = call i1 @iris_test_entry()\n  br i1 %test_ret, label %pass, label %fail\npass:\n  ret i32 0\nfail:\n  %test_ret_i32 = zext i1 %test_ret to i32\n  call void @iris_print_bool(i32 %test_ret_i32)\n  ret i32 1\n}\n".to_owned()
        }
        "i64" => {
            "\ndefine i32 @main(i32 %argc, ptr %argv) {\nentry:\n  call void @iris_set_argv(i32 %argc, ptr %argv)\n  %test_ret = call i64 @iris_test_entry()\n  %test_ok = icmp eq i64 %test_ret, 0\n  br i1 %test_ok, label %pass, label %fail\npass:\n  ret i32 0\nfail:\n  call void @iris_print_i64(i64 %test_ret)\n  ret i32 1\n}\n".to_owned()
        }
        "i32" => {
            "\ndefine i32 @main(i32 %argc, ptr %argv) {\nentry:\n  call void @iris_set_argv(i32 %argc, ptr %argv)\n  %test_ret = call i32 @iris_test_entry()\n  %test_ok = icmp eq i32 %test_ret, 0\n  br i1 %test_ok, label %pass, label %fail\npass:\n  ret i32 0\nfail:\n  call void @iris_print_i32(i32 %test_ret)\n  ret i32 1\n}\n".to_owned()
        }
        _ => format!(
            "\ndefine i32 @main(i32 %argc, ptr %argv) {{\nentry:\n  call void @iris_set_argv(i32 %argc, ptr %argv)\n  %test_ret = call {} @iris_test_entry()\n  ret i32 2\n}}\n",
            ret_llvm
        ),
    };
    base.push_str(&wrapper);
    Ok(base)
}

fn emit_llvm_ir_impl(
    module: &IrModule,
    for_binary: Option<()>,
    target_override: Option<&str>,
) -> Result<String, CodegenError> {
    let mut out = String::new();

    // Resolve target triple and data layout.
    let triple: &str = match target_override {
        Some(t) => target_preset_to_triple(t).unwrap_or(t),
        None => native_target_triple(),
    };
    let layout = target_data_layout(triple);

    // ── Header ────────────────────────────────────────────────────────────
    writeln!(out, "; IRIS Complete LLVM IR — phase 49")?;
    writeln!(
        out,
        "; Struct/array types lowered, typed calls, alloca for fixed arrays.\n"
    )?;
    writeln!(out, "target datalayout = \"{}\"", layout)?;
    writeln!(out, "target triple = \"{}\"\n", triple)?;

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

    // Continuation struct for effect handler resume support.
    writeln!(out, "%Continuation = type {{ i32, i64 }}")?;
    writeln!(out)?;

    // ── Collect global string constants ───────────────────────────────────
    let mut str_table: HashMap<String, usize> = HashMap::new();
    let mut str_vec: Vec<String> = Vec::new();
    for func in module.functions() {
        for block in func.blocks() {
            for instr in &block.instrs {
                match instr {
                    IrInstr::ConstStr { value, .. } | IrInstr::TapeRecord { op: value, .. } => {
                        if !str_table.contains_key(value) {
                            let idx = str_vec.len();
                            str_table.insert(value.clone(), idx);
                            str_vec.push(value.clone());
                        }
                    }
                    // An extern call is dispatched through the effect machinery
                    // using its own name as the effect name, so that name needs a
                    // string constant. Without this the lookup at the dispatch
                    // site fell back to index 0 and passed whatever string
                    // happened to be first in the table — producing runtime
                    // failures naming absurd effects like ',' or '=='.
                    IrInstr::CallExtern { name, .. } => {
                        if !str_table.contains_key(name) {
                            let idx = str_vec.len();
                            str_table.insert(name.clone(), idx);
                            str_vec.push(name.clone());
                        }
                    }
                    IrInstr::PushHandler { arms } => {
                        for arm in arms {
                            if !str_table.contains_key(&arm.effect_name) {
                                let idx = str_vec.len();
                                str_table.insert(arm.effect_name.clone(), idx);
                                str_vec.push(arm.effect_name.clone());
                            }
                            let fn_key = format!("__fn__{}", arm.func_name);
                            if !str_table.contains_key(&fn_key) {
                                let idx = str_vec.len();
                                str_table.insert(fn_key.clone(), idx);
                                str_vec.push(arm.func_name.clone());
                            }
                        }
                    }
                    _ => {}
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

    // ── Trait-object vtables ──────────────────────────────────────────────
    //
    // `MakeTraitObject` records both the trait and the *concrete* type at the
    // coercion site, so the set of methods a given trait object can dispatch to
    // is known statically. That makes a constant vtable per (trait, concrete)
    // pair sufficient: no runtime construction, and the slot layout is the
    // trait's own declaration order, which `IrType::TraitObject` already stores.
    //
    // Slots with no matching impl are emitted as `ptr null`. That can only be
    // reached by calling a method the concrete type does not implement, which
    // the front end rejects; emitting null keeps the table shape uniform rather
    // than silently shifting every later slot.
    let mut vtables: Vec<(String, String, Vec<String>)> = Vec::new();
    {
        let mut seen: HashSet<(String, String)> = HashSet::new();
        for func in module.functions() {
            for block in func.blocks() {
                for instr in &block.instrs {
                    if let IrInstr::MakeTraitObject {
                        target_trait,
                        concrete_ty,
                        result_ty,
                        ..
                    } = instr
                    {
                        if !seen.insert((target_trait.clone(), concrete_ty.clone())) {
                            continue;
                        }
                        // Slot order comes from the trait object's own type.
                        let slot_names: Vec<String> = match result_ty {
                            IrType::TraitObject { methods, .. } => {
                                methods.iter().map(|m| m.name.clone()).collect()
                            }
                            _ => module
                                .trait_def(target_trait)
                                .map(|ms| ms.iter().map(|m| m.name.clone()).collect())
                                .unwrap_or_default(),
                        };
                        let impls = module.trait_impl_methods();
                        let entries = impls.get(target_trait);
                        let mut slots = Vec::with_capacity(slot_names.len());
                        for mname in &slot_names {
                            let mangled = entries.and_then(|es| {
                                es.iter()
                                    .find(|(c, m, _)| c == concrete_ty && m == mname)
                                    .map(|(_, _, mangled)| mangled.clone())
                            });
                            slots.push(match mangled {
                                Some(f) => format!("ptr @{}", f),
                                None => "ptr null".to_owned(),
                            });
                        }
                        vtables.push((target_trait.clone(), concrete_ty.clone(), slots));
                    }
                }
            }
        }
        vtables.sort();
    }
    for (tname, cname, slots) in &vtables {
        writeln!(
            out,
            "@vt_{}_{} = private unnamed_addr constant [{} x ptr] [{}], align 8",
            tname,
            cname,
            slots.len().max(1),
            if slots.is_empty() {
                "ptr null".to_owned()
            } else {
                slots.join(", ")
            }
        )?;
    }
    if !vtables.is_empty() {
        writeln!(out)?;
    }

    // ── Runtime declarations ──────────────────────────────────────────────
    emit_runtime_declares(&mut out)?;

    // ── Extern (FFI) function declarations ───────────────────────────────
    for ext in &module.extern_fns {
        let ret_s = llvm_type_complete(&ext.ret_ty).unwrap_or_else(|_| "ptr".to_owned());
        let param_ss: Vec<String> = ext
            .param_types
            .iter()
            .map(|t| llvm_type_complete(t).unwrap_or_else(|_| "ptr".to_owned()))
            .collect();
        // `extern_weak`, not plain external linkage.
        //
        // `extern def` serves two purposes that codegen cannot distinguish: a
        // real FFI symbol (`iris_mlrt_onnx_load`, exported by the C runtime) and
        // an effect operation that exists only to be intercepted by a handler
        // (`extern def echo` in the effect tests, which has no implementation
        // anywhere). Both are dispatched through
        // `iris_effect_dispatch_or_call`, which takes the real function's
        // address and falls back to an installed handler when it is null.
        //
        // Taking the address of a plain `declare` would make a handler-only
        // extern an undefined symbol at link time. Weak linkage resolves to the
        // real function when one is linked in and to null when none is, which is
        // exactly the distinction the dispatcher already expects.
        //
        // Tradeoff, deliberately accepted: a misspelled FFI extern is no longer
        // a link error. It becomes a runtime "no handler for effect 'x' and no
        // real implementation", which names the symbol and is clear enough.
        writeln!(
            out,
            "declare extern_weak {} @{}({})",
            ret_s,
            ext.name,
            param_ss.join(", ")
        )?;
    }
    if !module.extern_fns.is_empty() {
        writeln!(out)?;
    }

    // ── Build function signature map for typed calls ──────────────────────
    // Maps function name → (return_type_string, Vec<param_type_string>)
    let mut fn_sigs: HashMap<String, (String, Vec<String>)> = HashMap::new();
    for func in module.functions() {
        let ret_s = llvm_type_complete(&func.return_ty).unwrap_or_else(|_| "ptr".to_owned());
        let is_lambda = func.name.starts_with("__lambda_");
        let param_ss: Vec<String> = if is_lambda {
            // All lambdas use uniform calling convention: (ptr %env, user_params...)
            let mut ss = vec!["ptr".to_owned()]; // env
            for p in func.params.iter().skip(func.capture_count) {
                ss.push(llvm_type_complete(&p.ty).unwrap_or_else(|_| "ptr".to_owned()));
            }
            ss
        } else {
            func.params
                .iter()
                .map(|p| llvm_type_complete(&p.ty).unwrap_or_else(|_| "ptr".to_owned()))
                .collect()
        };
        fn_sigs.insert(func.name.clone(), (ret_s, param_ss));
    }

    let entry_llvm_name: Option<String> = for_binary.and_then(|_| {
        module
            .functions()
            .iter()
            .find(|f| f.name == "main")
            .map(|f| f.name.clone())
            .or_else(|| {
                module
                    .functions()
                    .iter()
                    .find(|f| f.params.is_empty())
                    .map(|f| f.name.clone())
            })
    });

    // ── Function definitions ──────────────────────────────────────────────
    for func in module.functions() {
        let llvm_name = if entry_llvm_name.as_deref() == Some(func.name.as_str()) {
            "iris_main"
        } else {
            &func.name
        };
        let entry_rename = entry_llvm_name.as_deref().map(|orig| (orig, "iris_main"));
        emit_function_ir_with_name(
            func,
            llvm_name,
            entry_rename,
            &str_table,
            &fn_sigs,
            module,
            &mut out,
        )?;
    }

    // ── Spawn trampolines ─────────────────────────────────────────────────
    // For each __spawn_N function with parameters (captures), generate a
    // trampoline wrapper that takes a single `ptr` (array of boxed captures),
    // unpacks them, and calls the real spawn body function.
    for func in module.functions() {
        if !(func.name.starts_with("__spawn_") || func.name.starts_with("__async_spawn_")) {
            continue;
        }
        let tramp_name = format!("{}_trampoline", func.name);
        writeln!(out, "define ptr @{}(ptr %arg) {{", tramp_name)?;
        writeln!(out, "entry:")?;
        for (i, p) in func.params.iter().enumerate() {
            let slot = format!("%slot{}", i);
            writeln!(out, "  {} = getelementptr ptr, ptr %arg, i64 {}", slot, i)?;
            let raw = format!("%raw{}", i);
            writeln!(out, "  {} = load ptr, ptr {}", raw, slot)?;
            // Unbox to the expected parameter type.
            let param_llvm_ty = llvm_type_complete(&p.ty).unwrap_or_else(|_| "ptr".to_owned());
            if param_llvm_ty == "i64" {
                writeln!(out, "  %p{} = call i64 @iris_unbox_i64(ptr {})", i, raw)?;
            } else if param_llvm_ty == "i32" {
                writeln!(out, "  %p{} = call i64 @iris_unbox_i64(ptr {})", i, raw)?;
                writeln!(out, "  %p{}t = trunc i64 %p{} to i32", i, i)?;
            } else if param_llvm_ty == "double" {
                writeln!(out, "  %p{} = call double @iris_unbox_f64(ptr {})", i, raw)?;
            } else if param_llvm_ty == "i1" {
                writeln!(out, "  %p{}i = call i32 @iris_unbox_bool(ptr {})", i, raw)?;
                writeln!(out, "  %p{} = trunc i32 %p{}i to i1", i, i)?;
            } else if let Some(unbox_fn) = runtime_unbox_helper_for_type(&p.ty) {
                writeln!(out, "  %p{} = call ptr @{}(ptr {})", i, unbox_fn, raw)?;
            } else {
                // ptr types (closures, structs, etc.) — already unboxed ptr.
                writeln!(out, "  %p{} = bitcast ptr {} to ptr", i, raw)?;
            }
        }
        // Build call args.
        let call_args: Vec<String> = func
            .params
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let ty_s = llvm_type_complete(&p.ty).unwrap_or_else(|_| "ptr".to_owned());
                if ty_s == "i32" {
                    format!("i32 %p{}t", i)
                } else {
                    format!("{} %p{}", ty_s, i)
                }
            })
            .collect();
        let spawn_ret_ty = fn_sigs
            .get(&func.name)
            .map(|(ret_s, _)| ret_s.as_str())
            .unwrap_or("i64");
        if spawn_ret_ty == "void" {
            writeln!(out, "  call void @{}({})", func.name, call_args.join(", "))?;
        } else {
            writeln!(
                out,
                "  %spawn_ret = call {} @{}({})",
                spawn_ret_ty,
                func.name,
                call_args.join(", ")
            )?;
        }
        writeln!(out, "  call void @free(ptr %arg)")?;
        writeln!(out, "  ret ptr null")?;
        writeln!(out, "}}\n")?;
    }

    if entry_llvm_name.is_some() {
        writeln!(out, "define i32 @main(i32 %argc, ptr %argv) {{")?;
        writeln!(out, "  call void @iris_set_argv(i32 %argc, ptr %argv)")?;
        writeln!(out, "  %r = call i64 @iris_main()")?;
        writeln!(out, "  %r32 = trunc i64 %r to i32")?;
        writeln!(out, "  call void @exit(i32 %r32)")?;
        writeln!(out, "  unreachable")?;
        writeln!(out, "}}\n")?;
    }
    // Post-process emitted IR to correct occasional FP-width mismatches
    // where an instruction like `fptosi float %vN to i64` was emitted but
    // the `%vN` value is actually defined as `double` earlier in the IR.
    // This can happen when value-type bookkeeping and emitted instructions
    // get out of sync for constants/folded expressions. Fixing here avoids
    // clang rejecting the IR.
    {
        let mut lines: Vec<String> = out.lines().map(|s| s.to_owned()).collect();
        // Build a map of value -> its first-definition line (to check its type).
        let mut val_type: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        for ln in &lines {
            if let Some(idx) = ln.find(" = ") {
                let lhs = ln[..idx].trim();
                if lhs.starts_with('%') {
                    // lhs like "%v3"
                    // Try to extract a type from the rhs, e.g. "fmul double ..." or "fadd float ..."
                    let rhs = ln[idx + 3..].trim();
                    if rhs.starts_with("fmul ")
                        || rhs.starts_with("fadd ")
                        || rhs.starts_with("fsub ")
                    {
                        if rhs.contains(" double ") {
                            val_type.insert(lhs.to_string(), "double".to_string());
                        } else if rhs.contains(" float ") {
                            val_type.insert(lhs.to_string(), "float".to_string());
                        }
                    } else if rhs.starts_with("fptrunc ") {
                        val_type.insert(lhs.to_string(), "float".to_string());
                    } else if rhs.starts_with("fpext ") {
                        val_type.insert(lhs.to_string(), "double".to_string());
                    }
                }
            }
        }
        // Now fix fptosi float %vN occurrences when val_type says %vN is double.
        for ln in &mut lines {
            if ln.contains("fptosi float %") {
                // find the %vN
                if let Some(p) = ln.find("fptosi float %") {
                    let rest = &ln[p + "fptosi float ".len()..];
                    if let Some(end) = rest.find(' ') {
                        let vname = &rest[..end];
                        let key = vname.to_string();
                        if let Some(t) = val_type.get(&key) {
                            if t == "double" {
                                *ln = ln.replacen("fptosi float ", "fptosi double ", 1);
                            }
                        }
                    }
                }
            }
        }
        out = lines.join("\n");
        out.push('\n');
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Function emission
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn emit_function_ir(
    func: &IrFunction,
    str_table: &HashMap<String, usize>,
    fn_sigs: &HashMap<String, (String, Vec<String>)>,
    module: &IrModule,
    out: &mut String,
) -> Result<(), CodegenError> {
    emit_function_ir_with_name(func, &func.name, None, str_table, fn_sigs, module, out)
}

/// Emit one function with an optional LLVM name override (e.g. "iris_main") and
/// optional entry rename so that calls to the entry are emitted with the override name.
fn emit_function_ir_with_name(
    func: &IrFunction,
    llvm_name: &str,
    entry_rename: Option<(&str, &str)>,
    str_table: &HashMap<String, usize>,
    fn_sigs: &HashMap<String, (String, Vec<String>)>,
    module: &IrModule,
    out: &mut String,
) -> Result<(), CodegenError> {
    let ret = llvm_type_complete(&func.return_ty)?;

    let cc = func.capture_count;
    let is_lambda = func.name.starts_with("__lambda_");
    let is_par_body = func.name.starts_with("__par_body_");

    // All lambdas use uniform calling convention: (ptr %env, user_params...).
    let params_str = if is_lambda {
        let mut parts = vec!["ptr %env".to_owned()];
        for p in func.params.iter().skip(cc) {
            parts.push(format!("{} %{}", llvm_type_complete(&p.ty)?, p.name));
        }
        parts.join(", ")
    } else if is_par_body {
        // `iris_par_for` invokes the body as `fn(int64_t i, void* env)`, so the
        // signature is fixed at two parameters no matter how many values the
        // body captured. The lowerer emits them as params 1.. — those are
        // unpacked from `%env` in the entry block instead. See issue 12.
        let loop_var = func
            .params
            .first()
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "__par_i".to_owned());
        format!("i64 %{}, ptr %env", loop_var)
    } else {
        let params: Result<Vec<String>, CodegenError> = func
            .params
            .iter()
            .map(|p| Ok(format!("{} %{}", llvm_type_complete(&p.ty)?, p.name)))
            .collect();
        params?.join(", ")
    };

    // Determine function attributes.
    // - Pure functions (no side effects): nounwind willreturn (helps optimizer).
    // - Complex functions (≥15 blocks): optnone noinline to work around an
    //   LLVM 17.0.1 crash in "X86 DAG→DAG Instruction Selection" that triggers
    //   on highly complex IR functions. Using optnone forces the -O0 instruction
    //   selector for these functions only, avoiding the crash with no semantic cost.
    let is_pure = func
        .blocks()
        .iter()
        .all(|b| b.instrs.iter().all(|i| !is_side_effecting(i)));
    let block_count = func.blocks().len();
    let attrs = if block_count >= 15 {
        " optnone noinline"
    } else if is_pure {
        " nounwind willreturn"
    } else {
        ""
    };

    writeln!(
        out,
        "define {} @{}({}){} {{",
        ret, llvm_name, params_str, attrs
    )?;

    // For lambdas with captures: emit capture extraction preamble.
    // (Preamble is emitted inside the entry block by emit_function_body.)

    emit_function_body(func, entry_rename, str_table, fn_sigs, module, out)?;
    writeln!(out, "}}\n")?;
    Ok(())
}

fn emit_function_body(
    func: &IrFunction,
    entry_rename: Option<(&str, &str)>,
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

    // Pre-compute blocks that contain a Panic instruction — these end with
    // `unreachable` in LLVM IR and must not appear as phi predecessors.
    let panic_blocks: HashSet<BlockId> = func
        .blocks()
        .iter()
        .filter(|b| b.instrs.iter().any(|i| matches!(i, IrInstr::Panic { .. })))
        .map(|b| b.id)
        .collect();

    // Sub-pass B: phi sources for block-param SSA → LLVM phi conversion.
    // Skip phi contributions from panic blocks (they branch nowhere reachable).
    let mut phi_src: HashMap<(BlockId, usize), Vec<(BlockId, ValueId)>> = HashMap::new();
    for block in func.blocks() {
        if panic_blocks.contains(&block.id) {
            continue; // this block ends with unreachable — skip all branch args
        }
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

    // Sub-pass C: collect AllocArray instructions that use scalar elem types.
    // These will be emitted as `alloca` at the entry block.
    let mut scalar_arrays: HashSet<ValueId> = HashSet::new();
    for block in func.blocks() {
        for instr in &block.instrs {
            if let IrInstr::AllocArray {
                result, elem_ty, ..
            } = instr
            {
                if is_scalar_type(elem_ty) {
                    scalar_arrays.insert(*result);
                }
            }
        }
    }

    // Precompute closure fn_name mapping (for distinguishing lambda from regular fn).
    // Traces through MakeClosure, GetField, and scope lookups to find the
    // original function name for any closure value.
    let mut closure_fn_map: HashMap<ValueId, String> = HashMap::new();
    // Pass 1: direct MakeClosure instructions.
    for block in func.blocks() {
        for instr in &block.instrs {
            if let IrInstr::MakeClosure { result, fn_name, .. } = instr {
                closure_fn_map.insert(*result, fn_name.clone());
            }
        }
    }
    // Pass 2: propagate through MakeStruct + GetField chains.
    // A GetField on a struct returns the field value; if the field was set
    // via MakeStruct, we can trace back to the original MakeClosure.
    let mut struct_fields: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
    for block in func.blocks() {
        for instr in &block.instrs {
            if let IrInstr::MakeStruct { result, fields, .. } = instr {
                struct_fields.insert(*result, fields.clone());
            }
        }
    }
    for block in func.blocks() {
        for instr in &block.instrs {
            if let IrInstr::GetField { result, base, field_index, .. } = instr {
                if let Some(fields) = struct_fields.get(base) {
                    if let Some(field_val) = fields.get(*field_index) {
                        if let Some(fn_name) = closure_fn_map.get(field_val) {
                            closure_fn_map.insert(*result, fn_name.clone());
                        }
                    }
                }
            }
        }
    }

    let entry_id = func.blocks()[0].id;
    let mut gep_counter: u32 = 0;

    // Sub-pass D: Build a map of emitted LLVM types for every SSA value.
    // This lets the Call handler detect type mismatches and insert coercions.
    let mut emitted_types: HashMap<ValueId, String> = HashMap::new();
    // Function parameters.
    for p in &func.params {
        if let Ok(ty_s) = llvm_type_complete(&p.ty) {
            emitted_types.insert(
                ValueId(
                    func.params
                        .iter()
                        .position(|pp| pp.name == p.name)
                        .unwrap_or(0) as u32,
                ),
                ty_s,
            );
        }
    }
    // Block parameters (phi nodes).
    for block in func.blocks() {
        for param in &block.params {
            if let Ok(ty_s) = llvm_type_complete(&param.ty) {
                emitted_types.insert(param.id, ty_s);
            }
        }
    }
    // Instructions.
    for block in func.blocks() {
        for instr in &block.instrs {
            match instr {
                IrInstr::ConstStr { result, .. } => {
                    emitted_types.insert(*result, "ptr".to_owned());
                }
                IrInstr::ConstInt { result, ty, .. } => {
                    let s = llvm_type_complete(ty).unwrap_or_else(|_| "i64".to_owned());
                    emitted_types.insert(*result, s);
                }
                IrInstr::ConstFloat { result, ty, .. } => {
                    let s = llvm_type_complete(ty).unwrap_or_else(|_| "double".to_owned());
                    emitted_types.insert(*result, s);
                }
                IrInstr::ConstBool { result, .. } => {
                    emitted_types.insert(*result, "i1".to_owned());
                }
                // Sparsify yields a sparse handle (a heap object); Densify is typed
                // i64 by the lowerer and returns an nnz count. Registering these
                // lets downstream type checks — notably the return-value cast —
                // see the real types instead of guessing.
                IrInstr::Sparsify { result, .. } => {
                    emitted_types.insert(*result, "ptr".to_owned());
                }
                IrInstr::Densify { result, .. } => {
                    emitted_types.insert(*result, "ptr".to_owned());
                }
                IrInstr::SparseNnz { result, .. } => {
                    emitted_types.insert(*result, "i64".to_owned());
                }
                IrInstr::Call {
                    result: Some(r),
                    callee,
                    result_ty,
                    ..
                } => {
                    if let Some((ret_s, _)) = fn_sigs.get(callee) {
                        emitted_types.insert(*r, ret_s.clone());
                    } else {
                        let ty_s = result_ty
                            .as_ref()
                            .and_then(|t| llvm_type_complete(t).ok())
                            .unwrap_or_else(|| "ptr".to_owned());
                        emitted_types.insert(*r, ty_s);
                    }
                }
                IrInstr::BinOp { result, ty, .. } => {
                    let s = llvm_type_complete(ty).unwrap_or_else(|_| "i64".to_owned());
                    emitted_types.insert(*result, s);
                }
                IrInstr::IsSome { result, .. } | IrInstr::IsOk { result, .. } => {
                    emitted_types.insert(*result, "i1".to_owned());
                }
                IrInstr::MakeStruct { result, .. }
                | IrInstr::MakeTuple { result, .. }
                | IrInstr::MakeClosure { result, .. }
                | IrInstr::AllocArray { result, .. }
                | IrInstr::MakeSome { result, .. }
                | IrInstr::MakeNone { result, .. }
                | IrInstr::MakeOk { result, .. }
                | IrInstr::MakeErr { result, .. } => {
                    emitted_types.insert(*result, "ptr".to_owned());
                }
                IrInstr::MakeVariant { result, fields, .. } => {
                    if fields.is_empty() {
                        emitted_types.insert(*result, "i64".to_owned());
                    } else {
                        emitted_types.insert(*result, "ptr".to_owned());
                    }
                }
                IrInstr::TapeRecord { result, .. } => {
                    emitted_types.insert(*result, "ptr".to_owned());
                }
                IrInstr::OptionUnwrap {
                    result, result_ty, ..
                }
                | IrInstr::ResultUnwrap {
                    result, result_ty, ..
                }
                | IrInstr::ResultUnwrapErr {
                    result, result_ty, ..
                } => {
                    let s = match result_ty {
                        IrType::Scalar(DType::I64) | IrType::Scalar(DType::I32) => "i64".to_owned(),
                        IrType::Scalar(DType::F64) => "double".to_owned(),
                        IrType::Scalar(DType::F32) => "float".to_owned(),
                        IrType::Scalar(DType::Bool) => "i1".to_owned(),
                        IrType::Str => "ptr".to_owned(),
                        _ => "ptr".to_owned(),
                    };
                    emitted_types.insert(*result, s);
                }
                // CallClosure: inline indirect call returns the native type.
                IrInstr::CallClosure {
                    result: Some(result),
                    result_ty,
                    ..
                } => {
                    let s = llvm_type_complete(result_ty).unwrap_or_else(|_| "ptr".to_owned());
                    emitted_types.insert(*result, s);
                }
                // ListGet/ListPop/MapGet/ChanRecv: runtime returns boxed IrisVal*,
                // we unbox to the element type, so emitted type matches the elem type.
                IrInstr::ListGet {
                    result,
                    list,
                    elem_ty,
                    ..
                }
                | IrInstr::ListPop {
                    result,
                    list,
                    elem_ty,
                    ..
                } => {
                    let resolved_elem_ty = if matches!(elem_ty, IrType::Infer) {
                        if let Some(IrType::List(inner)) = func.value_type(*list) {
                            (**inner).clone()
                        } else {
                            elem_ty.clone()
                        }
                    } else {
                        elem_ty.clone()
                    };
                    let s = match &resolved_elem_ty {
                        IrType::Scalar(DType::I64) => "i64".to_owned(),
                        IrType::Scalar(DType::I32) => "i32".to_owned(),
                        IrType::Scalar(DType::F64) => "double".to_owned(),
                        IrType::Scalar(DType::F32) => "float".to_owned(),
                        IrType::Scalar(DType::Bool) => "i1".to_owned(),
                        _ => "ptr".to_owned(),
                    };
                    emitted_types.insert(*result, s);
                }
                IrInstr::ChanRecv {
                    result, elem_ty, ..
                } => {
                    let s = match elem_ty {
                        IrType::Scalar(DType::I64) => "i64".to_owned(),
                        IrType::Scalar(DType::I32) => "i32".to_owned(),
                        IrType::Scalar(DType::F64) => "double".to_owned(),
                        IrType::Scalar(DType::F32) => "float".to_owned(),
                        IrType::Scalar(DType::Bool) => "i1".to_owned(),
                        _ => "ptr".to_owned(),
                    };
                    emitted_types.insert(*result, s);
                }
                IrInstr::MapGet { result, .. } => {
                    emitted_types.insert(*result, "ptr".to_owned());
                }
                IrInstr::GetField {
                    result, result_ty, ..
                }
                | IrInstr::GetElement {
                    result, result_ty, ..
                } => {
                    let s = llvm_type_complete(result_ty).unwrap_or_else(|_| "ptr".to_owned());
                    emitted_types.insert(*result, s);
                }
                IrInstr::UnaryOp { result, ty, .. }
                | IrInstr::Cast {
                    result, to_ty: ty, ..
                } => {
                    let s = llvm_type_complete(ty).unwrap_or_else(|_| "i64".to_owned());
                    emitted_types.insert(*result, s);
                }
                // ValueToStr always returns a string pointer.
                IrInstr::ValueToStr { result, .. } => {
                    emitted_types.insert(*result, "ptr".to_owned());
                }
                // Instructions that always return i64.
                IrInstr::StrLen { result, .. }
                | IrInstr::ListLen { result, .. }
                | IrInstr::MapLen { result, .. } => {
                    emitted_types.insert(*result, "i64".to_owned());
                }
                // Retain/Release have no result.
                _ => {
                    // For other instructions that produce a result, fall back to
                    // func.value_type if available.
                    if let Some(r) = instr_result_id(instr) {
                        if let Some(ty) = func.value_type(r) {
                            if let Ok(s) = llvm_type_complete(ty) {
                                emitted_types.insert(r, s);
                            }
                        }
                    }
                }
            }
        }
    }

    // ── Alloca preamble (entry block) ─────────────────────────────────────
    // Emit all scalar array allocas at the top of the entry block.
    let mut alloca_emitted: HashSet<ValueId> = HashSet::new();
    for block in func.blocks() {
        for instr in &block.instrs {
            if let IrInstr::AllocArray {
                result,
                elem_ty,
                size,
                ..
            } = instr
            {
                if scalar_arrays.contains(result) && !alloca_emitted.contains(result) {
                    // Will be emitted inline at the alloca instruction site.
                    alloca_emitted.insert(*result);
                    let _ = (elem_ty, size); // used below in emit_instr_ir
                }
            }
        }
    }

    // Sub-pass E: Pre-compute phi arm coercions.
    // When a phi incoming value has an emitted LLVM type that differs from
    // the phi's declared type, we insert a coercion instruction in the
    // predecessor block (before its terminator) and use the coerced name
    // in the phi node.
    let mut phi_casts: HashMap<(BlockId, usize, BlockId), String> = HashMap::new();
    let mut phi_cast_instrs: HashMap<BlockId, Vec<String>> = HashMap::new();
    for block in func.blocks() {
        if block.id == entry_id {
            continue;
        }
        for (i, param) in block.params.iter().enumerate() {
            let expected_ty = llvm_type_complete(&param.ty)?;
            if let Some(srcs) = phi_src.get(&(block.id, i)) {
                for (pred_id, v) in srcs {
                    // Skip constants – they are untyped literals in LLVM IR.
                    if consts.contains_key(v) {
                        continue;
                    }
                    if let Some(actual_ty) = emitted_types.get(v) {
                        if *actual_ty != expected_ty {
                            gep_counter += 1;
                            let vstr = llvm_val(*v, &consts, func);
                            let cast_name = format!("%phi_cast{}", gep_counter);
                            let cast_instr = if actual_ty == "ptr" && expected_ty.starts_with('i') {
                                format!(
                                    "  {} = ptrtoint ptr {} to {}",
                                    cast_name, vstr, expected_ty
                                )
                            } else if expected_ty == "ptr" && actual_ty.starts_with('i') {
                                format!("  {} = inttoptr {} {} to ptr", cast_name, actual_ty, vstr)
                            } else if actual_ty.starts_with('i') && expected_ty.starts_with('i') {
                                let aw = bit_width(actual_ty);
                                let ew = bit_width(&expected_ty);
                                if aw > ew {
                                    format!(
                                        "  {} = trunc {} {} to {}",
                                        cast_name, actual_ty, vstr, expected_ty
                                    )
                                } else {
                                    format!(
                                        "  {} = zext {} {} to {}",
                                        cast_name, actual_ty, vstr, expected_ty
                                    )
                                }
                            } else if (actual_ty == "float" || actual_ty == "double")
                                && (expected_ty == "float" || expected_ty == "double")
                            {
                                // Float widening/narrowing — use proper LLVM FP casts.
                                if actual_ty == "float" && expected_ty == "double" {
                                    format!("  {} = fpext float {} to double", cast_name, vstr)
                                } else {
                                    format!("  {} = fptrunc double {} to float", cast_name, vstr)
                                }
                            } else {
                                format!(
                                    "  {} = bitcast {} {} to {}",
                                    cast_name, actual_ty, vstr, expected_ty
                                )
                            };
                            phi_casts.insert((block.id, i, *pred_id), cast_name);
                            phi_cast_instrs
                                .entry(*pred_id)
                                .or_default()
                                .push(cast_instr);
                        }
                    }
                }
            }
        }
    }

    // Compute reachable blocks from entry so we skip dead blocks left by
    // loop unrolling (unreachable blocks may reference undefined values).
    let reachable: HashSet<BlockId> = {
        let mut visited = HashSet::new();
        let mut stack = vec![entry_id];
        while let Some(bid) = stack.pop() {
            if !visited.insert(bid) {
                continue;
            }
            if let Some(b) = func.blocks().iter().find(|b| b.id == bid) {
                for instr in &b.instrs {
                    match instr {
                        IrInstr::Br { target, .. } => {
                            stack.push(*target);
                        }
                        IrInstr::CondBr {
                            then_block,
                            else_block,
                            ..
                        } => {
                            stack.push(*then_block);
                            stack.push(*else_block);
                        }
                        IrInstr::SwitchVariant {
                            arms,
                            default_block,
                            ..
                        } => {
                            for (_, bb) in arms {
                                stack.push(*bb);
                            }
                            if let Some(def_bb) = default_block {
                                stack.push(*def_bb);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        visited
    };

    for block in func.blocks() {
        if !reachable.contains(&block.id) {
            continue;
        }
        let blabel = block_label(block.name.as_deref(), block.id);
        writeln!(out, "{}:", blabel)?;

        // Capture extraction in the entry block, for both conventions:
        //
        //   lambdas    — signature (ptr %env, args...); captures are
        //                params[0..capture_count] and live in env slot i.
        //   par bodies — signature (i64 %var, ptr %env), fixed by the runtime's
        //                `fn(int64_t, void*)`; param 0 is the loop variable and
        //                the captures are params[1..], in env slot i-1.
        //                See docs/known-issues.md issue 12.
        //
        // Both name the extracted value `%{param.name}`, which is what the rest
        // of the body already refers to.
        let cap_params: Vec<(usize, &crate::ir::function::Param)> =
            if func.name.starts_with("__par_body_") {
                func.params
                    .iter()
                    .enumerate()
                    .skip(1)
                    .map(|(i, p)| (i - 1, p))
                    .collect()
            } else if func.capture_count > 0 {
                func.params
                    .iter()
                    .take(func.capture_count)
                    .enumerate()
                    .collect()
            } else {
                Vec::new()
            };
        if block.id == entry_id && !cap_params.is_empty() {
            for (i, p) in cap_params {
                writeln!(
                    out,
                    "  %__cap_raw_{} = call ptr @iris_closure_get_capture(ptr %env, i32 {})",
                    i, i
                )?;
                match &p.ty {
                    crate::ir::types::IrType::Scalar(crate::ir::types::DType::I64)
                    | crate::ir::types::IrType::Scalar(crate::ir::types::DType::U64)
                    | crate::ir::types::IrType::Scalar(crate::ir::types::DType::USize) => {
                        writeln!(
                            out,
                            "  %{} = call i64 @iris_unbox_i64(ptr %__cap_raw_{})",
                            p.name, i
                        )?;
                    }
                    crate::ir::types::IrType::Scalar(crate::ir::types::DType::I32)
                    | crate::ir::types::IrType::Scalar(crate::ir::types::DType::U32) => {
                        writeln!(
                            out,
                            "  %__cap_i64_{} = call i64 @iris_unbox_i64(ptr %__cap_raw_{})",
                            i, i
                        )?;
                        writeln!(out, "  %{} = trunc i64 %__cap_i64_{} to i32", p.name, i)?;
                    }
                    crate::ir::types::IrType::Scalar(crate::ir::types::DType::F64) => {
                        writeln!(
                            out,
                            "  %{} = call double @iris_unbox_f64(ptr %__cap_raw_{})",
                            p.name, i
                        )?;
                    }
                    crate::ir::types::IrType::Scalar(crate::ir::types::DType::F32) => {
                        writeln!(
                            out,
                            "  %__cap_f64_{} = call double @iris_unbox_f64(ptr %__cap_raw_{})",
                            i, i
                        )?;
                        writeln!(
                            out,
                            "  %{} = fptrunc double %__cap_f64_{} to float",
                            p.name, i
                        )?;
                    }
                    crate::ir::types::IrType::Scalar(crate::ir::types::DType::Bool) => {
                        writeln!(
                            out,
                            "  %__cap_i32_{} = call i32 @iris_unbox_bool(ptr %__cap_raw_{})",
                            i, i
                        )?;
                        writeln!(out, "  %{} = trunc i32 %__cap_i32_{} to i1", p.name, i)?;
                    }
                    crate::ir::types::IrType::Str => {
                        writeln!(
                            out,
                            "  %{} = call ptr @iris_unbox_str(ptr %__cap_raw_{})",
                            p.name, i
                        )?;
                    }
                    _ => {
                        // Heap kinds are boxed on the way in (`iris_box_atomic`,
                        // `iris_box_list`, …), so they must be unboxed on the way
                        // out. Returning the box unchanged handed an IrisVal
                        // wrapper to code expecting the payload — e.g.
                        // `iris_atomic_add` received the box and dereferenced the
                        // wrong pointer. See docs/known-issues.md issue 12.
                        if let Some(unbox) = runtime_unbox_helper_for_type(&p.ty) {
                            writeln!(
                                out,
                                "  %{} = call ptr @{}(ptr %__cap_raw_{})",
                                p.name, unbox, i
                            )?;
                        } else {
                            writeln!(
                                out,
                                "  %{} = call ptr @iris_closure_get_capture(ptr %env, i32 {})",
                                p.name, i
                            )?;
                        }
                    }
                }
            }
        }

        // Phi nodes for non-entry blocks.
        if block.id != entry_id {
            for (i, param) in block.params.iter().enumerate() {
                let ty_s = llvm_type_complete(&param.ty)?;
                let phi_name = format!("%v{}", param.id.0);
                let arms: Vec<String> = phi_src
                    .get(&(block.id, i))
                    .map(|srcs| {
                        srcs.iter()
                            .filter(|(pred_id, _)| reachable.contains(pred_id))
                            .map(|(pred_id, v)| {
                                // Use the coerced value name if a phi cast was needed.
                                let vstr = if let Some(cast_name) =
                                    phi_casts.get(&(block.id, i, *pred_id))
                                {
                                    cast_name.clone()
                                } else {
                                    llvm_val(*v, &consts, func)
                                };
                                let pred = block_label_by_id(func.blocks(), *pred_id);
                                format!("[ {}, %{} ]", vstr, pred)
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                // Skip phi nodes with no arms (can happen when predecessors
                // were converted to direct returns by the lowerer).
                // Emit a dummy constant so downstream references are satisfied
                // (the block is effectively unreachable in this case).
                if arms.is_empty() {
                    if ty_s == "ptr" {
                        writeln!(
                            out,
                            "  {} = inttoptr i64 0 to ptr  ; unreachable phi stub",
                            phi_name
                        )?;
                    } else if ty_s == "double" || ty_s == "float" {
                        writeln!(
                            out,
                            "  {} = fadd {} 0.0, 0.0  ; unreachable phi stub",
                            phi_name, ty_s
                        )?;
                    } else {
                        // integer types (i1, i32, i64, etc.)
                        writeln!(
                            out,
                            "  {} = add {} 0, 0  ; unreachable phi stub",
                            phi_name, ty_s
                        )?;
                    }
                } else {
                    writeln!(out, "  {} = phi {} {}", phi_name, ty_s, arms.join(", "))?;
                }
            }
        }

        let mut panic_emitted = false; // track if we've already emitted unreachable via Panic
        for instr in &block.instrs {
            // Skip everything after Panic — unreachable is already emitted as
            // terminator. Non-terminator instructions (e.g. GC release calls
            // inserted between Panic and Return) are dead code and must not
            // appear after `unreachable`, which is invalid LLVM IR.
            if panic_emitted {
                continue;
            }
            // Emit phi coercion casts right before block terminators.
            if matches!(instr, IrInstr::Br { .. } | IrInstr::CondBr { .. }) {
                if let Some(casts) = phi_cast_instrs.get(&block.id) {
                    for cast_instr in casts {
                        writeln!(out, "{}", cast_instr)?;
                    }
                }
            }
            if matches!(instr, IrInstr::Panic { .. }) {
                panic_emitted = true;
            }
            emit_instr_ir(
                instr,
                &consts,
                func,
                entry_rename,
                fn_sigs,
                module,
                &scalar_arrays,
                &mut gep_counter,
                str_table,
                &emitted_types,
                &closure_fn_map,
                out,
            )?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Boxing helper
// ---------------------------------------------------------------------------

fn runtime_box_helper_for_type(ty: &IrType) -> Option<&'static str> {
    match ty {
        IrType::List(_) => Some("iris_box_list"),
        IrType::Map(_, _) => Some("iris_box_map"),
        IrType::Option(_) => Some("iris_box_option"),
        IrType::ResultType(_, _) => Some("iris_box_result"),
        IrType::Chan(_) => Some("iris_box_chan"),
        IrType::Atomic(_) => Some("iris_box_atomic"),
        IrType::Mutex(_) => Some("iris_box_mutex"),
        IrType::Grad(_) => Some("iris_box_grad"),
        IrType::Sparse(_) => Some("iris_box_sparse"),
        _ => None,
    }
}

fn runtime_unbox_helper_for_type(ty: &IrType) -> Option<&'static str> {
    match ty {
        IrType::Str => Some("iris_unbox_str"),
        IrType::List(_) => Some("iris_unbox_list"),
        IrType::Map(_, _) => Some("iris_unbox_map"),
        IrType::Option(_) => Some("iris_unbox_option"),
        IrType::ResultType(_, _) => Some("iris_unbox_result"),
        IrType::Chan(_) => Some("iris_unbox_chan"),
        IrType::Atomic(_) => Some("iris_unbox_atomic"),
        IrType::Mutex(_) => Some("iris_unbox_mutex"),
        IrType::Grad(_) => Some("iris_unbox_grad"),
        IrType::Sparse(_) => Some("iris_unbox_sparse"),
        _ => None,
    }
}

fn runtime_rc_kind_for_type(ty: &IrType) -> Option<i32> {
    match ty {
        IrType::Struct { .. } | IrType::Tuple(_) | IrType::Fn { .. } => Some(0),
        IrType::Str => Some(1),
        IrType::List(_) => Some(2),
        IrType::Map(_, _) => Some(3),
        IrType::Option(_) => Some(4),
        IrType::ResultType(_, _) => Some(5),
        IrType::Chan(_) => Some(6),
        IrType::Atomic(_) => Some(7),
        IrType::Mutex(_) => Some(8),
        IrType::Grad(_) => Some(9),
        IrType::Sparse(_) => Some(10),
        _ => None,
    }
}

fn runtime_box_helper_for_value(
    func: &IrFunction,
    value_id: ValueId,
    value_ty: Option<&IrType>,
) -> Option<&'static str> {
    if let Some(ty) = value_ty {
        if let Some(helper) = runtime_box_helper_for_type(ty) {
            return Some(helper);
        }
    }
    for block in func.blocks() {
        for instr in &block.instrs {
            if instr_result_id(instr) != Some(value_id) {
                continue;
            }
            return match instr {
                IrInstr::Call {
                    result_ty: Some(ty),
                    ..
                }
                | IrInstr::CallClosure { result_ty: ty, .. }
                | IrInstr::GetField { result_ty: ty, .. }
                | IrInstr::GetElement { result_ty: ty, .. }
                | IrInstr::OptionUnwrap { result_ty: ty, .. }
                | IrInstr::ResultUnwrap { result_ty: ty, .. }
                | IrInstr::ResultUnwrapErr { result_ty: ty, .. }
                | IrInstr::AtomicLoad { result_ty: ty, .. }
                | IrInstr::MutexLock { result_ty: ty, .. } => runtime_box_helper_for_type(ty),
                IrInstr::CallExtern { ret_ty, .. } => runtime_box_helper_for_type(ret_ty),
                IrInstr::ListGet { elem_ty, .. }
                | IrInstr::ListPop { elem_ty, .. }
                | IrInstr::ChanRecv { elem_ty, .. } => runtime_box_helper_for_type(elem_ty),
                IrInstr::MapGet { val_ty, .. } => runtime_box_helper_for_type(val_ty),
                IrInstr::ListNew { .. }
                | IrInstr::MapKeys { .. }
                | IrInstr::MapValues { .. }
                | IrInstr::ListConcat { .. }
                | IrInstr::ListSlice { .. }
                | IrInstr::ProcessArgs { .. } => Some("iris_box_list"),
                IrInstr::MapNew { .. } => Some("iris_box_map"),
                IrInstr::MakeSome { .. } | IrInstr::MakeNone { .. } | IrInstr::EnvVar { .. } => {
                    Some("iris_box_option")
                }
                IrInstr::MakeOk { .. } | IrInstr::MakeErr { .. } => Some("iris_box_result"),
                IrInstr::ChanNew { .. } => Some("iris_box_chan"),
                IrInstr::AtomicNew { .. } => Some("iris_box_atomic"),
                IrInstr::MutexNew { .. } => Some("iris_box_mutex"),
                IrInstr::MakeGrad { .. } => Some("iris_box_grad"),
                IrInstr::Sparsify { .. } => Some("iris_box_sparse"),
                _ => None,
            };
        }
    }
    None
}

/// If `value_ty` is a scalar type, emit a boxing call and return the resulting
/// `%boxN` ptr name. Otherwise, the value is already a ptr — return it unchanged.
fn box_to_ptr(
    out: &mut String,
    func: &IrFunction,
    value_id: ValueId,
    value_str: &str,
    value_ty: Option<&IrType>,
    emitted_ty: Option<&str>,
    counter: &mut u32,
) -> Result<String, CodegenError> {
    let inferred_ty = inferred_value_type(func, value_id, value_ty);
    let value_ty = inferred_ty.as_ref();
    let idx = *counter;
    match value_ty {
        Some(IrType::Scalar(DType::I64)) => {
            *counter += 1;
            let boxed = format!("%box{}", idx);
            writeln!(
                out,
                "  {} = call ptr @iris_box_i64(i64 {})",
                boxed, value_str
            )?;
            Ok(boxed)
        }
        Some(IrType::Scalar(DType::I32)) => {
            *counter += 1;
            let boxed = format!("%box{}", idx);
            writeln!(
                out,
                "  {} = call ptr @iris_box_i32(i32 {})",
                boxed, value_str
            )?;
            Ok(boxed)
        }
        Some(IrType::Scalar(DType::U64 | DType::USize)) => {
            *counter += 1;
            let boxed = format!("%box{}", idx);
            writeln!(
                out,
                "  {} = call ptr @iris_box_i64(i64 {})",
                boxed, value_str
            )?;
            Ok(boxed)
        }
        Some(IrType::Scalar(DType::U32)) => {
            *counter += 1;
            let widened = format!("%box_widen{}", idx);
            writeln!(out, "  {} = zext i32 {} to i64", widened, value_str)?;
            *counter += 1;
            let boxed = format!("%box{}", *counter);
            writeln!(out, "  {} = call ptr @iris_box_i64(i64 {})", boxed, widened)?;
            Ok(boxed)
        }
        Some(IrType::Scalar(DType::I8)) => {
            *counter += 1;
            let widened = format!("%box_widen{}", idx);
            writeln!(out, "  {} = sext i8 {} to i64", widened, value_str)?;
            *counter += 1;
            let boxed = format!("%box{}", *counter);
            writeln!(out, "  {} = call ptr @iris_box_i64(i64 {})", boxed, widened)?;
            Ok(boxed)
        }
        Some(IrType::Scalar(DType::U8)) => {
            *counter += 1;
            let widened = format!("%box_widen{}", idx);
            writeln!(out, "  {} = zext i8 {} to i64", widened, value_str)?;
            *counter += 1;
            let boxed = format!("%box{}", *counter);
            writeln!(out, "  {} = call ptr @iris_box_i64(i64 {})", boxed, widened)?;
            Ok(boxed)
        }
        Some(IrType::Scalar(DType::F64)) => {
            *counter += 1;
            let boxed = format!("%box{}", idx);
            writeln!(
                out,
                "  {} = call ptr @iris_box_f64(double {})",
                boxed, value_str
            )?;
            Ok(boxed)
        }
        Some(IrType::Scalar(DType::F32)) => {
            *counter += 1;
            let boxed = format!("%box{}", idx);
            writeln!(
                out,
                "  {} = call ptr @iris_box_f32(float {})",
                boxed, value_str
            )?;
            Ok(boxed)
        }
        Some(IrType::Scalar(DType::Bool)) => {
            *counter += 1;
            let boxed = format!("%box{}", idx);
            writeln!(
                out,
                "  {} = call ptr @iris_box_bool(i1 {})",
                boxed, value_str
            )?;
            Ok(boxed)
        }
        Some(IrType::Str) => {
            *counter += 1;
            let boxed = format!("%box{}", idx);
            writeln!(
                out,
                "  {} = call ptr @iris_box_str(ptr {})",
                boxed, value_str
            )?;
            Ok(boxed)
        }
        Some(ty) if runtime_box_helper_for_type(ty).is_some() => {
            *counter += 1;
            let boxed = format!("%box{}", idx);
            writeln!(
                out,
                "  {} = call ptr @{}(ptr {})",
                boxed,
                runtime_box_helper_for_type(ty).unwrap(),
                value_str
            )?;
            Ok(boxed)
        }
        _ if runtime_box_helper_for_value(func, value_id, value_ty).is_some() => {
            *counter += 1;
            let boxed = format!("%box{}", idx);
            writeln!(
                out,
                "  {} = call ptr @{}(ptr {})",
                boxed,
                runtime_box_helper_for_value(func, value_id, value_ty).unwrap(),
                value_str
            )?;
            Ok(boxed)
        }
        _ => match emitted_ty {
            Some("i64") => {
                *counter += 1;
                let boxed = format!("%box{}", idx);
                writeln!(
                    out,
                    "  {} = call ptr @iris_box_i64(i64 {})",
                    boxed, value_str
                )?;
                Ok(boxed)
            }
            Some("i32") => {
                *counter += 1;
                let boxed = format!("%box{}", idx);
                writeln!(
                    out,
                    "  {} = call ptr @iris_box_i32(i32 {})",
                    boxed, value_str
                )?;
                Ok(boxed)
            }
            Some("double") => {
                *counter += 1;
                let boxed = format!("%box{}", idx);
                writeln!(
                    out,
                    "  {} = call ptr @iris_box_f64(double {})",
                    boxed, value_str
                )?;
                Ok(boxed)
            }
            Some("float") => {
                *counter += 1;
                let boxed = format!("%box{}", idx);
                writeln!(
                    out,
                    "  {} = call ptr @iris_box_f32(float {})",
                    boxed, value_str
                )?;
                Ok(boxed)
            }
            Some("i1") => {
                *counter += 1;
                let boxed = format!("%box{}", idx);
                writeln!(
                    out,
                    "  {} = call ptr @iris_box_bool(i1 {})",
                    boxed, value_str
                )?;
                Ok(boxed)
            }
            Some("i8") => {
                *counter += 1;
                let widened = format!("%box_widen{}", idx);
                writeln!(out, "  {} = zext i8 {} to i64", widened, value_str)?;
                *counter += 1;
                let boxed = format!("%box{}", *counter);
                writeln!(out, "  {} = call ptr @iris_box_i64(i64 {})", boxed, widened)?;
                Ok(boxed)
            }
            _ => {
                // Not a scalar — already a ptr. No boxing needed.
                if emitted_ty == Some("ptr") && value_ty != Some(&IrType::Infer) {
                    // Just debug the type
                    // println!("box_to_ptr fallback for type {:?}", value_ty);
                }
                let _ = value_ty;
                Ok(value_str.to_owned())
            }
        },
    }
}

/// Unbox a raw `IrisVal*` ptr into the result register `%v{result_id}`.
/// For non-scalar types (or when no unbox function exists), keeps it as ptr.
fn unbox_ptr_to_result(
    out: &mut String,
    raw: String, // e.g. "%raw_ouw0"
    result_id: u32,
    result_ty: &IrType,
    counter: &mut u32,
) -> Result<(), CodegenError> {
    match result_ty {
        IrType::Scalar(DType::I64) | IrType::Scalar(DType::U64 | DType::USize) => {
            writeln!(
                out,
                "  %v{} = call i64 @iris_unbox_i64(ptr {})",
                result_id, raw
            )?;
        }
        IrType::Scalar(DType::I32 | DType::U32) => {
            let tmp = format!("%unbox_i64_for_i32_{}", counter);
            *counter += 1;
            writeln!(out, "  {} = call i64 @iris_unbox_i64(ptr {})", tmp, raw)?;
            writeln!(out, "  %v{} = trunc i64 {} to i32", result_id, tmp)?;
        }
        IrType::Scalar(DType::I8 | DType::U8) => {
            let tmp = format!("%unbox_i64_for_i8_{}", counter);
            *counter += 1;
            writeln!(out, "  {} = call i64 @iris_unbox_i64(ptr {})", tmp, raw)?;
            writeln!(out, "  %v{} = trunc i64 {} to i8", result_id, tmp)?;
        }
        IrType::Scalar(DType::F64) => {
            writeln!(
                out,
                "  %v{} = call double @iris_unbox_f64(ptr {})",
                result_id, raw
            )?;
        }
        IrType::Scalar(DType::F32) => {
            // No iris_unbox_f32 — unbox as f64 then truncate.
            let tmp = format!("%unbox_f64_for_f32_{}", counter);
            *counter += 1;
            writeln!(out, "  {} = call double @iris_unbox_f64(ptr {})", tmp, raw)?;
            writeln!(out, "  %v{} = fptrunc double {} to float", result_id, tmp)?;
        }
        IrType::Scalar(DType::Bool) => {
            let tmp = format!("%unbox_bool{}", counter);
            *counter += 1;
            writeln!(out, "  {} = call i32 @iris_unbox_bool(ptr {})", tmp, raw)?;
            writeln!(out, "  %v{} = trunc i32 {} to i1", result_id, tmp)?;
        }
        IrType::Str => {
            writeln!(
                out,
                "  %v{} = call ptr @iris_unbox_str(ptr {})",
                result_id, raw
            )?;
        }
        ty if runtime_unbox_helper_for_type(ty).is_some() => {
            writeln!(
                out,
                "  %v{} = call ptr @{}(ptr {})",
                result_id,
                runtime_unbox_helper_for_type(ty).unwrap(),
                raw
            )?;
        }
        _ => {
            // Non-scalar — already a ptr, use a zero-offset GEP to name it.
            writeln!(
                out,
                "  %v{} = getelementptr i8, ptr {}, i32 0",
                result_id, raw
            )?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Type coercion helper — emit ptrtoint / inttoptr / trunc / zext as needed
// ---------------------------------------------------------------------------

/// If the value's emitted LLVM type differs from `expected_ty`, emit a coercion
/// instruction and return the coerced name. Otherwise return the raw value string.
fn coerce_to_type(
    v: ValueId,
    expected_ty: &str,
    consts: &HashMap<ValueId, String>,
    func: &IrFunction,
    emitted_types: &HashMap<ValueId, String>,
    gep_counter: &mut u32,
    out: &mut String,
) -> Result<String, CodegenError> {
    let v_str = llvm_val(v, consts, func);
    // Constants: when used in an expected type context we may need to
    // materialize a properly-typed temporary. In particular, LLVM expects
    // `float` constants to be written as hexadecimal float literals; so
    // for inline constants used as `float` we emit a small tmp holding the
    // properly-typed float literal.
    if consts.contains_key(&v) {
        if expected_ty == "float" {
            // Parse stored decimal and produce an f32 hex literal.
            let dv = consts[&v].clone();
            if let Ok(_d) = dv.parse::<f64>() {
                *gep_counter += 1;
                let tmp = format!("%coerce{}", gep_counter);
                // Use fptrunc from a double literal to produce a correctly-typed
                // float value. Decimal double literals are accepted by clang.
                writeln!(out, "  {} = fptrunc double {} to float", tmp, dv)?;
                return Ok(tmp);
            }
            return Ok(v_str);
        }
        return Ok(v_str);
    }
    if let Some(actual_ty) = emitted_types.get(&v) {
        if actual_ty != expected_ty {
            *gep_counter += 1;
            let tmp = format!("%coerce{}", gep_counter);
            // Two sources describe this value's type and they can disagree:
            // `emitted_types` records the LLVM type actually written for the SSA
            // value, while `llvm_type_complete(value_type)` records what the IR
            // type says it should be. The IR type is preferred here because it
            // carries the authoritative float *width* (emitting
            // `fptosi float %x` for a value that is really a double is rejected).
            //
            // But it must not be preferred across the integer/pointer boundary.
            // A tag-only enum is emitted as `i64` while its IrType maps to `ptr`,
            // and trusting the IR type there selected the bitcast fallback,
            // emitting `bitcast ptr %v14 to ptr` for a value that is an i64 —
            // which clang rejects with
            //   '%v14' defined with type 'i64' but expected 'ptr'
            // That is known-issues #2, and it is the same emitted-vs-declared
            // disagreement as #6. When the two disagree about int-vs-pointer, the
            // emitted form is the truth about the operand.
            let actual_ty_str = if let Some(ty) = func.value_type(v) {
                let from_ir = llvm_type_complete(ty).unwrap_or(actual_ty.clone());
                if (from_ir == "ptr") != (actual_ty == "ptr") {
                    actual_ty.clone()
                } else {
                    from_ir
                }
            } else {
                actual_ty.clone()
            };
            if actual_ty_str == "ptr" && expected_ty.starts_with('i') {
                writeln!(out, "  {} = ptrtoint ptr {} to {}", tmp, v_str, expected_ty)?;
            } else if expected_ty == "ptr" && actual_ty_str.starts_with('i') {
                writeln!(out, "  {} = inttoptr {} {} to ptr", tmp, actual_ty, v_str)?;
            } else if actual_ty_str.starts_with('i') && expected_ty.starts_with('i') {
                let src_w = bit_width(&actual_ty_str);
                let dst_w = bit_width(expected_ty);
                if src_w == dst_w {
                    return Ok(v_str);
                }
                let zero_extend_src = matches!(
                    func.value_type(v),
                    Some(IrType::Scalar(DType::U8 | DType::U32 | DType::U64 | DType::USize | DType::Bool))
                );
                let op = if src_w > dst_w {
                    "trunc"
                } else if zero_extend_src {
                    "zext"
                } else {
                    "sext"
                };
                writeln!(
                    out,
                    "  {} = {} {} {} to {}",
                    tmp, op, actual_ty_str, v_str, expected_ty
                )?;
            } else if (actual_ty == "float" || actual_ty == "double")
                && expected_ty.starts_with('i')
            {
                if expected_ty == "i1" {
                    let zero = "0.0";
                    let cmp = if actual_ty == "float" {
                        format!("fcmp one float {}, {}", v_str, zero)
                    } else {
                        format!("fcmp one double {}, {}", v_str, zero)
                    };
                    writeln!(out, "  {} = {}", tmp, cmp)?;
                } else {
                    // Ensure the operand is represented with the authoritative
                    // FP width (`actual_ty_str`) before performing the
                    // float->int conversion. This avoids emitting e.g.
                    // `fptosi float %x to i64` when `%x` is actually a
                    // `double` value, which clang rejects.
                    let v_use = if actual_ty != &actual_ty_str {
                        *gep_counter += 1;
                        let tmp2 = format!("%coerce{}", *gep_counter);
                        if actual_ty == "double" && actual_ty_str == "float" {
                            writeln!(out, "  {} = fptrunc double {} to float", tmp2, v_str)?;
                        } else if actual_ty == "float" && actual_ty_str == "double" {
                            writeln!(out, "  {} = fpext float {} to double", tmp2, v_str)?;
                        } else {
                            writeln!(
                                out,
                                "  {} = bitcast {} {} to {}",
                                tmp2, actual_ty, v_str, actual_ty_str
                            )?;
                        }
                        tmp2
                    } else {
                        v_str
                    };
                    writeln!(
                        out,
                        "  {} = fptosi {} {} to {}",
                        tmp, actual_ty_str, v_use, expected_ty
                    )?;
                }
            } else if actual_ty.starts_with('i')
                && (expected_ty == "float" || expected_ty == "double")
            {
                writeln!(
                    out,
                    "  {} = sitofp {} {} to {}",
                    tmp, actual_ty_str, v_str, expected_ty
                )?;
            } else {
                writeln!(
                    out,
                    "  {} = bitcast {} {} to {}",
                    tmp, actual_ty_str, v_str, expected_ty
                )?;
            }
            return Ok(tmp);
        }
    }
    Ok(v_str)
}

fn coerce_scalar_to_f64(
    v: ValueId,
    consts: &HashMap<ValueId, String>,
    func: &IrFunction,
    gep_counter: &mut u32,
    out: &mut String,
) -> Result<String, CodegenError> {
    let v_str = llvm_val(v, consts, func);
    match func.value_type(v) {
        Some(IrType::Scalar(DType::F64)) => Ok(v_str),
        Some(IrType::Scalar(DType::F32)) => {
            *gep_counter += 1;
            let tmp = format!("%coerce_f64{}", gep_counter);
            if consts.contains_key(&v) {
                writeln!(out, "  {} = fadd double {}, 0.0", tmp, v_str)?;
            } else {
                writeln!(out, "  {} = fpext float {} to double", tmp, v_str)?;
            }
            Ok(tmp)
        }
        Some(IrType::Scalar(DType::I64)) => {
            *gep_counter += 1;
            let tmp = format!("%coerce_f64{}", gep_counter);
            writeln!(out, "  {} = sitofp i64 {} to double", tmp, v_str)?;
            Ok(tmp)
        }
        Some(IrType::Scalar(DType::I32)) => {
            *gep_counter += 1;
            let tmp = format!("%coerce_f64{}", gep_counter);
            writeln!(out, "  {} = sitofp i32 {} to double", tmp, v_str)?;
            Ok(tmp)
        }
        Some(IrType::Scalar(DType::I8)) => {
            *gep_counter += 1;
            let tmp = format!("%coerce_f64{}", gep_counter);
            writeln!(out, "  {} = sitofp i8 {} to double", tmp, v_str)?;
            Ok(tmp)
        }
        Some(IrType::Scalar(DType::U8)) => {
            *gep_counter += 1;
            let tmp = format!("%coerce_f64{}", gep_counter);
            writeln!(out, "  {} = uitofp i8 {} to double", tmp, v_str)?;
            Ok(tmp)
        }
        Some(IrType::Scalar(DType::U32)) => {
            *gep_counter += 1;
            let tmp = format!("%coerce_f64{}", gep_counter);
            writeln!(out, "  {} = uitofp i32 {} to double", tmp, v_str)?;
            Ok(tmp)
        }
        Some(IrType::Scalar(DType::U64 | DType::USize)) => {
            *gep_counter += 1;
            let tmp = format!("%coerce_f64{}", gep_counter);
            writeln!(out, "  {} = uitofp i64 {} to double", tmp, v_str)?;
            Ok(tmp)
        }
        Some(other) => Err(CodegenError::Unsupported {
            backend: "llvm".into(),
            detail: format!("reverse-mode AD expects scalar parents, got {}", other),
        }),
        None => Err(CodegenError::Unsupported {
            backend: "llvm".into(),
            detail: format!(
                "reverse-mode AD could not determine the type of value {}",
                v.0
            ),
        }),
    }
}

fn emit_zero_value(
    result: ValueId,
    emitted_types: &HashMap<ValueId, String>,
    out: &mut String,
) -> Result<(), CodegenError> {
    match emitted_types
        .get(&result)
        .map(|s| s.as_str())
        .unwrap_or("i1")
    {
        "double" => writeln!(out, "  %v{} = fadd double 0.0, 0.0", result.0)?,
        "float" => writeln!(out, "  %v{} = fadd float 0.0, 0.0", result.0)?,
        "ptr" => writeln!(out, "  %v{} = inttoptr i64 0 to ptr", result.0)?,
        ty if ty.starts_with('i') => writeln!(out, "  %v{} = add {} 0, 0", result.0, ty)?,
        _ => writeln!(out, "  %v{} = add i1 0, 0", result.0)?,
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Instruction emission
// ---------------------------------------------------------------------------

/// Check if an einsum notation represents a standard matmul pattern.
/// Matches: "mk,kn->mn", "ij,jk->ik", or any 2-char,2-char->2-char where
/// the inner index is contracted.
pub(crate) fn is_matmul_notation(notation: &str) -> bool {
    let parts: Vec<&str> = notation.split("->").collect();
    if parts.len() != 2 {
        return false;
    }
    let inputs: Vec<&str> = parts[0].split(',').collect();
    if inputs.len() != 2 {
        return false;
    }
    let lhs: Vec<char> = inputs[0].chars().collect();
    let rhs: Vec<char> = inputs[1].chars().collect();
    let out: Vec<char> = parts[1].chars().collect();
    // Standard matmul: lhs=[a,b], rhs=[b,c], out=[a,c]
    lhs.len() == 2
        && rhs.len() == 2
        && out.len() == 2
        && lhs[1] == rhs[0]
        && out[0] == lhs[0]
        && out[1] == rhs[1]
}

#[allow(clippy::too_many_arguments)]
fn emit_instr_ir(
    instr: &IrInstr,
    consts: &HashMap<ValueId, String>,
    func: &IrFunction,
    entry_rename: Option<(&str, &str)>,
    fn_sigs: &HashMap<String, (String, Vec<String>)>,
    module: &IrModule,
    scalar_arrays: &HashSet<ValueId>,
    gep_counter: &mut u32,
    str_table: &HashMap<String, usize>,
    emitted_types: &HashMap<ValueId, String>,
    closure_fn_map: &HashMap<ValueId, String>,
    out: &mut String,
) -> Result<(), CodegenError> {
    let val = |v: ValueId| llvm_val(v, consts, func);

    match instr {
        // Constants are inlined at use sites.
        IrInstr::ConstFloat { .. } | IrInstr::ConstInt { .. } | IrInstr::ConstBool { .. } => {}

        IrInstr::BinOp {
            result,
            op,
            lhs,
            rhs,
            ty,
        } => {
            let comparison_op = matches!(
                op,
                BinOp::CmpEq
                    | BinOp::CmpNe
                    | BinOp::CmpLt
                    | BinOp::CmpLe
                    | BinOp::CmpGt
                    | BinOp::CmpGe
            );
            let semantic_operand_ty = func.value_type(*lhs).or_else(|| func.value_type(*rhs));
            // String comparison: use iris_str_eq / iris_str_cmp (pointer comparison is wrong).
            let lhs_ety = emitted_types.get(lhs).map(|s| s.as_str());
            let rhs_ety = emitted_types.get(rhs).map(|s| s.as_str());
            // Aggregates are also emitted as `ptr`, so "both operands are ptr"
            // is not evidence of a string. Treating it as one sent struct
            // pointers into `iris_str_eq`, which `strcmp`s raw struct memory:
            // `P { x: 1, y: 2 }` and `P { x: 1, y: 3 }` both read as the
            // one-byte string "" (the first field, then its zero padding)
            // and compared **equal**. A silent wrong answer, and the opposite
            // of the truth. See known-issues #26.
            let is_aggregate = matches!(
                semantic_operand_ty,
                Some(IrType::Struct { .. })
                    | Some(IrType::Tuple(_))
                    | Some(IrType::Array { .. })
                    | Some(IrType::List(_))
                    | Some(IrType::Map(_, _))
                    | Some(IrType::Option(_))
                    | Some(IrType::ResultType(_, _))
            );
            let is_str_cmp = semantic_operand_ty == Some(&IrType::Str)
                || (!is_aggregate
                    && lhs_ety == Some("ptr")
                    && rhs_ety == Some("ptr")
                    && matches!(op, BinOp::CmpEq | BinOp::CmpNe));

            // Structural equality for records, field by field.
            if matches!(op, BinOp::CmpEq | BinOp::CmpNe) {
                if let Some(sty @ IrType::Struct { .. }) = semantic_operand_ty {
                    let lv = coerce_to_type(
                        *lhs, "ptr", consts, func, emitted_types, gep_counter, out,
                    )?;
                    let rv = coerce_to_type(
                        *rhs, "ptr", consts, func, emitted_types, gep_counter, out,
                    )?;
                    let eq = emit_struct_eq(sty, &lv, &rv, gep_counter, out)?;
                    if *op == BinOp::CmpEq {
                        writeln!(out, "  %v{} = and i1 {}, true", result.0, eq)?;
                    } else {
                        writeln!(out, "  %v{} = xor i1 {}, true", result.0, eq)?;
                    }
                    // Result type is recorded by the shared comparison path,
                    // same as every other `i1`-producing compare.
                    return Ok(());
                }
            }
            if is_str_cmp && matches!(op, BinOp::CmpEq | BinOp::CmpNe) {
                let lv =
                    coerce_to_type(*lhs, "ptr", consts, func, emitted_types, gep_counter, out)?;
                let rv =
                    coerce_to_type(*rhs, "ptr", consts, func, emitted_types, gep_counter, out)?;
                if *op == BinOp::CmpEq {
                    writeln!(
                        out,
                        "  %v{} = call i1 @iris_str_eq(ptr {}, ptr {})",
                        result.0, lv, rv
                    )?;
                } else {
                    let tmp = format!("%str_eq_tmp{}", gep_counter);
                    *gep_counter += 1;
                    writeln!(
                        out,
                        "  {} = call i1 @iris_str_eq(ptr {}, ptr {})",
                        tmp, lv, rv
                    )?;
                    writeln!(out, "  %v{} = xor i1 {}, true", result.0, tmp)?;
                }
                // skip the rest of BinOp handling
            } else if is_str_cmp && matches!(op, BinOp::CmpLt | BinOp::CmpLe | BinOp::CmpGt | BinOp::CmpGe) {
                let lv =
                    coerce_to_type(*lhs, "ptr", consts, func, emitted_types, gep_counter, out)?;
                let rv =
                    coerce_to_type(*rhs, "ptr", consts, func, emitted_types, gep_counter, out)?;
                let cmp_tmp = format!("%str_cmp_tmp{}", gep_counter);
                *gep_counter += 1;
                writeln!(
                    out,
                    "  {} = call i64 @iris_str_cmp(ptr {}, ptr {})",
                    cmp_tmp, lv, rv
                )?;
                let llvm_pred = match op {
                    BinOp::CmpLt => "slt",
                    BinOp::CmpLe => "sle",
                    BinOp::CmpGt => "sgt",
                    BinOp::CmpGe => "sge",
                    _ => unreachable!(),
                };
                writeln!(
                    out,
                    "  %v{} = icmp {} i64 {}, 0",
                    result.0, llvm_pred, cmp_tmp
                )?;
            } else {
                let ty_s = if comparison_op {
                    lhs_ety
                        .or(rhs_ety)
                        .map(str::to_owned)
                        .or_else(|| {
                            semantic_operand_ty
                                .and_then(|operand_ty| llvm_type_complete(operand_ty).ok())
                        })
                        .unwrap_or_else(|| "i1".to_owned())
                } else {
                    llvm_type_complete(semantic_operand_ty.unwrap_or(ty))?
                };
                // Coerce both operands to the expected type.
                let lv =
                    coerce_to_type(*lhs, &ty_s, consts, func, emitted_types, gep_counter, out)?;
                let rv =
                    coerce_to_type(*rhs, &ty_s, consts, func, emitted_types, gep_counter, out)?;
                // Arithmetic on a boxed (pointer-represented) operand has no valid
                // LLVM form: `mul ptr %a, %b` is not an instruction. It arose for
                // types whose values are heap objects rather than machine scalars
                // — `grad<T>` dual numbers and sparse tensors — where the native
                // backend has no arithmetic helpers at all, only accessors like
                // iris_grad_value. Previously this emitted invalid IR and clang
                // killed the process.
                //
                // Reporting Unsupported instead is both honest and useful: the
                // eval path falls back to the interpreter on a codegen error, and
                // the interpreter does implement dual-number arithmetic, so these
                // programs now produce the right answer rather than crashing.
                // Comparisons are excluded — `icmp` on pointers is legal.
                if ty_s == "ptr"
                    && matches!(
                        op,
                        BinOp::Add
                            | BinOp::Sub
                            | BinOp::Mul
                            | BinOp::Div
                            | BinOp::FloorDiv
                            | BinOp::Mod
                            | BinOp::Pow
                    )
                {
                    return Err(CodegenError::Unsupported {
                        backend: "native".into(),
                        detail: format!(
                            "arithmetic ({:?}) on a heap-represented value is not implemented by \
                             the native backend — types such as grad<T> and sparse tensors have \
                             no arithmetic runtime helpers. Use the interpreter (--emit eval).",
                            op
                        ),
                    });
                }

                let is_float = ty_s == "float" || ty_s == "double";
                let llvm_op = match (op, is_float) {
                    (BinOp::Add, true) => format!("fadd {} {}, {}", ty_s, lv, rv),
                    (BinOp::Sub, true) => format!("fsub {} {}, {}", ty_s, lv, rv),
                    (BinOp::Mul, true) => format!("fmul {} {}, {}", ty_s, lv, rv),
                    (BinOp::Div, true) => format!("fdiv {} {}, {}", ty_s, lv, rv),
                    // The overflow-checked runtime helpers are i64-only. They were
                    // being used for every integer width, emitting
                    // `call i64 @iris_add_checked(i64 %a, ...)` where `%a` was an
                    // i32 or i8 parameter and assigning the i64 result to a
                    // narrower value — invalid IR that clang rejected, so any
                    // program doing arithmetic on i32/i8/u8/u32 failed to build.
                    //
                    // Narrower widths therefore use native LLVM arithmetic. Plain
                    // wrapping, without nsw/nuw: those assert no-wrap and make
                    // overflow undefined behaviour, which would be a worse answer
                    // than the wrap. Limit to state plainly: overflow is checked
                    // at i64 and wraps at narrower widths.
                    (BinOp::Add, false) if ty_s == "i64" => {
                        format!("call i64 @iris_add_checked(i64 {}, i64 {})", lv, rv)
                    }
                    (BinOp::Sub, false) if ty_s == "i64" => {
                        format!("call i64 @iris_sub_checked(i64 {}, i64 {})", lv, rv)
                    }
                    (BinOp::Mul, false) if ty_s == "i64" => {
                        format!("call i64 @iris_mul_checked(i64 {}, i64 {})", lv, rv)
                    }
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
                    (BinOp::Pow, true) => format!(
                        "call {} @llvm.pow.f64({} {}, {} {})",
                        ty_s, ty_s, lv, ty_s, rv
                    ),
                    (BinOp::Pow, false) => {
                        format!("call i64 @iris_pow_i64(i64 {}, i64 {})", lv, rv)
                    }
                    (BinOp::Min, true) => format!(
                        "call {} @llvm.minnum.f64({} {}, {} {})",
                        ty_s, ty_s, lv, ty_s, rv
                    ),
                    (BinOp::Min, false) => {
                        format!("call i64 @iris_min_i64(i64 {}, i64 {})", lv, rv)
                    }
                    (BinOp::Max, true) => format!(
                        "call {} @llvm.maxnum.f64({} {}, {} {})",
                        ty_s, ty_s, lv, ty_s, rv
                    ),
                    (BinOp::Max, false) => {
                        format!("call i64 @iris_max_i64(i64 {}, i64 {})", lv, rv)
                    }
                    (BinOp::BitAnd, false) => format!("and {} {}, {}", ty_s, lv, rv),
                    (BinOp::BitOr, false) => format!("or {} {}, {}", ty_s, lv, rv),
                    (BinOp::BitXor, false) => format!("xor {} {}, {}", ty_s, lv, rv),
                    (BinOp::Shl, false) => format!("shl {} {}, {}", ty_s, lv, rv),
                    (BinOp::Shr, false) => format!("ashr {} {}, {}", ty_s, lv, rv),
                    _ => {
                        return Err(CodegenError::Unsupported {
                            backend: "llvm".into(),
                            detail: format!(
                                "unsupported binary operation {:?} (float={})",
                                op, is_float
                            ),
                        });
                    }
                };
                writeln!(out, "  %v{} = {}", result.0, llvm_op)?;
            } // end else (non-string-comparison BinOp)
        }

        IrInstr::UnaryOp {
            result,
            op,
            operand,
            ty,
        } => {
            let ty_s = llvm_type_complete(ty)?;
            let ov = coerce_to_type(
                *operand,
                &ty_s,
                consts,
                func,
                emitted_types,
                gep_counter,
                out,
            )?;
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
                    writeln!(
                        out,
                        "  %v{} = call {} @llvm.sqrt.f64({} {})",
                        result.0, ty_s, ty_s, ov
                    )?;
                }
                ScalarUnaryOp::Abs if is_float => {
                    writeln!(
                        out,
                        "  %v{} = call {} @llvm.fabs.f64({} {})",
                        result.0, ty_s, ty_s, ov
                    )?;
                }
                ScalarUnaryOp::Abs => {
                    writeln!(out, "  %v{} = call i64 @iris_abs_i64(i64 {})", result.0, ov)?;
                }
                ScalarUnaryOp::Floor => {
                    writeln!(
                        out,
                        "  %v{} = call {} @llvm.floor.f64({} {})",
                        result.0, ty_s, ty_s, ov
                    )?;
                }
                ScalarUnaryOp::Ceil => {
                    writeln!(
                        out,
                        "  %v{} = call {} @llvm.ceil.f64({} {})",
                        result.0, ty_s, ty_s, ov
                    )?;
                }
                ScalarUnaryOp::BitNot => {
                    writeln!(out, "  %v{} = xor {} {}, -1", result.0, ty_s, ov)?;
                }
                ScalarUnaryOp::Sin => {
                    writeln!(
                        out,
                        "  %v{} = call {} @llvm.sin.f64({} {})",
                        result.0, ty_s, ty_s, ov
                    )?;
                }
                ScalarUnaryOp::Cos => {
                    writeln!(
                        out,
                        "  %v{} = call {} @llvm.cos.f64({} {})",
                        result.0, ty_s, ty_s, ov
                    )?;
                }
                ScalarUnaryOp::Tan => {
                    writeln!(out, "  %v{} = call double @tan(double {})", result.0, ov)?;
                }
                ScalarUnaryOp::Exp => {
                    writeln!(
                        out,
                        "  %v{} = call {} @llvm.exp.f64({} {})",
                        result.0, ty_s, ty_s, ov
                    )?;
                }
                ScalarUnaryOp::Log => {
                    writeln!(
                        out,
                        "  %v{} = call {} @llvm.log.f64({} {})",
                        result.0, ty_s, ty_s, ov
                    )?;
                }
                ScalarUnaryOp::Log2 => {
                    writeln!(
                        out,
                        "  %v{} = call {} @llvm.log2.f64({} {})",
                        result.0, ty_s, ty_s, ov
                    )?;
                }
                ScalarUnaryOp::Round => {
                    // Prefer standard C `round`/`roundf` functions instead of
                    // LLVM intrinsics to avoid linker issues with intrinsics.
                    if ty_s == "double" {
                        writeln!(out, "  %v{} = call double @round(double {})", result.0, ov)?;
                    } else if ty_s == "float" {
                        writeln!(out, "  %v{} = call float @roundf(float {})", result.0, ov)?;
                    } else {
                        writeln!(
                            out,
                            "  %v{} = call {} @llvm.round.f64({} {})",
                            result.0, ty_s, ty_s, ov
                        )?;
                    }
                }
                ScalarUnaryOp::Sign => {
                    writeln!(
                        out,
                        "  %v{} = call double @iris_sign_f64(double {})",
                        result.0, ov
                    )?;
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
            let from_s = llvm_type_complete(from_ty)?;
            let to_s = llvm_type_complete(to_ty)?;
            // If we know the emitted LLVM type for the operand, prefer that
            // when emitting cast instructions — this prevents generating
            // ops that use the wrong floating-point width for the operand.
            let mut actual_from_s = emitted_types
                .get(operand)
                .cloned()
                .unwrap_or_else(|| from_s.clone());
            // Prefer the authoritative IR value type when available to avoid
            // emitting casts with the wrong source width (e.g., using
            // `float` when the actual value is `double`).
            if let Some(fty) = func.value_type(*operand) {
                if let Ok(s) = llvm_type_complete(fty) {
                    actual_from_s = s;
                }
            }
            let is_from_float = matches!(from_ty, IrType::Scalar(DType::F32 | DType::F64));
            let is_to_float = matches!(to_ty, IrType::Scalar(DType::F32 | DType::F64));
            let is_from_int = matches!(
                from_ty,
                IrType::Scalar(
                    DType::I8
                        | DType::U8
                        | DType::I32
                        | DType::U32
                        | DType::I64
                        | DType::U64
                        | DType::USize
                        | DType::Bool
                )
            );
            let is_to_int = matches!(
                to_ty,
                IrType::Scalar(
                    DType::I8
                        | DType::U8
                        | DType::I32
                        | DType::U32
                        | DType::I64
                        | DType::U64
                        | DType::USize
                        | DType::Bool
                )
            );
            let is_from_f64 = matches!(from_ty, IrType::Scalar(DType::F64));
            let is_to_f64 = matches!(to_ty, IrType::Scalar(DType::F64));
            if from_ty == to_ty {
                writeln!(
                    out,
                    "  %v{} = bitcast {} {} to {}",
                    result.0, from_s, ov, to_s
                )?;
            } else if is_from_float && is_to_int {
                let ov_coerced = coerce_to_type(
                    *operand,
                    &from_s,
                    consts,
                    func,
                    emitted_types,
                    gep_counter,
                    out,
                )?;
                writeln!(
                    out,
                    "  %v{} = fptosi {} {} to {}",
                    result.0, from_s, ov_coerced, to_s
                )?;
            } else if is_from_int && is_to_float {
                let ov_coerced = coerce_to_type(
                    *operand,
                    &from_s,
                    consts,
                    func,
                    emitted_types,
                    gep_counter,
                    out,
                )?;
                writeln!(
                    out,
                    "  %v{} = sitofp {} {} to {}",
                    result.0, from_s, ov_coerced, to_s
                )?;
            } else if is_from_float && is_to_float {
                if !is_from_f64 && is_to_f64 {
                    // If the operand is an inline constant, emit it directly
                    // as a double literal.  LLVM does not accept decimal float
                    // literals that are not exactly representable in f32.
                    if consts.contains_key(operand) {
                        let dv = consts[operand].clone();
                        writeln!(out, "  %v{} = fadd double {}, 0.0", result.0, dv)?;
                    } else {
                        // Use the actual operand type when emitting the FP
                        // extension instruction.
                        if actual_from_s == "float" && to_s == "double" {
                            writeln!(out, "  %v{} = fpext float {} to double", result.0, ov)?;
                        } else {
                            writeln!(
                                out,
                                "  %v{} = fpext {} {} to {}",
                                result.0, actual_from_s, ov, to_s
                            )?;
                        }
                    }
                } else {
                    writeln!(
                        out,
                        "  %v{} = fptrunc {} {} to {}",
                        result.0, from_s, ov, to_s
                    )?;
                }
            } else if is_from_int && is_to_int {
                let ov_coerced = coerce_to_type(
                    *operand,
                    &from_s,
                    consts,
                    func,
                    emitted_types,
                    gep_counter,
                    out,
                )?;
                let src_w = bit_width(&from_s);
                let dst_w = bit_width(&to_s);
                if src_w == dst_w {
                    writeln!(
                        out,
                        "  %v{} = bitcast {} {} to {}",
                        result.0, from_s, ov_coerced, to_s
                    )?;
                } else {
                    let zero_extend_src = matches!(
                        from_ty,
                        IrType::Scalar(DType::U8 | DType::U32 | DType::U64 | DType::USize | DType::Bool)
                    );
                    if src_w > dst_w {
                        writeln!(
                            out,
                            "  %v{} = trunc {} {} to {}",
                            result.0, from_s, ov_coerced, to_s
                        )?;
                    } else if zero_extend_src {
                        writeln!(
                            out,
                            "  %v{} = zext {} {} to {}",
                            result.0, from_s, ov_coerced, to_s
                        )?;
                    } else {
                        writeln!(
                            out,
                            "  %v{} = sext {} {} to {}",
                            result.0, from_s, ov_coerced, to_s
                        )?;
                    }
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
                // If the function has a non-void return type but the IR
                // produced an empty return (common for if-else branches that
                // don't explicitly return), emit a zero/null return value
                // instead of `ret void` which would be invalid LLVM IR.
                let ret_ty = llvm_type_complete(&func.return_ty)?;
                if ret_ty == "void" {
                    writeln!(out, "  ret void")?;
                } else if ret_ty == "ptr" {
                    writeln!(out, "  ret ptr null")?;
                } else if ret_ty.starts_with('i') {
                    writeln!(out, "  ret {} 0", ret_ty)?;
                } else if ret_ty == "double" || ret_ty == "float" {
                    writeln!(out, "  ret {} 0.0", ret_ty)?;
                } else {
                    writeln!(out, "  ret {} zeroinitializer", ret_ty)?;
                }
            } else {
                let v_id = values[0];
                let v = val(v_id);
                let ty_s = llvm_type_complete(&func.return_ty)?;
                // Check for type mismatch between emitted value and return type.
                if let Some(actual_ty) = emitted_types.get(&v_id) {
                    if *actual_ty != ty_s && !consts.contains_key(&v_id) {
                        *gep_counter += 1;
                        let cast_name = format!("%ret_cast{}", gep_counter);
                        if actual_ty == "ptr" && ty_s.starts_with('i') {
                            writeln!(out, "  {} = ptrtoint ptr {} to {}", cast_name, v, ty_s)?;
                        } else if ty_s == "ptr" && actual_ty.starts_with('i') {
                            writeln!(out, "  {} = inttoptr {} {} to ptr", cast_name, actual_ty, v)?;
                        } else {
                            writeln!(
                                out,
                                "  {} = bitcast {} {} to {}",
                                cast_name, actual_ty, v, ty_s
                            )?;
                        }
                        writeln!(out, "  ret {} {}", ty_s, cast_name)?;
                        return Ok(());
                    }
                }
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

        // ── Typed user-defined function calls ─────────────────────────────
        IrInstr::Call {
            result,
            callee,
            args,
            result_ty,
        } => {
            let callee_name = entry_rename
                .and_then(|(orig, llvm)| if *callee == orig { Some(llvm) } else { None })
                .unwrap_or(callee);
            if let Some((ret_s, param_ss)) = fn_sigs.get(callee) {
                // Build typed arg list, inserting coercions when the actual
                // LLVM type of a value differs from the callee's parameter type.
                let mut typed_args: Vec<String> = Vec::with_capacity(args.len());
                for (a, expected_ty) in args.iter().zip(param_ss.iter()) {
                    let actual_ty = emitted_types
                        .get(a)
                        .cloned()
                        .or_else(|| func.value_type(*a).and_then(|t| llvm_type_complete(t).ok()))
                        .unwrap_or_else(|| expected_ty.clone());
                    let v = val(*a);
                    if actual_ty != *expected_ty {
                        // Need a coercion.
                        let tmp = format!("%cast{}", gep_counter);
                        *gep_counter += 1;
                        if actual_ty == "ptr" && expected_ty == "i64" {
                            writeln!(out, "  {} = ptrtoint ptr {} to i64", tmp, v)?;
                        } else if actual_ty == "i64" && expected_ty == "ptr" {
                            writeln!(out, "  {} = inttoptr i64 {} to ptr", tmp, v)?;
                        } else if actual_ty == "ptr" && expected_ty == "i32" {
                            writeln!(out, "  {} = ptrtoint ptr {} to i32", tmp, v)?;
                        } else if actual_ty == "i32" && expected_ty == "ptr" {
                            writeln!(out, "  {} = inttoptr i32 {} to ptr", tmp, v)?;
                        } else if (actual_ty == "i64" && expected_ty == "i1")
                            || (actual_ty == "i1" && expected_ty == "i64")
                            || (actual_ty == "i32" && expected_ty == "i64")
                            || (actual_ty == "i64" && expected_ty == "i32")
                        {
                            // integer truncation/extension
                            let op = if bit_width(&actual_ty) > bit_width(expected_ty) {
                                "trunc"
                            } else {
                                "zext"
                            };
                            writeln!(
                                out,
                                "  {} = {} {} {} to {}",
                                tmp, op, actual_ty, v, expected_ty
                            )?;
                        } else {
                            writeln!(
                                out,
                                "  {} = bitcast {} {} to {}",
                                tmp, actual_ty, v, expected_ty
                            )?;
                        }
                        typed_args.push(format!("{} {}", expected_ty, tmp));
                    } else {
                        typed_args.push(format!("{} {}", expected_ty, v));
                    }
                }
                if let Some(r) = result {
                    writeln!(
                        out,
                        "  %v{} = call {} @{}({})",
                        r.0,
                        ret_s,
                        callee_name,
                        typed_args.join(", ")
                    )?;
                } else {
                    writeln!(
                        out,
                        "  call {} @{}({})",
                        ret_s,
                        callee_name,
                        typed_args.join(", ")
                    )?;
                }
            } else {
                // Unknown callee (runtime intrinsic) — opaque call.
                let ret_ty_s = result_ty
                    .as_ref()
                    .and_then(|t| llvm_type_complete(t).ok())
                    .unwrap_or_else(|| "ptr".to_owned());
                let args_str: Vec<String> =
                    args.iter().map(|a| format!("ptr {}", val(*a))).collect();
                if let Some(r) = result {
                    writeln!(
                        out,
                        "  %v{} = call {} @{}({})",
                        r.0,
                        ret_ty_s,
                        callee_name,
                        args_str.join(", ")
                    )?;
                } else {
                    writeln!(out, "  call void @{}({})", callee_name, args_str.join(", "))?;
                }
            }
        }

        // ── Struct ops ─────────────────────────────────────────────────────
        IrInstr::MakeStruct {
            result,
            fields,
            result_ty,
        } => {
            if let IrType::Struct {
                name,
                fields: field_tys,
            } = result_ty
            {
                // Heap-allocate struct so it survives function returns.
                let struct_ty = format!("%{}", name);
                // Compute struct size via GEP-from-null trick.
                writeln!(
                    out,
                    "  %struct_sz{r} = getelementptr {sty}, ptr null, i32 1",
                    r = result.0,
                    sty = struct_ty
                )?;
                writeln!(
                    out,
                    "  %struct_bytes{r} = ptrtoint ptr %struct_sz{r} to i64",
                    r = result.0
                )?;
                writeln!(
                    out,
                    "  %struct_alloc{r} = call ptr @malloc(i64 %struct_bytes{r})",
                    r = result.0
                )?;
                for (i, (fv, (_, fty))) in fields.iter().zip(field_tys.iter()).enumerate() {
                    let fty_s = llvm_type_complete(fty)?;
                    let gep_name = format!("%sgep{}_{}", result.0, i);
                    writeln!(
                        out,
                        "  {} = getelementptr inbounds {}, ptr %struct_alloc{}, i32 0, i32 {}",
                        gep_name, struct_ty, result.0, i
                    )?;

                    // The value's *emitted* type is authoritative for the store,
                    // not the declared field type — the two disagree for a
                    // tag-only `choice`. `MakeVariant` emits a bare `i64` tag
                    // when the variant carries no payload, while
                    // `llvm_type_complete` maps every enum to `ptr`: it is a free
                    // function with no `IrModule` access, so it cannot tell a
                    // tag-only enum from a payload-carrying one (payload info
                    // lives in `IrModule.enum_defs`, not in `IrType::Enum`).
                    //
                    // Storing the i64 into the ptr-typed slot emitted
                    // `store ptr %v0, ...` for an i64 value and clang rejected the
                    // module. Coercing here keeps the slot consistently `ptr`,
                    // which is what `GetField` already loads it as. See
                    // docs/known-issues.md issue 6.
                    let mut store_val = val(*fv);
                    let actual_ty = emitted_types.get(fv).cloned().or_else(|| {
                        func.value_type(*fv)
                            .and_then(|t| llvm_type_complete(t).ok())
                    });
                    if let Some(actual) = actual_ty {
                        if actual != fty_s {
                            if fty_s == "ptr" && actual.starts_with('i') {
                                *gep_counter += 1;
                                let tmp = format!("%fcoerce{}", gep_counter);
                                writeln!(
                                    out,
                                    "  {} = inttoptr {} {} to ptr",
                                    tmp, actual, store_val
                                )?;
                                store_val = tmp;
                            } else if actual == "ptr" && fty_s.starts_with('i') {
                                *gep_counter += 1;
                                let tmp = format!("%fcoerce{}", gep_counter);
                                writeln!(
                                    out,
                                    "  {} = ptrtoint ptr {} to {}",
                                    tmp, store_val, fty_s
                                )?;
                                store_val = tmp;
                            }
                        }
                    }

                    writeln!(
                        out,
                        "  store {} {}, ptr {}, align 8",
                        fty_s, store_val, gep_name
                    )?;
                }
                // Use the malloc'd pointer directly as the struct value.
                writeln!(
                    out,
                    "  %v{r} = getelementptr inbounds {sty}, ptr %struct_alloc{r}, i32 0",
                    r = result.0,
                    sty = struct_ty
                )?;
            } else {
                let mut args_str: Vec<String> = Vec::with_capacity(fields.len());
                for f in fields {
                    let fv = val(*f);
                    let fty = func.value_type(*f);
                    let ptr_f = box_to_ptr(
                        out,
                        func,
                        *f,
                        &fv,
                        fty,
                        emitted_types.get(f).map(|s| s.as_str()),
                        gep_counter,
                    )?;
                    args_str.push(format!("ptr {}", ptr_f));
                }
                writeln!(
                    out,
                    "  %v{} = call ptr @iris_make_struct(i32 {}, {})",
                    result.0,
                    fields.len(),
                    args_str.join(", ")
                )?;
            }
        }

        IrInstr::GetField {
            result,
            base,
            field_index,
            result_ty,
        } => {
            let bv = val(*base);
            // Try to determine struct type from value type.
            let base_ty = inferred_value_type(func, *base, func.value_type(*base));
            if let Some(IrType::Struct {
                name,
                fields: field_tys,
            }) = base_ty.as_ref()
            {
                let struct_ty = format!("%{}", name);
                let fty_s = llvm_type_complete(result_ty)?;
                let gep_name = format!("%fgep{}_{}", result.0, field_index);
                writeln!(
                    out,
                    "  {} = getelementptr inbounds {}, ptr {}, i32 0, i32 {}",
                    gep_name, struct_ty, bv, field_index
                )?;
                writeln!(
                    out,
                    "  %v{} = load {}, ptr {}, align 8",
                    result.0, fty_s, gep_name
                )?;
                let _ = field_tys; // suppress unused warning
            } else {
                let raw = format!("%raw_gf{}", gep_counter);
                *gep_counter += 1;
                writeln!(
                    out,
                    "  {} = call ptr @iris_get_field(ptr {}, i32 {})",
                    raw, bv, field_index
                )?;
                unbox_ptr_to_result(out, raw, result.0, result_ty, gep_counter)?;
            }
        }

        // ── Enum ops ───────────────────────────────────────────────────────
        IrInstr::MakeVariant {
            result,
            variant_idx,
            fields,
            ..
        } => {
            if fields.is_empty() {
                // Tag-only variants are represented as immediate tags.
                writeln!(out, "  %v{} = add i64 0, {}", result.0, variant_idx)?;
            } else {
                let mut args_str = Vec::new();
                for f in fields {
                    let fv = val(*f);
                    let fty = func.value_type(*f);
                    let ptr_f = box_to_ptr(
                        out,
                        func,
                        *f,
                        &fv,
                        fty,
                        emitted_types.get(f).map(|s| s.as_str()),
                        gep_counter,
                    )?;
                    args_str.push(format!("ptr {}", ptr_f));
                }
                writeln!(
                    out,
                    "  %v{} = call ptr (i64, i32, ...) @iris_make_variant(i64 {}, i32 {}, {})",
                    result.0,
                    variant_idx,
                    fields.len(),
                    args_str.join(", ")
                )?;
            }
        }

        IrInstr::SwitchVariant {
            scrutinee,
            arms,
            default_block,
        } => {
            let sv = val(*scrutinee);
            let scrutinee_emitted_ty = emitted_types.get(scrutinee).map(|s| s.as_str());
            let tag_val = if scrutinee_emitted_ty == Some("ptr") {
                let raw_tag = format!("%tag_{}", gep_counter);
                *gep_counter += 1;
                writeln!(
                    out,
                    "  {} = call i64 @iris_get_variant_tag(ptr {})",
                    raw_tag, sv
                )?;
                raw_tag
            } else {
                sv
            };

            let blocks = func.blocks();
            let default = default_block
                .map(|bb| format!("label %{}", block_label_by_id(blocks, bb)))
                .unwrap_or_else(|| {
                    arms.first()
                        .map(|(_, bb)| format!("label %{}", block_label_by_id(blocks, *bb)))
                        .unwrap_or_else(|| "label %unreachable".to_owned())
                });
            write!(out, "  switch i64 {}, {} [", tag_val, default)?;
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

        IrInstr::ExtractVariantField {
            result,
            operand,
            field_idx,
            result_ty,
            ..
        } => {
            let raw = format!("%raw_evf_{}", gep_counter);
            *gep_counter += 1;
            writeln!(
                out,
                "  {} = call ptr @iris_extract_variant_field(ptr {}, i64 {})",
                raw,
                val(*operand),
                field_idx
            )?;
            unbox_ptr_to_result(out, raw, result.0, result_ty, gep_counter)?;
        }

        // ── Tuple ops ──────────────────────────────────────────────────────
        IrInstr::MakeTuple {
            result, elements, ..
        } => {
            let mut args_str: Vec<String> = Vec::with_capacity(elements.len());
            for e in elements {
                let ev = val(*e);
                let ety = func.value_type(*e);
                let ptr_e = box_to_ptr(
                    out,
                    func,
                    *e,
                    &ev,
                    ety,
                    emitted_types.get(e).map(|s| s.as_str()),
                    gep_counter,
                )?;
                args_str.push(format!("ptr {}", ptr_e));
            }
            writeln!(
                out,
                "  %v{} = call ptr @iris_make_tuple(i32 {}, {})",
                result.0,
                elements.len(),
                args_str.join(", ")
            )?;
        }

        IrInstr::GetElement {
            result,
            base,
            index,
            result_ty,
        } => {
            let raw = format!("%raw_ge{}", gep_counter);
            *gep_counter += 1;
            writeln!(
                out,
                "  {} = call ptr @iris_get_element(ptr {}, i32 {})",
                raw,
                val(*base),
                index
            )?;
            unbox_ptr_to_result(out, raw, result.0, result_ty, gep_counter)?;
        }

        // ── Array ops ─────────────────────────────────────────────────────
        // Scalar-element fixed arrays: use alloca + GEP.
        IrInstr::AllocArray {
            result,
            elem_ty,
            size,
            init,
        } => {
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
                    writeln!(
                        out,
                        "  store {} {}, ptr {}, align {}",
                        ety_s,
                        val(*iv),
                        gep,
                        ety_align(elem_ty)
                    )?;
                }
            } else {
                writeln!(out, "  %v{} = call ptr @iris_alloc_array()", result.0)?;
            }
        }

        IrInstr::ArrayLoad {
            result,
            array,
            index,
            elem_ty,
        } => {
            if scalar_arrays.contains(array) {
                let ety_s = llvm_type_complete(elem_ty)?;
                let arr_size = find_alloc_size(func, *array);
                let gep = format!("%agep{}_{}", result.0, gep_counter);
                *gep_counter += 1;
                if let Some(sz) = arr_size {
                    let bc = *gep_counter;
                    *gep_counter += 1;
                    let idx_val = val(*index);
                    writeln!(
                        out,
                        "  %bci{}_0 = icmp ult i64 {}, {}",
                        bc, idx_val, sz
                    )?;
                    writeln!(
                        out,
                        "  br i1 %bci{}_0, label %bco{}_ok, label %bco{}_fail",
                        bc, bc, bc
                    )?;
                    writeln!(out, "bco{}_fail:", bc)?;
                    writeln!(
                        out,
                        "  call void @iris_bounds_check_abort(i64 {}, i64 {})",
                        idx_val, sz
                    )?;
                    writeln!(out, "  unreachable")?;
                    writeln!(out, "bco{}_ok:", bc)?;
                    writeln!(
                        out,
                        "  {} = getelementptr inbounds [{} x {}], ptr %v{}, i64 0, i64 {}",
                        gep,
                        sz,
                        ety_s,
                        array.0,
                        val(*index)
                    )?;
                } else {
                    writeln!(
                        out,
                        "  {} = getelementptr inbounds {}, ptr %v{}, i64 {}",
                        gep,
                        ety_s,
                        array.0,
                        val(*index)
                    )?;
                }
                writeln!(
                    out,
                    "  %v{} = load {}, ptr {}, align {}",
                    result.0,
                    ety_s,
                    gep,
                    ety_align(elem_ty)
                )?;
            } else {
                let raw = format!("%raw_arr{}", gep_counter);
                *gep_counter += 1;
                writeln!(
                    out,
                    "  {} = call ptr @iris_array_load(ptr {}, i64 {})",
                    raw,
                    val(*array),
                    val(*index)
                )?;
                unbox_ptr_to_result(out, raw, result.0, elem_ty, gep_counter)?;
            }
        }

        IrInstr::ArrayStore {
            array,
            index,
            value,
        } => {
            if scalar_arrays.contains(array) {
                let vty = func.value_type(*value);
                let ety_s = vty
                    .and_then(|t| llvm_type_complete(t).ok())
                    .unwrap_or_else(|| "i64".to_owned());
                let arr_size = find_alloc_size(func, *array);
                let gep = format!("%asgep_{}", gep_counter);
                *gep_counter += 1;
                if let Some(sz) = arr_size {
                    let bc = *gep_counter;
                    *gep_counter += 1;
                    let idx_val = val(*index);
                    writeln!(
                        out,
                        "  %bsci{}_0 = icmp ult i64 {}, {}",
                        bc, idx_val, sz
                    )?;
                    writeln!(
                        out,
                        "  br i1 %bsci{}_0, label %bsco{}_ok, label %bsco{}_fail",
                        bc, bc, bc
                    )?;
                    writeln!(out, "bsco{}_fail:", bc)?;
                    writeln!(
                        out,
                        "  call void @iris_bounds_check_abort(i64 {}, i64 {})",
                        idx_val, sz
                    )?;
                    writeln!(out, "  unreachable")?;
                    writeln!(out, "bsco{}_ok:", bc)?;
                    writeln!(
                        out,
                        "  {} = getelementptr inbounds [{} x {}], ptr %v{}, i64 0, i64 {}",
                        gep,
                        sz,
                        ety_s,
                        array.0,
                        val(*index)
                    )?;
                } else {
                    writeln!(
                        out,
                        "  {} = getelementptr inbounds {}, ptr %v{}, i64 {}",
                        gep,
                        ety_s,
                        array.0,
                        val(*index)
                    )?;
                }
                writeln!(out, "  store {} {}, ptr {}", ety_s, val(*value), gep)?;
            } else {
                let vv = val(*value);
                let vty = func.value_type(*value);
                let ptr_v = box_to_ptr(
                    out,
                    func,
                    *value,
                    &vv,
                    vty,
                    emitted_types.get(value).map(|s| s.as_str()),
                    gep_counter,
                )?;
                writeln!(
                    out,
                    "  call void @iris_array_store(ptr {}, i64 {}, ptr {})",
                    val(*array),
                    val(*index),
                    ptr_v
                )?;
            }
        }

        // ── Memory / Tensor ops ────────────────────────────────────────────
        IrInstr::TensorOp {
            result, op, inputs, ..
        } => {
            match op {
                TensorOp::Einsum { notation } => {
                    // Dispatch known einsum patterns to C runtime functions.
                    // "mk,kn->mn" and similar 2-input contraction → matmul
                    if inputs.len() == 2 && is_matmul_notation(notation) {
                        writeln!(
                            out,
                            "  %v{} = call ptr @iris_tensor_matmul(ptr {}, ptr {})",
                            result.0,
                            val(inputs[0]),
                            val(inputs[1])
                        )?;
                    } else {
                        return Err(CodegenError::Unsupported {
                            backend: "llvm".into(),
                            detail: format!(
                                "unsupported einsum notation '{}' with {} inputs",
                                notation,
                                inputs.len()
                            ),
                        });
                    }
                }
                TensorOp::Unary { op: unary_op } => {
                    if inputs.len() == 1 {
                        let fn_name = match unary_op.as_str() {
                            "relu" => "iris_tensor_relu",
                            "sigmoid" => "iris_tensor_sigmoid",
                            "tanh" => "iris_tensor_tanh_act",
                            "neg" => "iris_tensor_neg",
                            "exp" => "iris_tensor_exp",
                            "log" => "iris_tensor_log",
                            "sqrt" => "iris_tensor_sqrt",
                            "abs" => "iris_tensor_abs",
                            _ => {
                                return Err(CodegenError::Unsupported {
                                    backend: "llvm".into(),
                                    detail: format!("unsupported tensor unary op '{}'", unary_op),
                                });
                            }
                        };
                        writeln!(
                            out,
                            "  %v{} = call ptr @{}(ptr {})",
                            result.0,
                            fn_name,
                            val(inputs[0])
                        )?;
                    } else {
                        return Err(CodegenError::Unsupported {
                            backend: "llvm".into(),
                            detail: "tensor unary op requires exactly 1 input".into(),
                        });
                    }
                }
                TensorOp::Reshape => {
                    // Reshape: first input is tensor, remaining are dimension i64s.
                    // For now, generate a call to iris_tensor_reshape with ndim + dims.
                    if inputs.len() >= 2 {
                        let ndim = inputs.len() - 1;
                        // Build shape array on the stack
                        let shape_arr = format!("%reshape_shape_{}", result.0);
                        writeln!(out, "  {} = alloca i64, i32 {}", shape_arr, ndim)?;
                        for (i, dim_val) in inputs[1..].iter().enumerate() {
                            let gep = format!("%reshape_gep_{}_{}", result.0, i);
                            writeln!(
                                out,
                                "  {} = getelementptr i64, ptr {}, i32 {}",
                                gep, shape_arr, i
                            )?;
                            writeln!(out, "  store i64 {}, ptr {}", val(*dim_val), gep)?;
                        }
                        writeln!(
                            out,
                            "  %v{} = call ptr @iris_tensor_reshape(ptr {}, i32 {}, ptr {})",
                            result.0,
                            val(inputs[0]),
                            ndim,
                            shape_arr
                        )?;
                    } else {
                        // Single input: flatten (reshape to 1D)
                        writeln!(
                            out,
                            "  %v{} = call ptr @iris_tensor_reshape(ptr {}, i32 0, ptr null)",
                            result.0,
                            val(inputs[0])
                        )?;
                    }
                }
                TensorOp::Transpose { axes } => {
                    if inputs.len() == 1 {
                        if axes.is_empty() {
                            // Default reverse transpose
                            writeln!(
                                out,
                                "  %v{} = call ptr @iris_tensor_transpose(ptr {}, i32 0, ptr null)",
                                result.0,
                                val(inputs[0])
                            )?;
                        } else {
                            let ndim = axes.len();
                            let axes_arr = format!("%trans_axes_{}", result.0);
                            writeln!(out, "  {} = alloca i32, i32 {}", axes_arr, ndim)?;
                            for (i, &ax) in axes.iter().enumerate() {
                                let gep = format!("%trans_gep_{}_{}", result.0, i);
                                writeln!(
                                    out,
                                    "  {} = getelementptr i32, ptr {}, i32 {}",
                                    gep, axes_arr, i
                                )?;
                                writeln!(out, "  store i32 {}, ptr {}", ax, gep)?;
                            }
                            writeln!(
                                out,
                                "  %v{} = call ptr @iris_tensor_transpose(ptr {}, i32 {}, ptr {})",
                                result.0,
                                val(inputs[0]),
                                ndim,
                                axes_arr
                            )?;
                        }
                    } else {
                        return Err(CodegenError::Unsupported {
                            backend: "llvm".into(),
                            detail: "tensor transpose requires exactly 1 input".into(),
                        });
                    }
                }
                TensorOp::Reduce {
                    op: reduce_op,
                    axes,
                    keepdims,
                } => {
                    if inputs.len() == 1 && axes.len() == 1 {
                        let fn_name = match reduce_op.as_str() {
                            "sum" => "iris_tensor_reduce_sum",
                            "max" => "iris_tensor_reduce_max",
                            "mean" => "iris_tensor_reduce_mean",
                            _ => {
                                return Err(CodegenError::Unsupported {
                                    backend: "llvm".into(),
                                    detail: format!("unsupported tensor reduce op '{}'", reduce_op),
                                });
                            }
                        };
                        writeln!(
                            out,
                            "  %v{} = call ptr @{}(ptr {}, i32 {}, i32 {})",
                            result.0,
                            fn_name,
                            val(inputs[0]),
                            axes[0],
                            if *keepdims { 1 } else { 0 }
                        )?;
                    } else {
                        return Err(CodegenError::Unsupported {
                            backend: "llvm".into(),
                            detail: format!("tensor reduce requires 1 input and 1 axis, got {} inputs and {} axes", inputs.len(), axes.len()),
                        });
                    }
                }
            }
        }

        IrInstr::Load {
            result,
            tensor,
            indices,
            result_ty,
        } => {
            let tv = val(*tensor);
            let ty_s = llvm_type_complete(result_ty)?;
            match indices.as_slice() {
                [] => {
                    writeln!(out, "  %v{} = load {}, ptr {}", result.0, ty_s, tv)?;
                }
                [idx] => {
                    let gep = format!("%gep{}", gep_counter);
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
                .and_then(|ty| llvm_type_complete(ty).ok())
                .unwrap_or_else(|| "ptr".to_owned());
            match indices.as_slice() {
                [] => {
                    writeln!(out, "  store {} {}, ptr {}", ty_s, vv, tv)?;
                }
                [idx] => {
                    let gep = format!("%gep{}", gep_counter);
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
                    let mut args = vec![format!("ptr {}", tv), format!("{} {}", ty_s, vv)];
                    for idx in indices {
                        args.push(format!("i64 {}", val(*idx)));
                    }
                    writeln!(out, "  call void @iris_tensor_store({})", args.join(", "))?;
                }
            }
        }

        // ── Concurrency ───────────────────────────────────────────────────
        IrInstr::ParFor {
            body_fn,
            start,
            end,
            args,
            ..
        } => {
            // The lowerer lambda-lifts the body to
            //   __par_body_N(loop_var, capture0, capture1, ...)
            // but the runtime can only call it as `fn(int64_t i, void* env)`.
            // The captures in `args` were previously ignored here, so every
            // capture parameter received an uninitialised register — a garbage
            // pointer that segfaulted the moment the body touched it (an
            // `atomic_add` inside a `par for` was the reported case). A body
            // with no captures happened to work, which is why this survived.
            //
            // Fix: pack the captures into a closure environment with the same
            // helper closures already use, and pass it as the runtime's `arg`.
            // `emit_function_ir_with_name` gives `__par_body_*` the matching
            // `(i64 %var, ptr %env)` signature and unpacks them on entry.
            // See docs/known-issues.md issue 12.
            let mut cap_args = vec![];
            for c in args {
                let cv = val(*c);
                let cty = func.value_type(*c);
                let ptr_c = box_to_ptr(
                    out,
                    func,
                    *c,
                    &cv,
                    cty,
                    emitted_types.get(c).map(|s| s.as_str()),
                    gep_counter,
                )?;
                cap_args.push(format!("ptr {}", ptr_c));
            }

            let env = if args.is_empty() {
                "null".to_owned()
            } else {
                *gep_counter += 1;
                let env_name = format!("%par_env{}", gep_counter);
                let mut mk_args = vec![
                    format!("ptr @{}", body_fn),
                    format!("i32 {}", args.len()),
                ];
                mk_args.extend(cap_args);
                writeln!(
                    out,
                    "  {} = call ptr @iris_make_closure({})",
                    env_name,
                    mk_args.join(", ")
                )?;
                env_name
            };

            writeln!(
                out,
                "  call void @iris_par_for(ptr @{}, i64 {}, i64 {}, ptr {})",
                body_fn,
                val(*start),
                val(*end),
                env
            )?;
        }

        IrInstr::ChanNew { result, capacity, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_chan_new(i64 {})", result.0, val(*capacity))?;
        }
        IrInstr::ChanSend { chan, value } => {
            let vv = val(*value);
            let vty = func.value_type(*value);
            let ptr_v = box_to_ptr(
                out,
                func,
                *value,
                &vv,
                vty,
                emitted_types.get(value).map(|s| s.as_str()),
                gep_counter,
            )?;
            writeln!(
                out,
                "  call void @iris_chan_send(ptr {}, ptr {})",
                val(*chan),
                ptr_v
            )?;
        }
        IrInstr::ChanRecv {
            result,
            chan,
            elem_ty,
        } => {
            // iris_chan_recv returns IrisVal* (boxed); unbox to the element type.
            let raw = format!("%raw_recv{}", gep_counter);
            *gep_counter += 1;
            writeln!(
                out,
                "  {} = call ptr @iris_chan_recv(ptr {})",
                raw,
                val(*chan)
            )?;
            match elem_ty {
                IrType::Scalar(DType::I64) | IrType::Scalar(DType::U64) => {
                    writeln!(
                        out,
                        "  %v{} = call i64 @iris_unbox_i64(ptr {})",
                        result.0, raw
                    )?;
                }
                IrType::Scalar(DType::I32) | IrType::Scalar(DType::U32) => {
                    let tmp = format!("%raw_recv_i64_{}", gep_counter);
                    *gep_counter += 1;
                    writeln!(out, "  {} = call i64 @iris_unbox_i64(ptr {})", tmp, raw)?;
                    writeln!(out, "  %v{} = trunc i64 {} to i32", result.0, tmp)?;
                }
                IrType::Scalar(DType::F64) => {
                    writeln!(
                        out,
                        "  %v{} = call double @iris_unbox_f64(ptr {})",
                        result.0, raw
                    )?;
                }
                IrType::Scalar(DType::F32) => {
                    let tmp = format!("%raw_recv_f64_{}", gep_counter);
                    *gep_counter += 1;
                    writeln!(out, "  {} = call double @iris_unbox_f64(ptr {})", tmp, raw)?;
                    writeln!(out, "  %v{} = fptrunc double {} to float", result.0, tmp)?;
                }
                IrType::Scalar(DType::Bool) => {
                    let tmp = format!("%raw_recv_bool_{}", gep_counter);
                    *gep_counter += 1;
                    writeln!(out, "  {} = call i32 @iris_unbox_bool(ptr {})", tmp, raw)?;
                    writeln!(out, "  %v{} = trunc i32 {} to i1", result.0, tmp)?;
                }
                IrType::Str => {
                    writeln!(
                        out,
                        "  %v{} = call ptr @iris_unbox_str(ptr {})",
                        result.0, raw
                    )?;
                }
                _ => {
                    unbox_ptr_to_result(out, raw, result.0, elem_ty, gep_counter)?;
                }
            }
        }
        IrInstr::Spawn { body_fn, args } => {
            let tramp_name = format!("{}_trampoline", body_fn);
            if args.is_empty() {
                writeln!(
                    out,
                    "  call void @iris_spawn_fn(ptr @{}, ptr null)",
                    tramp_name
                )?;
            } else {
                let arg_buf = format!("%spawn_args{}", gep_counter);
                *gep_counter += 1;
                let alloc_size = (args.len() as i64) * 8;
                writeln!(out, "  {} = call ptr @malloc(i64 {})", arg_buf, alloc_size)?;
                for (i, arg_id) in args.iter().enumerate() {
                    let slot = format!("%spawn_arg_slot{}_{}", gep_counter, i);
                    writeln!(
                        out,
                        "  {} = getelementptr ptr, ptr {}, i64 {}",
                        slot, arg_buf, i
                    )?;
                    let arg_val = val(*arg_id);
                    let boxed = box_to_ptr(
                        out,
                        func,
                        *arg_id,
                        &arg_val,
                        func.value_type(*arg_id),
                        emitted_types.get(arg_id).map(|s| s.as_str()),
                        gep_counter,
                    )?;
                    writeln!(out, "  store ptr {}, ptr {}", boxed, slot)?;
                }
                writeln!(
                    out,
                    "  call void @iris_spawn_fn(ptr @{}, ptr {})",
                    tramp_name, arg_buf
                )?;
            }
        }

        // ── Atomics ───────────────────────────────────────────────────────
        IrInstr::AtomicNew { result, value, .. } => {
            let vv = val(*value);
            let vty = func.value_type(*value);
            let ptr_v = box_to_ptr(
                out,
                func,
                *value,
                &vv,
                vty,
                emitted_types.get(value).map(|s| s.as_str()),
                gep_counter,
            )?;
            writeln!(
                out,
                "  %v{} = call ptr @iris_atomic_new(ptr {})",
                result.0, ptr_v
            )?;
        }
        IrInstr::AtomicLoad {
            result,
            atomic,
            result_ty,
        } => {
            let tmp = format!("%raw_atomic{}", gep_counter);
            *gep_counter += 1;
            writeln!(
                out,
                "  {} = call ptr @iris_atomic_load(ptr {})",
                tmp,
                val(*atomic)
            )?;
            unbox_ptr_to_result(out, tmp, result.0, result_ty, gep_counter)?;
        }
        IrInstr::AtomicStore { atomic, value } => {
            let vv = val(*value);
            let vty = func.value_type(*value);
            let ptr_v = box_to_ptr(
                out,
                func,
                *value,
                &vv,
                vty,
                emitted_types.get(value).map(|s| s.as_str()),
                gep_counter,
            )?;
            writeln!(
                out,
                "  call void @iris_atomic_store(ptr {}, ptr {})",
                val(*atomic),
                ptr_v
            )?;
        }
        IrInstr::AtomicAdd {
            result,
            atomic,
            value,
            result_ty,
        } => {
            let vv = val(*value);
            let vty = func.value_type(*value);
            let ptr_v = box_to_ptr(
                out,
                func,
                *value,
                &vv,
                vty,
                emitted_types.get(value).map(|s| s.as_str()),
                gep_counter,
            )?;
            let tmp = format!("%raw_atomic_add{}", gep_counter);
            *gep_counter += 1;
            writeln!(
                out,
                "  {} = call ptr @iris_atomic_add(ptr {}, ptr {})",
                tmp,
                val(*atomic),
                ptr_v
            )?;
            unbox_ptr_to_result(out, tmp, result.0, result_ty, gep_counter)?;
        }
        IrInstr::MutexNew { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_mutex_new()", result.0)?;
        }
        IrInstr::MutexLock { result, mutex, .. } => {
            writeln!(
                out,
                "  %v{} = call ptr @iris_mutex_lock(ptr {})",
                result.0,
                val(*mutex)
            )?;
        }
        IrInstr::MutexUnlock { mutex } => {
            writeln!(out, "  call void @iris_mutex_unlock(ptr {})", val(*mutex))?;
        }

        // ── Option / Result ────────────────────────────────────────────────
        IrInstr::MakeSome { result, value, .. } => {
            let vv = val(*value);
            let vty = func.value_type(*value);
            let ptr_v = box_to_ptr(
                out,
                func,
                *value,
                &vv,
                vty,
                emitted_types.get(value).map(|s| s.as_str()),
                gep_counter,
            )?;
            writeln!(
                out,
                "  %v{} = call ptr @iris_make_some(ptr {})",
                result.0, ptr_v
            )?;
        }
        IrInstr::MakeNone { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_make_none()", result.0)?;
        }
        IrInstr::IsSome { result, operand } => {
            writeln!(
                out,
                "  %v{}_raw = call i32 @iris_is_some(ptr {})",
                result.0,
                val(*operand)
            )?;
            writeln!(
                out,
                "  %v{} = trunc i32 %v{}_raw to i1",
                result.0, result.0
            )?;
        }
        IrInstr::OptionUnwrap {
            result,
            operand,
            result_ty,
            ..
        } => {
            // If the operand was already eagerly unboxed (e.g. by MapGet/ListGet),
            // emitted_types will show a scalar type rather than "ptr" — just copy it.
            let operand_emitted = emitted_types
                .get(operand)
                .map(|s| s.as_str())
                .unwrap_or("ptr");
            if operand_emitted != "ptr" {
                // Already unboxed — copy the scalar value directly.
                let expected_ty = llvm_type_complete(result_ty)?;
                if operand_emitted == expected_ty {
                    writeln!(
                        out,
                        "  %v{} = add {} {}, 0",
                        result.0,
                        expected_ty,
                        val(*operand)
                    )?;
                } else {
                    writeln!(
                        out,
                        "  %v{} = bitcast {} {} to {}",
                        result.0,
                        operand_emitted,
                        val(*operand),
                        expected_ty
                    )?;
                }
                let _ = expected_ty; // type already in emitted_types from sub-pass D
            } else {
                let tmp = format!("%raw_ouw{}", gep_counter);
                *gep_counter += 1;
                writeln!(
                    out,
                    "  {} = call ptr @iris_option_unwrap(ptr {})",
                    tmp,
                    val(*operand)
                )?;
                unbox_ptr_to_result(out, tmp, result.0, result_ty, gep_counter)?;
            }
        }
        IrInstr::MakeOk { result, value, .. } => {
            let vv = val(*value);
            let vty = func.value_type(*value);
            let ptr_v = box_to_ptr(
                out,
                func,
                *value,
                &vv,
                vty,
                emitted_types.get(value).map(|s| s.as_str()),
                gep_counter,
            )?;
            writeln!(
                out,
                "  %v{} = call ptr @iris_make_ok(ptr {})",
                result.0, ptr_v
            )?;
        }
        IrInstr::MakeErr { result, value, .. } => {
            let vv = val(*value);
            let vty = func.value_type(*value);
            let ptr_v = box_to_ptr(
                out,
                func,
                *value,
                &vv,
                vty,
                emitted_types.get(value).map(|s| s.as_str()),
                gep_counter,
            )?;
            writeln!(
                out,
                "  %v{} = call ptr @iris_make_err(ptr {})",
                result.0, ptr_v
            )?;
        }
        IrInstr::IsOk { result, operand } => {
            writeln!(
                out,
                "  %v{}_raw = call i32 @iris_is_ok(ptr {})",
                result.0,
                val(*operand)
            )?;
            writeln!(
                out,
                "  %v{} = trunc i32 %v{}_raw to i1",
                result.0, result.0
            )?;
        }
        IrInstr::ResultUnwrap {
            result,
            operand,
            result_ty,
            ..
        } => {
            let tmp = format!("%raw_ruw{}", gep_counter);
            *gep_counter += 1;
            writeln!(
                out,
                "  {} = call ptr @iris_result_unwrap(ptr {})",
                tmp,
                val(*operand)
            )?;
            unbox_ptr_to_result(out, tmp, result.0, result_ty, gep_counter)?;
        }
        IrInstr::ResultUnwrapErr {
            result,
            operand,
            result_ty,
            ..
        } => {
            let tmp = format!("%raw_rwe{}", gep_counter);
            *gep_counter += 1;
            writeln!(
                out,
                "  {} = call ptr @iris_result_unwrap_err(ptr {})",
                tmp,
                val(*operand)
            )?;
            unbox_ptr_to_result(out, tmp, result.0, result_ty, gep_counter)?;
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
        IrInstr::StrContains {
            result,
            haystack,
            needle,
        } => {
            writeln!(
                out,
                "  %v{} = call i1 @iris_str_contains(ptr {}, ptr {})",
                result.0,
                val(*haystack),
                val(*needle)
            )?;
        }
        IrInstr::StrStartsWith {
            result,
            haystack,
            prefix,
        } => {
            writeln!(
                out,
                "  %v{} = call i1 @iris_str_starts_with(ptr {}, ptr {})",
                result.0,
                val(*haystack),
                val(*prefix)
            )?;
        }
        IrInstr::StrEndsWith {
            result,
            haystack,
            suffix,
        } => {
            writeln!(
                out,
                "  %v{} = call i1 @iris_str_ends_with(ptr {}, ptr {})",
                result.0,
                val(*haystack),
                val(*suffix)
            )?;
        }
        IrInstr::StrToUpper { result, operand } => {
            writeln!(
                out,
                "  %v{} = call ptr @iris_str_to_upper(ptr {})",
                result.0,
                val(*operand)
            )?;
        }
        IrInstr::StrToLower { result, operand } => {
            writeln!(
                out,
                "  %v{} = call ptr @iris_str_to_lower(ptr {})",
                result.0,
                val(*operand)
            )?;
        }
        IrInstr::StrTrim { result, operand } => {
            writeln!(
                out,
                "  %v{} = call ptr @iris_str_trim(ptr {})",
                result.0,
                val(*operand)
            )?;
        }
        IrInstr::StrRepeat {
            result,
            operand,
            count,
        } => {
            writeln!(
                out,
                "  %v{} = call ptr @iris_str_repeat(ptr {}, i64 {})",
                result.0,
                val(*operand),
                val(*count)
            )?;
        }
        IrInstr::StrIndex {
            result,
            string,
            index,
        } => {
            let idx_v =
                coerce_to_type(*index, "i64", consts, func, emitted_types, gep_counter, out)?;
            writeln!(
                out,
                "  %v{} = call i64 @iris_str_index(ptr {}, i64 {})",
                result.0,
                val(*string),
                idx_v
            )?;
        }
        IrInstr::StrSlice {
            result,
            string,
            start,
            end,
        } => {
            let start_v =
                coerce_to_type(*start, "i64", consts, func, emitted_types, gep_counter, out)?;
            let end_v = coerce_to_type(*end, "i64", consts, func, emitted_types, gep_counter, out)?;
            writeln!(
                out,
                "  %v{} = call ptr @iris_str_slice(ptr {}, i64 {}, i64 {})",
                result.0,
                val(*string),
                start_v,
                end_v
            )?;
        }
        IrInstr::StrFind {
            result,
            haystack,
            needle,
        } => {
            writeln!(
                out,
                "  %v{} = call ptr @iris_str_find(ptr {}, ptr {})",
                result.0,
                val(*haystack),
                val(*needle)
            )?;
        }
        IrInstr::StrReplace {
            result,
            string,
            from,
            to,
        } => {
            writeln!(
                out,
                "  %v{} = call ptr @iris_str_replace(ptr {}, ptr {}, ptr {})",
                result.0,
                val(*string),
                val(*from),
                val(*to)
            )?;
        }

        // ── Collections ────────────────────────────────────────────────────
        IrInstr::ListNew { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_list_new()", result.0)?;
        }
        IrInstr::ListPush { list, value } => {
            let vv = val(*value);
            let vty = func.value_type(*value);
            let ptr_v = box_to_ptr(
                out,
                func,
                *value,
                &vv,
                vty,
                emitted_types.get(value).map(|s| s.as_str()),
                gep_counter,
            )?;
            writeln!(
                out,
                "  call void @iris_list_push(ptr {}, ptr {})",
                val(*list),
                ptr_v
            )?;
        }
        IrInstr::ListLen { result, list } => {
            writeln!(
                out,
                "  %v{} = call i64 @iris_list_len(ptr {})",
                result.0,
                val(*list)
            )?;
        }
        IrInstr::ListGet {
            result,
            list,
            index,
            elem_ty,
        } => {
            // If elem_ty is Infer, resolve from the list's value type
            let resolved_elem_ty = if matches!(elem_ty, IrType::Infer) {
                if let Some(IrType::List(inner)) = func.value_type(*list) {
                    (**inner).clone()
                } else {
                    elem_ty.clone()
                }
            } else {
                elem_ty.clone()
            };

            let idx_v =
                coerce_to_type(*index, "i64", consts, func, emitted_types, gep_counter, out)?;
            // iris_list_get returns IrisVal* (boxed); unbox to the element type.
            match &resolved_elem_ty {
                IrType::Scalar(DType::I64) => {
                    let tmp = format!("%raw_get{}", gep_counter);
                    *gep_counter += 1;
                    writeln!(
                        out,
                        "  {} = call ptr @iris_list_get(ptr {}, i64 {})",
                        tmp,
                        val(*list),
                        idx_v
                    )?;
                    writeln!(
                        out,
                        "  %v{} = call i64 @iris_unbox_i64(ptr {})",
                        result.0, tmp
                    )?;
                }
                IrType::Scalar(DType::I32) => {
                    let tmp = format!("%raw_get{}", gep_counter);
                    *gep_counter += 1;
                    let tmp2 = format!("%raw_i64_{}", gep_counter);
                    *gep_counter += 1;
                    writeln!(
                        out,
                        "  {} = call ptr @iris_list_get(ptr {}, i64 {})",
                        tmp,
                        val(*list),
                        idx_v
                    )?;
                    writeln!(out, "  {} = call i64 @iris_unbox_i64(ptr {})", tmp2, tmp)?;
                    writeln!(out, "  %v{} = trunc i64 {} to i32", result.0, tmp2)?;
                }
                IrType::Scalar(DType::F64) => {
                    let tmp = format!("%raw_get{}", gep_counter);
                    *gep_counter += 1;
                    writeln!(
                        out,
                        "  {} = call ptr @iris_list_get(ptr {}, i64 {})",
                        tmp,
                        val(*list),
                        idx_v
                    )?;
                    writeln!(
                        out,
                        "  %v{} = call double @iris_unbox_f64(ptr {})",
                        result.0, tmp
                    )?;
                }
                IrType::Scalar(DType::F32) => {
                    let tmp = format!("%raw_get{}", gep_counter);
                    *gep_counter += 1;
                    let tmp2 = format!("%raw_f64_{}", gep_counter);
                    *gep_counter += 1;
                    writeln!(
                        out,
                        "  {} = call ptr @iris_list_get(ptr {}, i64 {})",
                        tmp,
                        val(*list),
                        idx_v
                    )?;
                    writeln!(out, "  {} = call double @iris_unbox_f64(ptr {})", tmp2, tmp)?;
                    writeln!(out, "  %v{} = fptrunc double {} to float", result.0, tmp2)?;
                }
                IrType::Scalar(DType::Bool) => {
                    let tmp = format!("%raw_get{}", gep_counter);
                    *gep_counter += 1;
                    let tmp2 = format!("%raw_bool{}", gep_counter);
                    *gep_counter += 1;
                    writeln!(
                        out,
                        "  {} = call ptr @iris_list_get(ptr {}, i64 {})",
                        tmp,
                        val(*list),
                        idx_v
                    )?;
                    writeln!(out, "  {} = call i32 @iris_unbox_bool(ptr {})", tmp2, tmp)?;
                    writeln!(out, "  %v{} = trunc i32 {} to i1", result.0, tmp2)?;
                }
                IrType::Str => {
                    let tmp = format!("%raw_get{}", gep_counter);
                    *gep_counter += 1;
                    writeln!(
                        out,
                        "  {} = call ptr @iris_list_get(ptr {}, i64 {})",
                        tmp,
                        val(*list),
                        idx_v
                    )?;
                    writeln!(
                        out,
                        "  %v{} = call ptr @iris_unbox_str(ptr {})",
                        result.0, tmp
                    )?;
                }
                _ => {
                    let tmp = format!("%raw_get{}", gep_counter);
                    *gep_counter += 1;
                    writeln!(
                        out,
                        "  {} = call ptr @iris_list_get(ptr {}, i64 {})",
                        tmp,
                        val(*list),
                        idx_v
                    )?;
                    unbox_ptr_to_result(out, tmp, result.0, &resolved_elem_ty, gep_counter)?;
                }
            }
        }
        IrInstr::ListSet { list, index, value } => {
            let vv = val(*value);
            let vty = func.value_type(*value);
            let ptr_v = box_to_ptr(
                out,
                func,
                *value,
                &vv,
                vty,
                emitted_types.get(value).map(|s| s.as_str()),
                gep_counter,
            )?;
            let idx_v =
                coerce_to_type(*index, "i64", consts, func, emitted_types, gep_counter, out)?;
            writeln!(
                out,
                "  call void @iris_list_set(ptr {}, i64 {}, ptr {})",
                val(*list),
                idx_v,
                ptr_v
            )?;
        }
        IrInstr::ListPop {
            result,
            list,
            elem_ty,
        } => {
            // If elem_ty is Infer, resolve from the list's value type
            let resolved_elem_ty = if matches!(elem_ty, IrType::Infer) {
                if let Some(IrType::List(inner)) = func.value_type(*list) {
                    (**inner).clone()
                } else {
                    elem_ty.clone()
                }
            } else {
                elem_ty.clone()
            };

            // iris_list_pop returns IrisVal* (boxed); unbox to the element type.
            match &resolved_elem_ty {
                IrType::Scalar(DType::I64) => {
                    let tmp = format!("%raw_pop{}", gep_counter);
                    *gep_counter += 1;
                    writeln!(
                        out,
                        "  {} = call ptr @iris_list_pop(ptr {})",
                        tmp,
                        val(*list)
                    )?;
                    writeln!(
                        out,
                        "  %v{} = call i64 @iris_unbox_i64(ptr {})",
                        result.0, tmp
                    )?;
                }
                IrType::Scalar(DType::F64) => {
                    let tmp = format!("%raw_pop{}", gep_counter);
                    *gep_counter += 1;
                    writeln!(
                        out,
                        "  {} = call ptr @iris_list_pop(ptr {})",
                        tmp,
                        val(*list)
                    )?;
                    writeln!(
                        out,
                        "  %v{} = call double @iris_unbox_f64(ptr {})",
                        result.0, tmp
                    )?;
                }
                IrType::Scalar(DType::Bool) => {
                    let tmp = format!("%raw_pop{}", gep_counter);
                    *gep_counter += 1;
                    let tmp2 = format!("%raw_popbool{}", gep_counter);
                    *gep_counter += 1;
                    writeln!(
                        out,
                        "  {} = call ptr @iris_list_pop(ptr {})",
                        tmp,
                        val(*list)
                    )?;
                    writeln!(out, "  {} = call i32 @iris_unbox_bool(ptr {})", tmp2, tmp)?;
                    writeln!(out, "  %v{} = trunc i32 {} to i1", result.0, tmp2)?;
                }
                IrType::Str => {
                    let tmp = format!("%raw_pop{}", gep_counter);
                    *gep_counter += 1;
                    writeln!(
                        out,
                        "  {} = call ptr @iris_list_pop(ptr {})",
                        tmp,
                        val(*list)
                    )?;
                    writeln!(
                        out,
                        "  %v{} = call ptr @iris_unbox_str(ptr {})",
                        result.0, tmp
                    )?;
                }
                _ => {
                    let tmp = format!("%raw_pop{}", gep_counter);
                    *gep_counter += 1;
                    writeln!(
                        out,
                        "  {} = call ptr @iris_list_pop(ptr {})",
                        tmp,
                        val(*list)
                    )?;
                    unbox_ptr_to_result(out, tmp, result.0, &resolved_elem_ty, gep_counter)?;
                }
            }
        }
        IrInstr::MapNew { result, .. } => {
            writeln!(out, "  %v{} = call ptr @iris_map_new()", result.0)?;
        }
        IrInstr::MapSet { map, key, value } => {
            let kv = val(*key);
            let kty = func.value_type(*key);
            let ptr_k = box_to_ptr(
                out,
                func,
                *key,
                &kv,
                kty,
                emitted_types.get(key).map(|s| s.as_str()),
                gep_counter,
            )?;
            let vv = val(*value);
            let vty = func.value_type(*value);
            let ptr_v = box_to_ptr(
                out,
                func,
                *value,
                &vv,
                vty,
                emitted_types.get(value).map(|s| s.as_str()),
                gep_counter,
            )?;
            writeln!(
                out,
                "  call void @iris_map_set(ptr {}, ptr {}, ptr {})",
                val(*map),
                ptr_k,
                ptr_v
            )?;
        }
        IrInstr::MapGet {
            result,
            map,
            key,
            val_ty: _val_ty,
        } => {
            let kv = val(*key);
            let kty = func.value_type(*key);
            let ptr_k = box_to_ptr(
                out,
                func,
                *key,
                &kv,
                kty,
                emitted_types.get(key).map(|s| s.as_str()),
                gep_counter,
            )?;
            // iris_map_get returns an option<T> as an IrisVal*; keep as ptr.
            // IsSome/OptionUnwrap will handle unboxing.
            writeln!(
                out,
                "  %v{} = call ptr @iris_map_get(ptr {}, ptr {})",
                result.0,
                val(*map),
                ptr_k
            )?;
        }
        IrInstr::MapContains { result, map, key } => {
            let kv = val(*key);
            let kty = func.value_type(*key);
            let ptr_k = box_to_ptr(
                out,
                func,
                *key,
                &kv,
                kty,
                emitted_types.get(key).map(|s| s.as_str()),
                gep_counter,
            )?;
            writeln!(
                out,
                "  %v{} = call i1 @iris_map_contains(ptr {}, ptr {})",
                result.0,
                val(*map),
                ptr_k
            )?;
        }
        IrInstr::MapRemove { map, key } => {
            let kv = val(*key);
            let kty = func.value_type(*key);
            let ptr_k = box_to_ptr(
                out,
                func,
                *key,
                &kv,
                kty,
                emitted_types.get(key).map(|s| s.as_str()),
                gep_counter,
            )?;
            writeln!(
                out,
                "  call void @iris_map_remove(ptr {}, ptr {})",
                val(*map),
                ptr_k
            )?;
        }
        IrInstr::MapLen { result, map } => {
            writeln!(
                out,
                "  %v{} = call i64 @iris_map_len(ptr {})",
                result.0,
                val(*map)
            )?;
        }

        // ── Closures ──────────────────────────────────────────────────────
        IrInstr::MakeClosure {
            result,
            fn_name,
            captures,
            ..
        } => {
            let mut cap_args = vec![];
            for c in captures {
                let cv = val(*c);
                let cty = func.value_type(*c);
                let ptr_c = box_to_ptr(
                    out,
                    func,
                    *c,
                    &cv,
                    cty,
                    emitted_types.get(c).map(|s| s.as_str()),
                    gep_counter,
                )?;
                cap_args.push(format!("ptr {}", ptr_c));
            }
            let mut args = vec![
                format!("ptr @{}", fn_name),
                format!("i32 {}", captures.len()),
            ];
            args.extend(cap_args);
            writeln!(
                out,
                "  %v{} = call ptr @iris_make_closure({})",
                result.0,
                args.join(", ")
            )?;
        }
        IrInstr::CallClosure {
            result,
            closure,
            args,
            result_ty,
            ..
        } => {
            let closure_v = val(*closure);

            // Extract function pointer from the closure struct.
            *gep_counter += 1;
            let fn_ptr_name = format!("%closure_fn{}", *gep_counter);
            writeln!(
                out,
                "  {} = call ptr @iris_closure_fn(ptr {})",
                fn_ptr_name, closure_v
            )?;

            // Build argument list.
            // Autodetect calling convention from the closure's fn_name:
            // - Lambda functions (__lambda_*) expect (ptr %env, user_args...)
            // - Regular functions expect only (user_args...)
            let is_lambda = closure_fn_map
                .get(closure)
                .map(|n| n.starts_with("__lambda_"))
                .unwrap_or(true); // default to lambda convention for safety
            let mut call_args: Vec<String> = Vec::new();
            if is_lambda {
                call_args.push(format!("ptr {}", closure_v));
            }

            // Add passed arguments (already native-typed in the IR).
            for a in args {
                let av = val(*a);
                let aty = func.value_type(*a);
                let llvm_ty = match aty {
                    Some(t) => llvm_type_complete(t)?,
                    None => "i64".to_owned(),
                };
                call_args.push(format!("{} {}", llvm_ty, av));
            }

            let ret_llvm_ty = llvm_type_complete(result_ty)?;
            let args_str = call_args.join(", ");

            if let Some(r) = result {
                if ret_llvm_ty == "void" {
                    writeln!(out, "  call void {}({})", fn_ptr_name, args_str)?;
                } else {
                    writeln!(
                        out,
                        "  %v{} = call {} {}({})",
                        r.0, ret_llvm_ty, fn_ptr_name, args_str
                    )?;
                }
            } else {
                writeln!(out, "  call void {}({})", fn_ptr_name, args_str)?;
            }
        }

        // ── Grad / Sparse ─────────────────────────────────────────────────
        IrInstr::MakeGrad {
            result,
            value,
            tangent,
            ..
        } => {
            // value and tangent are f64 dual-number components.
            writeln!(
                out,
                "  %v{} = call ptr @iris_make_grad(double {}, double {})",
                result.0,
                val(*value),
                val(*tangent)
            )?;
        }
        IrInstr::GradValue {
            result, operand, ..
        } => {
            writeln!(
                out,
                "  %v{} = call double @iris_grad_value(ptr {})",
                result.0,
                val(*operand)
            )?;
        }
        IrInstr::GradTangent {
            result, operand, ..
        } => {
            writeln!(
                out,
                "  %v{} = call double @iris_grad_tangent(ptr {})",
                result.0,
                val(*operand)
            )?;
        }
        IrInstr::TapeRecord {
            result,
            value,
            op,
            parents,
        } => {
            let primal = coerce_scalar_to_f64(*value, consts, func, gep_counter, out)?;
            let op_idx = str_table
                .get(op)
                .copied()
                .ok_or_else(|| CodegenError::Unsupported {
                    backend: "llvm".into(),
                    detail: format!("missing reverse-mode AD op string constant '{}'", op),
                })?;
            let op_len = op.len() + 1;
            *gep_counter += 1;
            let op_ptr = format!("%tape_op{}", gep_counter);
            writeln!(
                out,
                "  {} = getelementptr inbounds [{} x i8], ptr @.str.{}, i32 0, i32 0",
                op_ptr, op_len, op_idx
            )?;

            if parents.is_empty() {
                writeln!(
                    out,
                    "  %v{} = call ptr @iris_tape_record(double {}, ptr {}, i64 0, ptr null, ptr null)",
                    result.0, primal, op_ptr
                )?;
            } else {
                *gep_counter += 1;
                let handle_arr = format!("%tape_handles{}", gep_counter);
                *gep_counter += 1;
                let primal_arr = format!("%tape_parent_primals{}", gep_counter);
                writeln!(
                    out,
                    "  {} = alloca [{} x ptr], align 8",
                    handle_arr,
                    parents.len()
                )?;
                writeln!(
                    out,
                    "  {} = alloca [{} x double], align 8",
                    primal_arr,
                    parents.len()
                )?;

                for (idx, parent) in parents.iter().enumerate() {
                    *gep_counter += 1;
                    let handle_slot = format!("%tape_handle_slot{}", gep_counter);
                    writeln!(
                        out,
                        "  {} = getelementptr inbounds [{} x ptr], ptr {}, i64 0, i64 {}",
                        handle_slot,
                        parents.len(),
                        handle_arr,
                        idx
                    )?;
                    if emitted_types.get(parent).map(|s| s.as_str()) == Some("ptr") {
                        let parent_handle = coerce_to_type(
                            *parent,
                            "ptr",
                            consts,
                            func,
                            emitted_types,
                            gep_counter,
                            out,
                        )?;
                        writeln!(
                            out,
                            "  store ptr {}, ptr {}, align 8",
                            parent_handle, handle_slot
                        )?;
                    } else {
                        writeln!(out, "  store ptr null, ptr {}, align 8", handle_slot)?;
                    }

                    *gep_counter += 1;
                    let primal_slot = format!("%tape_primal_slot{}", gep_counter);
                    writeln!(
                        out,
                        "  {} = getelementptr inbounds [{} x double], ptr {}, i64 0, i64 {}",
                        primal_slot,
                        parents.len(),
                        primal_arr,
                        idx
                    )?;
                    if emitted_types.get(parent).map(|s| s.as_str()) == Some("ptr") {
                        writeln!(out, "  store double 0.0, ptr {}, align 8", primal_slot)?;
                    } else {
                        let parent_primal =
                            coerce_scalar_to_f64(*parent, consts, func, gep_counter, out)?;
                        writeln!(
                            out,
                            "  store double {}, ptr {}, align 8",
                            parent_primal, primal_slot
                        )?;
                    }
                }

                *gep_counter += 1;
                let handle_base = format!("%tape_handles_base{}", gep_counter);
                writeln!(
                    out,
                    "  {} = getelementptr inbounds [{} x ptr], ptr {}, i64 0, i64 0",
                    handle_base,
                    parents.len(),
                    handle_arr
                )?;
                *gep_counter += 1;
                let primal_base = format!("%tape_primal_base{}", gep_counter);
                writeln!(
                    out,
                    "  {} = getelementptr inbounds [{} x double], ptr {}, i64 0, i64 0",
                    primal_base,
                    parents.len(),
                    primal_arr
                )?;
                writeln!(
                    out,
                    "  %v{} = call ptr @iris_tape_record(double {}, ptr {}, i64 {}, ptr {}, ptr {})",
                    result.0,
                    primal,
                    op_ptr,
                    parents.len(),
                    handle_base,
                    primal_base
                )?;
            }
        }
        IrInstr::Backward { result, loss } => {
            if emitted_types.get(loss).map(|s| s.as_str()) != Some("ptr") {
                return Err(CodegenError::Unsupported {
                    backend: "llvm".into(),
                    detail: "reverse-mode AD backward requires a lowered tape handle; use tape(...) on leaf values before calling backward(...)".into(),
                });
            }
            let loss_handle =
                coerce_to_type(*loss, "ptr", consts, func, emitted_types, gep_counter, out)?;
            writeln!(out, "  call void @iris_backward(ptr {})", loss_handle)?;
            emit_zero_value(*result, emitted_types, out)?;
        }
        IrInstr::TapeGrad { result, tape_node } => {
            if emitted_types.get(tape_node).map(|s| s.as_str()) != Some("ptr") {
                return Err(CodegenError::Unsupported {
                    backend: "llvm".into(),
                    detail: "reverse-mode AD grad requires a lowered tape handle; use tape(...) on differentiable leaf values before calling grad(...)".into(),
                });
            }
            let tape_handle = coerce_to_type(
                *tape_node,
                "ptr",
                consts,
                func,
                emitted_types,
                gep_counter,
                out,
            )?;
            writeln!(
                out,
                "  %v{} = call double @iris_tape_grad(ptr {})",
                result.0, tape_handle
            )?;
        }
        IrInstr::Sparsify {
            result, operand, ..
        } => {
            // `iris_sparsify` takes an IrisList*. A fixed-size array literal is
            // lowered to an `alloca [N x T]`, i.e. a raw data pointer with no list
            // header, so passing it here made the runtime read a bogus header and
            // report an empty sparse value — `densify(sparsify([1,0,3,0,5]))`
            // returned 0 natively while the interpreter returned 3. The runtime
            // provides array-specific entry points that take a pointer and a
            // length; use them when the operand really is an array.
            match func.value_type(*operand) {
                Some(IrType::Array { elem, len }) => {
                    let helper = match **elem {
                        IrType::Scalar(DType::F64) | IrType::Scalar(DType::F32) => {
                            "iris_sparsify_f64_array"
                        }
                        _ => "iris_sparsify_i64_array",
                    };
                    writeln!(
                        out,
                        "  %v{} = call ptr @{}(ptr {}, i64 {})",
                        result.0,
                        helper,
                        val(*operand),
                        len
                    )?;
                }
                _ => {
                    writeln!(
                        out,
                        "  %v{} = call ptr @iris_sparsify(ptr {})",
                        result.0,
                        val(*operand)
                    )?;
                }
            }
        }
        IrInstr::Densify {
            result, operand, ..
        } => {
            // Reconstructs the dense list, which is what the name, the documented
            // signature and this runtime function all mean. The non-zero count is
            // SparseNnz below.
            writeln!(
                out,
                "  %v{} = call ptr @iris_densify(ptr {})",
                result.0,
                val(*operand)
            )?;
        }
        IrInstr::SparseNnz { result, operand } => {
            writeln!(
                out,
                "  %v{} = call i64 @iris_sparse_nnz(ptr {})",
                result.0,
                val(*operand)
            )?;
        }

        // ── I/O ────────────────────────────────────────────────────────────
        IrInstr::Print { operand } => {
            // Typed print: use specialised helper for scalars.
            let oty = inferred_value_type(func, *operand, func.value_type(*operand));
            match oty.as_ref() {
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
                    writeln!(out, "  %bool{} = zext i1 {} to i32", *gep_counter, val(*operand))?;
                    writeln!(out, "  call void @iris_print_bool(i32 %bool{})", *gep_counter)?;
                    *gep_counter += 1;
                }
                Some(IrType::Str) => {
                    writeln!(out, "  call void @iris_print_str(ptr {})", val(*operand))?;
                }
                // Option<Scalar> — MapGet/ListGet eagerly unbox to the scalar type.
                Some(IrType::Option(inner)) => match inner.as_ref() {
                    IrType::Scalar(DType::I64) => {
                        writeln!(out, "  call void @iris_print_i64(i64 {})", val(*operand))?;
                    }
                    IrType::Scalar(DType::I32) => {
                        writeln!(out, "  call void @iris_print_i32(i32 {})", val(*operand))?;
                    }
                    IrType::Scalar(DType::F64) => {
                        writeln!(out, "  call void @iris_print_f64(double {})", val(*operand))?;
                    }
                    IrType::Scalar(DType::F32) => {
                        writeln!(out, "  call void @iris_print_f32(float {})", val(*operand))?;
                    }
                    IrType::Scalar(DType::Bool) => {
                        writeln!(out, "  %bool{} = zext i1 {} to i32", *gep_counter, val(*operand))?;
                        writeln!(out, "  call void @iris_print_bool(i32 %bool{})", *gep_counter)?;
                        *gep_counter += 1;
                    }
                    IrType::Str => {
                        writeln!(out, "  call void @iris_print_str(ptr {})", val(*operand))?;
                    }
                    _ => {
                        writeln!(out, "  call void @iris_print(ptr {})", val(*operand))?;
                    }
                },
                _ => {
                    // IR type unknown — check emitted LLVM type; ptr means string
                    let ety = emitted_types.get(operand).map(|s| s.as_str());
                    match ety {
                        Some("i64") => {
                            writeln!(out, "  call void @iris_print_i64(i64 {})", val(*operand))?;
                        }
                        Some("i32") => {
                            writeln!(out, "  call void @iris_print_i32(i32 {})", val(*operand))?;
                        }
                        Some("double") => {
                            writeln!(out, "  call void @iris_print_f64(double {})", val(*operand))?;
                        }
                        Some("float") => {
                            writeln!(out, "  call void @iris_print_f32(float {})", val(*operand))?;
                        }
                        Some("i1") => {
                            writeln!(out, "  %bool{} = zext i1 {} to i32", *gep_counter, val(*operand))?;
                            writeln!(out, "  call void @iris_print_bool(i32 %bool{})", *gep_counter)?;
                            *gep_counter += 1;
                        }
                        Some("ptr") => {
                            // For ptr-typed values (list, str, map, etc.), box and use generic print.
                            let boxed = box_to_ptr(out, func, *operand, &val(*operand), func.value_type(*operand), emitted_types.get(operand).map(|s| s.as_str()), gep_counter)?;
                            writeln!(out, "  call void @iris_print(ptr {})", boxed)?;
                        }
                        _ => {
                            writeln!(out, "  call void @iris_print(ptr {})", val(*operand))?;
                        }
                    }
                }
            }
        }
        IrInstr::Panic { msg, .. } => {
            // iris_panic is declared noreturn; LLVM will treat this as a terminating call.
            // We emit unreachable so the block has a proper terminator after it.
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
            writeln!(
                out,
                "  %v{} = call ptr @iris_parse_i64(ptr {})",
                result.0,
                val(*operand)
            )?;
        }
        IrInstr::ParseF64 { result, operand } => {
            writeln!(
                out,
                "  %v{} = call ptr @iris_parse_f64(ptr {})",
                result.0,
                val(*operand)
            )?;
        }
        IrInstr::ValueToStr { result, operand } => {
            let oty = inferred_value_type(func, *operand, func.value_type(*operand));
            // Check the actual emitted LLVM type; if it's ptr but IR thinks scalar,
            // insert a ptrtoint before calling the typed to_str function.
            let emitted_ty = emitted_types.get(operand).map(|s| s.as_str());
            match oty.as_ref() {
                Some(IrType::Scalar(DType::I64)) => {
                    let arg = if emitted_ty == Some("ptr") {
                        let tmp = format!("%cast{}", gep_counter);
                        *gep_counter += 1;
                        writeln!(out, "  {} = ptrtoint ptr {} to i64", tmp, val(*operand))?;
                        tmp
                    } else {
                        val(*operand)
                    };
                    writeln!(
                        out,
                        "  %v{} = call ptr @iris_i64_to_str(i64 {})",
                        result.0, arg
                    )?;
                }
                Some(IrType::Scalar(DType::I32)) => {
                    let arg = if emitted_ty == Some("ptr") {
                        let tmp = format!("%cast{}", gep_counter);
                        *gep_counter += 1;
                        writeln!(out, "  {} = ptrtoint ptr {} to i32", tmp, val(*operand))?;
                        tmp
                    } else {
                        val(*operand)
                    };
                    writeln!(
                        out,
                        "  %v{} = call ptr @iris_i32_to_str(i32 {})",
                        result.0, arg
                    )?;
                }
                Some(IrType::Scalar(DType::F64)) => {
                    writeln!(
                        out,
                        "  %v{} = call ptr @iris_f64_to_str(double {})",
                        result.0,
                        val(*operand)
                    )?;
                }
                Some(IrType::Scalar(DType::F32)) => {
                    writeln!(
                        out,
                        "  %v{} = call ptr @iris_f32_to_str(float {})",
                        result.0,
                        val(*operand)
                    )?;
                }
                Some(IrType::Scalar(DType::Bool)) => {
                    writeln!(
                        out,
                        "  %bool{} = zext i1 {} to i32",
                        gep_counter,
                        val(*operand)
                    )?;
                    writeln!(
                        out,
                        "  %v{} = call ptr @iris_bool_to_str(i32 %bool{})",
                        result.0,
                        *gep_counter
                    )?;
                    *gep_counter += 1;
                }
                Some(IrType::Str) => {
                    writeln!(
                        out,
                        "  %v{} = call ptr @iris_str_to_str(ptr {})",
                        result.0,
                        val(*operand)
                    )?;
                }
                _ => {
                    // IR type unknown — fall back to emitted LLVM type
                    match emitted_ty {
                        Some("i64") => {
                            writeln!(
                                out,
                                "  %v{} = call ptr @iris_i64_to_str(i64 {})",
                                result.0,
                                val(*operand)
                            )?;
                        }
                        Some("i32") => {
                            writeln!(
                                out,
                                "  %v{} = call ptr @iris_i32_to_str(i32 {})",
                                result.0,
                                val(*operand)
                            )?;
                        }
                        Some("double") => {
                            writeln!(
                                out,
                                "  %v{} = call ptr @iris_f64_to_str(double {})",
                                result.0,
                                val(*operand)
                            )?;
                        }
                        Some("float") => {
                            writeln!(
                                out,
                                "  %v{} = call ptr @iris_f32_to_str(float {})",
                                result.0,
                                val(*operand)
                            )?;
                        }
                        Some("i1") => {
                            writeln!(
                                out,
                                "  %bool{} = zext i1 {} to i32",
                                gep_counter,
                                val(*operand)
                            )?;
                            writeln!(
                                out,
                                "  %v{} = call ptr @iris_bool_to_str(i32 %bool{})",
                                result.0,
                                *gep_counter
                            )?;
                            *gep_counter += 1;
                        }
                        _ => {
                            writeln!(
                                out,
                                "  %v{} = call ptr @iris_value_to_str(ptr {})",
                                result.0,
                                val(*operand)
                            )?;
                        }
                    }
                }
            }
        }

        // ── Barrier ───────────────────────────────────────────────────────
        IrInstr::Barrier => {
            writeln!(out, "  call void @iris_barrier()")?;
        }

        // ── Phase 56: File I/O ─────────────────────────────────────────────
        IrInstr::FileReadAll { result, path } => {
            // `iris_file_read_all` returns an `IrisResult*` that is already
            // ok-or-err — see iris_runtime.c, which returns
            // `iris_make_err(...)` for every failure path.
            //
            // This used to null-check the return and re-wrap it: on the
            // non-null branch it did `iris_make_ok(iris_box_str(result))`. Since
            // an err result is *also* non-null, a missing file took the ok
            // branch and got wrapped a second time, so `is_ok(...)` reported
            // `true` for a file that does not exist. Codegen was written against
            // an older `char*`-returning signature and never updated.
            //
            // Pass the result through unchanged.
            writeln!(
                out,
                "  %v{} = call ptr @iris_file_read_all(ptr {})",
                result.0,
                val(*path)
            )?;
        }
        IrInstr::FileWriteAll {
            result,
            path,
            content,
        } => {
            // Same as FileReadAll above: `iris_file_write_all` returns an
            // `IrisResult*` that is already ok-or-err, and an err result is
            // non-null, so the old null-check-and-rewrap reported success for
            // every failed write.
            writeln!(
                out,
                "  %v{} = call ptr @iris_file_write_all(ptr {}, ptr {})",
                result.0,
                val(*path),
                val(*content)
            )?;
        }
        IrInstr::FileExists { result, path } => {
            writeln!(
                out,
                "  %v{}_raw = call i32 @iris_file_exists(ptr {})",
                result.0,
                val(*path)
            )?;
            writeln!(
                out,
                "  %v{} = trunc i32 %v{}_raw to i1",
                result.0, result.0
            )?;
        }
        IrInstr::FileLines { result, path } => {
            writeln!(
                out,
                "  %v{} = call ptr @iris_file_lines(ptr {})",
                result.0,
                val(*path)
            )?;
        }

        // Database operations
        IrInstr::DbOpen { result, path } => {
            writeln!(
                out,
                "  %v{} = call i64 @iris_db_open(ptr {})",
                result.0,
                val(*path)
            )?;
        }
        IrInstr::DbExec { result, db, sql } => {
            writeln!(
                out,
                "  %v{} = call i64 @iris_db_exec(i64 {}, ptr {})",
                result.0,
                val(*db),
                val(*sql)
            )?;
        }
        IrInstr::DbExecParams {
            result,
            db,
            sql,
            params,
        } => {
            writeln!(
                out,
                "  %v{} = call i64 @iris_db_exec_params(i64 {}, ptr {}, ptr {})",
                result.0,
                val(*db),
                val(*sql),
                val(*params)
            )?;
        }
        IrInstr::DbQuery { result, db, sql } => {
            writeln!(
                out,
                "  %v{} = call ptr @iris_db_query(i64 {}, ptr {})",
                result.0,
                val(*db),
                val(*sql)
            )?;
        }
        IrInstr::DbQueryParams {
            result,
            db,
            sql,
            params,
        } => {
            writeln!(
                out,
                "  %v{} = call ptr @iris_db_query_params(i64 {}, ptr {}, ptr {})",
                result.0,
                val(*db),
                val(*sql),
                val(*params)
            )?;
        }
        IrInstr::DbClose { result, db } => {
            writeln!(
                out,
                "  %v{} = call i64 @iris_db_close(i64 {})",
                result.0,
                val(*db)
            )?;
        }

        // ── Phase 58: Extended collections ────────────────────────────────
        IrInstr::ListContains {
            result,
            list,
            value,
        } => {
            let vv = val(*value);
            let vty = func.value_type(*value);
            let ptr_v = box_to_ptr(
                out,
                func,
                *value,
                &vv,
                vty,
                emitted_types.get(value).map(|s| s.as_str()),
                gep_counter,
            )?;
            writeln!(
                out,
                "  %v{} = call i1 @iris_list_contains(ptr {}, ptr {})",
                result.0,
                val(*list),
                ptr_v
            )?;
        }
        IrInstr::ListSort { list } => {
            writeln!(out, "  call void @iris_list_sort(ptr {})", val(*list))?;
        }
        IrInstr::MapKeys { result, map } => {
            writeln!(
                out,
                "  %v{} = call ptr @iris_map_keys(ptr {})",
                result.0,
                val(*map)
            )?;
        }
        IrInstr::MapValues { result, map } => {
            writeln!(
                out,
                "  %v{} = call ptr @iris_map_values(ptr {})",
                result.0,
                val(*map)
            )?;
        }
        IrInstr::ListConcat { result, lhs, rhs } => {
            writeln!(
                out,
                "  %v{} = call ptr @iris_list_concat(ptr {}, ptr {})",
                result.0,
                val(*lhs),
                val(*rhs)
            )?;
        }
        IrInstr::ListSlice {
            result,
            list,
            start,
            end,
        } => {
            writeln!(
                out,
                "  %v{} = call ptr @iris_list_slice(ptr {}, i64 {}, i64 {})",
                result.0,
                val(*list),
                val(*start),
                val(*end)
            )?;
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
            writeln!(
                out,
                "  %v{} = call ptr @iris_env_var(ptr {})",
                result.0,
                val(*name)
            )?;
        }
        // Phase 61: Pattern matching helpers
        IrInstr::GetVariantTag { result, operand } => {
            writeln!(
                out,
                "  %v{} = call i64 @iris_get_variant_tag(ptr {})",
                result.0,
                val(*operand)
            )?;
        }
        IrInstr::StrEq { result, lhs, rhs } => {
            writeln!(
                out,
                "  %v{} = call i1 @iris_str_eq(ptr {}, ptr {})",
                result.0,
                val(*lhs),
                val(*rhs)
            )?;
        }
        // Phase 83: GC retain/release
        IrInstr::Retain { ptr } => {
            let emitted_ptr = emitted_types.get(ptr).map(|s| s == "ptr").unwrap_or(false);
            if emitted_ptr {
                if let Some(kind) = func.value_type(*ptr).and_then(runtime_rc_kind_for_type) {
                    writeln!(
                        out,
                        "  call void @iris_retain_kind(ptr {}, i32 {})",
                        val(*ptr),
                        kind
                    )?;
                }
            }
        }
        IrInstr::Release { ptr, ty } => {
            let emitted_ptr = emitted_types.get(ptr).map(|s| s == "ptr").unwrap_or(false);
            if emitted_ptr {
                let kind = runtime_rc_kind_for_type(ty)
                    .or_else(|| func.value_type(*ptr).and_then(runtime_rc_kind_for_type));
                if let Some(kind) = kind {
                    writeln!(
                        out,
                        "  call void @iris_release_kind(ptr {}, i32 {})",
                        val(*ptr),
                        kind
                    )?;
                }
            }
        }
        // Phase 81: FFI extern calls — route through effect handler dispatch.
        IrInstr::CallExtern {
            result,
            name,
            args,
            ret_ty,
        } => {
            let llvm_ret = llvm_type_complete(ret_ty).unwrap_or_else(|_| "ptr".to_owned());
            let nargs = args.len();

            // Call the extern directly unless some handler in this module could
            // actually intercept it.
            //
            // iris_effect_dispatch_or_call casts every target to
            //   int64_t (*)(int64_t, int64_t, ...)
            // regardless of its real signature. On x86-64 that is an ABI
            // violation for floating point: f64 arguments belong in xmm0-3, not
            // the integer registers, and an f64 result is returned in xmm0, not
            // rax. So every extern taking or returning f64 read and wrote garbage
            // — `adaptive_learning_rate` returned 3.39e-312 instead of 0.01, a
            // denormal formed by reading an integer bit pattern as a double.
            //
            // Whether interception is possible is a static property: an effect can
            // only be handled by a `handle` block naming it, and those appear as
            // PushHandler arms. If no arm in the module names this extern, no
            // handler can ever intercept it, so a direct typed call is both
            // correct and cheaper. Externs that *are* handled still go through the
            // trampoline, and the f64 limitation remains for those.
            let interceptable = module.functions().iter().any(|f| {
                f.blocks().iter().any(|b| {
                    b.instrs.iter().any(|i| match i {
                        IrInstr::PushHandler { arms } => {
                            arms.iter().any(|a| a.effect_name == *name)
                        }
                        _ => false,
                    })
                })
            });

            if !interceptable {
                let arg_list = args
                    .iter()
                    .map(|a| {
                        let ty = emitted_types
                            .get(a)
                            .cloned()
                            .or_else(|| {
                                func.value_type(*a)
                                    .and_then(|t| llvm_type_complete(t).ok())
                            })
                            .unwrap_or_else(|| "i64".to_owned());
                        format!("{} {}", ty, val(*a))
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                match result {
                    Some(r) if llvm_ret != "void" => {
                        writeln!(
                            out,
                            "  %v{} = call {} @{}({})",
                            r.0, llvm_ret, name, arg_list
                        )?;
                    }
                    _ => {
                        writeln!(out, "  call {} @{}({})", llvm_ret, name, arg_list)?;
                    }
                }
                return Ok(());
            }

            // A miss means the string-collection pass never saw this extern call.
            // Falling back to index 0 silently passed an unrelated string as the
            // effect name, which surfaced at runtime as "no handler for effect ','"
            // — a wrong answer instead of an error. Fail loudly instead.
            let name_idx = *str_table.get(name).ok_or_else(|| CodegenError::Unsupported {
                backend: "llvm".into(),
                detail: format!(
                    "internal: extern name '{}' is missing from the string constant table",
                    name
                ),
            })?;

            // Pack args into an i64 array for the dispatch function.
            *gep_counter += 1;
            let args_buf_idx = *gep_counter;
            if nargs > 0 {
                writeln!(
                    out,
                    "  %_eff_args_{} = alloca [{} x i64], align 8",
                    args_buf_idx, nargs
                )?;
                for (i, a) in args.iter().enumerate() {
                    let arg_val = val(*a);
                    let arg_ty = func.value_type(*a).cloned().unwrap_or(IrType::Infer);
                    let arg_llvm_ty =
                        llvm_type_complete(&arg_ty).unwrap_or_else(|_| "ptr".to_owned());
                    let i64_val = if arg_llvm_ty == "i64" {
                        arg_val.to_string()
                    } else if arg_llvm_ty == "double" {
                        *gep_counter += 1;
                        let tmp = format!("%_eff_f2i_{}", gep_counter);
                        writeln!(
                            out,
                            "  {} = bitcast double {} to i64",
                            tmp, arg_val
                        )?;
                        tmp
                    } else {
                        // ptr or other → ptrtoint to i64
                        *gep_counter += 1;
                        let tmp = format!("%_eff_p2i_{}", gep_counter);
                        writeln!(
                            out,
                            "  {} = ptrtoint {} {} to i64",
                            tmp, arg_llvm_ty, arg_val
                        )?;
                        tmp
                    };
                    *gep_counter += 1;
                    writeln!(
                        out,
                        "  %_eff_ap_{} = getelementptr [{} x i64], ptr %_eff_args_{}, i32 0, i32 {}",
                        gep_counter, nargs, args_buf_idx, i
                    )?;
                    writeln!(
                        out,
                        "  store i64 {}, ptr %_eff_ap_{}",
                        i64_val, gep_counter
                    )?;
                }
            }

            // Allocate a Continuation on the stack for resume support.
            *gep_counter += 1;
            let cont_var = format!("%_cont_{}", gep_counter);
            writeln!(
                out,
                "  {} = alloca %Continuation, align 8",
                cont_var
            )?;
            writeln!(
                out,
                "  store %Continuation {{ i32 0, i64 0 }}, ptr {}",
                cont_var
            )?;

            // Call the dispatch function.
            let args_buf_str = if nargs > 0 {
                format!("%_eff_args_{}", args_buf_idx)
            } else {
                "null".to_string()
            };

            // LLVM has no copy instruction, so `%vN = %_eff_result_M` is not
            // valid IR. When the dispatch already returns the expected type,
            // name the call's destination directly rather than aliasing it
            // afterwards; every other type still goes through a real cast below.
            let result_i64 = match result {
                Some(r) if llvm_ret == "i64" => format!("%v{}", r.0),
                _ => format!("%_eff_result_{}", gep_counter),
            };
            // Pass the symbol's address in both cases. An IRIS-defined function
            // is referenced directly; anything else is the `extern_weak`
            // declaration emitted above, whose address is the real
            // implementation when one is linked and null when it is not.
            //
            // This previously passed `ptr null` for everything not defined in the
            // module, which is every C extern — so no real FFI function could be
            // reached through this path at all. It failed at runtime as "no
            // handler for effect 'iris_mlrt_onnx_load' and no real
            // implementation" even though the runtime exports that symbol.
            let real_fn_ref = format!("ptr @{}", name);
            writeln!(
                out,
                "  {} = call i64 @iris_effect_dispatch_or_call(ptr @.str.{}, {}, ptr {}, i64 {}, ptr {})",
                result_i64, name_idx, real_fn_ref, cont_var, nargs, args_buf_str
            )?;

            // Cast the i64 result to the expected return type.
            if let Some(r) = result {
                if llvm_ret == "i64" {
                    // Nothing to do: the call above wrote straight into %v{r}.
                } else if llvm_ret == "ptr" {
                    writeln!(
                        out,
                        "  %v{} = inttoptr i64 {} to ptr",
                        r.0, result_i64
                    )?;
                } else if llvm_ret == "double" {
                    writeln!(
                        out,
                        "  %v{} = bitcast i64 {} to double",
                        r.0, result_i64
                    )?;
                } else if llvm_ret == "i32" {
                    writeln!(
                        out,
                        "  %v{} = trunc i64 {} to i32",
                        r.0, result_i64
                    )?;
                } else if llvm_ret == "i1" {
                    writeln!(
                        out,
                        "  %v{} = trunc i64 {} to i1",
                        r.0, result_i64
                    )?;
                } else {
                    // Fallback: bitcast or inttoptr
                    writeln!(
                        out,
                        "  %v{} = inttoptr i64 {} to {}",
                        r.0, result_i64, llvm_ret
                    )?;
                }
            }
        }
        // Phase 88: TCP network I/O
        IrInstr::TcpConnect { result, host, port } => {
            writeln!(
                out,
                "  %v{} = call i64 @iris_tcp_connect(ptr {}, i64 {})",
                result.0,
                val(*host),
                val(*port)
            )?;
        }
        IrInstr::TcpListen { result, port } => {
            writeln!(
                out,
                "  %v{} = call i64 @iris_tcp_listen(i64 {})",
                result.0,
                val(*port)
            )?;
        }
        IrInstr::TcpAccept { result, listener } => {
            writeln!(
                out,
                "  %v{} = call i64 @iris_tcp_accept(i64 {})",
                result.0,
                val(*listener)
            )?;
        }
        IrInstr::TcpRead { result, conn } => {
            writeln!(
                out,
                "  %v{} = call ptr @iris_tcp_read(i64 {})",
                result.0,
                val(*conn)
            )?;
        }
        IrInstr::TcpWrite { conn, data } => {
            writeln!(
                out,
                "  call void @iris_tcp_write(i64 {}, ptr {})",
                val(*conn),
                val(*data)
            )?;
        }
        IrInstr::TcpClose { conn } => {
            writeln!(out, "  call void @iris_tcp_close(i64 {})", val(*conn))?;
        }
        IrInstr::StrSplit {
            result,
            str_val,
            delim,
        } => {
            writeln!(
                out,
                "  %v{} = call ptr @iris_str_split(ptr {}, ptr {})",
                result.0,
                val(*str_val),
                val(*delim)
            )?;
        }
        IrInstr::StrJoin {
            result,
            list_val,
            delim,
        } => {
            writeln!(
                out,
                "  %v{} = call ptr @iris_str_join(ptr {}, ptr {})",
                result.0,
                val(*list_val),
                val(*delim)
            )?;
        }
        IrInstr::NowMs { result } => {
            writeln!(out, "  %v{} = call i64 @iris_now_ms()", result.0)?;
        }
        IrInstr::SleepMs { result, ms } => {
            writeln!(out, "  call void @iris_sleep_ms(i64 {})", val(*ms))?;
            writeln!(out, "  %v{} = add i64 0, 0", result.0)?;
        }
        // ── Trait objects ─────────────────────────────────────────────────
        //
        // A trait object is a two-word value: `{ ptr data, ptr vtable }`. These
        // two instructions were empty no-ops, so every value they should have
        // defined dangled and any module using `dyn Trait` failed verification.
        // `--emit eval` builds natively and falls back to the interpreter, so
        // the failure looked like an interpreter-only feature rather than a
        // missing backend. See known-issues #18b.
        IrInstr::MakeTraitObject {
            result,
            value,
            target_trait,
            concrete_ty,
            ..
        } => {
            let data = val(*value);
            writeln!(out, "  %v{} = call ptr @malloc(i64 16)", result.0)?;
            writeln!(out, "  store ptr {}, ptr %v{}, align 8", data, result.0)?;
            writeln!(
                out,
                "  %tovt{r} = getelementptr inbounds ptr, ptr %v{r}, i32 1",
                r = result.0
            )?;
            writeln!(
                out,
                "  store ptr @vt_{}_{}, ptr %tovt{}, align 8",
                target_trait, concrete_ty, result.0
            )?;
        }

        IrInstr::DynCall {
            result,
            obj,
            method_name,
            args,
            result_ty,
        } => {
            // The trait object's *type* carries the method list in vtable slot
            // order, so the slot index is a static lookup and no runtime name
            // matching is needed.
            let obj_ty = inferred_value_type(func, *obj, func.value_type(*obj));
            let methods = match obj_ty.as_ref() {
                Some(IrType::TraitObject { methods, .. }) => methods.clone(),
                _ => {
                    return Err(CodegenError::Unsupported {
                        backend: "llvm".into(),
                        detail: format!(
                            "dynamic call to `{}` on a value that is not a trait object",
                            method_name
                        ),
                    })
                }
            };
            let slot = methods
                .iter()
                .position(|m| &m.name == method_name)
                .ok_or_else(|| CodegenError::Unsupported {
                    backend: "llvm".into(),
                    detail: format!(
                        "method `{}` is not a slot of this trait object (has: {})",
                        method_name,
                        methods
                            .iter()
                            .map(|m| m.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                })?;

            let n = *gep_counter;
            *gep_counter += 1;
            let ov = val(*obj);
            // self data pointer
            writeln!(out, "  %dcd{} = load ptr, ptr {}, align 8", n, ov)?;
            // vtable pointer
            writeln!(
                out,
                "  %dcvp{} = getelementptr inbounds ptr, ptr {}, i32 1",
                n, ov
            )?;
            writeln!(out, "  %dcvt{n} = load ptr, ptr %dcvp{n}, align 8", n = n)?;
            // slot -> function pointer
            writeln!(
                out,
                "  %dcs{n} = getelementptr inbounds [{len} x ptr], ptr %dcvt{n}, i32 0, i32 {slot}",
                n = n,
                len = methods.len().max(1),
                slot = slot
            )?;
            writeln!(out, "  %dcf{n} = load ptr, ptr %dcs{n}, align 8", n = n)?;

            // Build the argument list: `self` first, then the call's own args.
            let ret_s = llvm_type_complete(result_ty)?;
            let mut arg_s: Vec<String> = vec![format!("ptr %dcd{}", n)];
            for a in args {
                let aty = emitted_types
                    .get(a)
                    .cloned()
                    .or_else(|| {
                        func.value_type(*a)
                            .and_then(|t| llvm_type_complete(t).ok())
                    })
                    .unwrap_or_else(|| "ptr".to_owned());
                arg_s.push(format!("{} {}", aty, val(*a)));
            }
            writeln!(
                out,
                "  %v{} = call {} %dcf{}({})",
                result.0,
                ret_s,
                n,
                arg_s.join(", ")
            )?;
            let _ = module;
        }
        // Phase 113: TaskGroup instructions (no-op in simple codegen)
        IrInstr::TaskGroupNew { .. } => {}
        IrInstr::TaskGroupSpawn { .. } => {}
        IrInstr::TaskGroupJoin { .. } => {}
        IrInstr::TaskGroupCancel { .. } => {}
        // Phase 104: BuiltinCall — unified dispatch for new builtins
        IrInstr::BuiltinCall {
            result,
            name,
            args,
            result_ty,
        } => {
            let fn_name = format!("iris_{}", name);
            // Use each arg's emitted LLVM type so scalars (i64, double, i1) are
            // passed with the correct type instead of always "ptr".
            let mut arg_strs: Vec<String> = args
                .iter()
                .map(|a| {
                    let ty_s = emitted_types.get(a).map(|s| s.as_str()).unwrap_or("ptr");
                    format!("{} {}", ty_s, val(*a))
                })
                .collect();

            // `iris_select(int64_t n, ...)` takes the channel count first — a
            // variadic callee cannot discover its own argument count. Codegen
            // passed only the channels, so the first channel *pointer* was read
            // as `n`, giving an astronomically large count; `va_arg` then treated
            // the second channel as index 0 and returned the wrong index (0
            // instead of 1) while walking off the end of the argument list for
            // any longer select. Prepend the count.
            if name == "select" {
                arg_strs.insert(0, format!("i64 {}", args.len()));
            }

            // `iris_json_stringify(IrisVal*)` takes a *boxed* value, but the
            // generic path above passes each argument in its own emitted form —
            // for a `str` that is the raw `char*` of the string constant. The C
            // side then read those bytes as an IrisVal header, got a garbage tag
            // and returned the literal string "null" for every input.
            if name == "json_stringify" {
                arg_strs.clear();
                for a in args {
                    let av = val(*a);
                    let aty = func.value_type(*a);
                    let boxed = box_to_ptr(
                        out,
                        func,
                        *a,
                        &av,
                        aty,
                        emitted_types.get(a).map(|s| s.as_str()),
                        gep_counter,
                    )?;
                    arg_strs.push(format!("ptr {}", boxed));
                }
            }
            // Determine LLVM return type from result_ty
            // Bool builtins are declared as returning i32 (C int) — call as i32
            // then truncate to i1 for IRIS-level boolean usage.
            let is_bool = matches!(result_ty, IrType::Scalar(DType::Bool));
            let ret_llvm = match result_ty {
                IrType::Scalar(DType::I64) => "i64",
                IrType::Scalar(DType::F64) => "double",
                IrType::Scalar(DType::Bool) => "i32",
                _ => "ptr", // str, list, map, infer → ptr
            };
            if is_bool {
                writeln!(
                    out,
                    "  %v{}_raw = call {} @{}({})",
                    result.0,
                    ret_llvm,
                    fn_name,
                    arg_strs.join(", ")
                )?;
                writeln!(
                    out,
                    "  %v{} = trunc i32 %v{}_raw to i1",
                    result.0, result.0
                )?;
            } else {
                writeln!(
                    out,
                    "  %v{} = call {} @{}({})",
                    result.0,
                    ret_llvm,
                    fn_name,
                    arg_strs.join(", ")
                )?;
            }
        }
        IrInstr::PushHandler { arms } => {
            // Push each arm onto the runtime handler stack.
            for arm in arms {
                let name_idx = str_table.get(&arm.effect_name).copied().unwrap_or(0);
                let fn_idx = str_table.get(&format!("__fn__{}", arm.func_name)).copied().unwrap_or(0);
                let has_resume = if arm.has_resume { 1i32 } else { 0i32 };
                writeln!(
                    out,
                    "  call void @iris_push_handler_arm(ptr @.str.{}, ptr @.str.{}, i64 {}, i32 {})",
                    name_idx, fn_idx, arm.num_args, has_resume
                )?;
                // Store the actual function pointer for native dispatch.
                writeln!(
                    out,
                    "  call void @iris_push_handler_fn(ptr @{})",
                    arm.func_name
                )?;
            }
            writeln!(out, "  call void @iris_push_handler_frame()")?;
        }
        IrInstr::PopHandler => {
            writeln!(out, "  call void @iris_pop_handler()")?;
        }
        IrInstr::ResumeCont { cont, value, result } => {
            let val_ty = func.value_type(*value).cloned().unwrap_or(IrType::Infer);
            let raw_v = val(*value);
            let val_i64 = if raw_v.starts_with("%v") || raw_v.starts_with('@') || llvm_type_complete(&val_ty).unwrap_or_default() == "ptr" {
                *gep_counter += 1;
                let tmp = format!("%_res_i64_{}", gep_counter);
                writeln!(out, "  {} = ptrtoint ptr {} to i64", tmp, raw_v)?;
                tmp
            } else {
                raw_v
            };
            writeln!(
                out,
                "  call void @iris_resume_cont(ptr {}, i64 {})",
                val(*cont), val_i64
            )?;
            // The handler should return immediately after resume.
            // Store the value as the result so the handler can return it.
            let res_ty = func.value_type(*result).ok_or_else(|| CodegenError::Unsupported {
                backend: "llvm".into(),
                detail: format!("ResumeCont result v{} has no type", result.0),
            })?;
            let llvm_ret = llvm_type_complete(&res_ty)?;
            // Coerce the value to the proper type if needed
            write!(out, "  {} = ", val(*result))?;
            if llvm_ret == "i64" {
                // Already the carrier width. `trunc i64 %x to i64` is rejected
                // by LLVM as an invalid cast, and this path is reached whenever
                // a handler *uses* its parameter: handler bindings carry no
                // declared type, so they default to i64 (known-issues #31) and
                // land here. `test_resume_handler.iris` passed natively only
                // because its handler ignores its argument.
                writeln!(out, "add i64 {}, 0", val_i64)?;
            } else if llvm_ret.starts_with('i') && llvm_ret != "i1" {
                writeln!(out, "trunc i64 {} to {}", val_i64, llvm_ret)?;
            } else if llvm_ret == "ptr" {
                writeln!(out, "inttoptr i64 {} to ptr", val_i64)?;
            } else if llvm_ret == "double" {
                writeln!(out, "bitcast i64 {} to double", val_i64)?;
            } else if llvm_ret == "float" {
                writeln!(out, "bitcast i64 {} to float", val_i64)?;
            } else if llvm_ret == "i1" {
                writeln!(out, "trunc i64 {} to i1", val_i64)?;
            } else {
                writeln!(out, "inttoptr i64 {} to ptr", val_i64)?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Complete LLVM type mapping — includes proper struct, array, and scalar types.
/// Emits a field-by-field equality test for two record pointers, returning the
/// name of the resulting `i1`.
///
/// Records are `ptr` in LLVM, so there is no single instruction that compares
/// them. Before this existed, `==` on two records fell into the string path and
/// `strcmp`-ed the raw struct bytes, which reported records with *different*
/// contents as equal. Comparing the pointers instead would be no better: two
/// separately built records with identical fields would compare unequal, and
/// the interpreter compares by value.
///
/// Scalar fields compare with `icmp`/`fcmp`, `str` fields with `iris_str_eq`,
/// and nested records recurse. Anything else is refused rather than guessed at
/// -- a wrong answer here is invisible, so an explicit error is worth more than
/// coverage. See known-issues #26.
fn emit_struct_eq(
    ty: &IrType,
    lhs_ptr: &str,
    rhs_ptr: &str,
    gep_counter: &mut u32,
    out: &mut String,
) -> Result<String, CodegenError> {
    let (name, fields) = match ty {
        IrType::Struct { name, fields } => (name, fields),
        other => {
            return Err(CodegenError::Unsupported {
                backend: "llvm".into(),
                detail: format!("structural equality is not supported for {:?}", other),
            })
        }
    };

    if fields.is_empty() {
        // No fields to disagree on.
        let id = *gep_counter;
        *gep_counter += 1;
        writeln!(out, "  %seq{} = and i1 true, true", id)?;
        return Ok(format!("%seq{}", id));
    }

    let mut acc: Option<String> = None;
    for (idx, (fname, fty)) in fields.iter().enumerate() {
        let id = *gep_counter;
        *gep_counter += 1;
        let lgep = format!("%seq_l{}", id);
        let rgep = format!("%seq_r{}", id);
        writeln!(
            out,
            "  {} = getelementptr inbounds %{}, ptr {}, i32 0, i32 {}",
            lgep, name, lhs_ptr, idx
        )?;
        writeln!(
            out,
            "  {} = getelementptr inbounds %{}, ptr {}, i32 0, i32 {}",
            rgep, name, rhs_ptr, idx
        )?;
        let fty_s = llvm_type_complete(fty)?;
        let lval = format!("%seq_lv{}", id);
        let rval = format!("%seq_rv{}", id);
        writeln!(out, "  {} = load {}, ptr {}, align 8", lval, fty_s, lgep)?;
        writeln!(out, "  {} = load {}, ptr {}, align 8", rval, fty_s, rgep)?;

        let cmp = format!("%seq_c{}", id);
        match fty {
            IrType::Scalar(DType::F32) | IrType::Scalar(DType::F64) => {
                writeln!(out, "  {} = fcmp oeq {} {}, {}", cmp, fty_s, lval, rval)?;
            }
            IrType::Scalar(_) => {
                writeln!(out, "  {} = icmp eq {} {}, {}", cmp, fty_s, lval, rval)?;
            }
            IrType::Str => {
                writeln!(
                    out,
                    "  {} = call i1 @iris_str_eq(ptr {}, ptr {})",
                    cmp, lval, rval
                )?;
            }
            IrType::Struct { .. } => {
                let inner = emit_struct_eq(fty, &lval, &rval, gep_counter, out)?;
                writeln!(out, "  {} = and i1 {}, true", cmp, inner)?;
            }
            other => {
                return Err(CodegenError::Unsupported {
                    backend: "llvm".into(),
                    detail: format!(
                        "`==` on record `{}` is not supported: field `{}` has type {:?}, \
                         which has no structural comparison. Compare the fields directly.",
                        name, fname, other
                    ),
                })
            }
        }

        acc = Some(match acc {
            None => cmp,
            Some(prev) => {
                let aid = *gep_counter;
                *gep_counter += 1;
                writeln!(out, "  %seq_a{} = and i1 {}, {}", aid, prev, cmp)?;
                format!("%seq_a{}", aid)
            }
        });
    }
    Ok(acc.expect("at least one field"))
}

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
        // Enum -> ptr (heap-allocated variant)
        IrType::Enum { .. } => Ok("ptr".to_owned()),
        // Fixed scalar arrays → LLVM array type (via pointer for args).
        IrType::Array { .. } => Ok("ptr".to_owned()), // arrays passed as ptr to [N x T]
        IrType::Tuple(_) => Ok("ptr".to_owned()),
        IrType::Str => Ok("ptr".to_owned()),
        IrType::Tensor { .. } => Ok("ptr".to_owned()),
        IrType::Option(_) | IrType::ResultType(_, _) => Ok("ptr".to_owned()),
        IrType::Chan(_)
        | IrType::Atomic(_)
        | IrType::Mutex(_)
        | IrType::Grad(_)
        | IrType::Sparse(_) => Ok("ptr".to_owned()),
        IrType::List(_) | IrType::Map(_, _) => Ok("ptr".to_owned()),
        IrType::Fn { .. } => Ok("ptr".to_owned()), // function pointer
        IrType::TaskGroup => Ok("ptr".to_owned()),
        IrType::WeakRef(_) => Ok("ptr".to_owned()),
        IrType::TraitObject { .. } => Ok("ptr".to_owned()),
        // Infer appears for spawn-wrapper parameters whose types couldn't
        // be resolved at lowering time. All such cases are heap-allocated
        // (chan, list, map, etc.) so `ptr` is safe.
        IrType::Infer => Ok("ptr".to_owned()),
    }
}

fn is_scalar_type(ty: &IrType) -> bool {
    matches!(ty, IrType::Scalar(_))
}

/// Extract the primary result ValueId from an instruction (if any).
fn instr_result_id(instr: &IrInstr) -> Option<ValueId> {
    instr.result()
}

fn inferred_value_type(
    func: &IrFunction,
    value_id: ValueId,
    value_ty: Option<&IrType>,
) -> Option<IrType> {
    if let Some(ty) = value_ty.or_else(|| func.value_type(value_id)) {
        return Some(ty.clone());
    }

    for block in func.blocks() {
        for instr in &block.instrs {
            if instr_result_id(instr) != Some(value_id) {
                continue;
            }
            return match instr {
                IrInstr::Call {
                    result_ty: Some(ty),
                    ..
                }
                | IrInstr::CallClosure { result_ty: ty, .. }
                | IrInstr::GetField { result_ty: ty, .. }
                | IrInstr::GetElement { result_ty: ty, .. }
                | IrInstr::OptionUnwrap { result_ty: ty, .. }
                | IrInstr::ResultUnwrap { result_ty: ty, .. }
                | IrInstr::ResultUnwrapErr { result_ty: ty, .. }
                | IrInstr::AtomicLoad { result_ty: ty, .. }
                | IrInstr::MutexLock { result_ty: ty, .. }
                | IrInstr::MakeStruct { result_ty: ty, .. }
                | IrInstr::MakeTuple { result_ty: ty, .. }
                | IrInstr::MakeClosure { result_ty: ty, .. }
                | IrInstr::MakeSome { result_ty: ty, .. }
                | IrInstr::MakeNone { result_ty: ty, .. }
                | IrInstr::MakeOk { result_ty: ty, .. }
                | IrInstr::MakeErr { result_ty: ty, .. }
                | IrInstr::MakeGrad { ty, .. }
                | IrInstr::GradValue { ty, .. }
                | IrInstr::GradTangent { ty, .. }
                | IrInstr::Sparsify { ty, .. }
                | IrInstr::Densify { ty, .. }
                | IrInstr::CallExtern { ret_ty: ty, .. } => Some(ty.clone()),
                // Carries no type field: an element count is always i64.
                IrInstr::SparseNnz { .. } => Some(IrType::Scalar(DType::I64)),
                IrInstr::ListNew { elem_ty, .. } => Some(IrType::List(Box::new(elem_ty.clone()))),
                IrInstr::ListGet { elem_ty, list, .. } => {
                    // If elem_ty is Infer, try to resolve it from the list's inferred type.
                    if matches!(elem_ty, IrType::Infer) {
                        if let Some(IrType::List(inner)) = func.value_type(*list) {
                            return Some((**inner).clone());
                        }
                    }
                    Some(elem_ty.clone())
                }
                IrInstr::ListPop { elem_ty, list, .. } => {
                    // If elem_ty is Infer, try to resolve it from the list's inferred type.
                    if matches!(elem_ty, IrType::Infer) {
                        if let Some(IrType::List(inner)) = func.value_type(*list) {
                            return Some((**inner).clone());
                        }
                    }
                    Some(elem_ty.clone())
                }
                IrInstr::MapNew { key_ty, val_ty, .. } => Some(IrType::Map(
                    Box::new(key_ty.clone()),
                    Box::new(val_ty.clone()),
                )),
                IrInstr::MapGet { val_ty, .. } => Some(val_ty.clone()),
                IrInstr::MapKeys { map, .. } => {
                    inferred_value_type(func, *map, func.value_type(*map)).and_then(|ty| match ty {
                        IrType::Map(key_ty, _) => Some(IrType::List(key_ty)),
                        _ => None,
                    })
                }
                IrInstr::MapValues { map, .. } => {
                    inferred_value_type(func, *map, func.value_type(*map)).and_then(|ty| match ty {
                        IrType::Map(_, val_ty) => Some(IrType::List(val_ty)),
                        _ => None,
                    })
                }
                IrInstr::ListConcat { lhs, .. } | IrInstr::ListSlice { list: lhs, .. } => {
                    inferred_value_type(func, *lhs, func.value_type(*lhs))
                }
                IrInstr::ConstStr { .. }
                | IrInstr::StrConcat { .. }
                | IrInstr::StrToUpper { .. }
                | IrInstr::StrToLower { .. }
                | IrInstr::StrTrim { .. }
                | IrInstr::StrRepeat { .. }
                | IrInstr::StrSlice { .. }
                | IrInstr::StrReplace { .. }
                | IrInstr::ValueToStr { .. }
                | IrInstr::ReadLine { .. } => Some(IrType::Str),
                IrInstr::ProcessArgs { .. } => Some(IrType::List(Box::new(IrType::Str))),
                IrInstr::EnvVar { .. } => Some(IrType::Option(Box::new(IrType::Str))),
                _ => None,
            };
        }
    }

    None
}

/// Return the bit width of an LLVM integer type string (e.g. "i64" → 64).
fn bit_width(ty: &str) -> u32 {
    match ty {
        "i1" => 1,
        "i8" => 8,
        "i16" => 16,
        "i32" => 32,
        "i64" => 64,
        _ => 64, // fallback
    }
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
            | IrInstr::DbExec { .. }
            | IrInstr::DbClose { .. }
            | IrInstr::ProcessExit { .. }
    )
}

fn llvm_val(v: ValueId, consts: &HashMap<ValueId, String>, func: &IrFunction) -> String {
    if let Some(c) = consts.get(&v) {
        // Emit the stored literal string unchanged.  Emit decimal literals
        // for both `float` and `double` cases — keeping the formatting
        // consistent avoids producing hex literals that may be parsed in
        // a mismatched type context later.
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

fn _f32_to_llvm_hex(f: f32) -> String {
    if f == 0.0 {
        if f.is_sign_negative() {
            return "-0x0.0p+0".to_owned();
        } else {
            return "0x0.0p+0".to_owned();
        }
    }
    if f.is_nan() {
        return "0x7fc00000".to_owned();
    }
    if f.is_infinite() {
        return if f.is_sign_negative() {
            "-inf".to_owned()
        } else {
            "inf".to_owned()
        };
    }
    let bits = f.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;
    if exp == 0 {
        // subnormal: fall back to decimal formatting
        return format!("{}", f);
    }
    let exponent = exp - 127;
    // mantissa_val includes the implicit leading 1 (24 bits)
    let mantissa_val = mant | 0x800000;
    // hex6 = 2 * (mantissa_val - 2^23)
    let hex6 = 2u32.wrapping_mul(mantissa_val.wrapping_sub(0x800000));
    if sign == 1 {
        format!("-0x1.{:06x}p{:+}", hex6, exponent)
    } else {
        format!("0x1.{:06x}p{:+}", hex6, exponent)
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
        "declare void @iris_print_bool(i32)",
        "declare void @iris_print_str(ptr)",
        "declare void @iris_panic(ptr)",
        "declare ptr @iris_read_line()",
        "declare i64 @iris_read_i64()",
        "declare double @iris_read_f64()",
        // String ops
        "declare i64 @iris_str_len(ptr)",
        "declare ptr @iris_str_concat(ptr, ptr)",
        "declare i1 @iris_str_eq(ptr, ptr)",
        "declare i64 @iris_str_cmp(ptr, ptr)",
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
        "declare i32 @iris_is_some(ptr)",
        "declare ptr @iris_option_unwrap(ptr)",
        "declare ptr @iris_make_ok(ptr)",
        "declare ptr @iris_make_err(ptr)",
        "declare i32 @iris_is_ok(ptr)",
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
        "declare i32 @iris_file_exists(ptr)",
        "declare ptr @iris_file_lines(ptr)",
        // Streaming file I/O
        "declare i64 @iris_file_open(ptr, ptr)",
        "declare i32 @iris_file_close(i64)",
        "declare ptr @iris_file_read(i64, i64)",
        "declare i32 @iris_file_write(i64, ptr)",
        // Database
        "declare i64 @iris_db_open(ptr)",
        "declare i64 @iris_db_exec(i64, ptr)",
        "declare i64 @iris_db_exec_params(i64, ptr, ptr)",
        "declare ptr @iris_db_query(i64, ptr)",
        "declare ptr @iris_db_query_params(i64, ptr, ptr)",
        "declare i64 @iris_db_close(i64)",
        // Process / environment (Phase 59)
        "declare void @exit(i32)",
        "declare ptr @malloc(i64)",
        "declare void @free(ptr)",
        "declare void @iris_set_argv(i32, ptr)",
        "declare ptr @iris_process_args()",
        "declare ptr @iris_env_var(ptr)",
        // Arrays / Tensors
        "declare ptr @iris_alloc_array()",
        "declare void @iris_bounds_check_abort(i64, i64)",
        "declare ptr @iris_array_load(ptr, i64)",
        "declare void @iris_array_store(ptr, i64, ptr)",
        "declare ptr @iris_tensor_op()",
        "declare ptr @iris_tensor_load(ptr, ...)",
        "declare void @iris_tensor_store(ptr, ...)",
        // Tensor runtime (real ops)
        "declare ptr @iris_tensor_matmul(ptr, ptr)",
        "declare ptr @iris_tensor_add(ptr, ptr)",
        "declare ptr @iris_tensor_sub(ptr, ptr)",
        "declare ptr @iris_tensor_mul(ptr, ptr)",
        "declare ptr @iris_tensor_div(ptr, ptr)",
        "declare ptr @iris_tensor_neg(ptr)",
        "declare ptr @iris_tensor_relu(ptr)",
        "declare ptr @iris_tensor_sigmoid(ptr)",
        "declare ptr @iris_tensor_tanh_act(ptr)",
        "declare ptr @iris_tensor_exp(ptr)",
        "declare ptr @iris_tensor_log(ptr)",
        "declare ptr @iris_tensor_sqrt(ptr)",
        "declare ptr @iris_tensor_abs(ptr)",
        "declare ptr @iris_tensor_reshape(ptr, i32, ptr)",
        "declare ptr @iris_tensor_transpose(ptr, i32, ptr)",
        "declare ptr @iris_tensor_reduce_sum(ptr, i32, i32)",
        "declare ptr @iris_tensor_reduce_max(ptr, i32, i32)",
        "declare ptr @iris_tensor_reduce_mean(ptr, i32, i32)",
        // GC reference counting
        "declare void @iris_retain(ptr)",
        "declare void @iris_release(ptr)",
        "declare void @iris_retain_kind(ptr, i32)",
        // Effect handler runtime
        "declare void @iris_push_handler_arm(ptr, ptr, i64, i32)",
        "declare void @iris_push_handler_fn(ptr)",
        "declare void @iris_push_handler_frame()",
        "declare void @iris_pop_handler()",
        "declare i32 @iris_handler_depth()",
        "declare i32 @iris_can_handle(ptr)",
        "declare i32 @iris_handler_has_resume(ptr)",
        "declare ptr @iris_find_handler_fn(ptr)",
        "declare void @iris_resume_cont(ptr, i64)",
        "declare i64 @iris_effect_dispatch_or_call(ptr, ptr, ptr, i64, ptr)",
        "declare void @iris_release_kind(ptr, i32)",
        // Channels / Concurrency
        "declare ptr @iris_chan_new(i64)",
        "declare void @iris_chan_send(ptr, ptr)",
        "declare ptr @iris_chan_recv(ptr)",
        "declare void @iris_spawn_fn(ptr, ptr)",
        // 4th parameter is the capture environment. The C runtime has always
        // had it (`void* arg`, forwarded to the body as its second argument);
        // codegen simply never passed it. See known-issues issue 12.
        "declare void @iris_par_for(ptr, i64, i64, ptr)",
        "declare void @iris_barrier()",
        // Structs / Tuples / Closures
        "declare ptr @iris_make_struct(i32, ...)",
        "declare ptr @iris_get_field(ptr, i32)",
        "declare ptr @iris_make_tuple(i32, ...)",
        "declare ptr @iris_get_element(ptr, i32)",
        "declare ptr @iris_make_closure(ptr, i32, ...)",
        "declare ptr @iris_call_closure(ptr, ...)",
        "declare void @iris_call_closure_void(ptr, ...)",
        "declare ptr @iris_closure_fn(ptr)",
        "declare ptr @iris_closure_get_capture(ptr, i32)",
        // Variants
        "declare ptr @iris_make_variant(i64, i32, ...)",
        "declare i64 @iris_get_variant_tag(ptr)",
        "declare ptr @iris_extract_variant_field(ptr, i64)",
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
        "declare ptr @iris_tape_record(double, ptr, i64, ptr, ptr)",
        "declare void @iris_backward(ptr)",
        "declare double @iris_tape_grad(ptr)",
        "declare ptr @iris_sparsify(ptr)",
        "declare ptr @iris_densify(ptr)",
        "declare i64 @iris_sparse_nnz(ptr)",
        "declare ptr @iris_sparsify_i64_array(ptr, i64)",
        "declare ptr @iris_sparsify_f64_array(ptr, i64)",
        // Boxing helpers (scalar → IrisVal*)
        "declare ptr @iris_box_i64(i64)",
        "declare ptr @iris_box_i32(i32)",
        "declare ptr @iris_box_f64(double)",
        "declare ptr @iris_box_f32(float)",
        "declare ptr @iris_box_bool(i1)",
        "declare ptr @iris_box_str(ptr)",
        "declare ptr @iris_box_list(ptr)",
        "declare ptr @iris_box_map(ptr)",
        "declare ptr @iris_box_option(ptr)",
        "declare ptr @iris_box_result(ptr)",
        "declare ptr @iris_box_chan(ptr)",
        "declare ptr @iris_box_atomic(ptr)",
        "declare ptr @iris_box_mutex(ptr)",
        "declare ptr @iris_box_grad(ptr)",
        "declare ptr @iris_box_sparse(ptr)",
        // Unboxing helpers (IrisVal* → scalar)
        "declare i64 @iris_unbox_i64(ptr)",
        "declare double @iris_unbox_f64(ptr)",
        "declare i32 @iris_unbox_bool(ptr)",
        "declare ptr @iris_unbox_str(ptr)",
        "declare ptr @iris_unbox_list(ptr)",
        "declare ptr @iris_unbox_map(ptr)",
        "declare ptr @iris_unbox_option(ptr)",
        "declare ptr @iris_unbox_result(ptr)",
        "declare ptr @iris_unbox_chan(ptr)",
        "declare ptr @iris_unbox_atomic(ptr)",
        "declare ptr @iris_unbox_mutex(ptr)",
        "declare ptr @iris_unbox_grad(ptr)",
        "declare ptr @iris_unbox_sparse(ptr)",
        // Typed to-string conversions
        "declare ptr @iris_i64_to_str(i64)",
        "declare ptr @iris_i32_to_str(i32)",
        "declare ptr @iris_f64_to_str(double)",
        "declare ptr @iris_f32_to_str(float)",
        "declare ptr @iris_bool_to_str(i32)",
        "declare ptr @iris_str_to_str(ptr)",
        // Math helpers
        "declare i64 @iris_pow_i64(i64, i64)",
        "declare i64 @iris_min_i64(i64, i64)",
        "declare i64 @iris_max_i64(i64, i64)",
        "declare i64 @iris_abs_i64(i64)",
        // Integer overflow-checked arithmetic
        "declare i64 @iris_add_checked(i64, i64)",
        "declare i64 @iris_sub_checked(i64, i64)",
        "declare i64 @iris_mul_checked(i64, i64)",
        "declare double @iris_sign_f64(double)",
        "declare double @tan(double)",
        // LLVM intrinsics
        "declare double @llvm.sqrt.f64(double)",
        "declare double @llvm.fabs.f64(double)",
        "declare double @llvm.floor.f64(double)",
        "declare double @llvm.ceil.f64(double)",
        // Prefer C math library `round` functions rather than llvm intrinsics
        "declare double @round(double)",
        "declare float @roundf(float)",
        "declare double @llvm.sin.f64(double)",
        "declare double @llvm.cos.f64(double)",
        "declare double @llvm.exp.f64(double)",
        "declare double @llvm.log.f64(double)",
        "declare double @llvm.log2.f64(double)",
        "declare double @llvm.pow.f64(double, double)",
        "declare double @llvm.minnum.f64(double, double)",
        "declare double @llvm.maxnum.f64(double, double)",
        // String split/join (Phase 95)
        "declare ptr @iris_str_split(ptr, ptr)",
        "declare ptr @iris_str_join(ptr, ptr)",
        // Time/OS (Phase 97)
        "declare i64 @iris_now_ms()",
        "declare void @iris_sleep_ms(i64)",
        // TCP networking
        "declare i64 @iris_tcp_connect(ptr, i64)",
        "declare i64 @iris_tcp_listen(i64)",
        "declare i64 @iris_tcp_accept(i64)",
        "declare ptr @iris_tcp_read(i64)",
        "declare void @iris_tcp_write(i64, ptr)",
        "declare void @iris_tcp_close(i64)",
        // UDP networking
        "declare i64 @iris_udp_open(i64)",
        "declare void @iris_udp_send(i64, ptr, i64)",
        "declare ptr @iris_udp_recv(i64)",
        "declare void @iris_udp_close(i64)",
        // Phase 104: New builtins
        // HTTP
        "declare ptr @iris_http_get(ptr)",
        "declare ptr @iris_http_post(ptr, ptr, ptr)",
        "declare ptr @iris_http_post_json(ptr, ptr)",
        "declare ptr @iris_http_request(ptr, ptr, ptr, ptr)",
        // JSON
        "declare ptr @iris_json_parse(ptr)",
        "declare ptr @iris_json_stringify(ptr)",
        // Set
        "declare ptr @iris_set_new()",
        "declare ptr @iris_set_add(ptr, ptr)",
        "declare i1 @iris_set_contains(ptr, ptr)",
        "declare ptr @iris_set_remove(ptr, ptr)",
        "declare i64 @iris_set_len(ptr)",
        "declare ptr @iris_set_to_list(ptr)",
        // Regex
        "declare i1 @iris_regex_match(ptr, ptr)",
        "declare ptr @iris_regex_find_all(ptr, ptr)",
        "declare ptr @iris_regex_replace(ptr, ptr, ptr)",
        "declare ptr @iris_regex_replace_all(ptr, ptr, ptr)",
        // DateTime
        "declare ptr @iris_datetime_now()",
        "declare double @iris_datetime_timestamp()",
        "declare ptr @iris_datetime_format(ptr)",
        // OS / Path
        "declare ptr @iris_cwd()",
        "declare ptr @iris_listdir(ptr)",
        "declare ptr @iris_path_join(ptr, ptr)",
        "declare i1 @iris_path_exists(ptr)",
        "declare i1 @iris_mkdir(ptr)",
        "declare i1 @iris_remove_file(ptr)",
        // Type introspection
        "declare ptr @iris_type_of(ptr)",
        // Random
        "declare i64 @iris_seed(i64)",
        "declare i64 @iris_random_seed()",
        "declare double @iris_random()",
        "declare i64 @iris_random_range(i64, i64)",
        // Hash / Encoding
        "declare i64 @iris_hash(ptr)",
        "declare ptr @iris_base64_encode(ptr)",
        "declare ptr @iris_base64_decode(ptr)",
        // String extras
        "declare ptr @iris_char_at(ptr, i64)",
        "declare ptr @iris_str_reverse(ptr)",
        // Phase 105: Extended builtins
        // String extras
        "declare ptr @iris_str_pad_left(ptr, i64, ptr)",
        "declare ptr @iris_str_pad_right(ptr, i64, ptr)",
        "declare ptr @iris_str_chars(ptr)",
        "declare ptr @iris_str_bytes(ptr)",
        "declare i64 @iris_str_count(ptr, ptr)",
        // Math constants/predicates
        "declare double @iris_math_pi()",
        "declare double @iris_math_e()",
        "declare double @iris_math_inf()",
        "declare i1 @iris_is_nan(double)",
        "declare i1 @iris_is_inf(double)",
        // OS / System
        "declare ptr @iris_env_get(ptr)",
        "declare void @iris_env_set(ptr, ptr)",
        "declare void @iris_exit_code(i64)",
        "declare ptr @iris_exec_cmd(ptr)",
        "declare i64 @iris_pid()",
        // Crypto / UUID
        "declare ptr @iris_uuid()",
        "declare ptr @iris_sha256(ptr)",
        "declare ptr @iris_hex_encode(ptr)",
        "declare ptr @iris_hex_decode(ptr)",
        // Deque
        "declare ptr @iris_deque_new()",
        "declare ptr @iris_deque_push_front(ptr, i64)",
        "declare ptr @iris_deque_push_back(ptr, i64)",
        "declare i64 @iris_deque_pop_front(ptr)",
        "declare i64 @iris_deque_pop_back(ptr)",
        "declare i64 @iris_deque_len(ptr)",
        // FFI
        "declare ptr @iris_ffi_open(ptr)",
        "declare i64 @iris_ffi_call(ptr, ptr)",
        "declare i1 @iris_ffi_close(ptr)",
        // Expanded C FFI
        "declare i64 @iris_ffi_call_i64(ptr, ptr, ptr, i32)",
        "declare i64 @iris_ffi_out_new(i64)",
        "declare void @iris_ffi_out_free(i64)",
        "declare double @iris_ffi_out_get_f64(i64, i64)",
        "declare i64 @iris_ffi_out_get_i64(i64, i64)",
        "declare ptr @iris_ffi_out_get_str(i64)",
        "declare void @iris_ffi_out_set_f64(i64, i64, double)",
        "declare void @iris_ffi_out_set_i64(i64, i64, i64)",
        "declare double @iris_ffi_call_f64(ptr, ptr, ptr, i32)",
        "declare ptr @iris_ffi_call_str(ptr, ptr, ptr, i32)",
        "declare void @iris_ffi_call_void(ptr, ptr, ptr, i32)",
        // Python FFI
        "declare ptr @iris_python_eval(ptr)",
        "declare i64 @iris_python_exec(ptr)",
        "declare ptr @iris_python_call(ptr, ptr, ptr)",
        "declare ptr @iris_python_version()",
        // Rust FFI (cdylib)
        "declare ptr @iris_rust_lib_open(ptr)",
        "declare i64 @iris_rust_call_i64(ptr, ptr, ptr, i32)",
        "declare double @iris_rust_call_f64(ptr, ptr, ptr, i32)",
        "declare void @iris_rust_call_void(ptr, ptr, ptr, i32)",
        // Functional list ops
        "declare double @iris_list_sum(ptr)",
        "declare i64 @iris_list_min(ptr)",
        "declare i64 @iris_list_max(ptr)",
        "declare i64 @iris_list_index_of(ptr, i64)",
        "declare i64 @iris_list_count(ptr, i64)",
        "declare ptr @iris_list_reverse(ptr)",
        "declare ptr @iris_list_take(ptr, i64)",
        "declare ptr @iris_list_drop(ptr, i64)",
        // Deque front/back accessors
        "declare i64 @iris_deque_front(ptr)",
        "declare i64 @iris_deque_back(ptr)",
        // Channel extras
        "declare ptr @iris_chan_try_recv(ptr)",
        "declare i64 @iris_chan_len(ptr)",
        // First parameter is the channel count, matching
        // `int64_t iris_select(int64_t n, ...)` in iris_runtime.c.
        "declare i64 @iris_select(i64, ...)",
        "declare i1 @iris_timeout(i64)",
        // FFI variadic call
        "declare i64 @iris_ffi_call_args(ptr, ptr, ptr, i32)",
        // Concurrency extras
        "declare i64 @iris_thread_count()",
        // Terminal / Interactive Input
        "declare i64 @iris_read_key()",
        "declare ptr @iris_read_password(ptr)",
        "declare void @iris_term_clear()",
        "declare void @iris_term_cursor(i64, i64)",
        "declare void @iris_term_show_cursor(i32)",
        "declare void @iris_term_set_color(i64, i64)",
        "declare void @iris_term_reset()",
        "declare i64 @iris_term_rows()",
        "declare i64 @iris_term_cols()",
        // BitSet
        "declare ptr @iris_bitset_new(i64)",
        "declare ptr @iris_bitset_set(ptr, i64)",
        "declare i1 @iris_bitset_get(ptr, i64)",
        "declare i64 @iris_bitset_count(ptr)",
        "declare ptr @iris_bitset_clear(ptr, i64)",
        // New builtins (list ops, map_entries, recv_timeout, weak refs, gc)
        "declare i64 @iris_list_remove(ptr, i64)",
        "declare i64 @iris_list_insert(ptr, i64, ptr)",
        "declare ptr @iris_map_entries(ptr)",
        "declare ptr @iris_recv_timeout(ptr, i64)",
        "declare void @iris_chan_send_b(ptr, ptr)",
        "declare ptr @iris_weak_ref(ptr)",
        "declare i32 @iris_weak_alive(ptr)",
        "declare ptr @iris_weak_upgrade(ptr)",
        "declare ptr @iris_gc_stats_map()",
        "declare i32 @iris_gc_collect_call()",
    ];
    for decl in declares {
        writeln!(out, "{}", decl)?;
    }
    writeln!(out)?;
    Ok(())
}
