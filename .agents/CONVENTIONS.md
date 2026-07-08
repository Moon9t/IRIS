# IRIS Compiler Architecture & Coding Standards

<stack_declaration>
- Language: Rust (Strict mode, idiomatic, safety-first)
- Core Crates: `inkwell` (or `llvm-sys`), `tokio` (for async infrastructure)
- Backend: LLVM TargetMachine API, LLVM ORC JIT
</stack_declaration>

<architecture_invariants>
Do:
- Use explicit idiomatic Rust error propagation (`Result<T, E>`) mapping back to `CompilerError`.
- Use the programmatic LLVM ORC JIT layers (`ExecutionSession`, `JITDylib`) to compile IR directly in-memory to native machine code.
- Initialize native targets programmatically via `Target::initialize_native_target()`.
- Use `inkwell`'s `Builder`, `Module`, and `Type` abstractions to construct an AST-backed LLVM IR module.

Don't:
- Never use `unsafe` blocks without a preceding `// SAFETY:` comment detailing the explicit invariants.
- Never use text-based LLVM IR generation via string manipulation or macros (e.g., `writeln!`, `format!`).
- Never shell out to external toolchain executables (like `clang`) via `std::process::Command` for compilation.
- Never use `.unwrap()` or `.expect()` outside of initialization check failures.
</architecture_invariants>

<example_patterns>
<example type="negative">
// DON'T: Text-based IR generation
writeln!(out, "define i32 @main() {{ ret i32 0 }}");
</example>

<example type="positive">
// DO: Programmatic LLVM Builder
let i32_type = context.i32_type();
let fn_type = i32_type.fn_type(&[], false);
let function = module.add_function("main", fn_type, None);
let basic_block = context.append_basic_block(function, "entry");
builder.position_at_end(basic_block);
builder.build_return(Some(&i32_type.const_int(0, false)));
</example>
</example_patterns>