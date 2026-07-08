# Execution & Review Protocol

Before writing or modifying any code for the IRIS compiler, you must strictly follow this execution lifecycle. 

<execution_phases>
Phase 1: Plan Mode
1. Research the codebase to find relevant files and existing LLVM mapping patterns.
2. Ask clarifying questions if the architectural approach is ambiguous.
3. Create a detailed implementation plan with file paths and explicit API endpoints.
4. Wait for user approval before generating the code block.

Phase 2: Atomic Execution
1. Limit the scope to one single concern at a time (e.g., refactoring `compile_native` first, before touching module serialization).
2. Keep changes minimal. Only change what is absolutely necessary; do not attempt unsolicited refactoring in unrelated blocks.

Phase 3: Debug Mode
If a JIT execution or pointer mapping fails:
1. Generate multiple hypotheses about the root cause (e.g., target triple mismatch, layout alignment issues).
2. Instrument the code with targeted logging statements.
3. Analyze the actual runtime behavior instead of blindly guessing fixes.
</execution_phases>

<static_analysis_checklist>
Before finalizing the output, verify:
- Lifetime Verification: Are all Rust lifetimes explicitly bounded?
- Type Correctness: Do the programmatic LLVM types correctly match the target host machine's layout invariants?
- Zero-Macro Compliance: Have all `writeln!` strings been fully stripped from the target module?
</static_analysis_checklist>