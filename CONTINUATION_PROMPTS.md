# IRIS Compiler — Continuation Prompts for Claude Opus 4.6

Each section below is a self-contained prompt. Copy the entire section
(from the `---` line to the next `---` line) into a new Claude Opus 4.6
session that has the IRIS project directory open.

---

## PHASE 9 — Loop Constructs (`while` / `loop` / `break` / `continue`)

```
You are continuing development of the IRIS compiler, a Rust-based ML DSL
compiler located at c:\Users\Moon\Desktop\Projects\IRIS.

CURRENT STATE
=============
83 tests pass (cargo test with /c/Users/Moon/.cargo/bin/cargo).
The language supports: fn definitions, let bindings, if/else, scalar
arithmetic (+,-,*,/), comparisons (==,!=,<,<=,>,>=), unary (-,!),
boolean literals, integer/float/bool types, tensor types, einsum intrinsic,
call expressions, model DSL for ML layer graphs.

WHAT IS MISSING
===============
The language has NO loop constructs. `while`, `loop`, `for`, `break`,
`continue` are not keywords. Any loop in user code causes a parse error.

YOUR TASK: Phase 9 — implement while-loop and loop-forever constructs.

DELIVERABLES
============
1. `while <cond> { <body> }` — loop while cond is true
2. `loop { <body> }` — loop forever (must contain a break)
3. `break` — exit the innermost loop
4. `continue` — skip to the next iteration of the innermost loop
5. 8 integration tests in tests/phase9.rs (all must pass)
6. All 91 tests pass, zero warnings.

ARCHITECTURE CONSTRAINTS
========================
- IR design: Block-parameter SSA (MLIR-style). Loops are represented with
  back-edge blocks. A while loop lowers to THREE IR blocks:
    header (loop condition check) → body_block (loop body) → merge (after loop)
  The body_block terminates with Br { target: header, args: [] }.
  The header terminates with CondBr { cond, then_block: body, else_block: merge }.

- `break` lowers to Br { target: merge_block, args: [] }
- `continue` lowers to Br { target: header_block, args: [] }
- The lowerer needs a "loop context stack" (Vec of (header_id, merge_id)) to
  resolve break/continue targets. Push on loop entry, pop on loop exit.

- `loop { body }` desugars to: create a header block that immediately Br's
  to body, body terminates with Br back to header unless break is hit.
  Alternatively: header IS the body block, with a back-edge to itself.
  Simplest approach: header and body are the same block; break exits.

- `ValidatePass` must NOT reject back-edges (a loop's Br targeting a block
  that appears earlier in the function). Currently ValidatePass checks that
  all used values are defined before use in dominator order — this is fine
  because loop-body values are not used in the header condition.

- `ShapeCheckPass` and `CsePass` must handle or skip back-edge cycles safely.
  The simplest approach: these passes iterate over instructions in block order
  and do NOT follow back-edges for analysis. This is already safe because
  CsePass and ShapeCheckPass do not do dataflow across loop iterations.

FILE CHANGES REQUIRED
=====================
src/parser/lexer.rs    — Add Token::While, Token::Loop, Token::Break,
                         Token::Continue to the enum and their Display arms.
                         Add keyword mappings in lex_ident_or_keyword.

src/parser/ast.rs      — Add AstStmt::While { cond, body, span }
                         Add AstStmt::Loop { body, span }
                         Add AstStmt::Break { span }
                         Add AstStmt::Continue { span }

src/parser/parse.rs    — parse_stmt: match Token::While → parse_while_stmt
                         match Token::Loop → parse_loop_stmt
                         match Token::Break → consume + return AstStmt::Break
                         match Token::Continue → consume + return AstStmt::Continue
                         Add parse_while_stmt and parse_loop_stmt helpers.

src/lower/mod.rs       — Add loop_stack: Vec<(BlockId, BlockId)> field to Lowerer.
                         lower_stmt: handle AstStmt::While, Loop, Break, Continue.
                         While lowering: create header/body/merge blocks,
                         push (header, merge) onto loop_stack, lower body,
                         pop loop_stack.

src/pass/validate.rs   — Ensure ValidatePass does NOT reject Br instructions
                         whose target block index is <= current block index
                         (back-edges are valid in loop SSA).

tests/phase9.rs        — 8 tests (see below).

TESTS (tests/phase9.rs)
=======================
1. test_while_basic — `fn count() -> i64 { let x = 0; while x < 5 { let x = x + 1 } x }`
   IR contains "cmplt" and multiple blocks (a back-edge Br exists).

   NOTE: since IRIS uses immutable let-bindings, while loops that update
   a variable need a mutable rebind pattern. The simplest approach for
   Phase 9 is to treat while-loop bodies as using BLOCK PARAMETERS to
   carry loop variables. The while sugar is:

     while cond(vars) { body(vars) → new_vars }

   which lowers to:
     header(vars):   %cond = cond(vars);  CondBr %cond, body(vars), merge
     body(vars):     ...; Br header(new_vars)
     merge:          (uses vars as-is from header's else path)

   For the test, use a simpler form: the loop variable is a block parameter
   of the header block, so re-binding `let x = x + 1` inside the body works
   by passing the new value as a block argument to the header Br.

   IMPORTANT: The lowerer must detect which variables are modified inside the
   loop body and thread them as block parameters. The simplest approach:
   - Before entering loop body, snapshot the env.
   - After lowering body, detect which names in env were rebound.
   - Those are the "loop variables" — add them as block params to header.
   - The entry Br to header passes the initial values.
   - The body Br back to header passes the updated values.
   - The merge block receives the final values via its params (from CondBr).

   This is the standard SSA loop lowering.

2. test_while_zero_iterations — `fn zero() -> i64 { let x = 0; while false { let x = x + 1 } x }`
   Compiles without error (IR is valid; ConstFold may simplify).

3. test_loop_with_break — `fn find() -> bool { loop { break } false }`
   IR must have a Br to the merge block.

4. test_while_ir_has_back_edge — check that IR output for a while loop
   contains two Br instructions (one back-edge, one fallthrough).

5. test_break_exits_loop — end-to-end compile to IR succeeds.

6. test_continue_skips_body — end-to-end compile to IR succeeds.

7. test_nested_while — nested while loops compile without error.

8. test_while_llvm — simple while compiles with EmitKind::Llvm and contains "br label".

IMPORTANT NOTES
===============
- Run `/c/Users/Moon/.cargo/bin/cargo test` after each file change to catch
  regressions early.
- The `thiserror` crate is the only dependency in Cargo.toml.
- Rust toolchain is at /c/Users/Moon/.cargo/bin/cargo (not in PATH).
- Do NOT use `cd` — use absolute paths in all cargo invocations.
- Target: 91 passing tests, 0 warnings.
```

---

## PHASE 10 — Tensor Indexing Syntax, Modulo Operator, and Cast Instructions

```
You are continuing development of the IRIS compiler, a Rust-based ML DSL
compiler located at c:\Users\Moon\Desktop\Projects\IRIS.

CURRENT STATE
=============
The compiler has 83 passing tests (or more if Phase 9 was done).
The IR already has IrInstr::Load { tensor, indices, result_ty } and
IrInstr::Store { tensor, indices, value } — these instructions exist but
there is NO parser/AST support for them. The syntax `t[i]` and `t[i] = v`
does not parse.

Additionally:
- The `%` (modulo) operator is missing from the lexer/parser/lowerer/IR.
- There are no cast instructions: `x as f32`, `x as i64` etc do not parse.

YOUR TASK: Phase 10 — implement tensor indexing, modulo, and casts.

DELIVERABLES
============
1. Tensor read indexing: `t[i]`, `t[i,j]`, `m[r,c]` → IrInstr::Load
2. Tensor write indexing: `t[i] = v` (as a statement) → IrInstr::Store
3. Modulo operator: `a % b` → BinOp::Mod
4. Cast expression: `x as f32`, `x as i64` → IrInstr::Cast
5. 8 integration tests in tests/phase10.rs (all must pass)
6. All tests pass, zero warnings.

ARCHITECTURE
============

TENSOR INDEXING
---------------
Parse `expr[expr, expr, ...]` as AstExpr::Index { base, indices, span }.
In the parser: after parsing a primary expr, if the next token is `[`,
consume it, parse comma-separated index expressions, consume `]`, wrap in
AstExpr::Index. This is similar to how function calls are parsed as postfix.

Lowering AstExpr::Index:
  - Lower `base` → (tensor_val, tensor_ty)
  - Lower each index expr → (idx_val, idx_ty)
  - Emit IrInstr::Load { result, tensor: tensor_val, indices, result_ty: element_ty }
  - element_ty = extract dtype from tensor_ty (IrType::Tensor { dtype, .. } → IrType::Scalar(dtype))

For store `t[i] = v` (as AstStmt::Assign):
  Parse as: AstStmt::Assign { target: AstExpr::Index { .. }, value: AstExpr, span }
  Lower to IrInstr::Store { tensor, indices, value }.

  Parser change: in parse_stmt, after parsing an expression, if next token
  is `=` (not `==`), treat the expression as an lvalue and parse as assign.
  Only AstExpr::Index is a valid lvalue for now.

MODULO OPERATOR
---------------
- Add `Token::Percent` to lexer (b'%' → Token::Percent).
- Add `AstBinOp::Mod` to AST.
- Add `BinOp::Mod` to IR (src/ir/instr.rs), with Display "mod".
- Wire it through: parse_mul_expr handles `%` same as `*` and `/`.
- Lower: AstBinOp::Mod → BinOp::Mod.
- Codegen printer: BinOp::Mod → "mod".
- LLVM codegen: i32/i64 → "srem", f32/f64 → "frem".
- TypeInferPass: same type constraints as Add/Mul.
- ConstFoldPass: `a % b` with both known → `a % b` (use `%` in Rust).

CAST EXPRESSIONS
----------------
Add `Token::As` keyword (lexer: "as" → Token::As).
Add AstExpr::Cast { expr, ty, span } to AST.
In parser: parse_unary checks for `as` suffix (it is a postfix operator with
  lower precedence — actually parse it in a new parse_cast() level between
  parse_add_expr and parse_cmp_expr, or after parse_unary):

  Simplest: in parse_primary or as a postfix after parse_unary:
  after parsing a unary expr, if next token is `Token::As`, consume it,
  parse a type annotation (reuse parse_type()), wrap in AstExpr::Cast.

Add IrInstr::Cast { result, operand, from_ty: IrType, to_ty: IrType } to IR.
Lower AstExpr::Cast: emit IrInstr::Cast.
Printer: `%vN = cast %vM : f32 -> i64`
LLVM codegen:
  - f32/f64 → i32/i64: `fptosi float %v, i32`
  - i32/i64 → f32/f64: `sitofp i32 %v, float`
  - f32 → f64: `fpext float %v to double`
  - f64 → f32: `fptrunc double %v to float`
  - i32 → i64: `sext i32 %v to i64`
  - i64 → i32: `trunc i64 %v to i32`
  - same type: emit the operand directly (no-op)
TypeInferPass: validate that from_ty matches operand type; to_ty is valid.
IrInstr::Cast: result() → Some(result), operands() → vec![operand].
ValidatePass: handle Cast in the existing result/operand scanning.

FILE CHANGES REQUIRED
=====================
src/parser/lexer.rs    — Token::Percent, Token::As keyword
src/parser/ast.rs      — AstBinOp::Mod, AstExpr::Index, AstExpr::Cast,
                         AstStmt::Assign
src/parser/parse.rs    — parse index postfix, parse_cast level, parse_stmt
                         assignment detection, BinOp::Mod in parse_mul_expr
src/ir/instr.rs        — BinOp::Mod, IrInstr::Cast (result/operands/is_terminator)
src/lower/mod.rs       — lower_expr: Index → Load, Cast → Cast;
                         lower_stmt: Assign → Store; lower_binop: Mod → Mod
src/codegen/printer.rs — IrInstr::Cast emission
src/codegen/llvm_stub.rs — BinOp::Mod (srem/frem), IrInstr::Cast emission
src/pass/type_infer.rs — Cast type check; Mod same as Mul
src/pass/const_fold.rs — Mod const folding
src/pass/validate.rs   — Cast in result/operand scan
src/pass/opt.rs        — CseKey::Cast variant; apply_replacements Cast arm
tests/phase10.rs       — 8 tests

TESTS (tests/phase10.rs)
========================
1. test_tensor_load_ir — `fn get(t: tensor<f32,[8]>, i: i64) -> f32 { t[i] }`
   IR contains "load"
2. test_tensor_store_ir — function that does `t[0] = 1.0` compiles without error
3. test_modulo_ir — `fn rem(a: i64, b: i64) -> i64 { a % b }` IR contains "mod"
4. test_modulo_const_fold — `fn rem() -> i64 { 10 % 3 }` IR contains "3" (folded)
5. test_cast_f32_to_i64 — `fn conv(x: f32) -> i64 { x as i64 }` IR contains "cast"
6. test_cast_llvm_fptosi — same fn with EmitKind::Llvm → contains "fptosi"
7. test_cast_llvm_sitofp — `fn conv(x: i64) -> f32 { x as f32 }` → "sitofp"
8. test_modulo_llvm — `fn rem(a: i64, b: i64) -> i64 { a % b }` EmitKind::Llvm
   → contains "srem"

IMPORTANT NOTES
===============
- Rust toolchain: /c/Users/Moon/.cargo/bin/cargo
- Only dependency: thiserror = "1.0"
- Do not use `cd` — use absolute paths.
- Run tests after each file: /c/Users/Moon/.cargo/bin/cargo test
```

---

## PHASE 11 — Tree-Walking Interpreter (Execute IRIS IR)

```
You are continuing development of the IRIS compiler, a Rust-based ML DSL
compiler located at c:\Users\Moon\Desktop\Projects\IRIS.

CURRENT STATE
=============
The compiler can parse, lower, optimize, and emit IR/LLVM text for IRIS
source files. But it CANNOT execute any code — there is no interpreter or
runtime. `--emit ir` and `--emit llvm` produce text only.

YOUR TASK: Phase 11 — implement a tree-walking IR interpreter.

DELIVERABLES
============
1. New module: src/interp/mod.rs — IR interpreter
2. New EmitKind::Eval variant — "run" the first function with no arguments
   (for testing constant-returning functions)
3. Public function: iris::interp::eval_function(func: &IrFunction, args: &[IrValue]) -> Result<Vec<IrValue>, InterpError>
4. Integration via new --emit eval CLI flag (main.rs)
5. 8 integration tests in tests/phase11.rs (all must pass)
6. All tests pass, zero warnings.

INTERPRETER DESIGN
==================
IrValue enum (the runtime value type):
  pub enum IrValue {
      F32(f32),
      F64(f64),
      I32(i32),
      I64(i64),
      Bool(bool),
      Tensor(Vec<f32>, Vec<usize>),  // flat data + shape
  }

InterpError (thiserror, new variant in error.rs or separate file):
  - UndefinedValue { id: u32 }
  - DivisionByZero
  - IndexOutOfBounds { idx: i64, len: usize }
  - TypeError { detail: String }
  - Unsupported { detail: String }

Interpreter state:
  - values: HashMap<ValueId, IrValue>  — SSA value map
  - blocks: indexed by BlockId
  - current_block: BlockId

Algorithm (straightforward SSA interpreter):
  1. Bind function arguments: for each block param of the entry block,
     bind the corresponding arg from `args`.
  2. Execute the entry block's instructions in order.
  3. On IrInstr::Br { target, args }: bind target block's params to args,
     jump to target block.
  4. On IrInstr::CondBr { cond, then_block, then_args, else_block, else_args }:
     evaluate cond, choose branch, bind params, jump.
  5. On IrInstr::Return { values }: return the evaluated values.
  6. Detect infinite loops with a step counter (max 1_000_000 iterations).

Instruction evaluation:
  - ConstFloat/ConstInt/ConstBool → insert literal into values map
  - BinOp: extract lhs/rhs from values map, compute, insert result
    - Add/Sub/Mul/Div for f32/f64/i32/i64
    - CmpEq/Ne/Lt/Le/Gt/Ge → IrValue::Bool
    - FloorDiv → integer floor division
    - Mod → remainder
  - UnaryOp: Neg (arithmetic), Not (bool)
  - Cast: f32↔i64 etc
  - Load: index into Tensor flat data
  - Store: update Tensor flat data
  - TensorOp::Einsum: implement naive einsum for 2D matrix multiply ("mk,kn->mn")
  - TensorOp::Unary: apply elementwise fn (relu=max(0,x), sigmoid, tanh)
  - Call: if callee matches a function in the module, recurse; else error

FILE CHANGES
============
src/interp/mod.rs      — new file: IrValue, InterpError, Interpreter struct,
                         eval_function(), eval_block() impl
src/error.rs           — add Error::Interp(InterpError) variant; add InterpError enum
src/lib.rs             — pub mod interp; add EmitKind::Eval; wire in compile()
src/main.rs            — add "eval" to --emit parse_args match
tests/phase11.rs       — 8 tests

TESTS (tests/phase11.rs)
========================
1. test_eval_const_return — `fn answer() -> i64 { 42 }` → eval returns [IrValue::I64(42)]
2. test_eval_arithmetic — `fn add(a: f32, b: f32) -> f32 { a + b }` with args [1.0, 2.0] → [3.0]
3. test_eval_if_true — `fn max(a: f32, b: f32) -> f32 { if a > b { a } else { b } }` → correct
4. test_eval_if_false — same function, b > a
5. test_eval_neg — `fn neg(x: f32) -> f32 { -x }` with 3.0 → -3.0
6. test_eval_not — `fn inv(b: bool) -> bool { !b }` with true → false
7. test_eval_tensor_load — build IrModule with Load instruction; eval returns element
8. test_eval_emit_kind — compile("fn answer() -> i64 { 42 }", "test", EmitKind::Eval)
   returns Ok("42\n") (the interpreter result printed as a string)

IMPORTANT NOTES
===============
- Rust toolchain: /c/Users/Moon/.cargo/bin/cargo
- Only dependency: thiserror = "1.0"  (do NOT add any new dependencies)
- The interpreter only needs to handle the scalar subset correctly for
  the tests. Tensor operations can return InterpError::Unsupported for
  complex cases beyond the tests.
- Do not use `cd` — use absolute paths.
```

---

## PHASE 12 — Error Quality, Source Diagnostics, and CLI Polish

```
You are continuing development of the IRIS compiler, a Rust-based ML DSL
compiler located at c:\Users\Moon\Desktop\Projects\IRIS.

CURRENT STATE
=============
Errors show byte offsets, not line:col numbers. Example:
  parse error: unexpected character '@' at byte 42
Instead of:
  error[E001]: unexpected character '@'
   --> input.iris:3:7

Also missing:
- `--help` flag
- `-o <file>` output redirect
- `--dump-ir-after <pass>` diagnostic flag

YOUR TASK: Phase 12 — improve error messages and CLI.

DELIVERABLES
============
1. All ParseError / LowerError variants that carry a `Span` or `pos: u32`
   get line:col computed from the source string and included in the message.
2. New function: `iris::diagnostics::byte_to_line_col(source: &str, byte: u32) -> (u32, u32)`
   Returns 1-based (line, col).
3. New function: `iris::diagnostics::render_error(source: &str, err: &Error) -> String`
   Renders a rustc-style diagnostic with the source line and a `^` pointer.
4. main.rs: use render_error instead of `eprintln!("error: {}", e)`.
5. main.rs: `--help` flag prints usage and exits 0.
6. main.rs: `-o <file>` flag writes output to a file instead of stdout.
7. main.rs: `--dump-ir-after <pass-name>` flag prints IR after that pass runs.
   Requires PassManager to support a dump hook.
8. 8 integration tests in tests/phase12.rs

FILE CHANGES
============
src/diagnostics.rs     — new file: byte_to_line_col, render_error, span_to_line_col
src/lib.rs             — pub mod diagnostics
src/main.rs            — --help, -o, --dump-ir-after, use render_error
src/pass/mod.rs        — PassManager gains optional dump_after: Option<String>
                         field; after each pass, if pass.name() == dump_after,
                         print emit_ir_text to stderr.
tests/phase12.rs       — 8 tests

TESTS (tests/phase12.rs)
========================
1. test_byte_to_line_col_basic — "abc\ndef\n", byte 4 → (2, 1)
2. test_byte_to_line_col_first_line — "hello", byte 2 → (1, 3)
3. test_render_error_contains_caret — render_error for a parse error at a
   known position contains "^" in the output
4. test_render_error_contains_line_number — output contains "line" or ":" line number
5. test_parse_error_unknown_char_message — compile source with '@' in it;
   render the error; assert output contains "3:" or the correct line number
6. test_help_flag — parse_args with ["iris", "--help"] → returns a special
   HelpRequested variant (or the binary exits 0 — test via parse_args logic)
7. test_output_flag — parse_args with ["-o", "out.ll", "file.iris"] succeeds
   and captures the output path
8. test_span_to_line_col_multiline — source with 5 lines; test several offsets

IMPORTANT NOTES
===============
- Rust toolchain: /c/Users/Moon/.cargo/bin/cargo
- Only dependency: thiserror = "1.0"
- Prioritize correctness of byte_to_line_col — iterate bytes, count \n chars.
- The `render_error` function needs access to the original source string.
  The compile() function does NOT currently return the source. You may need
  to thread source into error types OR have render_error accept (source, err).
  The standalone diagnostic approach (render_error(source, err)) is cleaner.
```

---

## PHASE 13 — Struct Types and Named Records

```
You are continuing development of the IRIS compiler, a Rust-based ML DSL
compiler located at c:\Users\Moon\Desktop\Projects\IRIS.

CURRENT STATE
=============
IRIS only supports scalar types (f32, f64, i32, i64, bool) and tensor types.
There are no aggregate/record types. Functions cannot return multiple named
fields. There are no `struct` definitions.

YOUR TASK: Phase 13 — add struct type definitions and field access.

DELIVERABLES
============
Syntax to support:
  struct Point { x: f32, y: f32 }
  fn make_point(x: f32, y: f32) -> Point { Point { x, y } }
  fn get_x(p: Point) -> f32 { p.x }

1. `struct` keyword + struct definition AST + parser
2. `IrType::Struct { name: String, fields: Vec<(String, IrType)> }` in types.rs
3. `IrInstr::MakeStruct { result, fields: Vec<ValueId>, result_ty: IrType }` in instr.rs
4. `IrInstr::GetField { result, base: ValueId, field_index: usize, result_ty: IrType }` in instr.rs
5. AstExpr::StructLit { name: Ident, fields: Vec<(Ident, AstExpr)>, span }
6. AstExpr::FieldAccess { base: Box<AstExpr>, field: Ident, span }
7. Lowering, IR printing, ValidatePass, TypeInferPass support for new instrs
8. 8 integration tests in tests/phase13.rs

FILE CHANGES
============
src/parser/lexer.rs    — Token::Struct keyword
src/parser/ast.rs      — AstItem::Struct { name, fields: Vec<(Ident, AstType)>, span }
                         AstExpr::StructLit, AstExpr::FieldAccess
                         AstModule gains structs: Vec<AstStruct> field
src/parser/parse.rs    — parse_item: match Token::Struct → parse_struct_def
                         parse_primary: match Ident followed by { → struct lit
                         parse_postfix: match . → field access
src/ir/types.rs        — IrType::Struct { name, fields }
src/ir/instr.rs        — IrInstr::MakeStruct, IrInstr::GetField
                         (result/operands/is_terminator updates)
src/lower/mod.rs       — lower struct defs to a type registry;
                         lower AstExpr::StructLit → MakeStruct;
                         lower AstExpr::FieldAccess → GetField (look up field index)
src/codegen/printer.rs — emit MakeStruct and GetField
src/pass/validate.rs   — handle MakeStruct/GetField in result/operand scan
src/pass/type_infer.rs — GetField: result type = field type; MakeStruct: check field types
tests/phase13.rs       — 8 tests

TESTS (tests/phase13.rs)
========================
1. test_struct_parse — struct definition parses without error
2. test_struct_field_access_ir — `fn get_x(p: Point) -> f32 { p.x }` → IR contains "getfield"
3. test_struct_literal_ir — `fn make() -> Point { Point { x: 1.0, y: 2.0 } }` → IR has "makestruct"
4. test_struct_round_trip — make then get: `fn f() -> f32 { let p = Point {x:3.0,y:4.0}; p.x }` → IR valid
5. test_struct_type_error — assigning wrong field type → LowerError or PassError
6. test_struct_field_not_found — accessing nonexistent field → LowerError
7. test_nested_struct — struct containing another struct → compiles
8. test_struct_ir_text — IR text for a struct function contains "struct" type annotation

IMPORTANT NOTES
===============
- Rust toolchain: /c/Users/Moon/.cargo/bin/cargo
- Only dependency: thiserror = "1.0"
- Keep the struct type registry simple: store in IrModule as a HashMap<String, Vec<(String, IrType)>>.
- MakeStruct and GetField do NOT need LLVM codegen in this phase — emit a comment or opaque call.
```

---

## PHASE 14 — Real ONNX Binary Output

```
You are continuing development of the IRIS compiler, a Rust-based ML DSL
compiler located at c:\Users\Moon\Desktop\Projects\IRIS.

CURRENT STATE
=============
`--emit onnx` produces a human-readable text representation of an ONNX graph
(src/codegen/onnx.rs). It is NOT a real ONNX binary protobuf. Real ML tools
(ONNX Runtime, netron, torch.onnx) require binary .onnx files.

The current onnx.rs uses a text format like:
  node { op_type: "MatMul" input: "x" input: "w" output: "y" }
This is not valid protobuf binary.

YOUR TASK: Phase 14 — emit valid binary ONNX protobuf.

CONSTRAINT: You may NOT add external dependencies beyond `thiserror = "1.0"`.
This means you must write a minimal hand-rolled protobuf encoder.

APPROACH: Write a minimal protobuf binary encoder in src/proto/mod.rs.
ONNX model proto (onnx.proto3 schema) field numbers are well-documented.
You only need to encode:
  - ModelProto (field 1=ir_version:int64, 7=opset_import:OperatorSetIdProto,
                8=graph:GraphProto)
  - GraphProto (field 1=node:NodeProto[], 11=input:ValueInfoProto[],
                12=output:ValueInfoProto[], 2=name:string)
  - NodeProto (field 1=input:string[], 2=output:string[], 3=name:string,
               4=op_type:string, 7=attribute:AttributeProto[])
  - ValueInfoProto (field 1=name:string, 2=type:TypeProto)
  - TypeProto: tensor_type (field 1=elem_type:int32, 2=shape:TensorShapeProto)
  - TensorShapeProto: dim (field 1=dim_value:int64)

Protobuf wire encoding rules (you implement these):
  - varint (wire type 0): encode u64 as LEB128
  - length-delimited (wire type 2): tag + varint(len) + bytes
  - field tag: (field_number << 3) | wire_type
  - string: wire type 2, utf-8 bytes
  - embedded message: wire type 2, recursively encoded bytes
  - repeated fields: one tag+value per element

DELIVERABLES
============
1. src/proto/mod.rs — protobuf encoder helpers
2. src/codegen/onnx_binary.rs — emit_onnx_binary(graph, shapes) -> Vec<u8>
3. EmitKind::OnnxBinary variant → CLI --emit onnx-binary → writes bytes to stdout
   (or -o file.onnx)
4. 8 tests in tests/phase14.rs

TESTS (tests/phase14.rs)
========================
1. test_proto_varint_1 — encode_varint(1) == [0x01]
2. test_proto_varint_300 — encode_varint(300) == [0xAC, 0x02]
3. test_proto_string_field — encode_string_field(1, "hello") == expected bytes
4. test_onnx_binary_starts_with_magic — emit a simple model; result starts
   with the ModelProto ir_version field encoding (field 1, varint, value >= 7)
5. test_onnx_binary_matmul_node — a MatMul model encodes a NodeProto with
   op_type "MatMul" (look for 0x4D 0x61 0x74 0x4D 0x75 0x6C bytes)
6. test_onnx_binary_has_graph — output contains "MatMul" or "Linear" as bytes
7. test_onnx_binary_opset — encoded opset_import has domain="" version=17
8. test_onnx_binary_roundtrip_size — encoding a 2-node graph produces > 20 bytes

IMPORTANT NOTES
===============
- Rust toolchain: /c/Users/Moon/.cargo/bin/cargo
- Only dependency: thiserror = "1.0" — do NOT add prost, protobuf, etc.
- The hand-rolled encoder is ~150 lines. Focus on correctness of varint/LEB128
  first, then build up the ONNX-specific message structure.
- ONNX ir_version for ONNX 1.14 is 8. Opset version 17 or 19 is fine.
- Field numbers from the ONNX protobuf spec:
  ModelProto: ir_version=1, opset_import=7, graph=8
  GraphProto: name=2, node=1, input=11, output=12, initializer=5
  NodeProto: input=1, output=2, name=3, op_type=4, attribute=7
  ValueInfoProto: name=1, type=2
  TypeProto: tensor_type=1 (oneof)
  TypeProto.Tensor: elem_type=1, shape=2
  TensorShapeProto: dim=1
  TensorShapeProto.Dimension: dim_value=1
  OperatorSetIdProto: domain=1, version=2
```

---

## MASTER PROMPT — Use This First to Orient a New Session

```
You are working on the IRIS compiler, a Rust-based ML DSL compiler.

PROJECT LOCATION: c:\Users\Moon\Desktop\Projects\IRIS
CARGO: /c/Users/Moon/.cargo/bin/cargo  (NOT in PATH — always use full path)
DEPENDENCY: only `thiserror = "1.0"` in Cargo.toml
SHELL: bash (Unix syntax, forward slashes, even on Windows)
TESTS: /c/Users/Moon/.cargo/bin/cargo test  — currently 83 tests, all pass

ARCHITECTURE OVERVIEW
=====================
Pipeline: .iris source → Lexer → Tokens → Parser → AST → Lowerer → IrModule
          → PassManager → Codegen → output

IR Design: Block-parameter SSA (MLIR-style). Blocks have typed parameters
instead of phi nodes. Entry block params = function args.
Index-based arenas: Vec<T> indexed by newtype IDs (BlockId(u32), ValueId(u32)).

KEY FILES
=========
src/lib.rs                  — compile() entry point, EmitKind enum
src/main.rs                 — CLI arg parsing, file I/O
src/error.rs                — ParseError, LowerError, PassError, CodegenError
src/parser/lexer.rs         — Lexer, Token enum, Span
src/parser/ast.rs           — AST node types
src/parser/parse.rs         — Recursive-descent parser
src/ir/instr.rs             — IrInstr enum (all instructions), BinOp, TensorOp,
                              ScalarUnaryOp
src/ir/module.rs            — IrModule, IrFunctionBuilder
src/ir/types.rs             — IrType, DType, Shape, Dim
src/lower/mod.rs            — AST→IR lowering
src/pass/mod.rs             — Pass trait, PassManager
src/pass/validate.rs        — ValidatePass (SSA invariants)
src/pass/type_infer.rs      — TypeInferPass
src/pass/const_fold.rs      — ConstFoldPass
src/pass/opt.rs             — DcePass, CsePass, OpExpandPass
src/pass/shape_check.rs     — ShapeCheckPass
src/codegen/printer.rs      — IR text emitter
src/codegen/llvm_stub.rs    — LLVM IR emitter (scalar, full body with phi nodes)
src/codegen/graph_printer.rs — graph text emitter
src/codegen/onnx.rs         — ONNX text stub emitter

CURRENT LANGUAGE FEATURES
==========================
- fn definitions with typed params and return type
- let bindings (semicolon optional), block tail expressions
- Scalar types: f32, f64, i32, i64, bool
- Tensor types: tensor<f32, [M, K]>
- Arithmetic: +, -, *, /  (no % yet)
- Comparisons: ==, !=, <, <=, >, >=
- Unary: -, !
- if/else expressions (with else optional)
- Function calls
- einsum("notation", t1, t2, ...) intrinsic
- model DSL for layer graphs

MISSING FEATURES (choose one per session)
==========================================
Phase 9:  while/loop/break/continue
Phase 10: t[i] tensor indexing, % modulo, `as` casts
Phase 11: tree-walking interpreter (--emit eval)
Phase 12: line:col error messages, --help, -o, --dump-ir-after
Phase 13: struct types and field access
Phase 14: real ONNX binary protobuf output

PASS PIPELINE ORDER
===================
1. ValidatePass      — SSA structural correctness, rejects IrType::Infer
2. TypeInferPass     — type consistency (read-only, does not mutate types)
3. ConstFoldPass     — constant arithmetic simplification
4. OpExpandPass      — expand activation calls to TensorOp::Unary
5. DcePass           — dead code elimination
6. CsePass           — common subexpression elimination
7. ShapeCheckPass    — tensor shape consistency

WORKFLOW
========
1. Read relevant source files BEFORE making changes.
2. Make minimal targeted changes — do not refactor unrelated code.
3. Run /c/Users/Moon/.cargo/bin/cargo test after each file change.
4. Write tests LAST (after implementation compiles).
5. Target: all existing tests still pass + 8 new tests.
6. Zero compiler warnings.
```

---
