# How rustc Is Wired, and What IRIS Should Borrow

**Date**: 2026-08-05
**Sources**: [rustc-dev-guide overview](https://rustc-dev-guide.rust-lang.org/overview.html);
IRIS architecture read directly from the source tree.

The purpose here is not to imitate rustc. It is to identify *which* of rustc's
structural choices explain the specific classes of bug IRIS is currently hitting,
and which are irrelevant at IRIS's scale.

---

## 1. The rustc pipeline

| # | Stage | Operates on | Responsible for |
|---|---|---|---|
| 1 | Lexing (`rustc_lexer`) | source text | tokens |
| 2 | Parsing | tokens | AST |
| 3 | Expansion & resolution | AST | macro expansion, AST validation, **name resolution**, early lints |
| 4 | AST lowering | AST | **HIR** — desugared, name-resolved |
| 5 | Type check & inference | **HIR** | type inference, trait solving, type safety |
| 6 | THIR lowering | HIR | **THIR** — fully typed, explicit method calls and derefs |
| 7 | MIR lowering | THIR | **MIR** — control-flow graph of basic blocks |
| 8 | Borrow checking | **MIR** | dataflow-based memory safety (NLL) |
| 9 | MIR optimisation | MIR | generic, pre-monomorphisation optimisation |
| 10 | Monomorphisation collection | MIR | worklist of concrete instantiations |
| 11 | Codegen | MIR → LLVM IR | machine code |

Two cross-cutting systems: **queries** (every stage is a memoised query hanging off
`TyCtxt`, giving incremental compilation via dependency tracking) and
**diagnostics** (`Diag`; the parser accepts a superset of the grammar so it can
recover and report many errors per run).

**The critical ordering fact:** rustc types the program at stage 5, on HIR — a
tree that still mirrors the source. MIR, the CFG, is only produced at stage 7,
*after* everything is fully typed. Types flow **downhill** into the CFG; they are
never re-derived from it.

---

## 2. The IRIS pipeline as it stands

| # | Stage | Operates on | Notes |
|---|---|---|---|
| 1 | Lexer | source | ~46 keywords |
| 2 | Parser | tokens | AST, 35 expr / 18 stmt / 23 type forms |
| 3 | Preprocessor | source | `#ifdef` family, before lexing |
| 4 | AST passes | AST | `ast_exhaustive`, `variance_checker`, **`borrow_checker`**, `effect_checker` |
| 5 | Lowering | AST → IR | **also performs monomorphisation** (`mono_sigs`, `generic_fns`, `resolve_generic_struct_type`) |
| 6 | `HmTypeInferPass` | **IR** | Hindley-Milner over block-parameter SSA; **unresolved slots default to `i64`** |
| 7 | `ValidatePass` | IR | rejects any surviving `IrType::Infer` |
| 8 | ~20 further passes | IR | const-fold, LICM, inline, DCE, CSE, GC annotate, shape check … |
| 9 | Codegen | IR | LLVM text, CUDA, SIMD, WASM, ONNX |

Mapped onto rustc: IRIS's AST ≈ rustc's AST, and `IrModule` ≈ MIR (a CFG, using
block parameters instead of phi nodes — an MLIR-style choice, and a good one).

**There is no HIR and no THIR.** Nothing sits between the syntax tree and the CFG.

---

## 3. The three structural differences that explain the bugs

### 3.1 Type inference runs *after* lowering — this is the big one

rustc types HIR (stage 5) and lowers to MIR (stage 7). IRIS lowers first (stage 5)
and infers types afterwards, on the CFG (stage 6).

The consequence is that type information which was obvious in the source must be
**re-derived from IR context** once the source structure is gone. That is exactly
the shape of the defects found:

- `str` field in a record inside `result<T, E>` mis-typed as `i64`, failing LLVM
  verification (`'%v14' defined with type 'i64' but expected 'ptr'`).
- Named arguments silently evaluating to `0` — argument identity is resolved
  during lowering, before any type information exists to check it against.
- `IrType::Infer` existing at all as a lowering output.

And the fallback makes it worse: **unresolved inference slots default to `i64`**.
That converts "I could not determine this type" into "it is an integer", which is
precisely how a `str` becomes an `i64` and a wrong answer reaches the user instead
of an error. `ValidatePass` then rejects only *surviving* `Infer` — but the default
already removed them.

`MEMORY.md` records the workaround this forced: the lowerer must eagerly propagate
concrete scalar types "to avoid `Infer` at validation time", and unknown calls had
to start erroring explicitly because inference would otherwise silently resolve
them to `i64`. Those are symptoms of the ordering, not the cause.

### 3.2 Borrow checking runs on the AST, not the CFG

rustc borrow-checks MIR (stage 8) using dataflow over a control-flow graph — that
is what makes non-lexical lifetimes possible at all.

IRIS borrow-checks the AST (`src/pass/borrow_checker.rs`), with scope-based
lifetimes and borrows cleared at scope exit. A tree has no notion of "this path
reaches that use", so the analysis cannot be path-sensitive. It will accept unsound
programs and reject sound ones, and no amount of work on the AST version fixes
that — the information is not present in the representation.

### 3.3 Monomorphisation happens inside the lowerer

rustc collects a monomorphisation worklist as a distinct stage (10) over already
optimised, already type-checked MIR.

IRIS monomorphises *during* lowering, which means the lowerer is simultaneously
desugaring, resolving generics, substituting types, and building SSA. `src/lower/mod.rs`
is **17,262 lines** — by far the largest file in the project. That is the direct
cost of one stage owning four responsibilities, and it is why bugs there are hard
to localise.

---

## 4. What to actually do

Ordered by value-per-unit-of-disruption. This is a refactor of *ordering*, not a
rewrite — most existing code moves rather than disappears.

### 4.1 Stop defaulting unresolved types to `i64` — do this first

A one-line policy change with outsized effect: make an unresolved inference slot a
**compile error** with a span, not an `i64`.

This will surface a set of latent failures immediately, which is the point — every
one of them is currently a silent wrong answer. Do it after Phase 1.2 of the
release plan (the test-assertion conversion), so the fallout is visible.

**Effort:** hours to change, days to work through the fallout.

### 4.2 Introduce a typed HIR between AST and lowering

The central fix. Add a representation that is:

- name-resolved (a distinct resolution stage, as rustc has at stage 3),
- desugared (f-strings, map literals, `for`-destructuring — the "cosmetic"
  desugarings the no-desugaring audit catalogued),
- **fully typed** before lowering begins.

Then lowering consumes a typed tree and *cannot* produce `IrType::Infer` — the
variant becomes unrepresentable in its output rather than something a later pass
must clean up.

Move `HmTypeInferPass` from operating on IR to operating on this tree. The
union-find machinery is reusable; only its substrate changes.

**Effort:** weeks. This is the largest item and the one that pays off most.

### 4.3 Move borrow checking to the IR CFG

Once the IR is reliably typed, re-target `borrow_checker.rs` from AST to
`IrModule` and make it a dataflow analysis over the block graph. IRIS already has
the CFG; it is simply not being used for this.

**Effort:** weeks. Do after 4.2.

### 4.4 Split monomorphisation out of the lowerer

Make it a distinct stage over typed IR, as rustc does. The immediate benefit is
that `src/lower/mod.rs` stops being a 17k-line file with four jobs, and generic
resolution bugs become localisable.

**Effort:** weeks. Can proceed in parallel with 4.3.

### 4.5 Adopt rustc's diagnostics discipline, not its diagnostics code

Two cheap wins already visible as problems:

- Errors should carry a **source span** and point at user syntax. The current
  `variable '%30' is used before it has been assigned` names an internal SSA value
   — meaningless to a user.
- Keep parsing after an error to report many per run. IRIS already does this via
  `parse_recovering()`; extend the same discipline to the type and lowering stages.

**Effort:** days, and it disproportionately improves the impression the compiler
makes on a first-time user.

---

## 5. What *not* to copy

- **The query system.** rustc's `TyCtxt`/query architecture exists to make
  incremental compilation viable across a huge codebase with separate crate
  compilation. IRIS is a single crate compiling single projects, and already has a
  file-level `BuildCache`. Adopting queries now would be a very large refactor
  serving a problem IRIS does not yet have. Revisit only if whole-project rebuild
  times become the binding constraint.
- **Four IRs.** IRIS does not need both HIR and THIR. One typed, desugared,
  name-resolved tree between AST and the SSA IR is sufficient — rustc's split
  exists partly for historical reasons.
- **Block parameters → phi nodes.** IRIS's MLIR-style block parameters are a
  *better* choice than LLVM-style phi nodes. Keep them.

---

## 6. Summary

IRIS's IR design is sound; its **stage ordering** is not. rustc establishes types
on a source-shaped tree and lets them flow downhill into the CFG. IRIS builds the
CFG first and then tries to recover types from it, with an `i64` default when it
cannot — which turns unknown types into wrong answers rather than errors.

The single highest-value change is **4.1** (fail instead of defaulting to `i64`),
because it is cheap and converts a class of silent corruption into visible errors.
The single most important change is **4.2** (a typed layer before lowering),
because it removes the conditions that make that class possible at all.

Neither is a rewrite. The lexer, parser, IR design, pass framework, and all 15
backends are unaffected.
