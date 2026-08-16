//! Tail-call elimination for self-recursive functions.
//!
//! IRIS claimed tail-call optimisation for a long time without having it.
//! `tests/test_tco.iris` was titled "Test tail-call optimization (TCO) in the
//! interpreter" and its `sum_to(50000)` ran only because the interpreter's depth
//! limit used to be 5,000 — it genuinely built 50,000 frames. There was no
//! `musttail` in codegen either. See known-issues #35.
//!
//! # Why this is an IR pass
//!
//! Doing it here rather than per-backend means both the interpreter and the
//! native backend get it from one implementation, and neither can drift from the
//! other. It also means the transformation is visible in `--emit ir`, so it can
//! be inspected rather than trusted.
//!
//! # The transformation
//!
//! A self-call whose result is returned becomes a jump back to the top of the
//! function with new arguments. The complication is that the entry block cannot
//! be a branch target — LLVM rejects a branch to the entry block — so the body
//! moves into a fresh *loop header* and the entry block becomes a single jump
//! into it:
//!
//! ```text
//!   entry(n, acc):                     entry(n, acc):
//!     ...body...            ==>          br header(n, acc)
//!     %r = call @self(a, b)            header(n', acc'):
//!     br merge(%r)                       ...body...
//!   merge(%v):                           br header(a, b)      <- was the call
//!     return %v                        merge(%v):
//!                                        return %v            <- now unreachable
//! ```
//!
//! # What counts as a tail call
//!
//! A call to the enclosing function whose result is returned without any further
//! computation. In IRIS's block-parameter SSA the call is rarely adjacent to the
//! `Return`: an `if`/`else` puts the call in one arm and the `Return` in the
//! merge block, so the result travels as a block argument. This pass follows one
//! hop of that — the shape every tail-recursive function in the corpus has.
//!
//! # Deliberate limits
//!
//! Only *self* recursion is handled. Mutual recursion between two functions
//! needs a shared trampoline and is a different transformation; it is left alone
//! rather than half-done. Calls whose result is used for anything before being
//! returned — `n * fact(n - 1)` — are not tail calls at all and are untouched.

use crate::error::PassError;
use crate::ir::block::IrBlock;
use crate::ir::function::{FunctionId, IrFunction, Param};
use crate::ir::instr::{BinOp, IrInstr};
use crate::ir::module::IrModule;
use crate::ir::types::{DType, IrType};
use crate::ir::value::{BlockParam, ValueId};
use crate::ir::BlockId;
use crate::pass::Pass;
use std::collections::HashMap;

pub struct TailCallPass;

impl Pass for TailCallPass {
    fn name(&self) -> &'static str {
        "tail-call"
    }

    fn run(&mut self, module: &mut IrModule) -> Result<(), PassError> {
        // Mutual recursion first: merging an SCC turns its inter-function tail
        // calls into branches, so the self pass afterwards has less to do and
        // never sees a half-transformed function.
        eliminate_mutual_tail_calls(module);
        for fi in 0..module.functions.len() {
            eliminate_self_tail_calls(module, fi);
        }
        Ok(())
    }
}

/// One tail call: where it sits, who it calls, and what it passes.
struct TailSite {
    bi: usize,
    ii: usize,
    callee: String,
    args: Vec<ValueId>,
}

/// Every tail call in `func`, to itself or to anything else.
fn find_tail_sites(func: &IrFunction) -> Vec<TailSite> {
    let mut sites = Vec::new();
    for (bi, block) in func.blocks.iter().enumerate() {
        for (ii, instr) in block.instrs.iter().enumerate() {
            let IrInstr::Call {
                result: Some(result),
                callee,
                args,
                ..
            } = instr
            else {
                continue;
            };
            // The call must be the last non-terminator instruction, and the
            // terminator must hand its result straight to a returning block.
            if ii + 2 != block.instrs.len() {
                continue;
            }
            let IrInstr::Br {
                target,
                args: br_args,
            } = &block.instrs[ii + 1]
            else {
                continue;
            };
            // Passing the result twice, or alongside other values, is not a
            // shape this pass reasons about.
            if br_args.len() != 1 {
                continue;
            }
            let Some(slot) = br_args.iter().position(|v| v == result) else {
                continue;
            };
            if !block_just_returns_param(func, *target, slot) {
                continue;
            }
            sites.push(TailSite {
                bi,
                ii,
                callee: callee.clone(),
                args: args.clone(),
            });
        }
    }
    sites
}

/// True when branching to `target` with `value` in argument slot `slot`
/// immediately returns that value and nothing else.
fn block_just_returns_param(func: &crate::ir::function::IrFunction, target: BlockId, slot: usize) -> bool {
    let Some(block) = func.block(target) else {
        return false;
    };
    // Exactly one instruction, and it returns the parameter that this argument
    // slot feeds. Anything else -- a retain, a print, a second return value --
    // means the call's result is not simply handed back to the caller.
    if block.instrs.len() != 1 {
        return false;
    }
    match &block.instrs[0] {
        IrInstr::Return { values } if values.len() == 1 => {
            block.params.get(slot).map(|p| p.id) == Some(values[0])
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Mutual recursion: the trampoline
// ---------------------------------------------------------------------------
//
// `is_even`/`is_odd` hand control to each other in tail position. Neither call
// is a *self* call, so the transformation above does nothing, and interpreted
// they die at `call depth exceeded 250`. (Natively they happen to survive,
// because clang does its own tail-call optimisation on the emitted IR -- which
// is precisely the kind of backend disagreement this pass exists to remove.)
//
// The classic trampoline returns a "call this next" thunk to a driver loop.
// That needs a tagged return value, so every mutually-recursive function would
// have to change its return type -- unacceptable in a statically typed language
// where `is_even` must keep returning `i64`.
//
// So the SCC is *merged* instead. The members become blocks of one function
// with a leading selector parameter, and their tail calls to each other become
// branches -- a genuine loop, in constant stack:
//
//   def is_even(n) { if n==0 {1} else { is_odd(n-1)  } }
//   def is_odd(n)  { if n==0 {0} else { is_even(n-1) } }
//        |
//        v
//   __tramp$is_even$is_odd(sel, n):
//     dispatch0(sel, n):     condbr sel==0 -> is_even$entry(n) : is_odd$entry(n)
//     is_even$entry(n):      ... br is_odd$entry(n-1)      <- was a call
//     is_odd$entry(n):       ... br is_even$entry(n-1)     <- was a call
//
//   def is_even(n) { __tramp$is_even$is_odd(0, n) }   <- thin forwarder
//   def is_odd(n)  { __tramp$is_even$is_odd(1, n) }
//
// The originals survive as forwarders, so non-tail calls, function values and
// anything referencing them by name keep working; only the depth changes.
//
// # Deliberate limit
//
// The members must share one parameter list (same arity, same types) and one
// return type. That is what lets a single merged signature serve all of them.
// Where they differ, the merged function would need a parameter union and a
// synthesised dummy value for every slot the entering member does not supply --
// and for `str`, records or lists there is no dummy to synthesise. Such an SCC
// is left alone rather than merged wrongly. This covers the shape mutual tail
// recursion actually takes: state machines, parity, scanners.

/// Rewrites `Br`/`CondBr`/`SwitchVariant` targets by a fixed offset.
///
/// Only these three carry a `BlockId` -- verified against the `IrInstr`
/// declaration. A new *terminator* variant would need an arm here; the
/// catch-all is safe for anything else because nothing else names a block.
/// `ValidatePass` range-checks targets downstream, so a missed arm surfaces as
/// a validation failure rather than as a wrong jump.
fn shift_block_targets(instr: &mut IrInstr, offset: u32) {
    match instr {
        IrInstr::Br { target, .. } => target.0 += offset,
        IrInstr::CondBr {
            then_block,
            else_block,
            ..
        } => {
            then_block.0 += offset;
            else_block.0 += offset;
        }
        IrInstr::SwitchVariant {
            arms,
            default_block,
            ..
        } => {
            for (_, b) in arms.iter_mut() {
                b.0 += offset;
            }
            if let Some(b) = default_block {
                b.0 += offset;
            }
        }
        _ => {}
    }
}

/// Groups `nodes` into strongly connected components of `adj`.
///
/// Pairwise reachability rather than Tarjan: the graph here is only the
/// functions that tail-call another function, which is a handful even in a
/// large module, and this needs no recursion (the compiler already has one
/// stack-depth problem, which is the subject of this file).
fn tail_sccs(nodes: &[usize], adj: &HashMap<usize, Vec<usize>>) -> Vec<Vec<usize>> {
    let reach = |from: usize| -> Vec<usize> {
        let mut seen = vec![from];
        let mut stack = vec![from];
        while let Some(n) = stack.pop() {
            for &m in adj.get(&n).map(|v| v.as_slice()).unwrap_or(&[]) {
                if !seen.contains(&m) {
                    seen.push(m);
                    stack.push(m);
                }
            }
        }
        seen
    };

    let mut assigned: Vec<usize> = Vec::new();
    let mut out: Vec<Vec<usize>> = Vec::new();
    for &n in nodes {
        if assigned.contains(&n) {
            continue;
        }
        let fwd = reach(n);
        // Mutually reachable == same component.
        let mut group: Vec<usize> = fwd
            .iter()
            .copied()
            .filter(|&m| m == n || reach(m).contains(&n))
            .collect();
        group.sort_unstable();
        assigned.extend(group.iter().copied());
        out.push(group);
    }
    out
}

fn eliminate_mutual_tail_calls(module: &mut IrModule) {
    // Edges are tail calls to a *different* function in this module.
    let name_to_idx: HashMap<String, usize> = module
        .functions
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name.clone(), i))
        .collect();

    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, f) in module.functions.iter().enumerate() {
        for s in find_tail_sites(f) {
            if s.callee == f.name {
                continue;
            }
            if let Some(&j) = name_to_idx.get(&s.callee) {
                adj.entry(i).or_default().push(j);
            }
        }
    }
    if adj.is_empty() {
        return;
    }

    // Only functions that participate in an inter-function tail call can be in a
    // non-trivial SCC. Sorted so the merge order -- and therefore the generated
    // names and block numbering -- is identical run to run.
    let mut nodes: Vec<usize> = adj.keys().copied().collect();
    for vs in adj.values() {
        for &v in vs {
            if !nodes.contains(&v) {
                nodes.push(v);
            }
        }
    }
    nodes.sort_unstable();

    for scc in tail_sccs(&nodes, &adj) {
        if scc.len() < 2 {
            continue;
        }
        if scc_is_mergeable(module, &scc) {
            merge_scc(module, &scc);
        }
    }
}

/// One signature has to serve every member of the merged function.
fn scc_is_mergeable(module: &IrModule, scc: &[usize]) -> bool {
    let proto = &module.functions[scc[0]];
    if proto.blocks.is_empty() {
        return false;
    }
    scc.iter().all(|&i| {
        let f = &module.functions[i];
        !f.blocks.is_empty()
            // The entry block's params *are* the arguments; a mismatch here
            // means the function was built unusually and is not worth guessing at.
            && f.blocks[0].params.len() == f.params.len()
            && f.params.len() == proto.params.len()
            && f.params
                .iter()
                .zip(proto.params.iter())
                .all(|(a, b)| a.ty == b.ty)
            && f.return_ty == proto.return_ty
            // Captures are passed as leading params; a closure body is not a
            // free-standing function and must not be folded into a shared one.
            && f.capture_count == 0
            // `const def` runs at compile time; leave that path untouched.
            && !f.is_const
    })
}

fn merge_scc(module: &mut IrModule, scc: &[usize]) {
    let k = scc.len();
    let member_names: Vec<String> = scc
        .iter()
        .map(|&i| module.functions[i].name.clone())
        .collect();
    let tramp_name = format!("__tramp${}", member_names.join("$"));
    if module.functions.iter().any(|f| f.name == tramp_name) {
        return;
    }

    let i64_ty = IrType::Scalar(DType::I64);
    let bool_ty = IrType::Scalar(DType::Bool);
    let proto_params: Vec<Param> = module.functions[scc[0]].params.clone();
    let return_ty = module.functions[scc[0]].return_ty.clone();

    let mut t = IrFunction {
        id: FunctionId(0),
        name: tramp_name.clone(),
        params: std::iter::once(Param {
            name: "__sel".to_owned(),
            ty: i64_ty.clone(),
        })
        .chain(proto_params.iter().cloned())
        .collect(),
        return_ty: return_ty.clone(),
        blocks: Vec::new(),
        value_defs: HashMap::new(),
        value_types: HashMap::new(),
        next_value: 0,
        attrs: Vec::new(),
        span_table: Default::default(),
        capture_count: 0,
        is_const: false,
    };

    // Layout: `k - 1` dispatch blocks, then each member's blocks contiguously.
    // A member's entry is therefore at `base[mi]`, which is what a tail call to
    // it becomes a branch to.
    let n_dispatch = k - 1;
    let mut base = Vec::with_capacity(k);
    let mut off = n_dispatch;
    for &fi in scc {
        base.push(off);
        off += module.functions[fi].blocks.len();
    }

    // Dispatch: a chain of `sel == j` tests. Each block carries the selector and
    // the arguments onward, so no value has to outlive its block.
    for j in 0..n_dispatch {
        let mut b = IrBlock::new(BlockId(j as u32), Some(format!("tramp_dispatch{j}")));
        let sel = t.fresh_value();
        t.value_types.insert(sel, i64_ty.clone());
        b.params.push(BlockParam {
            id: sel,
            ty: i64_ty.clone(),
            name: Some("__sel".to_owned()),
        });
        let mut ps = Vec::with_capacity(proto_params.len());
        for p in &proto_params {
            let v = t.fresh_value();
            t.value_types.insert(v, p.ty.clone());
            b.params.push(BlockParam {
                id: v,
                ty: p.ty.clone(),
                name: Some(p.name.clone()),
            });
            ps.push(v);
        }
        let cst = t.fresh_value();
        t.value_types.insert(cst, i64_ty.clone());
        let cnd = t.fresh_value();
        t.value_types.insert(cnd, bool_ty.clone());
        b.instrs.push(IrInstr::ConstInt {
            result: cst,
            value: j as i64,
            ty: i64_ty.clone(),
        });
        b.instrs.push(IrInstr::BinOp {
            result: cnd,
            op: BinOp::CmpEq,
            lhs: sel,
            rhs: cst,
            ty: bool_ty.clone(),
        });
        // The last test's `else` is the final member -- no `sel == k-1` check is
        // needed once every other value has been excluded.
        let (else_block, else_args) = if j + 1 < n_dispatch {
            (
                BlockId((j + 1) as u32),
                std::iter::once(sel).chain(ps.iter().copied()).collect(),
            )
        } else {
            (BlockId(base[k - 1] as u32), ps.clone())
        };
        b.instrs.push(IrInstr::CondBr {
            cond: cnd,
            then_block: BlockId(base[j] as u32),
            then_args: ps.clone(),
            else_block,
            else_args,
        });
        t.blocks.push(b);
    }

    // Copy each member in, renumbering its values and blocks. Members number
    // their values from zero independently, so without this they collide.
    let mut vmaps: Vec<HashMap<ValueId, ValueId>> = Vec::with_capacity(k);
    let mut sites_per_member: Vec<Vec<TailSite>> = Vec::with_capacity(k);
    for (mi, &fi) in scc.iter().enumerate() {
        let src = module.functions[fi].clone();
        sites_per_member.push(find_tail_sites(&src));

        let mut vmap: HashMap<ValueId, ValueId> = HashMap::new();
        for blk in &src.blocks {
            for p in &blk.params {
                let nv = t.fresh_value();
                t.value_types.insert(
                    nv,
                    src.value_types.get(&p.id).cloned().unwrap_or(p.ty.clone()),
                );
                vmap.insert(p.id, nv);
            }
            for ins in &blk.instrs {
                if let Some(r) = ins.result() {
                    let nv = t.fresh_value();
                    if let Some(ty) = src.value_types.get(&r) {
                        t.value_types.insert(nv, ty.clone());
                    }
                    vmap.insert(r, nv);
                }
            }
        }

        for (bj, blk) in src.blocks.iter().enumerate() {
            let nid = BlockId((base[mi] + bj) as u32);
            let label = match &blk.name {
                Some(n) => format!("{}${}", src.name, n),
                None => format!("{}$bb{}", src.name, bj),
            };
            let mut nb = IrBlock::new(nid, Some(label));
            nb.params = blk
                .params
                .iter()
                .map(|p| BlockParam {
                    id: vmap[&p.id],
                    ty: p.ty.clone(),
                    name: p.name.clone(),
                })
                .collect();
            for ins in &blk.instrs {
                let mut ni = ins.clone();
                crate::pass::opt::apply_replacements(&mut ni, &vmap);
                if let Some(r) = ins.result() {
                    crate::pass::inline::set_result(&mut ni, vmap[&r]);
                }
                shift_block_targets(&mut ni, base[mi] as u32);
                nb.instrs.push(ni);
            }
            t.blocks.push(nb);
        }
        vmaps.push(vmap);
    }

    // Tail calls between members become branches to the callee's entry block.
    // A member's *self* tail calls are included: inside the merged function its
    // entry is an ordinary block, so it can be branched to directly.
    for mi in 0..k {
        for s in &sites_per_member[mi] {
            let Some(target_mi) = member_names.iter().position(|n| *n == s.callee) else {
                continue;
            };
            let args: Vec<ValueId> = s
                .args
                .iter()
                .map(|v| *vmaps[mi].get(v).unwrap_or(v))
                .collect();
            let blk = &mut t.blocks[base[mi] + s.bi];
            blk.instrs.truncate(s.ii);
            blk.instrs.push(IrInstr::Br {
                target: BlockId(base[target_mi] as u32),
                args,
            });
        }
    }

    if module.add_function(t).is_err() {
        return;
    }

    // The originals become forwarders. Everything that referred to them by name
    // still works; only the stack depth changes.
    for (mi, &fi) in scc.iter().enumerate() {
        let f = &mut module.functions[fi];
        let entry_params: Vec<BlockParam> = f.blocks[0].params.clone();
        let ret_ty = f.return_ty.clone();
        let sel = f.fresh_value();
        f.value_types.insert(sel, i64_ty.clone());
        let ret = f.fresh_value();
        f.value_types.insert(ret, ret_ty.clone());

        let mut b = IrBlock::new(BlockId(0), Some("entry".to_owned()));
        b.params = entry_params.clone();
        b.instrs = vec![
            IrInstr::ConstInt {
                result: sel,
                value: mi as i64,
                ty: i64_ty.clone(),
            },
            IrInstr::Call {
                result: Some(ret),
                callee: tramp_name.clone(),
                args: std::iter::once(sel)
                    .chain(entry_params.iter().map(|p| p.id))
                    .collect(),
                result_ty: Some(ret_ty),
            },
            IrInstr::Return { values: vec![ret] },
        ];
        f.blocks = vec![b];
        // Definition sites all referred to blocks that no longer exist.
        f.value_defs.clear();
    }
}

fn eliminate_self_tail_calls(module: &mut IrModule, fn_idx: usize) {
    let fn_name = module.functions[fn_idx].name.clone();

    // 1. Find the self tail calls: (block index, instr index, call arguments).
    let sites: Vec<(usize, usize, Vec<ValueId>)> = find_tail_sites(&module.functions[fn_idx])
        .into_iter()
        .filter(|s| s.callee == fn_name)
        .map(|s| (s.bi, s.ii, s.args))
        .collect();
    if sites.is_empty() {
        return;
    }

    // 2. Move the body into a fresh loop header, because the entry block cannot
    //    be a branch target.
    let _entry_id = module.functions[fn_idx].blocks[0].id;
    let entry_params: Vec<BlockParam> = module.functions[fn_idx].blocks[0].params.clone();

    // `BlockId` *is* the index -- `IrFunction::block` is `blocks.get(id.0)`. So the
    // header's id must be the position it is about to occupy, not `max(id) + 1`.
    // Those two agree only while ids are contiguous, which is true today (no pass
    // removes a block) but is a landmine for the first pass that does: the id
    // would resolve to a different block, or to none, in every backend at once.
    let header_pos = module.functions[fn_idx].blocks.len();
    let header_id = BlockId(header_pos as u32);

    // Fresh parameters for the header, carrying the entry's types.
    let mut remap: HashMap<ValueId, ValueId> = HashMap::new();
    let mut header_params: Vec<BlockParam> = Vec::with_capacity(entry_params.len());
    for p in &entry_params {
        let nid = module.functions[fn_idx].fresh_value();
        module.functions[fn_idx]
            .value_types
            .insert(nid, p.ty.clone());
        remap.insert(p.id, nid);
        header_params.push(BlockParam {
            id: nid,
            ty: p.ty.clone(),
            name: p.name.clone(),
        });
    }

    let body_instrs = std::mem::take(&mut module.functions[fn_idx].blocks[0].instrs);

    // The entry's parameters are the function's parameters and may be referenced
    // from any block, not only the entry -- IRIS IR permits cross-block
    // references. So the rename has to cover the whole function.
    for block in module.functions[fn_idx].blocks.iter_mut() {
        for instr in block.instrs.iter_mut() {
            crate::pass::opt::apply_replacements(instr, &remap);
        }
    }

    let mut header = crate::ir::block::IrBlock::new(header_id, Some("tail_header".to_owned()));
    header.params = header_params;
    header.instrs = body_instrs;
    for instr in header.instrs.iter_mut() {
        crate::pass::opt::apply_replacements(instr, &remap);
    }
    module.functions[fn_idx].blocks.push(header);

    // The entry block keeps its own parameters and simply enters the loop.
    module.functions[fn_idx].blocks[0].instrs = vec![IrInstr::Br {
        target: header_id,
        args: entry_params.iter().map(|p| p.id).collect(),
    }];

    // 3. Rewrite each call site into a jump back to the header.
    //
    // The block indices captured in step 1 still refer to the same blocks: the
    // header was appended, and the entry block's body moved wholesale, so the
    // sites recorded for block 0 now live in the header.
    debug_assert_eq!(module.functions[fn_idx].blocks[header_pos].id, header_id);
    for (bi, ii, call_args) in sites {
        let target_block = if bi == 0 { header_pos } else { bi };
        let args: Vec<ValueId> = call_args
            .iter()
            .map(|v| *remap.get(v).unwrap_or(v))
            .collect();
        let block = &mut module.functions[fn_idx].blocks[target_block];
        // Drop the call and replace the branch that carried its result.
        block.instrs.truncate(ii);
        block.instrs.push(IrInstr::Br {
            target: header_id,
            args,
        });
    }
}
