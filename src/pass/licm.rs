//! Loop-Invariant Code Motion (LICM) pass for `IrModule`.
//!
//! Moves instructions that compute the same value on every loop iteration
//! out of the loop body into the loop preheader. An instruction is loop-
//! invariant if all its operands are either:
//! - Defined outside the loop, or
//! - Loop-invariant themselves (transitively).
//!
//! Only pure (non-side-effecting) instructions are candidates for hoisting.

use std::collections::{HashMap, HashSet};

use crate::error::PassError;
use crate::ir::block::BlockId;
use crate::ir::function::IrFunction;
use crate::ir::instr::IrInstr;
use crate::ir::module::IrModule;
use crate::ir::value::ValueId;
use crate::pass::Pass;

pub struct LicmPass;

impl Pass for LicmPass {
    fn name(&self) -> &'static str {
        "licm"
    }

    fn run(&mut self, module: &mut IrModule) -> Result<(), PassError> {
        for func in &mut module.functions {
            licm_func(func);
        }
        Ok(())
    }
}

/// Detect natural loops and hoist invariant instructions.
fn licm_func(func: &mut IrFunction) {
    if func.blocks.len() < 2 {
        return;
    }

    // Build CFG: map block index → set of successor block indices.
    let num_blocks = func.blocks.len();
    let mut successors: Vec<HashSet<usize>> = vec![HashSet::new(); num_blocks];
    let mut predecessors: Vec<HashSet<usize>> = vec![HashSet::new(); num_blocks];

    for (i, block) in func.blocks.iter().enumerate() {
        if let Some(last) = block.instrs.last() {
            match last {
                IrInstr::Br { target, .. } => {
                    if let Some(idx) = block_index(func, *target) {
                        successors[i].insert(idx);
                        predecessors[idx].insert(i);
                    }
                }
                IrInstr::CondBr {
                    then_block,
                    else_block,
                    ..
                } => {
                    if let Some(idx) = block_index(func, *then_block) {
                        successors[i].insert(idx);
                        predecessors[idx].insert(i);
                    }
                    if let Some(idx) = block_index(func, *else_block) {
                        successors[i].insert(idx);
                        predecessors[idx].insert(i);
                    }
                }
                _ => {}
            }
        }
    }

    // Find back edges (target dominates source) → natural loops.
    // Simple dominator computation: iterative dataflow.
    let mut dom: Vec<HashSet<usize>> = vec![HashSet::new(); num_blocks];
    dom[0].insert(0);
    for block_dom in dom.iter_mut().take(num_blocks).skip(1) {
        *block_dom = (0..num_blocks).collect();
    }

    let mut changed = true;
    while changed {
        changed = false;
        for i in 1..num_blocks {
            let new_dom: HashSet<usize> = if predecessors[i].is_empty() {
                let mut s = HashSet::new();
                s.insert(i);
                s
            } else {
                let mut new = (0..num_blocks).collect::<HashSet<usize>>();
                for &pred in &predecessors[i] {
                    new = new.intersection(&dom[pred]).copied().collect();
                }
                new.insert(i);
                new
            };
            if new_dom != dom[i] {
                dom[i] = new_dom;
                changed = true;
            }
        }
    }

    // Find back edges: edge (src → tgt) where tgt dominates src.
    let mut loops: Vec<(usize, HashSet<usize>)> = Vec::new(); // (header, body blocks)
    for src in 0..num_blocks {
        for &tgt in &successors[src] {
            if dom[src].contains(&tgt) {
                // Back edge found: tgt is loop header.
                // Compute loop body: all blocks that can reach src without going through tgt.
                let mut body: HashSet<usize> = HashSet::new();
                body.insert(tgt);
                if src != tgt {
                    body.insert(src);
                    let mut worklist = vec![src];
                    while let Some(n) = worklist.pop() {
                        for &pred in &predecessors[n] {
                            if body.insert(pred) {
                                worklist.push(pred);
                            }
                        }
                    }
                }
                loops.push((tgt, body));
            }
        }
    }

    // Back edges that share a header belong to the *same* natural loop, and its
    // body is the union of theirs. Treating them as separate loops was wrong in
    // a way that only showed up with a labelled `continue`:
    //
    //     for outer i in 0..4 { for j in 0..4 { if j > i { continue outer; } } }
    //
    // gives the outer header two back edges — one from the inner loop's exit,
    // one from the `continue outer` block. Considered separately, each body
    // excludes the other's latch, so that latch looks like a block *outside*
    // the loop and became a candidate preheader. Hoisting into it placed a
    // definition where it could not reach its use. See known-issues #17.
    {
        let mut merged: HashMap<usize, HashSet<usize>> = HashMap::new();
        for (header, body) in loops.drain(..) {
            merged.entry(header).or_default().extend(body);
        }
        let mut headers: Vec<usize> = merged.keys().copied().collect();
        headers.sort_unstable();
        loops = headers
            .into_iter()
            .map(|h| {
                let body = merged.remove(&h).unwrap_or_default();
                (h, body)
            })
            .collect();
    }

    if loops.is_empty() {
        return;
    }

    // For each loop, identify loop-invariant instructions and hoist them.
    // Collect all definitions: ValueId → (block_index, instr_index).
    let mut def_block: HashMap<ValueId, usize> = HashMap::new();
    for (bi, block) in func.blocks.iter().enumerate() {
        for param in &block.params {
            def_block.insert(param.id, bi);
        }
        for instr in &block.instrs {
            if let Some(result) = instr.result() {
                def_block.insert(result, bi);
            }
        }
    }

    for (header, body) in &loops {
        // Find preheader: a predecessor of the header, outside the loop, that
        // *dominates* the header.
        //
        // The dominance requirement is what makes the hoist sound. A block that
        // merely sits outside the loop body can still fail to reach parts of it,
        // and hoisting a definition into such a block strands its uses. Since
        // the preheader dominates the header, and the header dominates every
        // block in a natural loop, it dominates every use we are hoisting past.
        //
        // Selection is also made deterministic by taking the lowest index rather
        // than an arbitrary `HashSet` element. Without that, the same source
        // compiled to different IR on different runs — three distinct outputs in
        // six runs of the same file, three of which were invalid — because the
        // hash seed changes per process.
        let mut cands: Vec<usize> = predecessors[*header]
            .iter()
            .copied()
            .filter(|p| !body.contains(p) && dom[*header].contains(p))
            .collect();
        cands.sort_unstable();
        let preheader = match cands.first() {
            Some(&p) => p,
            None => continue, // No sound preheader available — skip this loop.
        };

        // Identify loop-invariant instructions.
        let mut invariant: HashSet<ValueId> = HashSet::new();
        let mut changed = true;
        while changed {
            changed = false;
            for &bi in body {
                for instr in &func.blocks[bi].instrs {
                    if let Some(result) = instr.result() {
                        if invariant.contains(&result) {
                            continue;
                        }
                        if is_side_effecting_for_licm(instr) {
                            continue;
                        }
                        // All operands must be either defined outside loop or invariant.
                        let operands = instr.operands();
                        let all_invariant = operands.iter().all(|op| {
                            if let Some(&def_bi) = def_block.get(op) {
                                !body.contains(&def_bi) || invariant.contains(op)
                            } else {
                                true // Unknown def = parameter, treat as outside.
                            }
                        });
                        if all_invariant {
                            invariant.insert(result);
                            changed = true;
                        }
                    }
                }
            }
        }

        if invariant.is_empty() {
            continue;
        }

        // Hoist: move invariant instructions from loop body to preheader.
        // Insert before the terminator of the preheader block.
        // Sort body blocks by index so dependent instructions are hoisted in
        // definition order (block 2 before block 5, etc.), preventing
        // "undefined value" errors from non-deterministic HashSet iteration.
        let mut body_sorted: Vec<usize> = body.iter().copied().collect();
        body_sorted.sort();
        let mut hoisted: Vec<IrInstr> = Vec::new();
        for &bi in &body_sorted {
            let block = &mut func.blocks[bi];
            let mut remaining = Vec::new();
            for instr in block.instrs.drain(..) {
                if let Some(result) = instr.result() {
                    if invariant.contains(&result) {
                        hoisted.push(instr);
                        continue;
                    }
                }
                remaining.push(instr);
            }
            block.instrs = remaining;
        }

        // Insert hoisted instructions before the terminator of the preheader.
        let pre_block = &mut func.blocks[preheader];
        let term_pos = pre_block
            .instrs
            .iter()
            .position(|i| i.is_terminator())
            .unwrap_or(pre_block.instrs.len());
        for (i, instr) in hoisted.into_iter().enumerate() {
            pre_block.instrs.insert(term_pos + i, instr);
        }
    }
}

fn block_index(func: &IrFunction, bid: BlockId) -> Option<usize> {
    func.blocks.iter().position(|b| b.id == bid)
}

fn is_side_effecting_for_licm(instr: &IrInstr) -> bool {
    !matches!(
        instr,
        IrInstr::BinOp { .. }
            | IrInstr::ConstFloat { .. }
            | IrInstr::ConstInt { .. }
            | IrInstr::ConstBool { .. }
            | IrInstr::ConstStr { .. }
            | IrInstr::UnaryOp { .. }
            | IrInstr::TensorOp { .. }
            | IrInstr::Cast { .. }
            | IrInstr::MakeStruct { .. }
            | IrInstr::GetField { .. }
            | IrInstr::MakeTraitObject { .. }
            | IrInstr::DynCall { .. }
            | IrInstr::MakeVariant { .. }
            | IrInstr::ExtractVariantField { .. }
            | IrInstr::MakeTuple { .. }
            | IrInstr::GetElement { .. }
            | IrInstr::MakeSome { .. }
            | IrInstr::MakeNone { .. }
            | IrInstr::IsSome { .. }
            | IrInstr::MakeOk { .. }
            | IrInstr::MakeErr { .. }
            | IrInstr::IsOk { .. }
            | IrInstr::StrLen { .. }
            | IrInstr::StrConcat { .. }
            | IrInstr::StrContains { .. }
            | IrInstr::StrStartsWith { .. }
            | IrInstr::StrEndsWith { .. }
            | IrInstr::StrToUpper { .. }
            | IrInstr::StrToLower { .. }
            | IrInstr::StrTrim { .. }
            | IrInstr::StrRepeat { .. }
            | IrInstr::ValueToStr { .. }
            | IrInstr::ParseI64 { .. }
            | IrInstr::ParseF64 { .. }
            | IrInstr::StrIndex { .. }
            | IrInstr::StrSlice { .. }
            | IrInstr::StrFind { .. }
            | IrInstr::StrReplace { .. }
            | IrInstr::StrSplit { .. }
            | IrInstr::StrJoin { .. }
            | IrInstr::GetVariantTag { .. }
            | IrInstr::GradValue { .. }
            | IrInstr::GradTangent { .. }
            | IrInstr::TapeGrad { .. }
            | IrInstr::Sparsify { .. }
            | IrInstr::Densify { .. }
            | IrInstr::MakeGrad { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{compile, EmitKind};

    #[test]
    fn test_licm_pass_name() {
        let pass = LicmPass;
        assert_eq!(pass.name(), "licm");
    }

    #[test]
    fn fresh_loop_allocations_remain_per_iteration() {
        let src = r#"
            def main() -> i64 {
                val rows: list<list<i64>> = list()
                var i = 0
                while i < 2 {
                    val row: list<i64> = list()
                    val _ = list_push(row, i)
                    val _ = list_push(rows, row)
                    i = i + 1
                }
                list_get(list_get(rows, 0), 0) * 10 + list_get(list_get(rows, 1), 0)
            }
        "#;

        let out = compile(src, "test", EmitKind::Eval).expect("should compile and eval");
        assert_eq!(out.trim(), "1");
    }
}
