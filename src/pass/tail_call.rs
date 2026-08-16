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
use crate::ir::instr::IrInstr;
use crate::ir::module::IrModule;
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
        for fi in 0..module.functions.len() {
            eliminate_self_tail_calls(module, fi);
        }
        Ok(())
    }
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

fn eliminate_self_tail_calls(module: &mut IrModule, fn_idx: usize) {
    let fn_name = module.functions[fn_idx].name.clone();

    // 1. Find the tail-call sites: (block index, instr index, call arguments).
    let mut sites: Vec<(usize, usize, Vec<ValueId>)> = Vec::new();
    {
        let func = &module.functions[fn_idx];
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
                if *callee != fn_name {
                    continue;
                }
                // The call must be the last non-terminator instruction, and the
                // terminator must hand its result straight to a returning block.
                if ii + 2 != block.instrs.len() {
                    continue;
                }
                let IrInstr::Br { target, args: br_args } = &block.instrs[ii + 1] else {
                    continue;
                };
                let Some(slot) = br_args.iter().position(|v| v == result) else {
                    continue;
                };
                // Passing the result twice, or alongside other values, is not a
                // shape this pass reasons about.
                if br_args.len() != 1 {
                    continue;
                }
                if !block_just_returns_param(func, *target, slot) {
                    continue;
                }
                sites.push((bi, ii, args.clone()));
            }
        }
    }
    if sites.is_empty() {
        return;
    }

    // 2. Move the body into a fresh loop header, because the entry block cannot
    //    be a branch target.
    let _entry_id = module.functions[fn_idx].blocks[0].id;
    let entry_params: Vec<BlockParam> = module.functions[fn_idx].blocks[0].params.clone();

    let header_id = BlockId(
        module.functions[fn_idx]
            .blocks
            .iter()
            .map(|b| b.id.0)
            .max()
            .unwrap_or(0)
            + 1,
    );

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
    let header_pos = module.functions[fn_idx].blocks.len() - 1;
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
