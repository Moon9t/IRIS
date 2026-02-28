/// Phase 83: GC Annotation Pass
///
/// Inserts `Retain` after each heap-allocated value (list, map, option, result, channel, etc.)
/// is created, and `Release` before each `Return` terminator in the function.
///
/// In the interpreter, Retain/Release are no-ops.
/// In LLVM IR, they lower to `@iris_retain` / `@iris_release` calls.

use std::collections::HashSet;

use crate::error::PassError;
use crate::ir::block::BlockId;
use crate::ir::instr::IrInstr;
use crate::ir::module::IrModule;
use crate::ir::types::IrType;
use crate::ir::value::ValueId;
use crate::pass::Pass;

pub struct GcAnnotatePass;

impl Pass for GcAnnotatePass {
    fn name(&self) -> &'static str { "GcAnnotate" }

    fn run(&mut self, module: &mut IrModule) -> Result<(), PassError> {
        for func in &mut module.functions {
            // Collect heap-allocated value IDs and their types (from creation instrs).
            let mut heap_vals: Vec<(ValueId, IrType)> = Vec::new();
            for block in &func.blocks {
                for instr in &block.instrs {
                    if let Some(result) = instr.result() {
                        if let Some(ty) = func.value_types.get(&result) {
                            if is_heap_ty(ty) {
                                heap_vals.push((result, ty.clone()));
                            }
                        }
                    }
                }
            }
            // Avoid retaining the same value twice.
            let retained: HashSet<ValueId> = heap_vals.iter().map(|(v, _)| *v).collect();

            // For each block, after a heap-creating instruction, insert Retain.
            // For each block ending with Return, insert Release for all retained values before it.
            let block_ids: Vec<BlockId> = func.blocks.iter().map(|b| b.id).collect();

            for bid in &block_ids {
                let bidx = bid.0 as usize;

                // Collect positions of heap-creating instrs and Return positions.
                let mut inserts_after: Vec<(usize, IrInstr)> = Vec::new(); // (after idx, instr to insert)
                let mut return_pos: Option<usize> = None;

                for (i, instr) in func.blocks[bidx].instrs.iter().enumerate() {
                    if let Some(result) = instr.result() {
                        if retained.contains(&result) {
                            if let Some(ty) = func.value_types.get(&result) {
                                if is_heap_ty(ty) {
                                    inserts_after.push((i, IrInstr::Retain { ptr: result }));
                                }
                            }
                        }
                    }
                    if matches!(instr, IrInstr::Return { .. }) {
                        return_pos = Some(i);
                    }
                }

                // Insert Release before Return.
                if let Some(ret_idx) = return_pos {
                    // Build the new instruction list.
                    let mut new_instrs = Vec::new();
                    for (i, instr) in func.blocks[bidx].instrs.iter().enumerate() {
                        // Insert Retain after heap-creating instrs.
                        new_instrs.push(instr.clone());
                        if let Some((_, retain_instr)) = inserts_after.iter().find(|(idx, _)| *idx == i) {
                            new_instrs.push(retain_instr.clone());
                        }
                        // Insert Releases before Return.
                        if i + 1 == ret_idx {
                            for (val, ty) in &heap_vals {
                                new_instrs.push(IrInstr::Release { ptr: *val, ty: ty.clone() });
                            }
                        }
                    }
                    // Also handle the case where Return is at index 0.
                    if ret_idx == 0 && !heap_vals.is_empty() {
                        let mut new2 = Vec::new();
                        for (val, ty) in &heap_vals {
                            new2.push(IrInstr::Release { ptr: *val, ty: ty.clone() });
                        }
                        new2.extend(new_instrs);
                        new_instrs = new2;
                    }
                    func.blocks[bidx].instrs = new_instrs;
                } else {
                    // No return in this block — just insert Retains.
                    let mut new_instrs = Vec::new();
                    for (i, instr) in func.blocks[bidx].instrs.iter().enumerate() {
                        new_instrs.push(instr.clone());
                        if let Some((_, retain_instr)) = inserts_after.iter().find(|(idx, _)| *idx == i) {
                            new_instrs.push(retain_instr.clone());
                        }
                    }
                    func.blocks[bidx].instrs = new_instrs;
                }
            }
        }
        Ok(())
    }
}

fn is_heap_ty(ty: &IrType) -> bool {
    matches!(ty,
        IrType::List(_)
        | IrType::Map(_, _)
        | IrType::Option(_)
        | IrType::ResultType(_, _)
        | IrType::Chan(_)
        | IrType::Atomic(_)
        | IrType::Mutex(_)
        | IrType::Grad(_)
        | IrType::Sparse(_)
    )
}
