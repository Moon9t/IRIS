use crate::ir::instr::IrInstr;
use crate::ir::value::{BlockParam, ValueId};

/// An opaque index identifying a basic block within an `IrFunction`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub u32);

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bb{}", self.0)
    }
}

/// A basic block in SSA form.
///
/// Invariants enforced by `IrFunctionBuilder::build()`:
/// 1. `instrs` is non-empty — at minimum a terminator must be present.
/// 2. Exactly one terminator exists and it is always the last element of `instrs`.
/// 3. `params` are considered defined before any instruction in this block.
/// 4. Each `ValueId` in `instrs` and `params` is unique within the function.
#[derive(Debug, Clone)]
pub struct IrBlock {
    pub id: BlockId,
    /// Block parameters model phi nodes (block-param SSA style).
    pub params: Vec<BlockParam>,
    /// Instructions in program order. Terminator is last.
    pub instrs: Vec<IrInstr>,
    /// Optional display name used by the pretty-printer.
    pub name: Option<String>,
}

impl IrBlock {
    pub fn new(id: BlockId, name: Option<String>) -> Self {
        Self {
            id,
            params: Vec::new(),
            instrs: Vec::new(),
            name,
        }
    }

    /// Returns the terminator instruction if the block is sealed.
    pub fn terminator(&self) -> Option<&IrInstr> {
        self.instrs.last().filter(|i| i.is_terminator())
    }

    /// A block is sealed when it ends with a terminator.
    pub fn is_sealed(&self) -> bool {
        self.terminator().is_some()
    }

    /// Iterates over all `ValueId`s used as operands across all instructions.
    pub fn all_operands(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.instrs.iter().flat_map(|i| i.operands())
    }

    /// Iterates over all `ValueId`s defined in this block (params + instr results).
    pub fn all_defs(&self) -> impl Iterator<Item = ValueId> + '_ {
        let param_ids = self.params.iter().map(|p| p.id);
        let result_ids = self.instrs.iter().filter_map(|i| i.result());
        param_ids.chain(result_ids)
    }
}
