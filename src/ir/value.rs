use crate::ir::types::IrType;

/// An opaque, index-based reference to an SSA value within a function.
///
/// Invariant: `ValueId(n)` is only valid within the `IrFunction` that produced
/// it. Do not store `ValueId`s across function boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(pub u32);

impl std::fmt::Display for ValueId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// A block parameter in SSA form.
///
/// IRIS uses block-parameter style SSA (as in MLIR/Swift SIL) rather than
/// explicit phi instructions. Entry-block parameters are function arguments.
#[derive(Debug, Clone)]
pub struct BlockParam {
    pub id: ValueId,
    pub ty: IrType,
    pub name: Option<String>,
}

/// The definition site of an SSA value.
/// Every `ValueId` in a function must have exactly one `ValueDef`.
#[derive(Debug, Clone)]
pub enum ValueDef {
    /// Defined as a block parameter (entry block params are function args).
    BlockParam { block: crate::ir::block::BlockId },
    /// Defined as the result of an instruction.
    InstrResult {
        block: crate::ir::block::BlockId,
        instr: crate::ir::instr::InstrId,
    },
}
