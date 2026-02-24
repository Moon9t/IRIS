use std::collections::HashMap;

use crate::ir::block::{BlockId, IrBlock};
use crate::ir::types::IrType;
use crate::ir::value::{ValueDef, ValueId};

/// Uniquely identifies a function within an `IrModule`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FunctionId(pub u32);

/// A named, typed parameter of a function.
#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: IrType,
}

/// A compiled function in SSA form.
///
/// Internal representation uses flat `Vec`s indexed by `BlockId`. The entry
/// block is always `blocks[0]`; its block params are the function arguments.
///
/// Post-construction, the function is immutable to callers outside this crate.
/// Passes receive `&mut IrModule` but only mutate through the `pub(crate)` fields.
#[derive(Debug, Clone)]
pub struct IrFunction {
    pub id: FunctionId,
    pub name: String,
    pub params: Vec<Param>,
    pub return_ty: IrType,
    /// Flat list of blocks. `BlockId(n)` indexes `blocks[n]`.
    pub(crate) blocks: Vec<IrBlock>,
    /// Maps `ValueId` → its definition site. Populated during construction.
    pub(crate) value_defs: HashMap<ValueId, ValueDef>,
    /// Maps `ValueId` → its type. Populated during lowering.
    pub(crate) value_types: HashMap<ValueId, IrType>,
    /// Counter for allocating fresh `ValueId`s.
    pub(crate) next_value: u32,
}

impl IrFunction {
    /// Returns the entry block (always `BlockId(0)`).
    pub fn entry_block(&self) -> &IrBlock {
        &self.blocks[0]
    }

    pub fn block(&self, id: BlockId) -> Option<&IrBlock> {
        self.blocks.get(id.0 as usize)
    }

    pub fn blocks(&self) -> &[IrBlock] {
        &self.blocks
    }

    /// Returns the type of a value, if known.
    pub fn value_type(&self, v: ValueId) -> Option<&IrType> {
        self.value_types.get(&v)
    }

    /// Returns the definition site of a value.
    pub fn value_def(&self, v: ValueId) -> Option<&ValueDef> {
        self.value_defs.get(&v)
    }

    /// Allocates a fresh `ValueId`. Used by the builder only.
    pub(crate) fn fresh_value(&mut self) -> ValueId {
        let id = ValueId(self.next_value);
        self.next_value += 1;
        id
    }
}
