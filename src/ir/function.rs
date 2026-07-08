use std::collections::HashMap;

use crate::ir::block::{BlockId, IrBlock};
use crate::ir::instr::IrInstr;
use crate::ir::types::IrType;
use crate::ir::value::{ValueDef, ValueId};

/// Maps `(block_id, instr_index)` to the start byte offset of the source
/// statement that produced the instruction. Populated during lowering.
/// Used by the debugger for source-level breakpoints and step tracing.
#[derive(Debug, Default, Clone)]
pub struct SpanTable {
    /// Key: `(block_id.0, instr_index)`, value: byte offset into source text.
    pub(crate) entries: HashMap<(u32, usize), u32>,
}

impl SpanTable {
    /// Returns the source byte offset for the given instruction, if known.
    pub fn get(&self, block_id: u32, instr_idx: usize) -> Option<u32> {
        self.entries.get(&(block_id, instr_idx)).copied()
    }

    /// Returns `true` if any spans have been recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the number of recorded spans.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Uniquely identifies a function within an `IrModule`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FunctionId(pub u32);

/// A named, typed parameter of a function.
#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: IrType,
}

/// An attribute annotation with arguments, e.g. `@adaptive(learning_rate=0.01)`.
#[derive(Debug, Clone)]
pub struct IrAttribute {
    pub name: String,
    pub args: Vec<IrInstr>,  // Arguments as IR instructions (constants, etc.)
}

impl PartialEq<str> for IrAttribute {
    fn eq(&self, other: &str) -> bool {
        self.name == other
    }
}

impl PartialEq<String> for IrAttribute {
    fn eq(&self, other: &String) -> bool {
        self.name == *other
    }
}

impl PartialEq for IrAttribute {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
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
    /// Function attributes, e.g. `@adaptive(learning_rate=0.01)`.
    pub attrs: Vec<IrAttribute>,
    /// Source position table for the debugger: maps `(block_id, instr_idx)` to
    /// the byte offset of the statement that produced the instruction.
    pub span_table: SpanTable,
    /// Number of leading parameters that are lambda captures (0 for normal fns).
    /// Used by LLVM codegen to emit env-based capture extraction.
    pub capture_count: usize,
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

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::block::BlockId;
    use crate::ir::types::{DType, IrType};

    // -- SpanTable ------------------------------------------------------------

    #[test]
    fn span_table_empty() {
        let st = SpanTable::default();
        assert!(st.is_empty());
        assert_eq!(st.len(), 0);
    }

    #[test]
    fn span_table_insert_and_get() {
        let mut st = SpanTable::default();
        st.entries.insert((0, 0), 42);
        assert!(!st.is_empty());
        assert_eq!(st.len(), 1);
        assert_eq!(st.get(0, 0), Some(42));
    }

    #[test]
    fn span_table_get_missing() {
        let st = SpanTable::default();
        assert_eq!(st.get(0, 0), None);
    }

    #[test]
    fn span_table_multiple_entries() {
        let mut st = SpanTable::default();
        st.entries.insert((0, 0), 10);
        st.entries.insert((0, 1), 20);
        st.entries.insert((1, 0), 30);
        assert_eq!(st.len(), 3);
        assert_eq!(st.get(0, 0), Some(10));
        assert_eq!(st.get(0, 1), Some(20));
        assert_eq!(st.get(1, 0), Some(30));
    }

    // -- FunctionId -----------------------------------------------------------

    #[test]
    fn function_id_equality() {
        assert_eq!(FunctionId(0), FunctionId(0));
        assert_ne!(FunctionId(0), FunctionId(1));
    }

    #[test]
    fn function_id_ordering() {
        assert!(FunctionId(0) < FunctionId(1));
    }

    // -- IrFunction -----------------------------------------------------------

    fn make_test_func() -> IrFunction {
        use crate::ir::module::IrFunctionBuilder;
        let params = vec![Param {
            name: "x".into(),
            ty: IrType::Scalar(DType::I64),
        }];
        let mut builder = IrFunctionBuilder::new("test_fn", params, IrType::Scalar(DType::I64));
        let entry = builder.create_block(Some("entry"));
        builder.set_current_block(entry);
        let _param_v = builder.add_block_param(entry, Some("x"), IrType::Scalar(DType::I64));
        let const_v = builder.fresh_value();
        builder.push_instr(
            crate::ir::instr::IrInstr::ConstInt {
                result: const_v,
                value: 42,
                ty: IrType::Scalar(DType::I64),
            },
            Some(IrType::Scalar(DType::I64)),
        );
        builder.push_instr(
            crate::ir::instr::IrInstr::Return {
                values: vec![const_v],
            },
            None,
        );
        builder.build()
    }

    #[test]
    fn function_entry_block() {
        let f = make_test_func();
        assert_eq!(f.entry_block().id, BlockId(0));
    }

    #[test]
    fn function_block_by_id() {
        let f = make_test_func();
        assert!(f.block(BlockId(0)).is_some());
        assert!(f.block(BlockId(99)).is_none());
    }

    #[test]
    fn function_blocks_count() {
        let f = make_test_func();
        assert_eq!(f.blocks().len(), 1);
    }

    #[test]
    fn function_value_type() {
        let f = make_test_func();
        // The block param and const should have types
        let has_typed = f
            .value_types
            .values()
            .any(|t| matches!(t, IrType::Scalar(DType::I64)));
        assert!(has_typed);
    }

    #[test]
    fn function_fresh_value_increments() {
        let mut f = make_test_func();
        let before = f.next_value;
        let v = f.fresh_value();
        assert_eq!(v.0, before);
        assert_eq!(f.next_value, before + 1);
    }

    #[test]
    fn function_capture_count_default() {
        let f = make_test_func();
        assert_eq!(f.capture_count, 0);
    }
}
