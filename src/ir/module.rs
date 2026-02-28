use std::collections::HashMap;

use crate::ir::block::{BlockId, IrBlock};
use crate::ir::function::{FunctionId, IrFunction, Param};
use crate::ir::instr::{InstrId, IrInstr};
use crate::ir::types::IrType;
use crate::ir::value::{BlockParam, ValueDef, ValueId};

/// The top-level IR container.
///
/// Invariants:
/// - Function names are unique within a module.
/// - `FunctionId(n)` always indexes `functions[n]`.
/// - Once a function is added via `add_function()`, it is immutable to external
///   callers. Passes may mutate through the `pub(crate)` fields.
/// An extern function declaration (C-linkage FFI).
#[derive(Debug, Clone)]
pub struct IrExternFn {
    pub name: String,
    pub param_types: Vec<IrType>,
    pub ret_ty: IrType,
}

#[derive(Debug, Default)]
pub struct IrModule {
    pub name: String,
    pub(crate) functions: Vec<IrFunction>,
    pub(crate) function_index: HashMap<String, FunctionId>,
    /// Struct type definitions: name → ordered list of (field_name, field_type).
    pub(crate) struct_defs: HashMap<String, Vec<(String, IrType)>>,
    /// Enum type definitions: name → (variant_names, variant_field_types).
    /// `variant_field_types[i]` is the list of payload types for variant `i`.
    pub(crate) enum_defs: HashMap<String, (Vec<String>, Vec<Vec<IrType>>)>,
    /// Type alias definitions: alias name → concrete IrType.
    pub(crate) type_aliases: HashMap<String, IrType>,
    /// Extern function declarations: name → signature.
    pub extern_fns: Vec<IrExternFn>,
}

impl IrModule {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            functions: Vec::new(),
            function_index: HashMap::new(),
            struct_defs: HashMap::new(),
            enum_defs: HashMap::new(),
            type_aliases: HashMap::new(),
            extern_fns: Vec::new(),
        }
    }

    /// Registers a struct definition. Returns `Err` if the name already exists.
    pub fn add_struct_def(
        &mut self,
        name: impl Into<String>,
        fields: Vec<(String, IrType)>,
    ) -> Result<(), String> {
        let name = name.into();
        if self.struct_defs.contains_key(&name) {
            return Err(format!("struct '{}' already defined", name));
        }
        self.struct_defs.insert(name, fields);
        Ok(())
    }

    /// Looks up a struct definition by name.
    pub fn struct_def(&self, name: &str) -> Option<&Vec<(String, IrType)>> {
        self.struct_defs.get(name)
    }

    /// Registers an enum definition. Returns `Err` if the name already exists.
    pub fn add_enum_def(
        &mut self,
        name: impl Into<String>,
        variants: Vec<String>,
        variant_fields: Vec<Vec<IrType>>,
    ) -> Result<(), String> {
        let name = name.into();
        if self.enum_defs.contains_key(&name) {
            return Err(format!("enum '{}' already defined", name));
        }
        self.enum_defs.insert(name, (variants, variant_fields));
        Ok(())
    }

    /// Looks up an enum definition by name (returns variant names).
    pub fn enum_def(&self, name: &str) -> Option<&Vec<String>> {
        self.enum_defs.get(name).map(|(variants, _)| variants)
    }

    /// Looks up the payload field types for each variant of an enum.
    pub fn enum_variant_fields(&self, name: &str) -> Option<&Vec<Vec<IrType>>> {
        self.enum_defs.get(name).map(|(_, fields)| fields)
    }

    /// Registers a type alias. Returns `Err` if the name already exists.
    pub fn add_type_alias(&mut self, name: impl Into<String>, ty: IrType) -> Result<(), String> {
        let name = name.into();
        if self.type_aliases.contains_key(&name) {
            return Err(format!("type alias '{}' already defined", name));
        }
        self.type_aliases.insert(name, ty);
        Ok(())
    }

    /// Looks up a type alias by name.
    pub fn type_alias(&self, name: &str) -> Option<&IrType> {
        self.type_aliases.get(name)
    }

    pub fn function(&self, id: FunctionId) -> Option<&IrFunction> {
        self.functions.get(id.0 as usize)
    }

    pub fn function_by_name(&self, name: &str) -> Option<&IrFunction> {
        let id = self.function_index.get(name)?;
        self.functions.get(id.0 as usize)
    }

    pub fn functions(&self) -> &[IrFunction] {
        &self.functions
    }

    /// Seals and registers a function built by `IrFunctionBuilder`.
    /// Returns `Err` if the name is already taken.
    pub fn add_function(&mut self, mut func: IrFunction) -> Result<FunctionId, String> {
        if self.function_index.contains_key(&func.name) {
            return Err(format!("function '{}' already defined", func.name));
        }
        let id = FunctionId(self.functions.len() as u32);
        func.id = id;
        self.function_index.insert(func.name.clone(), id);
        self.functions.push(func);
        Ok(id)
    }
}

/// Builder for constructing an `IrFunction` incrementally.
///
/// Call order:
/// 1. `create_block()` — allocate one or more blocks
/// 2. `add_block_param()` — add typed params to each block
/// 3. `set_current_block()` — point the cursor at a block
/// 4. `push_instr()` — emit instructions into the current block
/// 5. `build()` — consume the builder and return the completed `IrFunction`
///
/// `build()` panics in debug builds if any block lacks a terminator.
pub struct IrFunctionBuilder {
    func: IrFunction,
    current_block: Option<BlockId>,
}

impl IrFunctionBuilder {
    pub fn new(name: impl Into<String>, params: Vec<Param>, return_ty: IrType) -> Self {
        let func = IrFunction {
            id: FunctionId(0), // reassigned by IrModule::add_function
            name: name.into(),
            params,
            return_ty,
            blocks: Vec::new(),
            value_defs: HashMap::new(),
            value_types: HashMap::new(),
            next_value: 0,
        };
        Self {
            func,
            current_block: None,
        }
    }

    /// Creates a new block and returns its `BlockId`.
    pub fn create_block(&mut self, name: Option<&str>) -> BlockId {
        let id = BlockId(self.func.blocks.len() as u32);
        self.func
            .blocks
            .push(IrBlock::new(id, name.map(str::to_owned)));
        id
    }

    /// Adds a typed parameter to a block. Returns the `ValueId` of the new param.
    pub fn add_block_param(&mut self, block: BlockId, name: Option<&str>, ty: IrType) -> ValueId {
        let value_id = self.func.fresh_value();
        let param = BlockParam {
            id: value_id,
            ty: ty.clone(),
            name: name.map(str::to_owned),
        };
        self.func.blocks[block.0 as usize].params.push(param);
        self.func
            .value_defs
            .insert(value_id, ValueDef::BlockParam { block });
        self.func.value_types.insert(value_id, ty);
        value_id
    }

    /// Returns the current insertion block.
    pub fn current_block(&self) -> BlockId {
        self.current_block.expect("IrFunctionBuilder: no current block set")
    }

    /// Sets the current insertion block.
    pub fn set_current_block(&mut self, block: BlockId) {
        self.current_block = Some(block);
    }

    /// Appends an instruction to the current block.
    ///
    /// `result_ty` is the type of the instruction's result value, if any.
    /// It must be `Some` iff the instruction produces a result.
    ///
    /// Panics in debug builds if the current block is already sealed.
    pub fn push_instr(&mut self, instr: IrInstr, result_ty: Option<IrType>) -> Option<ValueId> {
        let block_id = self
            .current_block
            .expect("IrFunctionBuilder: no current block set before push_instr");

        let block = &self.func.blocks[block_id.0 as usize];
        debug_assert!(
            !block.is_sealed(),
            "push_instr called on already-sealed block {:?}",
            block_id
        );

        let result = instr.result();

        if let (Some(result_id), Some(ty)) = (result, result_ty) {
            let instr_idx = InstrId(self.func.blocks[block_id.0 as usize].instrs.len() as u32);
            self.func.value_defs.insert(
                result_id,
                ValueDef::InstrResult {
                    block: block_id,
                    instr: instr_idx,
                },
            );
            self.func.value_types.insert(result_id, ty);
        }

        self.func.blocks[block_id.0 as usize].instrs.push(instr);
        result
    }

    /// Returns true if the current block already ends with a terminator.
    pub fn is_current_block_terminated(&self) -> bool {
        if let Some(block_id) = self.current_block {
            self.func.blocks[block_id.0 as usize].is_sealed()
        } else {
            false
        }
    }

    /// Allocates a fresh `ValueId` without attaching it to any instruction.
    /// Used by the lowerer when pre-allocating result values.
    pub fn fresh_value(&mut self) -> ValueId {
        self.func.fresh_value()
    }

    /// Emits a `ConstStr` instruction and returns the result `ValueId`.
    pub fn emit_const_str(&mut self, value: String) -> ValueId {
        let result = self.func.fresh_value();
        self.push_instr(
            crate::ir::instr::IrInstr::ConstStr { result, value },
            Some(crate::ir::types::IrType::Str),
        );
        result
    }

    /// Emits a `StrLen` instruction and returns the result `ValueId`.
    pub fn emit_str_len(&mut self, operand: ValueId) -> ValueId {
        let result = self.func.fresh_value();
        let ty = crate::ir::types::IrType::Scalar(crate::ir::types::DType::I64);
        self.push_instr(
            crate::ir::instr::IrInstr::StrLen { result, operand },
            Some(ty),
        );
        result
    }

    /// Emits a `StrConcat` instruction and returns the result `ValueId`.
    pub fn emit_str_concat(&mut self, lhs: ValueId, rhs: ValueId) -> ValueId {
        let result = self.func.fresh_value();
        self.push_instr(
            crate::ir::instr::IrInstr::StrConcat { result, lhs, rhs },
            Some(crate::ir::types::IrType::Str),
        );
        result
    }

    /// Emits a `Print` instruction (no result).
    pub fn emit_print(&mut self, operand: ValueId) {
        self.push_instr(
            crate::ir::instr::IrInstr::Print { operand },
            None,
        );
    }

    /// Terminates any unsealed blocks with `Return { values: [] }`.
    /// Call this before `build()` if early-return paths may leave orphan blocks.
    pub fn seal_unterminated_blocks(&mut self) {
        use crate::ir::instr::IrInstr;
        let block_ids: Vec<crate::ir::block::BlockId> = self
            .func
            .blocks
            .iter()
            .map(|b| b.id)
            .collect();
        for bid in block_ids {
            if !self.func.blocks[bid.0 as usize].is_sealed() {
                self.func.blocks[bid.0 as usize]
                    .instrs
                    .push(IrInstr::Return { values: vec![] });
            }
        }
    }

    /// Consumes the builder and returns the completed `IrFunction`.
    ///
    /// Panics in debug builds if any block is not sealed (lacks a terminator).
    pub fn build(self) -> IrFunction {
        #[cfg(debug_assertions)]
        for block in &self.func.blocks {
            assert!(
                block.is_sealed(),
                "build() called with unsealed block {:?} ('{:?}')",
                block.id,
                block.name
            );
        }
        self.func
    }
}
