/// Phase 85: Hindley-Milner style type inference pass.
///
/// Resolves `IrType::Infer` placeholders left after lowering by building
/// equality constraints from the IR and solving them with union-find
/// unification.
///
/// Algorithm:
///   1. Assign a fresh type variable (slot) to every value whose type is
///      `IrType::Infer`.  Known types are stored as concrete slots.
///   2. Walk every instruction and emit equality constraints between operand
///      and result types derived from the instruction's typing rules.
///   3. Solve: union-find with path compression.  Concrete types dominate;
///      unifying two distinct concrete types is a type error.
///   4. Substitute: replace every `IrType::Infer` in `value_types` with the
///      inferred type.  Any slot still unresolved becomes `IrType::I64`
///      (default integer).
use std::collections::HashMap;

use crate::error::PassError;
use crate::ir::instr::IrInstr;
use crate::ir::module::IrModule;
use crate::ir::types::{DType, IrType};
use crate::ir::value::ValueId;
use crate::pass::Pass;

// ---------------------------------------------------------------------------
// Union-find node
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum Slot {
    /// Points to another slot (union-find parent).
    Link(usize),
    /// Root node: either a concrete type or still unknown.
    Root(Option<IrType>),
}

struct UnionFind {
    slots: Vec<Slot>,
}

impl UnionFind {
    fn new() -> Self { Self { slots: Vec::new() } }

    /// Allocate a new slot with an optional known concrete type.
    fn new_slot(&mut self, ty: Option<IrType>) -> usize {
        let id = self.slots.len();
        self.slots.push(Slot::Root(ty));
        id
    }

    /// Find the root of the slot, applying path compression.
    fn find(&mut self, mut id: usize) -> usize {
        loop {
            match self.slots[id].clone() {
                Slot::Link(parent) => {
                    // Path compression: point directly to grandparent.
                    if let Slot::Link(gp) = self.slots[parent].clone() {
                        self.slots[id] = Slot::Link(gp);
                        id = gp;
                    } else {
                        id = parent;
                    }
                }
                Slot::Root(_) => return id,
            }
        }
    }

    /// Return the concrete type at the root, if any.
    fn get_type(&mut self, id: usize) -> Option<IrType> {
        let root = self.find(id);
        if let Slot::Root(ty) = &self.slots[root] { ty.clone() } else { None }
    }

    /// Unify two slots. Concrete types must match; otherwise record an error.
    fn unify(&mut self, a: usize, b: usize, errors: &mut Vec<String>) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb { return; }
        let ta = self.get_type(ra);
        let tb = self.get_type(rb);
        match (ta, tb) {
            (None, None) => {
                // Both unknown: merge.
                self.slots[ra] = Slot::Link(rb);
            }
            (Some(t), None) => {
                // a is concrete, b is unknown: propagate a → b.
                self.slots[rb] = Slot::Root(Some(t));
                self.slots[ra] = Slot::Link(rb);
            }
            (None, Some(t)) => {
                // b is concrete, a is unknown: propagate b → a.
                self.slots[ra] = Slot::Root(Some(t));
                self.slots[rb] = Slot::Link(ra);
            }
            (Some(t1), Some(t2)) => {
                if t1 != t2 {
                    errors.push(format!("type mismatch: {:?} vs {:?}", t1, t2));
                }
                // Even on mismatch, keep one root to avoid further explosions.
                self.slots[ra] = Slot::Link(rb);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

pub struct HmTypeInferPass;

impl Pass for HmTypeInferPass {
    fn name(&self) -> &'static str { "hm-type-infer" }

    fn run(&mut self, module: &mut IrModule) -> Result<(), PassError> {
        let num_fns = module.functions.len();
        for fn_idx in 0..num_fns {
            infer_function(module, fn_idx)?;
        }
        Ok(())
    }
}

fn infer_function(module: &mut IrModule, fn_idx: usize) -> Result<(), PassError> {
    let mut uf = UnionFind::new();
    // Map from ValueId → slot index in union-find.
    let mut slots: HashMap<ValueId, usize> = HashMap::new();
    let mut errors: Vec<String> = Vec::new();

    // Pass 1: collect constraints by walking all instructions.
    let num_blocks = module.functions[fn_idx].blocks.len();
    for bi in 0..num_blocks {
        let num_instrs = module.functions[fn_idx].blocks[bi].instrs.len();
        for ii in 0..num_instrs {
            let instr = module.functions[fn_idx].blocks[bi].instrs[ii].clone();
            collect_constraints(&instr, &mut uf, &mut slots, &mut errors);
        }
    }

    // Pass 2: substitute resolved types back into value_types.
    let value_ids: Vec<ValueId> = module.functions[fn_idx].value_types.keys().cloned().collect();
    for vid in value_ids {
        if module.functions[fn_idx].value_types.get(&vid) == Some(&IrType::Infer) {
            if let Some(&s) = slots.get(&vid) {
                let resolved = uf.get_type(s).unwrap_or(IrType::Scalar(DType::I64));
                module.functions[fn_idx].value_types.insert(vid, resolved);
            }
        }
    }

    if !errors.is_empty() {
        return Err(PassError::TypeError {
            func: module.functions[fn_idx].name.clone(),
            detail: errors.join("; "),
        });
    }
    Ok(())
}

/// Emit equality constraints from a single instruction.
fn collect_constraints(
    instr: &IrInstr,
    uf: &mut UnionFind,
    slots: &mut HashMap<ValueId, usize>,
    errors: &mut Vec<String>,
) {
    // Helper to get-or-create a slot inline.
    let mut slot = |v: ValueId, known: Option<IrType>| -> usize {
        if let Some(&s) = slots.get(&v) {
            s
        } else {
            let s = uf.new_slot(known);
            slots.insert(v, s);
            s
        }
    };

    match instr {
        // BinOp: ty is the result type; lhs/rhs have the same operand type.
        IrInstr::BinOp { result, lhs, rhs, ty, .. } => {
            let sr = slot(*result, Some(ty.clone()));
            let sl = slot(*lhs, None);
            let srs = slot(*rhs, None);
            // Unify lhs and rhs (same numeric type).
            uf.unify(sl, srs, errors);
            // For non-Bool results (i.e., non-comparison ops), result type = operand type.
            if !matches!(ty, IrType::Scalar(DType::Bool)) {
                uf.unify(sr, sl, errors);
            }
        }
        IrInstr::UnaryOp { result, ty, operand, .. } => {
            let sr = slot(*result, Some(ty.clone()));
            let so = slot(*operand, None);
            uf.unify(sr, so, errors);
        }
        IrInstr::ConstInt { result, ty, .. } => { slot(*result, Some(ty.clone())); }
        IrInstr::ConstFloat { result, ty, .. } => { slot(*result, Some(ty.clone())); }
        IrInstr::ConstBool { result, .. } => { slot(*result, Some(IrType::Scalar(DType::Bool))); }
        IrInstr::ConstStr { result, .. } => { slot(*result, Some(IrType::Str)); }
        IrInstr::Cast { result, to_ty, .. } => { slot(*result, Some(to_ty.clone())); }
        // Return: each returned value should match the corresponding function return type.
        // (We don't have function return_ty here; leave for a separate pass.)
        IrInstr::Return { .. } => {}
        // Everything else: if there's a result with a known result_ty, record it.
        _ => {
            if let Some(r) = instr.result() {
                // Most instructions already have a concrete type stored in value_types;
            // this is a no-op if it's already known.
                slot(r, None);
            }
        }
    }
}
