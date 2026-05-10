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

// Local helper: returns true if `ty` contains an unresolved `IrType::Infer`.
fn local_contains_infer(ty: &IrType) -> bool {
    match ty {
        IrType::Infer => true,
        IrType::Option(_) | IrType::ResultType(..) | IrType::Chan(_) | IrType::Atomic(_) | IrType::Mutex(_) => false,
        IrType::Scalar(_) | IrType::Str | IrType::Enum { .. } | IrType::Struct { .. } => false,
        IrType::Tensor { .. } => false,
        IrType::Tuple(elems) => elems.iter().any(local_contains_infer),
        IrType::Array { elem, .. } => local_contains_infer(elem),
        IrType::Grad(inner) | IrType::Sparse(inner) | IrType::List(inner) => local_contains_infer(inner),
        IrType::Map(k, v) => local_contains_infer(k) || local_contains_infer(v),
        IrType::Fn { params, ret } => params.iter().any(local_contains_infer) || local_contains_infer(ret),
    }
}

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
    fn new() -> Self {
        Self { slots: Vec::new() }
    }

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
        if let Slot::Root(ty) = &self.slots[root] {
            ty.clone()
        } else {
            None
        }
    }

    /// Unify two slots. Concrete types must match; otherwise record an error.
    fn unify(&mut self, a: usize, b: usize, errors: &mut Vec<String>) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
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
    fn name(&self) -> &'static str {
        "hm-type-infer"
    }

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
    // Track a union-find slot for each list value's element type.
    let mut list_elem_slots: HashMap<ValueId, usize> = HashMap::new();
    for bi in 0..num_blocks {
        let num_instrs = module.functions[fn_idx].blocks[bi].instrs.len();
        for ii in 0..num_instrs {
            let instr = module.functions[fn_idx].blocks[bi].instrs[ii].clone();
            collect_constraints(&instr, &mut uf, &mut slots, &mut errors, &mut list_elem_slots);
        }
    }

    // Pass 2: substitute resolved types back into value_types.
    let value_ids: Vec<ValueId> = module.functions[fn_idx]
        .value_types
        .keys()
        .cloned()
        .collect();
    for vid in value_ids {
        let current_ty = module.functions[fn_idx].value_types.get(&vid).cloned();
        if let Some(ty) = current_ty {
            // If the type contains an Infer somewhere (e.g., List(Infer)),
            // attempt to replace the inner Infer from collected slots.
            if local_contains_infer(&ty) {
                match ty {
                    IrType::List(_) => {
                        // If we tracked an element slot for this list, use it.
                        if let Some(&elem_slot) = list_elem_slots.get(&vid) {
                            let elem_ty = uf
                                .get_type(elem_slot)
                                .unwrap_or(IrType::Scalar(DType::I64));
                            let resolved = IrType::List(Box::new(elem_ty));
                            module.functions[fn_idx].value_types.insert(vid, resolved);
                        } else if let Some(&s) = slots.get(&vid) {
                            let resolved = uf.get_type(s).unwrap_or(IrType::Scalar(DType::I64));
                            module.functions[fn_idx].value_types.insert(vid, resolved);
                        }
                    }
                    IrType::Infer => {
                        if let Some(&s) = slots.get(&vid) {
                            let resolved = uf.get_type(s).unwrap_or(IrType::Scalar(DType::I64));
                            module.functions[fn_idx].value_types.insert(vid, resolved);
                        }
                    }
                    _ => {
                        // Other compound types with nested Infer are not expected
                        // here for now; skip and let the validator catch them.
                    }
                }
            }
        }
    }

    if !errors.is_empty() {
        // Append diagnostic info mapping slots -> resolved types to help debugging.
        let mut diag = String::new();
        diag.push_str("--- slot diagnostics ---\n");
        for (vid, &s) in slots.iter() {
            if let Some(ty) = uf.get_type(s) {
                diag.push_str(&format!("Value {:?} -> slot {} => {:?}\n", vid, s, ty));
            }
        }
        diag.push_str("--- list element slots ---\n");
        for (vid, &s) in list_elem_slots.iter() {
            if let Some(ty) = uf.get_type(s) {
                diag.push_str(&format!("List Value {:?} elem slot {} => {:?}\n", vid, s, ty));
            } else {
                diag.push_str(&format!("List Value {:?} elem slot {} => <unknown>\n", vid, s));
            }
        }
        diag.push_str("--- function blocks ---\n");
        diag.push_str(&format!("{:?}\n", module.functions[fn_idx].blocks));
        return Err(PassError::TypeError {
            func: module.functions[fn_idx].name.clone(),
            detail: format!("{}\n{}", errors.join("; "), diag),
        });
    }
    // If any `IrType::Infer` remains in the function value types, report
    // a detailed UnresolvedInfer error listing remaining unresolved ids.
    let mut unresolved: Vec<ValueId> = Vec::new();
    for (vid, ty) in module.functions[fn_idx].value_types.iter() {
        if matches!(ty, IrType::Infer) {
            unresolved.push(*vid);
        }
    }
    if !unresolved.is_empty() {
        let mut detail = String::new();
        detail.push_str("unresolved Infer for values:\n");
        for v in unresolved.iter() {
            detail.push_str(&format!(" - {:?}\n", v));
            if let Some(&s) = slots.get(v) {
                if let Some(ty) = uf.get_type(s) {
                    detail.push_str(&format!("    slot {} => {:?}\n", s, ty));
                }
            }
            if let Some(&es) = list_elem_slots.get(v) {
                if let Some(ty) = uf.get_type(es) {
                    detail.push_str(&format!("    elem slot {} => {:?}\n", es, ty));
                } else {
                    detail.push_str(&format!("    elem slot {} => <unknown>\n", es));
                }
            }
        }
        detail.push_str("--- function blocks ---\n");
        detail.push_str(&format!("{:?}\n", module.functions[fn_idx].blocks));
        return Err(PassError::TypeError {
            func: module.functions[fn_idx].name.clone(),
            detail,
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
    list_elem_slots: &mut HashMap<ValueId, usize>,
) {
    // Helper to get-or-create a slot for a value id.
    fn get_or_create_slot(
        uf: &mut UnionFind,
        slots: &mut HashMap<ValueId, usize>,
        v: ValueId,
        known: Option<IrType>,
    ) -> usize {
        if let Some(&s) = slots.get(&v) {
            s
        } else {
            let s = uf.new_slot(known);
            slots.insert(v, s);
            s
        }
    }

    // Local helper to unify with better diagnostics when both sides are concrete.
    fn try_unify(
        uf: &mut UnionFind,
        errors: &mut Vec<String>,
        a: usize,
        b: usize,
        ctx: &str,
    ) {
        let ta = uf.get_type(a);
        let tb = uf.get_type(b);
        if let (Some(ref ta2), Some(ref tb2)) = (ta.as_ref(), tb.as_ref()) {
            if ta2 != tb2 {
                errors.push(format!("type mismatch: {:?} vs {:?} -- {}", ta2, tb2, ctx));
                return;
            }
        }
        uf.unify(a, b, errors);
    }

    match instr {
        // BinOp: ty is the result type; lhs/rhs have the same operand type.
        IrInstr::BinOp {
            result,
            lhs,
            rhs,
            ty,
            ..
        } => {
            let sr = get_or_create_slot(uf, slots, *result, Some(ty.clone()));
            let sl = get_or_create_slot(uf, slots, *lhs, None);
            let srs = get_or_create_slot(uf, slots, *rhs, None);
            // Unify lhs and rhs (same numeric type).
            try_unify(uf, errors, sl, srs, &format!("BinOp lhs {:?} rhs {:?}", lhs, rhs));
            // For non-Bool results (i.e., non-comparison ops), result type = operand type.
            if !matches!(ty, IrType::Scalar(DType::Bool)) {
                try_unify(uf, errors, sr, sl, &format!("BinOp result {:?} lhs {:?}", result, lhs));
            }
        }
        IrInstr::UnaryOp {
            result,
            ty,
            operand,
            ..
        } => {
            let sr = get_or_create_slot(uf, slots, *result, Some(ty.clone()));
            let so = get_or_create_slot(uf, slots, *operand, None);
            try_unify(uf, errors, sr, so, &format!("UnaryOp result {:?} operand {:?}", result, operand));
        }
        IrInstr::ConstInt { result, ty, .. } => {
            let _ = get_or_create_slot(uf, slots, *result, Some(ty.clone()));
        }
        IrInstr::ConstFloat { result, ty, .. } => {
            let _ = get_or_create_slot(uf, slots, *result, Some(ty.clone()));
        }
        IrInstr::ConstBool { result, .. } => {
            let _ = get_or_create_slot(
                uf,
                slots,
                *result,
                Some(IrType::Scalar(DType::Bool)),
            );
        }
        IrInstr::ConstStr { result, .. } => {
            let _ = get_or_create_slot(uf, slots, *result, Some(IrType::Str));
        }
        IrInstr::ListNew { result, elem_ty } => {
            // Create or record a slot representing the element type for this list.
            let elem_slot = if matches!(elem_ty, IrType::Infer) {
                uf.new_slot(None)
            } else {
                uf.new_slot(Some(elem_ty.clone()))
            };
            list_elem_slots.insert(*result, elem_slot);
            // Leave the list value itself as unknown (slot registered below if needed).
            let _ = get_or_create_slot(uf, slots, *result, None);
        }
        IrInstr::ListPush { list, value } => {
            let s_val = get_or_create_slot(uf, slots, *value, None);
            // Ensure the list has an element slot registered.
            let elem_slot = *list_elem_slots.entry(*list).or_insert_with(|| uf.new_slot(None));
            // Unify the element slot with the pushed value's slot.
            try_unify(uf, errors, elem_slot, s_val, &format!("ListPush list {:?} value {:?}", list, value));
            let _ = get_or_create_slot(uf, slots, *list, None);
        }
        IrInstr::ListGet { result, list, .. } => {
            let s_res = get_or_create_slot(uf, slots, *result, None);
            let elem_slot = *list_elem_slots.entry(*list).or_insert_with(|| uf.new_slot(None));
            try_unify(uf, errors, s_res, elem_slot, &format!("ListGet result {:?} list {:?}", result, list));
        }
        IrInstr::Cast { result, to_ty, .. } => {
            let _ = get_or_create_slot(uf, slots, *result, Some(to_ty.clone()));
        }
        // Return: each returned value should match the corresponding function return type.
        // (We don't have function return_ty here; leave for a separate pass.)
        IrInstr::Return { .. } => {}
        // Everything else: if there's a result with a known result_ty, record it.
        _ => {
                if let Some(r) = instr.result() {
                // Most instructions already have a concrete type stored in value_types;
                // this is a no-op if it's already known.
                let _ = get_or_create_slot(uf, slots, r, None);
            }
        }
    }
}
