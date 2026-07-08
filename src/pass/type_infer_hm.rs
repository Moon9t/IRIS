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
        IrType::Option(inner) => local_contains_infer(inner),
        IrType::ResultType(ok, err) => local_contains_infer(ok) || local_contains_infer(err),
        IrType::Chan(inner) => local_contains_infer(inner),
        IrType::Atomic(inner) => local_contains_infer(inner),
        IrType::Mutex(inner) => local_contains_infer(inner),
        IrType::Scalar(_) | IrType::Str | IrType::Enum { .. } | IrType::Struct { .. } => false,
        IrType::Tensor { .. } => false,
        IrType::Tuple(elems) => elems.iter().any(local_contains_infer),
        IrType::Array { elem, .. } => local_contains_infer(elem),
        IrType::Grad(inner) | IrType::Sparse(inner) | IrType::List(inner) => {
            local_contains_infer(inner)
        }
        IrType::Map(k, v) => local_contains_infer(k) || local_contains_infer(v),
        IrType::Fn { params, ret } => {
            params.iter().any(local_contains_infer) || local_contains_infer(ret)
        }
    }
}

// Replace unresolved Infer nodes with a conservative scalar default.
fn default_infer(ty: &IrType) -> IrType {
    match ty {
        IrType::Infer => IrType::Scalar(DType::I64),
        IrType::Option(inner) => IrType::Option(Box::new(default_infer(inner))),
        IrType::ResultType(ok, err) => {
            IrType::ResultType(Box::new(default_infer(ok)), Box::new(default_infer(err)))
        }
        IrType::Chan(inner) => IrType::Chan(Box::new(default_infer(inner))),
        IrType::Atomic(inner) => IrType::Atomic(Box::new(default_infer(inner))),
        IrType::Mutex(inner) => IrType::Mutex(Box::new(default_infer(inner))),
        IrType::Tuple(elems) => IrType::Tuple(elems.iter().map(default_infer).collect()),
        IrType::Array { elem, len } => IrType::Array {
            elem: Box::new(default_infer(elem)),
            len: *len,
        },
        IrType::Grad(inner) => IrType::Grad(Box::new(default_infer(inner))),
        IrType::Sparse(inner) => IrType::Sparse(Box::new(default_infer(inner))),
        IrType::List(inner) => IrType::List(Box::new(default_infer(inner))),
        IrType::Map(k, v) => IrType::Map(Box::new(default_infer(k)), Box::new(default_infer(v))),
        IrType::Fn { params, ret } => IrType::Fn {
            params: params.iter().map(default_infer).collect(),
            ret: Box::new(default_infer(ret)),
        },
        other => other.clone(),
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
                let merged = structural_unify(&t1, &t2, errors);
                // Even on mismatch, keep one root to avoid further explosions.
                self.slots[ra] = Slot::Root(Some(merged));
                self.slots[rb] = Slot::Link(ra);
            }
        }
    }
}

fn structural_unify(t1: &IrType, t2: &IrType, errors: &mut Vec<String>) -> IrType {
    match (t1, t2) {
        (IrType::ResultType(ok1, err1), IrType::ResultType(ok2, err2)) => {
            let merged_ok = structural_unify(ok1, ok2, errors);
            let merged_err = structural_unify(err1, err2, errors);
            IrType::ResultType(Box::new(merged_ok), Box::new(merged_err))
        }
        (IrType::Option(inner1), IrType::Option(inner2)) => {
            let merged_inner = structural_unify(inner1, inner2, errors);
            IrType::Option(Box::new(merged_inner))
        }
        (IrType::Tuple(elems1), IrType::Tuple(elems2)) => {
            if elems1.len() != elems2.len() {
                errors.push(format!("tuple length mismatch: {:?} vs {:?}", t1, t2));
                t1.clone()
            } else {
                let merged = elems1.iter().zip(elems2.iter()).map(|(e1, e2)| structural_unify(e1, e2, errors)).collect();
                IrType::Tuple(merged)
            }
        }
        (IrType::List(inner1), IrType::List(inner2)) => {
            let merged = structural_unify(inner1, inner2, errors);
            IrType::List(Box::new(merged))
        }
        (IrType::Map(k1, v1), IrType::Map(k2, v2)) => {
            let mk = structural_unify(k1, k2, errors);
            let mv = structural_unify(v1, v2, errors);
            IrType::Map(Box::new(mk), Box::new(mv))
        }
        (IrType::Fn { params: p1, ret: r1 }, IrType::Fn { params: p2, ret: r2 }) => {
            if p1.len() != p2.len() {
                errors.push(format!("fn param length mismatch: {:?} vs {:?}", t1, t2));
                t1.clone()
            } else {
                let mp = p1.iter().zip(p2.iter()).map(|(e1, e2)| structural_unify(e1, e2, errors)).collect();
                let mr = structural_unify(r1, r2, errors);
                IrType::Fn { params: mp, ret: Box::new(mr) }
            }
        }
        (IrType::Infer, other) | (other, IrType::Infer) => other.clone(),
        (a, b) => {
            if a != b {
                errors.push(format!("type mismatch: {:?} vs {:?}", a, b));
            }
            a.clone()
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
    let initial_return_ty = module.functions[fn_idx].return_ty.clone();
    let ret_slot = uf.new_slot(if local_contains_infer(&initial_return_ty) {
        None
    } else {
        Some(initial_return_ty.clone())
    });

    // Pre-initialize slots for all ValueIds that have a type in value_types.
    for (&vid, ty) in module.functions[fn_idx].value_types.iter() {
        let known = if local_contains_infer(ty) {
            None
        } else {
            Some(ty.clone())
        };
        let s = uf.new_slot(known);
        slots.insert(vid, s);
    }

    // Pass 1: collect constraints by walking all instructions.
    let num_blocks = module.functions[fn_idx].blocks.len();
    // Track a union-find slot for each list value's element type.
    let mut list_elem_slots: HashMap<ValueId, usize> = HashMap::new();
    // Track number of captured values for closures (MakeClosure result -> captures.len()).
    // Map from MakeClosure result -> captured ValueIds
    let mut closure_captures: HashMap<ValueId, Vec<ValueId>> = HashMap::new();
    for bi in 0..num_blocks {
        let num_instrs = module.functions[fn_idx].blocks[bi].instrs.len();
        for ii in 0..num_instrs {
            let instr = module.functions[fn_idx].blocks[bi].instrs[ii].clone();
            collect_constraints(
                &instr,
                &mut uf,
                &mut slots,
                &mut errors,
                &mut list_elem_slots,
                &mut closure_captures,
                ret_slot,
            );
        }

        // Unify block arguments with target block parameters on branches
        if let Some(terminator) = module.functions[fn_idx].blocks[bi].terminator().cloned() {
            match terminator {
                IrInstr::Br { target, args } => {
                    if let Some(target_block) = module.functions[fn_idx].block(target) {
                        for (arg, param) in args.iter().zip(target_block.params.iter()) {
                            let s_arg = get_or_create_slot(&mut uf, &mut slots, *arg, None);
                            let s_param = get_or_create_slot(&mut uf, &mut slots, param.id, None);
                            try_unify(
                                &mut uf,
                                &mut errors,
                                s_arg,
                                s_param,
                                &format!("Br arg {:?} to param {:?}", arg, param.id),
                            );
                        }
                    }
                }
                IrInstr::CondBr {
                    then_block,
                    then_args,
                    else_block,
                    else_args,
                    ..
                } => {
                    if let Some(then_b) = module.functions[fn_idx].block(then_block) {
                        for (arg, param) in then_args.iter().zip(then_b.params.iter()) {
                            let s_arg = get_or_create_slot(&mut uf, &mut slots, *arg, None);
                            let s_param = get_or_create_slot(&mut uf, &mut slots, param.id, None);
                            try_unify(
                                &mut uf,
                                &mut errors,
                                s_arg,
                                s_param,
                                &format!("CondBr then arg {:?} to param {:?}", arg, param.id),
                            );
                        }
                    }
                    if let Some(else_b) = module.functions[fn_idx].block(else_block) {
                        for (arg, param) in else_args.iter().zip(else_b.params.iter()) {
                            let s_arg = get_or_create_slot(&mut uf, &mut slots, *arg, None);
                            let s_param = get_or_create_slot(&mut uf, &mut slots, param.id, None);
                            try_unify(
                                &mut uf,
                                &mut errors,
                                s_arg,
                                s_param,
                                &format!("CondBr else arg {:?} to param {:?}", arg, param.id),
                            );
                        }
                    }
                }
                _ => {}
            }
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
                            let elem_ty =
                                uf.get_type(elem_slot).unwrap_or(IrType::Scalar(DType::I64));
                            let resolved = IrType::List(Box::new(elem_ty));
                            module.functions[fn_idx].value_types.insert(vid, resolved);
                        } else if let Some(&s) = slots.get(&vid) {
                            let resolved = uf.get_type(s).unwrap_or(IrType::Scalar(DType::I64));
                            module.functions[fn_idx].value_types.insert(vid, resolved);
                        } else {
                            module.functions[fn_idx]
                                .value_types
                                .insert(vid, IrType::List(Box::new(IrType::Scalar(DType::I64))));
                        }
                    }
                    IrType::Fn { params, ret } => {
                        // Try to substitute nested Infer inside lifted function
                        // signatures using captured values recorded earlier.
                        if let Some(captures) = closure_captures.get(&vid) {
                            let mut new_params: Vec<IrType> = Vec::new();
                            for (i, p) in params.iter().enumerate() {
                                let mut p_new = p.clone();
                                // If this param corresponds to a captured value,
                                // try to replace List(Infer) or Infer with the
                                // captured value's concrete type.
                                if i < captures.len() {
                                    let cap_vid = captures[i];
                                    if let Some(&elem_slot) = list_elem_slots.get(&cap_vid) {
                                        let cap_elem_ty = uf
                                            .get_type(elem_slot)
                                            .unwrap_or(IrType::Scalar(DType::I64));
                                        if matches!(p_new, IrType::List(_)) {
                                            p_new = IrType::List(Box::new(cap_elem_ty));
                                        }
                                    } else if let Some(&s) = slots.get(&cap_vid) {
                                        if let Some(cap_ty) = uf.get_type(s) {
                                            if matches!(p_new, IrType::Infer) {
                                                p_new = cap_ty;
                                            }
                                        }
                                    }
                                }
                                new_params.push(p_new);
                            }
                            let new_ret = if local_contains_infer(&ret) {
                                // If return contains infer, try to leave it as-is
                                // (other passes or constraints may resolve it).
                                *ret.clone()
                            } else {
                                *ret.clone()
                            };
                            let resolved = IrType::Fn {
                                params: new_params,
                                ret: Box::new(new_ret),
                            };
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
                        let resolved = if let Some(&s) = slots.get(&vid) {
                            uf.get_type(s).unwrap_or_else(|| default_infer(&ty))
                        } else {
                            default_infer(&ty)
                        };
                        module.functions[fn_idx]
                            .value_types
                            .insert(vid, resolved);
                    }
                }
            }
        }
    }

    // Resolve function return type from accumulated return constraints.
    let inferred_ret = uf
        .get_type(ret_slot)
        .unwrap_or_else(|| default_infer(&module.functions[fn_idx].return_ty));
    module.functions[fn_idx].return_ty = if local_contains_infer(&inferred_ret) {
        default_infer(&inferred_ret)
    } else {
        inferred_ret
    };

    // Final sweep: normalize any remaining nested Infer to defaults so
    // ValidatePass does not fail on partially unresolved compounds.
    let value_ids: Vec<ValueId> = module.functions[fn_idx]
        .value_types
        .keys()
        .cloned()
        .collect();
    for vid in value_ids {
        if let Some(ty) = module.functions[fn_idx].value_types.get(&vid).cloned() {
            if local_contains_infer(&ty) {
                let new_ty = default_infer(&ty);
                module.functions[fn_idx]
                    .value_types
                    .insert(vid, new_ty);
            }
        }
    }

    // Sweep block parameters to replace any nested IrType::Infer in block parameter type fields
    // with their fully resolved types from module.functions[fn_idx].value_types.
    let value_types = module.functions[fn_idx].value_types.clone();
    let num_blocks = module.functions[fn_idx].blocks.len();
    for bi in 0..num_blocks {
        let num_params = module.functions[fn_idx].blocks[bi].params.len();
        for pi in 0..num_params {
            let param = &mut module.functions[fn_idx].blocks[bi].params[pi];
            if let Some(resolved) = value_types.get(&param.id) {
                param.ty = resolved.clone();
            }
        }
    }

    // Sweep func.params to match entry block param resolved types.
    let bp_ids: Vec<ValueId> = module.functions[fn_idx]
        .blocks
        .first()
        .map(|b| b.params.iter().map(|p| p.id).collect())
        .unwrap_or_default();
    for (pi, param) in module.functions[fn_idx].params.iter_mut().enumerate() {
        if pi < bp_ids.len() {
            if let Some(resolved) = value_types.get(&bp_ids[pi]) {
                param.ty = resolved.clone();
            }
        }
    }

    // Sweep instructions to replace any nested IrType::Infer in instruction type fields
    // with their fully resolved types from module.functions[fn_idx].value_types.
    for bi in 0..num_blocks {
        let num_instrs = module.functions[fn_idx].blocks[bi].instrs.len();
        for ii in 0..num_instrs {
            let instr = &mut module.functions[fn_idx].blocks[bi].instrs[ii];
            match instr {
                IrInstr::BinOp { result, op, ty, .. } => {
                    let is_cmp = matches!(
                        op,
                        crate::ir::instr::BinOp::CmpEq
                            | crate::ir::instr::BinOp::CmpNe
                            | crate::ir::instr::BinOp::CmpLt
                            | crate::ir::instr::BinOp::CmpLe
                            | crate::ir::instr::BinOp::CmpGt
                            | crate::ir::instr::BinOp::CmpGe
                    );
                    if is_cmp {
                        *ty = IrType::Scalar(DType::Bool);
                    } else if let Some(resolved) = value_types.get(result) {
                        *ty = resolved.clone();
                    }
                }
                IrInstr::UnaryOp { result, ty, .. } => {
                    if let Some(resolved) = value_types.get(result) {
                        *ty = resolved.clone();
                    }
                }
                IrInstr::Cast { result, to_ty, .. } => {
                    if let Some(resolved) = value_types.get(result) {
                        *to_ty = resolved.clone();
                    }
                }
                IrInstr::ListGet {
                    result, elem_ty, ..
                } => {
                    if let Some(resolved) = value_types.get(result) {
                        *elem_ty = resolved.clone();
                    }
                }
                IrInstr::ListPop {
                    result, elem_ty, ..
                } => {
                    if let Some(resolved) = value_types.get(result) {
                        *elem_ty = resolved.clone();
                    }
                }
                IrInstr::ArrayLoad {
                    result, elem_ty, ..
                } => {
                    if let Some(resolved) = value_types.get(result) {
                        *elem_ty = resolved.clone();
                    }
                }
                IrInstr::MapGet { result, val_ty, .. } => {
                    if let Some(resolved) = value_types.get(result) {
                        *val_ty = resolved.clone();
                    }
                }
                IrInstr::OptionUnwrap {
                    result, result_ty, ..
                } => {
                    if let Some(resolved) = value_types.get(result) {
                        *result_ty = resolved.clone();
                    }
                }
                IrInstr::ResultUnwrap {
                    result, result_ty, ..
                } => {
                    if let Some(resolved) = value_types.get(result) {
                        *result_ty = resolved.clone();
                    }
                }
                IrInstr::ResultUnwrapErr {
                    result, result_ty, ..
                } => {
                    if let Some(resolved) = value_types.get(result) {
                        *result_ty = resolved.clone();
                    }
                }
                IrInstr::AtomicLoad {
                    result, result_ty, ..
                } => {
                    if let Some(resolved) = value_types.get(result) {
                        *result_ty = resolved.clone();
                    }
                }
                IrInstr::AtomicAdd {
                    result, result_ty, ..
                } => {
                    if let Some(resolved) = value_types.get(result) {
                        *result_ty = resolved.clone();
                    }
                }
                IrInstr::GetField {
                    result, result_ty, ..
                } => {
                    if let Some(resolved) = value_types.get(result) {
                        *result_ty = resolved.clone();
                    }
                }
                IrInstr::GetElement {
                    result, result_ty, ..
                } => {
                    if let Some(resolved) = value_types.get(result) {
                        *result_ty = resolved.clone();
                    }
                }
                IrInstr::MakeStruct {
                    result, result_ty, ..
                } => {
                    if let Some(resolved) = value_types.get(result) {
                        *result_ty = resolved.clone();
                    }
                }
                IrInstr::MakeTuple {
                    result, result_ty, ..
                } => {
                    if let Some(resolved) = value_types.get(result) {
                        *result_ty = resolved.clone();
                    }
                }
                IrInstr::MakeClosure {
                    result, result_ty, ..
                } => {
                    if let Some(resolved) = value_types.get(result) {
                        *result_ty = resolved.clone();
                    }
                }
                IrInstr::MakeSome {
                    result, result_ty, ..
                } => {
                    if let Some(resolved) = value_types.get(result) {
                        *result_ty = resolved.clone();
                    }
                }
                IrInstr::MakeNone {
                    result, result_ty, ..
                } => {
                    if let Some(resolved) = value_types.get(result) {
                        *result_ty = resolved.clone();
                    }
                }
                IrInstr::MakeOk {
                    result, result_ty, ..
                } => {
                    if let Some(resolved) = value_types.get(result) {
                        *result_ty = resolved.clone();
                    }
                }
                IrInstr::MakeErr {
                    result, result_ty, ..
                } => {
                    if let Some(resolved) = value_types.get(result) {
                        *result_ty = resolved.clone();
                    }
                }
                IrInstr::Call {
                    result: Some(r),
                    result_ty,
                    ..
                } => {
                    if let Some(resolved) = value_types.get(r) {
                        *result_ty = Some(resolved.clone());
                    }
                }
                IrInstr::CallClosure {
                    result: Some(r),
                    result_ty,
                    ..
                } => {
                    if let Some(resolved) = value_types.get(r) {
                        *result_ty = resolved.clone();
                    }
                }
                IrInstr::BuiltinCall {
                    result, result_ty, ..
                } => {
                    if let Some(resolved) = value_types.get(result) {
                        *result_ty = resolved.clone();
                    }
                }
                IrInstr::ListNew { result, elem_ty } => {
                    if let Some(IrType::List(inner)) = value_types.get(result) {
                        *elem_ty = (**inner).clone();
                    }
                }
                IrInstr::AllocArray {
                    result, elem_ty, ..
                } => {
                    if let Some(IrType::Array { elem, .. }) = value_types.get(result) {
                        *elem_ty = (**elem).clone();
                    }
                }
                IrInstr::MapNew {
                    result,
                    key_ty,
                    val_ty,
                } => {
                    if let Some(IrType::Map(k, v)) = value_types.get(result) {
                        *key_ty = (**k).clone();
                        *val_ty = (**v).clone();
                    }
                }
                IrInstr::Release { ptr, ty } => {
                    if let Some(resolved) = value_types.get(ptr) {
                        *ty = resolved.clone();
                    }
                }
                _ => {}
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
                diag.push_str(&format!(
                    "List Value {:?} elem slot {} => {:?}\n",
                    vid, s, ty
                ));
            } else {
                diag.push_str(&format!(
                    "List Value {:?} elem slot {} => <unknown>\n",
                    vid, s
                ));
            }
        }
        diag.push_str("--- function blocks ---\n");
        diag.push_str(&format!("{:?}\n", module.functions[fn_idx].blocks));
        diag.push_str("--- value types ---\n");
        for (vid, ty) in module.functions[fn_idx].value_types.iter() {
            diag.push_str(&format!("Value {:?} => {:?}\n", vid, ty));
        }
        diag.push_str("--- closure captures ---\n");
        for (vid, caps) in closure_captures.iter() {
            diag.push_str(&format!("Closure {:?} captures {:?}\n", vid, caps));
        }
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
        detail.push_str("--- value types ---\n");
        for (vid, ty) in module.functions[fn_idx].value_types.iter() {
            detail.push_str(&format!("Value {:?} => {:?}\n", vid, ty));
        }
        detail.push_str("--- closure captures ---\n");
        for (vid, caps) in closure_captures.iter() {
            detail.push_str(&format!("Closure {:?} captures {:?}\n", vid, caps));
        }
        return Err(PassError::TypeError {
            func: module.functions[fn_idx].name.clone(),
            detail,
        });
    }
    Ok(())
}

/// Helper to get-or-create a slot for a value id.
fn get_or_create_slot(
    uf: &mut UnionFind,
    slots: &mut HashMap<ValueId, usize>,
    v: ValueId,
    known: Option<IrType>,
) -> usize {
    if let Some(&s) = slots.get(&v) {
        s
    } else {
        // Treat `Infer` as unknown: don't register it as a concrete type.
        let concrete = match known {
            Some(t) if local_contains_infer(&t) => None,
            other => other,
        };
        let s = uf.new_slot(concrete);
        slots.insert(v, s);
        s
    }
}

/// Helper to unify with better diagnostics when both sides are concrete.
fn try_unify(uf: &mut UnionFind, errors: &mut Vec<String>, a: usize, b: usize, ctx: &str) {
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

/// Emit equality constraints from a single instruction.
fn collect_constraints(
    instr: &IrInstr,
    uf: &mut UnionFind,
    slots: &mut HashMap<ValueId, usize>,
    errors: &mut Vec<String>,
    list_elem_slots: &mut HashMap<ValueId, usize>,
    closure_captures: &mut HashMap<ValueId, Vec<ValueId>>,
    ret_slot: usize,
) {
    match instr {
        // BinOp: ty is the operand type; lhs/rhs have the same operand type.
        IrInstr::BinOp {
            result,
            op,
            lhs,
            rhs,
            ty,
        } => {
            let is_cmp = matches!(
                op,
                crate::ir::instr::BinOp::CmpEq
                    | crate::ir::instr::BinOp::CmpNe
                    | crate::ir::instr::BinOp::CmpLt
                    | crate::ir::instr::BinOp::CmpLe
                    | crate::ir::instr::BinOp::CmpGt
                    | crate::ir::instr::BinOp::CmpGe
            );
            let res_ty = if is_cmp {
                IrType::Scalar(DType::Bool)
            } else {
                ty.clone()
            };
            let sr = get_or_create_slot(uf, slots, *result, Some(res_ty));
            let sl = get_or_create_slot(uf, slots, *lhs, None);
            let srs = get_or_create_slot(uf, slots, *rhs, None);
            // Unify lhs and rhs (same operand type).
            try_unify(
                uf,
                errors,
                sl,
                srs,
                &format!("BinOp lhs {:?} rhs {:?}", lhs, rhs),
            );
            // For non-comparison ops, result type = operand type.
            if !is_cmp {
                try_unify(
                    uf,
                    errors,
                    sr,
                    sl,
                    &format!("BinOp result {:?} lhs {:?}", result, lhs),
                );
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
            try_unify(
                uf,
                errors,
                sr,
                so,
                &format!("UnaryOp result {:?} operand {:?}", result, operand),
            );
        }
        IrInstr::ConstInt { result, ty, .. } => {
            let _ = get_or_create_slot(uf, slots, *result, Some(ty.clone()));
        }
        IrInstr::ConstFloat { result, ty, .. } => {
            let _ = get_or_create_slot(uf, slots, *result, Some(ty.clone()));
        }
        IrInstr::ConstBool { result, .. } => {
            let _ = get_or_create_slot(uf, slots, *result, Some(IrType::Scalar(DType::Bool)));
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
            let elem_slot = *list_elem_slots
                .entry(*list)
                .or_insert_with(|| uf.new_slot(None));
            // Unify the element slot with the pushed value's slot.
            try_unify(
                uf,
                errors,
                elem_slot,
                s_val,
                &format!("ListPush list {:?} value {:?}", list, value),
            );
            let _ = get_or_create_slot(uf, slots, *list, None);
        }
        IrInstr::ListGet { result, list, .. } => {
            let s_res = get_or_create_slot(uf, slots, *result, None);
            let elem_slot = *list_elem_slots
                .entry(*list)
                .or_insert_with(|| uf.new_slot(None));
            try_unify(
                uf,
                errors,
                s_res,
                elem_slot,
                &format!("ListGet result {:?} list {:?}", result, list),
            );
        }
        IrInstr::MakeClosure {
            result,
            result_ty,
            captures,
            ..
        } => {
            // Record the closure value's concrete function type (if any).
            let _ = get_or_create_slot(uf, slots, *result, Some(result_ty.clone()));
            // Remember the captured value ids so substitution can resolve
            // nested Infer types within the lifted function signature.
            closure_captures.insert(*result, captures.clone());
        }
        IrInstr::CallClosure {
            result,
            closure,
            args,
            result_ty,
            ..
        } => {
            // If the call produces a result, register its expected type.
            if let Some(rid) = result {
                let _ = get_or_create_slot(uf, slots, *rid, Some(result_ty.clone()));
            }
            // Try to inspect the closure's known function type and unify
            // argument slots with parameter types, and unify return type.
            if let Some(&cslot) = slots.get(closure) {
                if let Some(IrType::Fn { params, ret }) = uf.get_type(cslot) {
                    // Unify each argument with the corresponding param type.
                    // Skip leading captured params (if any): the MakeClosure
                    // encodes captures as leading parameters in the lifted
                    // function type, but CallClosure supplies only the user
                    // visible args.
                    let skip = closure_captures.get(closure).map(|v| v.len()).unwrap_or(0);
                    for (i, arg) in args.iter().enumerate() {
                        let param_idx = skip + i;
                        if param_idx < params.len() {
                            let arg_slot = get_or_create_slot(uf, slots, *arg, None);
                            let p_ty = params[param_idx].clone();
                            if !matches!(p_ty, IrType::Infer) {
                                let p_slot = uf.new_slot(Some(p_ty));
                                try_unify(
                                    uf,
                                    errors,
                                    arg_slot,
                                    p_slot,
                                    &format!("CallClosure arg {:?} param {}", arg, param_idx),
                                );
                            }
                        }
                    }
                    // Unify result with function return type.
                    if let Some(rid) = result {
                        let res_slot = get_or_create_slot(uf, slots, *rid, None);
                        let ret_ty = *ret.clone();
                        if !matches!(ret_ty, IrType::Infer) {
                            let rslot = uf.new_slot(Some(ret_ty));
                            try_unify(
                                uf,
                                errors,
                                res_slot,
                                rslot,
                                &format!("CallClosure result {:?}", rid),
                            );
                        }
                    }
                }
            }
        }
        IrInstr::ListSlice { result, list, .. } => {
            // The resulting slice has the same element type as the source list.
            let elem_slot = *list_elem_slots
                .entry(*list)
                .or_insert_with(|| uf.new_slot(None));
            // Propagate the element slot to the result list value.
            list_elem_slots.insert(*result, elem_slot);
            let _ = get_or_create_slot(uf, slots, *result, None);
        }
        IrInstr::ListConcat { result, lhs, rhs } => {
            // Concatenation produces a list whose element type must unify
            // with both operands' element types.
            let lhs_slot = *list_elem_slots
                .entry(*lhs)
                .or_insert_with(|| uf.new_slot(None));
            let rhs_slot = *list_elem_slots
                .entry(*rhs)
                .or_insert_with(|| uf.new_slot(None));
            // Ensure lhs and rhs element types unify.
            try_unify(
                uf,
                errors,
                lhs_slot,
                rhs_slot,
                &format!("ListConcat lhs {:?} rhs {:?}", lhs, rhs),
            );
            // Result list shares the same element slot.
            list_elem_slots.insert(*result, lhs_slot);
            let _ = get_or_create_slot(uf, slots, *result, None);
        }
        IrInstr::Cast { result, to_ty, .. } => {
            let _ = get_or_create_slot(uf, slots, *result, Some(to_ty.clone()));
        }
        // Return: each returned value should match the corresponding function return type.
        // (We don't have function return_ty here; leave for a separate pass.)
        IrInstr::Return { values } => {
            for v in values {
                let sv = get_or_create_slot(uf, slots, *v, None);
                try_unify(uf, errors, sv, ret_slot, &format!("Return value {:?}", v));
            }
        }
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
