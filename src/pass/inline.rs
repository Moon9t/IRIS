/// Phase 84: Function inlining pass.
use std::collections::HashMap;
use crate::ir::instr::IrInstr;
use crate::ir::module::IrModule;
use crate::ir::value::ValueId;
use crate::pass::PassError;
use crate::pass::opt::apply_replacements;

pub struct InlinePass {
    pub max_instrs: usize,
}

impl Default for InlinePass {
    fn default() -> Self { Self { max_instrs: 10 } }
}

impl super::Pass for InlinePass {
    fn name(&self) -> &'static str { "inline" }

    fn run(&mut self, module: &mut IrModule) -> Result<(), PassError> {
        let threshold = self.max_instrs;
        let inlineable: HashMap<String, usize> = module.functions.iter().enumerate()
            .filter(|(_, f)| {
                f.blocks.len() == 1 && {
                    let non_term = f.blocks[0].instrs.iter().filter(|i| !i.is_terminator()).count();
                    non_term <= threshold
                }
            })
            .map(|(idx, f)| (f.name.clone(), idx))
            .collect();
        if inlineable.is_empty() { return Ok(()); }

        let num_functions = module.functions.len();
        for caller_idx in 0..num_functions {
            let num_blocks = module.functions[caller_idx].blocks.len();
            for block_idx in 0..num_blocks {
                let instrs: Vec<IrInstr> =
                    module.functions[caller_idx].blocks[block_idx].instrs.clone();
                let mut alias_map: HashMap<ValueId, ValueId> = HashMap::new();
                let mut new_instrs: Vec<IrInstr> = Vec::new();

                for mut instr in instrs {
                    if !alias_map.is_empty() { apply_replacements(&mut instr, &alias_map); }

                    let inlined = if let IrInstr::Call { result: call_result, callee, args, .. } = &instr {
                        if let Some(&callee_idx) = inlineable.get(callee) {
                            if callee_idx != caller_idx {
                                Some(inline_call(module, caller_idx, callee_idx, args, call_result, &mut alias_map))
                            } else { None }
                        } else { None }
                    } else { None };

                    if let Some(mut extra) = inlined {
                        new_instrs.append(&mut extra);
                    } else {
                        new_instrs.push(instr);
                    }
                }
                module.functions[caller_idx].blocks[block_idx].instrs = new_instrs;
            }
        }
        Ok(())
    }
}

fn inline_call(
    module: &mut IrModule,
    caller_idx: usize,
    callee_idx: usize,
    args: &[ValueId],
    call_result: &Option<ValueId>,
    alias_map: &mut HashMap<ValueId, ValueId>,
) -> Vec<IrInstr> {
    let mut val_map: HashMap<ValueId, ValueId> = HashMap::new();
    let callee_params: Vec<ValueId> = module.functions[callee_idx].blocks[0]
        .params.iter().map(|p| p.id).collect();
    for (i, pid) in callee_params.iter().enumerate() {
        if i < args.len() { val_map.insert(*pid, args[i]); }
    }
    let callee_instrs: Vec<IrInstr> = module.functions[callee_idx].blocks[0].instrs.iter()
        .filter(|i| !i.is_terminator()).cloned().collect();
    let callee_ret: Vec<ValueId> = module.functions[callee_idx].blocks[0].instrs.last()
        .and_then(|i| if let IrInstr::Return { values } = i { Some(values.clone()) } else { None })
        .unwrap_or_default();

    let mut emitted: Vec<IrInstr> = Vec::new();
    for mut ci in callee_instrs {
        if let Some(old_result) = ci.result() {
            let fresh = module.functions[caller_idx].fresh_value();
            val_map.insert(old_result, fresh);
            set_result(&mut ci, fresh);
        }
        apply_replacements(&mut ci, &val_map);
        emitted.push(ci);
    }

    if let Some(cr) = call_result {
        if let Some(&callee_ret_val) = callee_ret.first() {
            if let Some(&mapped_ret) = val_map.get(&callee_ret_val) {
                alias_map.insert(*cr, mapped_ret);
            }
        }
    }
    emitted
}

pub(crate) fn set_result(instr: &mut IrInstr, v: ValueId) {
    match instr {
        IrInstr::BinOp { result, .. } => *result = v,
        IrInstr::UnaryOp { result, .. } => *result = v,
        IrInstr::ConstFloat { result, .. } => *result = v,
        IrInstr::ConstInt { result, .. } => *result = v,
        IrInstr::ConstBool { result, .. } => *result = v,
        IrInstr::ConstStr { result, .. } => *result = v,
        IrInstr::TensorOp { result, .. } => *result = v,
        IrInstr::Cast { result, .. } => *result = v,
        IrInstr::Load { result, .. } => *result = v,
        IrInstr::Call { result, .. } => *result = Some(v),
        IrInstr::CallExtern { result, .. } => *result = Some(v),
        IrInstr::CallClosure { result, .. } => *result = Some(v),
        IrInstr::MakeStruct { result, .. } => *result = v,
        IrInstr::GetField { result, .. } => *result = v,
        IrInstr::MakeVariant { result, .. } => *result = v,
        IrInstr::ExtractVariantField { result, .. } => *result = v,
        IrInstr::MakeTuple { result, .. } => *result = v,
        IrInstr::GetElement { result, .. } => *result = v,
        IrInstr::MakeClosure { result, .. } => *result = v,
        IrInstr::AllocArray { result, .. } => *result = v,
        IrInstr::ArrayLoad { result, .. } => *result = v,
        IrInstr::ChanNew { result, .. } => *result = v,
        IrInstr::ChanRecv { result, .. } => *result = v,
        IrInstr::AtomicNew { result, .. } => *result = v,
        IrInstr::AtomicLoad { result, .. } => *result = v,
        IrInstr::AtomicAdd { result, .. } => *result = v,
        IrInstr::MutexNew { result, .. } => *result = v,
        IrInstr::MutexLock { result, .. } => *result = v,
        IrInstr::MakeSome { result, .. } => *result = v,
        IrInstr::MakeNone { result, .. } => *result = v,
        IrInstr::IsSome { result, .. } => *result = v,
        IrInstr::OptionUnwrap { result, .. } => *result = v,
        IrInstr::MakeOk { result, .. } => *result = v,
        IrInstr::MakeErr { result, .. } => *result = v,
        IrInstr::IsOk { result, .. } => *result = v,
        IrInstr::ResultUnwrap { result, .. } => *result = v,
        IrInstr::ResultUnwrapErr { result, .. } => *result = v,
        IrInstr::Sparsify { result, .. } => *result = v,
        IrInstr::Densify { result, .. } => *result = v,
        IrInstr::MakeGrad { result, .. } => *result = v,
        IrInstr::GradValue { result, .. } => *result = v,
        IrInstr::GradTangent { result, .. } => *result = v,
        IrInstr::StrLen { result, .. } => *result = v,
        IrInstr::StrConcat { result, .. } => *result = v,
        IrInstr::StrContains { result, .. } => *result = v,
        IrInstr::StrStartsWith { result, .. } => *result = v,
        IrInstr::StrEndsWith { result, .. } => *result = v,
        IrInstr::StrToUpper { result, .. } => *result = v,
        IrInstr::StrToLower { result, .. } => *result = v,
        IrInstr::StrTrim { result, .. } => *result = v,
        IrInstr::StrRepeat { result, .. } => *result = v,
        IrInstr::ValueToStr { result, .. } => *result = v,
        IrInstr::ReadLine { result } => *result = v,
        IrInstr::ReadI64 { result } => *result = v,
        IrInstr::ReadF64 { result } => *result = v,
        IrInstr::ParseI64 { result, .. } => *result = v,
        IrInstr::ParseF64 { result, .. } => *result = v,
        IrInstr::StrIndex { result, .. } => *result = v,
        IrInstr::StrSlice { result, .. } => *result = v,
        IrInstr::StrFind { result, .. } => *result = v,
        IrInstr::StrReplace { result, .. } => *result = v,
        IrInstr::ListNew { result, .. } => *result = v,
        IrInstr::ListLen { result, .. } => *result = v,
        IrInstr::ListGet { result, .. } => *result = v,
        IrInstr::ListPop { result, .. } => *result = v,
        IrInstr::MapNew { result, .. } => *result = v,
        IrInstr::MapGet { result, .. } => *result = v,
        IrInstr::MapContains { result, .. } => *result = v,
        IrInstr::MapLen { result, .. } => *result = v,
        IrInstr::FileReadAll { result, .. } => *result = v,
        IrInstr::FileWriteAll { result, .. } => *result = v,
        IrInstr::FileExists { result, .. } => *result = v,
        IrInstr::FileLines { result, .. } => *result = v,
        IrInstr::ListContains { result, .. } => *result = v,
        IrInstr::MapKeys { result, .. } => *result = v,
        IrInstr::MapValues { result, .. } => *result = v,
        IrInstr::ListConcat { result, .. } => *result = v,
        IrInstr::ListSlice { result, .. } => *result = v,
        IrInstr::ProcessArgs { result } => *result = v,
        IrInstr::EnvVar { result, .. } => *result = v,
        IrInstr::GetVariantTag { result, .. } => *result = v,
        IrInstr::StrEq { result, .. } => *result = v,
        _ => {}
    }
}
