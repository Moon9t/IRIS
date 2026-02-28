//! IR pretty-printer.
//!
//! Emits a human-readable text representation of an `IrModule`.
//! Output is deterministic: functions are printed in `FunctionId` order,
//! blocks in `BlockId` order, instructions in program order.

use std::fmt::Write;

use crate::error::CodegenError;
use crate::ir::instr::{IrInstr, TensorOp};
use crate::ir::module::IrModule;

/// Emits a full text dump of the IR module.
pub fn emit_ir_text(module: &IrModule) -> Result<String, CodegenError> {
    let mut out = String::new();
    writeln!(out, "// IRIS module: {}", module.name)?;

    for func in module.functions() {
        write!(out, "\ndef {}(", func.name)?;
        for (i, param) in func.params.iter().enumerate() {
            if i > 0 {
                write!(out, ", ")?;
            }
            write!(out, "{}: {}", param.name, param.ty)?;
        }
        writeln!(out, ") -> {} {{", func.return_ty)?;

        for block in func.blocks() {
            let label = block.name.as_deref().unwrap_or("bb");
            write!(out, "  {}{}(", label, block.id.0)?;
            for (i, param) in block.params.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ")?;
                }
                let name = param.name.as_deref().unwrap_or("_");
                write!(out, "{} {}", param.id, name)?;
            }
            writeln!(out, "):")?;

            for instr in &block.instrs {
                write!(out, "    ")?;
                emit_instr(&mut out, instr)?;
                writeln!(out)?;
            }
        }
        writeln!(out, "}}")?;
    }
    Ok(out)
}

fn emit_instr(out: &mut String, instr: &IrInstr) -> Result<(), CodegenError> {
    match instr {
        IrInstr::BinOp {
            result,
            op,
            lhs,
            rhs,
            ..
        } => {
            write!(out, "{} = {} {}, {}", result, op, lhs, rhs)?;
        }

        IrInstr::UnaryOp {
            result,
            op,
            operand,
            ..
        } => {
            write!(out, "{} = {} {}", result, op, operand)?;
        }

        IrInstr::ConstFloat { result, value, ty } => {
            write!(out, "{} = const.f {} : {}", result, value, ty)?;
        }

        IrInstr::ConstInt { result, value, ty } => {
            write!(out, "{} = const.i {} : {}", result, value, ty)?;
        }

        IrInstr::ConstBool { result, value } => {
            write!(out, "{} = const.bool {}", result, value)?;
        }

        IrInstr::TensorOp {
            result,
            op,
            inputs,
            result_ty,
        } => {
            let op_name = match op {
                TensorOp::Einsum { notation } => format!("einsum[\"{}\"]", notation),
                TensorOp::Unary { op } => format!("unary.{}", op),
                TensorOp::Reshape => "reshape".to_owned(),
                TensorOp::Transpose { axes } => {
                    let axes_str: Vec<String> = axes.iter().map(|a| a.to_string()).collect();
                    format!("transpose[{}]", axes_str.join(", "))
                }
                TensorOp::Reduce { op, axes, keepdims } => {
                    let axes_str: Vec<String> = axes.iter().map(|a| a.to_string()).collect();
                    format!(
                        "reduce.{}[{}](keepdims={})",
                        op,
                        axes_str.join(", "),
                        keepdims
                    )
                }
            };
            write!(out, "{} = tensorop.{}", result, op_name)?;
            if !inputs.is_empty() {
                write!(out, "(")?;
                for (i, inp) in inputs.iter().enumerate() {
                    if i > 0 {
                        write!(out, ", ")?;
                    }
                    write!(out, "{}", inp)?;
                }
                write!(out, ")")?;
            }
            write!(out, " : {}", result_ty)?;
        }

        IrInstr::Cast {
            result,
            operand,
            from_ty,
            to_ty,
        } => {
            write!(
                out,
                "{} = cast {} {} : {} -> {}",
                result, to_ty, operand, from_ty, to_ty
            )?;
        }

        IrInstr::Load {
            result,
            tensor,
            indices,
            result_ty,
        } => {
            write!(out, "{} = load {}[", result, tensor)?;
            for (i, idx) in indices.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ")?;
                }
                write!(out, "{}", idx)?;
            }
            write!(out, "] : {}", result_ty)?;
        }

        IrInstr::Store {
            tensor,
            indices,
            value,
        } => {
            write!(out, "store {}[", tensor)?;
            for (i, idx) in indices.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ")?;
                }
                write!(out, "{}", idx)?;
            }
            write!(out, "], {}", value)?;
        }

        IrInstr::Br { target, args } => {
            write!(out, "br {}", target)?;
            if !args.is_empty() {
                write!(out, "(")?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(out, ", ")?;
                    }
                    write!(out, "{}", a)?;
                }
                write!(out, ")")?;
            }
        }

        IrInstr::CondBr {
            cond,
            then_block,
            then_args,
            else_block,
            else_args,
        } => {
            write!(out, "condbr {}, {}(", cond, then_block)?;
            for (i, a) in then_args.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ")?;
                }
                write!(out, "{}", a)?;
            }
            write!(out, "), {}(", else_block)?;
            for (i, a) in else_args.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ")?;
                }
                write!(out, "{}", a)?;
            }
            write!(out, ")")?;
        }

        IrInstr::Return { values } => {
            write!(out, "return")?;
            if !values.is_empty() {
                write!(out, " ")?;
                for (i, v) in values.iter().enumerate() {
                    if i > 0 {
                        write!(out, ", ")?;
                    }
                    write!(out, "{}", v)?;
                }
            }
        }

        IrInstr::Call {
            result,
            callee,
            args,
            ..
        } => {
            if let Some(r) = result {
                write!(out, "{} = ", r)?;
            }
            write!(out, "call @{}", callee)?;
            write!(out, "(")?;
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ")?;
                }
                write!(out, "{}", a)?;
            }
            write!(out, ")")?;
        }

        IrInstr::MakeStruct {
            result,
            fields,
            result_ty,
        } => {
            write!(out, "{} = make_struct {{", result)?;
            for (i, f) in fields.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ")?;
                }
                write!(out, "{}", f)?;
            }
            write!(out, "}} : {}", result_ty)?;
        }

        IrInstr::GetField {
            result,
            base,
            field_index,
            result_ty,
        } => {
            write!(
                out,
                "{} = get_field {}[{}] : {}",
                result, base, field_index, result_ty
            )?;
        }

        IrInstr::MakeVariant {
            result,
            variant_idx,
            fields,
            result_ty,
        } => {
            if fields.is_empty() {
                write!(
                    out,
                    "{} = make_variant {} : {}",
                    result, variant_idx, result_ty
                )?;
            } else {
                write!(out, "{} = make_variant {}(", result, variant_idx)?;
                for (i, f) in fields.iter().enumerate() {
                    if i > 0 { write!(out, ", ")?; }
                    write!(out, "{}", f)?;
                }
                write!(out, ") : {}", result_ty)?;
            }
        }

        IrInstr::ExtractVariantField {
            result,
            operand,
            variant_idx,
            field_idx,
            result_ty,
        } => {
            write!(
                out,
                "{} = extract_variant_field {}[{}.{}] : {}",
                result, operand, variant_idx, field_idx, result_ty
            )?;
        }

        IrInstr::SwitchVariant {
            scrutinee,
            arms,
            default_block,
        } => {
            write!(out, "switch_variant {}", scrutinee)?;
            for (idx, bb) in arms {
                write!(out, ", {} -> {}", idx, bb)?;
            }
            if let Some(def) = default_block {
                write!(out, ", default -> {}", def)?;
            }
        }

        IrInstr::MakeTuple {
            result,
            elements,
            result_ty,
        } => {
            write!(out, "{} = make_tuple(", result)?;
            for (i, e) in elements.iter().enumerate() {
                if i > 0 {
                    write!(out, ", ")?;
                }
                write!(out, "{}", e)?;
            }
            write!(out, ") : {}", result_ty)?;
        }

        IrInstr::GetElement {
            result,
            base,
            index,
            result_ty,
        } => {
            write!(
                out,
                "{} = get_element {}[{}] : {}",
                result, base, index, result_ty
            )?;
        }

        IrInstr::AllocArray {
            result,
            elem_ty,
            size,
            init,
        } => {
            write!(out, "{} = alloc_array [{}; {}](", result, elem_ty, size)?;
            for (i, v) in init.iter().enumerate() {
                if i > 0 { write!(out, ", ")?; }
                write!(out, "{}", v)?;
            }
            write!(out, ")")?;
        }

        IrInstr::ArrayLoad {
            result,
            array,
            index,
            elem_ty,
        } => {
            write!(out, "{} = array_load {}[{}] : {}", result, array, index, elem_ty)?;
        }

        IrInstr::ArrayStore { array, index, value } => {
            write!(out, "array_store {}[{}] = {}", array, index, value)?;
        }

        IrInstr::ParFor { var: _, start, end, body_fn, args } => {
            write!(out, "par_for @{}({}, {})", body_fn, start, end)?;
            if !args.is_empty() {
                write!(out, " captures({})", args.iter().map(|v| format!("{}", v)).collect::<Vec<_>>().join(", "))?;
            }
        }

        IrInstr::ChanNew { result, elem_ty } => {
            write!(out, "{} = chan_new : chan<{}>", result, elem_ty)?;
        }

        IrInstr::ChanSend { chan, value } => {
            write!(out, "chan_send {}, {}", chan, value)?;
        }

        IrInstr::ChanRecv { result, chan, elem_ty } => {
            write!(out, "{} = chan_recv {} : {}", result, chan, elem_ty)?;
        }

        IrInstr::Spawn { body_fn, args } => {
            let arg_strs: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
            write!(out, "spawn @{}({})", body_fn, arg_strs.join(", "))?;
        }

        IrInstr::AtomicNew { result, value, result_ty } => {
            write!(out, "{} = atomic_new {} : {}", result, value, result_ty)?;
        }

        IrInstr::AtomicLoad { result, atomic, result_ty } => {
            write!(out, "{} = atomic_load {} : {}", result, atomic, result_ty)?;
        }

        IrInstr::AtomicStore { atomic, value } => {
            write!(out, "atomic_store {}, {}", atomic, value)?;
        }

        IrInstr::AtomicAdd { result, atomic, value, result_ty } => {
            write!(out, "{} = atomic_add {}, {} : {}", result, atomic, value, result_ty)?;
        }

        IrInstr::MutexNew { result, value, result_ty } => {
            write!(out, "{} = mutex_new {} : {}", result, value, result_ty)?;
        }

        IrInstr::MutexLock { result, mutex, result_ty } => {
            write!(out, "{} = mutex_lock {} : {}", result, mutex, result_ty)?;
        }

        IrInstr::MutexUnlock { mutex } => {
            write!(out, "mutex_unlock {}", mutex)?;
        }

        IrInstr::MakeSome { result, value, result_ty } => {
            write!(out, "{} = make_some {} : {}", result, value, result_ty)?;
        }

        IrInstr::MakeNone { result, result_ty } => {
            write!(out, "{} = make_none : {}", result, result_ty)?;
        }

        IrInstr::IsSome { result, operand } => {
            write!(out, "{} = is_some {}", result, operand)?;
        }

        IrInstr::OptionUnwrap { result, operand, result_ty } => {
            write!(out, "{} = option_unwrap {} : {}", result, operand, result_ty)?;
        }

        IrInstr::MakeOk { result, value, result_ty } => {
            write!(out, "{} = make_ok {} : {}", result, value, result_ty)?;
        }

        IrInstr::MakeErr { result, value, result_ty } => {
            write!(out, "{} = make_err {} : {}", result, value, result_ty)?;
        }

        IrInstr::IsOk { result, operand } => {
            write!(out, "{} = is_ok {}", result, operand)?;
        }

        IrInstr::ResultUnwrap { result, operand, result_ty } => {
            write!(out, "{} = result_unwrap {} : {}", result, operand, result_ty)?;
        }

        IrInstr::ResultUnwrapErr { result, operand, result_ty } => {
            write!(out, "{} = result_unwrap_err {} : {}", result, operand, result_ty)?;
        }

        IrInstr::ConstStr { result, value } => {
            write!(out, "{} = const.str \"{}\"", result, value)?;
        }

        IrInstr::StrLen { result, operand } => {
            write!(out, "{} = str_len {}", result, operand)?;
        }

        IrInstr::StrConcat { result, lhs, rhs } => {
            write!(out, "{} = str_concat {}, {}", result, lhs, rhs)?;
        }

        IrInstr::MakeGrad { result, value, tangent, .. } => {
            write!(out, "{} = make_grad {}, {}", result, value, tangent)?;
        }

        IrInstr::GradValue { result, operand, .. } => {
            write!(out, "{} = grad_value {}", result, operand)?;
        }

        IrInstr::GradTangent { result, operand, .. } => {
            write!(out, "{} = grad_tangent {}", result, operand)?;
        }

        IrInstr::Sparsify { result, operand, .. } => {
            write!(out, "{} = sparsify {}", result, operand)?;
        }

        IrInstr::Densify { result, operand, .. } => {
            write!(out, "{} = densify {}", result, operand)?;
        }

        IrInstr::Barrier => {
            write!(out, "barrier")?;
        }

        IrInstr::Print { operand } => {
            write!(out, "print {}", operand)?;
        }

        IrInstr::StrContains { result, haystack, needle } => {
            write!(out, "{} = str_contains {}, {}", result, haystack, needle)?;
        }
        IrInstr::StrStartsWith { result, haystack, prefix } => {
            write!(out, "{} = str_starts_with {}, {}", result, haystack, prefix)?;
        }
        IrInstr::StrEndsWith { result, haystack, suffix } => {
            write!(out, "{} = str_ends_with {}, {}", result, haystack, suffix)?;
        }
        IrInstr::StrToUpper { result, operand } => {
            write!(out, "{} = str_to_upper {}", result, operand)?;
        }
        IrInstr::StrToLower { result, operand } => {
            write!(out, "{} = str_to_lower {}", result, operand)?;
        }
        IrInstr::StrTrim { result, operand } => {
            write!(out, "{} = str_trim {}", result, operand)?;
        }
        IrInstr::StrRepeat { result, operand, count } => {
            write!(out, "{} = str_repeat {}, {}", result, operand, count)?;
        }

        IrInstr::Panic { msg } => {
            write!(out, "panic {}", msg)?;
        }

        IrInstr::ValueToStr { result, operand } => {
            write!(out, "{} = to_str {}", result, operand)?;
        }

        IrInstr::ReadLine { result } => {
            write!(out, "{} = read_line", result)?;
        }

        IrInstr::ReadI64 { result } => {
            write!(out, "{} = read_i64", result)?;
        }

        IrInstr::ReadF64 { result } => {
            write!(out, "{} = read_f64", result)?;
        }

        IrInstr::ParseI64 { result, operand } => {
            write!(out, "{} = parse_i64 {}", result, operand)?;
        }

        IrInstr::ParseF64 { result, operand } => {
            write!(out, "{} = parse_f64 {}", result, operand)?;
        }

        IrInstr::StrIndex { result, string, index } => {
            write!(out, "{} = str_index {}, {}", result, string, index)?;
        }

        IrInstr::StrSlice { result, string, start, end } => {
            write!(out, "{} = str_slice {}, {}, {}", result, string, start, end)?;
        }

        IrInstr::StrFind { result, haystack, needle } => {
            write!(out, "{} = str_find {}, {}", result, haystack, needle)?;
        }

        IrInstr::StrReplace { result, string, from, to } => {
            write!(out, "{} = str_replace {}, {}, {}", result, string, from, to)?;
        }

        IrInstr::ListNew { result, elem_ty } => {
            write!(out, "{} = list_new<{}>", result, elem_ty)?;
        }
        IrInstr::ListPush { list, value } => {
            write!(out, "list_push {}, {}", list, value)?;
        }
        IrInstr::ListLen { result, list } => {
            write!(out, "{} = list_len {}", result, list)?;
        }
        IrInstr::ListGet { result, list, index, .. } => {
            write!(out, "{} = list_get {}, {}", result, list, index)?;
        }
        IrInstr::ListSet { list, index, value } => {
            write!(out, "list_set {}, {}, {}", list, index, value)?;
        }
        IrInstr::ListPop { result, list, .. } => {
            write!(out, "{} = list_pop {}", result, list)?;
        }

        IrInstr::MapNew { result, key_ty, val_ty } => {
            write!(out, "{} = map_new<{}, {}>", result, key_ty, val_ty)?;
        }
        IrInstr::MapSet { map, key, value } => {
            write!(out, "map_set {}, {}, {}", map, key, value)?;
        }
        IrInstr::MapGet { result, map, key, .. } => {
            write!(out, "{} = map_get {}, {}", result, map, key)?;
        }
        IrInstr::MapContains { result, map, key } => {
            write!(out, "{} = map_contains {}, {}", result, map, key)?;
        }
        IrInstr::MapRemove { map, key } => {
            write!(out, "map_remove {}, {}", map, key)?;
        }
        IrInstr::MapLen { result, map } => {
            write!(out, "{} = map_len {}", result, map)?;
        }

        IrInstr::MakeClosure { result, fn_name, captures, .. } => {
            let caps: Vec<String> = captures.iter().map(|v| format!("{}", v)).collect();
            write!(out, "{} = make_closure @{} [{}]", result, fn_name, caps.join(", "))?;
        }

        IrInstr::CallClosure { result, closure, args, .. } => {
            let arg_strs: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
            if let Some(r) = result {
                write!(out, "{} = call_closure {}({})", r, closure, arg_strs.join(", "))?;
            } else {
                write!(out, "call_closure {}({})", closure, arg_strs.join(", "))?;
            }
        }

        // Phase 56: File I/O
        IrInstr::FileReadAll { result, path } => {
            write!(out, "{} = file_read_all {}", result, path)?;
        }
        IrInstr::FileWriteAll { result, path, content } => {
            write!(out, "{} = file_write_all {}, {}", result, path, content)?;
        }
        IrInstr::FileExists { result, path } => {
            write!(out, "{} = file_exists {}", result, path)?;
        }
        IrInstr::FileLines { result, path } => {
            write!(out, "{} = file_lines {}", result, path)?;
        }

        // Phase 58: Extended collections
        IrInstr::ListContains { result, list, value } => {
            write!(out, "{} = list_contains {}, {}", result, list, value)?;
        }
        IrInstr::ListSort { list } => {
            write!(out, "list_sort {}", list)?;
        }
        IrInstr::MapKeys { result, map } => {
            write!(out, "{} = map_keys {}", result, map)?;
        }
        IrInstr::MapValues { result, map } => {
            write!(out, "{} = map_values {}", result, map)?;
        }
        IrInstr::ListConcat { result, lhs, rhs } => {
            write!(out, "{} = list_concat {}, {}", result, lhs, rhs)?;
        }
        IrInstr::ListSlice { result, list, start, end } => {
            write!(out, "{} = list_slice {}, {}, {}", result, list, start, end)?;
        }

        // Phase 59: Process / environment
        IrInstr::ProcessExit { code } => {
            write!(out, "process_exit {}", code)?;
        }
        IrInstr::ProcessArgs { result } => {
            write!(out, "{} = process_args", result)?;
        }
        IrInstr::EnvVar { result, name } => {
            write!(out, "{} = env_var {}", result, name)?;
        }
        // Phase 61: Pattern matching helpers
        IrInstr::GetVariantTag { result, operand } => {
            write!(out, "{} = get_variant_tag {}", result, operand)?;
        }
        IrInstr::StrEq { result, lhs, rhs } => {
            write!(out, "{} = str_eq {}, {}", result, lhs, rhs)?;
        }
        // Phase 81: FFI
        IrInstr::CallExtern { result, name, args, .. } => {
            let args_str: Vec<String> = args.iter().map(|a| format!("{}", a)).collect();
            if let Some(r) = result {
                write!(out, "{} = call_extern @{}({})", r, name, args_str.join(", "))?;
            } else {
                write!(out, "call_extern @{}({})", name, args_str.join(", "))?;
            }
        }
        // Phase 83: GC
        IrInstr::Retain { ptr } => {
            write!(out, "retain {}", ptr)?;
        }
        IrInstr::Release { ptr, ty } => {
            write!(out, "release {}, {:?}", ptr, ty)?;
        }
        IrInstr::TcpConnect { result, host, port } => {
            write!(out, "{} = tcp_connect {}, {}", result, host, port)?;
        }
        IrInstr::TcpListen { result, port } => {
            write!(out, "{} = tcp_listen {}", result, port)?;
        }
        IrInstr::TcpAccept { result, listener } => {
            write!(out, "{} = tcp_accept {}", result, listener)?;
        }
        IrInstr::TcpRead { result, conn } => {
            write!(out, "{} = tcp_read {}", result, conn)?;
        }
        IrInstr::TcpWrite { conn, data } => {
            write!(out, "tcp_write {}, {}", conn, data)?;
        }
        IrInstr::TcpClose { conn } => {
            write!(out, "tcp_close {}", conn)?;
        }
        IrInstr::StrSplit { result, str_val, delim } => {
            write!(out, "{} = str_split {}, {}", result, str_val, delim)?;
        }
        IrInstr::StrJoin { result, list_val, delim } => {
            write!(out, "{} = str_join {}, {}", result, list_val, delim)?;
        }
    }
    Ok(())
}
