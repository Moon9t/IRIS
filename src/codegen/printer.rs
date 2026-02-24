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
            result_ty,
        } => {
            write!(
                out,
                "{} = make_variant {} : {}",
                result, variant_idx, result_ty
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

        IrInstr::ConstStr { result, value } => {
            write!(out, "{} = const.str \"{}\"", result, value)?;
        }

        IrInstr::StrLen { result, operand } => {
            write!(out, "{} = str_len {}", result, operand)?;
        }

        IrInstr::StrConcat { result, lhs, rhs } => {
            write!(out, "{} = str_concat {}, {}", result, lhs, rhs)?;
        }

        IrInstr::Print { operand } => {
            write!(out, "print {}", operand)?;
        }
    }
    Ok(())
}
