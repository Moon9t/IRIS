//! Binary serialization / deserialization of `IrModule`.
//!
//! ## Wire format
//!
//! ```text
//! Header: b"IRIS" | version(u8=1) | module_name(str) | func_count(u32)
//! For each function:
//!   name(str) | next_value(u32) | param_count(u32) | [param_name + param_ty]
//!   | return_ty | block_count(u32) | [block]
//! For each block:
//!   id(u32) | name(str, empty = None) | bp_count(u32)
//!   | [bp_vid(u32) + bp_ty + bp_name(str)] | instr_count(u32) | [instr]
//! Each instruction: opcode(u8) + fields (see OPCODES below)
//! ```

use std::collections::HashMap;

use crate::ir::block::{BlockId, IrBlock};
use crate::ir::function::{FunctionId, IrFunction, Param};
use crate::ir::instr::{BinOp, InstrId, IrInstr, ScalarUnaryOp, TensorOp};
use crate::ir::module::IrModule;
use crate::ir::types::{DType, Dim, IrType, Shape};
use crate::ir::value::{BlockParam, ValueDef, ValueId};

// ── opcodes ─────────────────────────────────────────────────────────────────
const OP_BINOP: u8             = 0x01;
const OP_UNARY: u8             = 0x02;
const OP_CONST_FLOAT: u8       = 0x03;
const OP_CONST_INT: u8         = 0x04;
const OP_CONST_BOOL: u8        = 0x05;
const OP_CONST_STR: u8         = 0x06;
const OP_TENSOR_OP: u8         = 0x07;
const OP_CAST: u8              = 0x08;
const OP_LOAD: u8              = 0x09;
const OP_STORE: u8             = 0x0A;
const OP_BR: u8                = 0x0B;
const OP_CONDBR: u8            = 0x0C;
const OP_RETURN: u8            = 0x0D;
const OP_CALL: u8              = 0x0E;
const OP_MAKE_STRUCT: u8       = 0x0F;
const OP_GET_FIELD: u8         = 0x10;
const OP_MAKE_VARIANT: u8      = 0x11;
const OP_SWITCH_VARIANT: u8    = 0x12;
const OP_EXTRACT_VARIANT: u8   = 0x13;
const OP_MAKE_TUPLE: u8        = 0x14;
const OP_GET_ELEMENT: u8       = 0x15;
const OP_MAKE_CLOSURE: u8      = 0x16;
const OP_CALL_CLOSURE: u8      = 0x17;
const OP_ALLOC_ARRAY: u8       = 0x18;
const OP_ARRAY_LOAD: u8        = 0x19;
const OP_ARRAY_STORE: u8       = 0x1A;
const OP_MAKE_SOME: u8         = 0x1B;
const OP_MAKE_NONE: u8         = 0x1C;
const OP_IS_SOME: u8           = 0x1D;
const OP_OPTION_UNWRAP: u8     = 0x1E;
const OP_MAKE_OK: u8           = 0x1F;
const OP_MAKE_ERR: u8          = 0x20;
const OP_IS_OK: u8             = 0x21;
const OP_RESULT_UNWRAP: u8     = 0x22;
const OP_RESULT_UNWRAP_ERR: u8 = 0x23;
const OP_CHAN_NEW: u8           = 0x24;
const OP_CHAN_SEND: u8         = 0x25;
const OP_CHAN_RECV: u8         = 0x26;
const OP_SPAWN: u8             = 0x27;
const OP_PAR_FOR: u8           = 0x28;
const OP_ATOMIC_NEW: u8        = 0x29;
const OP_ATOMIC_LOAD: u8       = 0x2A;
const OP_ATOMIC_STORE: u8      = 0x2B;
const OP_ATOMIC_ADD: u8        = 0x2C;
const OP_MUTEX_NEW: u8         = 0x2D;
const OP_MUTEX_LOCK: u8        = 0x2E;
const OP_MUTEX_UNLOCK: u8      = 0x2F;
const OP_BARRIER: u8           = 0x30;
const OP_MAKE_GRAD: u8         = 0x31;
const OP_GRAD_VALUE: u8        = 0x32;
const OP_GRAD_TANGENT: u8      = 0x33;
const OP_SPARSIFY: u8          = 0x34;
const OP_DENSIFY: u8           = 0x35;
const OP_STR_LEN: u8           = 0x36;
const OP_STR_CONCAT: u8        = 0x37;
const OP_PRINT: u8             = 0x38;
const OP_STR_CONTAINS: u8      = 0x39;
const OP_STR_STARTS_WITH: u8   = 0x3A;
const OP_STR_ENDS_WITH: u8     = 0x3B;
const OP_STR_TO_UPPER: u8      = 0x3C;
const OP_STR_TO_LOWER: u8      = 0x3D;
const OP_STR_TRIM: u8          = 0x3E;
const OP_STR_REPEAT: u8        = 0x3F;
const OP_PANIC: u8             = 0x40;
const OP_VALUE_TO_STR: u8      = 0x41;
const OP_READ_LINE: u8         = 0x42;
const OP_READ_I64: u8          = 0x43;
const OP_READ_F64: u8          = 0x44;
const OP_PARSE_I64: u8         = 0x45;
const OP_PARSE_F64: u8         = 0x46;
const OP_STR_INDEX: u8         = 0x47;
const OP_STR_SLICE: u8         = 0x48;
const OP_STR_FIND: u8          = 0x49;
const OP_STR_REPLACE: u8       = 0x4A;
const OP_LIST_NEW: u8          = 0x4B;
const OP_LIST_PUSH: u8         = 0x4C;
const OP_LIST_LEN: u8          = 0x4D;
const OP_LIST_GET: u8          = 0x4E;
const OP_LIST_SET: u8          = 0x4F;
const OP_LIST_POP: u8          = 0x50;
const OP_MAP_NEW: u8           = 0x51;
const OP_MAP_SET: u8           = 0x52;
const OP_MAP_GET: u8           = 0x53;
const OP_MAP_CONTAINS: u8      = 0x54;
const OP_MAP_REMOVE: u8        = 0x55;
const OP_MAP_LEN: u8           = 0x56;
const OP_FILE_READ_ALL: u8     = 0x57;
const OP_FILE_WRITE_ALL: u8    = 0x58;
const OP_FILE_EXISTS: u8       = 0x59;
const OP_FILE_LINES: u8        = 0x5A;
const OP_LIST_CONTAINS: u8     = 0x5B;
const OP_LIST_SORT: u8         = 0x5C;
const OP_MAP_KEYS: u8          = 0x5D;
const OP_MAP_VALUES: u8        = 0x5E;
const OP_LIST_CONCAT: u8       = 0x5F;
const OP_LIST_SLICE: u8        = 0x60;
const OP_PROCESS_EXIT: u8      = 0x61;
const OP_PROCESS_ARGS: u8      = 0x62;
const OP_ENV_VAR: u8           = 0x63;
const OP_GET_VARIANT_TAG: u8   = 0x64;
const OP_STR_EQ: u8            = 0x65;

const MAGIC: &[u8; 4] = b"IRIS";
const VERSION: u8 = 1;

// ── Writer ───────────────────────────────────────────────────────────────────

struct Writer { buf: Vec<u8> }

impl Writer {
    fn new() -> Self { Self { buf: Vec::new() } }
    fn u8(&mut self, v: u8)  { self.buf.push(v); }
    fn u32(&mut self, v: u32) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn u64(&mut self, v: u64) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn i64(&mut self, v: i64) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn f64(&mut self, v: f64) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn bool(&mut self, v: bool) { self.u8(if v { 1 } else { 0 }); }
    fn str(&mut self, s: &str) { self.u32(s.len() as u32); self.buf.extend_from_slice(s.as_bytes()); }
    fn vid(&mut self, v: ValueId)  { self.u32(v.0); }
    fn bid(&mut self, b: BlockId)  { self.u32(b.0); }
    fn vids(&mut self, vs: &[ValueId]) { self.u32(vs.len() as u32); for v in vs { self.vid(*v); } }
    fn opt_vid(&mut self, v: Option<ValueId>) { self.bool(v.is_some()); if let Some(id) = v { self.vid(id); } }
    fn opt_bid(&mut self, b: Option<BlockId>) { self.bool(b.is_some()); if let Some(id) = b { self.bid(id); } }
    fn opt_ty(&mut self, t: &Option<IrType>) { self.bool(t.is_some()); if let Some(ty) = t { self.ty(ty); } }

    fn dtype(&mut self, d: DType) {
        let tag = match d {
            DType::F32 => 1, DType::F64 => 2, DType::I32 => 3, DType::I64 => 4,
            DType::Bool => 5, DType::U8 => 6, DType::I8 => 7, DType::U32 => 8,
            DType::U64 => 9, DType::USize => 10,
        };
        self.u8(tag);
    }

    fn ty(&mut self, t: &IrType) {
        match t {
            IrType::Scalar(d) => { self.u8(0x01); self.dtype(*d); }
            IrType::Tensor { dtype, shape } => {
                self.u8(0x02); self.dtype(*dtype);
                self.u32(shape.0.len() as u32);
                for dim in &shape.0 {
                    match dim {
                        Dim::Literal(n) => { self.u8(1); self.u64(*n); }
                        Dim::Symbolic(s) => { self.u8(2); self.str(s); }
                    }
                }
            }
            IrType::Fn { params, ret } => {
                self.u8(0x03); self.u32(params.len() as u32);
                for p in params { self.ty(p); }
                self.ty(ret);
            }
            IrType::Infer => { self.u8(0x04); }
            IrType::Struct { name, fields } => {
                self.u8(0x05); self.str(name);
                self.u32(fields.len() as u32);
                for (fname, fty) in fields { self.str(fname); self.ty(fty); }
            }
            IrType::Enum { name, variants } => {
                self.u8(0x06); self.str(name);
                self.u32(variants.len() as u32);
                for v in variants { self.str(v); }
            }
            IrType::Tuple(elems) => {
                self.u8(0x07); self.u32(elems.len() as u32);
                for e in elems { self.ty(e); }
            }
            IrType::Str    => { self.u8(0x08); }
            IrType::Array { elem, len } => { self.u8(0x09); self.ty(elem); self.u32(*len as u32); }
            IrType::Option(inner)           => { self.u8(0x0A); self.ty(inner); }
            IrType::ResultType(ok, err)     => { self.u8(0x0B); self.ty(ok); self.ty(err); }
            IrType::Chan(elem)              => { self.u8(0x0C); self.ty(elem); }
            IrType::Atomic(inner)           => { self.u8(0x0D); self.ty(inner); }
            IrType::Mutex(inner)            => { self.u8(0x0E); self.ty(inner); }
            IrType::Grad(inner)             => { self.u8(0x0F); self.ty(inner); }
            IrType::Sparse(inner)           => { self.u8(0x10); self.ty(inner); }
            IrType::List(elem)              => { self.u8(0x11); self.ty(elem); }
            IrType::Map(k, v)               => { self.u8(0x12); self.ty(k); self.ty(v); }
        }
    }

    fn binop(&mut self, op: BinOp) {
        let tag: u8 = match op {
            BinOp::Add => 1, BinOp::Sub => 2, BinOp::Mul => 3, BinOp::Div => 4,
            BinOp::FloorDiv => 5, BinOp::Mod => 6, BinOp::Pow => 7, BinOp::Min => 8,
            BinOp::Max => 9, BinOp::BitAnd => 10, BinOp::BitOr => 11, BinOp::BitXor => 12,
            BinOp::Shl => 13, BinOp::Shr => 14, BinOp::CmpEq => 15, BinOp::CmpNe => 16,
            BinOp::CmpLt => 17, BinOp::CmpLe => 18, BinOp::CmpGt => 19, BinOp::CmpGe => 20,
        };
        self.u8(tag);
    }

    fn unaryop(&mut self, op: ScalarUnaryOp) {
        let tag: u8 = match op {
            ScalarUnaryOp::Neg => 1, ScalarUnaryOp::Not => 2, ScalarUnaryOp::Sqrt => 3,
            ScalarUnaryOp::Abs => 4, ScalarUnaryOp::Floor => 5, ScalarUnaryOp::Ceil => 6,
            ScalarUnaryOp::BitNot => 7, ScalarUnaryOp::Sin => 8, ScalarUnaryOp::Cos => 9,
            ScalarUnaryOp::Tan => 10, ScalarUnaryOp::Exp => 11, ScalarUnaryOp::Log => 12,
            ScalarUnaryOp::Log2 => 13, ScalarUnaryOp::Round => 14, ScalarUnaryOp::Sign => 15,
        };
        self.u8(tag);
    }

    fn tensor_op(&mut self, op: &TensorOp) {
        match op {
            TensorOp::Einsum { notation } => { self.u8(1); self.str(notation); }
            TensorOp::Unary { op }        => { self.u8(2); self.str(op); }
            TensorOp::Reshape             => { self.u8(3); }
            TensorOp::Transpose { axes }  => {
                self.u8(4); self.u32(axes.len() as u32);
                for a in axes { self.u32(*a as u32); }
            }
            TensorOp::Reduce { op, axes, keepdims } => {
                self.u8(5); self.str(op);
                self.u32(axes.len() as u32);
                for a in axes { self.u32(*a as u32); }
                self.bool(*keepdims);
            }
        }
    }

    fn instr(&mut self, i: &IrInstr) {
        match i {
            IrInstr::BinOp { result, op, lhs, rhs, ty } => {
                self.u8(OP_BINOP); self.vid(*result); self.binop(*op); self.vid(*lhs); self.vid(*rhs); self.ty(ty);
            }
            IrInstr::UnaryOp { result, op, operand, ty } => {
                self.u8(OP_UNARY); self.vid(*result); self.unaryop(*op); self.vid(*operand); self.ty(ty);
            }
            IrInstr::ConstFloat { result, value, ty } => {
                self.u8(OP_CONST_FLOAT); self.vid(*result); self.f64(*value); self.ty(ty);
            }
            IrInstr::ConstInt { result, value, ty } => {
                self.u8(OP_CONST_INT); self.vid(*result); self.i64(*value); self.ty(ty);
            }
            IrInstr::ConstBool { result, value } => {
                self.u8(OP_CONST_BOOL); self.vid(*result); self.bool(*value);
            }
            IrInstr::ConstStr { result, value } => {
                self.u8(OP_CONST_STR); self.vid(*result); self.str(value);
            }
            IrInstr::TensorOp { result, op, inputs, result_ty } => {
                self.u8(OP_TENSOR_OP); self.vid(*result); self.tensor_op(op);
                self.vids(inputs); self.ty(result_ty);
            }
            IrInstr::Cast { result, operand, from_ty, to_ty } => {
                self.u8(OP_CAST); self.vid(*result); self.vid(*operand); self.ty(from_ty); self.ty(to_ty);
            }
            IrInstr::Load { result, tensor, indices, result_ty } => {
                self.u8(OP_LOAD); self.vid(*result); self.vid(*tensor); self.vids(indices); self.ty(result_ty);
            }
            IrInstr::Store { tensor, indices, value } => {
                self.u8(OP_STORE); self.vid(*tensor); self.vids(indices); self.vid(*value);
            }
            IrInstr::Br { target, args } => {
                self.u8(OP_BR); self.bid(*target); self.vids(args);
            }
            IrInstr::CondBr { cond, then_block, then_args, else_block, else_args } => {
                self.u8(OP_CONDBR);
                self.vid(*cond); self.bid(*then_block); self.vids(then_args);
                self.bid(*else_block); self.vids(else_args);
            }
            IrInstr::Return { values } => {
                self.u8(OP_RETURN); self.vids(values);
            }
            IrInstr::Call { result, callee, args, result_ty } => {
                self.u8(OP_CALL); self.opt_vid(*result); self.str(callee); self.vids(args); self.opt_ty(result_ty);
            }
            IrInstr::MakeStruct { result, fields, result_ty } => {
                self.u8(OP_MAKE_STRUCT); self.vid(*result); self.vids(fields); self.ty(result_ty);
            }
            IrInstr::GetField { result, base, field_index, result_ty } => {
                self.u8(OP_GET_FIELD); self.vid(*result); self.vid(*base); self.u32(*field_index as u32); self.ty(result_ty);
            }
            IrInstr::MakeVariant { result, variant_idx, fields, result_ty } => {
                self.u8(OP_MAKE_VARIANT); self.vid(*result); self.u32(*variant_idx as u32); self.vids(fields); self.ty(result_ty);
            }
            IrInstr::SwitchVariant { scrutinee, arms, default_block } => {
                self.u8(OP_SWITCH_VARIANT); self.vid(*scrutinee);
                self.u32(arms.len() as u32);
                for (idx, bid) in arms { self.u32(*idx as u32); self.bid(*bid); }
                self.opt_bid(*default_block);
            }
            IrInstr::ExtractVariantField { result, operand, variant_idx, field_idx, result_ty } => {
                self.u8(OP_EXTRACT_VARIANT); self.vid(*result); self.vid(*operand);
                self.u32(*variant_idx as u32); self.u32(*field_idx as u32); self.ty(result_ty);
            }
            IrInstr::MakeTuple { result, elements, result_ty } => {
                self.u8(OP_MAKE_TUPLE); self.vid(*result); self.vids(elements); self.ty(result_ty);
            }
            IrInstr::GetElement { result, base, index, result_ty } => {
                self.u8(OP_GET_ELEMENT); self.vid(*result); self.vid(*base); self.u32(*index as u32); self.ty(result_ty);
            }
            IrInstr::MakeClosure { result, fn_name, captures, result_ty } => {
                self.u8(OP_MAKE_CLOSURE); self.vid(*result); self.str(fn_name); self.vids(captures); self.ty(result_ty);
            }
            IrInstr::CallClosure { result, closure, args, result_ty } => {
                self.u8(OP_CALL_CLOSURE); self.opt_vid(*result); self.vid(*closure); self.vids(args); self.ty(result_ty);
            }
            IrInstr::AllocArray { result, elem_ty, size, init } => {
                self.u8(OP_ALLOC_ARRAY); self.vid(*result); self.ty(elem_ty); self.u32(*size as u32); self.vids(init);
            }
            IrInstr::ArrayLoad { result, array, index, elem_ty } => {
                self.u8(OP_ARRAY_LOAD); self.vid(*result); self.vid(*array); self.vid(*index); self.ty(elem_ty);
            }
            IrInstr::ArrayStore { array, index, value } => {
                self.u8(OP_ARRAY_STORE); self.vid(*array); self.vid(*index); self.vid(*value);
            }
            IrInstr::MakeSome { result, value, result_ty } => {
                self.u8(OP_MAKE_SOME); self.vid(*result); self.vid(*value); self.ty(result_ty);
            }
            IrInstr::MakeNone { result, result_ty } => {
                self.u8(OP_MAKE_NONE); self.vid(*result); self.ty(result_ty);
            }
            IrInstr::IsSome { result, operand } => {
                self.u8(OP_IS_SOME); self.vid(*result); self.vid(*operand);
            }
            IrInstr::OptionUnwrap { result, operand, result_ty } => {
                self.u8(OP_OPTION_UNWRAP); self.vid(*result); self.vid(*operand); self.ty(result_ty);
            }
            IrInstr::MakeOk { result, value, result_ty } => {
                self.u8(OP_MAKE_OK); self.vid(*result); self.vid(*value); self.ty(result_ty);
            }
            IrInstr::MakeErr { result, value, result_ty } => {
                self.u8(OP_MAKE_ERR); self.vid(*result); self.vid(*value); self.ty(result_ty);
            }
            IrInstr::IsOk { result, operand } => {
                self.u8(OP_IS_OK); self.vid(*result); self.vid(*operand);
            }
            IrInstr::ResultUnwrap { result, operand, result_ty } => {
                self.u8(OP_RESULT_UNWRAP); self.vid(*result); self.vid(*operand); self.ty(result_ty);
            }
            IrInstr::ResultUnwrapErr { result, operand, result_ty } => {
                self.u8(OP_RESULT_UNWRAP_ERR); self.vid(*result); self.vid(*operand); self.ty(result_ty);
            }
            IrInstr::ChanNew { result, elem_ty } => {
                self.u8(OP_CHAN_NEW); self.vid(*result); self.ty(elem_ty);
            }
            IrInstr::ChanSend { chan, value } => {
                self.u8(OP_CHAN_SEND); self.vid(*chan); self.vid(*value);
            }
            IrInstr::ChanRecv { result, chan, elem_ty } => {
                self.u8(OP_CHAN_RECV); self.vid(*result); self.vid(*chan); self.ty(elem_ty);
            }
            IrInstr::Spawn { body_fn, args } => {
                self.u8(OP_SPAWN); self.str(body_fn); self.vids(args);
            }
            IrInstr::ParFor { var, start, end, body_fn, args } => {
                self.u8(OP_PAR_FOR); self.vid(*var); self.vid(*start); self.vid(*end);
                self.str(body_fn); self.vids(args);
            }
            IrInstr::AtomicNew { result, value, result_ty } => {
                self.u8(OP_ATOMIC_NEW); self.vid(*result); self.vid(*value); self.ty(result_ty);
            }
            IrInstr::AtomicLoad { result, atomic, result_ty } => {
                self.u8(OP_ATOMIC_LOAD); self.vid(*result); self.vid(*atomic); self.ty(result_ty);
            }
            IrInstr::AtomicStore { atomic, value } => {
                self.u8(OP_ATOMIC_STORE); self.vid(*atomic); self.vid(*value);
            }
            IrInstr::AtomicAdd { result, atomic, value, result_ty } => {
                self.u8(OP_ATOMIC_ADD); self.vid(*result); self.vid(*atomic); self.vid(*value); self.ty(result_ty);
            }
            IrInstr::MutexNew { result, value, result_ty } => {
                self.u8(OP_MUTEX_NEW); self.vid(*result); self.vid(*value); self.ty(result_ty);
            }
            IrInstr::MutexLock { result, mutex, result_ty } => {
                self.u8(OP_MUTEX_LOCK); self.vid(*result); self.vid(*mutex); self.ty(result_ty);
            }
            IrInstr::MutexUnlock { mutex } => {
                self.u8(OP_MUTEX_UNLOCK); self.vid(*mutex);
            }
            IrInstr::Barrier => { self.u8(OP_BARRIER); }
            IrInstr::MakeGrad { result, value, tangent, ty } => {
                self.u8(OP_MAKE_GRAD); self.vid(*result); self.vid(*value); self.vid(*tangent); self.ty(ty);
            }
            IrInstr::GradValue { result, operand, ty } => {
                self.u8(OP_GRAD_VALUE); self.vid(*result); self.vid(*operand); self.ty(ty);
            }
            IrInstr::GradTangent { result, operand, ty } => {
                self.u8(OP_GRAD_TANGENT); self.vid(*result); self.vid(*operand); self.ty(ty);
            }
            IrInstr::Sparsify { result, operand, ty } => {
                self.u8(OP_SPARSIFY); self.vid(*result); self.vid(*operand); self.ty(ty);
            }
            IrInstr::Densify { result, operand, ty } => {
                self.u8(OP_DENSIFY); self.vid(*result); self.vid(*operand); self.ty(ty);
            }
            IrInstr::StrLen { result, operand } => {
                self.u8(OP_STR_LEN); self.vid(*result); self.vid(*operand);
            }
            IrInstr::StrConcat { result, lhs, rhs } => {
                self.u8(OP_STR_CONCAT); self.vid(*result); self.vid(*lhs); self.vid(*rhs);
            }
            IrInstr::Print { operand } => { self.u8(OP_PRINT); self.vid(*operand); }
            IrInstr::StrContains { result, haystack, needle } => {
                self.u8(OP_STR_CONTAINS); self.vid(*result); self.vid(*haystack); self.vid(*needle);
            }
            IrInstr::StrStartsWith { result, haystack, prefix } => {
                self.u8(OP_STR_STARTS_WITH); self.vid(*result); self.vid(*haystack); self.vid(*prefix);
            }
            IrInstr::StrEndsWith { result, haystack, suffix } => {
                self.u8(OP_STR_ENDS_WITH); self.vid(*result); self.vid(*haystack); self.vid(*suffix);
            }
            IrInstr::StrToUpper { result, operand } => {
                self.u8(OP_STR_TO_UPPER); self.vid(*result); self.vid(*operand);
            }
            IrInstr::StrToLower { result, operand } => {
                self.u8(OP_STR_TO_LOWER); self.vid(*result); self.vid(*operand);
            }
            IrInstr::StrTrim { result, operand } => {
                self.u8(OP_STR_TRIM); self.vid(*result); self.vid(*operand);
            }
            IrInstr::StrRepeat { result, operand, count } => {
                self.u8(OP_STR_REPEAT); self.vid(*result); self.vid(*operand); self.vid(*count);
            }
            IrInstr::Panic { msg } => { self.u8(OP_PANIC); self.vid(*msg); }
            IrInstr::ValueToStr { result, operand } => {
                self.u8(OP_VALUE_TO_STR); self.vid(*result); self.vid(*operand);
            }
            IrInstr::ReadLine { result } => { self.u8(OP_READ_LINE); self.vid(*result); }
            IrInstr::ReadI64 { result }  => { self.u8(OP_READ_I64);  self.vid(*result); }
            IrInstr::ReadF64 { result }  => { self.u8(OP_READ_F64);  self.vid(*result); }
            IrInstr::ParseI64 { result, operand } => {
                self.u8(OP_PARSE_I64); self.vid(*result); self.vid(*operand);
            }
            IrInstr::ParseF64 { result, operand } => {
                self.u8(OP_PARSE_F64); self.vid(*result); self.vid(*operand);
            }
            IrInstr::StrIndex { result, string, index } => {
                self.u8(OP_STR_INDEX); self.vid(*result); self.vid(*string); self.vid(*index);
            }
            IrInstr::StrSlice { result, string, start, end } => {
                self.u8(OP_STR_SLICE); self.vid(*result); self.vid(*string); self.vid(*start); self.vid(*end);
            }
            IrInstr::StrFind { result, haystack, needle } => {
                self.u8(OP_STR_FIND); self.vid(*result); self.vid(*haystack); self.vid(*needle);
            }
            IrInstr::StrReplace { result, string, from, to } => {
                self.u8(OP_STR_REPLACE); self.vid(*result); self.vid(*string); self.vid(*from); self.vid(*to);
            }
            IrInstr::ListNew { result, elem_ty } => {
                self.u8(OP_LIST_NEW); self.vid(*result); self.ty(elem_ty);
            }
            IrInstr::ListPush { list, value } => {
                self.u8(OP_LIST_PUSH); self.vid(*list); self.vid(*value);
            }
            IrInstr::ListLen { result, list } => {
                self.u8(OP_LIST_LEN); self.vid(*result); self.vid(*list);
            }
            IrInstr::ListGet { result, list, index, elem_ty } => {
                self.u8(OP_LIST_GET); self.vid(*result); self.vid(*list); self.vid(*index); self.ty(elem_ty);
            }
            IrInstr::ListSet { list, index, value } => {
                self.u8(OP_LIST_SET); self.vid(*list); self.vid(*index); self.vid(*value);
            }
            IrInstr::ListPop { result, list, elem_ty } => {
                self.u8(OP_LIST_POP); self.vid(*result); self.vid(*list); self.ty(elem_ty);
            }
            IrInstr::MapNew { result, key_ty, val_ty } => {
                self.u8(OP_MAP_NEW); self.vid(*result); self.ty(key_ty); self.ty(val_ty);
            }
            IrInstr::MapSet { map, key, value } => {
                self.u8(OP_MAP_SET); self.vid(*map); self.vid(*key); self.vid(*value);
            }
            IrInstr::MapGet { result, map, key, val_ty } => {
                self.u8(OP_MAP_GET); self.vid(*result); self.vid(*map); self.vid(*key); self.ty(val_ty);
            }
            IrInstr::MapContains { result, map, key } => {
                self.u8(OP_MAP_CONTAINS); self.vid(*result); self.vid(*map); self.vid(*key);
            }
            IrInstr::MapRemove { map, key } => {
                self.u8(OP_MAP_REMOVE); self.vid(*map); self.vid(*key);
            }
            IrInstr::MapLen { result, map } => {
                self.u8(OP_MAP_LEN); self.vid(*result); self.vid(*map);
            }
            IrInstr::FileReadAll { result, path } => {
                self.u8(OP_FILE_READ_ALL); self.vid(*result); self.vid(*path);
            }
            IrInstr::FileWriteAll { result, path, content } => {
                self.u8(OP_FILE_WRITE_ALL); self.vid(*result); self.vid(*path); self.vid(*content);
            }
            IrInstr::FileExists { result, path } => {
                self.u8(OP_FILE_EXISTS); self.vid(*result); self.vid(*path);
            }
            IrInstr::FileLines { result, path } => {
                self.u8(OP_FILE_LINES); self.vid(*result); self.vid(*path);
            }
            IrInstr::ListContains { result, list, value } => {
                self.u8(OP_LIST_CONTAINS); self.vid(*result); self.vid(*list); self.vid(*value);
            }
            IrInstr::ListSort { list } => { self.u8(OP_LIST_SORT); self.vid(*list); }
            IrInstr::MapKeys { result, map } => {
                self.u8(OP_MAP_KEYS); self.vid(*result); self.vid(*map);
            }
            IrInstr::MapValues { result, map } => {
                self.u8(OP_MAP_VALUES); self.vid(*result); self.vid(*map);
            }
            IrInstr::ListConcat { result, lhs, rhs } => {
                self.u8(OP_LIST_CONCAT); self.vid(*result); self.vid(*lhs); self.vid(*rhs);
            }
            IrInstr::ListSlice { result, list, start, end } => {
                self.u8(OP_LIST_SLICE); self.vid(*result); self.vid(*list); self.vid(*start); self.vid(*end);
            }
            IrInstr::ProcessExit { code } => { self.u8(OP_PROCESS_EXIT); self.vid(*code); }
            IrInstr::ProcessArgs { result } => { self.u8(OP_PROCESS_ARGS); self.vid(*result); }
            IrInstr::EnvVar { result, name } => {
                self.u8(OP_ENV_VAR); self.vid(*result); self.vid(*name);
            }
            IrInstr::GetVariantTag { result, operand } => {
                self.u8(OP_GET_VARIANT_TAG); self.vid(*result); self.vid(*operand);
            }
            IrInstr::StrEq { result, lhs, rhs } => {
                self.u8(OP_STR_EQ); self.vid(*result); self.vid(*lhs); self.vid(*rhs);
            }
        }
    }
}

// ── Reader ───────────────────────────────────────────────────────────────────

struct Reader<'a> { data: &'a [u8], pos: usize }

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self { Self { data, pos: 0 } }

    fn u8(&mut self) -> Result<u8, String> {
        if self.pos >= self.data.len() { return Err("unexpected end of data".into()); }
        let v = self.data[self.pos]; self.pos += 1; Ok(v)
    }
    fn u32(&mut self) -> Result<u32, String> {
        if self.pos + 4 > self.data.len() { return Err("truncated u32".into()); }
        let v = u32::from_le_bytes(self.data[self.pos..self.pos+4].try_into().unwrap());
        self.pos += 4; Ok(v)
    }
    fn u64(&mut self) -> Result<u64, String> {
        if self.pos + 8 > self.data.len() { return Err("truncated u64".into()); }
        let v = u64::from_le_bytes(self.data[self.pos..self.pos+8].try_into().unwrap());
        self.pos += 8; Ok(v)
    }
    fn i64(&mut self) -> Result<i64, String> {
        if self.pos + 8 > self.data.len() { return Err("truncated i64".into()); }
        let v = i64::from_le_bytes(self.data[self.pos..self.pos+8].try_into().unwrap());
        self.pos += 8; Ok(v)
    }
    fn f64(&mut self) -> Result<f64, String> {
        if self.pos + 8 > self.data.len() { return Err("truncated f64".into()); }
        let v = f64::from_le_bytes(self.data[self.pos..self.pos+8].try_into().unwrap());
        self.pos += 8; Ok(v)
    }
    fn bool(&mut self) -> Result<bool, String> { Ok(self.u8()? != 0) }
    fn str(&mut self) -> Result<String, String> {
        let len = self.u32()? as usize;
        if self.pos + len > self.data.len() { return Err("truncated string".into()); }
        let s = std::str::from_utf8(&self.data[self.pos..self.pos+len])
            .map_err(|e| e.to_string())?.to_owned();
        self.pos += len; Ok(s)
    }
    fn vid(&mut self) -> Result<ValueId, String> { Ok(ValueId(self.u32()?)) }
    fn bid(&mut self) -> Result<BlockId, String> { Ok(BlockId(self.u32()?)) }
    fn vids(&mut self) -> Result<Vec<ValueId>, String> {
        let n = self.u32()? as usize;
        (0..n).map(|_| self.vid()).collect()
    }
    fn opt_vid(&mut self) -> Result<Option<ValueId>, String> {
        if self.bool()? { Ok(Some(self.vid()?)) } else { Ok(None) }
    }
    fn opt_bid(&mut self) -> Result<Option<BlockId>, String> {
        if self.bool()? { Ok(Some(self.bid()?)) } else { Ok(None) }
    }
    fn opt_ty(&mut self) -> Result<Option<IrType>, String> {
        if self.bool()? { Ok(Some(self.ty()?)) } else { Ok(None) }
    }

    fn dtype(&mut self) -> Result<DType, String> {
        Ok(match self.u8()? {
            1 => DType::F32, 2 => DType::F64, 3 => DType::I32, 4 => DType::I64,
            5 => DType::Bool, 6 => DType::U8, 7 => DType::I8, 8 => DType::U32,
            9 => DType::U64, 10 => DType::USize,
            t => return Err(format!("unknown dtype tag {}", t)),
        })
    }

    fn ty(&mut self) -> Result<IrType, String> {
        Ok(match self.u8()? {
            0x01 => IrType::Scalar(self.dtype()?),
            0x02 => {
                let dtype = self.dtype()?;
                let n = self.u32()? as usize;
                let mut dims = Vec::with_capacity(n);
                for _ in 0..n {
                    dims.push(match self.u8()? {
                        1 => Dim::Literal(self.u64()?),
                        2 => Dim::Symbolic(self.str()?),
                        t => return Err(format!("unknown dim tag {}", t)),
                    });
                }
                IrType::Tensor { dtype, shape: Shape(dims) }
            }
            0x03 => {
                let n = self.u32()? as usize;
                let params: Result<Vec<_>,_> = (0..n).map(|_| self.ty()).collect();
                let ret = self.ty()?;
                IrType::Fn { params: params?, ret: Box::new(ret) }
            }
            0x04 => IrType::Infer,
            0x05 => {
                let name = self.str()?;
                let n = self.u32()? as usize;
                let mut fields = Vec::with_capacity(n);
                for _ in 0..n { fields.push((self.str()?, self.ty()?)); }
                IrType::Struct { name, fields }
            }
            0x06 => {
                let name = self.str()?;
                let n = self.u32()? as usize;
                let variants: Result<Vec<_>,_> = (0..n).map(|_| self.str()).collect();
                IrType::Enum { name, variants: variants? }
            }
            0x07 => {
                let n = self.u32()? as usize;
                let elems: Result<Vec<_>,_> = (0..n).map(|_| self.ty()).collect();
                IrType::Tuple(elems?)
            }
            0x08 => IrType::Str,
            0x09 => { let elem = self.ty()?; let len = self.u32()? as usize; IrType::Array { elem: Box::new(elem), len } }
            0x0A => IrType::Option(Box::new(self.ty()?)),
            0x0B => { let ok = self.ty()?; let err = self.ty()?; IrType::ResultType(Box::new(ok), Box::new(err)) }
            0x0C => IrType::Chan(Box::new(self.ty()?)),
            0x0D => IrType::Atomic(Box::new(self.ty()?)),
            0x0E => IrType::Mutex(Box::new(self.ty()?)),
            0x0F => IrType::Grad(Box::new(self.ty()?)),
            0x10 => IrType::Sparse(Box::new(self.ty()?)),
            0x11 => IrType::List(Box::new(self.ty()?)),
            0x12 => { let k = self.ty()?; let v = self.ty()?; IrType::Map(Box::new(k), Box::new(v)) }
            t => return Err(format!("unknown type tag 0x{:02x}", t)),
        })
    }

    fn binop(&mut self) -> Result<BinOp, String> {
        Ok(match self.u8()? {
            1 => BinOp::Add, 2 => BinOp::Sub, 3 => BinOp::Mul, 4 => BinOp::Div,
            5 => BinOp::FloorDiv, 6 => BinOp::Mod, 7 => BinOp::Pow, 8 => BinOp::Min,
            9 => BinOp::Max, 10 => BinOp::BitAnd, 11 => BinOp::BitOr, 12 => BinOp::BitXor,
            13 => BinOp::Shl, 14 => BinOp::Shr, 15 => BinOp::CmpEq, 16 => BinOp::CmpNe,
            17 => BinOp::CmpLt, 18 => BinOp::CmpLe, 19 => BinOp::CmpGt, 20 => BinOp::CmpGe,
            t => return Err(format!("unknown binop tag {}", t)),
        })
    }

    fn unaryop(&mut self) -> Result<ScalarUnaryOp, String> {
        Ok(match self.u8()? {
            1 => ScalarUnaryOp::Neg, 2 => ScalarUnaryOp::Not, 3 => ScalarUnaryOp::Sqrt,
            4 => ScalarUnaryOp::Abs, 5 => ScalarUnaryOp::Floor, 6 => ScalarUnaryOp::Ceil,
            7 => ScalarUnaryOp::BitNot, 8 => ScalarUnaryOp::Sin, 9 => ScalarUnaryOp::Cos,
            10 => ScalarUnaryOp::Tan, 11 => ScalarUnaryOp::Exp, 12 => ScalarUnaryOp::Log,
            13 => ScalarUnaryOp::Log2, 14 => ScalarUnaryOp::Round, 15 => ScalarUnaryOp::Sign,
            t => return Err(format!("unknown unaryop tag {}", t)),
        })
    }

    fn tensor_op(&mut self) -> Result<TensorOp, String> {
        Ok(match self.u8()? {
            1 => TensorOp::Einsum { notation: self.str()? },
            2 => TensorOp::Unary { op: self.str()? },
            3 => TensorOp::Reshape,
            4 => {
                let n = self.u32()? as usize;
                let axes: Result<Vec<_>,_> = (0..n).map(|_| self.u32().map(|v| v as usize)).collect();
                TensorOp::Transpose { axes: axes? }
            }
            5 => {
                let op = self.str()?;
                let n = self.u32()? as usize;
                let axes: Result<Vec<_>,_> = (0..n).map(|_| self.u32().map(|v| v as usize)).collect();
                let keepdims = self.bool()?;
                TensorOp::Reduce { op, axes: axes?, keepdims }
            }
            t => return Err(format!("unknown tensor_op tag {}", t)),
        })
    }

    fn instr(&mut self) -> Result<IrInstr, String> {
        Ok(match self.u8()? {
            OP_BINOP => {
                let result = self.vid()?; let op = self.binop()?;
                let lhs = self.vid()?; let rhs = self.vid()?; let ty = self.ty()?;
                IrInstr::BinOp { result, op, lhs, rhs, ty }
            }
            OP_UNARY => {
                let result = self.vid()?; let op = self.unaryop()?;
                let operand = self.vid()?; let ty = self.ty()?;
                IrInstr::UnaryOp { result, op, operand, ty }
            }
            OP_CONST_FLOAT => {
                let result = self.vid()?; let value = self.f64()?; let ty = self.ty()?;
                IrInstr::ConstFloat { result, value, ty }
            }
            OP_CONST_INT => {
                let result = self.vid()?; let value = self.i64()?; let ty = self.ty()?;
                IrInstr::ConstInt { result, value, ty }
            }
            OP_CONST_BOOL => {
                let result = self.vid()?; let value = self.bool()?;
                IrInstr::ConstBool { result, value }
            }
            OP_CONST_STR => {
                let result = self.vid()?; let value = self.str()?;
                IrInstr::ConstStr { result, value }
            }
            OP_TENSOR_OP => {
                let result = self.vid()?; let op = self.tensor_op()?;
                let inputs = self.vids()?; let result_ty = self.ty()?;
                IrInstr::TensorOp { result, op, inputs, result_ty }
            }
            OP_CAST => {
                let result = self.vid()?; let operand = self.vid()?;
                let from_ty = self.ty()?; let to_ty = self.ty()?;
                IrInstr::Cast { result, operand, from_ty, to_ty }
            }
            OP_LOAD => {
                let result = self.vid()?; let tensor = self.vid()?;
                let indices = self.vids()?; let result_ty = self.ty()?;
                IrInstr::Load { result, tensor, indices, result_ty }
            }
            OP_STORE => {
                let tensor = self.vid()?; let indices = self.vids()?; let value = self.vid()?;
                IrInstr::Store { tensor, indices, value }
            }
            OP_BR => {
                let target = self.bid()?; let args = self.vids()?;
                IrInstr::Br { target, args }
            }
            OP_CONDBR => {
                let cond = self.vid()?;
                let then_block = self.bid()?; let then_args = self.vids()?;
                let else_block = self.bid()?; let else_args = self.vids()?;
                IrInstr::CondBr { cond, then_block, then_args, else_block, else_args }
            }
            OP_RETURN => { let values = self.vids()?; IrInstr::Return { values } }
            OP_CALL => {
                let result = self.opt_vid()?; let callee = self.str()?;
                let args = self.vids()?; let result_ty = self.opt_ty()?;
                IrInstr::Call { result, callee, args, result_ty }
            }
            OP_MAKE_STRUCT => {
                let result = self.vid()?; let fields = self.vids()?; let result_ty = self.ty()?;
                IrInstr::MakeStruct { result, fields, result_ty }
            }
            OP_GET_FIELD => {
                let result = self.vid()?; let base = self.vid()?;
                let field_index = self.u32()? as usize; let result_ty = self.ty()?;
                IrInstr::GetField { result, base, field_index, result_ty }
            }
            OP_MAKE_VARIANT => {
                let result = self.vid()?; let variant_idx = self.u32()? as usize;
                let fields = self.vids()?; let result_ty = self.ty()?;
                IrInstr::MakeVariant { result, variant_idx, fields, result_ty }
            }
            OP_SWITCH_VARIANT => {
                let scrutinee = self.vid()?;
                let n = self.u32()? as usize;
                let mut arms = Vec::with_capacity(n);
                for _ in 0..n { arms.push((self.u32()? as usize, self.bid()?)); }
                let default_block = self.opt_bid()?;
                IrInstr::SwitchVariant { scrutinee, arms, default_block }
            }
            OP_EXTRACT_VARIANT => {
                let result = self.vid()?; let operand = self.vid()?;
                let variant_idx = self.u32()? as usize; let field_idx = self.u32()? as usize;
                let result_ty = self.ty()?;
                IrInstr::ExtractVariantField { result, operand, variant_idx, field_idx, result_ty }
            }
            OP_MAKE_TUPLE => {
                let result = self.vid()?; let elements = self.vids()?; let result_ty = self.ty()?;
                IrInstr::MakeTuple { result, elements, result_ty }
            }
            OP_GET_ELEMENT => {
                let result = self.vid()?; let base = self.vid()?;
                let index = self.u32()? as usize; let result_ty = self.ty()?;
                IrInstr::GetElement { result, base, index, result_ty }
            }
            OP_MAKE_CLOSURE => {
                let result = self.vid()?; let fn_name = self.str()?;
                let captures = self.vids()?; let result_ty = self.ty()?;
                IrInstr::MakeClosure { result, fn_name, captures, result_ty }
            }
            OP_CALL_CLOSURE => {
                let result = self.opt_vid()?; let closure = self.vid()?;
                let args = self.vids()?; let result_ty = self.ty()?;
                IrInstr::CallClosure { result, closure, args, result_ty }
            }
            OP_ALLOC_ARRAY => {
                let result = self.vid()?; let elem_ty = self.ty()?;
                let size = self.u32()? as usize; let init = self.vids()?;
                IrInstr::AllocArray { result, elem_ty, size, init }
            }
            OP_ARRAY_LOAD => {
                let result = self.vid()?; let array = self.vid()?;
                let index = self.vid()?; let elem_ty = self.ty()?;
                IrInstr::ArrayLoad { result, array, index, elem_ty }
            }
            OP_ARRAY_STORE => {
                let array = self.vid()?; let index = self.vid()?; let value = self.vid()?;
                IrInstr::ArrayStore { array, index, value }
            }
            OP_MAKE_SOME => {
                let result = self.vid()?; let value = self.vid()?; let result_ty = self.ty()?;
                IrInstr::MakeSome { result, value, result_ty }
            }
            OP_MAKE_NONE => {
                let result = self.vid()?; let result_ty = self.ty()?;
                IrInstr::MakeNone { result, result_ty }
            }
            OP_IS_SOME => { let result = self.vid()?; let operand = self.vid()?; IrInstr::IsSome { result, operand } }
            OP_OPTION_UNWRAP => {
                let result = self.vid()?; let operand = self.vid()?; let result_ty = self.ty()?;
                IrInstr::OptionUnwrap { result, operand, result_ty }
            }
            OP_MAKE_OK => {
                let result = self.vid()?; let value = self.vid()?; let result_ty = self.ty()?;
                IrInstr::MakeOk { result, value, result_ty }
            }
            OP_MAKE_ERR => {
                let result = self.vid()?; let value = self.vid()?; let result_ty = self.ty()?;
                IrInstr::MakeErr { result, value, result_ty }
            }
            OP_IS_OK => { let result = self.vid()?; let operand = self.vid()?; IrInstr::IsOk { result, operand } }
            OP_RESULT_UNWRAP => {
                let result = self.vid()?; let operand = self.vid()?; let result_ty = self.ty()?;
                IrInstr::ResultUnwrap { result, operand, result_ty }
            }
            OP_RESULT_UNWRAP_ERR => {
                let result = self.vid()?; let operand = self.vid()?; let result_ty = self.ty()?;
                IrInstr::ResultUnwrapErr { result, operand, result_ty }
            }
            OP_CHAN_NEW => { let result = self.vid()?; let elem_ty = self.ty()?; IrInstr::ChanNew { result, elem_ty } }
            OP_CHAN_SEND => { let chan = self.vid()?; let value = self.vid()?; IrInstr::ChanSend { chan, value } }
            OP_CHAN_RECV => {
                let result = self.vid()?; let chan = self.vid()?; let elem_ty = self.ty()?;
                IrInstr::ChanRecv { result, chan, elem_ty }
            }
            OP_SPAWN => { let body_fn = self.str()?; let args = self.vids()?; IrInstr::Spawn { body_fn, args } }
            OP_PAR_FOR => {
                let var = self.vid()?; let start = self.vid()?; let end = self.vid()?;
                let body_fn = self.str()?; let args = self.vids()?;
                IrInstr::ParFor { var, start, end, body_fn, args }
            }
            OP_ATOMIC_NEW => {
                let result = self.vid()?; let value = self.vid()?; let result_ty = self.ty()?;
                IrInstr::AtomicNew { result, value, result_ty }
            }
            OP_ATOMIC_LOAD => {
                let result = self.vid()?; let atomic = self.vid()?; let result_ty = self.ty()?;
                IrInstr::AtomicLoad { result, atomic, result_ty }
            }
            OP_ATOMIC_STORE => {
                let atomic = self.vid()?; let value = self.vid()?;
                IrInstr::AtomicStore { atomic, value }
            }
            OP_ATOMIC_ADD => {
                let result = self.vid()?; let atomic = self.vid()?; let value = self.vid()?; let result_ty = self.ty()?;
                IrInstr::AtomicAdd { result, atomic, value, result_ty }
            }
            OP_MUTEX_NEW => {
                let result = self.vid()?; let value = self.vid()?; let result_ty = self.ty()?;
                IrInstr::MutexNew { result, value, result_ty }
            }
            OP_MUTEX_LOCK => {
                let result = self.vid()?; let mutex = self.vid()?; let result_ty = self.ty()?;
                IrInstr::MutexLock { result, mutex, result_ty }
            }
            OP_MUTEX_UNLOCK => { let mutex = self.vid()?; IrInstr::MutexUnlock { mutex } }
            OP_BARRIER => IrInstr::Barrier,
            OP_MAKE_GRAD => {
                let result = self.vid()?; let value = self.vid()?; let tangent = self.vid()?; let ty = self.ty()?;
                IrInstr::MakeGrad { result, value, tangent, ty }
            }
            OP_GRAD_VALUE   => { let result = self.vid()?; let operand = self.vid()?; let ty = self.ty()?; IrInstr::GradValue { result, operand, ty } }
            OP_GRAD_TANGENT => { let result = self.vid()?; let operand = self.vid()?; let ty = self.ty()?; IrInstr::GradTangent { result, operand, ty } }
            OP_SPARSIFY     => { let result = self.vid()?; let operand = self.vid()?; let ty = self.ty()?; IrInstr::Sparsify { result, operand, ty } }
            OP_DENSIFY      => { let result = self.vid()?; let operand = self.vid()?; let ty = self.ty()?; IrInstr::Densify { result, operand, ty } }
            OP_STR_LEN    => { let result = self.vid()?; let operand = self.vid()?; IrInstr::StrLen { result, operand } }
            OP_STR_CONCAT => { let result = self.vid()?; let lhs = self.vid()?; let rhs = self.vid()?; IrInstr::StrConcat { result, lhs, rhs } }
            OP_PRINT      => { let operand = self.vid()?; IrInstr::Print { operand } }
            OP_STR_CONTAINS    => { let result = self.vid()?; let haystack = self.vid()?; let needle = self.vid()?; IrInstr::StrContains { result, haystack, needle } }
            OP_STR_STARTS_WITH => { let result = self.vid()?; let haystack = self.vid()?; let prefix = self.vid()?; IrInstr::StrStartsWith { result, haystack, prefix } }
            OP_STR_ENDS_WITH   => { let result = self.vid()?; let haystack = self.vid()?; let suffix = self.vid()?; IrInstr::StrEndsWith { result, haystack, suffix } }
            OP_STR_TO_UPPER => { let result = self.vid()?; let operand = self.vid()?; IrInstr::StrToUpper { result, operand } }
            OP_STR_TO_LOWER => { let result = self.vid()?; let operand = self.vid()?; IrInstr::StrToLower { result, operand } }
            OP_STR_TRIM     => { let result = self.vid()?; let operand = self.vid()?; IrInstr::StrTrim { result, operand } }
            OP_STR_REPEAT   => { let result = self.vid()?; let operand = self.vid()?; let count = self.vid()?; IrInstr::StrRepeat { result, operand, count } }
            OP_PANIC        => { let msg = self.vid()?; IrInstr::Panic { msg } }
            OP_VALUE_TO_STR => { let result = self.vid()?; let operand = self.vid()?; IrInstr::ValueToStr { result, operand } }
            OP_READ_LINE    => { let result = self.vid()?; IrInstr::ReadLine { result } }
            OP_READ_I64     => { let result = self.vid()?; IrInstr::ReadI64 { result } }
            OP_READ_F64     => { let result = self.vid()?; IrInstr::ReadF64 { result } }
            OP_PARSE_I64    => { let result = self.vid()?; let operand = self.vid()?; IrInstr::ParseI64 { result, operand } }
            OP_PARSE_F64    => { let result = self.vid()?; let operand = self.vid()?; IrInstr::ParseF64 { result, operand } }
            OP_STR_INDEX    => { let result = self.vid()?; let string = self.vid()?; let index = self.vid()?; IrInstr::StrIndex { result, string, index } }
            OP_STR_SLICE    => { let result = self.vid()?; let string = self.vid()?; let start = self.vid()?; let end = self.vid()?; IrInstr::StrSlice { result, string, start, end } }
            OP_STR_FIND     => { let result = self.vid()?; let haystack = self.vid()?; let needle = self.vid()?; IrInstr::StrFind { result, haystack, needle } }
            OP_STR_REPLACE  => { let result = self.vid()?; let string = self.vid()?; let from = self.vid()?; let to = self.vid()?; IrInstr::StrReplace { result, string, from, to } }
            OP_LIST_NEW  => { let result = self.vid()?; let elem_ty = self.ty()?; IrInstr::ListNew { result, elem_ty } }
            OP_LIST_PUSH => { let list = self.vid()?; let value = self.vid()?; IrInstr::ListPush { list, value } }
            OP_LIST_LEN  => { let result = self.vid()?; let list = self.vid()?; IrInstr::ListLen { result, list } }
            OP_LIST_GET  => { let result = self.vid()?; let list = self.vid()?; let index = self.vid()?; let elem_ty = self.ty()?; IrInstr::ListGet { result, list, index, elem_ty } }
            OP_LIST_SET  => { let list = self.vid()?; let index = self.vid()?; let value = self.vid()?; IrInstr::ListSet { list, index, value } }
            OP_LIST_POP  => { let result = self.vid()?; let list = self.vid()?; let elem_ty = self.ty()?; IrInstr::ListPop { result, list, elem_ty } }
            OP_MAP_NEW      => { let result = self.vid()?; let key_ty = self.ty()?; let val_ty = self.ty()?; IrInstr::MapNew { result, key_ty, val_ty } }
            OP_MAP_SET      => { let map = self.vid()?; let key = self.vid()?; let value = self.vid()?; IrInstr::MapSet { map, key, value } }
            OP_MAP_GET      => { let result = self.vid()?; let map = self.vid()?; let key = self.vid()?; let val_ty = self.ty()?; IrInstr::MapGet { result, map, key, val_ty } }
            OP_MAP_CONTAINS => { let result = self.vid()?; let map = self.vid()?; let key = self.vid()?; IrInstr::MapContains { result, map, key } }
            OP_MAP_REMOVE   => { let map = self.vid()?; let key = self.vid()?; IrInstr::MapRemove { map, key } }
            OP_MAP_LEN      => { let result = self.vid()?; let map = self.vid()?; IrInstr::MapLen { result, map } }
            OP_FILE_READ_ALL  => { let result = self.vid()?; let path = self.vid()?; IrInstr::FileReadAll { result, path } }
            OP_FILE_WRITE_ALL => { let result = self.vid()?; let path = self.vid()?; let content = self.vid()?; IrInstr::FileWriteAll { result, path, content } }
            OP_FILE_EXISTS    => { let result = self.vid()?; let path = self.vid()?; IrInstr::FileExists { result, path } }
            OP_FILE_LINES     => { let result = self.vid()?; let path = self.vid()?; IrInstr::FileLines { result, path } }
            OP_LIST_CONTAINS => { let result = self.vid()?; let list = self.vid()?; let value = self.vid()?; IrInstr::ListContains { result, list, value } }
            OP_LIST_SORT     => { let list = self.vid()?; IrInstr::ListSort { list } }
            OP_MAP_KEYS      => { let result = self.vid()?; let map = self.vid()?; IrInstr::MapKeys { result, map } }
            OP_MAP_VALUES    => { let result = self.vid()?; let map = self.vid()?; IrInstr::MapValues { result, map } }
            OP_LIST_CONCAT   => { let result = self.vid()?; let lhs = self.vid()?; let rhs = self.vid()?; IrInstr::ListConcat { result, lhs, rhs } }
            OP_LIST_SLICE    => { let result = self.vid()?; let list = self.vid()?; let start = self.vid()?; let end = self.vid()?; IrInstr::ListSlice { result, list, start, end } }
            OP_PROCESS_EXIT  => { let code = self.vid()?; IrInstr::ProcessExit { code } }
            OP_PROCESS_ARGS  => { let result = self.vid()?; IrInstr::ProcessArgs { result } }
            OP_ENV_VAR       => { let result = self.vid()?; let name = self.vid()?; IrInstr::EnvVar { result, name } }
            OP_GET_VARIANT_TAG => { let result = self.vid()?; let operand = self.vid()?; IrInstr::GetVariantTag { result, operand } }
            OP_STR_EQ          => { let result = self.vid()?; let lhs = self.vid()?; let rhs = self.vid()?; IrInstr::StrEq { result, lhs, rhs } }
            t => return Err(format!("unknown opcode 0x{:02x}", t)),
        })
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Serializes an `IrModule` to a compact binary representation.
///
/// The output starts with the magic bytes `IRIS` followed by a version byte.
/// Modules with the same logical content will always produce the same bytes
/// (deterministic output).
pub fn serialize_module(module: &IrModule) -> Vec<u8> {
    let mut w = Writer::new();
    w.buf.extend_from_slice(MAGIC);
    w.u8(VERSION);
    w.str(&module.name);
    w.u32(module.functions.len() as u32);
    for func in &module.functions {
        w.str(&func.name);
        w.u32(func.next_value);
        w.u32(func.params.len() as u32);
        for p in &func.params { w.str(&p.name); w.ty(&p.ty); }
        w.ty(&func.return_ty);
        w.u32(func.blocks.len() as u32);
        for block in &func.blocks {
            w.u32(block.id.0);
            w.str(block.name.as_deref().unwrap_or(""));
            w.u32(block.params.len() as u32);
            for bp in &block.params {
                w.vid(bp.id); w.ty(&bp.ty);
                w.str(bp.name.as_deref().unwrap_or(""));
            }
            w.u32(block.instrs.len() as u32);
            for instr in &block.instrs { w.instr(instr); }
        }
    }
    w.buf
}

/// Deserializes an `IrModule` from bytes produced by `serialize_module`.
///
/// Returns `Err` if the data is truncated, has an invalid magic header,
/// or contains an unknown opcode or type tag.
pub fn deserialize_module(data: &[u8]) -> Result<IrModule, String> {
    let mut r = Reader::new(data);

    // magic + version
    if data.len() < 5 { return Err("data too short for header".into()); }
    let magic = &data[0..4];
    if magic != MAGIC { return Err(format!("bad magic {:?}", magic)); }
    r.pos = 4;
    let version = r.u8()?;
    if version != VERSION { return Err(format!("unknown version {}", version)); }

    let module_name = r.str()?;
    let mut module = IrModule::new(module_name);

    let func_count = r.u32()? as usize;
    for _ in 0..func_count {
        let func = deserialize_function(&mut r)?;
        module.add_function(func).map_err(|e| e)?;
    }

    Ok(module)
}

fn deserialize_function(r: &mut Reader) -> Result<IrFunction, String> {
    let name = r.str()?;
    let next_value = r.u32()?;

    let param_count = r.u32()? as usize;
    let mut params = Vec::with_capacity(param_count);
    for _ in 0..param_count {
        let pname = r.str()?;
        let ty = r.ty()?;
        params.push(Param { name: pname, ty });
    }
    let return_ty = r.ty()?;

    let block_count = r.u32()? as usize;
    let mut blocks = Vec::with_capacity(block_count);
    let mut value_defs: HashMap<ValueId, ValueDef> = HashMap::new();
    let mut value_types: HashMap<ValueId, IrType> = HashMap::new();

    // Register entry-block params (function arguments) as value defs.
    for (i, param) in params.iter().enumerate() {
        let vid = ValueId(i as u32);
        value_defs.insert(vid, ValueDef::BlockParam { block: BlockId(0) });
        value_types.insert(vid, param.ty.clone());
    }

    for _ in 0..block_count {
        let block_id = BlockId(r.u32()?);
        let block_name_raw = r.str()?;
        let block_name = if block_name_raw.is_empty() { None } else { Some(block_name_raw) };

        let bp_count = r.u32()? as usize;
        let mut block_params = Vec::with_capacity(bp_count);
        for _ in 0..bp_count {
            let vid = r.vid()?;
            let ty = r.ty()?;
            let pname_raw = r.str()?;
            let pname = if pname_raw.is_empty() { None } else { Some(pname_raw) };
            value_defs.insert(vid, ValueDef::BlockParam { block: block_id });
            value_types.insert(vid, ty.clone());
            block_params.push(BlockParam { id: vid, ty, name: pname });
        }

        let instr_count = r.u32()? as usize;
        let mut instrs = Vec::with_capacity(instr_count);
        for iidx in 0..instr_count {
            let instr = r.instr()?;
            if let Some(result) = instr.result() {
                value_defs.insert(result, ValueDef::InstrResult {
                    block: block_id,
                    instr: InstrId(iidx as u32),
                });
            }
            instrs.push(instr);
        }

        blocks.push(IrBlock { id: block_id, params: block_params, instrs, name: block_name });
    }

    Ok(IrFunction {
        id: FunctionId(0), // reassigned by add_function
        name,
        params,
        return_ty,
        blocks,
        value_defs,
        value_types,
        next_value,
    })
}
