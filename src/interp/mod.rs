//! Tree-walking IR interpreter.
//!
//! Executes an `IrFunction` by walking its SSA instructions and threading
//! values through block parameters at branches.

use std::collections::HashMap;
use std::fmt;

use crate::error::InterpError;
use crate::ir::block::BlockId;
use crate::ir::function::IrFunction;
use crate::ir::instr::{BinOp, IrInstr, ScalarUnaryOp, TensorOp};
use crate::ir::module::IrModule;
use crate::ir::types::{DType, IrType};
use crate::ir::value::ValueId;

/// A runtime value produced or consumed by the interpreter.
#[derive(Debug, Clone, PartialEq)]
pub enum IrValue {
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
    Bool(bool),
    /// Flat tensor: data in row-major order, shape as dimension sizes.
    Tensor(Vec<f32>, Vec<usize>),
    /// Struct value: ordered field values matching the struct definition.
    Struct(Vec<IrValue>),
    /// Enum variant value: tag index (0-indexed).
    Enum(usize),
    /// Tuple value: ordered element values.
    Tuple(Vec<IrValue>),
    /// A UTF-8 string value.
    Str(String),
    /// A fixed-length array of values.
    Array(Vec<IrValue>),
}

impl fmt::Display for IrValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IrValue::F32(x) => write!(f, "{}", x),
            IrValue::F64(x) => write!(f, "{}", x),
            IrValue::I32(n) => write!(f, "{}", n),
            IrValue::I64(n) => write!(f, "{}", n),
            IrValue::Bool(b) => write!(f, "{}", b),
            IrValue::Tensor(data, shape) => {
                write!(
                    f,
                    "tensor<{}>({} elements)",
                    shape
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join("x"),
                    data.len()
                )
            }
            IrValue::Struct(fields) => {
                write!(f, "{{")?;
                for (i, v) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "}}")
            }
            IrValue::Enum(tag) => write!(f, "variant({})", tag),
            IrValue::Tuple(elems) => {
                write!(f, "(")?;
                for (i, v) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, ")")
            }
            IrValue::Str(s) => write!(f, "\"{}\"", s),
            IrValue::Array(elems) => {
                write!(f, "[")?;
                for (i, v) in elems.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
        }
    }
}

/// Evaluates `func` with the given `args`, returning its return values.
///
/// Call instructions that refer to other functions will fail with
/// `InterpError::Unsupported`. Use `eval_function_in_module` if you need
/// cross-function calls.
pub fn eval_function(func: &IrFunction, args: &[IrValue]) -> Result<Vec<IrValue>, InterpError> {
    Interpreter::new(None).run(func, args)
}

/// Like `eval_function` but with access to a full module for cross-function calls.
pub fn eval_function_in_module(
    module: &IrModule,
    func: &IrFunction,
    args: &[IrValue],
) -> Result<Vec<IrValue>, InterpError> {
    Interpreter::new(Some(module)).run(func, args)
}

// ---------------------------------------------------------------------------
// Interpreter state
// ---------------------------------------------------------------------------

struct Interpreter<'m> {
    values: HashMap<ValueId, IrValue>,
    module: Option<&'m IrModule>,
}

impl<'m> Interpreter<'m> {
    fn new(module: Option<&'m IrModule>) -> Self {
        Self {
            values: HashMap::new(),
            module,
        }
    }

    fn run(
        &mut self,
        func: &IrFunction,
        entry_args: &[IrValue],
    ) -> Result<Vec<IrValue>, InterpError> {
        // Bind entry block params to function arguments.
        let entry = func.entry_block();
        for (param, arg) in entry.params.iter().zip(entry_args.iter()) {
            self.values.insert(param.id, arg.clone());
        }

        let mut current = BlockId(0);
        let mut steps = 0usize;
        const MAX_STEPS: usize = 1_000_000;

        'blocks: loop {
            steps += 1;
            if steps > MAX_STEPS {
                return Err(InterpError::Unsupported {
                    detail: "exceeded step limit (infinite loop?)".into(),
                });
            }

            let block = func
                .block(current)
                .ok_or(InterpError::UndefinedValue { id: current.0 })?;

            for instr in &block.instrs {
                match instr {
                    IrInstr::ConstFloat { result, value, ty } => {
                        let v = match ty {
                            IrType::Scalar(DType::F32) => IrValue::F32(*value as f32),
                            IrType::Scalar(DType::F64) => IrValue::F64(*value),
                            _ => {
                                return Err(InterpError::TypeError {
                                    detail: format!("ConstFloat with type {}", ty),
                                })
                            }
                        };
                        self.values.insert(*result, v);
                    }

                    IrInstr::ConstInt { result, value, ty } => {
                        let v = match ty {
                            IrType::Scalar(DType::I32) => IrValue::I32(*value as i32),
                            IrType::Scalar(DType::I64) => IrValue::I64(*value),
                            _ => {
                                return Err(InterpError::TypeError {
                                    detail: format!("ConstInt with type {}", ty),
                                })
                            }
                        };
                        self.values.insert(*result, v);
                    }

                    IrInstr::ConstBool { result, value } => {
                        self.values.insert(*result, IrValue::Bool(*value));
                    }

                    IrInstr::BinOp {
                        result,
                        op,
                        lhs,
                        rhs,
                        ..
                    } => {
                        let lv = self.get(*lhs)?;
                        let rv = self.get(*rhs)?;
                        let res = eval_binop(*op, &lv, &rv)?;
                        self.values.insert(*result, res);
                    }

                    IrInstr::UnaryOp {
                        result,
                        op,
                        operand,
                        ..
                    } => {
                        let v = self.get(*operand)?;
                        let res = eval_unary(*op, &v)?;
                        self.values.insert(*result, res);
                    }

                    IrInstr::Cast {
                        result,
                        operand,
                        to_ty,
                        ..
                    } => {
                        let v = self.get(*operand)?;
                        let res = eval_cast(&v, to_ty)?;
                        self.values.insert(*result, res);
                    }

                    IrInstr::Load {
                        result,
                        tensor,
                        indices,
                        ..
                    } => {
                        let tv = self.get(*tensor)?;
                        let flat = self.compute_flat_index(&tv, indices)?;
                        if let IrValue::Tensor(data, _) = tv {
                            self.values.insert(*result, IrValue::F32(data[flat]));
                        } else {
                            return Err(InterpError::TypeError {
                                detail: "load from non-tensor".into(),
                            });
                        }
                    }

                    IrInstr::Store {
                        tensor,
                        indices,
                        value,
                    } => {
                        let tv = self.get(*tensor)?;
                        let val = self.get(*value)?;
                        let flat = self.compute_flat_index(&tv, indices)?;
                        let val_f32 = to_f32_val(&val)?;
                        if let IrValue::Tensor(mut data, shape) = tv {
                            data[flat] = val_f32;
                            self.values.insert(*tensor, IrValue::Tensor(data, shape));
                        } else {
                            return Err(InterpError::TypeError {
                                detail: "store to non-tensor".into(),
                            });
                        }
                    }

                    IrInstr::TensorOp {
                        result, op, inputs, ..
                    } => match op {
                        TensorOp::Unary { op: unary_op } => {
                            if inputs.len() == 1 {
                                let tv = self.get(inputs[0])?;
                                if let IrValue::Tensor(data, shape) = tv {
                                    let new_data = data
                                        .iter()
                                        .map(|&x| apply_unary_f32(unary_op, x))
                                        .collect();
                                    self.values
                                        .insert(*result, IrValue::Tensor(new_data, shape));
                                } else {
                                    return Err(InterpError::TypeError {
                                        detail: "TensorOp::Unary on non-tensor".into(),
                                    });
                                }
                            } else {
                                return Err(InterpError::Unsupported {
                                    detail: "TensorOp::Unary requires exactly 1 input".into(),
                                });
                            }
                        }
                        TensorOp::Einsum { notation } => {
                            if notation == "mk,kn->mn" && inputs.len() == 2 {
                                let a = self.get(inputs[0])?;
                                let b = self.get(inputs[1])?;
                                if let (
                                    IrValue::Tensor(a_data, a_shape),
                                    IrValue::Tensor(b_data, b_shape),
                                ) = (a, b)
                                {
                                    if a_shape.len() < 2
                                        || b_shape.len() < 2
                                        || a_shape[1] != b_shape[0]
                                    {
                                        return Err(InterpError::TypeError {
                                            detail: "einsum dimension mismatch".into(),
                                        });
                                    }
                                    let (m, k, n) = (a_shape[0], a_shape[1], b_shape[1]);
                                    let mut out = vec![0.0f32; m * n];
                                    for i in 0..m {
                                        for j in 0..n {
                                            for l in 0..k {
                                                out[i * n + j] +=
                                                    a_data[i * k + l] * b_data[l * n + j];
                                            }
                                        }
                                    }
                                    self.values
                                        .insert(*result, IrValue::Tensor(out, vec![m, n]));
                                } else {
                                    return Err(InterpError::TypeError {
                                        detail: "einsum inputs must be tensors".into(),
                                    });
                                }
                            } else {
                                return Err(InterpError::Unsupported {
                                    detail: format!("einsum notation '{}' not supported", notation),
                                });
                            }
                        }
                        _ => {
                            return Err(InterpError::Unsupported {
                                detail: format!("TensorOp {:?}", op),
                            });
                        }
                    },

                    IrInstr::Call {
                        result,
                        callee,
                        args,
                        ..
                    } => {
                        let call_args: Vec<IrValue> = args
                            .iter()
                            .map(|&v| {
                                self.values
                                    .get(&v)
                                    .cloned()
                                    .ok_or(InterpError::UndefinedValue { id: v.0 })
                            })
                            .collect::<Result<Vec<_>, _>>()?;

                        if let Some(module) = self.module {
                            if let Some(callee_func) = module.function_by_name(callee) {
                                let mut sub = Interpreter::new(self.module);
                                let ret = sub.run(callee_func, &call_args)?;
                                if let Some(r) = result {
                                    if let Some(v) = ret.into_iter().next() {
                                        self.values.insert(*r, v);
                                    }
                                }
                            } else {
                                return Err(InterpError::Unsupported {
                                    detail: format!("undefined function '{}'", callee),
                                });
                            }
                        } else {
                            return Err(InterpError::Unsupported {
                                detail: format!("call to '{}' without module context", callee),
                            });
                        }
                    }

                    IrInstr::MakeStruct { result, fields, .. } => {
                        let field_vals: Vec<IrValue> = fields
                            .iter()
                            .map(|&v| self.get(v))
                            .collect::<Result<Vec<_>, _>>()?;
                        self.values.insert(*result, IrValue::Struct(field_vals));
                    }

                    IrInstr::GetField {
                        result,
                        base,
                        field_index,
                        ..
                    } => {
                        let sv = self.get(*base)?;
                        if let IrValue::Struct(fields) = sv {
                            let val = fields.get(*field_index).cloned().ok_or_else(|| {
                                InterpError::Unsupported {
                                    detail: format!(
                                        "field index {} out of bounds for struct with {} fields",
                                        field_index,
                                        fields.len()
                                    ),
                                }
                            })?;
                            self.values.insert(*result, val);
                        } else {
                            return Err(InterpError::TypeError {
                                detail: format!("GetField on non-struct value: {:?}", sv),
                            });
                        }
                    }

                    IrInstr::MakeVariant {
                        result,
                        variant_idx,
                        ..
                    } => {
                        self.values.insert(*result, IrValue::Enum(*variant_idx));
                    }

                    IrInstr::SwitchVariant {
                        scrutinee,
                        arms,
                        default_block,
                    } => {
                        let tag = match self.get(*scrutinee)? {
                            IrValue::Enum(t) => t,
                            other => {
                                return Err(InterpError::TypeError {
                                    detail: format!(
                                        "SwitchVariant scrutinee must be Enum, got {:?}",
                                        other
                                    ),
                                })
                            }
                        };
                        let target = arms
                            .iter()
                            .find(|(idx, _)| *idx == tag)
                            .map(|(_, bb)| *bb)
                            .or(*default_block)
                            .ok_or_else(|| InterpError::Unsupported {
                                detail: format!("SwitchVariant: no arm for tag {}", tag),
                            })?;
                        self.bind_block_params(func, target, &[])?;
                        current = target;
                        continue 'blocks;
                    }

                    IrInstr::MakeTuple {
                        result, elements, ..
                    } => {
                        let elem_vals: Vec<IrValue> = elements
                            .iter()
                            .map(|&v| self.get(v))
                            .collect::<Result<Vec<_>, _>>()?;
                        self.values.insert(*result, IrValue::Tuple(elem_vals));
                    }

                    IrInstr::GetElement {
                        result,
                        base,
                        index,
                        ..
                    } => {
                        let bv = self.get(*base)?;
                        match bv {
                            IrValue::Tuple(elems) => {
                                let val = elems.get(*index).cloned().ok_or_else(|| {
                                    InterpError::Unsupported {
                                        detail: format!(
                                            "tuple index {} out of bounds for {} elements",
                                            index,
                                            elems.len()
                                        ),
                                    }
                                })?;
                                self.values.insert(*result, val);
                            }
                            IrValue::Struct(fields) => {
                                let val = fields.get(*index).cloned().ok_or_else(|| {
                                    InterpError::Unsupported {
                                        detail: format!(
                                            "element index {} out of bounds for struct",
                                            index
                                        ),
                                    }
                                })?;
                                self.values.insert(*result, val);
                            }
                            other => {
                                return Err(InterpError::TypeError {
                                    detail: format!("GetElement on non-tuple value: {:?}", other),
                                });
                            }
                        }
                    }

                    IrInstr::AllocArray { result, init, .. } => {
                        let vals: Vec<IrValue> = init
                            .iter()
                            .map(|&v| self.get(v))
                            .collect::<Result<Vec<_>, _>>()?;
                        self.values.insert(*result, IrValue::Array(vals));
                    }

                    IrInstr::ArrayLoad { result, array, index, .. } => {
                        let arr = self.get(*array)?;
                        let idx = match self.get(*index)? {
                            IrValue::I64(n) => n as usize,
                            IrValue::I32(n) => n as usize,
                            other => {
                                return Err(InterpError::TypeError {
                                    detail: format!("ArrayLoad index must be integer, got {:?}", other),
                                });
                            }
                        };
                        match arr {
                            IrValue::Array(elems) => {
                                let val = elems.get(idx).cloned().ok_or_else(|| {
                                    InterpError::Unsupported {
                                        detail: format!("array index {} out of bounds ({} elements)", idx, elems.len()),
                                    }
                                })?;
                                self.values.insert(*result, val);
                            }
                            other => {
                                return Err(InterpError::TypeError {
                                    detail: format!("ArrayLoad on non-array: {:?}", other),
                                });
                            }
                        }
                    }

                    IrInstr::ArrayStore { array, index, value } => {
                        let arr = self.get(*array)?;
                        let idx = match self.get(*index)? {
                            IrValue::I64(n) => n as usize,
                            IrValue::I32(n) => n as usize,
                            other => {
                                return Err(InterpError::TypeError {
                                    detail: format!("ArrayStore index must be integer, got {:?}", other),
                                });
                            }
                        };
                        let val = self.get(*value)?;
                        match arr {
                            IrValue::Array(mut elems) => {
                                if idx >= elems.len() {
                                    return Err(InterpError::Unsupported {
                                        detail: format!("array index {} out of bounds ({} elements)", idx, elems.len()),
                                    });
                                }
                                elems[idx] = val;
                                self.values.insert(*array, IrValue::Array(elems));
                            }
                            other => {
                                return Err(InterpError::TypeError {
                                    detail: format!("ArrayStore on non-array: {:?}", other),
                                });
                            }
                        }
                    }

                    IrInstr::ConstStr { result, value } => {
                        self.values.insert(*result, IrValue::Str(value.clone()));
                    }

                    IrInstr::StrLen { result, operand } => {
                        let sv = self.get(*operand)?;
                        match sv {
                            IrValue::Str(s) => {
                                self.values.insert(*result, IrValue::I64(s.len() as i64));
                            }
                            other => {
                                return Err(InterpError::TypeError {
                                    detail: format!("StrLen on non-string: {:?}", other),
                                });
                            }
                        }
                    }

                    IrInstr::StrConcat { result, lhs, rhs } => {
                        let lv = self.get(*lhs)?;
                        let rv = self.get(*rhs)?;
                        match (lv, rv) {
                            (IrValue::Str(l), IrValue::Str(r)) => {
                                self.values.insert(*result, IrValue::Str(l + &r));
                            }
                            other => {
                                return Err(InterpError::TypeError {
                                    detail: format!("StrConcat on non-strings: {:?}", other),
                                });
                            }
                        }
                    }

                    IrInstr::Print { operand } => {
                        let v = self.get(*operand)?;
                        println!("{}", v);
                    }

                    IrInstr::Br { target, args } => {
                        self.bind_block_params(func, *target, args)?;
                        current = *target;
                        continue 'blocks;
                    }

                    IrInstr::CondBr {
                        cond,
                        then_block,
                        then_args,
                        else_block,
                        else_args,
                    } => {
                        let b = match self
                            .values
                            .get(cond)
                            .ok_or(InterpError::UndefinedValue { id: cond.0 })?
                        {
                            IrValue::Bool(b) => *b,
                            other => {
                                return Err(InterpError::TypeError {
                                    detail: format!(
                                        "CondBr condition must be bool, got {:?}",
                                        other
                                    ),
                                })
                            }
                        };
                        let (target, br_args) = if b {
                            (then_block, then_args)
                        } else {
                            (else_block, else_args)
                        };
                        self.bind_block_params(func, *target, br_args)?;
                        current = *target;
                        continue 'blocks;
                    }

                    IrInstr::Return { values } => {
                        let results = values
                            .iter()
                            .map(|&v| {
                                self.values
                                    .get(&v)
                                    .cloned()
                                    .ok_or(InterpError::UndefinedValue { id: v.0 })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        return Ok(results);
                    }
                }
            }

            // If we fall through the block without hitting a terminator,
            // something is wrong with the IR (ValidatePass would have caught it).
            return Err(InterpError::Unsupported {
                detail: format!("block {} has no terminator", current),
            });
        }
    }

    /// Looks up a value by ID, returning a clone.
    fn get(&self, id: ValueId) -> Result<IrValue, InterpError> {
        self.values
            .get(&id)
            .cloned()
            .ok_or(InterpError::UndefinedValue { id: id.0 })
    }

    /// Binds a target block's parameters to the provided argument values.
    fn bind_block_params(
        &mut self,
        func: &IrFunction,
        block: BlockId,
        args: &[ValueId],
    ) -> Result<(), InterpError> {
        let block_ref = func
            .block(block)
            .ok_or(InterpError::UndefinedValue { id: block.0 })?;
        let arg_vals: Vec<IrValue> = args
            .iter()
            .map(|&v| {
                self.values
                    .get(&v)
                    .cloned()
                    .ok_or(InterpError::UndefinedValue { id: v.0 })
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (param, val) in block_ref.params.iter().zip(arg_vals.into_iter()) {
            self.values.insert(param.id, val);
        }
        Ok(())
    }

    /// Computes a flat row-major index into a tensor from multi-dimensional indices.
    fn compute_flat_index(&self, tv: &IrValue, indices: &[ValueId]) -> Result<usize, InterpError> {
        if let IrValue::Tensor(data, shape) = tv {
            // Compute row-major strides.
            let mut strides = vec![1usize; shape.len()];
            for i in (0..shape.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            let mut flat = 0usize;
            for (dim_idx, &idx_vid) in indices.iter().enumerate() {
                let idx_v = self
                    .values
                    .get(&idx_vid)
                    .ok_or(InterpError::UndefinedValue { id: idx_vid.0 })?;
                let idx = match idx_v {
                    IrValue::I64(n) => *n,
                    IrValue::I32(n) => *n as i64,
                    _ => {
                        return Err(InterpError::TypeError {
                            detail: "tensor index must be integer".into(),
                        })
                    }
                };
                let dim_size = shape[dim_idx];
                if idx < 0 || idx as usize >= dim_size {
                    return Err(InterpError::IndexOutOfBounds { idx, len: dim_size });
                }
                flat += (idx as usize) * strides[dim_idx];
            }
            if flat >= data.len() {
                return Err(InterpError::IndexOutOfBounds {
                    idx: flat as i64,
                    len: data.len(),
                });
            }
            Ok(flat)
        } else {
            Err(InterpError::TypeError {
                detail: "expected tensor for index computation".into(),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn to_f32_val(v: &IrValue) -> Result<f32, InterpError> {
    match v {
        IrValue::F32(x) => Ok(*x),
        IrValue::F64(x) => Ok(*x as f32),
        IrValue::I32(n) => Ok(*n as f32),
        IrValue::I64(n) => Ok(*n as f32),
        _ => Err(InterpError::TypeError {
            detail: "expected numeric value for store".into(),
        }),
    }
}

fn apply_unary_f32(op: &str, x: f32) -> f32 {
    match op {
        "relu" => x.max(0.0),
        "sigmoid" => 1.0 / (1.0 + (-x).exp()),
        "tanh" => x.tanh(),
        _ => x,
    }
}

fn eval_unary(op: ScalarUnaryOp, v: &IrValue) -> Result<IrValue, InterpError> {
    match (op, v) {
        (ScalarUnaryOp::Neg, IrValue::F32(x)) => Ok(IrValue::F32(-x)),
        (ScalarUnaryOp::Neg, IrValue::F64(x)) => Ok(IrValue::F64(-x)),
        (ScalarUnaryOp::Neg, IrValue::I32(n)) => Ok(IrValue::I32(-n)),
        (ScalarUnaryOp::Neg, IrValue::I64(n)) => Ok(IrValue::I64(-n)),
        (ScalarUnaryOp::Not, IrValue::Bool(b)) => Ok(IrValue::Bool(!b)),
        _ => Err(InterpError::TypeError {
            detail: format!("invalid unary {:?} on {:?}", op, v),
        }),
    }
}

fn eval_cast(v: &IrValue, to_ty: &IrType) -> Result<IrValue, InterpError> {
    match to_ty {
        IrType::Scalar(DType::F32) => match v {
            IrValue::F32(x) => Ok(IrValue::F32(*x)),
            IrValue::F64(x) => Ok(IrValue::F32(*x as f32)),
            IrValue::I32(n) => Ok(IrValue::F32(*n as f32)),
            IrValue::I64(n) => Ok(IrValue::F32(*n as f32)),
            _ => Err(InterpError::TypeError {
                detail: "cannot cast to f32".into(),
            }),
        },
        IrType::Scalar(DType::F64) => match v {
            IrValue::F64(x) => Ok(IrValue::F64(*x)),
            IrValue::F32(x) => Ok(IrValue::F64(*x as f64)),
            IrValue::I32(n) => Ok(IrValue::F64(*n as f64)),
            IrValue::I64(n) => Ok(IrValue::F64(*n as f64)),
            _ => Err(InterpError::TypeError {
                detail: "cannot cast to f64".into(),
            }),
        },
        IrType::Scalar(DType::I32) => match v {
            IrValue::I32(n) => Ok(IrValue::I32(*n)),
            IrValue::I64(n) => Ok(IrValue::I32(*n as i32)),
            IrValue::F32(x) => Ok(IrValue::I32(*x as i32)),
            IrValue::F64(x) => Ok(IrValue::I32(*x as i32)),
            _ => Err(InterpError::TypeError {
                detail: "cannot cast to i32".into(),
            }),
        },
        IrType::Scalar(DType::I64) => match v {
            IrValue::I64(n) => Ok(IrValue::I64(*n)),
            IrValue::I32(n) => Ok(IrValue::I64(*n as i64)),
            IrValue::F32(x) => Ok(IrValue::I64(*x as i64)),
            IrValue::F64(x) => Ok(IrValue::I64(*x as i64)),
            _ => Err(InterpError::TypeError {
                detail: "cannot cast to i64".into(),
            }),
        },
        _ => Err(InterpError::Unsupported {
            detail: format!("cast to {}", to_ty),
        }),
    }
}

fn eval_binop(op: BinOp, lv: &IrValue, rv: &IrValue) -> Result<IrValue, InterpError> {
    use IrValue::*;
    match (op, lv, rv) {
        // F32 arithmetic
        (BinOp::Add, F32(a), F32(b)) => Ok(F32(a + b)),
        (BinOp::Sub, F32(a), F32(b)) => Ok(F32(a - b)),
        (BinOp::Mul, F32(a), F32(b)) => Ok(F32(a * b)),
        (BinOp::Div, F32(a), F32(b)) => Ok(F32(a / b)),
        (BinOp::Mod, F32(a), F32(b)) => Ok(F32(a % b)),
        // F32 comparisons
        (BinOp::CmpEq, F32(a), F32(b)) => Ok(Bool(a == b)),
        (BinOp::CmpNe, F32(a), F32(b)) => Ok(Bool(a != b)),
        (BinOp::CmpLt, F32(a), F32(b)) => Ok(Bool(a < b)),
        (BinOp::CmpLe, F32(a), F32(b)) => Ok(Bool(a <= b)),
        (BinOp::CmpGt, F32(a), F32(b)) => Ok(Bool(a > b)),
        (BinOp::CmpGe, F32(a), F32(b)) => Ok(Bool(a >= b)),
        // F64 arithmetic
        (BinOp::Add, F64(a), F64(b)) => Ok(F64(a + b)),
        (BinOp::Sub, F64(a), F64(b)) => Ok(F64(a - b)),
        (BinOp::Mul, F64(a), F64(b)) => Ok(F64(a * b)),
        (BinOp::Div, F64(a), F64(b)) => Ok(F64(a / b)),
        (BinOp::Mod, F64(a), F64(b)) => Ok(F64(a % b)),
        // F64 comparisons
        (BinOp::CmpEq, F64(a), F64(b)) => Ok(Bool(a == b)),
        (BinOp::CmpNe, F64(a), F64(b)) => Ok(Bool(a != b)),
        (BinOp::CmpLt, F64(a), F64(b)) => Ok(Bool(a < b)),
        (BinOp::CmpLe, F64(a), F64(b)) => Ok(Bool(a <= b)),
        (BinOp::CmpGt, F64(a), F64(b)) => Ok(Bool(a > b)),
        (BinOp::CmpGe, F64(a), F64(b)) => Ok(Bool(a >= b)),
        // I32 arithmetic
        (BinOp::Add, I32(a), I32(b)) => Ok(I32(a.wrapping_add(*b))),
        (BinOp::Sub, I32(a), I32(b)) => Ok(I32(a.wrapping_sub(*b))),
        (BinOp::Mul, I32(a), I32(b)) => Ok(I32(a.wrapping_mul(*b))),
        (BinOp::Div, I32(a), I32(b)) => {
            if *b == 0 {
                return Err(InterpError::DivisionByZero);
            }
            Ok(I32(a.wrapping_div(*b)))
        }
        (BinOp::FloorDiv, I32(a), I32(b)) => {
            if *b == 0 {
                return Err(InterpError::DivisionByZero);
            }
            Ok(I32((*a as f64 / *b as f64).floor() as i32))
        }
        (BinOp::Mod, I32(a), I32(b)) => {
            if *b == 0 {
                return Err(InterpError::DivisionByZero);
            }
            Ok(I32(a.wrapping_rem(*b)))
        }
        // I32 comparisons
        (BinOp::CmpEq, I32(a), I32(b)) => Ok(Bool(a == b)),
        (BinOp::CmpNe, I32(a), I32(b)) => Ok(Bool(a != b)),
        (BinOp::CmpLt, I32(a), I32(b)) => Ok(Bool(a < b)),
        (BinOp::CmpLe, I32(a), I32(b)) => Ok(Bool(a <= b)),
        (BinOp::CmpGt, I32(a), I32(b)) => Ok(Bool(a > b)),
        (BinOp::CmpGe, I32(a), I32(b)) => Ok(Bool(a >= b)),
        // I64 arithmetic
        (BinOp::Add, I64(a), I64(b)) => Ok(I64(a.wrapping_add(*b))),
        (BinOp::Sub, I64(a), I64(b)) => Ok(I64(a.wrapping_sub(*b))),
        (BinOp::Mul, I64(a), I64(b)) => Ok(I64(a.wrapping_mul(*b))),
        (BinOp::Div, I64(a), I64(b)) => {
            if *b == 0 {
                return Err(InterpError::DivisionByZero);
            }
            Ok(I64(a.wrapping_div(*b)))
        }
        (BinOp::FloorDiv, I64(a), I64(b)) => {
            if *b == 0 {
                return Err(InterpError::DivisionByZero);
            }
            Ok(I64((*a as f64 / *b as f64).floor() as i64))
        }
        (BinOp::Mod, I64(a), I64(b)) => {
            if *b == 0 {
                return Err(InterpError::DivisionByZero);
            }
            Ok(I64(a.wrapping_rem(*b)))
        }
        // I64 comparisons
        (BinOp::CmpEq, I64(a), I64(b)) => Ok(Bool(a == b)),
        (BinOp::CmpNe, I64(a), I64(b)) => Ok(Bool(a != b)),
        (BinOp::CmpLt, I64(a), I64(b)) => Ok(Bool(a < b)),
        (BinOp::CmpLe, I64(a), I64(b)) => Ok(Bool(a <= b)),
        (BinOp::CmpGt, I64(a), I64(b)) => Ok(Bool(a > b)),
        (BinOp::CmpGe, I64(a), I64(b)) => Ok(Bool(a >= b)),
        // Bool
        (BinOp::CmpEq, Bool(a), Bool(b)) => Ok(Bool(a == b)),
        (BinOp::CmpNe, Bool(a), Bool(b)) => Ok(Bool(a != b)),
        _ => Err(InterpError::TypeError {
            detail: format!("unsupported binop {:?} on {:?} and {:?}", op, lv, rv),
        }),
    }
}
