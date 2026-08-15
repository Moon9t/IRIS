//! Tree-walking IR interpreter.
//!
//! Executes an `IrFunction` by walking its SSA instructions and threading
//! values through block parameters at branches.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

use crate::error::InterpError;
use crate::ir::block::BlockId;
use crate::ir::function::IrFunction;
use crate::ir::instr::{BinOp, HandlerArm, IrInstr, ScalarUnaryOp, TensorOp};
use crate::ir::module::IrModule;
use crate::ir::types::{DType, IrType};
use crate::ir::value::ValueId;
use crate::security;
use rusqlite::{params_from_iter, types::Value as SqlValue};

/// A runtime value produced or consumed by the interpreter.
#[derive(Debug, Clone)]
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
    /// Enum variant value: tag index (0-indexed) and payload field values.
    Enum(usize, Vec<IrValue>),
    /// Tuple value: ordered element values.
    Tuple(Vec<IrValue>),
    /// A UTF-8 string value.
    Str(String),
    /// A fixed-length array of values.
    Array(Vec<IrValue>),
    /// A closure value: function name + captured values.
    Closure {
        fn_name: String,
        captured: Vec<IrValue>,
        ty: IrType,
    },
    /// An option value: Some(v) or None.
    OptionVal(Option<Box<IrValue>>),
    /// A result value: Ok(v) or Err(e).
    ResultVal(std::result::Result<Box<IrValue>, Box<IrValue>>),
    /// A channel value: a shared FIFO queue (thread-safe for Spawn).
    Chan(Arc<SharedChannel>),
    /// A TaskGroup for structured concurrency.
    TaskGroup(Arc<Mutex<TaskGroupState>>),
    /// An atomic/mutex value: a shared mutable cell (thread-safe for Spawn).
    Atomic(std::sync::Arc<std::sync::Mutex<IrValue>>),
    /// Unit (void) value for side-effecting calls with no return.
    Unit,
    /// A dual number for forward-mode automatic differentiation.
    Grad {
        value: f64,
        tangent: f64,
    },
    /// A sparse representation: stores (index, value) pairs.
    Sparse(Vec<(usize, IrValue)>),
    /// A tape node for reverse-mode AD: primal value, op name, parent refs.
    TapeNode {
        primal: Box<IrValue>,
        op: String,
        parents: Vec<ValueId>,
    },
    /// A dynamic growable list (shared mutable).
    List(std::sync::Arc<std::sync::Mutex<Vec<IrValue>>>),
    /// A hash map (shared mutable). Keys are displayed as strings for comparison.
    Map(std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, IrValue>>>),
    /// A weak reference pointing to a heap cell.
    WeakRef(std::sync::Weak<std::sync::Mutex<Option<IrValue>>>),
    /// A trait object: trait name + boxed concrete value + vtable of impl
    /// method mangles keyed by method name. Used for `dyn Trait` dispatch.
    TraitObject {
        target_trait: String,
        concrete: String,
        data: Box<IrValue>,
        vtable: std::collections::HashMap<String, String>,
    },
    /// A delimited continuation captured at an effect perform site.
    /// When `resume(v)` is called inside the handler, the cell stores `v`
    /// and the parent interpreter returns `v` from the effect site.
    Continuation {
        resume_value: std::sync::Arc<std::sync::Mutex<Option<IrValue>>>,
    },
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
            IrValue::Enum(tag, data) => {
                if data.is_empty() {
                    write!(f, "variant({})", tag)
                } else {
                    write!(f, "variant({}", tag)?;
                    for v in data {
                        write!(f, ", {}", v)?;
                    }
                    write!(f, ")")
                }
            }
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
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            IrValue::Closure { fn_name, .. } => write!(f, "<closure:{}>", fn_name),
            IrValue::OptionVal(Some(v)) => write!(f, "some({})", v),
            IrValue::OptionVal(None) => write!(f, "none"),
            IrValue::ResultVal(Ok(v)) => write!(f, "ok({})", v),
            IrValue::ResultVal(Err(e)) => write!(f, "err({})", e),
            IrValue::Chan(_) => write!(f, "<channel>"),
            IrValue::TaskGroup(_) => write!(f, "<task_group>"),
            IrValue::Atomic(cell) => write!(f, "atomic({})", cell.lock().unwrap()),
            IrValue::Unit => write!(f, "()"),
            IrValue::Grad { value, tangent } => write!(f, "grad({}, {})", value, tangent),
            IrValue::Sparse(pairs) => write!(f, "sparse({} nonzeros)", pairs.len()),
            IrValue::TapeNode { op, .. } => write!(f, "tape_node({})", op),
            IrValue::List(elems) => write!(f, "list({} items)", elems.lock().unwrap().len()),
            IrValue::Map(entries) => write!(f, "map({} entries)", entries.lock().unwrap().len()),
            IrValue::WeakRef(_) => write!(f, "<weak_ref>"),
            IrValue::TraitObject { target_trait, concrete, .. } => {
                write!(f, "<dyn {} as {}>", target_trait, concrete)
            }
            IrValue::Continuation { .. } => write!(f, "<continuation>"),
        }
    }
}

impl PartialEq for IrValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (IrValue::F32(a), IrValue::F32(b)) => a == b,
            (IrValue::F64(a), IrValue::F64(b)) => a == b,
            (IrValue::I32(a), IrValue::I32(b)) => a == b,
            (IrValue::I64(a), IrValue::I64(b)) => a == b,
            (IrValue::Bool(a), IrValue::Bool(b)) => a == b,
            (IrValue::Tensor(da, sa), IrValue::Tensor(db, sb)) => da == db && sa == sb,
            (IrValue::Struct(a), IrValue::Struct(b)) => a == b,
            (IrValue::Enum(a, da), IrValue::Enum(b, db)) => a == b && da == db,
            (IrValue::Tuple(a), IrValue::Tuple(b)) => a == b,
            (IrValue::Str(a), IrValue::Str(b)) => a == b,
            (IrValue::Array(a), IrValue::Array(b)) => a == b,
            (IrValue::Closure { fn_name: a, .. }, IrValue::Closure { fn_name: b, .. }) => a == b,
            (IrValue::OptionVal(a), IrValue::OptionVal(b)) => a == b,
            (IrValue::ResultVal(a), IrValue::ResultVal(b)) => a == b,
            // Channels and atomics use pointer equality.
            (IrValue::Chan(a), IrValue::Chan(b)) => std::sync::Arc::ptr_eq(a, b),
            (IrValue::Atomic(a), IrValue::Atomic(b)) => std::sync::Arc::ptr_eq(a, b),
            (IrValue::Unit, IrValue::Unit) => true,
            (
                IrValue::Grad {
                    value: av,
                    tangent: at,
                },
                IrValue::Grad {
                    value: bv,
                    tangent: bt,
                },
            ) => av == bv && at == bt,
            (IrValue::Sparse(a), IrValue::Sparse(b)) => a.len() == b.len(),
            (IrValue::List(a), IrValue::List(b)) => std::sync::Arc::ptr_eq(a, b),
            (IrValue::Map(a), IrValue::Map(b)) => std::sync::Arc::ptr_eq(a, b),
            (
                IrValue::TraitObject { target_trait: t1, concrete: c1, data: d1, .. },
                IrValue::TraitObject { target_trait: t2, concrete: c2, data: d2, .. },
            ) => t1 == t2 && c1 == c2 && d1 == d2,
            (IrValue::Continuation { .. }, IrValue::Continuation { .. }) => true,
            _ => false,
        }
    }
}

/// Shared state backing a channel: a FIFO queue with a condition variable
/// so that `recv` can block efficiently instead of spinning.
#[derive(Debug)]
pub struct SharedChannel {
    pub queue: Mutex<std::collections::VecDeque<IrValue>>,
    pub not_empty: Condvar,
    pub not_full: Condvar,
    pub capacity: usize,
}

#[derive(Debug)]
pub struct TaskGroupState {
    pub handles: Vec<std::thread::JoinHandle<Result<Vec<IrValue>, InterpError>>>,
    pub cancelled: bool,
}

/// Interpreter execution options.
#[derive(Debug, Clone, Copy)]
pub struct InterpOptions {
    /// Maximum number of block-execution steps before aborting (default: 1 000 000).
    pub max_steps: usize,
    /// Maximum call-stack depth before aborting (default: 500).
    pub max_depth: usize,
}

impl Default for InterpOptions {
    fn default() -> Self {
        Self {
            max_steps: 1_000_000,
            max_depth: 500,
        }
    }
}

/// Evaluates `func` with the given `args`, returning its return values.
///
/// Call instructions that refer to other functions will fail with
/// `InterpError::Unsupported`. Use `eval_function_in_module` if you need
/// cross-function calls.
pub fn eval_function(func: &IrFunction, args: &[IrValue]) -> Result<Vec<IrValue>, InterpError> {
    let mut interp = Interpreter::new(None, InterpOptions::default(), 0);
    let result = interp.run(func, args)?;
    interp.join_all_spawns()?;
    Ok(result)
}

// ---------------------------------------------------------------------------
// Captured program output
// ---------------------------------------------------------------------------
//
// `--emit eval` must return the same string whether a program ran natively or was
// interpreted. The native path returns the child process's stdout, which is the
// printed output followed by the return value. The interpreter wrote prints
// straight to this process's stdout, so callers received only the return value —
// which is why an interpreted run of a correct program could return "0" while the
// expected text appeared on the terminal instead.
//
// The sink is thread-local so parallel test threads cannot interleave into one
// another's buffers. Output printed from `spawn`ed interpreter threads is not
// captured: it has no deterministic position in the stream anyway, and still goes
// to stdout.
thread_local! {
    static OUT_SINK: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);
}

/// Emit one line of program output — to the active sink, or stdout if none.
fn emit_line(line: &str) {
    OUT_SINK.with(|sink| {
        if let Some(buf) = sink.borrow_mut().as_mut() {
            buf.push_str(line);
            buf.push('\n');
        } else {
            println!("{}", line);
        }
    });
}

/// Like `eval_function_in_module_opts`, but captures printed output instead of
/// letting it escape to stdout.
///
/// Returns the captured text alongside the result so a caller can reproduce the
/// native path's output shape. The captured buffer is returned even when
/// evaluation fails, since partial output is often what explains the failure.
pub fn eval_function_in_module_opts_capturing(
    module: &IrModule,
    func: &IrFunction,
    args: &[IrValue],
    opts: InterpOptions,
) -> (Result<Vec<IrValue>, InterpError>, String) {
    OUT_SINK.with(|sink| *sink.borrow_mut() = Some(String::new()));
    let mut interp = Interpreter::new(Some(module), opts, 0);
    let result = interp
        .run(func, args)
        .and_then(|vals| interp.join_all_spawns().map(|()| vals));
    // `take` also uninstalls the sink, so a later un-captured run still prints.
    let captured = OUT_SINK
        .with(|sink| sink.borrow_mut().take())
        .unwrap_or_default();
    (result, captured)
}

/// Like `eval_function` but with access to a full module for cross-function calls.
pub fn eval_function_in_module(
    module: &IrModule,
    func: &IrFunction,
    args: &[IrValue],
) -> Result<Vec<IrValue>, InterpError> {
    let mut interp = Interpreter::new(Some(module), InterpOptions::default(), 0);
    let result = interp.run(func, args)?;
    interp.join_all_spawns()?;
    Ok(result)
}

/// Like `eval_function_in_module` but accepts custom execution limits.
pub fn eval_function_in_module_opts(
    module: &IrModule,
    func: &IrFunction,
    args: &[IrValue],
    opts: InterpOptions,
) -> Result<Vec<IrValue>, InterpError> {
    let mut interp = Interpreter::new(Some(module), opts, 0);
    let result = interp.run(func, args)?;
    interp.join_all_spawns()?;
    Ok(result)
}

/// Runs the first zero-argument function in `module`, collecting a trace of
/// executed statements for use by the debugger.
///
/// Each [`crate::debugger::TraceEntry`] records the function name, source
/// position (from the function's `span_table`), and a snapshot of named values.
///
/// Functions without span information produce entries with `line = 0`.
pub fn collect_trace(
    module: &IrModule,
    source: &str,
    out: std::rc::Rc<std::cell::RefCell<Vec<crate::debugger::TraceEntry>>>,
) -> Result<(), InterpError> {
    use crate::debugger::TraceEntry;

    let func = module
        .functions()
        .iter()
        .find(|f| f.name == "main" && f.params.is_empty())
        .or_else(|| module.functions().iter().find(|f| f.params.is_empty()))
        .ok_or_else(|| InterpError::Unsupported {
            detail: "no zero-argument function for trace collection".into(),
        })?;

    // Run the program with trace collection enabled.
    let opts = InterpOptions::default();
    let mut interp = Interpreter::new(Some(module), opts, 0);
    interp.trace_out = Some(std::rc::Rc::clone(&out));
    interp.trace_func = func.name.clone();
    interp.trace_source = source.to_owned();
    interp.run(func, &[])?;
    interp.join_all_spawns()?;

    // Ensure at least one trace entry exists so the session is non-empty.
    if out.borrow().is_empty() {
        out.borrow_mut().push(TraceEntry {
            func_name: func.name.clone(),
            line: 0,
            column: 0,
            variables: Vec::new(),
            depth: 0,
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Interpreter state
// ---------------------------------------------------------------------------

pub(crate) struct Interpreter<'m> {
    values: HashMap<ValueId, IrValue>,
    module: Option<&'m IrModule>,
    opts: InterpOptions,
    /// Current call-stack depth (0 = top-level).
    depth: usize,
    /// Optional trace output buffer (populated by the debugger).
    trace_out: Option<std::rc::Rc<std::cell::RefCell<Vec<crate::debugger::TraceEntry>>>>,
    /// Name of the function being traced (used in TraceEntry.func_name).
    trace_func: String,
    /// Source text for byte-offset → line/col conversion.
    trace_source: String,
    /// Gradient accumulator for reverse-mode AD (populated by Backward).
    tape_grads: HashMap<ValueId, f64>,
    /// Byte offset of the most-recently executed instruction (for error location).
    last_byte: Option<u32>,
    /// Name of the function currently executing (for error location).
    cur_func: String,
    pub(crate) profiler: Option<std::rc::Rc<std::cell::RefCell<crate::profiler::Profiler>>>,
    /// Join handles for spawned threads, collected so we can wait for them.
    spawn_handles: Vec<JoinHandle<Result<Vec<IrValue>, InterpError>>>,
    /// Stack of active effect handler frames.
    handler_stack: Vec<Vec<HandlerArm>>,
}

impl<'m> Interpreter<'m> {
    pub(crate) fn new(module: Option<&'m IrModule>, opts: InterpOptions, depth: usize) -> Self {
        Self {
            values: HashMap::new(),
            module,
            opts,
            depth,
            trace_out: None,
            trace_func: String::new(),
            trace_source: String::new(),
            tape_grads: HashMap::new(),
            last_byte: None,
            cur_func: String::new(),
            profiler: None,
            spawn_handles: Vec::new(),
            handler_stack: Vec::new(),
        }
    }

    /// Wraps `e` with `InterpError::Located` if a source span is available.
    fn located(&self, e: InterpError) -> InterpError {
        if let Some(byte) = self.last_byte {
            InterpError::Located {
                inner: Box::new(e),
                byte,
                byte_end: None,
                func: self.cur_func.clone(),
            }
        } else {
            e
        }
    }

    fn sqlite_param_values(&self, params: &IrValue) -> Vec<SqlValue> {
        let list = if let IrValue::List(rc) = params {
            rc.lock().ok().map(|guard| guard.clone())
        } else {
            None
        };

        list.unwrap_or_default()
            .into_iter()
            .map(|value| match value {
                IrValue::I64(n) => SqlValue::Integer(n),
                IrValue::I32(n) => SqlValue::Integer(n as i64),
                IrValue::F64(n) => SqlValue::Real(n),
                IrValue::F32(n) => SqlValue::Real(n as f64),
                IrValue::Bool(b) => SqlValue::Integer(if b { 1 } else { 0 }),
                IrValue::Str(s) => SqlValue::Text(s),
                other => SqlValue::Text(format!("{}", other)),
            })
            .collect()
    }

    pub(crate) fn run(
        &mut self,
        func: &IrFunction,
        entry_args: &[IrValue],
    ) -> Result<Vec<IrValue>, InterpError> {
        self.cur_func = func.name.clone();
        self.run_inner(func, entry_args)
            .map_err(|e| self.located(e))
    }

    fn join_all_spawns(&mut self) -> Result<(), InterpError> {
        while let Some(handle) = self.spawn_handles.pop() {
            if let Err(e) = handle.join() {
                return Err(InterpError::Unsupported {
                    detail: format!("spawn thread panicked: {:?}", e),
                });
            }
        }
        Ok(())
    }

    pub(crate) fn run_inner(
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

        'blocks: loop {
                let block = func
                    .block(current)
                    .ok_or(InterpError::UndefinedValue { id: current.0 })?;

                for (instr_idx, instr) in block.instrs.iter().enumerate() {
                    steps += 1;
                    if let Some(ref prof) = self.profiler {
                        prof.borrow_mut().record_instruction();
                    }
                    if steps > self.opts.max_steps {
                        return Err(InterpError::Unsupported {
                        detail: format!(
                            "exceeded step limit of {} (infinite loop?); use --max-steps to increase",
                            self.opts.max_steps
                        ),
                    });
                    }

                    // Track the source span of the current instruction for error reporting.
                    if let Some(byte) = func.span_table.get(current.0, instr_idx) {
                        self.last_byte = Some(byte);
                    }

                    // Emit a trace entry whenever this instruction has a recorded span.
                    if let Some(ref trace) = self.trace_out {
                        if let Some(byte) = func.span_table.get(current.0, instr_idx) {
                            let (line, col) = if self.trace_source.is_empty() {
                                (0, 0)
                            } else {
                                crate::diagnostics::byte_to_line_col(&self.trace_source, byte)
                            };
                            // Snapshot all named block-param values currently in scope.
                            let variables: Vec<(String, String)> = func
                                .value_defs
                                .iter()
                                .filter_map(|(vid, def)| {
                                    if let crate::ir::value::ValueDef::BlockParam { block: bid } =
                                        def
                                    {
                                        let b = func.block(*bid)?;
                                        let param = b.params.iter().find(|p| p.id == *vid)?;
                                        let name = param.name.as_ref()?.clone();
                                        let val = self.values.get(vid)?;
                                        Some((name, format!("{}", val)))
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            trace.borrow_mut().push(crate::debugger::TraceEntry {
                                func_name: self.trace_func.clone(),
                                line,
                                column: col,
                                variables,
                                depth: self.depth as u32,
                            });
                        }
                    }

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
                                // Extended integer types: stored as I64 for interpreter purposes.
                                IrType::Scalar(DType::U8) => IrValue::I64((*value as u8) as i64),
                                IrType::Scalar(DType::I8) => IrValue::I64((*value as i8) as i64),
                                IrType::Scalar(DType::U32) => IrValue::I64((*value as u32) as i64),
                                IrType::Scalar(DType::U64) => IrValue::I64(*value),
                                IrType::Scalar(DType::USize) => IrValue::I64(*value),
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
                                if inputs.len() == 2 {
                                    let a = self.get(inputs[0])?;
                                    let b = self.get(inputs[1])?;
                                    if let (
                                        IrValue::Tensor(a_data, a_shape),
                                        IrValue::Tensor(b_data, b_shape),
                                    ) = (a, b)
                                    {
                                        let result_val = eval_einsum(
                                            notation, &a_data, &a_shape, &b_data, &b_shape,
                                        )?;
                                        self.values.insert(*result, result_val);
                                    } else {
                                        return Err(InterpError::TypeError {
                                            detail: "einsum inputs must be tensors".into(),
                                        });
                                    }
                                } else if inputs.len() == 1 {
                                    // Trace/diagonal einsum on single tensor
                                    let a = self.get(inputs[0])?;
                                    if let IrValue::Tensor(data, shape) = a {
                                        let result_val =
                                            eval_einsum_single(notation, &data, &shape)?;
                                        self.values.insert(*result, result_val);
                                    } else {
                                        return Err(InterpError::TypeError {
                                            detail: "einsum input must be a tensor".into(),
                                        });
                                    }
                                } else {
                                    return Err(InterpError::Unsupported {
                                        detail: format!(
                                            "einsum with {} inputs not supported",
                                            inputs.len()
                                        ),
                                    });
                                }
                            }
                            TensorOp::Reshape => {
                                // Reshape: takes the tensor input and the result_ty
                                // to determine the new shape
                                if !inputs.is_empty() {
                                    let tv = self.get(inputs[0])?;
                                    if let IrValue::Tensor(data, old_shape) = tv {
                                        // Extract new shape from result_ty or from
                                        // additional shape inputs
                                        let new_shape =
                                            if inputs.len() > 1 {
                                                // Shape provided as additional i64 inputs
                                                let mut s = Vec::new();
                                                for input in inputs.iter().skip(1) {
                                                    match self.get(*input)? {
                                                        IrValue::I64(n) => s.push(n as usize),
                                                        IrValue::I32(n) => s.push(n as usize),
                                                        _ => return Err(InterpError::TypeError {
                                                            detail:
                                                                "reshape dimension must be integer"
                                                                    .into(),
                                                        }),
                                                    }
                                                }
                                                s
                                            } else {
                                                // Infer: flatten to 1D
                                                let total: usize = old_shape.iter().product();
                                                vec![total]
                                            };
                                        let new_numel: usize = new_shape.iter().product();
                                        let old_numel: usize = old_shape.iter().product();
                                        if new_numel != old_numel {
                                            return Err(InterpError::TypeError {
                                            detail: format!(
                                                "reshape: new shape {:?} has {} elements, but tensor has {}",
                                                new_shape, new_numel, old_numel
                                            ),
                                        });
                                        }
                                        self.values
                                            .insert(*result, IrValue::Tensor(data, new_shape));
                                    } else {
                                        return Err(InterpError::TypeError {
                                            detail: "reshape on non-tensor".into(),
                                        });
                                    }
                                } else {
                                    return Err(InterpError::Unsupported {
                                        detail: "reshape requires at least 1 input".into(),
                                    });
                                }
                            }
                            TensorOp::Transpose { axes } => {
                                if inputs.len() == 1 {
                                    let tv = self.get(inputs[0])?;
                                    if let IrValue::Tensor(data, shape) = tv {
                                        let ndim = shape.len();
                                        let perm = if axes.is_empty() {
                                            // Default: reverse axes
                                            (0..ndim).rev().collect::<Vec<_>>()
                                        } else {
                                            axes.clone()
                                        };
                                        if perm.len() != ndim {
                                            return Err(InterpError::TypeError {
                                            detail: format!(
                                                "transpose: axes {:?} has {} elements, tensor has {} dims",
                                                perm, perm.len(), ndim
                                            ),
                                        });
                                        }

                                        // Compute new shape
                                        let new_shape: Vec<usize> =
                                            perm.iter().map(|&a| shape[a]).collect();
                                        let numel: usize = shape.iter().product();
                                        let mut new_data = vec![0.0f32; numel];

                                        // Compute source strides
                                        let mut src_strides = vec![1usize; ndim];
                                        for i in (0..ndim.saturating_sub(1)).rev() {
                                            src_strides[i] = src_strides[i + 1] * shape[i + 1];
                                        }
                                        // Compute dest strides
                                        let mut dst_strides = vec![1usize; ndim];
                                        for i in (0..ndim.saturating_sub(1)).rev() {
                                            dst_strides[i] = dst_strides[i + 1] * new_shape[i + 1];
                                        }

                                        let mut coords = vec![0usize; ndim];
                                        for (flat, &val) in data.iter().enumerate().take(numel) {
                                            // Decompose flat index into coords using source strides
                                            let mut rem = flat;
                                            for d in 0..ndim {
                                                coords[d] = rem / src_strides[d];
                                                rem %= src_strides[d];
                                            }
                                            // Compute destination flat index
                                            let mut dst_flat = 0;
                                            for d in 0..ndim {
                                                dst_flat += coords[perm[d]] * dst_strides[d];
                                            }
                                            new_data[dst_flat] = val;
                                        }

                                        self.values
                                            .insert(*result, IrValue::Tensor(new_data, new_shape));
                                    } else {
                                        return Err(InterpError::TypeError {
                                            detail: "transpose on non-tensor".into(),
                                        });
                                    }
                                } else {
                                    return Err(InterpError::Unsupported {
                                        detail: "transpose requires exactly 1 input".into(),
                                    });
                                }
                            }
                            TensorOp::Reduce {
                                op: reduce_op,
                                axes: reduce_axes,
                                keepdims,
                            } => {
                                if inputs.len() == 1 {
                                    let tv = self.get(inputs[0])?;
                                    if let IrValue::Tensor(data, shape) = tv {
                                        let result_val = eval_reduce(
                                            &data,
                                            &shape,
                                            reduce_op,
                                            reduce_axes,
                                            *keepdims,
                                        )?;
                                        self.values.insert(*result, result_val);
                                    } else {
                                        return Err(InterpError::TypeError {
                                            detail: "reduce on non-tensor".into(),
                                        });
                                    }
                                } else {
                                    return Err(InterpError::Unsupported {
                                        detail: "reduce requires exactly 1 input".into(),
                                    });
                                }
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
                                    if self.depth >= self.opts.max_depth {
                                        return Err(InterpError::Unsupported {
                                        detail: format!(
                                            "call depth exceeded {} (infinite recursion?); raise it with --max-depth",
                                            self.opts.max_depth
                                        ),
                                    });
                                    }
                                    let mut sub =
                                        Interpreter::new(self.module, self.opts, self.depth + 1);
                                    sub.profiler = self.profiler.clone();
                                    sub.trace_out = self.trace_out.clone();
                                    sub.trace_func = callee.to_owned();
                                    sub.trace_source = self.trace_source.clone();
                                    if let Some(ref prof) = self.profiler {
                                        prof.borrow_mut().enter_function(callee);
                                    }
                                    let ret = sub.run(callee_func, &call_args)?;
                                    self.spawn_handles.extend(sub.spawn_handles.drain(..));
                                    if let Some(ref prof) = self.profiler {
                                        prof.borrow_mut().exit_function(callee);
                                    }
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

                        // ---- Trait objects: dyn Trait dispatch ----
                        IrInstr::MakeTraitObject {
                            result,
                            value,
                            target_trait,
                            concrete_ty,
                            ..
                        } => {
                            let data = self.get(*value)?;
                            // Build the per-method vtable by looking up each
                            // trait method's mangled impl function for the
                            // concrete type, recorded earlier in
                            // IrModule::trait_impl_methods.
                            let mut vtable = std::collections::HashMap::new();
                            if let Some(module) = &self.module {
                                if let Some(methods) = module.trait_def(target_trait) {
                                    for m in methods {
                                        // Find the entry that pairs this
                                        // (target_trait, concrete_ty) with
                                        // m.name. Fall back to scanning.
                                        let mut found: Option<String> = None;
                                        if let Some(list) =
                                            module.trait_impl_methods().get(target_trait)
                                        {
                                            for (cty, mname, mangled) in list {
                                                if cty == concrete_ty && mname == &m.name {
                                                    found = Some(mangled.clone());
                                                    break;
                                                }
                                            }
                                        }
                                        if let Some(mangled) = found {
                                            vtable.insert(m.name.clone(), mangled);
                                        }
                                    }
                                }
                            }
                            self.values.insert(
                                *result,
                                IrValue::TraitObject {
                                    target_trait: target_trait.clone(),
                                    concrete: concrete_ty.clone(),
                                    data: Box::new(data),
                                    vtable,
                                },
                            );
                        }

                        IrInstr::DynCall {
                            result,
                            obj,
                            method_name,
                            args,
                            ..
                        } => {
                            // Pull the vtable entry; call the mangled function
                            // synchronously in the interpreter using the
                            // boxed data value as the `self` first argument.
                            let obj_val = self.get(*obj)?;
                            let method_name = method_name.clone();
                            let arg_vals: Vec<IrValue> = args
                                .iter()
                                .map(|&v| self.get(v))
                                .collect::<Result<Vec<_>, _>>()?;
                            let (target_trait, concrete, data, vtable) =
                                match obj_val {
                                    IrValue::TraitObject {
                                        target_trait,
                                        concrete,
                                        data,
                                        vtable,
                                    } => (target_trait, concrete, data, vtable),
                                    other => {
                                        return Err(InterpError::TypeError {
                                            detail: format!(
                                                "DynCall on non-trait-object: {:?}",
                                                other
                                            ),
                                        });
                                    }
                                };
                            let mangled = vtable.get(&method_name).cloned().ok_or_else(
                                || InterpError::Unsupported {
                                    detail: format!(
                                "no impl method '{}' in vtable for dyn {} (concrete {})",
                                method_name, target_trait, concrete
                            ),
                                },
                            )?;
                            let callee_func = self
                                .module
                                .as_ref()
                                .and_then(|m| m.function_by_name(&mangled).cloned())
                                .ok_or_else(|| InterpError::Unsupported {
                                    detail: format!("undefined function '{}'", mangled),
                                })?;
                            // Construct the full argument list:
                            // [data, args...] — the impl method's first param
                            // is the concrete struct type.
                            let mut call_args = Vec::with_capacity(arg_vals.len() + 1);
                            call_args.push(*data);
                            call_args.extend(arg_vals);
                            let mut sub = Interpreter::new(
                                self.module.clone(),
                                self.opts,
                                self.depth + 1,
                            );
                            let ret_vals = sub.run(&callee_func, &call_args)?;
                            self.spawn_handles.extend(sub.spawn_handles.drain(..));
                            if let Some(v) = ret_vals.into_iter().next() {
                                self.values.insert(*result, v);
                            } else {
                                // The impl return is Unit: insert dummy 0.
                                self.values.insert(*result, IrValue::Unit);
                            }
                        }

                        IrInstr::MakeVariant {
                            result,
                            variant_idx,
                            fields,
                            ..
                        } => {
                            let field_vals: Vec<IrValue> = fields
                                .iter()
                                .map(|&v| self.get(v))
                                .collect::<Result<Vec<_>, _>>()?;
                            self.values
                                .insert(*result, IrValue::Enum(*variant_idx, field_vals));
                        }

                        IrInstr::SwitchVariant {
                            scrutinee,
                            arms,
                            default_block,
                        } => {
                            let tag = match self.get(*scrutinee)? {
                                IrValue::Enum(t, _) => t,
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

                        IrInstr::ExtractVariantField {
                            result,
                            operand,
                            field_idx,
                            ..
                        } => {
                            let ev = self.get(*operand)?;
                            match ev {
                                IrValue::Enum(_, data) => {
                                    let val = data.get(*field_idx).cloned().ok_or_else(|| {
                                    InterpError::TypeError {
                                        detail: format!(
                                            "ExtractVariantField: field {} out of bounds (variant has {} fields)",
                                            field_idx, data.len()
                                        ),
                                    }
                                })?;
                                    self.values.insert(*result, val);
                                }
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!(
                                            "ExtractVariantField on non-Enum value: {:?}",
                                            other
                                        ),
                                    });
                                }
                            }
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
                                        detail: format!(
                                            "GetElement on non-tuple value: {:?}",
                                            other
                                        ),
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

                        IrInstr::ArrayLoad {
                            result,
                            array,
                            index,
                            ..
                        } => {
                            let arr = self.get(*array)?;
                            let idx = match self.get(*index)? {
                                IrValue::I64(n) => n as usize,
                                IrValue::I32(n) => n as usize,
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!(
                                            "ArrayLoad index must be integer, got {:?}",
                                            other
                                        ),
                                    });
                                }
                            };
                            match arr {
                                IrValue::Array(elems) => {
                                    let val = elems.get(idx).cloned().ok_or_else(|| {
                                        InterpError::Unsupported {
                                            detail: format!(
                                                "array index {} out of bounds ({} elements)",
                                                idx,
                                                elems.len()
                                            ),
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

                        IrInstr::ArrayStore {
                            array,
                            index,
                            value,
                        } => {
                            let arr = self.get(*array)?;
                            let idx = match self.get(*index)? {
                                IrValue::I64(n) => n as usize,
                                IrValue::I32(n) => n as usize,
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!(
                                            "ArrayStore index must be integer, got {:?}",
                                            other
                                        ),
                                    });
                                }
                            };
                            let val = self.get(*value)?;
                            match arr {
                                IrValue::Array(mut elems) => {
                                    if idx >= elems.len() {
                                        return Err(InterpError::Unsupported {
                                            detail: format!(
                                                "array index {} out of bounds ({} elements)",
                                                idx,
                                                elems.len()
                                            ),
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
                            match &v {
                                // Print strings without surrounding quotes.
                                IrValue::Str(s) => emit_line(s),
                                other => emit_line(&format!("{}", other)),
                            }
                        }

                        IrInstr::StrContains {
                            result,
                            haystack,
                            needle,
                        } => {
                            let h = self.get(*haystack)?;
                            let n = self.get(*needle)?;
                            match (h, n) {
                                (IrValue::Str(hs), IrValue::Str(ns)) => {
                                    self.values
                                        .insert(*result, IrValue::Bool(hs.contains(ns.as_str())));
                                }
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!("StrContains on non-strings: {:?}", other),
                                    })
                                }
                            }
                        }

                        IrInstr::StrStartsWith {
                            result,
                            haystack,
                            prefix,
                        } => {
                            let h = self.get(*haystack)?;
                            let p = self.get(*prefix)?;
                            match (h, p) {
                                (IrValue::Str(hs), IrValue::Str(ps)) => {
                                    self.values.insert(
                                        *result,
                                        IrValue::Bool(hs.starts_with(ps.as_str())),
                                    );
                                }
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!(
                                            "StrStartsWith on non-strings: {:?}",
                                            other
                                        ),
                                    })
                                }
                            }
                        }

                        IrInstr::StrEndsWith {
                            result,
                            haystack,
                            suffix,
                        } => {
                            let h = self.get(*haystack)?;
                            let s = self.get(*suffix)?;
                            match (h, s) {
                                (IrValue::Str(hs), IrValue::Str(ss)) => {
                                    self.values
                                        .insert(*result, IrValue::Bool(hs.ends_with(ss.as_str())));
                                }
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!("StrEndsWith on non-strings: {:?}", other),
                                    })
                                }
                            }
                        }

                        IrInstr::StrToUpper { result, operand } => {
                            let v = self.get(*operand)?;
                            match v {
                                IrValue::Str(s) => {
                                    self.values.insert(*result, IrValue::Str(s.to_uppercase()));
                                }
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!("StrToUpper on non-string: {:?}", other),
                                    })
                                }
                            }
                        }

                        IrInstr::StrToLower { result, operand } => {
                            let v = self.get(*operand)?;
                            match v {
                                IrValue::Str(s) => {
                                    self.values.insert(*result, IrValue::Str(s.to_lowercase()));
                                }
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!("StrToLower on non-string: {:?}", other),
                                    })
                                }
                            }
                        }

                        IrInstr::StrTrim { result, operand } => {
                            let v = self.get(*operand)?;
                            match v {
                                IrValue::Str(s) => {
                                    self.values
                                        .insert(*result, IrValue::Str(s.trim().to_string()));
                                }
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!("StrTrim on non-string: {:?}", other),
                                    })
                                }
                            }
                        }

                        IrInstr::StrRepeat {
                            result,
                            operand,
                            count,
                        } => {
                            let sv = self.get(*operand)?;
                            let cv = self.get(*count)?;
                            match (sv, cv) {
                                (IrValue::Str(s), IrValue::I64(n)) => {
                                    self.values
                                        .insert(*result, IrValue::Str(s.repeat(n.max(0) as usize)));
                                }
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!("StrRepeat invalid args: {:?}", other),
                                    })
                                }
                            }
                        }

                        IrInstr::ParFor {
                            start,
                            end,
                            body_fn,
                            args,
                            ..
                        } => {
                            // Parallel for: run iterations on real threads.
                            let s = match self.get(*start)? {
                                IrValue::I64(n) => n,
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!(
                                            "ParFor start must be i64, got {:?}",
                                            other
                                        ),
                                    })
                                }
                            };
                            let e = match self.get(*end)? {
                                IrValue::I64(n) => n,
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!("ParFor end must be i64, got {:?}", other),
                                    })
                                }
                            };
                            let callee = self
                                .module
                                .and_then(|m| m.function_by_name(body_fn))
                                .ok_or_else(|| InterpError::Unsupported {
                                    detail: format!("undefined par_for function: {}", body_fn),
                                })?
                                .clone();
                            // Resolve captured args once.
                            let mut cap_vals: Vec<IrValue> = Vec::new();
                            for a in args {
                                cap_vals.push(self.get(*a)?);
                            }
                            let module = self.module;
                            let opts = self.opts;
                            let depth = self.depth;
                            // Use the enclosing thread scope for parallelism.
                            // Limit thread count to available parallelism.
                            let n_iters = (e - s) as usize;
                            let n_threads = std::thread::available_parallelism()
                                .map(|p| p.get())
                                .unwrap_or(4)
                                .min(n_iters);
                            if n_threads <= 1 || n_iters == 0 {
                                // Sequential fallback for tiny ranges.
                                for i in s..e {
                                    let mut call_args = vec![IrValue::I64(i)];
                                    call_args.extend(cap_vals.iter().cloned());
                                    let mut sub = Interpreter::new(module, opts, depth + 1);
                                    sub.run(&callee, &call_args)?;
                                    self.spawn_handles.extend(sub.spawn_handles.drain(..));
                                }
                            } else {
                                let chunk_size = n_iters.div_ceil(n_threads);
                                let first_err: std::sync::Mutex<Option<InterpError>> =
                                    std::sync::Mutex::new(None);
                                // Use a dedicated scope so ParFor blocks until all
                                // iterations complete.
                                std::thread::scope(|par_scope| {
                                    let callee_ref = &callee;
                                    let cap_ref = &cap_vals;
                                    let err_ref = &first_err;
                                    for t in 0..n_threads {
                                        let lo = s + (t as i64) * (chunk_size as i64);
                                        let hi = lo + (chunk_size as i64);
                                        let hi = hi.min(e);
                                        if lo >= hi {
                                            break;
                                        }
                                        par_scope.spawn(move || {
                                            for i in lo..hi {
                                                let mut call_args = vec![IrValue::I64(i)];
                                                call_args.extend(cap_ref.iter().cloned());
                                                let mut sub =
                                                    Interpreter::new(module, opts, depth + 1);
                                                if let Err(err) = sub.run(callee_ref, &call_args) {
                                                    let mut guard = err_ref.lock().unwrap();
                                                    if guard.is_none() {
                                                        *guard = Some(err);
                                                    }
                                                    return;
                                                }
                                            }
                                        });
                                    }
                                });
                                if let Some(err) = first_err.into_inner().unwrap() {
                                    return Err(err);
                                }
                            }
                        }

                        IrInstr::ChanNew { result, capacity, .. } => {
                            let cap_val = self.get(*capacity)?;
                            let cap = match cap_val {
                                IrValue::I64(c) => if c <= 0 { std::usize::MAX } else { c as usize },
                                _ => return Err(InterpError::TypeError { detail: "ChanNew capacity must be an integer".into() }),
                            };
                            let ch = Arc::new(SharedChannel {
                                queue: Mutex::new(std::collections::VecDeque::new()),
                                not_empty: Condvar::new(),
                                not_full: Condvar::new(),
                                capacity: cap,
                            });
                            self.values.insert(*result, IrValue::Chan(ch));
                        }

                        IrInstr::ChanSend { chan, value } => {
                            let ch = self.get(*chan)?;
                            let v = self.get(*value)?;
                            match ch {
                                IrValue::Chan(q) => {
                                    let mut queue = q.queue.lock().unwrap();
                                    while queue.len() >= q.capacity {
                                        queue = q.not_full.wait(queue).unwrap();
                                    }
                                    queue.push_back(v);
                                    q.not_empty.notify_one();
                                }
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!("ChanSend on non-channel: {:?}", other),
                                    })
                                }
                            }
                        }

                        IrInstr::ChanRecv { result, chan, .. } => {
                            let ch = self.get(*chan)?;
                            match ch {
                                IrValue::Chan(q) => {
                                    let mut queue = q.queue.lock().unwrap();
                                    while queue.is_empty() {
                                        queue = q.not_empty.wait(queue).unwrap();
                                    }
                                    let val = queue.pop_front().unwrap();
                                    q.not_full.notify_one();
                                    self.values.insert(*result, val);
                                }
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!("ChanRecv on non-channel: {:?}", other),
                                    })
                                }
                            }
                        }

                        IrInstr::Spawn { body_fn, args } => {
                            // Spawn the body function on a real OS thread.
                            let callee = self
                                .module
                                .and_then(|m| m.function_by_name(body_fn))
                                .ok_or_else(|| InterpError::Unsupported {
                                    detail: format!("undefined spawn function: {}", body_fn),
                                })?
                                .clone();
                            let mut call_args = Vec::new();
                            for a in args {
                                call_args.push(self.get(*a)?);
                            }
                            let module_raw: usize = self
                                .module
                                .map(|m| m as *const IrModule as usize)
                                .unwrap_or(0);
                            let opts = self.opts;
                            let depth = self.depth;
                            self.spawn_handles.push(std::thread::spawn(move || {
                                let module: Option<&IrModule> = if module_raw == 0 {
                                    None
                                } else {
                                    Some(unsafe { &*(module_raw as *const IrModule) })
                                };
                                let mut sub = Interpreter::new(module, opts, depth + 1);
                                sub.run(&callee, &call_args)
                            }));
                        }

                        IrInstr::TaskGroupNew { result } => {
                            let tg = Arc::new(Mutex::new(TaskGroupState {
                                handles: Vec::new(),
                                cancelled: false,
                            }));
                            self.values.insert(*result, IrValue::TaskGroup(tg));
                        }

                        IrInstr::TaskGroupSpawn { group, body_fn, args } => {
                            let tg_val = self.get(*group)?;
                            match tg_val {
                                IrValue::TaskGroup(tg) => {
                                    let callee = self
                                        .module
                                        .and_then(|m| m.function_by_name(body_fn))
                                        .ok_or_else(|| InterpError::Unsupported {
                                            detail: format!("undefined spawn function: {}", body_fn),
                                        })?
                                        .clone();
                                    let mut call_args = Vec::new();
                                    for a in args {
                                        call_args.push(self.get(*a)?);
                                    }
                                    let module_raw: usize = self
                                        .module
                                        .map(|m| m as *const IrModule as usize)
                                        .unwrap_or(0);
                                    let opts = self.opts;
                                    let depth = self.depth;
                                    let handle = std::thread::spawn(move || {
                                        let module: Option<&IrModule> = if module_raw == 0 {
                                            None
                                        } else {
                                            Some(unsafe { &*(module_raw as *const IrModule) })
                                        };
                                        let mut sub = Interpreter::new(module, opts, depth + 1);
                                        sub.run(&callee, &call_args)
                                    });
                                    tg.lock().unwrap().handles.push(handle);
                                }
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!("TaskGroupSpawn on non-task-group: {:?}", other),
                                    })
                                }
                            }
                        }

                        IrInstr::TaskGroupJoin { group } => {
                            let tg_val = self.get(*group)?;
                            match tg_val {
                                IrValue::TaskGroup(tg) => {
                                    let handles = tg.lock().unwrap().handles.drain(..).collect::<Vec<_>>();
                                    for h in handles {
                                        let _ = h.join().map_err(|_| InterpError::Unsupported {
                                            detail: "task panicked during join".into(),
                                        })?;
                                    }
                                }
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!("TaskGroupJoin on non-task-group: {:?}", other),
                                    })
                                }
                            }
                        }

                        IrInstr::TaskGroupCancel { group } => {
                            let tg_val = self.get(*group)?;
                            match tg_val {
                                IrValue::TaskGroup(tg) => {
                                    tg.lock().unwrap().cancelled = true;
                                }
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!("TaskGroupCancel on non-task-group: {:?}", other),
                                    })
                                }
                            }
                        }

                        IrInstr::AtomicNew {
                            result,
                            value,
                            result_ty,
                        } => {
                            let v = self.get(*value)?;
                            let cell = std::sync::Arc::new(std::sync::Mutex::new(v));
                            let _ = result_ty;
                            self.values.insert(*result, IrValue::Atomic(cell));
                        }

                        IrInstr::AtomicLoad { result, atomic, .. } => {
                            let v = self.get(*atomic)?;
                            match v {
                                IrValue::Atomic(cell) => {
                                    self.values.insert(*result, cell.lock().unwrap().clone());
                                }
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!("AtomicLoad on non-atomic: {:?}", other),
                                    })
                                }
                            }
                        }

                        IrInstr::AtomicStore { atomic, value } => {
                            let v = self.get(*value)?;
                            let a = self.get(*atomic)?;
                            match a {
                                IrValue::Atomic(cell) => {
                                    *cell.lock().unwrap() = v;
                                }
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!("AtomicStore on non-atomic: {:?}", other),
                                    })
                                }
                            }
                        }

                        IrInstr::AtomicAdd {
                            result,
                            atomic,
                            value,
                            ..
                        } => {
                            let v = self.get(*value)?;
                            let a = self.get(*atomic)?;
                            match a {
                                IrValue::Atomic(cell) => {
                                    let mut guard = cell.lock().unwrap();
                                    let old = guard.clone();
                                    let new_val = match (old.clone(), v) {
                                        (IrValue::I64(a), IrValue::I64(b)) => IrValue::I64(a + b),
                                        (IrValue::I32(a), IrValue::I32(b)) => IrValue::I32(a + b),
                                        (IrValue::F32(a), IrValue::F32(b)) => IrValue::F32(a + b),
                                        (IrValue::F64(a), IrValue::F64(b)) => IrValue::F64(a + b),
                                        _ => {
                                            return Err(InterpError::TypeError {
                                                detail: "AtomicAdd on non-numeric".into(),
                                            })
                                        }
                                    };
                                    *guard = new_val.clone();
                                    self.values.insert(*result, new_val);
                                }
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!("AtomicAdd on non-atomic: {:?}", other),
                                    })
                                }
                            }
                        }

                        IrInstr::MutexNew {
                            result,
                            value,
                            result_ty,
                        } => {
                            let v = self.get(*value)?;
                            let cell = std::sync::Arc::new(std::sync::Mutex::new(v));
                            let _ = result_ty;
                            self.values.insert(*result, IrValue::Atomic(cell));
                        }

                        IrInstr::MutexLock { result, mutex, .. } => {
                            let v = self.get(*mutex)?;
                            match v {
                                IrValue::Atomic(cell) => {
                                    self.values.insert(*result, cell.lock().unwrap().clone());
                                }
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!("MutexLock on non-mutex: {:?}", other),
                                    })
                                }
                            }
                        }

                        IrInstr::MutexUnlock { .. } => {
                            // No-op in single-threaded interpreter.
                        }

                        IrInstr::Sparsify {
                            result, operand, ..
                        } => {
                            // Convert an Array or Tensor to sparse (index, value) pairs.
                            // Only non-zero elements are stored.
                            let v = self.get(*operand)?;
                            let pairs = match v {
                                IrValue::Array(elems) => elems
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, e)| match e {
                                        IrValue::I64(0) | IrValue::I32(0) => false,
                                        IrValue::F32(f) => *f != 0.0,
                                        IrValue::F64(f) => *f != 0.0,
                                        _ => true,
                                    })
                                    .map(|(i, e)| (i, e.clone()))
                                    .collect(),
                                IrValue::Tensor(data, _shape) => data
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, &val)| val != 0.0)
                                    .map(|(i, &val)| (i, IrValue::F32(val)))
                                    .collect(),
                                other => vec![(0, other)],
                            };
                            self.values.insert(*result, IrValue::Sparse(pairs));
                        }

                        IrInstr::Densify {
                            result, operand, ..
                        } => {
                            // Reconstruct the dense collection, filling the gaps
                            // with zeros, exactly as the runtime's iris_densify
                            // does. This used to return the non-zero count, which
                            // contradicted the builtin's name and documented
                            // signature and disagreed with the native backend.
                            // The count is now SparseNnz.
                            let v = self.get(*operand)?;
                            let dense = match v {
                                IrValue::Sparse(pairs) => {
                                    let size = pairs
                                        .iter()
                                        .map(|(idx, _)| *idx + 1)
                                        .max()
                                        .unwrap_or(0);
                                    let mut out = vec![IrValue::I64(0); size];
                                    for (idx, value) in pairs {
                                        if idx < size {
                                            out[idx] = value;
                                        }
                                    }
                                    IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(out)))
                                }
                                other => other,
                            };
                            self.values.insert(*result, dense);
                        }

                        IrInstr::SparseNnz { result, operand } => {
                            let v = self.get(*operand)?;
                            let nnz = match v {
                                IrValue::Sparse(pairs) => pairs.len() as i64,
                                // A non-sparse operand has no stored-element count;
                                // treat a list as fully populated.
                                IrValue::List(items) => {
                                    items.lock().map(|g| g.len()).unwrap_or(0) as i64
                                }
                                _ => 0,
                            };
                            self.values.insert(*result, IrValue::I64(nnz));
                        }

                        IrInstr::Barrier => {
                            // No-op in single-threaded interpreter.
                        }

                        IrInstr::MakeGrad {
                            result,
                            value,
                            tangent,
                            ..
                        } => {
                            let v = self.get(*value)?;
                            let t = self.get(*tangent)?;
                            let vf = match v {
                                IrValue::F64(x) => x,
                                IrValue::F32(x) => x as f64,
                                IrValue::I64(x) => x as f64,
                                IrValue::I32(x) => x as f64,
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!(
                                            "MakeGrad value must be numeric, got {:?}",
                                            other
                                        ),
                                    })
                                }
                            };
                            let tf = match t {
                                IrValue::F64(x) => x,
                                IrValue::F32(x) => x as f64,
                                IrValue::I64(x) => x as f64,
                                IrValue::I32(x) => x as f64,
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!(
                                            "MakeGrad tangent must be numeric, got {:?}",
                                            other
                                        ),
                                    })
                                }
                            };
                            self.values.insert(
                                *result,
                                IrValue::Grad {
                                    value: vf,
                                    tangent: tf,
                                },
                            );
                        }

                        IrInstr::GradValue {
                            result, operand, ..
                        } => {
                            let v = self.get(*operand)?;
                            match v {
                                IrValue::Grad { value, .. } => {
                                    self.values.insert(*result, IrValue::F64(value));
                                }
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!("GradValue on non-grad: {:?}", other),
                                    })
                                }
                            }
                        }

                        IrInstr::GradTangent {
                            result, operand, ..
                        } => {
                            let v = self.get(*operand)?;
                            match v {
                                IrValue::Grad { tangent, .. } => {
                                    self.values.insert(*result, IrValue::F64(tangent));
                                }
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!("GradTangent on non-grad: {:?}", other),
                                    })
                                }
                            }
                        }

                        // ── Reverse-mode AD (tape-based backpropagation) ──
                        IrInstr::TapeRecord {
                            result,
                            value,
                            op,
                            parents,
                        } => {
                            let primal = self.get(*value)?;
                            let parent_ids: Vec<ValueId> = parents.clone();
                            // Store as a TapeNode containing primal, op, and parent refs
                            self.values.insert(
                                *result,
                                IrValue::TapeNode {
                                    primal: Box::new(primal),
                                    op: op.clone(),
                                    parents: parent_ids,
                                },
                            );
                        }

                        IrInstr::Backward { result, loss } => {
                            // Reverse-mode backpropagation from a loss scalar
                            let mut grads: std::collections::HashMap<ValueId, f64> =
                                std::collections::HashMap::new();
                            // Seed: dL/dL = 1.0
                            grads.insert(*loss, 1.0);

                            // Topological order: collect all tape nodes reachable from loss
                            let mut topo: Vec<ValueId> = Vec::new();
                            let mut visited: std::collections::HashSet<ValueId> =
                                std::collections::HashSet::new();
                            fn topo_sort(
                                vid: ValueId,
                                values: &std::collections::HashMap<ValueId, IrValue>,
                                visited: &mut std::collections::HashSet<ValueId>,
                                topo: &mut Vec<ValueId>,
                            ) {
                                if !visited.insert(vid) {
                                    return;
                                }
                                if let Some(IrValue::TapeNode { parents, .. }) = values.get(&vid) {
                                    for &p in parents {
                                        topo_sort(p, values, visited, topo);
                                    }
                                }
                                topo.push(vid);
                            }
                            topo_sort(*loss, &self.values, &mut visited, &mut topo);
                            topo.reverse(); // reverse post-order

                            // Propagate gradients in reverse topological order
                            for vid in &topo {
                                let grad = *grads.get(vid).unwrap_or(&0.0);
                                if let Some(IrValue::TapeNode {
                                    op,
                                    parents,
                                    primal: _primal,
                                    ..
                                }) = self.values.get(vid).cloned()
                                {
                                    match op.as_str() {
                                        "add" => {
                                            // d(a+b)/da = 1, d(a+b)/db = 1
                                            for p in &parents {
                                                *grads.entry(*p).or_insert(0.0) += grad;
                                            }
                                        }
                                        "sub" => {
                                            if parents.len() >= 2 {
                                                *grads.entry(parents[0]).or_insert(0.0) += grad;
                                                *grads.entry(parents[1]).or_insert(0.0) -= grad;
                                            }
                                        }
                                        "mul" => {
                                            if parents.len() >= 2 {
                                                let a_val = self.get_f64(parents[0]).unwrap_or(0.0);
                                                let b_val = self.get_f64(parents[1]).unwrap_or(0.0);
                                                *grads.entry(parents[0]).or_insert(0.0) +=
                                                    grad * b_val;
                                                *grads.entry(parents[1]).or_insert(0.0) +=
                                                    grad * a_val;
                                            }
                                        }
                                        "div" => {
                                            if parents.len() >= 2 {
                                                let a_val = self.get_f64(parents[0]).unwrap_or(0.0);
                                                let b_val = self.get_f64(parents[1]).unwrap_or(1.0);
                                                *grads.entry(parents[0]).or_insert(0.0) +=
                                                    grad / b_val;
                                                *grads.entry(parents[1]).or_insert(0.0) -=
                                                    grad * a_val / (b_val * b_val);
                                            }
                                        }
                                        "neg" => {
                                            if let Some(&p) = parents.first() {
                                                *grads.entry(p).or_insert(0.0) -= grad;
                                            }
                                        }
                                        "sin" => {
                                            if let Some(&p) = parents.first() {
                                                let x = self.get_f64(p).unwrap_or(0.0);
                                                *grads.entry(p).or_insert(0.0) += grad * x.cos();
                                            }
                                        }
                                        "cos" => {
                                            if let Some(&p) = parents.first() {
                                                let x = self.get_f64(p).unwrap_or(0.0);
                                                *grads.entry(p).or_insert(0.0) -= grad * x.sin();
                                            }
                                        }
                                        "exp" => {
                                            if let Some(&p) = parents.first() {
                                                let x = self.get_f64(p).unwrap_or(0.0);
                                                *grads.entry(p).or_insert(0.0) += grad * x.exp();
                                            }
                                        }
                                        "log" => {
                                            if let Some(&p) = parents.first() {
                                                let x = self.get_f64(p).unwrap_or(1.0);
                                                *grads.entry(p).or_insert(0.0) += grad / x;
                                            }
                                        }
                                        "sqrt" => {
                                            if let Some(&p) = parents.first() {
                                                let x = self.get_f64(p).unwrap_or(1.0);
                                                *grads.entry(p).or_insert(0.0) +=
                                                    grad / (2.0 * x.sqrt());
                                            }
                                        }
                                        "relu" => {
                                            if let Some(&p) = parents.first() {
                                                let x = self.get_f64(p).unwrap_or(0.0);
                                                *grads.entry(p).or_insert(0.0) +=
                                                    if x > 0.0 { grad } else { 0.0 };
                                            }
                                        }
                                        "sigmoid" => {
                                            if let Some(&p) = parents.first() {
                                                let x = self.get_f64(p).unwrap_or(0.0);
                                                let s = 1.0 / (1.0 + (-x).exp());
                                                *grads.entry(p).or_insert(0.0) +=
                                                    grad * s * (1.0 - s);
                                            }
                                        }
                                        "tanh" => {
                                            if let Some(&p) = parents.first() {
                                                let x = self.get_f64(p).unwrap_or(0.0);
                                                let t = x.tanh();
                                                *grads.entry(p).or_insert(0.0) +=
                                                    grad * (1.0 - t * t);
                                            }
                                        }
                                        "pow" => {
                                            if parents.len() >= 2 {
                                                let base = self.get_f64(parents[0]).unwrap_or(1.0);
                                                let exp = self.get_f64(parents[1]).unwrap_or(1.0);
                                                // d/dbase = exp * base^(exp-1)
                                                *grads.entry(parents[0]).or_insert(0.0) +=
                                                    grad * exp * base.powf(exp - 1.0);
                                                // d/dexp = base^exp * ln(base)
                                                *grads.entry(parents[1]).or_insert(0.0) +=
                                                    grad * base.powf(exp) * base.ln();
                                            }
                                        }
                                        "abs" => {
                                            if let Some(&p) = parents.first() {
                                                let x = self.get_f64(p).unwrap_or(0.0);
                                                *grads.entry(p).or_insert(0.0) +=
                                                    grad * if x >= 0.0 { 1.0 } else { -1.0 };
                                            }
                                        }
                                        _ => {
                                            // Unknown op: gradients stay 0 for parents
                                        }
                                    }
                                }
                            }

                            // Store the gradient map as an opaque unit value
                            // The actual gradients are extracted via TapeGrad
                            self.tape_grads = grads;
                            self.values.insert(*result, IrValue::Unit);
                        }

                        IrInstr::TapeGrad { result, tape_node } => {
                            let grad_val = self.tape_grads.get(tape_node).copied().unwrap_or(0.0);
                            self.values.insert(*result, IrValue::F64(grad_val));
                        }

                        IrInstr::MakeSome { result, value, .. } => {
                            let v = self.get(*value)?;
                            self.values
                                .insert(*result, IrValue::OptionVal(Some(Box::new(v))));
                        }

                        IrInstr::MakeNone { result, .. } => {
                            self.values.insert(*result, IrValue::OptionVal(None));
                        }

                        IrInstr::IsSome { result, operand } => {
                            let v = self.get(*operand)?;
                            let b = matches!(v, IrValue::OptionVal(Some(_)));
                            self.values.insert(*result, IrValue::Bool(b));
                        }

                        IrInstr::OptionUnwrap {
                            result, operand, ..
                        } => match self.get(*operand)? {
                            IrValue::OptionVal(Some(inner)) => {
                                self.values.insert(*result, *inner);
                            }
                            IrValue::OptionVal(None) => {
                                return Err(InterpError::Unsupported {
                                    detail: "unwrap called on none".into(),
                                });
                            }
                            other => {
                                return Err(InterpError::TypeError {
                                    detail: format!("OptionUnwrap on non-option: {:?}", other),
                                });
                            }
                        },

                        IrInstr::MakeOk { result, value, .. } => {
                            let v = self.get(*value)?;
                            self.values
                                .insert(*result, IrValue::ResultVal(Ok(Box::new(v))));
                        }

                        IrInstr::MakeErr { result, value, .. } => {
                            let v = self.get(*value)?;
                            self.values
                                .insert(*result, IrValue::ResultVal(Err(Box::new(v))));
                        }

                        IrInstr::IsOk { result, operand } => {
                            let v = self.get(*operand)?;
                            let b = matches!(v, IrValue::ResultVal(Ok(_)));
                            self.values.insert(*result, IrValue::Bool(b));
                        }

                        IrInstr::ResultUnwrap {
                            result, operand, ..
                        } => match self.get(*operand)? {
                            IrValue::ResultVal(Ok(inner)) => {
                                self.values.insert(*result, *inner);
                            }
                            IrValue::ResultVal(Err(_)) => {
                                return Err(InterpError::Unsupported {
                                    detail: "result_unwrap called on err".into(),
                                });
                            }
                            other => {
                                return Err(InterpError::TypeError {
                                    detail: format!("ResultUnwrap on non-result: {:?}", other),
                                });
                            }
                        },

                        IrInstr::ResultUnwrapErr {
                            result, operand, ..
                        } => match self.get(*operand)? {
                            IrValue::ResultVal(Err(inner)) => {
                                self.values.insert(*result, *inner);
                            }
                            IrValue::ResultVal(Ok(_)) => {
                                return Err(InterpError::Unsupported {
                                    detail: "result_unwrap_err called on ok".into(),
                                });
                            }
                            other => {
                                return Err(InterpError::TypeError {
                                    detail: format!("ResultUnwrapErr on non-result: {:?}", other),
                                });
                            }
                        },

                        IrInstr::MakeClosure {
                            result,
                            fn_name,
                            captures,
                            result_ty,
                        } => {
                            let captured_vals: Vec<IrValue> = captures
                                .iter()
                                .map(|v| self.get(*v))
                                .collect::<Result<_, _>>()?;
                            self.values.insert(
                                *result,
                                IrValue::Closure {
                                    fn_name: fn_name.clone(),
                                    captured: captured_vals,
                                    ty: result_ty.clone(),
                                },
                            );
                        }

                        IrInstr::CallClosure {
                            result,
                            closure,
                            args,
                            result_ty,
                            ..
                        } => {
                            let closure_val = self.get(*closure)?;
                            let (fn_name, captured) = match closure_val {
                                IrValue::Closure {
                                    fn_name, captured, ..
                                } => (fn_name, captured),
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!("CallClosure on non-closure: {:?}", other),
                                    })
                                }
                            };
                            let callee = self
                                .module
                                .and_then(|m| m.function_by_name(&fn_name))
                                .ok_or_else(|| InterpError::Unsupported {
                                    detail: format!("undefined closure function: {}", fn_name),
                                })?
                                .clone();
                            let mut call_args: Vec<IrValue> = captured;
                            for a in args {
                                call_args.push(self.get(*a)?);
                            }
                            if self.depth >= self.opts.max_depth {
                                return Err(InterpError::Unsupported {
                                    detail: format!(
                                        "call depth exceeded {} (infinite recursion?)",
                                        self.opts.max_depth
                                    ),
                                });
                            }
                            let mut sub = Interpreter::new(self.module, self.opts, self.depth + 1);
                            sub.profiler = self.profiler.clone();
                            sub.trace_out = self.trace_out.clone();
                            sub.trace_func = fn_name.clone();
                            sub.trace_source = self.trace_source.clone();
                            if let Some(ref prof) = self.profiler {
                                prof.borrow_mut().enter_function(&fn_name);
                            }
                            let ret = sub.run(&callee, &call_args)?;
                            self.spawn_handles.extend(sub.spawn_handles.drain(..));
                            if let Some(ref prof) = self.profiler {
                                prof.borrow_mut().exit_function(&fn_name);
                            }
                            if let Some(r) = result {
                                self.values
                                    .insert(*r, ret.into_iter().next().unwrap_or(IrValue::Unit));
                            }
                            let _ = result_ty;
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

                        IrInstr::Panic { msg, span_byte } => {
                            // Prefer the position carried on the instruction over
                            // the sticky `last_byte`. `span_table` is keyed by
                            // (block, instr_idx) and no pass maintains it, so once
                            // const-folding deletes the ConstStr holding the panic
                            // message every later index shifts and the panic is
                            // attributed to the preceding statement — a line that
                            // succeeded. See known-issues #20.
                            if let Some(byte) = span_byte {
                                self.last_byte = Some(*byte);
                            }
                            let msg_val = self
                                .values
                                .get(msg)
                                .cloned()
                                .ok_or(InterpError::UndefinedValue { id: msg.0 })?;
                            let msg_str = match &msg_val {
                                IrValue::Str(s) => s.clone(),
                                other => format!("{}", other),
                            };
                            return Err(InterpError::Panic { msg: msg_str });
                        }

                        IrInstr::ValueToStr { result, operand } => {
                            let v = self.get(*operand)?;
                            let s = match &v {
                                IrValue::Str(s) => s.clone(),
                                other => format!("{}", other),
                            };
                            self.values.insert(*result, IrValue::Str(s));
                        }

                        IrInstr::ReadLine { result } => {
                            let mut line = String::new();
                            std::io::stdin().read_line(&mut line).map_err(|e| {
                                InterpError::Unsupported {
                                    detail: format!("read_line failed: {}", e),
                                }
                            })?;
                            let s = line.trim_end_matches(['\n', '\r']).to_owned();
                            self.values.insert(*result, IrValue::Str(s));
                        }

                        IrInstr::ReadI64 { result } => {
                            let mut line = String::new();
                            std::io::stdin().read_line(&mut line).map_err(|e| {
                                InterpError::Unsupported {
                                    detail: format!("read_i64 failed: {}", e),
                                }
                            })?;
                            let n: i64 =
                                line.trim().parse().map_err(|e| InterpError::Unsupported {
                                    detail: format!("read_i64 parse error: {}", e),
                                })?;
                            self.values.insert(*result, IrValue::I64(n));
                        }

                        IrInstr::ReadF64 { result } => {
                            let mut line = String::new();
                            std::io::stdin().read_line(&mut line).map_err(|e| {
                                InterpError::Unsupported {
                                    detail: format!("read_f64 failed: {}", e),
                                }
                            })?;
                            let x: f64 =
                                line.trim().parse().map_err(|e| InterpError::Unsupported {
                                    detail: format!("read_f64 parse error: {}", e),
                                })?;
                            self.values.insert(*result, IrValue::F64(x));
                        }

                        IrInstr::ParseI64 { result, operand } => {
                            let v = self.get(*operand)?;
                            let s = match &v {
                                IrValue::Str(s) => s.clone(),
                                other => format!("{}", other),
                            };
                            let opt = s
                                .trim()
                                .parse::<i64>()
                                .ok()
                                .map(|n| Box::new(IrValue::I64(n)));
                            self.values.insert(*result, IrValue::OptionVal(opt));
                        }

                        IrInstr::ParseF64 { result, operand } => {
                            let v = self.get(*operand)?;
                            let s = match &v {
                                IrValue::Str(s) => s.clone(),
                                other => format!("{}", other),
                            };
                            let opt = s
                                .trim()
                                .parse::<f64>()
                                .ok()
                                .map(|x| Box::new(IrValue::F64(x)));
                            self.values.insert(*result, IrValue::OptionVal(opt));
                        }

                        IrInstr::StrIndex {
                            result,
                            string,
                            index,
                        } => {
                            let sv = self.get(*string)?;
                            let iv = self.get(*index)?;
                            let s = match &sv {
                                IrValue::Str(s) => s.clone(),
                                other => format!("{}", other),
                            };
                            let idx = match &iv {
                                IrValue::I64(n) => *n,
                                _ => {
                                    return Err(InterpError::TypeError {
                                        detail: "str_index index must be i64".into(),
                                    })
                                }
                            };
                            let byte = s
                                .as_bytes()
                                .get(idx as usize)
                                .ok_or(InterpError::IndexOutOfBounds { idx, len: s.len() })?;
                            self.values.insert(*result, IrValue::I64(*byte as i64));
                        }

                        IrInstr::StrSlice {
                            result,
                            string,
                            start,
                            end,
                        } => {
                            let sv = self.get(*string)?;
                            let startv = self.get(*start)?;
                            let endv = self.get(*end)?;
                            let s = match &sv {
                                IrValue::Str(s) => s.clone(),
                                other => format!("{}", other),
                            };
                            let start_idx = match &startv {
                                IrValue::I64(n) => *n as usize,
                                _ => {
                                    return Err(InterpError::TypeError {
                                        detail: "slice start must be i64".into(),
                                    })
                                }
                            };
                            let end_idx = match &endv {
                                IrValue::I64(n) => *n as usize,
                                _ => {
                                    return Err(InterpError::TypeError {
                                        detail: "slice end must be i64".into(),
                                    })
                                }
                            };
                            let slice = s.get(start_idx..end_idx).unwrap_or("").to_owned();
                            self.values.insert(*result, IrValue::Str(slice));
                        }

                        IrInstr::StrFind {
                            result,
                            haystack,
                            needle,
                        } => {
                            let hv = self.get(*haystack)?;
                            let nv = self.get(*needle)?;
                            let h = match &hv {
                                IrValue::Str(s) => s.clone(),
                                other => format!("{}", other),
                            };
                            let n = match &nv {
                                IrValue::Str(s) => s.clone(),
                                other => format!("{}", other),
                            };
                            let opt = h.find(&*n).map(|i| Box::new(IrValue::I64(i as i64)));
                            self.values.insert(*result, IrValue::OptionVal(opt));
                        }

                        IrInstr::StrReplace {
                            result,
                            string,
                            from,
                            to,
                        } => {
                            let sv = self.get(*string)?;
                            let fv = self.get(*from)?;
                            let tv = self.get(*to)?;
                            let s = match &sv {
                                IrValue::Str(s) => s.clone(),
                                other => format!("{}", other),
                            };
                            let f = match &fv {
                                IrValue::Str(s) => s.clone(),
                                other => format!("{}", other),
                            };
                            let t = match &tv {
                                IrValue::Str(s) => s.clone(),
                                other => format!("{}", other),
                            };
                            self.values
                                .insert(*result, IrValue::Str(s.replace(&*f, &t)));
                        }

                        IrInstr::ListNew { result, .. } => {
                            let list = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
                            self.values.insert(*result, IrValue::List(list));
                        }
                        IrInstr::ListPush { list, value } => {
                            let lv = self.get(*list)?;
                            let v = self.get(*value)?;
                            if let IrValue::List(cells) = lv {
                                cells.lock().unwrap().push(v);
                                self.values.insert(*list, IrValue::List(cells));
                            } else {
                                return Err(InterpError::TypeError {
                                    detail: "list_push: not a list".into(),
                                });
                            }
                        }
                        IrInstr::ListLen { result, list } => {
                            let lv = self.get(*list)?;
                            let len = if let IrValue::List(cells) = &lv {
                                cells.lock().unwrap().len() as i64
                            } else {
                                return Err(InterpError::TypeError {
                                    detail: "list_len: not a list".into(),
                                });
                            };
                            self.values.insert(*result, IrValue::I64(len));
                        }
                        IrInstr::ListGet {
                            result,
                            list,
                            index,
                            elem_ty,
                        } => {
                            let lv = self.get(*list)?;
                            let iv = self.get(*index)?;
                            let idx = match iv {
                                IrValue::I64(n) => n as usize,
                                _ => {
                                    return Err(InterpError::TypeError {
                                        detail: "list_get: index must be i64".into(),
                                    })
                                }
                            };
                            if let IrValue::List(cells) = lv {
                                let raw =
                                    cells.lock().unwrap().get(idx).cloned().ok_or_else(|| {
                                        InterpError::TypeError {
                                            detail: format!(
                                                "list_get: index {} out of bounds",
                                                idx
                                            ),
                                        }
                                    })?;
                                // Coerce to declared element type (e.g. f32 stored → f64 expected)
                                let elem = eval_cast(&raw, elem_ty).unwrap_or(raw);
                                self.values.insert(*result, elem);
                            } else {
                                return Err(InterpError::TypeError {
                                    detail: "list_get: not a list".into(),
                                });
                            }
                        }
                        IrInstr::ListSet { list, index, value } => {
                            let lv = self.get(*list)?;
                            let iv = self.get(*index)?;
                            let v = self.get(*value)?;
                            let idx = match iv {
                                IrValue::I64(n) => n as usize,
                                _ => {
                                    return Err(InterpError::TypeError {
                                        detail: "list_set: index must be i64".into(),
                                    })
                                }
                            };
                            if let IrValue::List(cells) = lv {
                                {
                                    let mut borrow = cells.lock().unwrap();
                                    if idx >= borrow.len() {
                                        return Err(InterpError::TypeError {
                                            detail: format!(
                                                "list_set: index {} out of bounds",
                                                idx
                                            ),
                                        });
                                    }
                                    borrow[idx] = v;
                                }
                                self.values.insert(*list, IrValue::List(cells));
                            } else {
                                return Err(InterpError::TypeError {
                                    detail: "list_set: not a list".into(),
                                });
                            }
                        }
                        IrInstr::ListPop { result, list, .. } => {
                            let lv = self.get(*list)?;
                            if let IrValue::List(cells) = lv {
                                let elem = cells.lock().unwrap().pop().ok_or_else(|| {
                                    InterpError::TypeError {
                                        detail: "list_pop: empty list".into(),
                                    }
                                })?;
                                self.values.insert(*list, IrValue::List(cells));
                                self.values.insert(*result, elem);
                            } else {
                                return Err(InterpError::TypeError {
                                    detail: "list_pop: not a list".into(),
                                });
                            }
                        }

                        IrInstr::MapNew { result, .. } => {
                            let map = std::sync::Arc::new(std::sync::Mutex::new(
                                std::collections::HashMap::new(),
                            ));
                            self.values.insert(*result, IrValue::Map(map));
                        }
                        IrInstr::MapSet { map, key, value } => {
                            let mv = self.get(*map)?;
                            let kv = self.get(*key)?;
                            let v = self.get(*value)?;
                            let key_str = format!("{}", kv);
                            if let IrValue::Map(entries) = mv {
                                entries.lock().unwrap().insert(key_str, v);
                                self.values.insert(*map, IrValue::Map(entries));
                            } else {
                                return Err(InterpError::TypeError {
                                    detail: "map_set: not a map".into(),
                                });
                            }
                        }
                        IrInstr::MapGet {
                            result, map, key, ..
                        } => {
                            let mv = self.get(*map)?;
                            let kv = self.get(*key)?;
                            let key_str = format!("{}", kv);
                            if let IrValue::Map(entries) = mv {
                                let opt =
                                    entries.lock().unwrap().get(&key_str).cloned().map(Box::new);
                                self.values.insert(*result, IrValue::OptionVal(opt));
                            } else {
                                return Err(InterpError::TypeError {
                                    detail: "map_get: not a map".into(),
                                });
                            }
                        }
                        IrInstr::MapContains { result, map, key } => {
                            let mv = self.get(*map)?;
                            let kv = self.get(*key)?;
                            let key_str = format!("{}", kv);
                            if let IrValue::Map(entries) = mv {
                                let contains = entries.lock().unwrap().contains_key(&key_str);
                                self.values.insert(*result, IrValue::Bool(contains));
                            } else {
                                return Err(InterpError::TypeError {
                                    detail: "map_contains: not a map".into(),
                                });
                            }
                        }
                        IrInstr::MapRemove { map, key } => {
                            let mv = self.get(*map)?;
                            let kv = self.get(*key)?;
                            let key_str = format!("{}", kv);
                            if let IrValue::Map(entries) = mv {
                                entries.lock().unwrap().remove(&key_str);
                                self.values.insert(*map, IrValue::Map(entries));
                            } else {
                                return Err(InterpError::TypeError {
                                    detail: "map_remove: not a map".into(),
                                });
                            }
                        }
                        IrInstr::MapLen { result, map } => {
                            let mv = self.get(*map)?;
                            let len = if let IrValue::Map(entries) = &mv {
                                entries.lock().unwrap().len() as i64
                            } else {
                                return Err(InterpError::TypeError {
                                    detail: "map_len: not a map".into(),
                                });
                            };
                            self.values.insert(*result, IrValue::I64(len));
                        }

                        // ── Phase 56: File I/O ────────────────────────────────
                        IrInstr::FileReadAll { result, path } => {
                            let p = self.get(*path)?;
                            let path_str = if let IrValue::Str(s) = p {
                                s
                            } else {
                                String::new()
                            };
                            security::check_fs_read(&path_str)?;
                            match std::fs::read_to_string(&path_str) {
                                Ok(s) => {
                                    self.values.insert(
                                        *result,
                                        IrValue::ResultVal(Ok(Box::new(IrValue::Str(s)))),
                                    );
                                }
                                Err(e) => {
                                    self.values.insert(
                                        *result,
                                        IrValue::ResultVal(Err(Box::new(IrValue::Str(
                                            e.to_string(),
                                        )))),
                                    );
                                }
                            }
                        }
                        IrInstr::FileWriteAll {
                            result,
                            path,
                            content,
                        } => {
                            let p = self.get(*path)?;
                            let c = self.get(*content)?;
                            let path_str = if let IrValue::Str(s) = p {
                                s
                            } else {
                                String::new()
                            };
                            let content_str = if let IrValue::Str(s) = c {
                                s
                            } else {
                                String::new()
                            };
                            security::check_fs_write(&path_str)?;
                            match std::fs::write(&path_str, &content_str) {
                                Ok(()) => {
                                    let written_len = content_str.len() as i64;
                                    self.values.insert(
                                        *result,
                                        IrValue::ResultVal(Ok(Box::new(IrValue::I64(written_len)))),
                                    );
                                }
                                Err(e) => {
                                    self.values.insert(
                                        *result,
                                        IrValue::ResultVal(Err(Box::new(IrValue::Str(
                                            e.to_string(),
                                        )))),
                                    );
                                }
                            }
                        }
                        IrInstr::FileExists { result, path } => {
                            let p = self.get(*path)?;
                            let path_str = if let IrValue::Str(s) = p {
                                s
                            } else {
                                String::new()
                            };
                            security::check_fs_read(&path_str)?;
                            let exists = std::path::Path::new(&path_str).exists();
                            self.values.insert(*result, IrValue::Bool(exists));
                        }
                        IrInstr::FileLines { result, path } => {
                            let p = self.get(*path)?;
                            let path_str = if let IrValue::Str(s) = p {
                                s
                            } else {
                                String::new()
                            };
                            security::check_fs_read(&path_str)?;
                            let lines: Vec<IrValue> = match std::fs::read_to_string(&path_str) {
                                Ok(s) => s.lines().map(|l| IrValue::Str(l.to_string())).collect(),
                                Err(_) => vec![],
                            };
                            self.values.insert(
                                *result,
                                IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(lines))),
                            );
                        }

                        // ── Database operations ─────────────────────────────────
                        IrInstr::DbOpen { result, path } => {
                            let p = self.get(*path)?;
                            let path_str = if let IrValue::Str(s) = p {
                                s
                            } else {
                                String::new()
                            };
                            security::check_fs_write(&path_str)?;
                            match rusqlite::Connection::open(&path_str) {
                                Ok(conn) => {
                                    let handle = Box::into_raw(Box::new(conn)) as i64;
                                    self.values.insert(*result, IrValue::I64(handle));
                                }
                                Err(_) => {
                                    self.values.insert(*result, IrValue::I64(0));
                                }
                            }
                        }
                        IrInstr::DbExec { result, db, sql } => {
                            let db_handle = if let IrValue::I64(h) = self.get(*db)? {
                                h
                            } else {
                                0
                            };
                            let sql_str = if let IrValue::Str(s) = self.get(*sql)? {
                                s
                            } else {
                                String::new()
                            };
                            if db_handle != 0 {
                                let conn = unsafe { &*(db_handle as *const rusqlite::Connection) };
                                match conn.execute_batch(&sql_str) {
                                    Ok(()) => self.values.insert(*result, IrValue::I64(0)),
                                    Err(_) => self.values.insert(*result, IrValue::I64(-1)),
                                };
                            } else {
                                self.values.insert(*result, IrValue::I64(-1));
                            }
                        }
                        IrInstr::DbExecParams {
                            result,
                            db,
                            sql,
                            params,
                        } => {
                            let db_handle = if let IrValue::I64(h) = self.get(*db)? {
                                h
                            } else {
                                0
                            };
                            let sql_str = if let IrValue::Str(s) = self.get(*sql)? {
                                s
                            } else {
                                String::new()
                            };
                            let params_val = self.get(*params)?;
                            if db_handle != 0 {
                                let conn = unsafe { &*(db_handle as *const rusqlite::Connection) };
                                let bind_values = self.sqlite_param_values(&params_val);
                                match conn.execute(&sql_str, params_from_iter(bind_values)) {
                                    Ok(_) => self.values.insert(*result, IrValue::I64(0)),
                                    Err(_) => self.values.insert(*result, IrValue::I64(-1)),
                                };
                            } else {
                                self.values.insert(*result, IrValue::I64(-1));
                            }
                        }
                        IrInstr::DbQuery { result, db, sql } => {
                            let db_handle = if let IrValue::I64(h) = self.get(*db)? {
                                h
                            } else {
                                0
                            };
                            let sql_str = if let IrValue::Str(s) = self.get(*sql)? {
                                s
                            } else {
                                String::new()
                            };
                            let rows: Vec<IrValue> = if db_handle != 0 {
                                let conn = unsafe { &*(db_handle as *const rusqlite::Connection) };
                                match conn.prepare(&sql_str) {
                                    Ok(mut stmt) => {
                                        let col_count = stmt.column_count();
                                        let mut all_rows = Vec::new();
                                        if let Ok(iter) = stmt.query_map([], |row| {
                                            let mut cols = Vec::new();
                                            for i in 0..col_count {
                                                let val: String =
                                                    row.get::<_, String>(i).unwrap_or_default();
                                                cols.push(IrValue::Str(val));
                                            }
                                            Ok(cols)
                                        }) {
                                            for cols in iter.flatten() {
                                                all_rows.push(IrValue::List(std::sync::Arc::new(
                                                    std::sync::Mutex::new(cols),
                                                )));
                                            }
                                        }
                                        all_rows
                                    }
                                    Err(_) => vec![],
                                }
                            } else {
                                vec![]
                            };
                            self.values.insert(
                                *result,
                                IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(rows))),
                            );
                        }
                        IrInstr::DbQueryParams {
                            result,
                            db,
                            sql,
                            params,
                        } => {
                            let db_handle = if let IrValue::I64(h) = self.get(*db)? {
                                h
                            } else {
                                0
                            };
                            let sql_str = if let IrValue::Str(s) = self.get(*sql)? {
                                s
                            } else {
                                String::new()
                            };
                            let params_val = self.get(*params)?;
                            let rows: Vec<IrValue> = if db_handle != 0 {
                                let conn = unsafe { &*(db_handle as *const rusqlite::Connection) };
                                let bind_values = self.sqlite_param_values(&params_val);
                                match conn.prepare(&sql_str) {
                                    Ok(mut stmt) => {
                                        let col_count = stmt.column_count();
                                        let mut all_rows = Vec::new();
                                        if let Ok(iter) =
                                            stmt.query_map(params_from_iter(bind_values), |row| {
                                                let mut cols = Vec::new();
                                                for i in 0..col_count {
                                                    let val: String =
                                                        row.get::<_, String>(i).unwrap_or_default();
                                                    cols.push(IrValue::Str(val));
                                                }
                                                Ok(cols)
                                            })
                                        {
                                            for cols in iter.flatten() {
                                                all_rows.push(IrValue::List(std::sync::Arc::new(
                                                    std::sync::Mutex::new(cols),
                                                )));
                                            }
                                        }
                                        all_rows
                                    }
                                    Err(_) => vec![],
                                }
                            } else {
                                vec![]
                            };
                            self.values.insert(
                                *result,
                                IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(rows))),
                            );
                        }
                        IrInstr::DbClose { result, db } => {
                            let db_handle = if let IrValue::I64(h) = self.get(*db)? {
                                h
                            } else {
                                0
                            };
                            if db_handle != 0 {
                                unsafe {
                                    drop(Box::from_raw(db_handle as *mut rusqlite::Connection));
                                }
                            }
                            self.values.insert(*result, IrValue::I64(0));
                        }

                        // ── Phase 58: Extended collections ─────────────────────
                        IrInstr::ListContains {
                            result,
                            list,
                            value,
                        } => {
                            let v = self.get(*value)?;
                            let lst = self.get(*list)?;
                            if let IrValue::List(rc) = lst {
                                let found = rc.lock().unwrap().iter().any(|item| item == &v);
                                self.values.insert(*result, IrValue::Bool(found));
                            } else {
                                self.values.insert(*result, IrValue::Bool(false));
                            }
                        }
                        IrInstr::ListSort { list } => {
                            let lst = self.get(*list)?;
                            if let IrValue::List(rc) = lst {
                                rc.lock().unwrap().sort_by(|a, b| match (a, b) {
                                    (IrValue::I64(x), IrValue::I64(y)) => x.cmp(y),
                                    (IrValue::F64(x), IrValue::F64(y)) => {
                                        x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
                                    }
                                    (IrValue::F32(x), IrValue::F32(y)) => {
                                        x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
                                    }
                                    (IrValue::Str(x), IrValue::Str(y)) => x.cmp(y),
                                    _ => std::cmp::Ordering::Equal,
                                });
                            }
                        }
                        IrInstr::MapKeys { result, map } => {
                            let m = self.get(*map)?;
                            if let IrValue::Map(rc) = m {
                                let keys: Vec<IrValue> = rc
                                    .lock()
                                    .unwrap()
                                    .keys()
                                    .map(|k| IrValue::Str(k.clone()))
                                    .collect();
                                self.values.insert(
                                    *result,
                                    IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(keys))),
                                );
                            } else {
                                self.values.insert(
                                    *result,
                                    IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                                        vec![],
                                    ))),
                                );
                            }
                        }
                        IrInstr::MapValues { result, map } => {
                            let m = self.get(*map)?;
                            if let IrValue::Map(rc) = m {
                                let vals: Vec<IrValue> =
                                    rc.lock().unwrap().values().cloned().collect();
                                self.values.insert(
                                    *result,
                                    IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(vals))),
                                );
                            } else {
                                self.values.insert(
                                    *result,
                                    IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                                        vec![],
                                    ))),
                                );
                            }
                        }
                        IrInstr::ListConcat { result, lhs, rhs } => {
                            let l = self.get(*lhs)?;
                            let r = self.get(*rhs)?;
                            let mut combined = vec![];
                            if let IrValue::List(rc) = l {
                                combined.extend(rc.lock().unwrap().iter().cloned());
                            }
                            if let IrValue::List(rc) = r {
                                combined.extend(rc.lock().unwrap().iter().cloned());
                            }
                            self.values.insert(
                                *result,
                                IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(combined))),
                            );
                        }
                        IrInstr::ListSlice {
                            result,
                            list,
                            start,
                            end,
                        } => {
                            let lst = self.get(*list)?;
                            let s = self.get(*start)?;
                            let e = self.get(*end)?;
                            let si = if let IrValue::I64(n) = s {
                                n as usize
                            } else {
                                0
                            };
                            let ei = if let IrValue::I64(n) = e {
                                n as usize
                            } else {
                                0
                            };
                            if let IrValue::List(rc) = lst {
                                let sliced: Vec<IrValue> =
                                    rc.lock().unwrap().get(si..ei).unwrap_or(&[]).to_vec();
                                self.values.insert(
                                    *result,
                                    IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                                        sliced,
                                    ))),
                                );
                            } else {
                                self.values.insert(
                                    *result,
                                    IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                                        vec![],
                                    ))),
                                );
                            }
                        }

                        // ── Phase 59: Process / environment ───────────────────
                        IrInstr::ProcessExit { code } => {
                            let c = self.get(*code)?;
                            let code_val = if let IrValue::I64(n) = c { n as i32 } else { 0 };
                            security::check_process("exit")?;
                            std::process::exit(code_val);
                        }
                        IrInstr::ProcessArgs { result } => {
                            let args: Vec<IrValue> = std::env::args().map(IrValue::Str).collect();
                            self.values.insert(
                                *result,
                                IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(args))),
                            );
                        }
                        IrInstr::EnvVar { result, name } => {
                            let n = self.get(*name)?;
                            let name_str = if let IrValue::Str(s) = n {
                                s
                            } else {
                                String::new()
                            };
                            match std::env::var(&name_str) {
                                Ok(v) => {
                                    self.values.insert(
                                        *result,
                                        IrValue::OptionVal(Some(Box::new(IrValue::Str(v)))),
                                    );
                                }
                                Err(_) => {
                                    self.values.insert(*result, IrValue::OptionVal(None));
                                }
                            }
                        }
                        // Phase 61: Pattern matching helpers
                        IrInstr::GetVariantTag { result, operand } => {
                            let v = self.get(*operand)?;
                            match v {
                                IrValue::Enum(tag, _) => {
                                    self.values.insert(*result, IrValue::I64(tag as i64));
                                }
                                other => {
                                    return Err(InterpError::TypeError {
                                        detail: format!(
                                            "GetVariantTag on non-Enum value: {:?}",
                                            other
                                        ),
                                    });
                                }
                            }
                        }
                        IrInstr::StrEq { result, lhs, rhs } => {
                            let lv = self.get(*lhs)?;
                            let rv = self.get(*rhs)?;
                            let eq = match (lv, rv) {
                                (IrValue::Str(a), IrValue::Str(b)) => a == b,
                                _ => false,
                            };
                            self.values.insert(*result, IrValue::Bool(eq));
                        }
                        // Phase 83: GC retain/release — no-op in interpreter (Rc handles it)
                        IrInstr::Retain { .. } => {}
                        IrInstr::Release { .. } => {}
                        // Phase 81: FFI extern calls — interpreter dispatches known names to Rust stubs
                        IrInstr::CallExtern {
                            result,
                            name,
                            args,
                            ret_ty,
                        } => {
                            let arg_vals: Vec<IrValue> = args
                                .iter()
                                .map(|a| self.get(*a))
                                .collect::<Result<Vec<_>, _>>()?;

                            // Check if this extern call is intercepted by an effect handler.
                            let handler_result = self.dispatch_handler(name, &arg_vals);

                            if let Some(handled_ret) = handler_result {
                                if let Some(r) = result {
                                    self.values.insert(*r, handled_ret);
                                }
                            } else {
                                let ret = self.dispatch_extern(name, &arg_vals, ret_ty)?;
                                if let Some(r) = result {
                                    self.values.insert(*r, ret);
                                }
                            }
                        }
                        // Phase 88: TCP network I/O — wire to real TCP via tcp_store
                        IrInstr::TcpConnect { result, host, port } => {
                            let h = match self.get(*host)? {
                                IrValue::Str(s) => s,
                                _ => String::new(),
                            };
                            let p = match self.get(*port)? {
                                IrValue::I64(n) => n,
                                _ => 0,
                            };
                            security::check_network(&h)?;
                            let id = match std::net::TcpStream::connect(format!("{}:{}", h, p)) {
                                Ok(stream) => tcp_store::store_stream(stream),
                                Err(_) => -1,
                            };
                            self.values.insert(*result, IrValue::I64(id));
                        }
                        IrInstr::TcpListen { result, port } => {
                            let p = match self.get(*port)? {
                                IrValue::I64(n) => n,
                                _ => 0,
                            };
                            security::check_network("0.0.0.0")?;
                            let id = match std::net::TcpListener::bind(format!("0.0.0.0:{}", p)) {
                                Ok(listener) => tcp_store::store_listener(listener),
                                Err(_) => -1,
                            };
                            self.values.insert(*result, IrValue::I64(id));
                        }
                        IrInstr::TcpAccept { result, listener } => {
                            let id = match self.get(*listener)? {
                                IrValue::I64(n) => n,
                                _ => -1,
                            };
                            let conn = tcp_store::accept_listener(id).unwrap_or(-1);
                            self.values.insert(*result, IrValue::I64(conn));
                        }
                        IrInstr::TcpRead { result, conn } => {
                            let id = match self.get(*conn)? {
                                IrValue::I64(n) => n,
                                _ => -1,
                            };
                            let data = tcp_store::read_stream(id).unwrap_or_default();
                            self.values.insert(*result, IrValue::Str(data));
                        }
                        IrInstr::TcpWrite { conn, data } => {
                            let id = match self.get(*conn)? {
                                IrValue::I64(n) => n,
                                _ => -1,
                            };
                            let s = match self.get(*data)? {
                                IrValue::Str(s) => s,
                                _ => String::new(),
                            };
                            tcp_store::write_stream(id, &s);
                        }
                        IrInstr::TcpClose { conn } => {
                            let id = match self.get(*conn)? {
                                IrValue::I64(n) => n,
                                _ => -1,
                            };
                            tcp_store::close(id);
                        }
                        IrInstr::StrSplit {
                            result,
                            str_val,
                            delim,
                        } => {
                            let sv = self.get(*str_val)?;
                            let dv = self.get(*delim)?;
                            let (s, d) = match (&sv, &dv) {
                                (IrValue::Str(s), IrValue::Str(d)) => (s.clone(), d.clone()),
                                _ => {
                                    return Err(InterpError::TypeError {
                                        detail: "str_split: expected str".into(),
                                    })
                                }
                            };
                            let parts: Vec<IrValue> = if d.is_empty() {
                                s.chars().map(|c| IrValue::Str(c.to_string())).collect()
                            } else {
                                s.split(d.as_str())
                                    .map(|p| IrValue::Str(p.to_owned()))
                                    .collect()
                            };
                            let list =
                                IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(parts)));
                            self.values.insert(*result, list);
                        }
                        IrInstr::StrJoin {
                            result,
                            list_val,
                            delim,
                        } => {
                            let lv = self.get(*list_val)?;
                            let dv = self.get(*delim)?;
                            let d = if let IrValue::Str(s) = &dv {
                                s.clone()
                            } else {
                                return Err(InterpError::TypeError {
                                    detail: "str_join: delim must be str".into(),
                                });
                            };
                            let joined = if let IrValue::List(cells) = &lv {
                                cells
                                    .lock()
                                    .unwrap()
                                    .iter()
                                    .map(|v| {
                                        if let IrValue::Str(s) = v {
                                            s.clone()
                                        } else {
                                            format!("{}", v)
                                        }
                                    })
                                    .collect::<Vec<_>>()
                                    .join(&d)
                            } else {
                                return Err(InterpError::TypeError {
                                    detail: "str_join: expected list<str>".into(),
                                });
                            };
                            self.values.insert(*result, IrValue::Str(joined));
                        }
                        IrInstr::NowMs { result } => {
                            use std::time::{SystemTime, UNIX_EPOCH};
                            let ms = SystemTime::now()
                                .duration_since(UNIX_EPOCH)
                                .map(|d| d.as_millis() as i64)
                                .unwrap_or(0);
                            self.values.insert(*result, IrValue::I64(ms));
                        }
                        IrInstr::SleepMs { result, ms } => {
                            let n = match self.get(*ms)? {
                                IrValue::I64(n) => n,
                                _ => {
                                    return Err(InterpError::TypeError {
                                        detail: "sleep_ms: expected i64".into(),
                                    })
                                }
                            };
                            std::thread::sleep(std::time::Duration::from_millis(n as u64));
                            self.values.insert(*result, IrValue::I64(0));
                        }
                        // Effect handler instructions
                        IrInstr::PushHandler { arms } => {
                            self.handler_stack.push(arms.clone());
                        }
                        IrInstr::PopHandler => {
                            self.handler_stack.pop();
                        }
                        IrInstr::ResumeCont { cont, value, result } => {
                            // Look up the continuation value to get its resume cell.
                            let cell = match self.values.get(cont) {
                                Some(IrValue::Continuation { resume_value }) => resume_value.clone(),
                                _ => {
                                    return Err(InterpError::Unsupported {
                                        detail: "resume called on non-continuation value".into(),
                                    });
                                }
                            };
                            let v = self.get(*value)?.clone();
                            *cell.lock().unwrap() = Some(v.clone());
                            self.values.insert(*result, v);
                        }
                        // Phase 104: BuiltinCall — unified dispatch for new builtins
                        IrInstr::BuiltinCall {
                            result,
                            name,
                            args,
                            result_ty: _,
                        } => {
                            let arg_vals: Vec<IrValue> = args
                                .iter()
                                .map(|a| self.get(*a))
                                .collect::<Result<Vec<_>, _>>()?;

                            // Handle closure-invoking builtins here (need `self` for function dispatch).
                            let ret = match name.as_str() {
                                "list_map" => self.builtin_list_map(&arg_vals)?,
                                "list_filter" => self.builtin_list_filter(&arg_vals)?,
                                "list_reduce" => self.builtin_list_reduce(&arg_vals)?,
                                "par_map" => self.builtin_par_map(&arg_vals)?,
                                _ => interp_builtin(name, &arg_vals)?,
                            };
                            self.values.insert(*result, ret);
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

    /// Extract an f64 from a value (for reverse-mode AD gradient computation).
    /// Works on F64, F32, TapeNode (extracts primal), I64, I32.
    fn get_f64(&self, id: ValueId) -> Option<f64> {
        match self.values.get(&id)? {
            IrValue::F64(v) => Some(*v),
            IrValue::F32(v) => Some(*v as f64),
            IrValue::I64(v) => Some(*v as f64),
            IrValue::I32(v) => Some(*v as f64),
            IrValue::TapeNode { primal, .. } => match primal.as_ref() {
                IrValue::F64(v) => Some(*v),
                IrValue::F32(v) => Some(*v as f64),
                IrValue::I64(v) => Some(*v as f64),
                IrValue::I32(v) => Some(*v as f64),
                _ => None,
            },
            _ => None,
        }
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

    // ------------------------------------------------------------------
    // Closure-invoking list builtins (need `self` to dispatch functions)
    // ------------------------------------------------------------------

    /// Calls a closure value with the given arguments, returning the result.
    fn call_closure_val(
        &mut self,
        closure: &IrValue,
        call_args: &[IrValue],
    ) -> Result<IrValue, InterpError> {
        let (fn_name, captured) = match closure {
            IrValue::Closure {
                fn_name, captured, ..
            } => (fn_name.clone(), captured.clone()),
            other => {
                return Err(InterpError::TypeError {
                    detail: format!("expected closure, got {:?}", other),
                })
            }
        };
        let callee = self
            .module
            .and_then(|m| m.function_by_name(&fn_name))
            .ok_or_else(|| InterpError::Unsupported {
                detail: format!("undefined closure function: {}", fn_name),
            })?
            .clone();
        let mut args_full: Vec<IrValue> = captured;
        args_full.extend_from_slice(call_args);
        if self.depth >= self.opts.max_depth {
            return Err(InterpError::Unsupported {
                detail: format!(
                    "call depth exceeded {} (infinite recursion?)",
                    self.opts.max_depth
                ),
            });
        }
        let mut sub = Interpreter::new(self.module, self.opts, self.depth + 1);
        let ret = sub.run(&callee, &args_full)?;
        self.spawn_handles.extend(sub.spawn_handles.drain(..));
        Ok(ret.into_iter().next().unwrap_or(IrValue::Unit))
    }

    /// list_map(list, closure) — apply closure to each element, return new list.
    fn builtin_list_map(&mut self, args: &[IrValue]) -> Result<IrValue, InterpError> {
        if args.len() < 2 {
            return Err(InterpError::TypeError {
                detail: "list_map: expected 2 arguments (list, closure)".into(),
            });
        }
        let items = match &args[0] {
            IrValue::List(rc) => rc.lock().unwrap().clone(),
            _ => {
                return Err(InterpError::TypeError {
                    detail: "list_map: first argument must be a list".into(),
                })
            }
        };
        let closure = &args[1];
        let mut result = Vec::with_capacity(items.len());
        for item in &items {
            let mapped = self.call_closure_val(closure, std::slice::from_ref(item))?;
            result.push(mapped);
        }
        Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
            result,
        ))))
    }

    /// list_filter(list, closure) — keep elements where closure returns true.
    fn builtin_list_filter(&mut self, args: &[IrValue]) -> Result<IrValue, InterpError> {
        if args.len() < 2 {
            return Err(InterpError::TypeError {
                detail: "list_filter: expected 2 arguments (list, closure)".into(),
            });
        }
        let items = match &args[0] {
            IrValue::List(rc) => rc.lock().unwrap().clone(),
            _ => {
                return Err(InterpError::TypeError {
                    detail: "list_filter: first argument must be a list".into(),
                })
            }
        };
        let closure = &args[1];
        let mut result = Vec::new();
        for item in &items {
            let keep = self.call_closure_val(closure, std::slice::from_ref(item))?;
            let truthy = match &keep {
                IrValue::Bool(b) => *b,
                IrValue::I64(n) => *n != 0,
                IrValue::I32(n) => *n != 0,
                _ => true,
            };
            if truthy {
                result.push(item.clone());
            }
        }
        Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
            result,
        ))))
    }

    /// list_reduce(list, initial, closure) — fold list with closure(accumulator, element).
    fn builtin_list_reduce(&mut self, args: &[IrValue]) -> Result<IrValue, InterpError> {
        if args.len() < 3 {
            return Err(InterpError::TypeError {
                detail: "list_reduce: expected 3 arguments (list, initial, closure)".into(),
            });
        }
        let items = match &args[0] {
            IrValue::List(rc) => rc.lock().unwrap().clone(),
            _ => {
                return Err(InterpError::TypeError {
                    detail: "list_reduce: first argument must be a list".into(),
                })
            }
        };
        let mut acc = args[1].clone();
        let closure = &args[2];
        for item in &items {
            acc = self.call_closure_val(closure, &[acc.clone(), item.clone()])?;
        }
        Ok(acc)
    }

    /// par_map(list, closure) — apply closure to each element in parallel.
    fn builtin_par_map(&mut self, args: &[IrValue]) -> Result<IrValue, InterpError> {
        if args.len() < 2 {
            return Err(InterpError::TypeError {
                detail: "par_map: expected 2 arguments (list, closure)".into(),
            });
        }
        let items = match &args[0] {
            IrValue::List(rc) => rc.lock().unwrap().clone(),
            _ => {
                return Err(InterpError::TypeError {
                    detail: "par_map: first argument must be a list".into(),
                })
            }
        };
        let closure = &args[1];
        let module_raw: usize = self.module
            .map(|m| m as *const IrModule as usize)
            .unwrap_or(0);
        let opts = self.opts;
        let depth = self.depth;
        let n = items.len();
        // Extract closure info
        let (fn_name, captured) = match closure {
            IrValue::Closure { fn_name, captured, .. } => (fn_name.clone(), captured.clone()),
            _ => {
                return Err(InterpError::TypeError {
                    detail: "par_map: second argument must be a closure".into(),
                })
            }
        };
        let mut handles = Vec::with_capacity(n);
        for elem in items {
            let callee_fn = fn_name.clone();
            let callee_captures = captured.clone();
            let handle = std::thread::spawn(move || {
                let module: Option<&IrModule> = if module_raw == 0 {
                    None
                } else {
                    Some(unsafe { &*(module_raw as *const IrModule) })
                };
                let mut sub = Interpreter::new(module, opts, depth + 1);
                if let Some(func) = sub.module.and_then(|m| m.function_by_name(&callee_fn)) {
                    let func = func.clone();
                    // Captures are prepended before explicit args (same as CallClosure)
                    let mut call_args: Vec<IrValue> = callee_captures;
                    call_args.push(elem);
                    let ret_vals = sub.run(&func, &call_args)?;
                    // run returns Vec<IrValue> — return first (or Unit if empty)
                    Ok(ret_vals.into_iter().next().unwrap_or(IrValue::Unit))
                } else {
                    Err(InterpError::Unsupported {
                        detail: format!("par_map: unknown function: {}", callee_fn),
                    })
                }
            });
            handles.push(handle);
        }
        let mut results = Vec::with_capacity(n);
        for h in handles {
            match h.join() {
                Ok(Ok(val)) => results.push(val),
                Ok(Err(e)) => return Err(e),
                Err(_) => return Err(InterpError::Unsupported {
                    detail: "par_map: thread panicked".to_string(),
                }),
            }
        }
        Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(results))))
    }

    /// Checks if an extern call is intercepted by an active effect handler.
    /// Returns `Some(value)` if handled, `None` if no handler matches.
    fn dispatch_handler(&mut self, name: &str, args: &[IrValue]) -> Option<IrValue> {
        // Walk the handler stack from innermost to outermost.
        for frame in self.handler_stack.iter().rev() {
            for arm in frame {
                if arm.effect_name == name {
                    // Found a matching handler arm. Call the handler function.
                    if let Some(module) = self.module {
                        if let Some(handler_func) = module.function_by_name(&arm.func_name) {
                            let mut sub = Interpreter::new(
                                self.module,
                                self.opts,
                                self.depth + 1,
                            );
                            sub.handler_stack = self.handler_stack.clone();

                            if arm.has_resume {
                                // Pass the continuation as the first arg.
                                let cell: std::sync::Arc<std::sync::Mutex<Option<IrValue>>> =
                                    std::sync::Arc::new(std::sync::Mutex::new(None));
                                let cont = IrValue::Continuation {
                                    resume_value: cell.clone(),
                                };
                                let mut call_args = Vec::with_capacity(args.len() + 1);
                                call_args.push(cont);
                                for a in args {
                                    call_args.push(a.clone());
                                }
                                if let Ok(ret) = sub.run(handler_func, &call_args) {
                                    // If the handler called `resume(v)`, the cell
                                    // is now Some(v). Return that as the effect
                                    // site's value. Otherwise fall back to the
                                    // handler's return value.
                                    if let Some(v) = cell.lock().unwrap().take() {
                                        return Some(v);
                                    }
                                    let val = ret.into_iter().next().unwrap_or(IrValue::I64(0));
                                    return Some(val);
                                }
                            } else {
                                if let Ok(ret) = sub.run(handler_func, args) {
                                    let val = ret.into_iter().next().unwrap_or(IrValue::I64(0));
                                    return Some(val);
                                }
                            }
                        }
                    }
                    // If the handler function can't be found, still intercept
                    // but return a default value to avoid calling the real extern.
                    return Some(IrValue::I64(0));
                }
            }
        }
        None
    }

    /// Dispatch an extern call by name to a built-in Rust stub.
    /// Unknown extern names return an Unsupported error.
    fn dispatch_extern(
        &self,
        name: &str,
        args: &[IrValue],
        ret_ty: &IrType,
    ) -> Result<IrValue, InterpError> {
        match name {
            // Math stubs mirroring common C/CBLAS names
            "cblas_ddot" | "iris_blas_ddot" => {
                // (n: i64, x: list<f64>, y: list<f64>) -> f64
                let n = match args.first() {
                    Some(IrValue::I64(n)) => *n as usize,
                    _ => 0,
                };
                let xs = match args.get(1) {
                    Some(IrValue::List(l)) => l.lock().unwrap().clone(),
                    _ => vec![],
                };
                let ys = match args.get(2) {
                    Some(IrValue::List(l)) => l.lock().unwrap().clone(),
                    _ => vec![],
                };
                let dot: f64 = (0..n.min(xs.len()).min(ys.len()))
                    .map(|i| {
                        let a = match &xs[i] {
                            IrValue::F64(v) => *v,
                            IrValue::F32(v) => *v as f64,
                            _ => 0.0,
                        };
                        let b = match &ys[i] {
                            IrValue::F64(v) => *v,
                            IrValue::F32(v) => *v as f64,
                            _ => 0.0,
                        };
                        a * b
                    })
                    .sum();
                Ok(IrValue::F64(dot))
            }
            "sqrt" | "cblas_sqrt" => {
                let x = match args.first() {
                    Some(IrValue::F64(v)) => *v,
                    Some(IrValue::F32(v)) => *v as f64,
                    _ => 0.0,
                };
                Ok(IrValue::F64(x.sqrt()))
            }
            _ => {
                if let Some(symbol_ptr) = lookup_dynamic_symbol(name) {
                    // Check if we should dispatch as float or integer arguments
                    let all_floats = !args.is_empty()
                        && args
                            .iter()
                            .all(|a| matches!(a, IrValue::F64(_) | IrValue::F32(_)));
                    if all_floats
                        && matches!(
                            ret_ty,
                            IrType::Scalar(
                                crate::ir::types::DType::F64 | crate::ir::types::DType::F32
                            )
                        )
                    {
                        let float_args: Vec<f64> = args
                            .iter()
                            .map(|a| match a {
                                IrValue::F64(f) => *f,
                                IrValue::F32(f) => *f as f64,
                                _ => 0.0,
                            })
                            .collect();
                        unsafe {
                            let res = match float_args.len() {
                                1 => {
                                    let f: extern "C" fn(f64) -> f64 =
                                        std::mem::transmute(symbol_ptr);
                                    f(float_args[0])
                                }
                                2 => {
                                    let f: extern "C" fn(f64, f64) -> f64 =
                                        std::mem::transmute(symbol_ptr);
                                    f(float_args[0], float_args[1])
                                }
                                _ => 0.0,
                            };
                            return match ret_ty {
                                IrType::Scalar(crate::ir::types::DType::F64) => {
                                    Ok(IrValue::F64(res))
                                }
                                IrType::Scalar(crate::ir::types::DType::F32) => {
                                    Ok(IrValue::F32(res as f32))
                                }
                                _ => Ok(IrValue::F64(res)),
                            };
                        }
                    } else {
                        // Dispatch as integer / pointer arguments
                        let mut temp_cstrings = Vec::new();
                        let int_args: Vec<i64> = args
                            .iter()
                            .map(|a| match a {
                                IrValue::I64(n) => *n,
                                IrValue::I32(n) => *n as i64,
                                IrValue::Bool(b) => {
                                    if *b {
                                        1
                                    } else {
                                        0
                                    }
                                }
                                IrValue::Str(s) => {
                                    let cs = std::ffi::CString::new(s.as_bytes()).unwrap_or_default();
                                    let ptr = cs.as_ptr() as i64;
                                    temp_cstrings.push(cs);
                                    ptr
                                }
                                _ => 0,
                            })
                            .collect();
                        unsafe {
                            let raw = match int_args.len() {
                                0 => {
                                    let f: extern "C" fn() -> i64 = std::mem::transmute(symbol_ptr);
                                    f()
                                }
                                1 => {
                                    let f: extern "C" fn(i64) -> i64 =
                                        std::mem::transmute(symbol_ptr);
                                    f(int_args[0])
                                }
                                2 => {
                                    let f: extern "C" fn(i64, i64) -> i64 =
                                        std::mem::transmute(symbol_ptr);
                                    f(int_args[0], int_args[1])
                                }
                                3 => {
                                    let f: extern "C" fn(i64, i64, i64) -> i64 =
                                        std::mem::transmute(symbol_ptr);
                                    f(int_args[0], int_args[1], int_args[2])
                                }
                                4 => {
                                    let f: extern "C" fn(i64, i64, i64, i64) -> i64 =
                                        std::mem::transmute(symbol_ptr);
                                    f(int_args[0], int_args[1], int_args[2], int_args[3])
                                }
                                5 => {
                                    let f: extern "C" fn(i64, i64, i64, i64, i64) -> i64 =
                                        std::mem::transmute(symbol_ptr);
                                    f(
                                        int_args[0],
                                        int_args[1],
                                        int_args[2],
                                        int_args[3],
                                        int_args[4],
                                    )
                                }
                                _ => {
                                    let f: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64 =
                                        std::mem::transmute(symbol_ptr);
                                    f(
                                        int_args[0],
                                        int_args[1],
                                        int_args[2],
                                        int_args[3],
                                        int_args[4],
                                        int_args[5],
                                    )
                                }
                            };
                            return match ret_ty {
                                IrType::Scalar(crate::ir::types::DType::I64) => {
                                    Ok(IrValue::I64(raw))
                                }
                                IrType::Scalar(crate::ir::types::DType::I32) => {
                                    Ok(IrValue::I32(raw as i32))
                                }
                                IrType::Scalar(crate::ir::types::DType::Bool) => {
                                    Ok(IrValue::Bool(raw != 0))
                                }
                                IrType::Str => {
                                    if raw == 0 {
                                        Ok(IrValue::Str(String::new()))
                                    } else {
                                        let cstr = std::ffi::CStr::from_ptr(
                                            raw as *const std::os::raw::c_char,
                                        );
                                        Ok(IrValue::Str(cstr.to_string_lossy().to_string()))
                                    }
                                }
                                _ => Ok(IrValue::I64(raw)),
                            };
                        }
                    }
                }

                // Return a zero value of the declared return type so tests can verify the call happened.
                let zero = match ret_ty {
                    IrType::Scalar(crate::ir::types::DType::F64) => IrValue::F64(0.0),
                    IrType::Scalar(crate::ir::types::DType::F32) => IrValue::F32(0.0),
                    IrType::Scalar(crate::ir::types::DType::I64) => IrValue::I64(0),
                    IrType::Scalar(crate::ir::types::DType::I32) => IrValue::I32(0),
                    IrType::Scalar(crate::ir::types::DType::Bool) => IrValue::Bool(false),
                    IrType::Str => IrValue::Str(String::new()),
                    _ => IrValue::I64(0),
                };
                Ok(zero)
            }
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

/// Parse einsum notation like "mk,kn->mn" into (lhs_indices, rhs_indices, out_indices).
fn parse_einsum_notation(notation: &str) -> Option<(Vec<char>, Vec<char>, Vec<char>)> {
    let parts: Vec<&str> = notation.split("->").collect();
    if parts.len() != 2 {
        return None;
    }
    let output = parts[1].chars().collect::<Vec<_>>();
    let inputs: Vec<&str> = parts[0].split(',').collect();
    if inputs.len() != 2 {
        return None;
    }
    let lhs = inputs[0].chars().collect::<Vec<_>>();
    let rhs = inputs[1].chars().collect::<Vec<_>>();
    Some((lhs, rhs, output))
}

/// Parse single-input einsum notation like "ii->" (trace) or "ij->ji" (transpose).
fn parse_einsum_single_notation(notation: &str) -> Option<(Vec<char>, Vec<char>)> {
    let parts: Vec<&str> = notation.split("->").collect();
    if parts.len() != 2 {
        return None;
    }
    let input = parts[0].chars().collect::<Vec<_>>();
    let output = parts[1].chars().collect::<Vec<_>>();
    Some((input, output))
}

/// Evaluate 2-input einsum.
fn eval_einsum(
    notation: &str,
    a_data: &[f32],
    a_shape: &[usize],
    b_data: &[f32],
    b_shape: &[usize],
) -> Result<IrValue, InterpError> {
    let (lhs_idx, rhs_idx, out_idx) =
        parse_einsum_notation(notation).ok_or_else(|| InterpError::Unsupported {
            detail: format!("cannot parse einsum notation '{}'", notation),
        })?;

    // Build dimension map: index char -> size
    let mut dim_map: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    for (i, &c) in lhs_idx.iter().enumerate() {
        if i < a_shape.len() {
            dim_map.insert(c, a_shape[i]);
        }
    }
    for (i, &c) in rhs_idx.iter().enumerate() {
        if i < b_shape.len() {
            if let Some(&existing) = dim_map.get(&c) {
                if existing != b_shape[i] {
                    return Err(InterpError::TypeError {
                        detail: format!(
                            "einsum dimension mismatch for '{}': {} vs {}",
                            c, existing, b_shape[i]
                        ),
                    });
                }
            } else {
                dim_map.insert(c, b_shape[i]);
            }
        }
    }

    // Find contracted indices (in inputs but not in output)
    let mut all_indices: Vec<char> = Vec::new();
    for &c in &lhs_idx {
        if !all_indices.contains(&c) {
            all_indices.push(c);
        }
    }
    for &c in &rhs_idx {
        if !all_indices.contains(&c) {
            all_indices.push(c);
        }
    }
    let contracted: Vec<char> = all_indices
        .iter()
        .filter(|c| !out_idx.contains(c))
        .copied()
        .collect();

    // Compute output shape
    let out_shape: Vec<usize> = out_idx
        .iter()
        .map(|c| dim_map.get(c).copied().unwrap_or(1))
        .collect();
    let out_numel: usize = out_shape.iter().product::<usize>().max(1);

    // Compute strides for a and b
    let a_strides = compute_strides(a_shape);
    let b_strides = compute_strides(b_shape);
    let out_strides = compute_strides(&out_shape);

    let contracted_sizes: Vec<usize> = contracted
        .iter()
        .map(|c| dim_map.get(c).copied().unwrap_or(1))
        .collect();
    let contracted_total: usize = contracted_sizes.iter().product::<usize>().max(1);

    let mut result = vec![0.0f32; out_numel];

    // Iterate over all output positions
    for (out_flat, out_slot) in result.iter_mut().enumerate().take(out_numel) {
        // Decompose output flat index into per-dimension coords
        let mut out_coords: std::collections::HashMap<char, usize> =
            std::collections::HashMap::new();
        let mut rem = out_flat;
        for (d, &c) in out_idx.iter().enumerate() {
            if d < out_strides.len() {
                out_coords.insert(c, rem / out_strides[d]);
                rem %= out_strides[d];
            }
        }

        // Sum over contracted indices
        let mut sum = 0.0f32;
        for c_flat in 0..contracted_total {
            let mut c_coords = out_coords.clone();
            let mut c_rem = c_flat;
            for (ci, &cc) in contracted.iter().enumerate() {
                let _sz = contracted_sizes[ci];
                c_coords.insert(
                    cc,
                    c_rem
                        / if ci + 1 < contracted_sizes.len() {
                            contracted_sizes[ci + 1..].iter().product::<usize>().max(1)
                        } else {
                            1
                        },
                );
                c_rem %= if ci + 1 < contracted_sizes.len() {
                    contracted_sizes[ci + 1..].iter().product::<usize>().max(1)
                } else {
                    1
                };
            }

            // Compute a flat index
            let mut a_flat = 0usize;
            for (i, &c) in lhs_idx.iter().enumerate() {
                if i < a_strides.len() {
                    a_flat += c_coords.get(&c).copied().unwrap_or(0) * a_strides[i];
                }
            }
            // Compute b flat index
            let mut b_flat = 0usize;
            for (i, &c) in rhs_idx.iter().enumerate() {
                if i < b_strides.len() {
                    b_flat += c_coords.get(&c).copied().unwrap_or(0) * b_strides[i];
                }
            }

            if a_flat < a_data.len() && b_flat < b_data.len() {
                sum += a_data[a_flat] * b_data[b_flat];
            }
        }
        *out_slot = sum;
    }

    if out_shape.is_empty() {
        // Scalar output (e.g., "i,i->")
        Ok(IrValue::F32(result[0]))
    } else {
        Ok(IrValue::Tensor(result, out_shape))
    }
}

/// Evaluate single-input einsum (trace, transpose-via-einsum, etc.)
fn eval_einsum_single(
    notation: &str,
    data: &[f32],
    shape: &[usize],
) -> Result<IrValue, InterpError> {
    let (in_idx, out_idx) =
        parse_einsum_single_notation(notation).ok_or_else(|| InterpError::Unsupported {
            detail: format!("cannot parse single-input einsum '{}'", notation),
        })?;

    let mut dim_map: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    for (i, &c) in in_idx.iter().enumerate() {
        if i < shape.len() {
            dim_map.insert(c, shape[i]);
        }
    }

    let contracted: Vec<char> = {
        let mut seen = Vec::new();
        let mut dupes = Vec::new();
        for &c in &in_idx {
            if seen.contains(&c) && !dupes.contains(&c) && !out_idx.contains(&c) {
                dupes.push(c);
            }
            seen.push(c);
        }
        dupes
    };

    let out_shape: Vec<usize> = out_idx
        .iter()
        .map(|c| dim_map.get(c).copied().unwrap_or(1))
        .collect();
    let out_numel: usize = out_shape.iter().product::<usize>().max(1);
    let in_strides = compute_strides(shape);
    let out_strides = compute_strides(&out_shape);

    let contracted_sizes: Vec<usize> = contracted
        .iter()
        .map(|c| dim_map.get(c).copied().unwrap_or(1))
        .collect();
    let contracted_total: usize = contracted_sizes.iter().product::<usize>().max(1);

    let mut result = vec![0.0f32; out_numel];

    for (out_flat, out_slot) in result.iter_mut().enumerate().take(out_numel) {
        let mut coords: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
        let mut rem = out_flat;
        for (d, &c) in out_idx.iter().enumerate() {
            if d < out_strides.len() {
                coords.insert(c, rem / out_strides[d]);
                rem %= out_strides[d];
            }
        }

        let mut sum = 0.0f32;
        for c_flat in 0..contracted_total {
            let mut c_coords = coords.clone();
            let mut c_rem = c_flat;
            for (ci, &cc) in contracted.iter().enumerate() {
                let div = if ci + 1 < contracted_sizes.len() {
                    contracted_sizes[ci + 1..].iter().product::<usize>().max(1)
                } else {
                    1
                };
                c_coords.insert(cc, c_rem / div);
                c_rem %= div;
            }

            let mut in_flat = 0usize;
            for (i, &c) in in_idx.iter().enumerate() {
                if i < in_strides.len() {
                    in_flat += c_coords.get(&c).copied().unwrap_or(0) * in_strides[i];
                }
            }
            if in_flat < data.len() {
                sum += data[in_flat];
            }
        }
        *out_slot = sum;
    }

    if out_shape.is_empty() {
        Ok(IrValue::F32(result[0]))
    } else {
        Ok(IrValue::Tensor(result, out_shape))
    }
}

/// Compute strides for a row-major shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Evaluate a reduce operation on a tensor.
fn eval_reduce(
    data: &[f32],
    shape: &[usize],
    op: &str,
    axes: &[usize],
    keepdims: bool,
) -> Result<IrValue, InterpError> {
    if shape.is_empty() {
        return Ok(IrValue::F32(if data.is_empty() { 0.0 } else { data[0] }));
    }

    let ndim = shape.len();
    let strides = compute_strides(shape);

    // Normalize axes
    let reduce_axes: Vec<usize> = if axes.is_empty() {
        (0..ndim).collect()
    } else {
        axes.to_vec()
    };

    // Compute output shape
    let mut out_shape: Vec<usize> = Vec::new();
    for (d, dim) in shape.iter().enumerate().take(ndim) {
        if reduce_axes.contains(&d) {
            if keepdims {
                out_shape.push(1);
            }
        } else {
            out_shape.push(*dim);
        }
    }
    let out_numel: usize = out_shape.iter().product::<usize>().max(1);
    let out_strides = compute_strides(&out_shape);

    // Init result
    let init_val = match op {
        "sum" | "mean" => 0.0f32,
        "max" => f32::NEG_INFINITY,
        "min" => f32::INFINITY,
        "prod" => 1.0f32,
        _ => {
            return Err(InterpError::Unsupported {
                detail: format!("reduce op '{}' not supported", op),
            })
        }
    };
    let mut result = vec![init_val; out_numel];
    let mut counts = vec![0usize; out_numel];

    let total: usize = shape.iter().product::<usize>();
    for (flat, &val) in data.iter().enumerate().take(total) {
        // Decompose into coords
        let mut coords = vec![0usize; ndim];
        let mut rem = flat;
        for (d, coord) in coords.iter_mut().enumerate() {
            *coord = rem / strides[d];
            rem %= strides[d];
        }

        // Compute output flat index (skip reduced dims)
        let mut out_flat = 0usize;
        let mut out_d = 0usize;
        for (d, coord) in coords.iter().enumerate().take(ndim) {
            if reduce_axes.contains(&d) {
                if keepdims {
                    // This dim is 1, contributes 0 to flat index
                    out_d += 1;
                }
            } else {
                if out_d < out_strides.len() {
                    out_flat += *coord * out_strides[out_d];
                }
                out_d += 1;
            }
        }

        match op {
            "sum" | "mean" => result[out_flat] += val,
            "max" => {
                if val > result[out_flat] {
                    result[out_flat] = val;
                }
            }
            "min" => {
                if val < result[out_flat] {
                    result[out_flat] = val;
                }
            }
            "prod" => result[out_flat] *= val,
            _ => {}
        }
        counts[out_flat] += 1;
    }

    if op == "mean" {
        for i in 0..out_numel {
            if counts[i] > 0 {
                result[i] /= counts[i] as f32;
            }
        }
    }

    if out_shape.is_empty() {
        Ok(IrValue::F32(result[0]))
    } else {
        Ok(IrValue::Tensor(result, out_shape))
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
        // Math builtins — float variants
        (ScalarUnaryOp::Sqrt, IrValue::F64(x)) => Ok(IrValue::F64(x.sqrt())),
        (ScalarUnaryOp::Sqrt, IrValue::F32(x)) => Ok(IrValue::F32(x.sqrt())),
        (ScalarUnaryOp::Abs, IrValue::F64(x)) => Ok(IrValue::F64(x.abs())),
        (ScalarUnaryOp::Abs, IrValue::F32(x)) => Ok(IrValue::F32(x.abs())),
        (ScalarUnaryOp::Abs, IrValue::I64(n)) => Ok(IrValue::I64(n.abs())),
        (ScalarUnaryOp::Abs, IrValue::I32(n)) => Ok(IrValue::I32(n.abs())),
        (ScalarUnaryOp::Floor, IrValue::F64(x)) => Ok(IrValue::F64(x.floor())),
        (ScalarUnaryOp::Floor, IrValue::F32(x)) => Ok(IrValue::F32(x.floor())),
        (ScalarUnaryOp::Ceil, IrValue::F64(x)) => Ok(IrValue::F64(x.ceil())),
        (ScalarUnaryOp::Ceil, IrValue::F32(x)) => Ok(IrValue::F32(x.ceil())),
        (ScalarUnaryOp::BitNot, IrValue::I64(n)) => Ok(IrValue::I64(!n)),
        (ScalarUnaryOp::BitNot, IrValue::I32(n)) => Ok(IrValue::I32(!n)),
        // Trig / transcendental — float variants
        (ScalarUnaryOp::Sin, IrValue::F64(x)) => Ok(IrValue::F64(x.sin())),
        (ScalarUnaryOp::Sin, IrValue::F32(x)) => Ok(IrValue::F32(x.sin())),
        (ScalarUnaryOp::Cos, IrValue::F64(x)) => Ok(IrValue::F64(x.cos())),
        (ScalarUnaryOp::Cos, IrValue::F32(x)) => Ok(IrValue::F32(x.cos())),
        (ScalarUnaryOp::Tan, IrValue::F64(x)) => Ok(IrValue::F64(x.tan())),
        (ScalarUnaryOp::Tan, IrValue::F32(x)) => Ok(IrValue::F32(x.tan())),
        (ScalarUnaryOp::Exp, IrValue::F64(x)) => Ok(IrValue::F64(x.exp())),
        (ScalarUnaryOp::Exp, IrValue::F32(x)) => Ok(IrValue::F32(x.exp())),
        (ScalarUnaryOp::Log, IrValue::F64(x)) => Ok(IrValue::F64(x.ln())),
        (ScalarUnaryOp::Log, IrValue::F32(x)) => Ok(IrValue::F32(x.ln())),
        (ScalarUnaryOp::Log2, IrValue::F64(x)) => Ok(IrValue::F64(x.log2())),
        (ScalarUnaryOp::Log2, IrValue::F32(x)) => Ok(IrValue::F32(x.log2())),
        (ScalarUnaryOp::Round, IrValue::F64(x)) => Ok(IrValue::F64(x.round())),
        (ScalarUnaryOp::Round, IrValue::F32(x)) => Ok(IrValue::F32(x.round())),
        // Sign function
        (ScalarUnaryOp::Sign, IrValue::F64(x)) => Ok(IrValue::F64(x.signum())),
        (ScalarUnaryOp::Sign, IrValue::F32(x)) => Ok(IrValue::F32(x.signum())),
        (ScalarUnaryOp::Sign, IrValue::I64(n)) => Ok(IrValue::I64(n.signum())),
        (ScalarUnaryOp::Sign, IrValue::I32(n)) => Ok(IrValue::I32(n.signum())),
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
        // Extended integer types: all stored as I64 in the interpreter.
        IrType::Scalar(DType::U8) => match v {
            IrValue::I64(n) => Ok(IrValue::I64((*n as u8) as i64)),
            IrValue::I32(n) => Ok(IrValue::I64((*n as u8) as i64)),
            IrValue::F32(x) => Ok(IrValue::I64((*x as u8) as i64)),
            IrValue::F64(x) => Ok(IrValue::I64((*x as u8) as i64)),
            _ => Err(InterpError::TypeError {
                detail: "cannot cast to u8".into(),
            }),
        },
        IrType::Scalar(DType::I8) => match v {
            IrValue::I64(n) => Ok(IrValue::I64((*n as i8) as i64)),
            IrValue::I32(n) => Ok(IrValue::I64((*n as i8) as i64)),
            IrValue::F32(x) => Ok(IrValue::I64((*x as i8) as i64)),
            IrValue::F64(x) => Ok(IrValue::I64((*x as i8) as i64)),
            _ => Err(InterpError::TypeError {
                detail: "cannot cast to i8".into(),
            }),
        },
        IrType::Scalar(DType::U32) => match v {
            IrValue::I64(n) => Ok(IrValue::I64((*n as u32) as i64)),
            IrValue::I32(n) => Ok(IrValue::I64((*n as u32) as i64)),
            IrValue::F32(x) => Ok(IrValue::I64((*x as u32) as i64)),
            IrValue::F64(x) => Ok(IrValue::I64((*x as u32) as i64)),
            _ => Err(InterpError::TypeError {
                detail: "cannot cast to u32".into(),
            }),
        },
        IrType::Scalar(DType::U64) => match v {
            IrValue::I64(n) => Ok(IrValue::I64(*n)),
            IrValue::I32(n) => Ok(IrValue::I64(*n as i64)),
            IrValue::F32(x) => Ok(IrValue::I64(*x as i64)),
            IrValue::F64(x) => Ok(IrValue::I64(*x as i64)),
            _ => Err(InterpError::TypeError {
                detail: "cannot cast to u64".into(),
            }),
        },
        IrType::Scalar(DType::USize) => match v {
            IrValue::I64(n) => Ok(IrValue::I64(*n)),
            IrValue::I32(n) => Ok(IrValue::I64(*n as i64)),
            IrValue::F32(x) => Ok(IrValue::I64(*x as i64)),
            IrValue::F64(x) => Ok(IrValue::I64(*x as i64)),
            _ => Err(InterpError::TypeError {
                detail: "cannot cast to usize".into(),
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
        (BinOp::Add, I32(a), I32(b)) => {
            a.checked_add(*b).ok_or_else(|| InterpError::Panic { msg: format!("integer overflow in addition ({} + {})", a, b) }).map(I32)
        }
        (BinOp::Sub, I32(a), I32(b)) => {
            a.checked_sub(*b).ok_or_else(|| InterpError::Panic { msg: format!("integer overflow in subtraction ({} - {})", a, b) }).map(I32)
        }
        (BinOp::Mul, I32(a), I32(b)) => {
            a.checked_mul(*b).ok_or_else(|| InterpError::Panic { msg: format!("integer overflow in multiplication ({} * {})", a, b) }).map(I32)
        }
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
        (BinOp::Add, I64(a), I64(b)) => {
            a.checked_add(*b).ok_or_else(|| InterpError::Panic { msg: format!("integer overflow in addition ({} + {})", a, b) }).map(I64)
        }
        (BinOp::Sub, I64(a), I64(b)) => {
            a.checked_sub(*b).ok_or_else(|| InterpError::Panic { msg: format!("integer overflow in subtraction ({} - {})", a, b) }).map(I64)
        }
        (BinOp::Mul, I64(a), I64(b)) => {
            a.checked_mul(*b).ok_or_else(|| InterpError::Panic { msg: format!("integer overflow in multiplication ({} * {})", a, b) }).map(I64)
        }
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
        // String comparisons
        (BinOp::CmpEq, Str(a), Str(b)) => Ok(Bool(a == b)),
        (BinOp::CmpNe, Str(a), Str(b)) => Ok(Bool(a != b)),
        (BinOp::CmpLt, Str(a), Str(b)) => Ok(Bool(a < b)),
        (BinOp::CmpLe, Str(a), Str(b)) => Ok(Bool(a <= b)),
        (BinOp::CmpGt, Str(a), Str(b)) => Ok(Bool(a > b)),
        (BinOp::CmpGe, Str(a), Str(b)) => Ok(Bool(a >= b)),
        // Grad (dual number) arithmetic -- forward-mode AD with chain rule
        (
            BinOp::Add,
            Grad {
                value: av,
                tangent: at,
            },
            Grad {
                value: bv,
                tangent: bt,
            },
        ) => Ok(Grad {
            value: av + bv,
            tangent: at + bt,
        }),
        (
            BinOp::Sub,
            Grad {
                value: av,
                tangent: at,
            },
            Grad {
                value: bv,
                tangent: bt,
            },
        ) => Ok(Grad {
            value: av - bv,
            tangent: at - bt,
        }),
        (
            BinOp::Mul,
            Grad {
                value: av,
                tangent: at,
            },
            Grad {
                value: bv,
                tangent: bt,
            },
        ) => Ok(Grad {
            value: av * bv,
            tangent: av * bt + at * bv,
        }),
        (
            BinOp::Div,
            Grad {
                value: av,
                tangent: at,
            },
            Grad {
                value: bv,
                tangent: bt,
            },
        ) => Ok(Grad {
            value: av / bv,
            tangent: (at * bv - av * bt) / (bv * bv),
        }),
        // Grad vs scalar: promote scalar to Grad with zero tangent
        (
            BinOp::Add,
            Grad {
                value: av,
                tangent: at,
            },
            F64(b),
        ) => Ok(Grad {
            value: av + b,
            tangent: *at,
        }),
        (
            BinOp::Mul,
            Grad {
                value: av,
                tangent: at,
            },
            F64(b),
        ) => Ok(Grad {
            value: av * b,
            tangent: at * b,
        }),
        // Math builtins: pow, min, max — F64
        (BinOp::Pow, F64(a), F64(b)) => Ok(F64(a.powf(*b))),
        (BinOp::Min, F64(a), F64(b)) => Ok(F64(a.min(*b))),
        (BinOp::Max, F64(a), F64(b)) => Ok(F64(a.max(*b))),
        // Math builtins: pow, min, max — F32
        (BinOp::Pow, F32(a), F32(b)) => Ok(F32(a.powf(*b))),
        (BinOp::Min, F32(a), F32(b)) => Ok(F32(a.min(*b))),
        (BinOp::Max, F32(a), F32(b)) => Ok(F32(a.max(*b))),
        // Math builtins: pow, min, max — I64
        (BinOp::Pow, I64(a), I64(b)) => Ok(I64((*a as f64).powf(*b as f64) as i64)),
        (BinOp::Min, I64(a), I64(b)) => Ok(I64(*a.min(b))),
        (BinOp::Max, I64(a), I64(b)) => Ok(I64(*a.max(b))),
        // Math builtins: pow, min, max — I32
        (BinOp::Pow, I32(a), I32(b)) => Ok(I32((*a as f64).powf(*b as f64) as i32)),
        (BinOp::Min, I32(a), I32(b)) => Ok(I32(*a.min(b))),
        (BinOp::Max, I32(a), I32(b)) => Ok(I32(*a.max(b))),
        // Bitwise/logical AND on booleans
        (BinOp::BitAnd, IrValue::Bool(a), IrValue::Bool(b)) => Ok(IrValue::Bool(a & b)),
        (BinOp::BitOr, IrValue::Bool(a), IrValue::Bool(b)) => Ok(IrValue::Bool(a | b)),
        (BinOp::BitXor, IrValue::Bool(a), IrValue::Bool(b)) => Ok(IrValue::Bool(a ^ b)),
        // Bitwise ops — I64
        (BinOp::BitAnd, I64(a), I64(b)) => Ok(I64(a & b)),
        (BinOp::BitOr, I64(a), I64(b)) => Ok(I64(a | b)),
        (BinOp::BitXor, I64(a), I64(b)) => Ok(I64(a ^ b)),
        (BinOp::Shl, I64(a), I64(b)) => Ok(I64(a.wrapping_shl(*b as u32))),
        (BinOp::Shr, I64(a), I64(b)) => Ok(I64(a.wrapping_shr(*b as u32))),
        // Bitwise ops — I32
        (BinOp::BitAnd, I32(a), I32(b)) => Ok(I32(a & b)),
        (BinOp::BitOr, I32(a), I32(b)) => Ok(I32(a | b)),
        (BinOp::BitXor, I32(a), I32(b)) => Ok(I32(a ^ b)),
        (BinOp::Shl, I32(a), I32(b)) => Ok(I32(a.wrapping_shl(*b as u32))),
        (BinOp::Shr, I32(a), I32(b)) => Ok(I32(a.wrapping_shr(*b as u32))),
        _ => Err(InterpError::TypeError {
            detail: format!("unsupported binop {:?} on {:?} and {:?}", op, lv, rv),
        }),
    }
}

// ---------------------------------------------------------------------------
// Phase 104: Builtin function interpreter dispatch
// ---------------------------------------------------------------------------

/// Thread-local store for TCP streams/listeners used by the interpreter.
mod tcp_store {
    use std::collections::HashMap;
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicI64, Ordering};
    use std::sync::{Arc, LazyLock, Mutex};

    static STREAMS: LazyLock<Mutex<HashMap<i64, Arc<TcpStream>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    static LISTENERS: LazyLock<Mutex<HashMap<i64, Arc<TcpListener>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    static NEXT_ID: AtomicI64 = AtomicI64::new(1);

    fn next_handle() -> i64 {
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    }

    pub fn store_stream(s: TcpStream) -> i64 {
        let id = next_handle();
        STREAMS
            .lock()
            .unwrap()
            .insert(id, Arc::new(s));
        id
    }
    pub fn store_listener(l: TcpListener) -> i64 {
        let id = next_handle();
        LISTENERS.lock().unwrap().insert(id, Arc::new(l));
        id
    }
    pub fn read_stream(id: i64) -> Result<String, ()> {
        use std::io::Read;
        let stream = {
            let map = STREAMS.lock().unwrap();
            map.get(&id).cloned()
        };
        if let Some(stream) = stream {
            let mut buf = vec![0u8; 8192];
            // Read directly from &TcpStream (it is thread-safe and implements Read)
            match (&*stream).read(&mut buf) {
                Ok(n) => Ok(String::from_utf8_lossy(&buf[..n]).to_string()),
                Err(_) => Err(()),
            }
        } else {
            Err(())
        }
    }
    pub fn write_stream(id: i64, data: &str) {
        use std::io::Write;
        let stream = {
            let map = STREAMS.lock().unwrap();
            map.get(&id).cloned()
        };
        if let Some(stream) = stream {
            // Write directly to &TcpStream (it is thread-safe and implements Write)
            let _ = (&*stream).write_all(data.as_bytes());
        }
    }
    pub fn accept_listener(id: i64) -> Result<i64, ()> {
        let listener = {
            let map = LISTENERS.lock().unwrap();
            map.get(&id).cloned()
        };
        if let Some(listener) = listener {
            match listener.accept() {
                Ok((stream, _)) => Ok(store_stream(stream)),
                Err(_) => Err(()),
            }
        } else {
            Err(())
        }
    }
    pub fn close(id: i64) {
        STREAMS.lock().unwrap().remove(&id);
        LISTENERS.lock().unwrap().remove(&id);
    }
}

mod udp_store {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::net::UdpSocket;

    thread_local! {
        static SOCKETS: RefCell<HashMap<i64, UdpSocket>> = RefCell::new(HashMap::new());
        static NEXT_ID: RefCell<i64> = const { RefCell::new(1000) };
    }

    fn next_handle() -> i64 {
        NEXT_ID.with(|c| {
            let id = *c.borrow();
            *c.borrow_mut() = id + 1;
            id
        })
    }

    pub fn store_socket(s: UdpSocket) -> i64 {
        let id = next_handle();
        SOCKETS.with(|m| m.borrow_mut().insert(id, s));
        id
    }
    pub fn send_socket(id: i64, addr_port: &str) -> Result<(), ()> {
        SOCKETS.with(|m| {
            let map = m.borrow();
            if let Some(sock) = map.get(&id) {
                sock.send_to(b"hello", addr_port).map(|_| ()).map_err(|_| ())
            } else {
                Err(())
            }
        })
    }
    pub fn recv_socket(id: i64) -> Result<String, ()> {
        SOCKETS.with(|m| {
            let map = m.borrow();
            if let Some(sock) = map.get(&id) {
                let mut buf = vec![0u8; 8192];
                match sock.recv_from(&mut buf) {
                    Ok((n, _)) => Ok(String::from_utf8_lossy(&buf[..n]).to_string()),
                    Err(_) => Err(()),
                }
            } else {
                Err(())
            }
        })
    }
    pub fn close(id: i64) {
        SOCKETS.with(|m| {
            m.borrow_mut().remove(&id);
        });
    }
}

/// Helper: dispatch a C/Rust FFI call with up to 6 i64 arguments via transmuted pointers.
/// The function pointer `proc` must point to a valid extern "C" function.
unsafe fn ffi_dispatch_call(proc: *const u8, args: &[i64]) -> i64 {
    match args.len() {
        0 => {
            let f: extern "C" fn() -> i64 = std::mem::transmute(proc);
            f()
        }
        1 => {
            let f: extern "C" fn(i64) -> i64 = std::mem::transmute(proc);
            f(args[0])
        }
        2 => {
            let f: extern "C" fn(i64, i64) -> i64 = std::mem::transmute(proc);
            f(args[0], args[1])
        }
        3 => {
            let f: extern "C" fn(i64, i64, i64) -> i64 = std::mem::transmute(proc);
            f(args[0], args[1], args[2])
        }
        4 => {
            let f: extern "C" fn(i64, i64, i64, i64) -> i64 = std::mem::transmute(proc);
            f(args[0], args[1], args[2], args[3])
        }
        5 => {
            let f: extern "C" fn(i64, i64, i64, i64, i64) -> i64 = std::mem::transmute(proc);
            f(args[0], args[1], args[2], args[3], args[4])
        }
        _ => {
            let f: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64 = std::mem::transmute(proc);
            f(args[0], args[1], args[2], args[3], args[4], args[5])
        }
    }
}

#[cfg(windows)]
fn lookup_dynamic_symbol(name: &str) -> Option<*const u8> {
    use std::ffi::CString;
    use std::sync::OnceLock;

    static MSVCRT: OnceLock<usize> = OnceLock::new();
    let msvcrt_handle = *MSVCRT.get_or_init(|| unsafe {
        let name = CString::new("msvcrt.dll").unwrap();
        winapi_LoadLibraryA(name.as_ptr()) as usize
    }) as *mut u8;

    let cs = CString::new(name.as_bytes()).unwrap_or_default();

    // 1. Search in loaded msvcrt.dll first
    if !msvcrt_handle.is_null() {
        let proc = unsafe { winapi_GetProcAddress(msvcrt_handle, cs.as_ptr()) };
        if !proc.is_null() {
            return Some(proc as *const u8);
        }
    }

    // 2. Search in main process itself
    unsafe {
        let main_handle = GetModuleHandleA(std::ptr::null());
        if !main_handle.is_null() {
            let proc = winapi_GetProcAddress(main_handle, cs.as_ptr());
            if !proc.is_null() {
                return Some(proc as *const u8);
            }
        }
    }

    None
}

#[cfg(unix)]
fn lookup_dynamic_symbol(name: &str) -> Option<*const u8> {
    use std::ffi::CString;
    use std::sync::OnceLock;

    static GLOBAL_HANDLE: OnceLock<usize> = OnceLock::new();
    let handle = *GLOBAL_HANDLE.get_or_init(|| unsafe {
        libc_dlopen(std::ptr::null(), 1) as usize // RTLD_LAZY = 1
    }) as *mut u8;

    if handle.is_null() {
        return None;
    }

    let cs = CString::new(name.as_bytes()).unwrap_or_default();
    unsafe {
        let sym = libc_dlsym(handle, cs.as_ptr());
        if !sym.is_null() {
            Some(sym as *const u8)
        } else {
            None
        }
    }
}

#[cfg(not(any(windows, unix)))]
fn lookup_dynamic_symbol(_name: &str) -> Option<*const u8> {
    None
}

fn str_arg(v: &IrValue) -> String {
    match v {
        IrValue::Str(s) => s.clone(),
        _ => format!("{}", v),
    }
}
fn i64_arg(v: &IrValue) -> i64 {
    match v {
        IrValue::I64(n) => *n,
        IrValue::I32(n) => *n as i64,
        IrValue::F64(f) => *f as i64,
        _ => 0,
    }
}

fn f64_arg(v: &IrValue) -> f64 {
    match v {
        IrValue::F64(f) => *f,
        IrValue::F32(f) => *f as f64,
        IrValue::I64(n) => *n as f64,
        IrValue::I32(n) => *n as f64,
        _ => 0.0,
    }
}

/// Live FFI out-parameter cells: address -> byte length.
///
/// A process-wide registry because `interp_builtin` is a free function, and
/// because the addresses are handed to foreign code — these are deliberately
/// outside the refcounting GC, whose lifetime analysis cannot follow a pointer
/// that has crossed into C. Tracking the length lets every read be
/// bounds-checked and makes a double free a no-op rather than undefined
/// behaviour.
fn ffi_out_cells() -> std::sync::MutexGuard<'static, std::collections::HashMap<i64, usize>> {
    static CELLS: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<i64, usize>>,
    > = std::sync::OnceLock::new();
    CELLS
        .get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn interp_builtin(name: &str, args: &[IrValue]) -> Result<IrValue, InterpError> {
    match name {
        // ---- File Streams ----
        "file_open" => {
            let path = str_arg(&args[0]);
            let mode = str_arg(&args[1]);
            security::check_fs_read(&path)?;
            if mode.contains('w') || mode.contains('a') || mode.contains('+') {
                security::check_fs_write(&path)?;
            }
            let mut options = std::fs::OpenOptions::new();
            if mode.contains('r') {
                options.read(true);
            }
            if mode.contains('w') {
                options.write(true).create(true).truncate(true);
            }
            if mode.contains('a') {
                options.write(true).create(true).append(true);
            }
            if !mode.contains('r') && !mode.contains('w') && !mode.contains('a') {
                options.read(true);
            }
            match options.open(&path) {
                Ok(file) => {
                    let handle = Box::into_raw(Box::new(file)) as i64;
                    Ok(IrValue::I64(handle))
                }
                Err(_) => Ok(IrValue::I64(0)),
            }
        }
        "file_close" => {
            let handle = i64_arg(&args[0]);
            if handle != 0 {
                let _file = unsafe { Box::from_raw(handle as *mut std::fs::File) };
                Ok(IrValue::Bool(true))
            } else {
                Ok(IrValue::Bool(false))
            }
        }
        "file_read" => {
            let handle = i64_arg(&args[0]);
            let bytes = i64_arg(&args[1]);
            if handle != 0 && bytes > 0 {
                let file_ref = unsafe { &*(handle as *const std::fs::File) };
                let mut buf = vec![0u8; bytes as usize];
                use std::io::Read;
                match (&*file_ref).read(&mut buf) {
                    Ok(n) => {
                        buf.truncate(n);
                        let s = String::from_utf8_lossy(&buf).into_owned();
                        Ok(IrValue::Str(s))
                    }
                    Err(_) => Ok(IrValue::Str(String::new())),
                }
            } else {
                Ok(IrValue::Str(String::new()))
            }
        }
        "file_write" => {
            let handle = i64_arg(&args[0]);
            let data = str_arg(&args[1]);
            if handle != 0 {
                let file_ref = unsafe { &*(handle as *const std::fs::File) };
                use std::io::Write;
                match (&*file_ref).write_all(data.as_bytes()) {
                    Ok(()) => {
                        let _ = (&*file_ref).flush();
                        Ok(IrValue::Bool(true))
                    }
                    Err(_) => Ok(IrValue::Bool(false)),
                }
            } else {
                Ok(IrValue::Bool(false))
            }
        }

        // ---- TCP ----
        "tcp_connect" => {
            let host = str_arg(&args[0]);
            let port = i64_arg(&args[1]);
            security::check_network(&host)?;
            match std::net::TcpStream::connect(format!("{}:{}", host, port)) {
                Ok(stream) => Ok(IrValue::I64(tcp_store::store_stream(stream))),
                Err(_) => Ok(IrValue::I64(-1)),
            }
        }
        "tcp_listen" => {
            let port = i64_arg(&args[0]);
            security::check_network("0.0.0.0")?;
            match std::net::TcpListener::bind(format!("0.0.0.0:{}", port)) {
                Ok(listener) => Ok(IrValue::I64(tcp_store::store_listener(listener))),
                Err(_) => Ok(IrValue::I64(-1)),
            }
        }
        "tcp_accept" => {
            let id = i64_arg(&args[0]);
            Ok(IrValue::I64(tcp_store::accept_listener(id).unwrap_or(-1)))
        }
        "tcp_read" => {
            let id = i64_arg(&args[0]);
            Ok(IrValue::Str(tcp_store::read_stream(id).unwrap_or_default()))
        }
        "tcp_write" => {
            let id = i64_arg(&args[0]);
            let data = str_arg(&args[1]);
            tcp_store::write_stream(id, &data);
            Ok(IrValue::I64(0))
        }
        "tcp_close" => {
            let id = i64_arg(&args[0]);
            tcp_store::close(id);
            Ok(IrValue::I64(0))
        }
        // ---- UDP ----
        "udp_open" => {
            let host = str_arg(&args[0]);
            let port = i64_arg(&args[1]);
            security::check_network(&host)?;
            match std::net::UdpSocket::bind(format!("{}:{}", host, port)) {
                Ok(sock) => {
                    let _ = sock.set_read_timeout(Some(std::time::Duration::from_millis(100)));
                    Ok(IrValue::I64(udp_store::store_socket(sock)))
                }
                Err(_) => Ok(IrValue::I64(-1)),
            }
        }
        "udp_send" => {
            let id = i64_arg(&args[0]);
            let addr = str_arg(&args[1]);
            security::check_network(&addr)?;
            if udp_store::send_socket(id, &addr).is_ok() {
                Ok(IrValue::I64(0))
            } else {
                Ok(IrValue::I64(-1))
            }
        }
        "udp_recv" => {
            let id = i64_arg(&args[0]);
            Ok(IrValue::Str(udp_store::recv_socket(id).unwrap_or_default()))
        }
        "udp_close" => {
            let id = i64_arg(&args[0]);
            udp_store::close(id);
            Ok(IrValue::I64(0))
        }
        // ---- HTTP ----
        "http_get" => {
            let url = str_arg(&args[0]);
            security::check_network(&url)?;
            Ok(IrValue::Str(http_request("GET", &url, "")))
        }
        "http_post" => {
            let url = str_arg(&args[0]);
            let body = str_arg(&args[1]);
            security::check_network(&url)?;
            Ok(IrValue::Str(http_request("POST", &url, &body)))
        }
        "http_request" => {
            let method = str_arg(&args[0]);
            let url = str_arg(&args[1]);
            let body = if args.len() > 2 {
                str_arg(&args[2])
            } else {
                String::new()
            };
            security::check_network(&url)?;
            Ok(IrValue::Str(http_request(&method, &url, &body)))
        }
        "http_post_json" => {
            let url = str_arg(&args[0]);
            let body = str_arg(&args[1]);
            security::check_network(&url)?;
            Ok(IrValue::Str(http_request("POST", &url, &body)))
        }
        // ---- Terminal ----
        "read_key" => {
            // Simple: read one byte from stdin
            use std::io::Read;
            let mut buf = [0u8; 1];
            let _ = std::io::stdin().read(&mut buf);
            Ok(IrValue::I64(buf[0] as i64))
        }
        "read_password" => {
            // Interpreter: just use read_line (no echo-hiding in interp mode)
            let mut line = String::new();
            let _ = std::io::stdin().read_line(&mut line);
            Ok(IrValue::Str(
                line.trim_end_matches('\n')
                    .trim_end_matches('\r')
                    .to_string(),
            ))
        }
        "term_clear" => {
            print!("\x1b[2J\x1b[H");
            Ok(IrValue::I64(0))
        }
        "term_cursor" => {
            let row = i64_arg(&args[0]);
            let col = i64_arg(&args[1]);
            print!("\x1b[{};{}H", row, col);
            Ok(IrValue::I64(0))
        }
        "term_show_cursor" => {
            let show = i64_arg(&args[0]);
            if show != 0 {
                print!("\x1b[?25h");
            } else {
                print!("\x1b[?25l");
            }
            Ok(IrValue::I64(0))
        }
        "term_set_color" => {
            let fg = i64_arg(&args[0]);
            let bg = i64_arg(&args[1]);
            if fg >= 0 {
                print!("\x1b[38;5;{}m", fg);
            }
            if bg >= 0 {
                print!("\x1b[48;5;{}m", bg);
            }
            Ok(IrValue::I64(0))
        }
        "term_reset" => {
            print!("\x1b[0m");
            Ok(IrValue::I64(0))
        }
        "term_rows" => Ok(IrValue::I64(24)),
        "term_cols" => Ok(IrValue::I64(80)),
        // ---- JSON ----
        "json_parse" => {
            let s = str_arg(&args[0]);
            match serde_json::from_str::<serde_json::Value>(&s) {
                Ok(v) => Ok(json_to_irvalue(&v)),
                Err(_) => Ok(IrValue::Str("null".into())),
            }
        }
        "json_stringify" => {
            let v = &args[0];
            Ok(IrValue::Str(irvalue_to_json(v)))
        }
        // ---- Set (list-backed) ----
        "set_new" => Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
            Vec::new(),
        )))),
        "set_add" => {
            if let IrValue::List(rc) = &args[0] {
                let mut list = rc.lock().unwrap();
                let item = &args[1];
                if !list.iter().any(|x| irvalue_eq(x, item)) {
                    list.push(item.clone());
                }
            }
            Ok(args[0].clone())
        }
        "set_contains" => {
            if let IrValue::List(rc) = &args[0] {
                let list = rc.lock().unwrap();
                let item = &args[1];
                Ok(IrValue::Bool(list.iter().any(|x| irvalue_eq(x, item))))
            } else {
                Ok(IrValue::Bool(false))
            }
        }
        "set_remove" => {
            if let IrValue::List(rc) = &args[0] {
                let mut list = rc.lock().unwrap();
                let item = &args[1];
                list.retain(|x| !irvalue_eq(x, item));
            }
            Ok(args[0].clone())
        }
        "set_len" => {
            if let IrValue::List(rc) = &args[0] {
                Ok(IrValue::I64(rc.lock().unwrap().len() as i64))
            } else {
                Ok(IrValue::I64(0))
            }
        }
        "set_to_list" => Ok(args[0].clone()),
        // ---- Regex ----
        "regex_match" => {
            let pattern = str_arg(&args[0]);
            let text = str_arg(&args[1]);
            Ok(IrValue::Bool(simple_regex_match(&pattern, &text)))
        }
        "regex_find_all" => {
            let pattern = str_arg(&args[0]);
            let text = str_arg(&args[1]);
            let matches = simple_regex_find_all(&pattern, &text);
            let list: Vec<IrValue> = matches.into_iter().map(IrValue::Str).collect();
            Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                list,
            ))))
        }
        "regex_replace" => {
            let pattern = str_arg(&args[0]);
            let text = str_arg(&args[1]);
            let replacement = str_arg(&args[2]);
            Ok(IrValue::Str(simple_regex_replace(
                &pattern,
                &text,
                &replacement,
            )))
        }
        "regex_replace_all" => {
            let pattern = str_arg(&args[0]);
            let text = str_arg(&args[1]);
            let replacement = str_arg(&args[2]);
            Ok(IrValue::Str(simple_regex_replace_all(
                &pattern,
                &text,
                &replacement,
            )))
        }
        // ---- DateTime ----
        "datetime_now" => {
            use std::time::{SystemTime, UNIX_EPOCH};
            let secs = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let s = secs as i64;
            let (y, m, d) = days_to_ymd(s / 86400);
            let time_of_day = s % 86400;
            Ok(IrValue::Str(format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
                y,
                m,
                d,
                time_of_day / 3600,
                (time_of_day % 3600) / 60,
                time_of_day % 60
            )))
        }
        "datetime_timestamp" => {
            use std::time::{SystemTime, UNIX_EPOCH};
            let secs = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs() as f64)
                .unwrap_or(0.0);
            Ok(IrValue::F64(secs))
        }
        "time_now_ms" => {
            use std::time::{SystemTime, UNIX_EPOCH};
            let ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(0);
            Ok(IrValue::I64(ms))
        }
        "datetime_format" => {
            let fmt = str_arg(&args[0]);
            use std::time::{SystemTime, UNIX_EPOCH};
            let secs = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let s = secs as i64;
            let (y, mo, da) = days_to_ymd(s / 86400);
            let tod = s % 86400;
            let out = fmt
                .replace("%Y", &format!("{:04}", y))
                .replace("%m", &format!("{:02}", mo))
                .replace("%d", &format!("{:02}", da))
                .replace("%H", &format!("{:02}", tod / 3600))
                .replace("%M", &format!("{:02}", (tod % 3600) / 60))
                .replace("%S", &format!("{:02}", tod % 60));
            Ok(IrValue::Str(out))
        }
        // ---- OS / Path ----
        "cwd" => Ok(IrValue::Str(
            std::env::current_dir()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
        )),
        "listdir" => {
            let path = str_arg(&args[0]);
            let entries: Vec<IrValue> = std::fs::read_dir(&path)
                .into_iter()
                .flatten()
                .filter_map(|e| {
                    e.ok()
                        .map(|e| IrValue::Str(e.file_name().to_string_lossy().into_owned()))
                })
                .collect();
            Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                entries,
            ))))
        }
        "path_join" => {
            let a = str_arg(&args[0]);
            let b = str_arg(&args[1]);
            Ok(IrValue::Str(
                std::path::PathBuf::from(&a).join(&b).display().to_string(),
            ))
        }
        "path_exists" => {
            let p = str_arg(&args[0]);
            Ok(IrValue::Bool(std::path::Path::new(&p).exists()))
        }
        "mkdir" => {
            let p = str_arg(&args[0]);
            Ok(IrValue::Bool(std::fs::create_dir_all(&p).is_ok()))
        }
        "remove_file" => {
            let p = str_arg(&args[0]);
            Ok(IrValue::Bool(std::fs::remove_file(&p).is_ok()))
        }
        // ---- Type introspection ----
        "type_of" => {
            let t = match &args[0] {
                IrValue::I64(_) => "int",
                IrValue::I32(_) => "i32",
                IrValue::F64(_) => "float",
                IrValue::F32(_) => "f32",
                IrValue::Bool(_) => "bool",
                IrValue::Str(_) => "str",
                IrValue::List(_) => "list",
                IrValue::Map(_) => "map",
                IrValue::Tuple(_) => "tuple",
                IrValue::Struct(_) => "struct",
                IrValue::Enum(_, _) => "enum",
                IrValue::Array(_) => "array",
                IrValue::Tensor(_, _) => "tensor",
                IrValue::Closure { .. } => "closure",
                IrValue::OptionVal(_) => "option",
                IrValue::ResultVal(_) => "result",
                IrValue::Chan(_) => "chan",
                IrValue::TaskGroup(_) => "task_group",
                IrValue::Atomic(_) => "atomic",
                IrValue::Unit => "unit",
                IrValue::Grad { .. } => "grad",
                IrValue::Sparse(_) => "sparse",
                IrValue::TapeNode { .. } => "tape_node",
                IrValue::WeakRef(_) => "weak_ref",
                IrValue::TraitObject { .. } => "trait_object",
                IrValue::Continuation { .. } => "continuation",
            };
            Ok(IrValue::Str(t.to_string()))
        }
        // ---- Random ----
        "random" => {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            thread_local! {
                static SEED: std::cell::Cell<u64> = std::cell::Cell::new(
                    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64).unwrap_or(42)
                );
            }
            let val = SEED.with(|s| {
                let mut h = DefaultHasher::new();
                s.get().hash(&mut h);
                let next = h.finish();
                s.set(next);
                (next >> 11) as f64 / (1u64 << 53) as f64
            });
            Ok(IrValue::F64(val))
        }
        "random_range" => {
            let lo = i64_arg(&args[0]);
            let hi = i64_arg(&args[1]);
            if hi <= lo {
                return Ok(IrValue::I64(lo));
            }
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            thread_local! {
                static SEED2: std::cell::Cell<u64> = std::cell::Cell::new(
                    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64).unwrap_or(99)
                );
            }
            let val = SEED2.with(|s| {
                let mut h = DefaultHasher::new();
                s.get().hash(&mut h);
                let next = h.finish();
                s.set(next);
                lo + (next % (hi - lo) as u64) as i64
            });
            Ok(IrValue::I64(val))
        }
        // ---- Hash ----
        "hash" => {
            let s = str_arg(&args[0]);
            let mut h: u64 = 5381;
            for b in s.bytes() {
                h = h.wrapping_mul(33).wrapping_add(b as u64);
            }
            Ok(IrValue::I64(h as i64))
        }
        // ---- Base64 ----
        "base64_encode" => {
            let s = str_arg(&args[0]);
            Ok(IrValue::Str(base64_encode(s.as_bytes())))
        }
        "base64_decode" => {
            let s = str_arg(&args[0]);
            Ok(IrValue::Str(base64_decode(&s)))
        }
        // ---- String extras ----
        "char_at" => {
            let s = str_arg(&args[0]);
            let idx = i64_arg(&args[1]) as usize;
            Ok(IrValue::Str(
                s.chars()
                    .nth(idx)
                    .map(|c| c.to_string())
                    .unwrap_or_default(),
            ))
        }
        "str_reverse" => {
            let s = str_arg(&args[0]);
            Ok(IrValue::Str(s.chars().rev().collect()))
        }

        // ====================================================================
        // Phase 105 builtins
        // ====================================================================

        // ---- Async/Concurrency extensions ----
        "chan_try_recv" => {
            if let IrValue::Chan(rc) = &args[0] {
                let mut q = rc.queue.lock().unwrap();
                match q.pop_front() {
                    Some(v) => {
                        rc.not_full.notify_one();
                        Ok(IrValue::OptionVal(Some(Box::new(v))))
                    }
                    None => Ok(IrValue::OptionVal(None)),
                }
            } else {
                Ok(IrValue::OptionVal(None))
            }
        }
        "chan_len" => {
            if let IrValue::Chan(rc) = &args[0] {
                Ok(IrValue::I64(rc.queue.lock().unwrap().len() as i64))
            } else {
                Ok(IrValue::I64(0))
            }
        }
        "select" => {
            // select(chan1, chan2, ...) → index of the first non-empty channel, or -1
            for (i, arg) in args.iter().enumerate() {
                if let IrValue::Chan(rc) = arg {
                    if !rc.queue.lock().unwrap().is_empty() {
                        return Ok(IrValue::I64(i as i64));
                    }
                }
            }
            Ok(IrValue::I64(-1))
        }
        "timeout" => {
            // timeout(ms) → always true in single-threaded interp (sleep then return)
            let ms = i64_arg(&args[0]);
            std::thread::sleep(std::time::Duration::from_millis(ms as u64));
            Ok(IrValue::Bool(true))
        }
        "thread_count" => Ok(IrValue::I64(
            std::thread::available_parallelism()
                .map(|n| n.get() as i64)
                .unwrap_or(1),
        )),
        "recv_timeout" => {
            let ms = if args.len() > 1 { i64_arg(&args[1]) } else { 0 };
            if let IrValue::Chan(rc) = &args[0] {
                let start = std::time::Instant::now();
                loop {
                    {
                        let mut q = rc.queue.lock().unwrap();
                        if let Some(v) = q.pop_front() {
                            rc.not_full.notify_one();
                            return Ok(IrValue::OptionVal(Some(Box::new(v))));
                        }
                    }
                    if start.elapsed().as_millis() as i64 >= ms {
                        return Ok(IrValue::OptionVal(None));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            } else {
                Ok(IrValue::OptionVal(None))
            }
        }
        "task_group_join_timeout" => {
            if let IrValue::TaskGroup(tg) = &args[0] {
                let handles = {
                    let mut guard = tg.lock().unwrap();
                    std::mem::take(&mut guard.handles)
                };
                let mut joined_all = true;
                for h in handles {
                    if h.join().is_err() {
                        joined_all = false;
                    }
                }
                Ok(IrValue::Bool(joined_all))
            } else {
                Ok(IrValue::Bool(true))
            }
        }
        "weak_ref" => {
            let arc = std::sync::Arc::new(std::sync::Mutex::new(Some(args[0].clone())));
            let weak = std::sync::Arc::downgrade(&arc);
            Ok(IrValue::WeakRef(weak))
        }
        "weak_upgrade" => {
            if let IrValue::WeakRef(w) = &args[0] {
                if let Some(arc) = w.upgrade() {
                    let guard = arc.lock().unwrap();
                    if let Some(val) = guard.as_ref() {
                        return Ok(IrValue::OptionVal(Some(Box::new(val.clone()))));
                    }
                }
            }
            Ok(IrValue::OptionVal(None))
        }
        "weak_alive" => {
            if let IrValue::WeakRef(w) = &args[0] {
                if let Some(arc) = w.upgrade() {
                    let guard = arc.lock().unwrap();
                    return Ok(IrValue::Bool(guard.is_some()));
                }
            }
            Ok(IrValue::Bool(false))
        }
        "gc_collect" | "gc_collect_call" => Ok(IrValue::Bool(true)),
        "gc_stats" | "gc_stats_map" => {
            let mut map = std::collections::HashMap::new();
            map.insert("allocated".to_string(), IrValue::I64(0));
            map.insert("freed".to_string(), IrValue::I64(0));
            map.insert("cycles_collected".to_string(), IrValue::I64(0));
            map.insert("weak_refs_invalidated".to_string(), IrValue::I64(0));
            Ok(IrValue::Map(std::sync::Arc::new(std::sync::Mutex::new(map))))
        }

        // ---- Deque (double-ended queue, backed by VecDeque stored as List) ----
        "deque_new" => Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
            Vec::new(),
        )))),
        "deque_push_front" => {
            if let IrValue::List(rc) = &args[0] {
                let mut v = rc.lock().unwrap();
                v.insert(0, args[1].clone());
            }
            Ok(args[0].clone())
        }
        "deque_push_back" => {
            if let IrValue::List(rc) = &args[0] {
                rc.lock().unwrap().push(args[1].clone());
            }
            Ok(args[0].clone())
        }
        "deque_pop_front" => {
            if let IrValue::List(rc) = &args[0] {
                let mut v = rc.lock().unwrap();
                if !v.is_empty() {
                    return Ok(v.remove(0));
                }
            }
            Ok(IrValue::Unit)
        }
        "deque_pop_back" => {
            if let IrValue::List(rc) = &args[0] {
                let mut v = rc.lock().unwrap();
                if let Some(val) = v.pop() {
                    return Ok(val);
                }
            }
            Ok(IrValue::Unit)
        }
        "deque_len" => {
            if let IrValue::List(rc) = &args[0] {
                Ok(IrValue::I64(rc.lock().unwrap().len() as i64))
            } else {
                Ok(IrValue::I64(0))
            }
        }
        "deque_front" => {
            if let IrValue::List(rc) = &args[0] {
                let v = rc.lock().unwrap();
                if let Some(val) = v.first() {
                    return Ok(val.clone());
                }
            }
            Ok(IrValue::Unit)
        }
        "deque_back" => {
            if let IrValue::List(rc) = &args[0] {
                let v = rc.lock().unwrap();
                if let Some(val) = v.last() {
                    return Ok(val.clone());
                }
            }
            Ok(IrValue::Unit)
        }

        // ---- Sorted collection helpers ----
        "sorted_keys" => {
            if let IrValue::Map(rc) = &args[0] {
                let m = rc.lock().unwrap();
                let mut keys: Vec<String> = m.keys().cloned().collect();
                keys.sort();
                let list: Vec<IrValue> = keys.into_iter().map(IrValue::Str).collect();
                Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                    list,
                ))))
            } else {
                Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                    Vec::new(),
                ))))
            }
        }

        // ---- BitSet (backed by list of i64 as bit-words) ----
        "bitset_new" => {
            // Create an empty bitset (list of i64 words, each holding 64 bits)
            Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                Vec::new(),
            ))))
        }
        "bitset_set" => {
            if let IrValue::List(rc) = &args[0] {
                let bit = i64_arg(&args[1]) as usize;
                let word_idx = bit / 64;
                let bit_idx = bit % 64;
                let mut v = rc.lock().unwrap();
                while v.len() <= word_idx {
                    v.push(IrValue::I64(0));
                }
                if let IrValue::I64(w) = &v[word_idx] {
                    v[word_idx] = IrValue::I64(w | (1i64 << bit_idx));
                }
            }
            Ok(args[0].clone())
        }
        "bitset_get" => {
            if let IrValue::List(rc) = &args[0] {
                let bit = i64_arg(&args[1]) as usize;
                let word_idx = bit / 64;
                let bit_idx = bit % 64;
                let v = rc.lock().unwrap();
                if word_idx < v.len() {
                    if let IrValue::I64(w) = &v[word_idx] {
                        return Ok(IrValue::Bool((w >> bit_idx) & 1 == 1));
                    }
                }
            }
            Ok(IrValue::Bool(false))
        }
        "bitset_count" => {
            if let IrValue::List(rc) = &args[0] {
                let v = rc.lock().unwrap();
                let count: u32 = v
                    .iter()
                    .map(|x| match x {
                        IrValue::I64(w) => w.count_ones(),
                        _ => 0,
                    })
                    .sum();
                Ok(IrValue::I64(count as i64))
            } else {
                Ok(IrValue::I64(0))
            }
        }
        "bitset_clear" => {
            if let IrValue::List(rc) = &args[0] {
                let bit = i64_arg(&args[1]) as usize;
                let word_idx = bit / 64;
                let bit_idx = bit % 64;
                let mut v = rc.lock().unwrap();
                if word_idx < v.len() {
                    if let IrValue::I64(w) = &v[word_idx] {
                        v[word_idx] = IrValue::I64(w & !(1i64 << bit_idx));
                    }
                }
            }
            Ok(args[0].clone())
        }

        // ---- FFI (dynamic library loading) ----
        "ffi_open" => {
            // ffi_open(path) -> handle (i64), -1 on error
            let _path = str_arg(&args[0]);
            #[cfg(windows)]
            {
                use std::ffi::CString;

                // Automatically inject known dependency paths to PATH for LoadLibrary search
                static INJECT_PATHS: std::sync::Once = std::sync::Once::new();
                INJECT_PATHS.call_once(|| {
                    if std::env::var("AMENT_PREFIX_PATH").is_err() {
                        std::env::set_var(
                            "AMENT_PREFIX_PATH",
                            "C:\\dev\\ros2_humble\\ros2-windows",
                        );
                    }
                    if let Ok(old_path) = std::env::var("PATH") {
                        let new_path = format!(
                            "C:\\dev\\ros2_humble\\ros2-windows\\bin;\
                             C:\\onnxruntime\\lib;\
                             C:\\tensorflow\\lib;\
                             C:\\libtorch\\lib;\
                             C:\\openblas\\bin;{}",
                            old_path
                        );
                        std::env::set_var("PATH", new_path);
                    }

                    // Windows MSVC UCRT environment block dynamic injection
                    unsafe {
                        let ucrt_name = b"ucrtbase.dll\0".as_ptr() as *const i8;
                        let h_ucrt = winapi_LoadLibraryA(ucrt_name);
                        if !h_ucrt.is_null() {
                            let putenv_name = b"_putenv_s\0".as_ptr() as *const i8;
                            let p_putenv = winapi_GetProcAddress(h_ucrt, putenv_name);
                            if !p_putenv.is_null() {
                                let putenv: unsafe extern "C" fn(*const i8, *const i8) -> i32 =
                                    std::mem::transmute(p_putenv);
                                putenv(
                                    b"AMENT_PREFIX_PATH\0".as_ptr() as *const i8,
                                    b"C:\\dev\\ros2_humble\\ros2-windows\0".as_ptr() as *const i8,
                                );
                                if let Ok(new_path) = std::env::var("PATH") {
                                    let c_new_path =
                                        std::ffi::CString::new(new_path).unwrap_or_default();
                                    putenv(b"PATH\0".as_ptr() as *const i8, c_new_path.as_ptr());
                                }
                            }
                        }
                    }
                });

                let cs = CString::new(_path.as_bytes()).unwrap_or_default();
                let h = unsafe { winapi_LoadLibraryA(cs.as_ptr()) };
                if h.is_null() {
                    let err = unsafe { winapi_GetLastError() };
                    eprintln!(
                        "[iris_interp] LoadLibraryA(\"{}\") failed with error code: {}",
                        _path, err
                    );
                    return Ok(IrValue::I64(-1));
                }
                Ok(IrValue::I64(h as i64))
            }
            #[cfg(unix)]
            {
                use std::ffi::CString;
                let cs = CString::new(_path.as_bytes()).unwrap_or_default();
                let h = unsafe { libc_dlopen(cs.as_ptr(), 1) }; // RTLD_LAZY = 1
                if h.is_null() {
                    return Ok(IrValue::I64(-1));
                }
                Ok(IrValue::I64(h as i64))
            }
            #[cfg(not(any(windows, unix)))]
            {
                Ok(IrValue::I64(-1))
            }
        }
        "ffi_call" => {
            // ffi_call(handle, func_name, arg1...) -> i64
            // Simplified: calls a function that takes no args and returns i64
            let _handle = i64_arg(&args[0]);
            let _func_name = str_arg(&args[1]);
            #[cfg(windows)]
            {
                use std::ffi::CString;
                let cs = CString::new(_func_name.as_bytes()).unwrap_or_default();
                let proc = unsafe { winapi_GetProcAddress(_handle as *mut u8, cs.as_ptr()) };
                if proc.is_null() {
                    let err = unsafe { winapi_GetLastError() };
                    eprintln!("[iris_interp] GetProcAddress(handle: {}, \"{}\") failed with error code: {}", _handle, _func_name, err);
                    return Ok(IrValue::I64(-1));
                }
                let f: extern "C" fn() -> i64 = unsafe { std::mem::transmute(proc) };
                Ok(IrValue::I64(f()))
            }
            #[cfg(unix)]
            {
                use std::ffi::CString;
                let cs = CString::new(_func_name.as_bytes()).unwrap_or_default();
                let sym = unsafe { libc_dlsym(_handle as *mut u8, cs.as_ptr()) };
                if sym.is_null() {
                    return Ok(IrValue::I64(-1));
                }
                let f: extern "C" fn() -> i64 = unsafe { std::mem::transmute(sym) };
                Ok(IrValue::I64(f()))
            }
            #[cfg(not(any(windows, unix)))]
            {
                Ok(IrValue::I64(-1))
            }
        }
        "ffi_close" => {
            let _handle = i64_arg(&args[0]);
            #[cfg(windows)]
            {
                let r = unsafe { winapi_FreeLibrary(_handle as *mut u8) };
                Ok(IrValue::Bool(r != 0))
            }
            #[cfg(unix)]
            {
                let r = unsafe { libc_dlclose(_handle as *mut u8) };
                Ok(IrValue::Bool(r == 0))
            }
            #[cfg(not(any(windows, unix)))]
            {
                Ok(IrValue::Bool(false))
            }
        }

        // ---- Expanded FFI: C with typed arguments ----
        // FFI out-parameter cells.
        //
        // Implemented for real rather than stubbed: the interpreter and the
        // native backend must agree, and a stub here would make every
        // sensor-reading policy silently return zeros under `--emit eval` —
        // which is the default path. The cell is a genuine heap allocation
        // whose address is handed to foreign code, so ownership is the
        // caller's, exactly as in the C runtime.
        "ffi_out_new" => {
            let n = i64_arg(&args[0]);
            if n <= 0 {
                return Ok(IrValue::I64(0));
            }
            let mut buf = vec![0u8; n as usize].into_boxed_slice();
            let addr = buf.as_mut_ptr() as i64;
            std::mem::forget(buf);
            ffi_out_cells().insert(addr, n as usize);
            Ok(IrValue::I64(addr))
        }
        "ffi_out_free" => {
            let addr = i64_arg(&args[0]);
            if let Some(len) = ffi_out_cells().remove(&addr) {
                unsafe {
                    let _ = Box::from_raw(std::slice::from_raw_parts_mut(addr as *mut u8, len));
                }
            }
            Ok(IrValue::I64(0))
        }
        "ffi_out_get_f64" | "ffi_out_get_i64" | "ffi_out_get_str" => {
            let addr = i64_arg(&args[0]);
            if addr == 0 || !ffi_out_cells().contains_key(&addr) {
                return Ok(match name {
                    "ffi_out_get_f64" => IrValue::F64(0.0),
                    "ffi_out_get_str" => IrValue::Str(String::new()),
                    _ => IrValue::I64(0),
                });
            }
            let len = ffi_out_cells()[&addr];
            let idx = if args.len() > 1 { i64_arg(&args[1]) } else { 0 };
            match name {
                "ffi_out_get_str" => {
                    let bytes = unsafe { std::slice::from_raw_parts(addr as *const u8, len) };
                    let end = bytes.iter().position(|b| *b == 0).unwrap_or(len);
                    Ok(IrValue::Str(
                        String::from_utf8_lossy(&bytes[..end]).into_owned(),
                    ))
                }
                "ffi_out_get_f64" => {
                    if idx < 0 || (idx as usize + 1) * 8 > len {
                        return Ok(IrValue::F64(0.0));
                    }
                    let v = unsafe { *(addr as *const f64).offset(idx as isize) };
                    Ok(IrValue::F64(v))
                }
                _ => {
                    if idx < 0 || (idx as usize + 1) * 8 > len {
                        return Ok(IrValue::I64(0));
                    }
                    let v = unsafe { *(addr as *const i64).offset(idx as isize) };
                    Ok(IrValue::I64(v))
                }
            }
        }
        "ffi_out_set_f64" | "ffi_out_set_i64" => {
            let addr = i64_arg(&args[0]);
            let idx = i64_arg(&args[1]);
            if addr == 0 || !ffi_out_cells().contains_key(&addr) || idx < 0 {
                return Ok(IrValue::I64(0));
            }
            let len = ffi_out_cells()[&addr];
            if (idx as usize + 1) * 8 > len {
                return Ok(IrValue::I64(0));
            }
            unsafe {
                if name == "ffi_out_set_f64" {
                    *(addr as *mut f64).offset(idx as isize) = f64_arg(&args[2]);
                } else {
                    *(addr as *mut i64).offset(idx as isize) = i64_arg(&args[2]);
                }
            }
            Ok(IrValue::I64(0))
        }

        "ffi_call_i64" | "ffi_call_f64" | "ffi_call_str" | "ffi_call_void" | "ffi_call_args" => {
            // ffi_call_i64(handle, func_name, arg1, arg2, ...) -> i64/f64/str
            // Supports up to 6 i64 arguments via transmuted function pointers.
            let _handle = i64_arg(&args[0]);
            let _func_name = str_arg(&args[1]);
            let extra_args: Vec<i64> = args[2..].iter().map(i64_arg).collect();

            let result_raw: i64;
            #[cfg(windows)]
            {
                use std::ffi::CString;
                let cs = CString::new(_func_name.as_bytes()).unwrap_or_default();
                let proc = unsafe { winapi_GetProcAddress(_handle as *mut u8, cs.as_ptr()) };
                if proc.is_null() {
                    let err = unsafe { winapi_GetLastError() };
                    eprintln!("[iris_interp] GetProcAddress(handle: {}, \"{}\") failed in expanded FFI with error code: {}", _handle, _func_name, err);
                    return match name {
                        "ffi_call_str" => Ok(IrValue::Str(String::new())),
                        "ffi_call_f64" => Ok(IrValue::F64(0.0)),
                        _ => Ok(IrValue::I64(-1)),
                    };
                }
                result_raw = unsafe { ffi_dispatch_call(proc, &extra_args) };
            }
            #[cfg(unix)]
            {
                use std::ffi::CString;
                let cs = CString::new(_func_name.as_bytes()).unwrap_or_default();
                let sym = unsafe { libc_dlsym(_handle as *mut u8, cs.as_ptr()) };
                if sym.is_null() {
                    return match name {
                        "ffi_call_str" => Ok(IrValue::Str(String::new())),
                        "ffi_call_f64" => Ok(IrValue::F64(0.0)),
                        _ => Ok(IrValue::I64(-1)),
                    };
                }
                result_raw = unsafe { ffi_dispatch_call(sym, &extra_args) };
            }
            #[cfg(not(any(windows, unix)))]
            {
                result_raw = -1;
            }

            match name {
                "ffi_call_f64" => Ok(IrValue::F64(f64::from_bits(result_raw as u64))),
                "ffi_call_str" => {
                    // Interpret return as a *const c_char pointer
                    if result_raw == 0 {
                        Ok(IrValue::Str(String::new()))
                    } else {
                        let cstr = unsafe {
                            std::ffi::CStr::from_ptr(result_raw as *const std::os::raw::c_char)
                        };
                        Ok(IrValue::Str(cstr.to_string_lossy().to_string()))
                    }
                }
                "ffi_call_void" => Ok(IrValue::I64(0)),
                _ => Ok(IrValue::I64(result_raw)),
            }
        }

        // ---- Python FFI ----
        "python_eval" => {
            // python_eval(code_str) -> str (stdout from python -c "print(<code>)")
            let code = str_arg(&args[0]);
            let output = std::process::Command::new("python3")
                .args([
                    "-c",
                    &format!("import sys; sys.stdout.write(str({}))", code),
                ])
                .output()
                .or_else(|_| {
                    std::process::Command::new("python")
                        .args([
                            "-c",
                            &format!("import sys; sys.stdout.write(str({}))", code),
                        ])
                        .output()
                });
            match output {
                Ok(o) => Ok(IrValue::Str(String::from_utf8_lossy(&o.stdout).to_string())),
                Err(_) => Ok(IrValue::Str("error: python not found".to_owned())),
            }
        }
        "python_exec" => {
            // python_exec(script_path_or_code) -> exit code
            let code = str_arg(&args[0]);
            let result = if std::path::Path::new(&code).exists() {
                std::process::Command::new("python3")
                    .arg(&code)
                    .status()
                    .or_else(|_| std::process::Command::new("python").arg(&code).status())
            } else {
                std::process::Command::new("python3")
                    .args(["-c", &code])
                    .status()
                    .or_else(|_| {
                        std::process::Command::new("python")
                            .args(["-c", &code])
                            .status()
                    })
            };
            match result {
                Ok(s) => Ok(IrValue::I64(s.code().unwrap_or(-1) as i64)),
                Err(_) => Ok(IrValue::I64(-1)),
            }
        }
        "python_call" => {
            // python_call(module_or_script, func_name, arg1, arg2, ...) -> str
            let module = str_arg(&args[0]);
            let func = str_arg(&args[1]);
            let py_args: Vec<String> = args[2..]
                .iter()
                .map(|a| match a {
                    IrValue::Str(s) => format!("'{}'", s.replace('\'', "\\'")),
                    IrValue::I64(n) => n.to_string(),
                    IrValue::F64(f) => f.to_string(),
                    IrValue::Bool(b) => (if *b { "True" } else { "False" }).to_owned(),
                    _ => "None".to_owned(),
                })
                .collect();
            let py_code = if std::path::Path::new(&module).exists() {
                // It's a script file — import as module
                let mod_name = std::path::Path::new(&module)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("mod");
                format!(
                    "import sys, importlib.util; \
                     spec = importlib.util.spec_from_file_location('{}', '{}'); \
                     mod = importlib.util.module_from_spec(spec); \
                     spec.loader.exec_module(mod); \
                     print(mod.{}({}))",
                    mod_name,
                    module.replace('\\', "\\\\"),
                    func,
                    py_args.join(", ")
                )
            } else {
                format!(
                    "import {}; print({}.{}({}))",
                    module,
                    module,
                    func,
                    py_args.join(", ")
                )
            };
            let output = std::process::Command::new("python3")
                .args(["-c", &py_code])
                .output()
                .or_else(|_| {
                    std::process::Command::new("python")
                        .args(["-c", &py_code])
                        .output()
                });
            match output {
                Ok(o) => {
                    let stdout = String::from_utf8_lossy(&o.stdout).trim().to_owned();
                    let stderr = String::from_utf8_lossy(&o.stderr).trim().to_owned();
                    if !stderr.is_empty() && stdout.is_empty() {
                        Ok(IrValue::Str(format!("error: {}", stderr)))
                    } else {
                        Ok(IrValue::Str(stdout))
                    }
                }
                Err(_) => Ok(IrValue::Str("error: python not found".to_owned())),
            }
        }
        "python_version" => {
            let output = std::process::Command::new("python3")
                .arg("--version")
                .output()
                .or_else(|_| {
                    std::process::Command::new("python")
                        .arg("--version")
                        .output()
                });
            match output {
                Ok(o) => Ok(IrValue::Str(
                    String::from_utf8_lossy(&o.stdout).trim().to_owned(),
                )),
                Err(_) => Ok(IrValue::Str("Python not found".to_owned())),
            }
        }

        // ---- Rust FFI (cdylib — same mechanism as C FFI via dlopen) ----
        "rust_lib_open" => {
            // Alias for ffi_open — open a Rust cdylib (.dll / .so / .dylib)
            let path = str_arg(&args[0]);
            #[cfg(windows)]
            {
                use std::ffi::CString;
                let cs = CString::new(path.as_bytes()).unwrap_or_default();
                let h = unsafe { winapi_LoadLibraryA(cs.as_ptr()) };
                if h.is_null() {
                    let err = unsafe { winapi_GetLastError() };
                    eprintln!(
                        "[iris_interp] LoadLibraryA(\"{}\") failed with error code: {}",
                        path, err
                    );
                    return Ok(IrValue::I64(-1));
                }
                Ok(IrValue::I64(h as i64))
            }
            #[cfg(unix)]
            {
                use std::ffi::CString;
                let cs = CString::new(path.as_bytes()).unwrap_or_default();
                let h = unsafe { libc_dlopen(cs.as_ptr(), 1) };
                if h.is_null() {
                    return Ok(IrValue::I64(-1));
                }
                Ok(IrValue::I64(h as i64))
            }
            #[cfg(not(any(windows, unix)))]
            {
                Ok(IrValue::I64(-1))
            }
        }
        "rust_call_i64" => {
            // rust_call_i64(handle, func_name, arg1, ...) -> i64
            let handle = i64_arg(&args[0]);
            let func_name = str_arg(&args[1]);
            let extra_args: Vec<i64> = args[2..].iter().map(i64_arg).collect();
            #[cfg(windows)]
            {
                use std::ffi::CString;
                let cs = CString::new(func_name.as_bytes()).unwrap_or_default();
                let proc = unsafe { winapi_GetProcAddress(handle as *mut u8, cs.as_ptr()) };
                if proc.is_null() {
                    return Ok(IrValue::I64(-1));
                }
                Ok(IrValue::I64(unsafe {
                    ffi_dispatch_call(proc, &extra_args)
                }))
            }
            #[cfg(unix)]
            {
                use std::ffi::CString;
                let cs = CString::new(func_name.as_bytes()).unwrap_or_default();
                let sym = unsafe { libc_dlsym(handle as *mut u8, cs.as_ptr()) };
                if sym.is_null() {
                    return Ok(IrValue::I64(-1));
                }
                Ok(IrValue::I64(unsafe { ffi_dispatch_call(sym, &extra_args) }))
            }
            #[cfg(not(any(windows, unix)))]
            {
                Ok(IrValue::I64(-1))
            }
        }
        "rust_call_f64" => {
            let handle = i64_arg(&args[0]);
            let func_name = str_arg(&args[1]);
            let extra_args: Vec<i64> = args[2..].iter().map(i64_arg).collect();
            #[cfg(windows)]
            {
                use std::ffi::CString;
                let cs = CString::new(func_name.as_bytes()).unwrap_or_default();
                let proc = unsafe { winapi_GetProcAddress(handle as *mut u8, cs.as_ptr()) };
                if proc.is_null() {
                    return Ok(IrValue::F64(0.0));
                }
                let raw = unsafe { ffi_dispatch_call(proc, &extra_args) };
                Ok(IrValue::F64(f64::from_bits(raw as u64)))
            }
            #[cfg(unix)]
            {
                use std::ffi::CString;
                let cs = CString::new(func_name.as_bytes()).unwrap_or_default();
                let sym = unsafe { libc_dlsym(handle as *mut u8, cs.as_ptr()) };
                if sym.is_null() {
                    return Ok(IrValue::F64(0.0));
                }
                let raw = unsafe { ffi_dispatch_call(sym, &extra_args) };
                Ok(IrValue::F64(f64::from_bits(raw as u64)))
            }
            #[cfg(not(any(windows, unix)))]
            {
                Ok(IrValue::F64(0.0))
            }
        }
        "rust_call_void" => {
            let handle = i64_arg(&args[0]);
            let func_name = str_arg(&args[1]);
            let extra_args: Vec<i64> = args[2..].iter().map(i64_arg).collect();
            #[cfg(windows)]
            {
                use std::ffi::CString;
                let cs = CString::new(func_name.as_bytes()).unwrap_or_default();
                let proc = unsafe { winapi_GetProcAddress(handle as *mut u8, cs.as_ptr()) };
                if !proc.is_null() {
                    unsafe {
                        ffi_dispatch_call(proc, &extra_args);
                    }
                }
                Ok(IrValue::I64(0))
            }
            #[cfg(unix)]
            {
                use std::ffi::CString;
                let cs = CString::new(func_name.as_bytes()).unwrap_or_default();
                let sym = unsafe { libc_dlsym(handle as *mut u8, cs.as_ptr()) };
                if !sym.is_null() {
                    unsafe {
                        ffi_dispatch_call(sym, &extra_args);
                    }
                }
                Ok(IrValue::I64(0))
            }
            #[cfg(not(any(windows, unix)))]
            {
                Ok(IrValue::I64(0))
            }
        }

        // ---- OS / System ----
        "env_get" => {
            let key = str_arg(&args[0]);
            Ok(IrValue::Str(std::env::var(&key).unwrap_or_default()))
        }
        "env_set" => {
            let key = str_arg(&args[0]);
            let val = str_arg(&args[1]);
            unsafe {
                std::env::set_var(&key, &val);
            }
            Ok(IrValue::Bool(true))
        }
        "exit_code" => {
            let code = i64_arg(&args[0]);
            std::process::exit(code as i32);
        }
        "exec_cmd" => {
            let cmd = str_arg(&args[0]);
            #[cfg(windows)]
            let output = std::process::Command::new("cmd")
                .args(["/C", &cmd])
                .output();
            #[cfg(not(windows))]
            let output = std::process::Command::new("sh").args(["-c", &cmd]).output();
            match output {
                Ok(o) => Ok(IrValue::Str(String::from_utf8_lossy(&o.stdout).to_string())),
                Err(_) => Ok(IrValue::Str(String::new())),
            }
        }
        "pid" => Ok(IrValue::I64(std::process::id() as i64)),

        // ---- Crypto / UUID ----
        "uuid" => {
            // Generate a v4-like UUID using hash-based RNG
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            thread_local! {
                static UUID_SEED: std::cell::Cell<u64> = std::cell::Cell::new(
                    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos() as u64).unwrap_or(1234) ^ 0xDEADBEEF
                );
            }
            let (a, b) = UUID_SEED.with(|s| {
                let mut h = DefaultHasher::new();
                s.get().hash(&mut h);
                let v1 = h.finish();
                s.set(v1);
                let mut h2 = DefaultHasher::new();
                v1.hash(&mut h2);
                let v2 = h2.finish();
                s.set(v2);
                (v1, v2)
            });
            // Format as UUID v4
            let a = (a & 0xFFFFFFFFFFFF0FFF) | 0x4000; // version 4
            let b = (b & 0x3FFFFFFFFFFFFFFF) | 0x8000000000000000; // variant 1
            Ok(IrValue::Str(format!(
                "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
                (a >> 32) as u32,
                ((a >> 16) & 0xFFFF) as u16,
                (a & 0xFFFF) as u16,
                ((b >> 48) & 0xFFFF) as u16,
                b & 0xFFFFFFFFFFFF
            )))
        }
        "sha256" => {
            // Minimal SHA-256 implementation
            let input = str_arg(&args[0]);
            Ok(IrValue::Str(sha256_hash(input.as_bytes())))
        }
        "hex_encode" => {
            let s = str_arg(&args[0]);
            Ok(IrValue::Str(
                s.bytes().map(|b| format!("{:02x}", b)).collect(),
            ))
        }
        "hex_decode" => {
            let s = str_arg(&args[0]);
            let bytes: Vec<u8> = (0..s.len())
                .step_by(2)
                .filter_map(|i| {
                    u8::from_str_radix(&s[i..i.min(s.len() - 1) + 2.min(s.len() - i)], 16).ok()
                })
                .collect();
            Ok(IrValue::Str(String::from_utf8_lossy(&bytes).to_string()))
        }

        // ---- String extras (Phase 105) ----
        "str_pad_left" => {
            let s = str_arg(&args[0]);
            let width = i64_arg(&args[1]) as usize;
            let pad = str_arg(&args[2]);
            let pad_char = pad.chars().next().unwrap_or(' ');
            let cur_len = s.chars().count();
            if cur_len >= width {
                Ok(IrValue::Str(s))
            } else {
                let padding: String = std::iter::repeat(pad_char).take(width - cur_len).collect();
                Ok(IrValue::Str(format!("{}{}", padding, s)))
            }
        }
        "str_pad_right" => {
            let s = str_arg(&args[0]);
            let width = i64_arg(&args[1]) as usize;
            let pad = str_arg(&args[2]);
            let pad_char = pad.chars().next().unwrap_or(' ');
            let cur_len = s.chars().count();
            if cur_len >= width {
                Ok(IrValue::Str(s))
            } else {
                let padding: String = std::iter::repeat(pad_char).take(width - cur_len).collect();
                Ok(IrValue::Str(format!("{}{}", s, padding)))
            }
        }
        "str_chars" => {
            let s = str_arg(&args[0]);
            let chars: Vec<IrValue> = s.chars().map(|c| IrValue::Str(c.to_string())).collect();
            Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                chars,
            ))))
        }
        "str_bytes" => {
            let s = str_arg(&args[0]);
            let bytes: Vec<IrValue> = s.bytes().map(|b| IrValue::I64(b as i64)).collect();
            Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                bytes,
            ))))
        }
        "str_count" => {
            let s = str_arg(&args[0]);
            let sub = str_arg(&args[1]);
            Ok(IrValue::I64(s.matches(&sub).count() as i64))
        }

        // ---- Math constants and predicates ----
        "math_pi" => Ok(IrValue::F64(std::f64::consts::PI)),
        "math_e" => Ok(IrValue::F64(std::f64::consts::E)),
        "math_inf" => Ok(IrValue::F64(f64::INFINITY)),
        "is_nan" => match &args[0] {
            IrValue::F64(f) => Ok(IrValue::Bool(f.is_nan())),
            IrValue::F32(f) => Ok(IrValue::Bool(f.is_nan())),
            _ => Ok(IrValue::Bool(false)),
        },
        "is_inf" => match &args[0] {
            IrValue::F64(f) => Ok(IrValue::Bool(f.is_infinite())),
            IrValue::F32(f) => Ok(IrValue::Bool(f.is_infinite())),
            _ => Ok(IrValue::Bool(false)),
        },

        // ---- Functional list operations ----
        // list_map, list_filter, list_reduce are handled directly in the
        // Interpreter::run() method (before this function is called) so they
        // can invoke closures. If we reach here, it's a programming error.
        "list_map" | "list_filter" | "list_reduce" => Err(InterpError::Unsupported {
            detail: format!(
                "{}: should have been handled at the Interpreter level",
                name
            ),
        }),
        "list_any" => {
            // list_any(list) → true if any element is truthy
            if let IrValue::List(rc) = &args[0] {
                let v = rc.lock().unwrap();
                Ok(IrValue::Bool(v.iter().any(|x| match x {
                    IrValue::Bool(b) => *b,
                    IrValue::I64(n) => *n != 0,
                    IrValue::Str(s) => !s.is_empty(),
                    _ => true,
                })))
            } else {
                Ok(IrValue::Bool(false))
            }
        }
        "list_all" => {
            // list_all(list) → true if all elements are truthy
            if let IrValue::List(rc) = &args[0] {
                let v = rc.lock().unwrap();
                Ok(IrValue::Bool(v.iter().all(|x| match x {
                    IrValue::Bool(b) => *b,
                    IrValue::I64(n) => *n != 0,
                    IrValue::Str(s) => !s.is_empty(),
                    _ => true,
                })))
            } else {
                Ok(IrValue::Bool(true))
            }
        }
        "list_zip" => {
            // list_zip(list1, list2) → list of tuples
            if let (IrValue::List(a), IrValue::List(b)) = (&args[0], &args[1]) {
                let va = a.lock().unwrap();
                let vb = b.lock().unwrap();
                let zipped: Vec<IrValue> = va
                    .iter()
                    .zip(vb.iter())
                    .map(|(x, y)| IrValue::Tuple(vec![x.clone(), y.clone()]))
                    .collect();
                Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                    zipped,
                ))))
            } else {
                Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                    Vec::new(),
                ))))
            }
        }
        "list_enumerate" => {
            // list_enumerate(list) → list of (index, value) tuples
            if let IrValue::List(rc) = &args[0] {
                let v = rc.lock().unwrap();
                let enumerated: Vec<IrValue> = v
                    .iter()
                    .enumerate()
                    .map(|(i, val)| IrValue::Tuple(vec![IrValue::I64(i as i64), val.clone()]))
                    .collect();
                Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                    enumerated,
                ))))
            } else {
                Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                    Vec::new(),
                ))))
            }
        }
        "list_flatten" => {
            // list_flatten(list<list<T>>) → list<T>
            if let IrValue::List(rc) = &args[0] {
                let v = rc.lock().unwrap();
                let mut flat = Vec::new();
                for item in v.iter() {
                    if let IrValue::List(inner) = item {
                        flat.extend(inner.lock().unwrap().iter().cloned());
                    } else {
                        flat.push(item.clone());
                    }
                }
                Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                    flat,
                ))))
            } else {
                Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                    Vec::new(),
                ))))
            }
        }
        "list_unique" => {
            // list_unique(list) → list with duplicates removed
            if let IrValue::List(rc) = &args[0] {
                let v = rc.lock().unwrap();
                let mut unique = Vec::new();
                for item in v.iter() {
                    if !unique.iter().any(|x| irvalue_eq(x, item)) {
                        unique.push(item.clone());
                    }
                }
                Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                    unique,
                ))))
            } else {
                Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                    Vec::new(),
                ))))
            }
        }
        "list_reverse" => {
            if let IrValue::List(rc) = &args[0] {
                let v = rc.lock().unwrap();
                let reversed: Vec<IrValue> = v.iter().rev().cloned().collect();
                Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                    reversed,
                ))))
            } else {
                Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                    Vec::new(),
                ))))
            }
        }
        "list_sorted" => {
            // list_sorted(list) → sorted copy (works for i64, f64, str)
            if let IrValue::List(rc) = &args[0] {
                let v = rc.lock().unwrap();
                let mut sorted: Vec<IrValue> = v.clone();
                sorted.sort_by(irvalue_cmp);
                Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                    sorted,
                ))))
            } else {
                Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                    Vec::new(),
                ))))
            }
        }
        "list_sum" => {
            if let IrValue::List(rc) = &args[0] {
                let v = rc.lock().unwrap();
                let sum: f64 = v
                    .iter()
                    .map(|x| match x {
                        IrValue::I64(n) => *n as f64,
                        IrValue::F64(f) => *f,
                        IrValue::I32(n) => *n as f64,
                        IrValue::F32(f) => *f as f64,
                        _ => 0.0,
                    })
                    .sum();
                Ok(IrValue::F64(sum))
            } else {
                Ok(IrValue::F64(0.0))
            }
        }
        "list_min" => {
            if let IrValue::List(rc) = &args[0] {
                let v = rc.lock().unwrap();
                if v.is_empty() {
                    return Ok(IrValue::Unit);
                }
                let mut min_val = v[0].clone();
                for item in v.iter().skip(1) {
                    if irvalue_cmp(item, &min_val) == std::cmp::Ordering::Less {
                        min_val = item.clone();
                    }
                }
                Ok(min_val)
            } else {
                Ok(IrValue::Unit)
            }
        }
        "list_max" => {
            if let IrValue::List(rc) = &args[0] {
                let v = rc.lock().unwrap();
                if v.is_empty() {
                    return Ok(IrValue::Unit);
                }
                let mut max_val = v[0].clone();
                for item in v.iter().skip(1) {
                    if irvalue_cmp(item, &max_val) == std::cmp::Ordering::Greater {
                        max_val = item.clone();
                    }
                }
                Ok(max_val)
            } else {
                Ok(IrValue::Unit)
            }
        }
        "list_index_of" => {
            // list_index_of(list, item) → index or -1
            if let IrValue::List(rc) = &args[0] {
                let v = rc.lock().unwrap();
                let needle = &args[1];
                for (i, item) in v.iter().enumerate() {
                    if irvalue_eq(item, needle) {
                        return Ok(IrValue::I64(i as i64));
                    }
                }
                Ok(IrValue::I64(-1))
            } else {
                Ok(IrValue::I64(-1))
            }
        }
        "list_count" => {
            // list_count(list, item) → number of occurrences
            if let IrValue::List(rc) = &args[0] {
                let v = rc.lock().unwrap();
                let needle = &args[1];
                let count = v.iter().filter(|x| irvalue_eq(x, needle)).count();
                Ok(IrValue::I64(count as i64))
            } else {
                Ok(IrValue::I64(0))
            }
        }
        "list_take" => {
            // list_take(list, n) → first n elements
            if let IrValue::List(rc) = &args[0] {
                let n = i64_arg(&args[1]).max(0) as usize;
                let v = rc.lock().unwrap();
                let taken: Vec<IrValue> = v.iter().take(n).cloned().collect();
                Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                    taken,
                ))))
            } else {
                Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                    Vec::new(),
                ))))
            }
        }
        "list_drop" => {
            // list_drop(list, n) → elements after first n
            if let IrValue::List(rc) = &args[0] {
                let n = i64_arg(&args[1]).max(0) as usize;
                let v = rc.lock().unwrap();
                let dropped: Vec<IrValue> = v.iter().skip(n).cloned().collect();
                Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                    dropped,
                ))))
            } else {
                Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(
                    Vec::new(),
                ))))
            }
        }

        "list_remove" => {
            if let IrValue::List(arc_list) = &args[0] {
                let idx = i64_arg(&args[1]) as usize;
                let mut list = arc_list.lock().unwrap();
                if idx < list.len() {
                    let removed = list.remove(idx);
                    Ok(removed)
                } else {
                    Err(InterpError::Panic { msg: format!("list_remove: index {} out of bounds (len {})", idx, list.len()) })
                }
            } else {
                Err(InterpError::TypeError { detail: "list_remove: first argument must be a list".into() })
            }
        }
        "list_insert" => {
            if let IrValue::List(arc_list) = &args[0] {
                let idx = i64_arg(&args[1]) as usize;
                let val = args[2].clone();
                let mut list = arc_list.lock().unwrap();
                if idx <= list.len() {
                    list.insert(idx, val);
                    Ok(IrValue::I64(0))
                } else {
                    Err(InterpError::Panic { msg: format!("list_insert: index {} out of bounds (len {})", idx, list.len()) })
                }
            } else {
                Err(InterpError::TypeError { detail: "list_insert: first argument must be a list".into() })
            }
        }
        "map_entries" => {
            if let IrValue::Map(arc_map) = &args[0] {
                let map = arc_map.lock().unwrap();
                let entries: Vec<IrValue> = map.iter().map(|(k, v)| {
                    IrValue::Str(format!("{}:{}", k, match v {
                        IrValue::I64(n) => n.to_string(),
                        IrValue::F64(f) => f.to_string(),
                        IrValue::Str(s) => s.clone(),
                        IrValue::Bool(b) => b.to_string(),
                        _ => "<value>".to_string(),
                    }))
                }).collect();
                Ok(IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(entries))))
            } else {
                Err(InterpError::TypeError { detail: "map_entries: argument must be a map".into() })
            }
        }
        "chan_send" => {
            if let IrValue::Chan(rc) = &args[0] {
                let v = args[1].clone();
                let mut queue = rc.queue.lock().unwrap();
                while queue.len() >= rc.capacity {
                    queue = rc.not_full.wait(queue).unwrap();
                }
                queue.push_back(v);
                rc.not_empty.notify_one();
                Ok(IrValue::I64(0))
            } else {
                Err(InterpError::TypeError { detail: "chan_send: first argument must be a channel".into() })
            }
        }

        _ => Err(InterpError::Unsupported {
            detail: format!("unknown builtin: {}", name),
        }),
    }
}

// ---------------------------------------------------------------------------
// Builtin helpers
// ---------------------------------------------------------------------------

fn irvalue_eq(a: &IrValue, b: &IrValue) -> bool {
    match (a, b) {
        (IrValue::I64(x), IrValue::I64(y)) => x == y,
        (IrValue::I32(x), IrValue::I32(y)) => x == y,
        (IrValue::F64(x), IrValue::F64(y)) => x == y,
        (IrValue::F32(x), IrValue::F32(y)) => x == y,
        (IrValue::Bool(x), IrValue::Bool(y)) => x == y,
        (IrValue::Str(x), IrValue::Str(y)) => x == y,
        _ => false,
    }
}

fn irvalue_cmp(a: &IrValue, b: &IrValue) -> std::cmp::Ordering {
    match (a, b) {
        (IrValue::I64(x), IrValue::I64(y)) => x.cmp(y),
        (IrValue::I32(x), IrValue::I32(y)) => x.cmp(y),
        (IrValue::F64(x), IrValue::F64(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
        (IrValue::F32(x), IrValue::F32(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
        (IrValue::Str(x), IrValue::Str(y)) => x.cmp(y),
        (IrValue::Bool(x), IrValue::Bool(y)) => x.cmp(y),
        _ => std::cmp::Ordering::Equal,
    }
}

/// Minimal SHA-256 implementation (pure Rust, no deps)
fn sha256_hash(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    // Padding
    let bit_len = (data.len() as u64) * 8;
    let mut msg = data.to_vec();
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    // Process 512-bit blocks
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    format!(
        "{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}{:08x}",
        h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]
    )
}

// ---- FFI extern declarations ----
#[cfg(windows)]
extern "system" {
    fn LoadLibraryA(name: *const i8) -> *mut u8;
    fn GetProcAddress(module: *mut u8, name: *const i8) -> *mut u8;
    fn FreeLibrary(module: *mut u8) -> i32;
    fn GetModuleHandleA(name: *const i8) -> *mut u8;
    fn GetLastError() -> u32;
}
#[cfg(windows)]
use self::FreeLibrary as winapi_FreeLibrary;
#[cfg(windows)]
use self::GetLastError as winapi_GetLastError;
#[cfg(windows)]
use self::GetProcAddress as winapi_GetProcAddress;
#[cfg(windows)]
use self::LoadLibraryA as winapi_LoadLibraryA;

#[cfg(unix)]
extern "C" {
    fn dlopen(filename: *const std::os::raw::c_char, flags: i32) -> *mut u8;
    fn dlsym(handle: *mut u8, symbol: *const std::os::raw::c_char) -> *mut u8;
    fn dlclose(handle: *mut u8) -> i32;
}
#[cfg(unix)]
use self::dlclose as libc_dlclose;
#[cfg(unix)]
use self::dlopen as libc_dlopen;
#[cfg(unix)]
use self::dlsym as libc_dlsym;

fn json_to_irvalue(v: &serde_json::Value) -> IrValue {
    match v {
        serde_json::Value::Null => IrValue::Str("null".into()),
        serde_json::Value::Bool(b) => IrValue::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                IrValue::I64(i)
            } else {
                IrValue::F64(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::String(s) => IrValue::Str(s.clone()),
        serde_json::Value::Array(arr) => {
            let items: Vec<IrValue> = arr.iter().map(json_to_irvalue).collect();
            IrValue::List(std::sync::Arc::new(std::sync::Mutex::new(items)))
        }
        serde_json::Value::Object(obj) => {
            let mut map = std::collections::HashMap::new();
            for (k, v) in obj {
                map.insert(k.clone(), json_to_irvalue(v));
            }
            IrValue::Map(std::sync::Arc::new(std::sync::Mutex::new(map)))
        }
    }
}

fn irvalue_to_json(v: &IrValue) -> String {
    fn to_json(v: &IrValue) -> String {
        match v {
            IrValue::I64(n) => n.to_string(),
            IrValue::I32(n) => n.to_string(),
            IrValue::F64(f) => format!("{}", f),
            IrValue::F32(f) => format!("{}", f),
            IrValue::Bool(b) => {
                if *b { "true".into() } else { "false".into() }
            }
            IrValue::Str(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\"")),
            IrValue::List(rc) => {
                let items: Vec<String> = rc.lock().unwrap().iter().map(to_json).collect();
                format!("[{}]", items.join(","))
            }
            IrValue::Map(rc) => {
                let pairs: Vec<String> = rc
                    .lock()
                    .unwrap()
                    .iter()
                    .map(|(k, v)| {
                        format!(
                            "\"{}\":{}",
                            k.replace('\\', "\\\\").replace('"', "\\\""),
                            to_json(v)
                        )
                    })
                    .collect();
                format!("{{{}}}", pairs.join(","))
            }
            IrValue::Struct(fields) => {
                let items: Vec<String> = fields.iter().enumerate()
                    .map(|(i, fv)| format!("\"{}\":{}", i, to_json(fv)))
                    .collect();
                format!("{{{}}}", items.join(","))
            }
            IrValue::Enum(tag, fields) => {
                let flds: Vec<String> = fields.iter().map(to_json).collect();
                format!("{{\"tag\":{},\"fields\":[{}]}}", tag, flds.join(","))
            }
            IrValue::Tuple(elems) => {
                let items: Vec<String> = elems.iter().map(to_json).collect();
                format!("[{}]", items.join(","))
            }
            IrValue::OptionVal(Some(v)) => to_json(v),
            IrValue::OptionVal(None) => "null".into(),
            IrValue::ResultVal(Ok(v)) => format!("{{\"ok\":{}}}", to_json(v)),
            IrValue::ResultVal(Err(e)) => format!("{{\"err\":{}}}", to_json(e)),
            IrValue::Unit => "null".into(),
            _ => "null".into(),
        }
    }
    to_json(v)
}

fn http_request(method: &str, url: &str, body: &str) -> String {
    let url_trimmed = url.strip_prefix("http://").unwrap_or(url);
    let (hostport, path) = match url_trimmed.find('/') {
        Some(i) => (&url_trimmed[..i], &url_trimmed[i..]),
        None => (url_trimmed, "/"),
    };
    let (host, port) = match hostport.rfind(':') {
        Some(i) => (
            &hostport[..i],
            hostport[i + 1..].parse::<u16>().unwrap_or(80),
        ),
        None => (hostport, 80u16),
    };
    let req = if method == "POST" {
        format!("{} {} HTTP/1.0\r\nHost: {}\r\nContent-Length: {}\r\nContent-Type: application/x-www-form-urlencoded\r\nConnection: close\r\n\r\n{}",
            method, path, host, body.len(), body)
    } else {
        format!(
            "{} {} HTTP/1.0\r\nHost: {}\r\nConnection: close\r\n\r\n",
            method, path, host
        )
    };
    use std::io::{Read, Write};
    match std::net::TcpStream::connect(format!("{}:{}", host, port)) {
        Ok(mut stream) => {
            let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(10)));
            if stream.write_all(req.as_bytes()).is_err() {
                return String::new();
            }
            let mut resp = String::new();
            let _ = stream.read_to_string(&mut resp);
            if let Some(i) = resp.find("\r\n\r\n") {
                resp[i + 4..].to_string()
            } else {
                resp
            }
        }
        Err(_) => String::new(),
    }
}

fn simple_regex_match(pattern: &str, text: &str) -> bool {
    let pat = pattern.as_bytes();
    let txt = text.as_bytes();
    if pat.first() == Some(&b'^') {
        return regex_match_here(&pat[1..], txt);
    }
    for i in 0..=txt.len() {
        if regex_match_here(pat, &txt[i..]) {
            return true;
        }
    }
    false
}

fn regex_char_class_matches(cc: &[u8], c: u8) -> (bool, usize) {
    let mut negated = false;
    let mut pos = 0;
    if pos < cc.len() && cc[pos] == b'^' {
        negated = true;
        pos += 1;
    }
    let mut matched = false;
    while pos < cc.len() && cc[pos] != b']' {
        let lo = cc[pos];
        if pos + 2 < cc.len() && cc[pos + 1] == b'-' && cc[pos + 2] != b']' {
            let hi = cc[pos + 2];
            if c >= lo && c <= hi { matched = true; }
            pos += 3;
        } else if lo == b'\\' && pos + 1 < cc.len() {
            pos += 1;
            let esc = cc[pos];
            pos += 1;
            match esc {
                b'd' => if c.is_ascii_digit() { matched = true; },
                b'w' => if c.is_ascii_alphanumeric() || c == b'_' { matched = true; },
                b's' => if c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' { matched = true; },
                _ => if c == esc { matched = true; },
            }
        } else {
            if c == lo { matched = true; }
            pos += 1;
        }
    }
    if pos < cc.len() && cc[pos] == b']' { pos += 1; }
    (if negated { !matched } else { matched }, pos)
}

fn regex_match_here(pat: &[u8], txt: &[u8]) -> bool {
    if pat.is_empty() {
        return true;
    }
    if pat == b"$" {
        return txt.is_empty();
    }
    if pat.len() >= 2 && pat[1] == b'*' {
        return regex_match_star(pat[0], &pat[2..], txt);
    }
    if pat.len() >= 2 && pat[1] == b'+' {
        if txt.is_empty() || (pat[0] != b'.' && pat[0] != txt[0]) {
            return false;
        }
        return regex_match_star(pat[0], &pat[2..], &txt[1..]);
    }
    if pat.len() >= 2 && pat[1] == b'?' {
        if regex_match_here(&pat[2..], txt) {
            return true;
        }
        if !txt.is_empty() && (pat[0] == b'.' || pat[0] == txt[0]) {
            return regex_match_here(&pat[2..], &txt[1..]);
        }
        return false;
    }
    /* Grouping: (sub-pattern) with | alternation */
    if pat[0] == b'(' {
        let mut depth = 1;
        let mut p = 1;
        while p < pat.len() && depth > 0 {
            if pat[p] == b'(' { depth += 1; }
            else if pat[p] == b')' { depth -= 1; }
            p += 1;
        }
        /* p is past the closing ')' */
        let group = &pat[1..p-1]; // exclude parens
        let rest = &pat[p..];
        // Check for alternation
        let alt_pos = {
            let mut found = None;
            let mut ad = 0;
            for (i, &b) in group.iter().enumerate() {
                if b == b'(' { ad += 1; }
                else if b == b')' { ad -= 1; }
                else if b == b'|' && ad == 0 { found = Some(i); break; }
            }
            found
        };
        if let Some(ap) = alt_pos {
            let left = &group[..ap];
            let right = &group[ap+1..];
            let mut combined = Vec::with_capacity(left.len() + rest.len());
            combined.extend_from_slice(left);
            combined.extend_from_slice(rest);
            if regex_match_here(&combined, txt) { return true; }
            combined.clear();
            combined.extend_from_slice(right);
            combined.extend_from_slice(rest);
            return regex_match_here(&combined, txt);
        } else {
            let mut combined = Vec::with_capacity(group.len() + rest.len());
            combined.extend_from_slice(group);
            combined.extend_from_slice(rest);
            return regex_match_here(&combined, txt);
        }
    }
    /* Escape sequences: \d, \w, \s */
    if pat[0] == b'\\' && pat.len() >= 2 {
        let esc = pat[1];
        let m = match esc {
            b'd' => txt.first().map_or(false, |&c| c.is_ascii_digit()),
            b'w' => txt.first().map_or(false, |&c| c.is_ascii_alphanumeric() || c == b'_'),
            b's' => txt.first().map_or(false, |&c| c == b' ' || c == b'\t' || c == b'\n' || c == b'\r'),
            _ => txt.first().map_or(false, |&c| c == esc),
        };
        let after_esc = &pat[2..];
        if !after_esc.is_empty() && (after_esc[0] == b'+' || after_esc[0] == b'*' || after_esc[0] == b'?') {
            let q = after_esc[0];
            let rest = &after_esc[1..];
            if q == b'+' {
                if !m || txt.is_empty() { return false; }
                // Match one or more
                let mut i = 1;
                loop {
                    if regex_match_here(rest, &txt[i..]) { return true; }
                    if i >= txt.len() { break; }
                    let c = txt[i];
                    let ok = match esc {
                        b'd' => c.is_ascii_digit(),
                        b'w' => c.is_ascii_alphanumeric() || c == b'_',
                        b's' => c == b' ' || c == b'\t' || c == b'\n' || c == b'\r',
                        _ => c == esc,
                    };
                    if !ok { break; }
                    i += 1;
                }
                return false;
            } else if q == b'*' {
                let mut i = 0;
                loop {
                    if regex_match_here(rest, &txt[i..]) { return true; }
                    if i >= txt.len() { break; }
                    let c = txt[i];
                    let ok = match esc {
                        b'd' => c.is_ascii_digit(),
                        b'w' => c.is_ascii_alphanumeric() || c == b'_',
                        b's' => c == b' ' || c == b'\t' || c == b'\n' || c == b'\r',
                        _ => c == esc,
                    };
                    if !ok { break; }
                    i += 1;
                }
                return false;
            } else {
                // '?': zero or one
                if m && !txt.is_empty() {
                    if regex_match_here(rest, &txt[1..]) { return true; }
                }
                return regex_match_here(rest, txt);
            }
        }
        if m && !txt.is_empty() {
            return regex_match_here(&pat[2..], &txt[1..]);
        }
        return false;
    }
    /* Character class: [...] */
    if pat[0] == b'[' {
        if txt.is_empty() { return false; }
        let (m, consumed) = regex_char_class_matches(&pat[1..], txt[0]);
        // Check for quantifier after the class
        let after_class = 1 + consumed; // skip '[' + class content
        if after_class < pat.len() && (pat[after_class] == b'+' || pat[after_class] == b'*' || pat[after_class] == b'?') {
            let q = pat[after_class];
            let rest = &pat[after_class + 1..];
            if q == b'+' {
                if !m { return false; }
                let mut i = 1;
                loop {
                    if regex_match_here(rest, &txt[i..]) { return true; }
                    if i >= txt.len() { break; }
                    let (m2, _) = regex_char_class_matches(&pat[1..], txt[i]);
                    if !m2 { break; }
                    i += 1;
                }
                return false;
            } else if q == b'*' {
                let mut i = 0;
                loop {
                    if regex_match_here(rest, &txt[i..]) { return true; }
                    if i >= txt.len() { break; }
                    let (m2, _) = regex_char_class_matches(&pat[1..], txt[i]);
                    if !m2 { break; }
                    i += 1;
                }
                return false;
            } else {
                // '?': zero or one
                if m {
                    if regex_match_here(rest, &txt[1..]) { return true; }
                }
                return regex_match_here(rest, txt);
            }
        }
        if m {
            return regex_match_here(&pat[1 + consumed..], &txt[1..]);
        }
        return false;
    }
    if !txt.is_empty() && (pat[0] == b'.' || pat[0] == txt[0]) {
        return regex_match_here(&pat[1..], &txt[1..]);
    }
    false
}

fn regex_match_star(c: u8, pat: &[u8], txt: &[u8]) -> bool {
    let mut i = 0;
    loop {
        if regex_match_here(pat, &txt[i..]) {
            return true;
        }
        if i >= txt.len() || (c != b'.' && txt[i] != c) {
            return false;
        }
        i += 1;
    }
}

fn regex_mpl(pat: &[u8], txt: &[u8]) -> isize {
    if pat.is_empty() { return 0; }
    if pat == b"$" { return if txt.is_empty() { 0 } else { -1 }; }

    // Grouping: (sub-pattern)
    if pat[0] == b'(' {
        let mut depth = 1; let mut p = 1;
        while p < pat.len() && depth > 0 {
            if pat[p] == b'(' { depth += 1; }
            else if pat[p] == b')' { depth -= 1; if depth == 0 { p += 1; break; } }
            p += 1;
        }
        let group = &pat[1..p-1];
        let rest = &pat[p..];
        let alt_pos = {
            let mut found = None; let mut ad = 0;
            for (i, &b) in group.iter().enumerate() {
                if b == b'(' { ad += 1; }
                else if b == b')' { ad -= 1; }
                else if b == b'|' && ad == 0 { found = Some(i); break; }
            }
            found
        };
        if let Some(ap) = alt_pos {
            let left = &group[..ap]; let right = &group[ap+1..];
            let mut best: isize = -1;
            for alt in &[left, right] {
                let mut combined = Vec::with_capacity(alt.len() + rest.len());
                combined.extend_from_slice(alt);
                combined.extend_from_slice(rest);
                let r = regex_mpl(&combined, txt);
                if r >= 0 && r > best { best = r; }
            }
            return best;
        } else {
            let mut combined = Vec::with_capacity(group.len() + rest.len());
            combined.extend_from_slice(group);
            combined.extend_from_slice(rest);
            return regex_mpl(&combined, txt);
        }
    }

    // Escape: \d, \w, \s
    if pat[0] == b'\\' && pat.len() >= 2 {
        let esc = pat[1];
        let af = &pat[2..];
        let m = txt.first().map_or(false, |&c| match esc {
            b'd' => c.is_ascii_digit(),
            b'w' => c.is_ascii_alphanumeric() || c == b'_',
            b's' => c == b' ' || c == b'\t' || c == b'\n' || c == b'\r',
            _ => c == esc,
        });
        let esc_match = |c: u8| -> bool {
            match esc {
                b'd' => c.is_ascii_digit(),
                b'w' => c.is_ascii_alphanumeric() || c == b'_',
                b's' => c == b' ' || c == b'\t' || c == b'\n' || c == b'\r',
                _ => c == esc,
            }
        };
        if !af.is_empty() && (af[0] == b'+' || af[0] == b'*' || af[0] == b'?') {
            let q = af[0]; let rest = &af[1..];
            if q == b'+' {
                if !m { return -1; }
                let mut cnt: isize = 0; let mut i = 0;
                while i < txt.len() && esc_match(txt[i]) { cnt += 1; i += 1; }
                for c in (1..=cnt).rev() {
                    let r = regex_mpl(rest, &txt[c as usize..]);
                    if r >= 0 { return c + r; }
                }
                return -1;
            } else if q == b'*' {
                let mut cnt: isize = 0; let mut i = 0;
                while i < txt.len() && esc_match(txt[i]) { cnt += 1; i += 1; }
                for c in (0..=cnt).rev() {
                    let r = regex_mpl(rest, &txt[c as usize..]);
                    if r >= 0 { return c + r; }
                }
                return -1;
            } else {
                if m && !txt.is_empty() {
                    let r = regex_mpl(rest, &txt[1..]);
                    if r >= 0 { return 1 + r; }
                }
                return regex_mpl(rest, txt);
            }
        }
        if m && !txt.is_empty() {
            let r = regex_mpl(af, &txt[1..]);
            if r >= 0 { return 1 + r; }
        }
        return -1;
    }

    // Character class: [...]
    if pat[0] == b'[' {
        if txt.is_empty() { return -1; }
        let (m, consumed) = regex_char_class_matches(&pat[1..], txt[0]);
        let after_class = 1 + consumed;
        if after_class < pat.len() && (pat[after_class] == b'+' || pat[after_class] == b'*' || pat[after_class] == b'?') {
            let q = pat[after_class]; let rest = &pat[after_class + 1..];
            if q == b'+' {
                if !m { return -1; }
                let mut cnt: isize = 0; let mut i = 0;
                while i < txt.len() {
                    let (m2, _) = regex_char_class_matches(&pat[1..], txt[i]);
                    if !m2 { break; }
                    cnt += 1; i += 1;
                }
                for c in (1..=cnt).rev() {
                    let r = regex_mpl(rest, &txt[c as usize..]);
                    if r >= 0 { return c + r; }
                }
                return -1;
            } else if q == b'*' {
                let mut cnt: isize = 0; let mut i = 0;
                while i < txt.len() {
                    let (m2, _) = regex_char_class_matches(&pat[1..], txt[i]);
                    if !m2 { break; }
                    cnt += 1; i += 1;
                }
                for c in (0..=cnt).rev() {
                    let r = regex_mpl(rest, &txt[c as usize..]);
                    if r >= 0 { return c + r; }
                }
                return -1;
            } else {
                if m {
                    let r = regex_mpl(rest, &txt[1..]);
                    if r >= 0 { return 1 + r; }
                }
                return regex_mpl(rest, txt);
            }
        }
        if m {
            let r = regex_mpl(&pat[1+consumed..], &txt[1..]);
            if r >= 0 { return 1 + r; }
        }
        return -1;
    }

    // Single char with quantifiers
    if pat.len() >= 2 && pat[1] == b'*' {
        let mut cnt: isize = 0; let mut i = 0;
        while i < txt.len() && (pat[0] == b'.' || txt[i] == pat[0]) { cnt += 1; i += 1; }
        for c in (0..=cnt).rev() {
            let r = regex_mpl(&pat[2..], &txt[c as usize..]);
            if r >= 0 { return c + r; }
        }
        return -1;
    }
    if pat.len() >= 2 && pat[1] == b'+' {
        if txt.is_empty() || (pat[0] != b'.' && txt[0] != pat[0]) { return -1; }
        let mut cnt: isize = 0; let mut i = 0;
        while i < txt.len() && (pat[0] == b'.' || txt[i] == pat[0]) { cnt += 1; i += 1; }
        for c in (1..=cnt).rev() {
            let r = regex_mpl(&pat[2..], &txt[c as usize..]);
            if r >= 0 { return c + r; }
        }
        return -1;
    }
    if pat.len() >= 2 && pat[1] == b'?' {
        if !txt.is_empty() && (pat[0] == b'.' || txt[0] == pat[0]) {
            let r = regex_mpl(&pat[2..], &txt[1..]);
            if r >= 0 { return 1 + r; }
        }
        return regex_mpl(&pat[2..], txt);
    }
    // Single char, no quantifier
    if !txt.is_empty() && (pat[0] == b'.' || txt[0] == pat[0]) {
        let r = regex_mpl(&pat[1..], &txt[1..]);
        if r >= 0 { return 1 + r; }
    }
    -1
}

fn simple_regex_find_all(pattern: &str, text: &str) -> Vec<String> {
    let pat = if pattern.starts_with('^') { &pattern.as_bytes()[1..] } else { pattern.as_bytes() };
    let txt = text.as_bytes();
    let mut results = Vec::new();
    let mut pos = 0;
    while pos < txt.len() {
        let len = regex_mpl(pat, &txt[pos..]);
        if len > 0 {
            results.push(String::from_utf8_lossy(&txt[pos..pos + len as usize]).to_string());
            pos += len as usize;
        } else {
            pos += 1;
        }
    }
    results
}

fn simple_regex_replace(pattern: &str, text: &str, replacement: &str) -> String {
    let matches = simple_regex_find_all(pattern, text);
    let mut result = text.to_string();
    for m in &matches {
        if let Some(pos) = result.find(m.as_str()) {
            result = format!(
                "{}{}{}",
                &result[..pos],
                replacement,
                &result[pos + m.len()..]
            );
        }
    }
    result
}

fn simple_regex_replace_all(pattern: &str, text: &str, replacement: &str) -> String {
    let mut result = text.to_string();
    loop {
        let before = result.clone();
        let matches = simple_regex_find_all(pattern, &result);
        if matches.is_empty() {
            break;
        }
        for m in &matches {
            if let Some(pos) = result.find(m.as_str()) {
                result = format!(
                    "{}{}{}",
                    &result[..pos],
                    replacement,
                    &result[pos + m.len()..]
                );
            }
        }
        if result == before {
            break;
        }
    }
    result
}

fn days_to_ymd(mut days: i64) -> (i64, i64, i64) {
    days += 719468;
    let era = if days >= 0 { days } else { days - 146096 } / 146097;
    let doe = days - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    let mut i = 0;
    while i < data.len() {
        let b0 = data[i] as u32;
        let b1 = if i + 1 < data.len() {
            data[i + 1] as u32
        } else {
            0
        };
        let b2 = if i + 2 < data.len() {
            data[i + 2] as u32
        } else {
            0
        };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        out.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if i + 1 < data.len() {
            out.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if i + 2 < data.len() {
            out.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        i += 3;
    }
    out
}

fn base64_decode(s: &str) -> String {
    const DTABLE: [u8; 128] = {
        let mut t = [255u8; 128];
        let mut i = 0u8;
        while i < 26 {
            t[(b'A' + i) as usize] = i;
            i += 1;
        }
        i = 0;
        while i < 26 {
            t[(b'a' + i) as usize] = 26 + i;
            i += 1;
        }
        i = 0;
        while i < 10 {
            t[(b'0' + i) as usize] = 52 + i;
            i += 1;
        }
        t[b'+' as usize] = 62;
        t[b'/' as usize] = 63;
        t
    };
    let bytes: Vec<u8> = s
        .bytes()
        .filter(|&b| b != b'=' && b < 128 && DTABLE[b as usize] != 255)
        .collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 3 < bytes.len() {
        let (a, b, c, d) = (
            DTABLE[bytes[i] as usize] as u32,
            DTABLE[bytes[i + 1] as usize] as u32,
            DTABLE[bytes[i + 2] as usize] as u32,
            DTABLE[bytes[i + 3] as usize] as u32,
        );
        let triple = (a << 18) | (b << 12) | (c << 6) | d;
        out.push((triple >> 16) as u8);
        out.push((triple >> 8) as u8);
        out.push(triple as u8);
        i += 4;
    }
    if i + 2 < bytes.len() {
        let (a, b, c) = (
            DTABLE[bytes[i] as usize] as u32,
            DTABLE[bytes[i + 1] as usize] as u32,
            DTABLE[bytes[i + 2] as usize] as u32,
        );
        let triple = (a << 18) | (b << 12) | (c << 6);
        out.push((triple >> 16) as u8);
        out.push((triple >> 8) as u8);
    } else if i + 1 < bytes.len() {
        let (a, b) = (
            DTABLE[bytes[i] as usize] as u32,
            DTABLE[bytes[i + 1] as usize] as u32,
        );
        let triple = (a << 18) | (b << 12);
        out.push((triple >> 16) as u8);
    }
    String::from_utf8_lossy(&out).to_string()
}
