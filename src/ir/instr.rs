use crate::ir::block::BlockId;
use crate::ir::types::IrType;
use crate::ir::value::ValueId;

/// Index of an instruction within a block's instruction list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InstrId(pub u32);

/// Binary arithmetic operations on scalars.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    /// Integer floor division.
    FloorDiv,
    /// Modulo / remainder.
    Mod,
    /// Element-wise comparisons: yield a bool scalar.
    CmpEq,
    CmpNe,
    CmpLt,
    CmpLe,
    CmpGt,
    CmpGe,
}

impl std::fmt::Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::Mul => "mul",
            BinOp::Div => "div",
            BinOp::FloorDiv => "floordiv",
            BinOp::Mod => "mod",
            BinOp::CmpEq => "cmpeq",
            BinOp::CmpNe => "cmpne",
            BinOp::CmpLt => "cmplt",
            BinOp::CmpLe => "cmple",
            BinOp::CmpGt => "cmpgt",
            BinOp::CmpGe => "cmpge",
        };
        f.write_str(s)
    }
}

/// Scalar unary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarUnaryOp {
    /// Arithmetic negation: `-x`
    Neg,
    /// Boolean NOT: `!x`
    Not,
}

impl std::fmt::Display for ScalarUnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarUnaryOp::Neg => f.write_str("neg"),
            ScalarUnaryOp::Not => f.write_str("not"),
        }
    }
}

/// Tensor-level operations. These are high-level and subject to lowering passes.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorOp {
    /// Einstein summation: einsum("mk,kn->mn", [inputs])
    Einsum { notation: String },
    /// Element-wise unary: relu, sigmoid, tanh, etc.
    Unary { op: String },
    /// Reshape a tensor to a new shape (must have same total element count).
    Reshape,
    /// Transpose with explicit axis permutation.
    Transpose { axes: Vec<usize> },
    /// Reduction along specified axes.
    Reduce {
        op: String,
        axes: Vec<usize>,
        keepdims: bool,
    },
}

/// A single instruction in SSA form.
///
/// Invariants:
/// - Every instruction that produces a value has exactly one result `ValueId`.
/// - Terminators (`Br`, `CondBr`, `Return`) are the last instruction in a block.
/// - No instruction may appear after a terminator.
#[derive(Debug, Clone)]
pub enum IrInstr {
    // ---- Scalar arithmetic ----
    BinOp {
        result: ValueId,
        op: BinOp,
        lhs: ValueId,
        rhs: ValueId,
        ty: IrType,
    },

    // ---- Constants ----
    ConstFloat {
        result: ValueId,
        value: f64,
        ty: IrType,
    },
    ConstInt {
        result: ValueId,
        value: i64,
        ty: IrType,
    },
    ConstBool {
        result: ValueId,
        value: bool,
    },

    // ---- Scalar unary operations ----
    UnaryOp {
        result: ValueId,
        op: ScalarUnaryOp,
        operand: ValueId,
        ty: IrType,
    },

    // ---- Tensor operations ----
    TensorOp {
        result: ValueId,
        op: TensorOp,
        inputs: Vec<ValueId>,
        result_ty: IrType,
    },

    // ---- Type casts ----
    /// Cast a scalar value from one type to another.
    Cast {
        result: ValueId,
        operand: ValueId,
        from_ty: IrType,
        to_ty: IrType,
    },

    // ---- Memory ----
    /// Load a scalar from a tensor at given indices.
    Load {
        result: ValueId,
        tensor: ValueId,
        indices: Vec<ValueId>,
        result_ty: IrType,
    },
    /// Store a scalar value into a tensor at given indices.
    /// Produces no result (side-effecting).
    Store {
        tensor: ValueId,
        indices: Vec<ValueId>,
        value: ValueId,
    },

    // ---- Control flow (terminators) ----
    /// Unconditional branch with block arguments (SSA block params).
    Br {
        target: BlockId,
        args: Vec<ValueId>,
    },
    /// Conditional branch.
    CondBr {
        cond: ValueId,
        then_block: BlockId,
        then_args: Vec<ValueId>,
        else_block: BlockId,
        else_args: Vec<ValueId>,
    },
    /// Return from function. Values must match the function's return type.
    Return {
        values: Vec<ValueId>,
    },

    // ---- Function calls ----
    Call {
        result: Option<ValueId>,
        callee: String,
        args: Vec<ValueId>,
        result_ty: Option<IrType>,
    },

    // ---- Struct operations ----
    /// Construct a struct value from field values.
    MakeStruct {
        result: ValueId,
        fields: Vec<ValueId>,
        result_ty: IrType,
    },
    /// Extract a field from a struct value by index.
    GetField {
        result: ValueId,
        base: ValueId,
        field_index: usize,
        result_ty: IrType,
    },

    // ---- Enum operations ----
    /// Construct an enum variant (tag integer).
    MakeVariant {
        result: ValueId,
        variant_idx: usize,
        result_ty: IrType,
    },
    /// Dispatch to a block based on enum variant tag (terminator).
    SwitchVariant {
        scrutinee: ValueId,
        /// (variant_index, target_block) pairs — must cover all variants.
        arms: Vec<(usize, BlockId)>,
        /// Fallback block if tag matches none (may be None for exhaustive match).
        default_block: Option<BlockId>,
    },

    // ---- Tuple operations ----
    /// Construct a tuple from element values.
    MakeTuple {
        result: ValueId,
        elements: Vec<ValueId>,
        result_ty: IrType,
    },
    /// Extract an element from a tuple by index.
    GetElement {
        result: ValueId,
        base: ValueId,
        index: usize,
        result_ty: IrType,
    },

    // ---- Closure operations ----
    /// Create a closure value from a function name and captured values.
    MakeClosure {
        result: ValueId,
        fn_name: String,
        captures: Vec<ValueId>,
        result_ty: IrType,
    },
    /// Call a closure value with the given arguments.
    CallClosure {
        result: Option<ValueId>,
        closure: ValueId,
        args: Vec<ValueId>,
        result_ty: IrType,
    },

    // ---- Array operations ----
    /// Allocate a fixed-length array and initialise it from a list of values.
    AllocArray {
        result: ValueId,
        elem_ty: IrType,
        size: usize,
        init: Vec<ValueId>,
    },
    /// Load one element from an array by index.
    ArrayLoad {
        result: ValueId,
        array: ValueId,
        index: ValueId,
        elem_ty: IrType,
    },
    /// Store a value into an array element by index (side-effecting, no result).
    ArrayStore {
        array: ValueId,
        index: ValueId,
        value: ValueId,
    },

    // ---- Option operations ----
    /// Wrap a value in Some.
    MakeSome { result: ValueId, value: ValueId, result_ty: IrType },
    /// Create a None value.
    MakeNone { result: ValueId, result_ty: IrType },
    /// Test if an option is Some. Yields bool.
    IsSome { result: ValueId, operand: ValueId },
    /// Unwrap a Some value, panicking at runtime on None.
    OptionUnwrap { result: ValueId, operand: ValueId, result_ty: IrType },

    // ---- Result operations ----
    /// Wrap a value in Ok.
    MakeOk { result: ValueId, value: ValueId, result_ty: IrType },
    /// Wrap a value in Err.
    MakeErr { result: ValueId, value: ValueId, result_ty: IrType },
    /// Test if a result is Ok. Yields bool.
    IsOk { result: ValueId, operand: ValueId },
    /// Unwrap the Ok value of a result.
    ResultUnwrap { result: ValueId, operand: ValueId, result_ty: IrType },
    /// Unwrap the Err value of a result.
    ResultUnwrapErr { result: ValueId, operand: ValueId, result_ty: IrType },

    // ---- Channel operations ----
    /// Create a new channel.
    ChanNew { result: ValueId, elem_ty: IrType },
    /// Send a value on a channel (side-effecting, no result).
    ChanSend { chan: ValueId, value: ValueId },
    /// Receive a value from a channel.
    ChanRecv { result: ValueId, chan: ValueId, elem_ty: IrType },
    /// Spawn a concurrent task (body is a lifted function name).
    Spawn { body_fn: String, args: Vec<ValueId> },

    /// Parallel for-loop over a range (sequential simulation).
    ParFor {
        var: ValueId,   // loop variable (result placeholder)
        start: ValueId,
        end: ValueId,
        body_fn: String,
        /// Captured outer-scope values passed as extra params to body_fn.
        args: Vec<ValueId>,
    },

    // ---- Atomic / Mutex operations ----
    /// Create a new atomic value.
    AtomicNew { result: ValueId, value: ValueId, result_ty: IrType },
    /// Load from an atomic value.
    AtomicLoad { result: ValueId, atomic: ValueId, result_ty: IrType },
    /// Store into an atomic value (side-effecting, no result).
    AtomicStore { atomic: ValueId, value: ValueId },
    /// Atomically add a value and return the new value.
    AtomicAdd { result: ValueId, atomic: ValueId, value: ValueId, result_ty: IrType },
    /// Create a new mutex-protected value.
    MutexNew { result: ValueId, value: ValueId, result_ty: IrType },
    /// Lock a mutex and return the inner value.
    MutexLock { result: ValueId, mutex: ValueId, result_ty: IrType },
    /// Unlock a mutex (side-effecting, no result).
    MutexUnlock { mutex: ValueId },

    // ---- Concurrency barrier ----
    /// A synchronization barrier (no-op in interpreter, marks sync point in parallel code).
    Barrier,

    // ---- Grad (dual number) operations ----
    /// Create a dual number with given value and tangent.
    MakeGrad {
        result: ValueId,
        value: ValueId,
        tangent: ValueId,
        ty: IrType,
    },
    /// Extract the primal value from a dual number.
    GradValue {
        result: ValueId,
        operand: ValueId,
        ty: IrType,
    },
    /// Extract the tangent (gradient) from a dual number.
    GradTangent {
        result: ValueId,
        operand: ValueId,
        ty: IrType,
    },

    // ---- Sparse tensor operations ----
    /// Convert a dense array/tensor to sparse representation.
    Sparsify { result: ValueId, operand: ValueId, ty: IrType },
    /// Convert a sparse representation back to dense.
    Densify { result: ValueId, operand: ValueId, ty: IrType },

    // ---- String operations ----
    /// A compile-time string constant.
    ConstStr { result: ValueId, value: String },
    /// Get the length (number of bytes) of a string.
    StrLen { result: ValueId, operand: ValueId },
    /// Concatenate two strings.
    StrConcat { result: ValueId, lhs: ValueId, rhs: ValueId },
    /// Print a value to stdout (side-effecting, no result).
    Print { operand: ValueId },
}

impl IrInstr {
    /// Returns the `ValueId` produced by this instruction, if any.
    /// Terminators and `Store` produce no value.
    pub fn result(&self) -> Option<ValueId> {
        match self {
            IrInstr::BinOp { result, .. } => Some(*result),
            IrInstr::UnaryOp { result, .. } => Some(*result),
            IrInstr::ConstFloat { result, .. } => Some(*result),
            IrInstr::ConstInt { result, .. } => Some(*result),
            IrInstr::ConstBool { result, .. } => Some(*result),
            IrInstr::TensorOp { result, .. } => Some(*result),
            IrInstr::Cast { result, .. } => Some(*result),
            IrInstr::Load { result, .. } => Some(*result),
            IrInstr::Store { .. } => None,
            IrInstr::Br { .. } => None,
            IrInstr::CondBr { .. } => None,
            IrInstr::Return { .. } => None,
            IrInstr::Call { result, .. } => *result,
            IrInstr::MakeStruct { result, .. } => Some(*result),
            IrInstr::GetField { result, .. } => Some(*result),
            IrInstr::MakeVariant { result, .. } => Some(*result),
            IrInstr::SwitchVariant { .. } => None,
            IrInstr::MakeTuple { result, .. } => Some(*result),
            IrInstr::GetElement { result, .. } => Some(*result),
            IrInstr::MakeClosure { result, .. } => Some(*result),
            IrInstr::CallClosure { result, .. } => *result,
            IrInstr::AllocArray { result, .. } => Some(*result),
            IrInstr::ArrayLoad { result, .. } => Some(*result),
            IrInstr::ArrayStore { .. } => None,
            IrInstr::ParFor { .. } => None,
            IrInstr::ChanNew { result, .. } => Some(*result),
            IrInstr::ChanSend { .. } => None,
            IrInstr::ChanRecv { result, .. } => Some(*result),
            IrInstr::Spawn { .. } => None,
            IrInstr::AtomicNew { result, .. } => Some(*result),
            IrInstr::AtomicLoad { result, .. } => Some(*result),
            IrInstr::AtomicStore { .. } => None,
            IrInstr::AtomicAdd { result, .. } => Some(*result),
            IrInstr::MutexNew { result, .. } => Some(*result),
            IrInstr::MutexLock { result, .. } => Some(*result),
            IrInstr::MutexUnlock { .. } => None,
            IrInstr::MakeSome { result, .. } => Some(*result),
            IrInstr::MakeNone { result, .. } => Some(*result),
            IrInstr::IsSome { result, .. } => Some(*result),
            IrInstr::OptionUnwrap { result, .. } => Some(*result),
            IrInstr::MakeOk { result, .. } => Some(*result),
            IrInstr::MakeErr { result, .. } => Some(*result),
            IrInstr::IsOk { result, .. } => Some(*result),
            IrInstr::ResultUnwrap { result, .. } => Some(*result),
            IrInstr::ResultUnwrapErr { result, .. } => Some(*result),
            IrInstr::Barrier => None,
            IrInstr::Sparsify { result, .. } => Some(*result),
            IrInstr::Densify { result, .. } => Some(*result),
            IrInstr::MakeGrad { result, .. } => Some(*result),
            IrInstr::GradValue { result, .. } => Some(*result),
            IrInstr::GradTangent { result, .. } => Some(*result),
            IrInstr::ConstStr { result, .. } => Some(*result),
            IrInstr::StrLen { result, .. } => Some(*result),
            IrInstr::StrConcat { result, .. } => Some(*result),
            IrInstr::Print { .. } => None,
        }
    }

    /// Returns `true` if this instruction is a block terminator.
    pub fn is_terminator(&self) -> bool {
        matches!(
            self,
            IrInstr::Br { .. }
                | IrInstr::CondBr { .. }
                | IrInstr::Return { .. }
                | IrInstr::SwitchVariant { .. }
        )
    }


    /// Returns all `ValueId`s consumed by this instruction (operands).
    pub fn operands(&self) -> Vec<ValueId> {
        match self {
            IrInstr::BinOp { lhs, rhs, .. } => vec![*lhs, *rhs],
            IrInstr::UnaryOp { operand, .. } => vec![*operand],
            IrInstr::ConstFloat { .. } => vec![],
            IrInstr::ConstInt { .. } => vec![],
            IrInstr::ConstBool { .. } => vec![],
            IrInstr::Cast { operand, .. } => vec![*operand],
            IrInstr::TensorOp { inputs, .. } => inputs.clone(),
            IrInstr::Load {
                tensor, indices, ..
            } => {
                let mut ops = vec![*tensor];
                ops.extend_from_slice(indices);
                ops
            }
            IrInstr::Store {
                tensor,
                indices,
                value,
            } => {
                let mut ops = vec![*tensor, *value];
                ops.extend_from_slice(indices);
                ops
            }
            IrInstr::Br { args, .. } => args.clone(),
            IrInstr::CondBr {
                cond,
                then_args,
                else_args,
                ..
            } => {
                let mut ops = vec![*cond];
                ops.extend_from_slice(then_args);
                ops.extend_from_slice(else_args);
                ops
            }
            IrInstr::Return { values } => values.clone(),
            IrInstr::Call { args, .. } => args.clone(),
            IrInstr::MakeStruct { fields, .. } => fields.clone(),
            IrInstr::GetField { base, .. } => vec![*base],
            IrInstr::MakeVariant { .. } => vec![],
            IrInstr::SwitchVariant { scrutinee, .. } => vec![*scrutinee],
            IrInstr::MakeTuple { elements, .. } => elements.clone(),
            IrInstr::GetElement { base, .. } => vec![*base],
            IrInstr::MakeClosure { captures, .. } => captures.clone(),
            IrInstr::CallClosure { closure, args, .. } => {
                let mut ops = vec![*closure];
                ops.extend_from_slice(args);
                ops
            }
            IrInstr::AllocArray { init, .. } => init.clone(),
            IrInstr::ArrayLoad { array, index, .. } => vec![*array, *index],
            IrInstr::ArrayStore { array, index, value } => vec![*array, *index, *value],
            IrInstr::ParFor { start, end, args, .. } => {
                let mut ops = vec![*start, *end];
                ops.extend_from_slice(args);
                ops
            }
            IrInstr::ChanNew { .. } => vec![],
            IrInstr::ChanSend { chan, value } => vec![*chan, *value],
            IrInstr::ChanRecv { chan, .. } => vec![*chan],
            IrInstr::Spawn { args, .. } => args.clone(),
            IrInstr::AtomicNew { value, .. } => vec![*value],
            IrInstr::AtomicLoad { atomic, .. } => vec![*atomic],
            IrInstr::AtomicStore { atomic, value } => vec![*atomic, *value],
            IrInstr::AtomicAdd { atomic, value, .. } => vec![*atomic, *value],
            IrInstr::MutexNew { value, .. } => vec![*value],
            IrInstr::MutexLock { mutex, .. } => vec![*mutex],
            IrInstr::MutexUnlock { mutex } => vec![*mutex],
            IrInstr::MakeSome { value, .. } => vec![*value],
            IrInstr::MakeNone { .. } => vec![],
            IrInstr::IsSome { operand, .. } => vec![*operand],
            IrInstr::OptionUnwrap { operand, .. } => vec![*operand],
            IrInstr::MakeOk { value, .. } => vec![*value],
            IrInstr::MakeErr { value, .. } => vec![*value],
            IrInstr::IsOk { operand, .. } => vec![*operand],
            IrInstr::ResultUnwrap { operand, .. } => vec![*operand],
            IrInstr::ResultUnwrapErr { operand, .. } => vec![*operand],
            IrInstr::Barrier => vec![],
            IrInstr::Sparsify { operand, .. } => vec![*operand],
            IrInstr::Densify { operand, .. } => vec![*operand],
            IrInstr::MakeGrad { value, tangent, .. } => vec![*value, *tangent],
            IrInstr::GradValue { operand, .. } => vec![*operand],
            IrInstr::GradTangent { operand, .. } => vec![*operand],
            IrInstr::ConstStr { .. } => vec![],
            IrInstr::StrLen { operand, .. } => vec![*operand],
            IrInstr::StrConcat { lhs, rhs, .. } => vec![*lhs, *rhs],
            IrInstr::Print { operand } => vec![*operand],
        }
    }
}
