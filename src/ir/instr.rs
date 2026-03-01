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
    /// Power: `pow(base, exp)`
    Pow,
    /// Minimum: `min(a, b)`
    Min,
    /// Maximum: `max(a, b)`
    Max,
    /// Bitwise AND: `a & b`
    BitAnd,
    /// Bitwise OR: `a | b`
    BitOr,
    /// Bitwise XOR: `a ^ b`
    BitXor,
    /// Logical left shift: `a << b`
    Shl,
    /// Arithmetic right shift: `a >> b`
    Shr,
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
            BinOp::Add      => "add",
            BinOp::Sub      => "sub",
            BinOp::Mul      => "mul",
            BinOp::Div      => "div",
            BinOp::FloorDiv => "floordiv",
            BinOp::Mod      => "mod",
            BinOp::Pow      => "pow",
            BinOp::Min      => "min",
            BinOp::Max      => "max",
            BinOp::BitAnd   => "band",
            BinOp::BitOr    => "bor",
            BinOp::BitXor   => "bxor",
            BinOp::Shl      => "shl",
            BinOp::Shr      => "shr",
            BinOp::CmpEq    => "cmpeq",
            BinOp::CmpNe    => "cmpne",
            BinOp::CmpLt    => "cmplt",
            BinOp::CmpLe    => "cmple",
            BinOp::CmpGt    => "cmpgt",
            BinOp::CmpGe    => "cmpge",
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
    /// Square root: `sqrt(x)`
    Sqrt,
    /// Absolute value: `abs(x)`
    Abs,
    /// Floor: `floor(x)`
    Floor,
    /// Ceiling: `ceil(x)`
    Ceil,
    /// Bitwise NOT: `~x`
    BitNot,
    /// Sine: `sin(x)`
    Sin,
    /// Cosine: `cos(x)`
    Cos,
    /// Tangent: `tan(x)`
    Tan,
    /// Natural exponential: `exp(x)`
    Exp,
    /// Natural logarithm: `log(x)`
    Log,
    /// Base-2 logarithm: `log2(x)`
    Log2,
    /// Round to nearest integer: `round(x)`
    Round,
    /// Sign function: `sign(x)` → -1, 0, or 1
    Sign,
}

impl std::fmt::Display for ScalarUnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarUnaryOp::Neg    => f.write_str("neg"),
            ScalarUnaryOp::Not    => f.write_str("not"),
            ScalarUnaryOp::Sqrt   => f.write_str("sqrt"),
            ScalarUnaryOp::Abs    => f.write_str("abs"),
            ScalarUnaryOp::Floor  => f.write_str("floor"),
            ScalarUnaryOp::Ceil   => f.write_str("ceil"),
            ScalarUnaryOp::BitNot => f.write_str("bnot"),
            ScalarUnaryOp::Sin    => f.write_str("sin"),
            ScalarUnaryOp::Cos    => f.write_str("cos"),
            ScalarUnaryOp::Tan    => f.write_str("tan"),
            ScalarUnaryOp::Exp    => f.write_str("exp"),
            ScalarUnaryOp::Log    => f.write_str("log"),
            ScalarUnaryOp::Log2   => f.write_str("log2"),
            ScalarUnaryOp::Round  => f.write_str("round"),
            ScalarUnaryOp::Sign   => f.write_str("sign"),
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
    /// Call an extern (C-linkage) function declared with `extern def`.
    CallExtern {
        result: Option<ValueId>,
        name: String,
        args: Vec<ValueId>,
        ret_ty: IrType,
    },

    // ---- Phase 83: Ref-counting GC ----
    /// Increment the reference count of a heap value (list, map, option, …).
    Retain { ptr: ValueId },
    /// Decrement the reference count; frees the value when it reaches zero.
    Release { ptr: ValueId, ty: IrType },

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
    /// Construct an enum variant (tag integer, optional payload fields).
    MakeVariant {
        result: ValueId,
        variant_idx: usize,
        /// Payload field values (empty for unit variants).
        fields: Vec<ValueId>,
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
    /// Extract a payload field from an enum variant value.
    ExtractVariantField {
        result: ValueId,
        /// The enum value to extract from.
        operand: ValueId,
        /// Which variant index we expect (for documentation/verification).
        variant_idx: usize,
        /// Which field within that variant to extract (0-indexed).
        field_idx: usize,
        result_ty: IrType,
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
    // ---- Extended string operations ----
    /// `contains(s, sub)` → bool
    StrContains { result: ValueId, haystack: ValueId, needle: ValueId },
    /// `starts_with(s, prefix)` → bool
    StrStartsWith { result: ValueId, haystack: ValueId, prefix: ValueId },
    /// `ends_with(s, suffix)` → bool
    StrEndsWith { result: ValueId, haystack: ValueId, suffix: ValueId },
    /// `to_upper(s)` → str
    StrToUpper { result: ValueId, operand: ValueId },
    /// `to_lower(s)` → str
    StrToLower { result: ValueId, operand: ValueId },
    /// `trim(s)` → str  (strips leading/trailing ASCII whitespace)
    StrTrim { result: ValueId, operand: ValueId },
    /// `repeat(s, n)` → str
    StrRepeat { result: ValueId, operand: ValueId, count: ValueId },
    /// Unconditional abort with a message string. Terminates execution.
    Panic { msg: ValueId },
    /// Convert any scalar or string value to its string representation.
    ValueToStr { result: ValueId, operand: ValueId },

    // ---- User input ----
    /// Read a line from stdin (strips trailing newline). Returns str.
    ReadLine { result: ValueId },
    /// Read a line from stdin and parse it as i64.
    ReadI64 { result: ValueId },
    /// Read a line from stdin and parse it as f64.
    ReadF64 { result: ValueId },

    // ---- String parsing ----
    /// Parse a str as i64. Returns option<i64>: some(n) on success, none on failure.
    ParseI64 { result: ValueId, operand: ValueId },
    /// Parse a str as f64. Returns option<f64>: some(x) on success, none on failure.
    ParseF64 { result: ValueId, operand: ValueId },

    // ---- String indexing and slicing ----
    /// `str_index(s, i)` → i64 byte value at position i.
    StrIndex { result: ValueId, string: ValueId, index: ValueId },
    /// `slice(s, start, end)` → str substring [start..end).
    StrSlice { result: ValueId, string: ValueId, start: ValueId, end: ValueId },
    /// `find(s, sub)` → option<i64>: index of first occurrence of sub in s, or none.
    StrFind { result: ValueId, haystack: ValueId, needle: ValueId },
    /// `str_replace(s, old, new)` → str with all occurrences of old replaced by new.
    StrReplace { result: ValueId, string: ValueId, from: ValueId, to: ValueId },

    // ---- Dynamic list operations ----
    /// Create a new empty list of the given element type.
    ListNew { result: ValueId, elem_ty: IrType },
    /// Append a value to a list (side-effecting, no SSA result).
    ListPush { list: ValueId, value: ValueId },
    /// Get the length of a list. Returns i64.
    ListLen { result: ValueId, list: ValueId },
    /// Load an element from a list by index. Returns elem_ty.
    ListGet { result: ValueId, list: ValueId, index: ValueId, elem_ty: IrType },
    /// Store a value into a list element by index (side-effecting).
    ListSet { list: ValueId, index: ValueId, value: ValueId },
    /// Remove and return the last element of a list. Returns option<elem_ty>.
    ListPop { result: ValueId, list: ValueId, elem_ty: IrType },

    // ---- HashMap operations ----
    /// Create a new empty map with the given key/value types.
    MapNew { result: ValueId, key_ty: IrType, val_ty: IrType },
    /// Insert or update a key-value pair (side-effecting, no SSA result).
    MapSet { map: ValueId, key: ValueId, value: ValueId },
    /// Get the value for a key. Returns option<val_ty>.
    MapGet { result: ValueId, map: ValueId, key: ValueId, val_ty: IrType },
    /// Check whether a key exists. Returns bool.
    MapContains { result: ValueId, map: ValueId, key: ValueId },
    /// Remove a key from the map (side-effecting, no SSA result).
    MapRemove { map: ValueId, key: ValueId },
    /// Return the number of entries in the map. Returns i64.
    MapLen { result: ValueId, map: ValueId },

    // ---- File I/O operations (Phase 56) ----
    /// Read entire file as a string. Returns result<str, str>.
    FileReadAll { result: ValueId, path: ValueId },
    /// Write string to a file. Returns result<unit, str>.
    FileWriteAll { result: ValueId, path: ValueId, content: ValueId },
    /// Check if a file exists. Returns bool.
    FileExists { result: ValueId, path: ValueId },
    /// Read file as a list of lines. Returns list<str>.
    FileLines { result: ValueId, path: ValueId },

    // ---- Extended collection operations (Phase 58) ----
    /// Check if a list contains a value. Returns bool.
    ListContains { result: ValueId, list: ValueId, value: ValueId },
    /// Sort a list in-place (side-effecting, no result).
    ListSort { list: ValueId },
    /// Get all keys of a map as list<str>.
    MapKeys { result: ValueId, map: ValueId },
    /// Get all values of a map as a list.
    MapValues { result: ValueId, map: ValueId },
    /// Concatenate two lists into a new list.
    ListConcat { result: ValueId, lhs: ValueId, rhs: ValueId },
    /// Slice a list from start to end (exclusive). Returns list.
    ListSlice { result: ValueId, list: ValueId, start: ValueId, end: ValueId },

    // ---- Process / environment operations (Phase 59) ----
    /// Exit the process with the given i64 exit code (side-effecting, no result).
    ProcessExit { code: ValueId },
    /// Get command-line arguments as list<str>.
    ProcessArgs { result: ValueId },
    /// Get an environment variable by name. Returns option<str>.
    EnvVar { result: ValueId, name: ValueId },

    // ---- Pattern matching helpers (Phase 61) ----
    /// Extract the tag (variant index as i64) from an enum value.
    GetVariantTag { result: ValueId, operand: ValueId },
    /// Compare two string values for equality. Returns bool.
    StrEq { result: ValueId, lhs: ValueId, rhs: ValueId },

    // ---- Phase 88: TCP network I/O ----
    /// Connect to a TCP server. Returns socket fd (i64).
    TcpConnect { result: ValueId, host: ValueId, port: ValueId },
    /// Listen on a TCP port. Returns listener fd (i64).
    TcpListen { result: ValueId, port: ValueId },
    /// Accept a connection from a listener. Returns connection fd (i64).
    TcpAccept { result: ValueId, listener: ValueId },
    /// Read a line from a TCP connection. Returns str.
    TcpRead { result: ValueId, conn: ValueId },
    /// Write a string to a TCP connection. Side-effecting.
    TcpWrite { conn: ValueId, data: ValueId },
    /// Close a TCP connection or listener. Side-effecting.
    TcpClose { conn: ValueId },

    // ---- Phase 95: String split/join ----
    /// Split a string by a delimiter. Returns list<str>.
    StrSplit { result: ValueId, str_val: ValueId, delim: ValueId },
    /// Join a list<str> with a delimiter string. Returns str.
    StrJoin { result: ValueId, list_val: ValueId, delim: ValueId },

    // ---- Phase 97: Time / OS ----
    /// Returns current time in milliseconds since Unix epoch. Returns i64.
    NowMs { result: ValueId },
    /// Sleep for the given number of milliseconds. Side-effecting; returns i64 (0).
    SleepMs { result: ValueId, ms: ValueId },
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
            IrInstr::ExtractVariantField { result, .. } => Some(*result),
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
            IrInstr::StrContains { result, .. } => Some(*result),
            IrInstr::StrStartsWith { result, .. } => Some(*result),
            IrInstr::StrEndsWith { result, .. } => Some(*result),
            IrInstr::StrToUpper { result, .. } => Some(*result),
            IrInstr::StrToLower { result, .. } => Some(*result),
            IrInstr::StrTrim { result, .. } => Some(*result),
            IrInstr::StrRepeat { result, .. } => Some(*result),
            IrInstr::Panic { .. } => None,
            IrInstr::ValueToStr { result, .. } => Some(*result),
            IrInstr::ReadLine { result } => Some(*result),
            IrInstr::ReadI64 { result } => Some(*result),
            IrInstr::ReadF64 { result } => Some(*result),
            IrInstr::ParseI64 { result, .. } => Some(*result),
            IrInstr::ParseF64 { result, .. } => Some(*result),
            IrInstr::StrIndex { result, .. } => Some(*result),
            IrInstr::StrSlice { result, .. } => Some(*result),
            IrInstr::StrFind { result, .. } => Some(*result),
            IrInstr::StrReplace { result, .. } => Some(*result),
            IrInstr::ListNew { result, .. } => Some(*result),
            IrInstr::ListPush { .. } => None,
            IrInstr::ListLen { result, .. } => Some(*result),
            IrInstr::ListGet { result, .. } => Some(*result),
            IrInstr::ListSet { .. } => None,
            IrInstr::ListPop { result, .. } => Some(*result),
            IrInstr::MapNew { result, .. } => Some(*result),
            IrInstr::MapSet { .. } => None,
            IrInstr::MapGet { result, .. } => Some(*result),
            IrInstr::MapContains { result, .. } => Some(*result),
            IrInstr::MapRemove { .. } => None,
            IrInstr::MapLen { result, .. } => Some(*result),
            // Phase 56: File I/O
            IrInstr::FileReadAll { result, .. } => Some(*result),
            IrInstr::FileWriteAll { result, .. } => Some(*result),
            IrInstr::FileExists { result, .. } => Some(*result),
            IrInstr::FileLines { result, .. } => Some(*result),
            // Phase 58: Extended collections
            IrInstr::ListContains { result, .. } => Some(*result),
            IrInstr::ListSort { .. } => None,
            IrInstr::MapKeys { result, .. } => Some(*result),
            IrInstr::MapValues { result, .. } => Some(*result),
            IrInstr::ListConcat { result, .. } => Some(*result),
            IrInstr::ListSlice { result, .. } => Some(*result),
            // Phase 59: Process / environment
            IrInstr::ProcessExit { .. } => None,
            IrInstr::ProcessArgs { result } => Some(*result),
            IrInstr::EnvVar { result, .. } => Some(*result),
            // Phase 61: Pattern matching helpers
            IrInstr::GetVariantTag { result, .. } => Some(*result),
            IrInstr::StrEq { result, .. } => Some(*result),
            IrInstr::CallExtern { result, .. } => *result,
            IrInstr::Retain { .. } => None,
            IrInstr::Release { .. } => None,
            IrInstr::TcpConnect { result, .. } => Some(*result),
            IrInstr::TcpListen { result, .. } => Some(*result),
            IrInstr::TcpAccept { result, .. } => Some(*result),
            IrInstr::TcpRead { result, .. } => Some(*result),
            IrInstr::TcpWrite { .. } => None,
            IrInstr::TcpClose { .. } => None,
            IrInstr::StrSplit { result, .. } => Some(*result),
            IrInstr::StrJoin { result, .. } => Some(*result),
            IrInstr::NowMs { result } => Some(*result),
            IrInstr::SleepMs { result, .. } => Some(*result),
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
            IrInstr::MakeVariant { fields, .. } => fields.clone(),
            IrInstr::SwitchVariant { scrutinee, .. } => vec![*scrutinee],
            IrInstr::ExtractVariantField { operand, .. } => vec![*operand],
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
            IrInstr::StrContains { haystack, needle, .. } => vec![*haystack, *needle],
            IrInstr::StrStartsWith { haystack, prefix, .. } => vec![*haystack, *prefix],
            IrInstr::StrEndsWith { haystack, suffix, .. } => vec![*haystack, *suffix],
            IrInstr::StrToUpper { operand, .. } => vec![*operand],
            IrInstr::StrToLower { operand, .. } => vec![*operand],
            IrInstr::StrTrim { operand, .. } => vec![*operand],
            IrInstr::StrRepeat { operand, count, .. } => vec![*operand, *count],
            IrInstr::Panic { msg } => vec![*msg],
            IrInstr::ValueToStr { operand, .. } => vec![*operand],
            IrInstr::ReadLine { .. } => vec![],
            IrInstr::ReadI64 { .. } => vec![],
            IrInstr::ReadF64 { .. } => vec![],
            IrInstr::ParseI64 { operand, .. } => vec![*operand],
            IrInstr::ParseF64 { operand, .. } => vec![*operand],
            IrInstr::StrIndex { string, index, .. } => vec![*string, *index],
            IrInstr::StrSlice { string, start, end, .. } => vec![*string, *start, *end],
            IrInstr::StrFind { haystack, needle, .. } => vec![*haystack, *needle],
            IrInstr::StrReplace { string, from, to, .. } => vec![*string, *from, *to],
            IrInstr::ListNew { .. } => vec![],
            IrInstr::ListPush { list, value } => vec![*list, *value],
            IrInstr::ListLen { list, .. } => vec![*list],
            IrInstr::ListGet { list, index, .. } => vec![*list, *index],
            IrInstr::ListSet { list, index, value } => vec![*list, *index, *value],
            IrInstr::ListPop { list, .. } => vec![*list],
            IrInstr::MapNew { .. } => vec![],
            IrInstr::MapSet { map, key, value } => vec![*map, *key, *value],
            IrInstr::MapGet { map, key, .. } => vec![*map, *key],
            IrInstr::MapContains { map, key, .. } => vec![*map, *key],
            IrInstr::MapRemove { map, key } => vec![*map, *key],
            IrInstr::MapLen { map, .. } => vec![*map],
            // Phase 56: File I/O
            IrInstr::FileReadAll { path, .. } => vec![*path],
            IrInstr::FileWriteAll { path, content, .. } => vec![*path, *content],
            IrInstr::FileExists { path, .. } => vec![*path],
            IrInstr::FileLines { path, .. } => vec![*path],
            // Phase 58: Extended collections
            IrInstr::ListContains { list, value, .. } => vec![*list, *value],
            IrInstr::ListSort { list } => vec![*list],
            IrInstr::MapKeys { map, .. } => vec![*map],
            IrInstr::MapValues { map, .. } => vec![*map],
            IrInstr::ListConcat { lhs, rhs, .. } => vec![*lhs, *rhs],
            IrInstr::ListSlice { list, start, end, .. } => vec![*list, *start, *end],
            // Phase 59: Process / environment
            IrInstr::ProcessExit { code } => vec![*code],
            IrInstr::ProcessArgs { .. } => vec![],
            IrInstr::EnvVar { name, .. } => vec![*name],
            // Phase 61: Pattern matching helpers
            IrInstr::GetVariantTag { operand, .. } => vec![*operand],
            IrInstr::StrEq { lhs, rhs, .. } => vec![*lhs, *rhs],
            IrInstr::CallExtern { args, .. } => args.clone(),
            IrInstr::Retain { ptr } => vec![*ptr],
            IrInstr::Release { ptr, .. } => vec![*ptr],
            IrInstr::TcpConnect { host, port, .. } => vec![*host, *port],
            IrInstr::TcpListen { port, .. } => vec![*port],
            IrInstr::TcpAccept { listener, .. } => vec![*listener],
            IrInstr::TcpRead { conn, .. } => vec![*conn],
            IrInstr::TcpWrite { conn, data } => vec![*conn, *data],
            IrInstr::TcpClose { conn } => vec![*conn],
            IrInstr::StrSplit { str_val, delim, .. } => vec![*str_val, *delim],
            IrInstr::StrJoin { list_val, delim, .. } => vec![*list_val, *delim],
            IrInstr::NowMs { .. } => vec![],
            IrInstr::SleepMs { ms, .. } => vec![*ms],
        }
    }
}
