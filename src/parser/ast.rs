use crate::parser::lexer::Span;
use crate::ir::instr::BinOp;

/// A macro definition: `defmacro name(params) => body_expr`
#[derive(Debug, Clone)]
pub struct AstMacroDef {
    pub name: Ident,
    pub params: Vec<String>,
    pub body: Box<AstExpr>,
    pub span: Span,
    /// Doc comment (`/// ...`) preceding this item, if any.
    pub doc_comment: Option<String>,
}

/// An attribute annotation with optional arguments, e.g. `@adaptive(learning_rate=0.01)`.
#[derive(Debug, Clone)]
pub struct AstAttribute {
    pub name: String,
    pub args: Vec<AstExpr>,  // Positional and named arguments
    pub span: Span,
}

/// An identifier with its source location.
#[derive(Debug, Clone)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

/// A dimension in a tensor shape.
#[derive(Debug, Clone)]
pub enum AstDim {
    Literal(u64),
    Symbol(Ident),
}

/// Scalar kind as parsed from the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AstScalarKind {
    F32,
    F64,
    I32,
    I64,
    Bool,
    // Extended integer types (Phase 63)
    U8,
    I8,
    U32,
    U64,
    USize,
}

/// A parsed type expression.
#[derive(Debug, Clone)]
pub enum AstType {
    Scalar(AstScalarKind, Span),
    /// `tensor<dtype, [dims]>`
    Tensor {
        dtype: AstScalarKind,
        dims: Vec<AstDim>,
        span: Span,
    },
    /// A named struct type, e.g. `Point`.
    Named(String, Span),
    /// A tuple type, e.g. `(i64, f64, bool)`.
    Tuple(Vec<AstType>, Span),
    /// A fixed-length array type, e.g. `[i64; 5]` or `[T; N]` with const generic.
    Array {
        elem: Box<AstType>,
        len: usize,
        /// Expression for const-generic array length (e.g. `N` from `[T; N]`).
        len_expr: Option<Box<AstExpr>>,
        span: Span,
    },
    /// `option<T>` optional type.
    Option(Box<AstType>, Span),
    /// `result<T, E>` result type.
    Result(Box<AstType>, Box<AstType>, Span),
    /// `chan<T>` channel type.
    Chan(Box<AstType>, Span),
    /// `atomic<T>` atomic type.
    Atomic(Box<AstType>, Span),
    /// `mutex<T>` mutex type.
    Mutex(Box<AstType>, Span),
    /// `grad<T>` dual number type for automatic differentiation.
    Grad(Box<AstType>, Span),
    /// `sparse<T>` sparse tensor/array type.
    Sparse(Box<AstType>, Span),
    /// `list<T>` dynamic list type.
    List(Box<AstType>, Span),
    /// `map<K, V>` map type.
    Map(Box<AstType>, Box<AstType>, Span),
    /// `weak_ref<T>` weak reference type.
    WeakRef(Box<AstType>, Span),
    /// Function type, e.g. `(i64, bool) -> i64`.
    Fn {
        params: Vec<AstType>,
        ret: Box<AstType>,
        span: Span,
    },
    /// Generic type application, e.g. `Box<i64>`, `Pair<str, f64>`, `Array<i64, 16>`.
    Generic {
        name: String,
        args: Vec<AstType>,
        span: Span,
    },
    /// Constant integer literal used as a type-level argument (e.g. array length in generics).
    ConstInt(i64, Span),
    /// Associated type reference: `Self::Item` or `T::Item`.
    AssocType {
        base: String,
        assoc_name: String,
        span: Span,
    },
    /// `dyn Trait` — trait object type (fat pointer + vtable).
    DynTrait {
        trait_name: String,
        span: Span,
    },
    /// `with e1, e2` — effect mask type for first-class masks.
    MaskEffectType {
        effects: Vec<String>,
        span: Span,
    },
    /// `&T` immutable reference type.
    Ref(Box<AstType>, Span),
    /// `&mut T` mutable reference type.
    RefMut(Box<AstType>, Span),
}

impl PartialEq for AstType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (AstType::Scalar(k1, _), AstType::Scalar(k2, _)) => k1 == k2,
            (AstType::Tensor { dtype: dt1, dims: d1, .. }, AstType::Tensor { dtype: dt2, dims: d2, .. }) => {
                if dt1 != dt2 || d1.len() != d2.len() {
                    return false;
                }
                for (x, y) in d1.iter().zip(d2.iter()) {
                    match (x, y) {
                        (AstDim::Literal(l1), AstDim::Literal(l2)) => if l1 != l2 { return false; },
                        (AstDim::Symbol(i1), AstDim::Symbol(i2)) => if i1.name != i2.name { return false; },
                        _ => return false,
                    }
                }
                true
            }
            (AstType::Named(n1, _), AstType::Named(n2, _)) => n1 == n2,
            (AstType::Tuple(e1, _), AstType::Tuple(e2, _)) => e1 == e2,
            (AstType::Array { elem: el1, len: l1, .. }, AstType::Array { elem: el2, len: l2, .. }) => el1 == el2 && l1 == l2,
            (AstType::Option(i1, _), AstType::Option(i2, _)) => i1 == i2,
            (AstType::Result(ok1, err1, _), AstType::Result(ok2, err2, _)) => ok1 == ok2 && err1 == err2,
            (AstType::Chan(i1, _), AstType::Chan(i2, _)) => i1 == i2,
            (AstType::Atomic(i1, _), AstType::Atomic(i2, _)) => i1 == i2,
            (AstType::Mutex(i1, _), AstType::Mutex(i2, _)) => i1 == i2,
            (AstType::Grad(i1, _), AstType::Grad(i2, _)) => i1 == i2,
            (AstType::Sparse(i1, _), AstType::Sparse(i2, _)) => i1 == i2,
            (AstType::List(i1, _), AstType::List(i2, _)) => i1 == i2,
            (AstType::Map(k1, v1, _), AstType::Map(k2, v2, _)) => k1 == k2 && v1 == v2,
            (AstType::WeakRef(i1, _), AstType::WeakRef(i2, _)) => i1 == i2,
            (AstType::Fn { params: p1, ret: r1, .. }, AstType::Fn { params: p2, ret: r2, .. }) => p1 == p2 && r1 == r2,
            (AstType::Generic { name: n1, args: a1, .. }, AstType::Generic { name: n2, args: a2, .. }) => n1 == n2 && a1 == a2,
            (AstType::ConstInt(v1, _), AstType::ConstInt(v2, _)) => v1 == v2,
            (AstType::AssocType { base: b1, assoc_name: a1, .. }, AstType::AssocType { base: b2, assoc_name: a2, .. }) => b1 == b2 && a1 == a2,
            (AstType::DynTrait { trait_name: t1, .. }, AstType::DynTrait { trait_name: t2, .. }) => t1 == t2,
            (AstType::MaskEffectType { effects: e1, .. }, AstType::MaskEffectType { effects: e2, .. }) => e1 == e2,
            (AstType::Ref(i1, _), AstType::Ref(i2, _)) => i1 == i2,
            (AstType::RefMut(i1, _), AstType::RefMut(i2, _)) => i1 == i2,
            _ => false,
        }
    }
}

impl Eq for AstType {}

impl std::hash::Hash for AstType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            AstType::Scalar(k, _) => {
                (*k as u8).hash(state);
            }
            AstType::Tensor { dtype, dims, .. } => {
                (*dtype as u8).hash(state);
                for d in dims {
                    match d {
                        AstDim::Literal(l) => {
                            0u8.hash(state);
                            l.hash(state);
                        }
                        AstDim::Symbol(i) => {
                            1u8.hash(state);
                            i.name.hash(state);
                        }
                    }
                }
            }
            AstType::Named(name, _) => name.hash(state),
            AstType::Tuple(elems, _) => elems.hash(state),
            AstType::Array { elem, len, .. } => {
                elem.hash(state);
                len.hash(state);
            }
            AstType::Option(inner, _) => inner.hash(state),
            AstType::Result(ok, err, _) => {
                ok.hash(state);
                err.hash(state);
            }
            AstType::Chan(inner, _) => inner.hash(state),
            AstType::Atomic(inner, _) => inner.hash(state),
            AstType::Mutex(inner, _) => inner.hash(state),
            AstType::Grad(inner, _) => inner.hash(state),
            AstType::Sparse(inner, _) => inner.hash(state),
            AstType::List(inner, _) => inner.hash(state),
            AstType::Map(k, v, _) => {
                k.hash(state);
                v.hash(state);
            }
            AstType::WeakRef(inner, _) => inner.hash(state),
            AstType::Fn { params, ret, .. } => {
                params.hash(state);
                ret.hash(state);
            }
            AstType::Generic { name, args, .. } => {
                name.hash(state);
                args.hash(state);
            }
            AstType::ConstInt(v, _) => v.hash(state),
            AstType::AssocType { base, assoc_name, .. } => {
                base.hash(state);
                assoc_name.hash(state);
            }
            AstType::DynTrait { trait_name, .. } => trait_name.hash(state),
            AstType::MaskEffectType { effects, .. } => effects.hash(state),
            AstType::Ref(inner, _) => inner.hash(state),
            AstType::RefMut(inner, _) => inner.hash(state),
        }
    }
}

impl AstType {
    pub fn span(&self) -> Span {
        match self {
            AstType::Scalar(_, s) => *s,
            AstType::Tensor { span, .. } => *span,
            AstType::Named(_, s) => *s,
            AstType::Tuple(_, s) => *s,
            AstType::Array { span, .. } => *span,
            AstType::Option(_, s) => *s,
            AstType::Result(_, _, s) => *s,
            AstType::Chan(_, s) => *s,
            AstType::Atomic(_, s) => *s,
            AstType::Mutex(_, s) => *s,
            AstType::Grad(_, s) => *s,
            AstType::Sparse(_, s) => *s,
            AstType::List(_, s) => *s,
            AstType::Map(_, _, s) => *s,
            AstType::Fn { span, .. } => *span,
            AstType::Generic { span, .. } => *span,
            AstType::ConstInt(_, s) => *s,
            AstType::AssocType { span, .. } => *span,
            AstType::DynTrait { span, .. } => *span,
            AstType::MaskEffectType { span, .. } => *span,
            AstType::WeakRef(_, s) => *s,
            AstType::Ref(_, s) => *s,
            AstType::RefMut(_, s) => *s,
        }
    }
}

/// A function parameter.
#[derive(Debug, Clone)]
pub struct AstParam {
    pub name: Ident,
    pub ty: AstType,
    /// Optional default value expression (for `def f(x: i64 = 0)`).
    pub default: Option<AstExpr>,
}

/// Variance annotation for type parameters: `+T` (covariant), `-T` (contravariant), `T` (invariant).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variance {
    Covariant,
    Contravariant,
    Invariant,
}

/// A generic parameter — either a type param `T` or a const param `const N: usize`.
#[derive(Debug, Clone)]
pub enum AstGenericParam {
    /// Type parameter with optional trait bounds (e.g. `T where T: Ord, Sensory`) and variance.
    Type(String, Vec<String>, Variance),
    /// Higher-kinded type parameter: name, nested type parameters (e.g. `[T]`), bounds, variance
    Hkt(String, Vec<AstGenericParam>, Vec<String>, Variance),
    Const { name: String, kind: Box<AstType> },
}

/// A function definition.
#[derive(Debug, Clone)]
pub struct AstFunction {
    pub name: Ident,
    /// Whether this function is publicly exported (`pub def`).
    pub is_pub: bool,
    /// Type parameter names, e.g. `["T", "U"]` for `def f[T, U](...)`.
    pub type_params: Vec<AstGenericParam>,
    pub params: Vec<AstParam>,
    pub return_ty: AstType,
    /// Effect row: `effect io, fs, alloc`. Empty means pure.
    pub effects: Vec<String>,
    pub body: AstBlock,
    pub span: Span,
    pub is_async: bool,
    /// Whether this function is `const def` — can be evaluated at compile time.
    pub is_const: bool,
    /// Attribute annotations, e.g. `@adaptive(learning_rate=0.01)` for `@adaptive def f(...)`
    pub attrs: Vec<AstAttribute>,
    /// Doc comment (`/// ...`) preceding this item, if any.
    pub doc_comment: Option<String>,
}

/// A block of statements with an optional tail expression (the block's value).
#[derive(Debug, Clone)]
pub struct AstBlock {
    pub stmts: Vec<AstStmt>,
    /// The final expression in the block, if any. Its value is the block's value.
    pub tail: Option<Box<AstExpr>>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum AstStmt {
    /// `let <name>[: <ty>] = <expr>`
    Let {
        name: Ident,
        ty: Option<AstType>,
        init: Box<AstExpr>,
        is_var: bool,
        span: Span,
    },
    /// An expression used for its side effects (followed by `;`).
    Expr(Box<AstExpr>),
    While {
        label: Option<String>,
        cond: Box<AstExpr>,
        body: AstBlock,
        span: Span,
    },
    Loop {
        label: Option<String>,
        body: AstBlock,
        span: Span,
    },
    Break {
        label: Option<String>,
        span: Span,
    },
    Continue {
        label: Option<String>,
        span: Span,
    },
    /// `for <var> in <start>..<end> { <body> }` or `for <var> in <start>..=<end> { <body> }` range loop (sugar over while).
    ForRange {
        label: Option<String>,
        var: Ident,
        start: Box<AstExpr>,
        end: Box<AstExpr>,
        inclusive: bool,
        step: Option<Box<AstExpr>>,
        body: AstBlock,
        span: Span,
    },
    /// `lvalue = expr` or compound assignment `lvalue += expr`, etc.
    Assign {
        target: Box<AstExpr>,
        op: Option<BinOp>,
        value: Box<AstExpr>,
        span: Span,
    },
    /// `val (a, b, ...) = expr` destructuring tuple let.
    LetTuple {
        names: Vec<Ident>,
        init: Box<AstExpr>,
        is_var: bool,
        span: Span,
    },
    /// `return [expr]` early return from function.
    Return {
        value: Option<Box<AstExpr>>,
        span: Span,
    },
    /// `spawn { body }` — launch a concurrent task.
    /// `spawn(group) { body }` — launch a concurrent task in a TaskGroup.
    Spawn {
        body: Vec<AstStmt>,
        span: Span,
        group: Option<Box<AstExpr>>,
    },
    /// `par for <var> in <start>..<end> { body }` or `par for <var> in <start>..=<end> { body }` — parallel range iteration.
    ParFor {
        label: Option<String>,
        var: Ident,
        start: Box<AstExpr>,
        end: Box<AstExpr>,
        inclusive: bool,
        body: AstBlock,
        span: Span,
    },
    /// `for <var> in <list_expr> { body }` — foreach over a list.
    ForEach {
        label: Option<String>,
        var: Ident,
        iter: Box<AstExpr>,
        body: AstBlock,
        span: Span,
    },
    /// `with <effect1, effect2, ...> { <body> }` — restrict the body's effects.
    /// The body's call-site verification uses ONLY this effect row.
    MaskStmt {
        effects: Vec<String>,
        body: AstBlock,
        span: Span,
    },
    /// `handle <expr> with { <arm1>, <arm2>, ... }` — install algebraic-effect handlers.
    HandleStmt {
        expr: Box<AstExpr>,
        arms: Vec<AstHandlerArm>,
        return_ty: Box<AstType>,
        span: Span,
    },
    /// `defer <expr>` — evaluate expression on scope exit (LIFO order).
    Defer {
        expr: Box<AstExpr>,
        span: Span,
    },
    /// `select! { msg = ch => { body }, ... default => { body } }` — channel multiplexing.
    Select {
        arms: Vec<SelectArm>,
        default: Option<Box<AstBlock>>,
        span: Span,
    },
    /// `yield <expr>` — produce a value in a generator function (desugars to list push).
    Yield {
        expr: Box<AstExpr>,
        span: Span,
    },
}

/// A handler arm: `k1(p1, p2) -> resume(v) => body`.
#[derive(Debug, Clone)]
pub struct AstHandlerArm {
    /// Effect name, e.g. `yield` or `break`.
    pub effect_name: String,
    /// Patterns of effect payload, e.g. `k1(p1, p2)`. Stored as identifiers.
    pub params: Vec<Ident>,
    /// Optional binding for the resumed value: `k(...) -> resume(v) => body`.
    pub resume_param: Option<Ident>,
    /// Handler body. Runs when effect `effect_name` is raised inside `expr`.
    pub body: Box<AstExpr>,
    pub span: Span,
}

/// `select! { msg = ch => { body }, ... default => { body } }` — channel multiplexing arm.
#[derive(Debug, Clone)]
pub struct SelectArm {
    /// Channel expression to receive from.
    pub channel: AstExpr,
    /// Binding name for the received value.
    pub binding: String,
    /// Body to execute if this channel fires.
    pub body: AstBlock,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AstUnaryOp {
    /// Arithmetic negation: `-x`
    Neg,
    /// Boolean NOT: `!b`
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AstBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    CmpEq,
    CmpLt,
    CmpLe,
    CmpGt,
    CmpGe,
    CmpNe,
    /// Logical AND (`&&`), short-circuit.
    And,
    /// Logical OR (`||`), short-circuit.
    Or,
}

/// An expression in the AST.
#[derive(Debug, Clone)]
pub enum AstExpr {
    Ident(Ident),
    IntLit {
        value: i64,
        span: Span,
    },
    FloatLit {
        value: f64,
        span: Span,
    },
    BoolLit {
        value: bool,
        span: Span,
    },
    StringLit {
        value: String,
        span: Span,
    },
    BinOp {
        op: AstBinOp,
        lhs: Box<AstExpr>,
        rhs: Box<AstExpr>,
        span: Span,
    },
    /// `<callee>(<args...>)`
    Call {
        callee: Ident,
        args: Vec<AstExpr>,
        named_args: Vec<(String, AstExpr)>,
        span: Span,
    },
    /// `-x` or `!b` (prefix unary operators)
    UnaryOp {
        op: AstUnaryOp,
        expr: Box<AstExpr>,
        span: Span,
    },
    /// `if <cond> { <then> } [else { <else> }]`
    If {
        cond: Box<AstExpr>,
        then_block: AstBlock,
        else_block: Option<AstBlock>,
        span: Span,
    },
    /// A block expression: `{ stmts... tail }`
    Block(AstBlock),
    /// `expr[idx0, idx1, ...]` tensor index
    Index {
        base: Box<AstExpr>,
        indices: Vec<AstExpr>,
        span: Span,
    },
    /// `expr as Type` cast
    Cast {
        expr: Box<AstExpr>,
        ty: AstType,
        span: Span,
    },
    /// `Name { field: expr, ... }` struct literal
    StructLit {
        name: String,
        fields: Vec<(String, AstExpr)>,
        spread: Option<Box<AstExpr>>,
        span: Span,
    },
    /// `expr.field` field access
    FieldAccess {
        base: Box<AstExpr>,
        field: String,
        span: Span,
    },
    /// `when scrutinee { EnumName.Variant => expr, ... }` pattern match on enum
    When {
        scrutinee: Box<AstExpr>,
        arms: Vec<AstWhenArm>,
        span: Span,
    },
    /// `(expr, expr, ...)` tuple literal
    Tuple {
        elements: Vec<AstExpr>,
        span: Span,
    },
    /// `expr.0` tuple index access
    TupleIndex {
        base: Box<AstExpr>,
        index: usize,
        span: Span,
    },
    /// `[expr, expr, ...]` array literal
    ArrayLit {
        elems: Vec<AstExpr>,
        span: Span,
    },
    /// `|param: type, ...| body_expr` lambda / closure literal
    Lambda {
        params: Vec<AstParam>,
        body: Box<AstExpr>,
        span: Span,
    },
    /// `await expr` -- awaits an async expression (lowered as regular call).
    Await {
        expr: Box<AstExpr>,
        span: Span,
    },
    /// `expr?` early-return on error propagation.
    Try {
        expr: Box<AstExpr>,
        span: Span,
    },
    /// `expr ?? default` null coalescing: unwrap option/result or use default.
    NullCoal {
        expr: Box<AstExpr>,
        default: Box<AstExpr>,
        span: Span,
    },
    /// `base.method(args...)` method call on a struct.
    MethodCall {
        base: Box<AstExpr>,
        method: String,
        args: Vec<AstExpr>,
        span: Span,
    },
    /// `with <effect1, effect2, ...> { <body> }` — restrict the body's effects.
    Mask {
        effects: Vec<String>,
        body: AstBlock,
        span: Span,
    },
    /// `handle <expr> with { ... }` — install algebraic-effect handlers.
    Handle {
        expr: Box<AstExpr>,
        arms: Vec<AstHandlerArm>,
        return_ty: Box<AstType>,
        span: Span,
    },
    /// `{ key1: val1, key2: val2 }` map/dict literal.
    MapLiteral {
        entries: Vec<(AstExpr, AstExpr)>,
        span: Span,
    },
    /// `&expr` — take an immutable reference to an expression.
    Ref {
        expr: Box<AstExpr>,
        span: Span,
    },
    /// `&mut expr` — take a mutable reference to an expression.
    RefMut {
        expr: Box<AstExpr>,
        span: Span,
    },
    /// `*expr` — dereference a reference.
    Deref {
        expr: Box<AstExpr>,
        span: Span,
    },
    /// `try { body } catch e { handler }` — try/catch on result types.
    /// Inside body, `?` on result types jumps to catch on Err instead of early-returning.
    TryCatch {
        body: Box<AstExpr>,
        catch_param: String,
        catch_body: Box<AstExpr>,
        span: Span,
    },
    /// `raise effect_name(args)` — explicitly raise an algebraic effect.
    Raise {
        effect_name: String,
        args: Vec<AstExpr>,
        span: Span,
    },
    /// `move expr` — explicitly transfer ownership (marks value as consumed).
    Move {
        expr: Box<AstExpr>,
        span: Span,
    },
    /// `unsafe { body }` — marks a block as unsafe (currently a regular block, syntactic sugar for future use).
    Unsafe {
        body: Box<AstExpr>,
        span: Span,
    },
    /// `..expr` — splat expression: unpack array elements as individual arguments in a function call.
    Splat {
        expr: Box<AstExpr>,
        span: Span,
    },
    /// `name!(args...)` — macro invocation
    MacroCall {
        name: Ident,
        args: Vec<AstExpr>,
        span: Span,
    },
}

impl AstExpr {
    pub fn span(&self) -> Span {
        match self {
            AstExpr::Ident(i) => i.span,
            AstExpr::IntLit { span, .. } => *span,
            AstExpr::FloatLit { span, .. } => *span,
            AstExpr::BoolLit { span, .. } => *span,
            AstExpr::StringLit { span, .. } => *span,
            AstExpr::BinOp { span, .. } => *span,
            AstExpr::UnaryOp { span, .. } => *span,
            AstExpr::Call { span, .. } => *span,
            AstExpr::If { span, .. } => *span,
            AstExpr::Block(b) => b.span,
            AstExpr::Index { span, .. } => *span,
            AstExpr::Cast { span, .. } => *span,
            AstExpr::StructLit { span, .. } => *span,
            AstExpr::FieldAccess { span, .. } => *span,
            AstExpr::When { span, .. } => *span,
            AstExpr::Tuple { span, .. } => *span,
            AstExpr::TupleIndex { span, .. } => *span,
            AstExpr::ArrayLit { span, .. } => *span,
            AstExpr::Lambda { span, .. } => *span,
            AstExpr::Await { span, .. } => *span,
            AstExpr::Try { span, .. } => *span,
            AstExpr::NullCoal { span, .. } => *span,
            AstExpr::MethodCall { span, .. } => *span,
            AstExpr::Mask { span, .. } => *span,
            AstExpr::Handle { span, .. } => *span,
            AstExpr::MapLiteral { span, .. } => *span,
            AstExpr::Ref { span, .. } => *span,
            AstExpr::RefMut { span, .. } => *span,
            AstExpr::Deref { span, .. } => *span,
            AstExpr::TryCatch { span, .. } => *span,
            AstExpr::Raise { span, .. } => *span,
            AstExpr::Move { span, .. } => *span,
            AstExpr::Unsafe { span, .. } => *span,
            AstExpr::Splat { span, .. } => *span,
            AstExpr::MacroCall { span, .. } => *span,
        }
    }
}

/// A struct field definition: `name: type` optionally with `= default_expr`.
#[derive(Debug, Clone)]
pub struct AstFieldDef {
    pub name: Ident,
    pub ty: AstType,
    pub default: Option<AstExpr>,
}

/// A struct definition: `record Name { field: type, ... }`.
#[derive(Debug, Clone)]
pub struct AstStructDef {
    pub name: Ident,
    pub type_params: Vec<AstGenericParam>,
    pub fields: Vec<AstFieldDef>,
    pub span: Span,
    /// Whether this struct is publicly exported (`pub record`).
    pub is_pub: bool,
    /// Doc comment (`/// ...`) preceding this item, if any.
    pub doc_comment: Option<String>,
}

/// A single enum variant, optionally carrying typed fields.
#[derive(Debug, Clone)]
pub struct AstEnumVariant {
    pub name: Ident,
    /// Payload field types, empty for unit (tag-only) variants.
    pub fields: Vec<AstType>,
    pub span: Span,
}

/// An enum definition: `choice Name { Variant1, Variant2(T), ... }`.
#[derive(Debug, Clone)]
pub struct AstEnumDef {
    pub name: Ident,
    /// Ordered list of variants (may carry payload types).
    pub variants: Vec<AstEnumVariant>,
    pub span: Span,
    /// Whether this enum is publicly exported (`pub choice`).
    pub is_pub: bool,
    /// Doc comment (`/// ...`) preceding this item, if any.
    pub doc_comment: Option<String>,
}

/// The pattern in a `when` arm.
#[derive(Debug, Clone)]
pub enum AstWhenPattern {
    /// `EnumName.Variant` or `EnumName.Variant(a, b, ...)` — enum variant pattern.
    EnumVariant {
        enum_name: String,
        variant_name: String,
        bindings: Vec<String>,
    },
    /// `some(binding)` — option Some pattern with an optional bound name.
    OptionSome { binding: Option<String> },
    /// `none` — option None pattern.
    OptionNone,
    /// `ok(binding)` — result Ok pattern.
    ResultOk { binding: Option<String> },
    /// `err(binding)` — result Err pattern.
    ResultErr { binding: Option<String> },
    /// `_` — wildcard pattern, matches anything.
    Wildcard,
    /// Integer literal pattern, e.g. `0` or `1`.
    IntLit(i64),
    /// Float literal pattern, e.g. `1.0` or `3.14`.
    FloatLit(f64),
    /// Bool literal pattern, e.g. `true` or `false`.
    BoolLit(bool),
    /// String literal pattern, e.g. `"hello"`.
    StringLit(String),
    /// Tuple pattern, e.g. `(a, b)` or `(0, x)`.
    /// Each sub-pattern is a `AstWhenPattern`; variable names bind to the elements.
    Tuple(Vec<AstWhenPattern>),
    /// Inclusive integer range pattern, e.g. `1..=5`.
    Range { lo: i64, hi: i64 },
    /// Or-pattern: `pat1 | pat2 | ...` — matches if any sub-pattern matches.
    Or(Vec<AstWhenPattern>),
    /// Slice pattern: `[a, b, ..rest]` — matches list/array prefix with optional rest binding.
    Slice {
        prefix: Vec<AstWhenPattern>,
        rest: Option<String>, // None = exact match, Some(name) = bind rest
    },
    /// Binding pattern: `name @ sub_pattern` — matches `sub_pattern` and binds the whole matched value to `name`.
    Binding {
        name: String,
        pattern: Box<AstWhenPattern>,
        span: Span,
    },
    /// Struct pattern: `StructName { field1: pat1, field2: pat2 }` — matches a record by field values.
    Struct {
        struct_name: String,
        fields: Vec<(String, AstWhenPattern)>,
        span: Span,
    },
}

/// A single arm in a `when` expression.
#[derive(Debug, Clone)]
pub struct AstWhenArm {
    pub pattern: AstWhenPattern,
    /// Optional guard expression: `pattern if expr =>`.
    pub guard: Option<Box<AstExpr>>,
    pub body: Box<AstExpr>,
    pub span: Span,
    // Legacy fields kept for backward compatibility during transition.
    pub enum_name: String,
    pub variant_name: String,
}

// ---------------------------------------------------------------------------
// Model DSL AST nodes
// ---------------------------------------------------------------------------

/// A single hyperparameter in a layer: `key = value`.
#[derive(Debug, Clone)]
pub struct AstLayerParam {
    pub key: Ident,
    pub value: AstExpr,
    pub span: Span,
}

/// A layer declaration inside a model: `layer <name> <Op>([refs,] [key=val,]*)`.
///
/// `input_refs` holds bare ident arguments (explicit data-flow inputs).
/// `params` holds `key = value` keyword hyperparameters.
/// Both may appear in the same argument list.
#[derive(Debug, Clone)]
pub struct AstLayer {
    pub name: Ident,
    pub op: Ident,
    pub input_refs: Vec<Ident>,
    pub params: Vec<AstLayerParam>,
    pub span: Span,
}

/// A model input declaration: `input <name>: <type>`.
#[derive(Debug, Clone)]
pub struct AstModelInput {
    pub name: Ident,
    pub ty: AstType,
    pub span: Span,
}

/// A model output declaration: `output <name>`.
/// `name` must refer to a previously declared layer or input.
#[derive(Debug, Clone)]
pub struct AstModelOutput {
    pub name: Ident,
    pub span: Span,
}

/// A model definition: `model <Name> { inputs... layers... outputs... }`.
#[derive(Debug, Clone)]
pub struct AstModel {
    pub name: Ident,
    pub inputs: Vec<AstModelInput>,
    pub layers: Vec<AstLayer>,
    pub outputs: Vec<AstModelOutput>,
    pub span: Span,
    /// Doc comment (`/// ...`) preceding this item, if any.
    pub doc_comment: Option<String>,
}

/// A global constant declaration: `const NAME: type = value` or `const NAME = value`.
#[derive(Debug, Clone)]
pub struct AstConst {
    pub name: Ident,
    /// Optional explicit type annotation.
    pub ty: Option<AstType>,
    pub value: AstExpr,
    pub span: Span,
    /// Whether this const is publicly exported (`pub const`).
    pub is_pub: bool,
    /// Doc comment (`/// ...`) preceding this item, if any.
    pub doc_comment: Option<String>,
}

/// A type alias declaration: `type Name = Type`.
#[derive(Debug, Clone)]
pub struct AstTypeAlias {
    pub name: String,
    pub ty: AstType,
    pub span: Span,
    /// Whether this type alias is publicly exported (`pub type`).
    pub is_pub: bool,
    /// Doc comment (`/// ...`) preceding this item, if any.
    pub doc_comment: Option<String>,
}

// ---------------------------------------------------------------------------
// Module bring / import system
// ---------------------------------------------------------------------------

/// The path of a `bring` declaration.
#[derive(Debug, Clone)]
pub enum BringPath {
    /// `bring "path/to/file.iris"` — resolved from disk (or virtual source map).
    File(String),
    /// `bring std.name` — resolved from the embedded stdlib registry.
    Stdlib(String),
}

/// A `bring` declaration at module level.
#[derive(Debug, Clone)]
pub struct AstBring {
    pub path: BringPath,
    /// Optional selective items: `bring std.math.{gcd, lcm}` → items = Some(["gcd", "lcm"])
    pub items: Option<Vec<String>>,
    pub span: Span,
    /// Whether this bring is `pub bring` (re-exported to parent module).
    pub is_pub: bool,
}

/// A method signature inside a trait definition, with optional default body.
#[derive(Debug, Clone)]
pub struct AstTraitMethod {
    pub name: Ident,
    pub params: Vec<AstParam>,
    pub return_ty: AstType,
    /// Effect bound declared on the method, e.g. `def read(self) -> i64 effect io`.
    /// A trait could not state the effects its implementations may perform until
    /// this existed -- the clause was a parse error. See known-issues #10.
    pub effects: Vec<String>,
    pub body: Option<AstBlock>,
    pub span: Span,
}

/// An associated type declaration inside a trait: `type Name`.
#[derive(Debug, Clone)]
pub struct AstAssocTypeDecl {
    pub name: Ident,
    pub span: Span,
}

/// A trait definition: `trait Name { type AssocType; def method(params) -> type }`.
#[derive(Debug, Clone)]
pub struct AstTraitDef {
    pub name: Ident,
    /// Generic parameters on the trait itself: `trait Container[T]`, or
    /// `trait Mappable[F[_]]` for a parameter that is a type *constructor*.
    ///
    /// Traits accepted no generic parameters at all before this, so
    /// `trait Container[T]` was a parse error exactly like
    /// `trait Mappable[F[_]]` -- there were no type classes of any kind, over a
    /// constructor or otherwise. See known-issues #37.
    pub type_params: Vec<AstGenericParam>,
    pub assoc_types: Vec<AstAssocTypeDecl>,
    pub methods: Vec<AstTraitMethod>,
    pub span: Span,
    /// Doc comment (`/// ...`) preceding this item, if any.
    pub doc_comment: Option<String>,
}

/// An impl block: `impl TraitName for TypeName { ... }` or `impl[T where T: Trait] TraitName for T { ... }`.
#[derive(Debug, Clone)]
pub struct AstImplDef {
    /// The trait being implemented.
    pub trait_name: String,
    /// Arguments supplied to the trait's generic parameters:
    /// `impl Container[i64] for IntBox` binds the trait's `T` to `i64`.
    pub trait_args: Vec<AstType>,
    /// The type being implemented for (e.g. "i64", "Point", or "T" for blanket impls).
    pub type_name: String,
    /// Generic parameters (for blanket impls): `impl[T where T: Show] ...`.
    pub generic_params: Vec<AstGenericParam>,
    /// Associated type bindings: `type AssocType = ConcreteType`.
    pub assoc_type_bindings: Vec<(String, AstType)>,
    /// Full method bodies.
    pub methods: Vec<AstFunction>,
    pub span: Span,
    /// Doc comment (`/// ...`) preceding this item, if any.
    pub doc_comment: Option<String>,
}

/// An extern function declaration: `extern "C" def name(params) -> ret_ty`.
/// Declares a C-linkage function callable from IRIS but defined outside.
/// The `abi` field stores the calling convention (e.g. `"C"`), and `attrs`
/// may contain `@link(name = "lib")` to specify which library to link against.
#[derive(Debug, Clone)]
pub struct AstExternFn {
    pub name: Ident,
    pub params: Vec<AstParam>,
    pub ret_ty: AstType,
    pub abi: Option<String>,    // e.g. Some("C") or None for default
    pub link_lib: Option<String>, // e.g. Some("m") for -lm
    pub span: Span,
    /// Doc comment (`/// ...`) preceding this item, if any.
    pub doc_comment: Option<String>,
}

/// An operation signature within an algebraic effect: `def name(params) -> ret_ty`.
#[derive(Debug, Clone)]
pub struct AstEffectOperation {
    pub name: Ident,
    pub params: Vec<AstParam>,
    pub ret_ty: AstType,
    pub span: Span,
}

/// An algebraic effect declaration: `effect Name { def op1(...) -> ty; ... }`.
#[derive(Debug, Clone)]
pub struct AstEffectDef {
    pub name: Ident,
    pub operations: Vec<AstEffectOperation>,
    pub span: Span,
    /// Doc comment (`/// ...`) preceding this item, if any.
    pub doc_comment: Option<String>,
}

/// The top-level AST for an IRIS source file.
/// A file may contain any mix of `def`, `record`, `choice`, `model`, `const`, `type`, `trait`, `impl`, `effect`, `bring`, and `extern def` definitions.
#[derive(Debug, Clone)]
pub struct AstModule {
    pub enums: Vec<AstEnumDef>,
    pub structs: Vec<AstStructDef>,
    pub functions: Vec<AstFunction>,
    pub models: Vec<AstModel>,
    pub consts: Vec<AstConst>,
    pub type_aliases: Vec<AstTypeAlias>,
    pub traits: Vec<AstTraitDef>,
    pub impls: Vec<AstImplDef>,
    pub effects: Vec<AstEffectDef>,
    /// Bring declarations: `bring "file.iris"`, `bring std.name`, or `bring module_name`.
    pub brings: Vec<AstBring>,
    /// Extern function declarations: `extern def name(params) -> type`.
    pub extern_fns: Vec<AstExternFn>,
    /// Inline module definitions: `mod name { ... }`.
    pub modules: Vec<AstModuleDef>,
    /// Macro definitions: `defmacro name(params) => body`.
    pub macros: Vec<AstMacroDef>,
}

/// An inline module definition: `mod name { ... }`.
/// Contains the same item types as a top-level module, nested inside the current file.
#[derive(Debug, Clone)]
pub struct AstModuleDef {
    pub name: Ident,
    pub enums: Vec<AstEnumDef>,
    pub structs: Vec<AstStructDef>,
    pub functions: Vec<AstFunction>,
    pub models: Vec<AstModel>,
    pub consts: Vec<AstConst>,
    pub type_aliases: Vec<AstTypeAlias>,
    pub traits: Vec<AstTraitDef>,
    pub impls: Vec<AstImplDef>,
    pub effects: Vec<AstEffectDef>,
    pub extern_fns: Vec<AstExternFn>,
    pub modules: Vec<AstModuleDef>,
    pub macros: Vec<AstMacroDef>,
    pub span: Span,
}
