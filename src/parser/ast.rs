use crate::parser::lexer::Span;

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
    /// A fixed-length array type, e.g. `[i64; 5]`.
    Array { elem: Box<AstType>, len: usize, span: Span },
}

impl AstType {
    pub fn span(&self) -> Span {
        match self {
            AstType::Scalar(_, s) => *s,
            AstType::Tensor { span, .. } => *span,
            AstType::Named(_, s) => *s,
            AstType::Tuple(_, s) => *s,
            AstType::Array { span, .. } => *span,
        }
    }
}

/// A function parameter.
#[derive(Debug, Clone)]
pub struct AstParam {
    pub name: Ident,
    pub ty: AstType,
}

/// A function definition.
#[derive(Debug, Clone)]
pub struct AstFunction {
    pub name: Ident,
    pub params: Vec<AstParam>,
    pub return_ty: AstType,
    pub body: AstBlock,
    pub span: Span,
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
        span: Span,
    },
    /// An expression used for its side effects (followed by `;`).
    Expr(Box<AstExpr>),
    While {
        cond: Box<AstExpr>,
        body: AstBlock,
        span: Span,
    },
    Loop {
        body: AstBlock,
        span: Span,
    },
    Break {
        span: Span,
    },
    Continue {
        span: Span,
    },
    /// `for <var> in <start>..<end> { <body> }` range loop (sugar over while).
    ForRange {
        var: Ident,
        start: Box<AstExpr>,
        end: Box<AstExpr>,
        body: AstBlock,
        span: Span,
    },
    /// `lvalue = expr` tensor store assignment.
    Assign {
        target: Box<AstExpr>,
        value: Box<AstExpr>,
        span: Span,
    },
    /// `val (a, b, ...) = expr` destructuring tuple let.
    LetTuple {
        names: Vec<Ident>,
        init: Box<AstExpr>,
        span: Span,
    },
    /// `return [expr]` early return from function.
    Return {
        value: Option<Box<AstExpr>>,
        span: Span,
    },
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
        }
    }
}

/// A struct field definition: `name: type`.
#[derive(Debug, Clone)]
pub struct AstFieldDef {
    pub name: Ident,
    pub ty: AstType,
}

/// A struct definition: `record Name { field: type, ... }`.
#[derive(Debug, Clone)]
pub struct AstStructDef {
    pub name: Ident,
    pub fields: Vec<AstFieldDef>,
    pub span: Span,
}

/// An enum definition: `choice Name { Variant1, Variant2, ... }`.
#[derive(Debug, Clone)]
pub struct AstEnumDef {
    pub name: Ident,
    /// Ordered list of variant names.
    pub variants: Vec<Ident>,
    pub span: Span,
}

/// A single arm in a `when` expression: `EnumName.Variant => expr`.
#[derive(Debug, Clone)]
pub struct AstWhenArm {
    pub enum_name: String,
    pub variant_name: String,
    pub body: Box<AstExpr>,
    pub span: Span,
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
}

/// The top-level AST for an IRIS source file.
/// A file may contain any mix of `def`, `record`, `choice`, and `model` definitions.
#[derive(Debug, Clone)]
pub struct AstModule {
    pub enums: Vec<AstEnumDef>,
    pub structs: Vec<AstStructDef>,
    pub functions: Vec<AstFunction>,
    pub models: Vec<AstModel>,
}
