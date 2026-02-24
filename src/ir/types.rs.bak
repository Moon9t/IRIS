/// Scalar element types for tensors and scalar values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    Bool,
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::F32 => f.write_str("f32"),
            DType::F64 => f.write_str("f64"),
            DType::I32 => f.write_str("i32"),
            DType::I64 => f.write_str("i64"),
            DType::Bool => f.write_str("bool"),
        }
    }
}

/// A single dimension of a tensor shape.
/// Symbolic dims allow shapes like [M, K] to be tracked at compile time
/// without requiring concrete values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dim {
    /// Statically known at compile time (e.g., 3 in [3, 4]).
    Literal(u64),
    /// Symbolic name, resolved during type inference (e.g., M, K, N).
    Symbolic(String),
}

impl std::fmt::Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dim::Literal(n) => write!(f, "{}", n),
            Dim::Symbolic(s) => f.write_str(s),
        }
    }
}

/// An ordered list of dimensions forming a tensor shape.
/// Invariant: rank-0 tensors use an empty Shape, not a missing Shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(pub Vec<Dim>);

impl Shape {
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn is_fully_concrete(&self) -> bool {
        self.0.iter().all(|d| matches!(d, Dim::Literal(_)))
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("[")?;
        for (i, dim) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{}", dim)?;
        }
        f.write_str("]")
    }
}

/// The type of an IR value.
///
/// Invariant: `Fn` types appear only in function signatures, not in value
/// positions within a function body (IRIS v0 has no first-class functions).
/// `Infer` is a placeholder valid only before `TypeInferPass` completes;
/// `ValidatePass` rejects any module containing `Infer` values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IrType {
    /// A primitive scalar value.
    Scalar(DType),
    /// A tensor with element type and shape. Shape may contain symbolic dims.
    Tensor { dtype: DType, shape: Shape },
    /// A function type, used in call instruction signatures only.
    Fn {
        params: Vec<IrType>,
        ret: Box<IrType>,
    },
    /// An unresolved type — valid only before type inference completes.
    Infer,
    /// A named struct type with ordered fields.
    Struct {
        name: String,
        fields: Vec<(String, IrType)>,
    },
    /// A named enum type. Values are integer variant tags (0-indexed).
    Enum { name: String, variants: Vec<String> },
    /// An ordered tuple of heterogeneous types.
    Tuple(Vec<IrType>),
}

impl std::fmt::Display for IrType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IrType::Scalar(d) => write!(f, "{}", d),
            IrType::Tensor { dtype, shape } => write!(f, "tensor<{}, {}>", dtype, shape),
            IrType::Fn { params, ret } => {
                f.write_str("fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", ret)
            }
            IrType::Infer => f.write_str("_"),
            IrType::Struct { name, .. } => write!(f, "%{}", name),
            IrType::Enum { name, .. } => write!(f, "enum.{}", name),
            IrType::Tuple(elems) => {
                f.write_str("(")?;
                for (i, t) in elems.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", t)?;
                }
                f.write_str(")")
            }
        }
    }
}
