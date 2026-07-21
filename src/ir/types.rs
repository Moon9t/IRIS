/// Scalar element types for tensors and scalar values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
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

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::F32 => f.write_str("f32"),
            DType::F64 => f.write_str("f64"),
            DType::I32 => f.write_str("i32"),
            DType::I64 => f.write_str("i64"),
            DType::Bool => f.write_str("bool"),
            DType::U8 => f.write_str("u8"),
            DType::I8 => f.write_str("i8"),
            DType::U32 => f.write_str("u32"),
            DType::U64 => f.write_str("u64"),
            DType::USize => f.write_str("usize"),
        }
    }
}

impl DType {
    /// Returns true if this is an integer type (signed or unsigned).
    pub fn is_integer(&self) -> bool {
        matches!(self, DType::I8 | DType::U8 | DType::I32 | DType::U32 | DType::I64 | DType::U64 | DType::USize)
    }

    /// Returns true if this is a signed integer type.
    pub fn is_signed(&self) -> bool {
        matches!(self, DType::I8 | DType::I32 | DType::I64)
    }

    /// Returns true if this is a floating-point type.
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F32 | DType::F64)
    }

    /// Returns the bit width of this type.
    pub fn bit_width(&self) -> u32 {
        match self {
            DType::Bool | DType::I8 | DType::U8 => 8,
            DType::I32 | DType::U32 | DType::F32 => 32,
            DType::I64 | DType::U64 | DType::USize | DType::F64 => 64,
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
    /// A UTF-8 string value.
    Str,
    /// A fixed-length array of a single element type.
    Array { elem: Box<IrType>, len: usize },
    /// Option type: `option<T>` — either Some(T) or None.
    Option(Box<IrType>),
    /// Result type: `result<T, E>` — either Ok(T) or Err(E).
    ResultType(Box<IrType>, Box<IrType>),
    /// Channel type: `chan<T>` — a FIFO communication channel.
    Chan(Box<IrType>),
    /// Atomic type: `atomic<T>` — an atomically-accessible value.
    Atomic(Box<IrType>),
    /// Mutex type: `mutex<T>` — a mutex-protected value.
    Mutex(Box<IrType>),
    /// Grad type: `grad<T>` — a dual number for forward-mode automatic differentiation.
    Grad(Box<IrType>),
    /// Sparse type: `sparse<T>` — a sparse representation of a tensor or array.
    Sparse(Box<IrType>),
    /// Dynamic list type: `list<T>` — a growable sequence of values.
    List(Box<IrType>),
    /// Hash map type: `map<K, V>` — a key-value store.
    Map(Box<IrType>, Box<IrType>),
    /// TaskGroup type for structured concurrency.
    TaskGroup,
    /// WeakRef type: `weak_ref<T>` — non-owning reference.
    WeakRef(Box<IrType>),
    /// `dyn Trait` — fat pointer { data_ptr, vtable_ptr }.
    /// At the IR level, we store the trait name and the method signatures
    /// (param types + return type) so the codegen can lay out the vtable
    /// and DynCall sites can produce well-typed indirect calls. Method
    /// bodies are looked up at runtime via the impl dispatch table.
    TraitObject {
        name: String,
        /// Method signatures in vtable slot order (no `self`).
        methods: Vec<TraitMethodSig>,
    },
}

/// Signature of a single method on a trait object's vtable slot.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TraitMethodSig {
    pub name: String,
    pub params: Vec<IrType>,
    pub ret: Box<IrType>,
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
            IrType::Str => f.write_str("str"),
            IrType::Array { elem, len } => write!(f, "[{}; {}]", elem, len),
            IrType::Option(inner) => write!(f, "option<{}>", inner),
            IrType::ResultType(ok, err) => write!(f, "result<{},{}>", ok, err),
            IrType::Chan(elem) => write!(f, "chan<{}>", elem),
            IrType::Atomic(inner) => write!(f, "atomic<{}>", inner),
            IrType::Mutex(inner) => write!(f, "mutex<{}>", inner),
            IrType::Grad(inner) => write!(f, "grad<{}>", inner),
            IrType::Sparse(inner) => write!(f, "sparse<{}>", inner),
            IrType::List(elem) => write!(f, "list<{}>", elem),
            IrType::Map(k, v) => write!(f, "map<{}, {}>", k, v),
            IrType::TaskGroup => f.write_str("task_group"),
            IrType::WeakRef(inner) => write!(f, "weak_ref<{}>", inner),
            IrType::TraitObject { name, methods } => {
                write!(f, "dyn {}", name)?;
                if !methods.is_empty() {
                    f.write_str(" {")?;
                    for (i, m) in methods.iter().enumerate() {
                        if i > 0 {
                            f.write_str("; ")?;
                        }
                        write!(f, "{}(", m.name)?;
                        for (j, p) in m.params.iter().enumerate() {
                            if j > 0 {
                                f.write_str(", ")?;
                            }
                            write!(f, "{}", p)?;
                        }
                        write!(f, ") -> {}", m.ret)?;
                    }
                    f.write_str("}")?;
                }
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- DType Display ----------------------------------------------------

    #[test]
    fn dtype_display() {
        assert_eq!(format!("{}", DType::F32), "f32");
        assert_eq!(format!("{}", DType::F64), "f64");
        assert_eq!(format!("{}", DType::I32), "i32");
        assert_eq!(format!("{}", DType::I64), "i64");
        assert_eq!(format!("{}", DType::Bool), "bool");
        assert_eq!(format!("{}", DType::U8), "u8");
        assert_eq!(format!("{}", DType::I8), "i8");
        assert_eq!(format!("{}", DType::U32), "u32");
        assert_eq!(format!("{}", DType::U64), "u64");
        assert_eq!(format!("{}", DType::USize), "usize");
    }

    // -- Dim Display ------------------------------------------------------

    #[test]
    fn dim_display() {
        assert_eq!(format!("{}", Dim::Literal(3)), "3");
        assert_eq!(format!("{}", Dim::Symbolic("M".into())), "M");
    }

    #[test]
    fn dim_equality() {
        assert_eq!(Dim::Literal(5), Dim::Literal(5));
        assert_ne!(Dim::Literal(5), Dim::Literal(6));
        assert_eq!(Dim::Symbolic("N".into()), Dim::Symbolic("N".into()));
        assert_ne!(Dim::Symbolic("M".into()), Dim::Symbolic("N".into()));
    }

    // -- Shape ------------------------------------------------------------

    #[test]
    fn shape_rank() {
        let s0 = Shape(vec![]);
        assert_eq!(s0.rank(), 0);

        let s2 = Shape(vec![Dim::Literal(3), Dim::Literal(4)]);
        assert_eq!(s2.rank(), 2);
    }

    #[test]
    fn shape_is_fully_concrete() {
        let concrete = Shape(vec![Dim::Literal(2), Dim::Literal(3)]);
        assert!(concrete.is_fully_concrete());

        let symbolic = Shape(vec![Dim::Literal(2), Dim::Symbolic("N".into())]);
        assert!(!symbolic.is_fully_concrete());
    }

    #[test]
    fn shape_display() {
        assert_eq!(format!("{}", Shape(vec![])), "[]");
        assert_eq!(
            format!(
                "{}",
                Shape(vec![Dim::Literal(3), Dim::Symbolic("K".into())])
            ),
            "[3, K]"
        );
    }

    // -- IrType Display ---------------------------------------------------

    #[test]
    fn irtype_scalar_display() {
        assert_eq!(format!("{}", IrType::Scalar(DType::I64)), "i64");
        assert_eq!(format!("{}", IrType::Scalar(DType::F32)), "f32");
    }

    #[test]
    fn irtype_tensor_display() {
        let t = IrType::Tensor {
            dtype: DType::F32,
            shape: Shape(vec![Dim::Literal(3), Dim::Literal(4)]),
        };
        assert_eq!(format!("{}", t), "tensor<f32, [3, 4]>");
    }

    #[test]
    fn irtype_fn_display() {
        let t = IrType::Fn {
            params: vec![IrType::Scalar(DType::I64), IrType::Scalar(DType::Bool)],
            ret: Box::new(IrType::Str),
        };
        assert_eq!(format!("{}", t), "fn(i64, bool) -> str");
    }

    #[test]
    fn irtype_compound_display() {
        assert_eq!(format!("{}", IrType::Infer), "_");
        assert_eq!(format!("{}", IrType::Str), "str");
        assert_eq!(
            format!(
                "{}",
                IrType::Array {
                    elem: Box::new(IrType::Scalar(DType::I64)),
                    len: 5
                }
            ),
            "[i64; 5]"
        );
        assert_eq!(
            format!("{}", IrType::Option(Box::new(IrType::Scalar(DType::I64)))),
            "option<i64>"
        );
        assert_eq!(
            format!(
                "{}",
                IrType::ResultType(Box::new(IrType::Scalar(DType::I64)), Box::new(IrType::Str))
            ),
            "result<i64,str>"
        );
        assert_eq!(
            format!("{}", IrType::Chan(Box::new(IrType::Scalar(DType::I64)))),
            "chan<i64>"
        );
        assert_eq!(
            format!("{}", IrType::List(Box::new(IrType::Str))),
            "list<str>"
        );
        assert_eq!(
            format!(
                "{}",
                IrType::Map(Box::new(IrType::Str), Box::new(IrType::Scalar(DType::I64)))
            ),
            "map<str, i64>"
        );
        assert_eq!(
            format!("{}", IrType::Grad(Box::new(IrType::Scalar(DType::F64)))),
            "grad<f64>"
        );
        assert_eq!(
            format!("{}", IrType::Sparse(Box::new(IrType::Scalar(DType::F32)))),
            "sparse<f32>"
        );
        assert_eq!(
            format!("{}", IrType::Atomic(Box::new(IrType::Scalar(DType::I64)))),
            "atomic<i64>"
        );
        assert_eq!(
            format!("{}", IrType::Mutex(Box::new(IrType::Scalar(DType::I64)))),
            "mutex<i64>"
        );
    }

    #[test]
    fn irtype_tuple_display() {
        let t = IrType::Tuple(vec![
            IrType::Scalar(DType::I64),
            IrType::Scalar(DType::F64),
            IrType::Scalar(DType::Bool),
        ]);
        assert_eq!(format!("{}", t), "(i64, f64, bool)");
    }

    #[test]
    fn irtype_struct_display() {
        let t = IrType::Struct {
            name: "Point".into(),
            fields: vec![
                ("x".into(), IrType::Scalar(DType::F64)),
                ("y".into(), IrType::Scalar(DType::F64)),
            ],
        };
        assert_eq!(format!("{}", t), "%Point");
    }

    #[test]
    fn irtype_enum_display() {
        let t = IrType::Enum {
            name: "Color".into(),
            variants: vec!["Red".into(), "Green".into(), "Blue".into()],
        };
        assert_eq!(format!("{}", t), "enum.Color");
    }

    // -- Equality ---------------------------------------------------------

    #[test]
    fn irtype_equality() {
        assert_eq!(IrType::Scalar(DType::I64), IrType::Scalar(DType::I64));
        assert_ne!(IrType::Scalar(DType::I64), IrType::Scalar(DType::F64));
        assert_ne!(IrType::Infer, IrType::Str);
    }
}
