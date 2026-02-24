pub mod block;
pub mod function;
pub mod graph;
pub mod instr;
pub mod module;
pub mod types;
pub mod value;

pub use block::{BlockId, IrBlock};
pub use function::{FunctionId, IrFunction, Param};
pub use graph::{GraphIr, GraphNode, LayerParam, NodeId, ParamValue};
pub use instr::{BinOp, InstrId, IrInstr, TensorOp};
pub use module::{IrFunctionBuilder, IrModule};
pub use types::{DType, Dim, IrType, Shape};
pub use value::{BlockParam, ValueDef, ValueId};
