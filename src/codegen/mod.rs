pub mod graph_printer;
pub mod llvm_stub;
pub mod onnx;
pub mod onnx_binary;
pub mod printer;

pub use graph_printer::emit_graph_text;
pub use llvm_stub::emit_llvm_stub;
pub use onnx::emit_onnx_text;
pub use onnx_binary::emit_onnx_binary;
pub use printer::emit_ir_text;

use crate::error::CodegenError;
