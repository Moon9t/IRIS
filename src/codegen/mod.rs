pub mod build;
pub mod cuda;
pub mod graph_printer;
pub mod jit;
pub mod llvm_ir;
pub mod llvm_stub;
pub mod onnx;
pub mod onnx_binary;
pub mod pgo;
pub mod printer;
pub mod simd;

pub use build::{build_binary, emit_binary_ir, runtime_c_source, runtime_h_source};
pub use cuda::emit_cuda;
pub use graph_printer::emit_graph_text;
pub use jit::emit_jit;
pub use llvm_ir::emit_llvm_ir;
pub use llvm_stub::emit_llvm_stub;
pub use onnx::emit_onnx_text;
pub use onnx_binary::emit_onnx_binary;
pub use pgo::{emit_pgo_instrument, emit_pgo_optimize};
pub use printer::emit_ir_text;
pub use simd::emit_simd;

use crate::error::CodegenError;
