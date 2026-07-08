pub mod build;
pub mod cuda;
pub mod graph_printer;
pub mod ir_serial;
pub mod jit;
pub mod llvm_ir;
pub mod llvm_native;
pub mod llvm_stub;
pub mod onnx;
pub mod onnx_binary;
pub mod pgo;
pub mod printer;
pub mod simd;
pub mod tensorrt;

pub use build::{
    build_binary, build_binary_for_eval, build_binary_for_eval_with_target,
    build_binary_with_target, emit_binary_ir, execute_binary_for_eval,
    execute_binary_for_eval_with_target, runtime_c_source, runtime_h_source,
};
pub use cuda::{emit_cuda, emit_cuda_ptx};
pub use graph_printer::emit_graph_text;
pub use ir_serial::{deserialize_module, serialize_module};
pub use jit::emit_jit;
pub use llvm_ir::{
    emit_llvm_ir, emit_llvm_ir_with_target, native_target_triple, target_data_layout,
    target_preset_to_triple,
};
pub use llvm_stub::emit_llvm_stub;
pub use onnx::emit_onnx_text;
pub use onnx_binary::emit_onnx_binary;
pub use pgo::{emit_pgo_instrument, emit_pgo_optimize};
pub use printer::emit_ir_text;
pub use simd::emit_simd;
pub use tensorrt::emit_tensorrt;

use crate::error::CodegenError;
