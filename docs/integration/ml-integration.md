# ML Backend Integration Design

Goal: integrate ONNX Runtime, LibTorch (PyTorch), and TensorFlow C API into IRIS runtime (no stubs) so native `iris` binaries and the interpreter can run ML models (inference and training where supported).

High-level approach

- Provide a small stable C ABI for each backend (shim layer) compiled into the runtime or as a separate shared library.
- Expose minimal operations: load model, run/infer, run with provided inputs/outputs, destroy session/model.
- Map IRIS runtime `IrisTensor` <-> backend tensor memory (zero-copy when possible).
- Add safe wrappers in `iris_runtime.c` and `iris_runtime.h`; add Rust bindings to call these C functions from `src/`.
- Expose IRIS-facing bridge wrappers as `iris_mlrt_onnx_*`, `iris_mlrt_pytorch_*`, and `iris_mlrt_tf_*`.
- Add `std.ml` functions to load/run models, clean tabular data, prepare `(list<f64>, list<i64>)` tensor pairs, and convert to/from `tensor<T,shape>`.
- CI/build: optional linking flags per backend; provide helpful fallbacks when a backend is not available.

Memory & ownership rules

- `IrisTensor*` is the IRIS runtime representation (opaque in the shim) with documented ownership: caller owns returned `IrisTensor*`; `iris_tensor_free` frees it.
- For zero-copy: when the backend supports providing an external buffer, `IrisTensor` will point into a memory region managed by the runtime; if a backend allocates a buffer, the shim will copy into `IrisTensor` on return unless pinned.
- Concurrency: sessions are not thread-safe unless explicitly documented (user must manage concurrency).

API sketches (C)

- IRIS runtime bridge:
  - `int64_t iris_mlrt_onnx_load(const char* model_path);`
  - `IrisVal* iris_mlrt_onnx_run(int64_t session, IrisVal* input);`
  - `int64_t iris_mlrt_pytorch_load(const char* model_path);`
  - `IrisVal* iris_mlrt_pytorch_run(int64_t model, IrisVal* input);`
  - `int64_t iris_mlrt_tf_load(const char* model_path);`
  - `IrisVal* iris_mlrt_tf_run(int64_t model, IrisVal* input);`

- ONNX Runtime shim (C):
  - `void* iris_onnx_session_create(const char* model_path);`
  - `int iris_onnx_session_run(void* session, IrisTensor** inputs, size_t n_inputs, IrisTensor*** outputs, size_t* n_outputs);`
  - `void iris_onnx_session_free(void* session);`

- LibTorch shim (C++ -> C):
  - `void* iris_pytorch_load(const char* model_path);`
  - `int iris_pytorch_run(void* model, IrisTensor** inputs, size_t n_inputs, IrisTensor*** outputs, size_t* n_outputs);`
  - `void iris_pytorch_free(void* model);`

- TensorFlow C API shim:
  - `void* iris_tf_load_saved_model(const char* path);`
  - `int iris_tf_run(void* model, IrisTensor** inputs, size_t n_inputs, IrisTensor*** outputs, size_t* n_outputs);`
  - `void iris_tf_free(void* model);`

Engineering plan

1. Add design doc (this file).
2. Add lightweight ONNX C shim with build-time optional linking (stub if ONNX not present).
3. Add wrapper functions to `src/runtime/iris_runtime.h/.c` forwarding to the shim.
4. Add Rust bindings in `src/ffi` or `src/runtime_bindings.rs` that call the C functions and convert to/from `IrisTensor`.
5. Add `stdlib` bindings (IRIS-level functions) to clean datasets and load/run models from IRIS source.
6. Add unit/integration tests using a small exported ONNX model (matrix multiply) and an exported PyTorch model if LibTorch enabled.
7. Add CI matrix entries for Linux (preferred), macOS, and Windows. Provide scripts in `installer/` for obtaining prebuilt binaries where possible.

Next steps

- Implement ONNX shim stubs in `src/runtime/onnx_shim.c` + header prototypes in `src/runtime/iris_runtime.h`.
- Keep `examples/ml_full_pipeline.iris` as the end-to-end showcase: ingest rows, clean values, train/retrain, predict, and hand off tensors to ONNX/PyTorch/TensorFlow hooks when native SDKs are configured.

Platform note: building libtorch and TF can be heavy; recommend starting with ONNX Runtime for cross-framework inference, then add LibTorch and TF shims.

Platform-specific build/link steps

- Windows: set `ONNXRUNTIME_DIR`, `LIBTORCH_DIR`, or `TENSORFLOW_DIR` to the unpacked SDK directories before invoking `cargo build` or `cargo test`. Generated native IRIS programs link external ONNX/TensorFlow SDKs only when `IRIS_NATIVE_ML_BACKENDS=1` is set; otherwise the generated runtime uses safe stubs. The runtime includes a small pthread compatibility layer so the embedded C runtime can compile with MSVC.
- Linux: install the matching shared libraries and point the same environment variables at their root directories so `build.rs` can add the include and link search paths.
- Validation: `tests/ml_bindings.rs` covers the Rust tensor marshalling layer and ensures the ML stdlib facade is embedded.
