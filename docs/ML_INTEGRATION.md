# IRIS ML Integration Guide

This guide explains how to set up and test ONNX Runtime, LibTorch, and TensorFlow integration with IRIS.

## Architecture Overview

```
IRIS Source Code
    ↓
Stdlib Module: std.ml
    ↓
Rust FFI Bridge (src/runtime_bindings.rs)
    ↓
C Shim Layer (src/runtime/*.c)
    ↓
Native ML Framework APIs
    ↓ (ONNX Runtime, LibTorch, TensorFlow)
```

### Data Flow

- **IRIS Tensors**: Represented as lists of f64 values + shape list
- **IrisTensor (C struct)**: `float* data, int64_t* shape, int32_t ndim, int64_t numel`
- **Native Tensors**: Framework-specific (OrtValue, torch::Tensor, TfTensor)

## ONNX Runtime Integration

### Windows Setup

1. **Download ONNX Runtime SDK**:
   - Visit: https://github.com/microsoft/onnxruntime/releases
   - Download: `onnxruntime-win-x64-1.17.0.zip` (or latest)
   - Extract to: `C:\onnxruntime` (or your preferred location)

2. **Set Environment Variable**:
   ```powershell
   $env:ONNXRUNTIME_DIR = "C:\onnxruntime"
   ```

3. **Verify Installation**:
   ```powershell
   Test-Path "$env:ONNXRUNTIME_DIR\include\onnxruntime_c_api.h"
   Test-Path "$env:ONNXRUNTIME_DIR\lib\onnxruntime.lib"
   ```

4. **Build IRIS with ONNX Support**:
   ```powershell
   cargo clean
   cargo build
   ```

   Expected output during build:
   ```
   Linking ONNX Runtime from: C:\onnxruntime
   ```

5. **Generate Test Models**:
   ```bash
   cd tests
   pip install onnx numpy
   python create_onnx_model.py
   ```

   This creates:
   - `fixtures/identity.onnx`: Identity function (output = input)
   - `fixtures/add_const.onnx`: Add constant (output = input + 2.0)
   - `fixtures/matmul.onnx`: Matrix multiplication

6. **Run Tests**:
   ```bash
   cargo test --test ml_inference -- --nocapture
   ```

### Linux Setup

1. **Download ONNX Runtime**:
   ```bash
   cd /opt
   wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz
   tar xzf onnxruntime-linux-x64-1.17.0.tgz
   ```

2. **Set Environment Variable**:
   ```bash
   export ONNXRUNTIME_DIR=/opt/onnxruntime-linux-x64-1.17.0
   ```

3. **Build and Test**:
   ```bash
   cargo clean
   cargo build
   cd tests && python create_onnx_model.py
   cargo test --test ml_inference
   ```

## LibTorch Integration (PyTorch C++)

### Windows Setup

1. **Download LibTorch**:
   - Visit: https://pytorch.org/get-started/locally/
   - Select: C++ → CPU → Windows
   - Download: `libtorch-win-shared-with-deps-*.zip`
   - Extract to: `C:\libtorch`

2. **Set Environment Variable**:
   ```powershell
   $env:LIBTORCH_DIR = "C:\libtorch"
   ```

3. **Build**:
   ```powershell
   cargo clean
   cargo build
   ```

4. **Create Test Models**:
   ```python
   import torch
   model = torch.nn.Sequential(
       torch.nn.Linear(3, 4),
       torch.nn.ReLU()
   )
   scripted = torch.jit.script(model)
   torch.jit.save(scripted, 'tests/fixtures/model.pt')
   ```

5. **Run Tests**:
   ```bash
   cargo test --test ml_inference
   ```

### Linux Setup

Similar to ONNX Runtime, but with LibTorch:

```bash
export LIBTORCH_DIR=/opt/libtorch
cargo clean && cargo build
cargo test --test ml_inference
```

## TensorFlow Integration

### Windows Setup

1. **Download TensorFlow C API**:
   ```
   https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-windows-x86_64-2.13.0.zip
   ```
   Or CPU version (recommended for testing):
   ```
   https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-2.13.0.zip
   ```

2. **Extract and Set Environment**:
   ```powershell
   $env:TENSORFLOW_DIR = "C:\tensorflow"
   ```

3. **Build and Test**:
   ```bash
   cargo clean && cargo build
   cargo test --test ml_inference
   ```

## Testing End-to-End Model Execution

### Test a Simple ONNX Model

```rust
#[test]
fn test_onnx_add_model() {
    unsafe {
        // Load model
        let session = iris_onnx_session_create(CString::new("fixtures/add_const.onnx").unwrap().as_ptr());
        
        // Create input [2, 3] with values [1,2,3,4,5,6]
        let input_shape = [2i64, 3i64];
        let input = iris_tensor_alloc(2, input_shape.as_ptr());
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        std::ptr::copy_nonoverlapping(data.as_ptr(), (*input).data, 6);
        
        // Run inference
        let mut inputs = vec![input];
        let mut outputs: *mut *mut IrisTensor = std::ptr::null_mut();
        let mut n_outputs = 0;
        
        let rc = iris_onnx_session_run(session, inputs.as_mut_ptr(), 1, &mut outputs, &mut n_outputs);
        
        if rc == 0 {
            // Success! Output should be [3,4,5,6,7,8]
            let output_data = std::slice::from_raw_parts((*(*outputs)).data, 6);
            assert_eq!(output_data, &[3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        }
        
        // Cleanup
        iris_tensor_free(input);
        iris_tensor_free(*outputs);
        iris_onnx_session_free(session);
    }
}
```

### Using ML from IRIS Code

```iris
import std.ml;

def test_onnx_inference() {
    // Load ONNX model
    let model_id = std.ml.onnx_load("model.onnx");
    
    // Create input tensor: [2, 3]
    let input = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    
    // Run inference (shape is inferred)
    let output = std.ml.onnx_run(model_id, input);
    
    // Use output
    print(output);
    
    // Cleanup
    std.ml.onnx_free(model_id);
}
```

## Troubleshooting

### "ONNX Runtime not enabled at build time"

**Cause**: `ONNXRUNTIME_DIR` environment variable not set.

**Solution**:
```powershell
# PowerShell
$env:ONNXRUNTIME_DIR = "C:\onnxruntime"
cargo clean
cargo build
```

### "Cannot open include file: 'onnxruntime_c_api.h'"

**Cause**: SDK path is incorrect or incomplete.

**Solution**:
1. Verify SDK structure:
   ```powershell
   ls $env:ONNXRUNTIME_DIR\include\
   ls $env:ONNXRUNTIME_DIR\lib\
   ```
2. Both directories should exist with appropriate files.

### "unresolved external symbol iris_pytorch_load"

**Cause**: LibTorch SDK not available or path incorrect.

**Solution**:
```powershell
$env:LIBTORCH_DIR = "C:\libtorch"
cargo clean && cargo build
```

### Test models not found

**Cause**: `tests/fixtures/` directory missing model files.

**Solution**:
```bash
cd tests
pip install onnx numpy torch
python create_onnx_model.py
```

## Platform-Specific Notes

### Windows MSVC Specifics

- Uses Visual C++ compiler (cl.exe)
- Link with `.lib` files (not `.a` or `.so`)
- Ensure ONNXRUNTIME_DIR contains both:
  - `include/onnxruntime_c_api.h`
  - `lib/onnxruntime.lib` (or `onnxruntime.dll.a`)

### Linux Specifics

- Uses GCC/Clang compiler
- Link with `.so` files
- Set `LD_LIBRARY_PATH` if needed:
  ```bash
  export LD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$LD_LIBRARY_PATH
  ```

### macOS Specifics

- Uses Clang compiler
- Link with `.dylib` files
- May need to set `DYLD_LIBRARY_PATH`:
  ```bash
  export DYLD_LIBRARY_PATH=$ONNXRUNTIME_DIR/lib:$DYLD_LIBRARY_PATH
  ```

## Build System Integration

The build script (`build.rs`) automatically:

1. Checks for `ONNXRUNTIME_DIR`, `LIBTORCH_DIR`, `TENSORFLOW_DIR` env vars
2. Includes C/C++ headers from `include/` subdirectories
3. Links libraries from `lib/` subdirectories
4. Sets conditional compilation flags (`ONNX_RUNTIME_ENABLED`, etc.)
5. Compiles C/C++ shim files with appropriate flags

If an env var is not set, the corresponding backend compiles fallback stubs that return error codes.

## Next Steps

1. Choose a framework (start with **ONNX Runtime** - simplest)
2. Download and extract the SDK
3. Set the environment variable
4. Run `cargo clean && cargo build`
5. Generate test models
6. Run tests: `cargo test --test ml_inference -- --nocapture`

For questions or issues, check the build output logs in `target/debug/build/iris-*/output/`.
