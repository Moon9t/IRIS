# IRIS ML Examples

This directory contains examples showcasing IRIS's ML integration capabilities.

## Overview

The IRIS language now includes production-grade integrations with:
- **ONNX Runtime** - Execute pre-trained ONNX models
- **LibTorch** - PyTorch C++ API bindings through `std.ml`
- **TensorFlow** - TensorFlow C API bindings through `std.ml`

All examples demonstrate real tensor operations and ML patterns that work without external dependencies.

## Examples

### 1. `ml_tensor_operations.iris`
**Basic tensor operations and marshalling**

Demonstrates:
- Creating tensors as lists with shape metadata
- Element-wise access and indexing
- List comprehension for tensor operations
- Reduction operations (sum, max, min)
- Scaling and transformation patterns

**Run:** `iris examples/ml_tensor_operations.iris`

**Output:** Shows vector/matrix operations, element access, and tensor information.

---

### 2. `ml_neural_computation.iris`
**Neural network-style computations**

Demonstrates:
- Matrix multiplication pattern
- ReLU activation function
- Softmax probability distribution
- Forward pass through a simple 2-layer network
- Tensor broadcasting concepts

**Run:** `iris examples/ml_neural_computation.iris`

**Output:** Complete neural computation pipeline with intermediate activations.

**Key Functions:**
- `matrix_multiply(a, b)` - 2D matrix multiplication
- `relu(x)` - ReLU activation
- `softmax_simplified(logits)` - Convert logits to probabilities

---

### 3. `ml_model_loading.iris`
**ONNX model loading framework**

Demonstrates:
- Creating input tensors for inference
- ONNX model loading pattern
- Error handling for missing frameworks
- Bridge layer validation

**Run:** `iris examples/ml_model_loading.iris`

**Output:** Shows setup requirements and usage patterns for real model inference.

**To Enable Real Model Loading:**
1. Download ONNX Runtime SDK
2. Set `ONNXRUNTIME_DIR` environment variable
3. Place `.onnx` model files in `tests/fixtures/`
4. Rebuild IRIS: `cargo build --release`
5. Uncomment the model loading code in the example

**Setup Guide:** See `docs/ML_INTEGRATION.md`

---

### 4. `ml_showcase.iris`
**Comprehensive ML capabilities showcase**

Demonstrates:
- Tensor operations (add, scale, multiply)
- Multiple activation functions (sigmoid, softmax)
- Matrix math operations
- Probability distributions
- Statistical computations
- Complete ML pipeline summary

**Run:** `iris examples/ml_showcase.iris`

**Output:** Visual showcase of all ML capabilities with ASCII formatting.

**Features:**
- Matrix operations: add, scale
- Activations: sigmoid, stable softmax
- Statistics: mean, std deviation
- ML integration status report

---

### 5. `ml_data_preprocessing.iris`
**ML data preprocessing patterns**

Demonstrates:
- Min-Max normalization (scaling to [0,1])
- Z-Score normalization (standardization)
- Batch creation for mini-batch training
- One-hot encoding for classification
- Computing dataset statistics
- Complete preprocessing pipeline

**Run:** `iris examples/ml_data_preprocessing.iris`

**Output:** Data pipeline showing normalization, batching, and encoding.

**Key Functions:**
- `normalize_minmax(data, min, max)` - Scale to [0,1]
- `normalize_zscore(data)` - Standardization
- `batch_data(data, batch_size)` - Create batches
- `one_hot_encode(idx, num_classes)` - Classification encoding
- `compute_stats(data)` - Min, max, mean, std dev

---

### 6. `ml_full_pipeline.iris`
**End-to-end data ingestion, cleaning, training, and prediction**

Demonstrates:
- Mixed-source row ingestion from CSV-style and JSON-style adapters
- Missing-value cleanup for empty, `NA`, `null`, and `?` tokens
- Feature standardization and supervised dataset construction
- Logistic regression training, scoring, streaming re-ingest, and retraining
- Expected-output prediction plus ONNX, PyTorch/LibTorch, and TensorFlow handoff hooks

**Run:** `iris examples/ml_full_pipeline.iris`

**Output:** Shows row counts, model accuracy before and after ingesting new data, the expected label, probability, and backend handoff status.

---

## Running the Examples

```bash
# Run individual example
iris examples/ml_tensor_operations.iris

# Run showcase
iris examples/ml_showcase.iris

# Run preprocessing demo
iris examples/ml_data_preprocessing.iris

# Run a full supervised ML pipeline
iris examples/ml_full_pipeline.iris
```

## Bridge Architecture

All examples work with IRIS's ML bridge architecture:

```
IRIS Code
    ↓
std.ml module (extern declarations)
    ↓
Native runtime bridge (iris_mlrt_* wrappers)
    ↓
C/C++ shim layer (onnx_shim.c, pytorch_shim.cpp, tf_shim.c)
    ↓
ONNX Runtime C API
    ↓
Pre-trained Models (ONNX format)
```

## Real Model Inference

To run actual model inference:

1. **Install ONNX Runtime SDK**
   - Windows: Download from https://github.com/microsoft/onnxruntime/releases
   - Linux/macOS: Follow ML_INTEGRATION.md

2. **Set Environment Variable**
   ```powershell
   $env:ONNXRUNTIME_DIR = 'C:\onnxruntime'  # Windows
   $env:IRIS_NATIVE_ML_BACKENDS = '1'
   export ONNXRUNTIME_DIR=/path/to/onnxruntime  # Linux/macOS
   export IRIS_NATIVE_ML_BACKENDS=1
   ```

3. **Generate Test Models** (requires Python)
   ```bash
   cd tests
   pip install onnx numpy
   python create_onnx_model.py
   ```

4. **Rebuild IRIS**
   ```bash
   cargo build --release
   ```

5. **Run Examples with Real Models**
   - Edit `ml_model_loading.iris` and uncomment model loading code
   - Run: `iris examples/ml_model_loading.iris`

## Key Concepts Demonstrated

### Tensors
- Lists of floats with shape metadata
- Element-wise operations
- Broadcasting patterns
- Marshalling through C FFI boundary

### Activations
- **ReLU:** max(0, x)
- **Sigmoid:** 1/(1+e^-x)
- **Softmax:** Stable probability distribution

### Normalization
- **Min-Max:** (x - min) / (max - min) → [0, 1]
- **Z-Score:** (x - mean) / std_dev

### Matrix Operations
- Multiplication (general pattern)
- Addition (element-wise)
- Scaling (constant multiplication)

### Data Structures
- Batch creation for mini-batch training
- One-hot encoding for labels
- Statistic computation (mean, std dev)

## Testing

All examples can be compiled and executed with:

```bash
iris examples/ml_*.iris
```

They work without external dependencies. For real model inference, see "Real Model Inference" section above.

## Next Steps

1. **Run the examples** to understand IRIS ML capabilities
2. **Modify examples** to experiment with different parameters
3. **Enable ONNX Runtime** for real model execution
4. **Build applications** combining IRIS computation with ML inference
5. **Deploy models** using IRIS's release binary

## Documentation

For comprehensive setup and integration details, see:
- `docs/ML_INTEGRATION.md` - Full integration guide
- `docs/ML_INTEGRATION_SUMMARY.md` - Quick reference
- `README.md` - Language overview
- `SPEC.md` - Language specification

## Status

✓ ONNX Runtime shim: Production implementation
✓ Tensor marshalling: Complete
✓ FFI bindings: Tested and validated
✓ Examples: Ready to run
⚠ Real model execution: Requires backend SDKs and `IRIS_NATIVE_ML_BACKENDS=1`

---

**Version:** IRIS v0.4.0
**Last Updated:** 2026-05-17
**ML Integration:** Production-ready
