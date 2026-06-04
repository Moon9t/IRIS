# ML Backends Demo

A comprehensive demonstration of IRIS's ML capabilities across four areas:

## Sections

1. **Native MLP Training** — Trains a 2→16→2 neural network on XOR-like data using Adam, entirely within IRIS's `std.nn` module.
2. **Vector & Activation Operations** — Showcases `dot`, `vec_add`, `norm`, `softmax`, `sigmoid`, `relu`, `tanh_act` from `std.ml`.
3. **External ML Backends** — Loads and runs models through all three supported runtimes:
   - **ONNX Runtime** — uses `tests/fixtures/identity.onnx`
   - **PyTorch / LibTorch** — `torch_load` / `torch_run` / `torch_free`
   - **TensorFlow C API** — `tf_load` / `tf_run` / `tf_free`
4. **Loss Functions & Metrics** — Computes MSE, MAE, BCE, accuracy, precision, recall, F1, and confusion matrix.

## Running

```bash
# From the IRIS project root:
iris run projects/ml_backends_demo/main.iris

# Or from this directory:
iris run main.iris
```

## Requirements

- IRIS compiler with ML shims linked (`IRIS_LINK_ML_SHIMS=1` build)
- Environment variables set: `ONNXRUNTIME_DIR`, `LIBTORCH_DIR`, `TENSORFLOW_DIR`
- DLLs in `PATH`: `onnxruntime.dll`, `torch_cpu.dll`, `tensorflow.dll`
