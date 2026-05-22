# LLM Inference Demo

A demonstration of IRIS's C-bindings for running external ML models via `std.ml` (ONNX/Torch/TF).
This project uses the downloaded `tiny-random-gpt2.onnx` model to run an inference pass.

## Building

```bash
iris build projects/llm_inference/main.iris -o llm_inference
```

## Usage

```bash
./llm_inference
```
