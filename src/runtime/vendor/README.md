# Vendored third-party headers

These headers are vendored so the IRIS runtime can be **compiled without any ML
SDK installed**. The corresponding shared libraries are resolved lazily at
*runtime* via `dlopen`/`LoadLibrary` — see `iris_ml_dynload.h`.

Vendoring the header does **not** create a link-time dependency. A build of
`iris_runtime.c` never references `onnxruntime.dll`; it only learns the ABI
layout of the API structs.

## `onnxruntime_c_api.h`

- **Source**: ONNX Runtime, `include/onnxruntime_c_api.h`
- **Version**: `ORT_API_VERSION 26`
- **License**: MIT — Copyright (c) Microsoft Corporation. All rights reserved.
- **Local modification**: the trailing `#include "onnxruntime_ep_c_api.h"` was
  removed. It is the final line of the file, so no declaration above it can
  depend on it, and IRIS does not use the execution-provider plugin API.

The loader negotiates the API version downward at runtime (see
`iris_ort_api()`), so an `onnxruntime.dll` older than 26 still works as long as
it exposes the functions IRIS actually calls.

## TensorFlow

The TensorFlow C API is flat (plain `TF_*` functions rather than a struct of
function pointers), so no header is vendored. The ~20 entry points IRIS uses are
declared directly in `iris_ml_dynload.h` against the documented stable C ABI.

## LibTorch

Not vendored. LibTorch exposes a **C++** API, whose mangled symbols and ABI make
in-process `dlopen` impractical. It is handled as an optional, separately built
C-ABI plugin instead, so the core runtime never depends on it.
