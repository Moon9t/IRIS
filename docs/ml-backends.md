# ML Backends

IRIS supports three external inference backends: **ONNX Runtime**, **TensorFlow**
and **LibTorch**. As of the runtime-loading change, none of them is required at
build time, and none is linked into your binary.

## How they are resolved

| Backend | Mechanism | What you need at build time | What you need at run time |
|---|---|---|---|
| ONNX Runtime | `dlopen` on first use | nothing | `onnxruntime.dll` / `libonnxruntime.so` on the library path |
| TensorFlow | `dlopen` on first use | nothing | `tensorflow.dll` / `libtensorflow.so` on the library path |
| LibTorch | C-ABI plugin, `dlopen`ed | nothing | `iris_torch_plugin` + LibTorch on the library path |

If a backend is missing, the corresponding IRIS functions fail with a clear
message at the point of use. Programs that never call into ML are unaffected —
they do not reference the libraries at all.

This replaces the old scheme, where `-DONNX_RUNTIME_ENABLED` and friends were set
at compile time and `-lonnxruntime` was passed at link time. That approach had two
problems: `iris_runtime.o` differed depending on which SDKs happened to be
installed on the build machine (so no single object could be shipped, which forced
every user to have a C compiler), and a binary built with ONNX enabled refused to
start without `onnxruntime.dll` even if it never touched ML.

`IRIS_NATIVE_ML_BACKENDS` is obsolete and ignored.

## ONNX Runtime

Install ONNX Runtime and make sure the shared library is findable:

- **Windows** — `onnxruntime.dll` next to your binary, on `PATH`, or at
  `C:\onnxruntime\lib\`.
- **Linux** — `libonnxruntime.so` on `LD_LIBRARY_PATH` or in `/usr/local/lib`.
- **macOS** — `libonnxruntime.dylib` on `DYLD_LIBRARY_PATH`, `/usr/local/lib`, or
  `/opt/homebrew/lib`.

IRIS negotiates the ORT API version downward at load time, so a library older than
the one IRIS was built against still works provided it exposes the functions IRIS
uses. The old statically linked path could not do this.

## TensorFlow

Install the TensorFlow **C** library (`libtensorflow`) and make it findable, as
above. IRIS resolves roughly twenty `TF_*` entry points; if any are missing it
reports a version mismatch rather than crashing.

## LibTorch

LibTorch exposes a **C++** API. Mangled symbols and C++ ABI details make it
impractical to `dlopen` directly, so it is handled differently: `pytorch_shim.cpp`
(whose entry points are already `extern "C"`) is compiled into a small standalone
library that links LibTorch, and the IRIS runtime loads *that*.

Build it once:

```sh
# Requires LibTorch and a C++ compiler — only for this plugin, not for IRIS itself.
LIBTORCH_DIR=/path/to/libtorch cargo run --release -- <your build command>
```

Programmatically, the builder is `codegen::build::build_torch_plugin(libtorch_dir,
output_dir)`, which produces:

- `iris_torch_plugin.dll` (Windows)
- `libiris_torch_plugin.so` (Linux)
- `libiris_torch_plugin.dylib` (macOS)

Place the result next to your binary or on the library search path. The runtime
resolves four symbols from it: `iris_pytorch_load`, `iris_pytorch_run`,
`iris_pytorch_free`, `iris_pytorch_train_step`.

> **Note:** a CLI subcommand to drive `build_torch_plugin` is not wired up yet;
> the function is available but must currently be called from Rust.

## Vendored headers

`src/runtime/vendor/onnxruntime_c_api.h` is vendored (MIT, Microsoft) so the ONNX
shim can be compiled with no SDK present. It supplies ABI layout only and creates
no link-time dependency. The TensorFlow C ABI is declared by hand in
`src/runtime/iris_ml_dynload.h`. See `src/runtime/vendor/README.md` for details
and local modifications.
