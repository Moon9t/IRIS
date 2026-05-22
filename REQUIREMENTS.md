# IRIS System Requirements

> **IRIS — The C of Machine Learning**
> Comprehensive platform requirements for running, building, and deploying IRIS programs.

---

## Quick Start

```sh
# Verify installation
iris --version

# Run a program (interpreter — no dependencies beyond IRIS itself)
iris hello.iris

# Build a native binary (requires LLVM/clang)
iris build hello.iris
./hello
```

---

## Minimum Requirements (Interpreter / IR Emission)

These are the requirements to run `iris <file.iris>` for IR analysis, or
`iris repl`, `iris lsp`, `iris dap`, and `iris pkg`.

| Component     | Requirement                                    |
|---------------|------------------------------------------------|
| **OS**        | Windows 10+, macOS 12+, Linux (glibc 2.17+)   |
| **Arch**      | x86_64 (AMD64) or ARM64 (aarch64)             |
| **RAM**       | 256 MB minimum                                 |
| **Disk**      | 50 MB (IRIS binary + stdlib)                   |
| **Dependencies** | **None** — fully self-contained             |

---

## Native Compilation Requirements

These are required for `iris build`, `iris run`, `--emit eval`, `--emit jit`,
and `--emit binary`. IRIS compiles through LLVM IR → Clang → native binary.

| Component     | Requirement                                              |
|---------------|----------------------------------------------------------|
| **LLVM/Clang** | Version 17 or newer (18+ recommended)                   |
| **Linker**    | `lld` (bundled with LLVM) or system linker               |
| **C library** | Windows: MinGW UCRT64 · Linux: glibc · macOS: system SDK |
| **RAM**       | 512 MB minimum, 2 GB recommended for large projects      |
| **Disk**      | 500 MB (IRIS + bundled LLVM + sysroot)                   |

### Installing LLVM/Clang

**Ubuntu / Debian:**
```sh
sudo apt update
sudo apt install clang lld
# Or install a specific version:
sudo apt install clang-18 lld-18
```

**Fedora / RHEL:**
```sh
sudo dnf install clang lld
```

**Arch Linux:**
```sh
sudo pacman -S clang lld
```

**macOS (Homebrew):**
```sh
brew install llvm
# Add to PATH:
export PATH="$(brew --prefix llvm)/bin:$PATH"
```

**Windows:**
- The **Full installer** (`.exe` / `.msi`) bundles LLVM and MinGW UCRT64 — no extra steps needed.
- For **portable** installs: download LLVM from https://releases.llvm.org/ and install MSYS2 UCRT64:
  ```powershell
  # Install MSYS2 from https://www.msys2.org/
  # Then in MSYS2 UCRT64 shell:
  pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-headers-git
  ```

---

## ML Workload Requirements (Optional)

These are optional dependencies that unlock accelerated ML performance.

| Component           | Purpose                      | Install                              |
|---------------------|------------------------------|--------------------------------------|
| **OpenBLAS**        | Accelerated matmul (10-50×)  | `apt install libopenblas-dev`        |
| **Intel MKL**       | Accelerated matmul (Intel CPUs) | Via Intel oneAPI toolkit          |
| **CUDA Toolkit 12+** | GPU acceleration            | https://developer.nvidia.com/cuda    |
| **ONNX Runtime**    | ONNX model inference         | https://onnxruntime.ai/              |
| **LibTorch**        | PyTorch model inference      | https://pytorch.org/cppdocs/         |
| **TensorFlow C API** | TensorFlow model inference  | https://www.tensorflow.org/install/lang_c |
| **RAM**             | 4 GB+ for training workloads | —                                    |
| **GPU VRAM**        | 4 GB+ for CUDA workloads    | —                                    |

### Enabling BLAS Acceleration

```sh
# Linux (OpenBLAS)
sudo apt install libopenblas-dev
export IRIS_USE_BLAS=1
iris build my_ml_program.iris    # Links against OpenBLAS

# macOS (Accelerate framework is always available)
# BLAS is available via the system Accelerate framework — no extra install needed.

# Windows
# Install OpenBLAS from https://github.com/OpenMathLib/OpenBLAS/releases
# Set OPENBLAS_DIR environment variable
```

### Enabling ML Backend Inference

```sh
# ONNX Runtime
export ONNXRUNTIME_DIR=/path/to/onnxruntime
export IRIS_NATIVE_ML_BACKENDS=1
iris build examples/ml_full_pipeline.iris

# PyTorch (LibTorch)
export LIBTORCH=/path/to/libtorch
export IRIS_NATIVE_ML_BACKENDS=1

# TensorFlow
export TENSORFLOW_DIR=/path/to/tensorflow
export IRIS_NATIVE_ML_BACKENDS=1
```

---

## Supported Platforms (CI-Tested)

Every release is built and tested on all 6 targets via GitHub Actions.

| Platform          | Architecture | Binary | Installers                  | CI Runner          |
|-------------------|-------------|--------|-----------------------------|-------------------|
| **Windows 10/11** | x86_64      | ✅     | `.exe`, `.zip`, `.msi`      | `windows-latest`  |
| **Windows 11**    | ARM64       | ✅     | `.zip`                      | `windows-latest` (cross) |
| **Ubuntu 20.04+** | x86_64      | ✅     | `.deb`, `.rpm`, `.AppImage` | `ubuntu-latest`   |
| **Ubuntu 22.04+** | ARM64       | ✅     | `.deb`, `.rpm`              | `ubuntu-24.04-arm` |
| **macOS 12+**     | x86_64      | ✅     | `.pkg`, `.dmg`              | `macos-latest`    |
| **macOS 14+**     | ARM64       | ✅     | `.pkg`, `.dmg`              | `macos-14`        |

---

## Environment Variables

| Variable                  | Purpose                                              | Default |
|---------------------------|------------------------------------------------------|---------|
| `IRIS_CLANG`              | Path to clang binary (overrides PATH lookup)         | `clang` |
| `IRIS_USE_BLAS`           | Enable BLAS-accelerated tensor ops (`1` to enable)   | off     |
| `IRIS_NATIVE_ML_BACKENDS` | Link real ONNX/PyTorch/TF backends (`1` to enable)  | off     |
| `ONNXRUNTIME_DIR`         | Path to ONNX Runtime SDK                             | —       |
| `LIBTORCH`                | Path to LibTorch (PyTorch C++ API)                   | —       |
| `TENSORFLOW_DIR`          | Path to TensorFlow C API                             | —       |
| `OPENBLAS_DIR`            | Path to OpenBLAS installation                        | —       |
| `IRIS_LOG`                | Log level: `error`, `warn`, `info`, `debug`, `trace` | `warn`  |

---

## Building from Source

Requires **Rust 1.75+** (stable). No C compiler needed to build IRIS itself —
only for the programs IRIS compiles.

```sh
git clone https://github.com/moon9t/iris.git
cd iris

# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Run tests (1,400+ tests across 110+ suites)
cargo test

# Install globally
cargo install --path .
```

---

## Verification

After installation, verify everything works:

```sh
# Check version
iris --version

# Run a simple program
echo 'def main() -> i64 { print("Hello from IRIS!"); 0 }' > hello.iris
iris run hello.iris

# Run the ML pipeline example
iris run examples/ml_full_pipeline.iris

# Run the AIS agent example
iris run examples/ais_agent.iris

# Build a native binary
iris build hello.iris
./hello      # Linux/macOS
hello.exe    # Windows
```
