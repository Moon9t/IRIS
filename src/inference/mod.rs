//! Inference adapter abstractions. Provide a trait that different backends
//! (ONNX, libtorch, TF) can implement. This file contains concrete struct
//! implementations mapping to the native C/C++ shims in the IRIS runtime.

use std::os::raw::{c_char, c_void};
use std::path::Path;

pub trait InferenceBackend {
    /// Load a model from path. Implementations may return errors.
    fn load_model(path: &Path) -> Result<Box<dyn InferenceBackend>, String>
    where
        Self: Sized;

    /// Run inference on raw input bytes and receive raw output bytes.
    fn run(&self, input: &[f32]) -> Result<Vec<f32>, String>;
}

// ---------------------------------------------------------------------------
// Native C Runtime Structs and Declarations
// ---------------------------------------------------------------------------

#[repr(C)]
struct IrisTensor {
    data: *mut f32,
    shape: *mut i64,
    ndim: i32,
    numel: i64,
}

extern "C" {
    fn free(ptr: *mut c_void);

    fn iris_tensor_alloc(ndim: i32, shape: *const i64) -> *mut IrisTensor;
    fn iris_tensor_free(t: *mut IrisTensor);

    fn iris_onnx_session_create(model_path: *const c_char) -> *mut c_void;
    fn iris_onnx_session_run(
        session: *mut c_void,
        inputs: *mut *mut IrisTensor,
        n_inputs: usize,
        outputs: *mut *mut *mut IrisTensor,
        n_outputs: *mut usize,
    ) -> i32;
    fn iris_onnx_session_free(session: *mut c_void);

    #[cfg(libtorch_enabled)]
    fn iris_pytorch_load(model_path: *const c_char) -> *mut c_void;
    #[cfg(libtorch_enabled)]
    fn iris_pytorch_run(
        model: *mut c_void,
        inputs: *mut *mut IrisTensor,
        n_inputs: usize,
        outputs: *mut *mut *mut IrisTensor,
        n_outputs: *mut usize,
    ) -> i32;
    #[cfg(libtorch_enabled)]
    fn iris_pytorch_free(model: *mut c_void);

    fn iris_tf_load_saved_model(path: *const c_char) -> *mut c_void;
    fn iris_tf_run(
        model: *mut c_void,
        inputs: *mut *mut IrisTensor,
        n_inputs: usize,
        outputs: *mut *mut *mut IrisTensor,
        n_outputs: *mut usize,
    ) -> i32;
    fn iris_tf_free(model: *mut c_void);
}

// ---------------------------------------------------------------------------
// 1. ONNX Runtime Backend Implementation
// ---------------------------------------------------------------------------

pub struct OnnxBackend {
    session: *mut c_void,
}

impl InferenceBackend for OnnxBackend {
    fn load_model(path: &Path) -> Result<Box<dyn InferenceBackend>, String> {
        let path_str = path.to_str().ok_or_else(|| "invalid path".to_owned())?;
        let c_path = std::ffi::CString::new(path_str).map_err(|e| e.to_string())?;
        unsafe {
            let session = iris_onnx_session_create(c_path.as_ptr());
            if session.is_null() {
                return Err("Failed to load ONNX model via C shim".to_owned());
            }
            Ok(Box::new(OnnxBackend { session }))
        }
    }

    fn run(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        unsafe {
            let shape = [1i64, input.len() as i64];
            let iris_in = iris_tensor_alloc(2, shape.as_ptr());
            if iris_in.is_null() {
                return Err("Failed to allocate input tensor".to_owned());
            }
            std::ptr::copy_nonoverlapping(input.as_ptr(), (*iris_in).data, input.len());

            let mut inputs = [iris_in];
            let mut outputs: *mut *mut IrisTensor = std::ptr::null_mut();
            let mut n_outputs = 0;

            let rc = iris_onnx_session_run(
                self.session,
                inputs.as_mut_ptr(),
                1,
                &mut outputs,
                &mut n_outputs,
            );

            iris_tensor_free(iris_in);

            if rc != 0 || outputs.is_null() || n_outputs == 0 {
                if !outputs.is_null() {
                    free(outputs as *mut c_void);
                }
                return Err("ONNX session run failed".to_owned());
            }

            let out_tensor = *outputs;
            if out_tensor.is_null() {
                free(outputs as *mut c_void);
                return Err("ONNX session returned null output tensor".to_owned());
            }

            let numel = (*out_tensor).numel as usize;
            let out_slice = std::slice::from_raw_parts((*out_tensor).data, numel);
            let result = out_slice.to_vec();

            iris_tensor_free(out_tensor);
            free(outputs as *mut c_void);

            Ok(result)
        }
    }
}

impl Drop for OnnxBackend {
    fn drop(&mut self) {
        unsafe {
            iris_onnx_session_free(self.session);
        }
    }
}

// ---------------------------------------------------------------------------
// 2. PyTorch (LibTorch) Backend Implementation
// ---------------------------------------------------------------------------

#[cfg(libtorch_enabled)]
pub struct TorchBackend {
    model: *mut c_void,
}

#[cfg(libtorch_enabled)]
impl InferenceBackend for TorchBackend {
    fn load_model(path: &Path) -> Result<Box<dyn InferenceBackend>, String> {
        let path_str = path.to_str().ok_or_else(|| "invalid path".to_owned())?;
        let c_path = std::ffi::CString::new(path_str).map_err(|e| e.to_string())?;
        unsafe {
            let model = iris_pytorch_load(c_path.as_ptr());
            if model.is_null() {
                return Err("Failed to load PyTorch model via C++ shim".to_owned());
            }
            Ok(Box::new(TorchBackend { model }))
        }
    }

    fn run(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        unsafe {
            let shape = [1i64, input.len() as i64];
            let iris_in = iris_tensor_alloc(2, shape.as_ptr());
            if iris_in.is_null() {
                return Err("Failed to allocate input tensor".to_owned());
            }
            std::ptr::copy_nonoverlapping(input.as_ptr(), (*iris_in).data, input.len());

            let mut inputs = [iris_in];
            let mut outputs: *mut *mut IrisTensor = std::ptr::null_mut();
            let mut n_outputs = 0;

            let rc = iris_pytorch_run(
                self.model,
                inputs.as_mut_ptr(),
                1,
                &mut outputs,
                &mut n_outputs,
            );

            iris_tensor_free(iris_in);

            if rc != 0 || outputs.is_null() || n_outputs == 0 {
                if !outputs.is_null() {
                    free(outputs as *mut c_void);
                }
                return Err("PyTorch run failed".to_owned());
            }

            let out_tensor = *outputs;
            if out_tensor.is_null() {
                free(outputs as *mut c_void);
                return Err("PyTorch returned null output tensor".to_owned());
            }

            let numel = (*out_tensor).numel as usize;
            let out_slice = std::slice::from_raw_parts((*out_tensor).data, numel);
            let result = out_slice.to_vec();

            iris_tensor_free(out_tensor);
            free(outputs as *mut c_void);

            Ok(result)
        }
    }
}

#[cfg(libtorch_enabled)]
impl Drop for TorchBackend {
    fn drop(&mut self) {
        unsafe {
            iris_pytorch_free(self.model);
        }
    }
}

// Fallback TorchBackend when libtorch is disabled
#[cfg(not(libtorch_enabled))]
pub struct TorchBackend;

#[cfg(not(libtorch_enabled))]
impl InferenceBackend for TorchBackend {
    fn load_model(_path: &Path) -> Result<Box<dyn InferenceBackend>, String> {
        Err("LibTorch is not enabled at build time. Install SDK and set LIBTORCH_DIR.".to_owned())
    }
    fn run(&self, _input: &[f32]) -> Result<Vec<f32>, String> {
        Err("LibTorch is not enabled at build time.".to_owned())
    }
}

// ---------------------------------------------------------------------------
// 3. TensorFlow Backend Implementation
// ---------------------------------------------------------------------------

pub struct TfBackend {
    model: *mut c_void,
}

impl InferenceBackend for TfBackend {
    fn load_model(path: &Path) -> Result<Box<dyn InferenceBackend>, String> {
        let path_str = path.to_str().ok_or_else(|| "invalid path".to_owned())?;
        let c_path = std::ffi::CString::new(path_str).map_err(|e| e.to_string())?;
        unsafe {
            let model = iris_tf_load_saved_model(c_path.as_ptr());
            if model.is_null() {
                return Err("Failed to load TensorFlow model via C shim".to_owned());
            }
            Ok(Box::new(TfBackend { model }))
        }
    }

    fn run(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        unsafe {
            let shape = [1i64, input.len() as i64];
            let iris_in = iris_tensor_alloc(2, shape.as_ptr());
            if iris_in.is_null() {
                return Err("Failed to allocate input tensor".to_owned());
            }
            std::ptr::copy_nonoverlapping(input.as_ptr(), (*iris_in).data, input.len());

            let mut inputs = [iris_in];
            let mut outputs: *mut *mut IrisTensor = std::ptr::null_mut();
            let mut n_outputs = 0;

            let rc = iris_tf_run(
                self.model,
                inputs.as_mut_ptr(),
                1,
                &mut outputs,
                &mut n_outputs,
            );

            iris_tensor_free(iris_in);

            if rc != 0 || outputs.is_null() || n_outputs == 0 {
                if !outputs.is_null() {
                    free(outputs as *mut c_void);
                }
                return Err("TensorFlow run failed".to_owned());
            }

            let out_tensor = *outputs;
            if out_tensor.is_null() {
                free(outputs as *mut c_void);
                return Err("TensorFlow returned null output tensor".to_owned());
            }

            let numel = (*out_tensor).numel as usize;
            let out_slice = std::slice::from_raw_parts((*out_tensor).data, numel);
            let result = out_slice.to_vec();

            iris_tensor_free(out_tensor);
            free(outputs as *mut c_void);

            Ok(result)
        }
    }
}

impl Drop for TfBackend {
    fn drop(&mut self) {
        unsafe {
            iris_tf_free(self.model);
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Lightweight Stub Backend (Default Fallback)
// ---------------------------------------------------------------------------

pub struct StubBackend;

impl InferenceBackend for StubBackend {
    fn load_model(_path: &Path) -> Result<Box<dyn InferenceBackend>, String> {
        Ok(Box::new(StubBackend))
    }
    fn run(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        Ok(input.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn stub_runs() {
        let b = StubBackend::load_model(&PathBuf::from("/tmp/not_needed")).unwrap();
        let out = b.run(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }
}
