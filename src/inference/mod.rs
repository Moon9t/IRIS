//! Inference adapter abstractions. Provide a trait that different backends
//! (ONNX, libtorch, TF) can implement. This file contains a lightweight
//! stub implementation so higher-level code can compile without native deps.

use std::path::Path;

pub trait InferenceBackend {
    /// Load a model from path. Implementations may return errors.
    fn load_model(path: &Path) -> Result<Box<dyn InferenceBackend>, String>
    where
        Self: Sized;

    /// Run inference on raw input bytes and receive raw output bytes.
    fn run(&self, input: &[f32]) -> Result<Vec<f32>, String>;
}

/// A stub backend that just echoes the input.
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
