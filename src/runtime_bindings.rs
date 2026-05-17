use std::os::raw::{c_char, c_void};

#[repr(C)]
pub struct IrisTensor {
    pub data: *mut f32,
    pub shape: *mut i64,
    pub ndim: i32,
    pub numel: i64,
}

#[repr(C)]
pub struct IrisVal {
    pub tag: i32,
    pub data: IrisValData,
}

#[repr(C)]
pub union IrisValData {
    pub i64_: i64,
    pub i32_: i32,
    pub f64_: f64,
    pub f32_: f32,
    pub boolean: u8,
    pub str_: *mut c_char,
    pub ptr: *mut c_void,
}

#[repr(C)]
pub struct IrisList {
    pub data: *mut *mut IrisVal,
    pub len: usize,
    pub cap: usize,
}

#[repr(C)]
pub struct TensorPair {
    pub data: *mut IrisList,
    pub shape: *mut IrisList,
}

const IRIS_TAG_I64: i32 = 0;
const IRIS_TAG_F64: i32 = 2;

extern "C" {
    fn free(ptr: *mut c_void);

    fn iris_tensor_alloc(ndim: i32, shape: *const i64) -> *mut IrisTensor;
    fn iris_tensor_free(t: *mut IrisTensor);

    fn iris_box_f64(v: f64) -> *mut IrisVal;
    fn iris_box_i64(v: i64) -> *mut IrisVal;
    fn iris_list_new() -> *mut IrisList;
    fn iris_list_push(list: *mut IrisList, val: *mut IrisVal);

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

fn list_to_vec_f64(list: *mut IrisList) -> Option<Vec<f64>> {
    if list.is_null() {
        return None;
    }
    unsafe {
        let list_ref = &*list;
        let mut values = Vec::with_capacity(list_ref.len);
        for idx in 0..list_ref.len {
            let value_ptr = *list_ref.data.add(idx);
            if value_ptr.is_null() {
                return None;
            }
            let value_ref = &*value_ptr;
            match value_ref.tag {
                IRIS_TAG_F64 => values.push(value_ref.data.f64_),
                IRIS_TAG_I64 => values.push(value_ref.data.i64_ as f64),
                _ => return None,
            }
        }
        Some(values)
    }
}

fn list_to_vec_i64(list: *mut IrisList) -> Option<Vec<i64>> {
    if list.is_null() {
        return None;
    }
    unsafe {
        let list_ref = &*list;
        let mut values = Vec::with_capacity(list_ref.len);
        for idx in 0..list_ref.len {
            let value_ptr = *list_ref.data.add(idx);
            if value_ptr.is_null() {
                return None;
            }
            let value_ref = &*value_ptr;
            match value_ref.tag {
                IRIS_TAG_I64 => values.push(value_ref.data.i64_),
                IRIS_TAG_F64 => values.push(value_ref.data.f64_ as i64),
                _ => return None,
            }
        }
        Some(values)
    }
}

unsafe fn tensor_pair_to_native(pair: TensorPair) -> Option<*mut IrisTensor> {
    let data = list_to_vec_f64(pair.data)?;
    let shape = list_to_vec_i64(pair.shape)?;
    if shape.is_empty() {
        return None;
    }
    let tensor = iris_tensor_alloc(shape.len() as i32, shape.as_ptr());
    if tensor.is_null() {
        return None;
    }
    let tensor_ref = &mut *tensor;
    if tensor_ref.numel != data.len() as i64 {
        iris_tensor_free(tensor);
        return None;
    }
    for (idx, value) in data.iter().enumerate() {
        *tensor_ref.data.add(idx) = *value as f32;
    }
    Some(tensor)
}

unsafe fn native_tensor_to_pair(tensor: *mut IrisTensor) -> Option<TensorPair> {
    if tensor.is_null() {
        return None;
    }
    let tensor_ref = &*tensor;
    let data_list = iris_list_new();
    let shape_list = iris_list_new();
    if data_list.is_null() || shape_list.is_null() {
        return None;
    }

    for idx in 0..tensor_ref.numel {
        iris_list_push(
            data_list,
            iris_box_f64(*tensor_ref.data.add(idx as usize) as f64),
        );
    }
    for idx in 0..tensor_ref.ndim {
        iris_list_push(
            shape_list,
            iris_box_i64(*tensor_ref.shape.add(idx as usize)),
        );
    }

    Some(TensorPair {
        data: data_list,
        shape: shape_list,
    })
}

fn run_single_input(
    session: *mut c_void,
    runner: unsafe extern "C" fn(
        *mut c_void,
        *mut *mut IrisTensor,
        usize,
        *mut *mut *mut IrisTensor,
        *mut usize,
    ) -> i32,
    input: TensorPair,
) -> Option<TensorPair> {
    unsafe {
        let native = tensor_pair_to_native(input)?;
        let mut inputs = vec![native];
        let mut output_ptrs: *mut *mut IrisTensor = std::ptr::null_mut();
        let mut n_outputs: usize = 0;
        let status = runner(
            session,
            inputs.as_mut_ptr(),
            inputs.len(),
            &mut output_ptrs,
            &mut n_outputs,
        );
        iris_tensor_free(native);
        if status != 0 || output_ptrs.is_null() || n_outputs == 0 {
            return None;
        }

        let mut first_pair = None;
        for idx in 0..n_outputs {
            let output = *output_ptrs.add(idx);
            if output.is_null() {
                continue;
            }
            if first_pair.is_none() {
                first_pair = native_tensor_to_pair(output);
            }
            iris_tensor_free(output);
        }
        free(output_ptrs as *mut c_void);
        first_pair
    }
}

#[no_mangle]
pub extern "C" fn iris_ml_onnx_load(model_path: *const c_char) -> i64 {
    unsafe { iris_onnx_session_create(model_path) as i64 }
}

#[no_mangle]
pub extern "C" fn iris_ml_onnx_free(session: i64) -> i64 {
    unsafe { iris_onnx_session_free(session as *mut c_void) }
    0
}

#[no_mangle]
pub extern "C" fn iris_ml_onnx_run(session: i64, input: TensorPair) -> TensorPair {
    run_single_input(session as *mut c_void, iris_onnx_session_run, input).unwrap_or(TensorPair {
        data: std::ptr::null_mut(),
        shape: std::ptr::null_mut(),
    })
}

#[cfg(libtorch_enabled)]
#[no_mangle]
pub extern "C" fn iris_ml_pytorch_load(model_path: *const c_char) -> i64 {
    unsafe { iris_pytorch_load(model_path) as i64 }
}

#[cfg(libtorch_enabled)]
#[no_mangle]
pub extern "C" fn iris_ml_pytorch_free(model: i64) -> i64 {
    unsafe { iris_pytorch_free(model as *mut c_void) }
    0
}

#[cfg(libtorch_enabled)]
#[no_mangle]
pub extern "C" fn iris_ml_pytorch_run(model: i64, input: TensorPair) -> TensorPair {
    run_single_input(model as *mut c_void, iris_pytorch_run, input).unwrap_or(TensorPair {
        data: std::ptr::null_mut(),
        shape: std::ptr::null_mut(),
    })
}

#[cfg(not(libtorch_enabled))]
#[no_mangle]
pub extern "C" fn iris_ml_pytorch_load(_model_path: *const c_char) -> i64 {
    0
}

#[cfg(not(libtorch_enabled))]
#[no_mangle]
pub extern "C" fn iris_ml_pytorch_free(_model: i64) -> i64 {
    0
}

#[cfg(not(libtorch_enabled))]
#[no_mangle]
pub extern "C" fn iris_ml_pytorch_run(_model: i64, _input: TensorPair) -> TensorPair {
    TensorPair {
        data: std::ptr::null_mut(),
        shape: std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn iris_ml_tf_load(model_path: *const c_char) -> i64 {
    unsafe { iris_tf_load_saved_model(model_path) as i64 }
}

#[no_mangle]
pub extern "C" fn iris_ml_tf_free(model: i64) -> i64 {
    unsafe { iris_tf_free(model as *mut c_void) }
    0
}

#[no_mangle]
pub extern "C" fn iris_ml_tf_run(model: i64, input: TensorPair) -> TensorPair {
    run_single_input(model as *mut c_void, iris_tf_run, input).unwrap_or(TensorPair {
        data: std::ptr::null_mut(),
        shape: std::ptr::null_mut(),
    })
}

pub fn tensor_pair_from_slices(data: &[f64], shape: &[i64]) -> Option<TensorPair> {
    unsafe {
        let data_list = iris_list_new();
        let shape_list = iris_list_new();
        if data_list.is_null() || shape_list.is_null() {
            return None;
        }
        for value in data {
            iris_list_push(data_list, iris_box_f64(*value));
        }
        for dim in shape {
            iris_list_push(shape_list, iris_box_i64(*dim));
        }
        Some(TensorPair {
            data: data_list,
            shape: shape_list,
        })
    }
}

pub fn tensor_pair_to_slices(pair: TensorPair) -> Option<(Vec<f64>, Vec<i64>)> {
    Some((list_to_vec_f64(pair.data)?, list_to_vec_i64(pair.shape)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_pair_roundtrip() {
        let pair = tensor_pair_from_slices(&[1.0, 2.5, -3.0, 4.25], &[2, 2]).expect("pair");
        let (data, shape) = tensor_pair_to_slices(pair).expect("slices");
        assert_eq!(data, vec![1.0, 2.5, -3.0, 4.25]);
        assert_eq!(shape, vec![2, 2]);
    }
}
