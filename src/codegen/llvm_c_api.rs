//! LLVM C API binding for compiling `.ll` → `.o`/`.obj` without `clang`.
//!
//! Uses `libloading` to load `LLVM-C.dll` at runtime, then calls the stable
//! LLVM C API functions to parse LLVM IR text and emit a native object file.
//!
//! Supported on Windows (LLVM-C.dll) and POSIX (libLLVM.so / libLLVM.dylib).

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;
use std::ptr;

use libloading::{Library, Symbol};

use crate::error::CodegenError;

// ---------------------------------------------------------------------------
// Opaque LLVM C API types
// ---------------------------------------------------------------------------

#[repr(C)]
struct LLVMOpaqueContext;
#[repr(C)]
struct LLVMOpaqueModule;
#[repr(C)]
struct LLVMOpaqueMemoryBuffer;
#[repr(C)]
struct LLVMOpaqueTargetMachine;
#[repr(C)]
struct LLVMOpaqueTarget;

type LLVMContextRef = *mut LLVMOpaqueContext;
type LLVMModuleRef = *mut LLVMOpaqueModule;
type LLVMMemoryBufferRef = *mut LLVMOpaqueMemoryBuffer;
type LLVMTargetMachineRef = *mut LLVMOpaqueTargetMachine;
type LLVMTargetRef = *mut LLVMOpaqueTarget;

// ---------------------------------------------------------------------------
// Function pointer types for LLVM C API
// ---------------------------------------------------------------------------

type FnGetGlobalContext = unsafe extern "C" fn() -> LLVMContextRef;
type FnCreateMemoryBufferWithMemoryRangeCopy =
    unsafe extern "C" fn(*const c_char, usize, *const c_char) -> LLVMMemoryBufferRef;

type FnParseIRInContext = unsafe extern "C" fn(
    LLVMContextRef,
    LLVMMemoryBufferRef,
    *mut LLVMModuleRef,
    *mut *mut c_char,
) -> bool;

type FnGetTargetFromTriple =
    unsafe extern "C" fn(*const c_char, *mut LLVMTargetRef, *mut *mut c_char) -> bool;

type FnCreateTargetMachine = unsafe extern "C" fn(
    LLVMTargetRef,
    *const c_char,
    *const c_char,
    *const c_char,
    u32,
    u32,
    u32,
) -> LLVMTargetMachineRef;

type FnTargetMachineEmitToMemoryBuffer = unsafe extern "C" fn(
    LLVMTargetMachineRef,
    LLVMModuleRef,
    u32,
    *mut *mut c_char,
) -> LLVMMemoryBufferRef;

type FnGetBufferStart = unsafe extern "C" fn(LLVMMemoryBufferRef) -> *const c_char;
type FnGetBufferSize = unsafe extern "C" fn(LLVMMemoryBufferRef) -> u64;
type FnDisposeMemoryBuffer = unsafe extern "C" fn(LLVMMemoryBufferRef);
type FnDisposeModule = unsafe extern "C" fn(LLVMModuleRef);
type FnDisposeMessage = unsafe extern "C" fn(*mut c_char);
type FnSetTarget = unsafe extern "C" fn(LLVMModuleRef, *const c_char);
type FnSetDataLayout = unsafe extern "C" fn(LLVMModuleRef, *const c_char);

// ---------------------------------------------------------------------------
// Loaded library wrapper: stores raw function pointers
// ---------------------------------------------------------------------------

struct LlvmCApi {
    _lib: Library,

    get_global_context: FnGetGlobalContext,
    create_memory_buffer_with_memory_range_copy: FnCreateMemoryBufferWithMemoryRangeCopy,
    parse_ir_in_context: FnParseIRInContext,
    get_target_from_triple: FnGetTargetFromTriple,
    create_target_machine: FnCreateTargetMachine,
    target_machine_emit_to_memory_buffer: FnTargetMachineEmitToMemoryBuffer,
    get_buffer_start: FnGetBufferStart,
    get_buffer_size: FnGetBufferSize,
    dispose_memory_buffer: FnDisposeMemoryBuffer,
    dispose_module: FnDisposeModule,
    dispose_message: FnDisposeMessage,
    set_target: FnSetTarget,
    set_data_layout: FnSetDataLayout,
}

unsafe impl Send for LlvmCApi {}
unsafe impl Sync for LlvmCApi {}

impl LlvmCApi {
    fn load() -> Result<Self, CodegenError> {
        #[cfg(target_os = "windows")]
        let lib_name = "LLVM-C.dll";
        #[cfg(target_os = "macos")]
        let lib_name = "libLLVM.dylib";
        #[cfg(not(any(target_os = "windows", target_os = "macos")))]
        let lib_name = "libLLVM.so";

        let lib = (|| -> Result<Library, CodegenError> {
            // Try system PATH first (avoids DLL conflicts with TensorFlow/other loaded LLVMs)
            #[cfg(target_os = "windows")]
            {
                if let Ok(lib) = unsafe { Library::new(lib_name) } {
                    return Ok(lib);
                }
            }
            // Fall back to well-known install paths (prefer LLVM 20 over 17)
            #[cfg(target_os = "windows")]
            {
                let candidates = [
                    r"C:\llvm-20\bin\LLVM-C.dll",
                    r"C:\llvm-19\bin\LLVM-C.dll",
                    r"C:\llvm-18\bin\LLVM-C.dll",
                    r"C:\Program Files\LLVM\bin\LLVM-C.dll",
                ];
                for path in &candidates {
                    if std::path::Path::new(path).exists() {
                        return unsafe { Library::new::<&str>(path) }.map_err(|e| {
                            CodegenError::Unsupported {
                                backend: "llvm_c_api".into(),
                                detail: format!("failed to load '{}': {}", path, e),
                            }
                        });
                    }
                }
            }
            Err(CodegenError::Unsupported {
                backend: "llvm_c_api".into(),
                detail: format!("failed to load '{}' via PATH or any known install path. Is LLVM installed?", lib_name),
            })
        })()?;

        // Helper to load a function symbol
        macro_rules! load {
            ($lib:expr, $fn_ty:ty, $name:expr) => {{
                let sym: Symbol<$fn_ty> = unsafe {
                    $lib.get($name).map_err(|e| CodegenError::Unsupported {
                        backend: "llvm_c_api".into(),
                        detail: format!("symbol '{}' not found: {}", std::str::from_utf8($name).unwrap_or("?"), e),
                    })?
                };
                *sym // Deref to get the function pointer copy
            }};
        }

        Ok(Self {
            // Load all function pointers first (borrow `lib`), THEN move `lib` into `_lib`.
            get_global_context: load!(lib, FnGetGlobalContext, b"LLVMGetGlobalContext\0"),
            create_memory_buffer_with_memory_range_copy: load!(
                lib,
                FnCreateMemoryBufferWithMemoryRangeCopy,
                b"LLVMCreateMemoryBufferWithMemoryRangeCopy\0"
            ),
            parse_ir_in_context: load!(lib, FnParseIRInContext, b"LLVMParseIRInContext\0"),
            get_target_from_triple: load!(lib, FnGetTargetFromTriple, b"LLVMGetTargetFromTriple\0"),
            create_target_machine: load!(lib, FnCreateTargetMachine, b"LLVMCreateTargetMachine\0"),
            target_machine_emit_to_memory_buffer: load!(
                lib,
                FnTargetMachineEmitToMemoryBuffer,
                b"LLVMTargetMachineEmitToMemoryBuffer\0"
            ),
            get_buffer_start: load!(lib, FnGetBufferStart, b"LLVMGetBufferStart\0"),
            get_buffer_size: load!(lib, FnGetBufferSize, b"LLVMGetBufferSize\0"),
            dispose_memory_buffer: load!(lib, FnDisposeMemoryBuffer, b"LLVMDisposeMemoryBuffer\0"),
            dispose_module: load!(lib, FnDisposeModule, b"LLVMDisposeModule\0"),
            dispose_message: load!(lib, FnDisposeMessage, b"LLVMDisposeMessage\0"),
            set_target: load!(lib, FnSetTarget, b"LLVMSetTarget\0"),
            set_data_layout: load!(lib, FnSetDataLayout, b"LLVMSetDataLayout\0"),
            _lib: lib,
        })
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compile an LLVM IR string (`.ll` text) to a native object file.
///
/// `source_text` — the complete LLVM IR module as a string.
/// `output_path` — where to write the `.o` / `.obj` file.
/// `target_triple` — e.g. `"x86_64-w64-windows-gnu"`.
///   If `None`, detects the host target.
pub fn compile_llvm_ir_to_object(
    source_text: &str,
    output_path: &Path,
    target_triple: Option<&str>,
) -> Result<(), CodegenError> {
    let api = LlvmCApi::load()?;

    let triple = target_triple.unwrap_or_else(|| {
        if cfg!(target_os = "windows") {
            "x86_64-pc-windows-msvc"
        } else if cfg!(target_os = "macos") {
            "x86_64-apple-darwin"
        } else {
            "x86_64-unknown-linux-gnu"
        }
    });

    let context = unsafe { (api.get_global_context)() };
    let module = parse_ir(&api, context, source_text)?;

    // Set target triple
    let triple_c = CString::new(triple).map_err(|_| CodegenError::Unsupported {
        backend: "llvm_c_api".into(),
        detail: format!("invalid target triple: '{}'", triple),
    })?;
    unsafe {
        (api.set_target)(module, triple_c.as_ptr());
        // Let LLVM infer the data layout from the target triple
        (api.set_data_layout)(module, CString::new("").unwrap().as_ptr());
    }

    // Get target
    let mut target: LLVMTargetRef = ptr::null_mut();
    let mut target_error: *mut c_char = ptr::null_mut();
    let ok = unsafe { (api.get_target_from_triple)(triple_c.as_ptr(), &mut target, &mut target_error) };
    if !ok || target.is_null() {
        let err = if !target_error.is_null() {
            unsafe { CStr::from_ptr(target_error).to_string_lossy().into_owned() }
        } else {
            "unknown target".to_owned()
        };
        unsafe { (api.dispose_message)(target_error) };
        unsafe { (api.dispose_module)(module) };
        return Err(CodegenError::Unsupported {
            backend: "llvm_c_api".into(),
            detail: format!("unknown target triple '{}': {}", triple, err),
        });
    }

    // Create target machine
    let cpu = CString::new("generic").unwrap();
    let features = CString::new("").unwrap();
    let machine = unsafe {
        (api.create_target_machine)(target, triple_c.as_ptr(), cpu.as_ptr(), features.as_ptr(), 2, 1, 0)
    };
    if machine.is_null() {
        unsafe { (api.dispose_module)(module) };
        return Err(CodegenError::Unsupported {
            backend: "llvm_c_api".into(),
            detail: format!("failed to create target machine for '{}'", triple),
        });
    }

    // Emit object file to memory buffer
    let mut emit_error: *mut c_char = ptr::null_mut();
    let obj_buf = unsafe {
        (api.target_machine_emit_to_memory_buffer)(machine, module, 1, &mut emit_error)
    };
    if obj_buf.is_null() {
        let err = if !emit_error.is_null() {
            unsafe { CStr::from_ptr(emit_error).to_string_lossy().into_owned() }
        } else {
            "unknown error".to_owned()
        };
        unsafe { (api.dispose_message)(emit_error) };
        unsafe { (api.dispose_module)(module) };
        return Err(CodegenError::Unsupported {
            backend: "llvm_c_api".into(),
            detail: format!("failed to emit object code: {}", err),
        });
    }

    // Write object data to file
    let data_ptr = unsafe { (api.get_buffer_start)(obj_buf) };
    let data_len = unsafe { (api.get_buffer_size)(obj_buf) as usize };
    let data = unsafe { std::slice::from_raw_parts(data_ptr as *const u8, data_len) };
    std::fs::write(output_path, data).map_err(|e| CodegenError::Unsupported {
        backend: "llvm_c_api".into(),
        detail: format!("failed to write object file: {}", e),
    })?;

    unsafe {
        (api.dispose_memory_buffer)(obj_buf);
        (api.dispose_module)(module);
    }

    Ok(())
}

/// Check if the LLVM C API library is available on this system.
pub fn is_llvm_c_api_available() -> bool {
    #[cfg(target_os = "windows")]
    {
        let candidates = [
            r"C:\llvm-20\bin\LLVM-C.dll",
            r"C:\llvm-19\bin\LLVM-C.dll",
            r"C:\llvm-18\bin\LLVM-C.dll",
            r"C:\Program Files\LLVM\bin\LLVM-C.dll",
            "LLVM-C.dll",
        ];
        for path in &candidates {
            if unsafe { Library::new::<&str>(path) }.is_ok() {
                return true;
            }
        }
        return false;
    }
    #[cfg(target_os = "macos")]
    let lib_name = "libLLVM.dylib";
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let lib_name = "libLLVM.so";
    #[cfg(not(target_os = "windows"))]
    unsafe { Library::new(lib_name).is_ok() }
}

// ---------------------------------------------------------------------------
// Internal: parse LLVM IR text into a module
// ---------------------------------------------------------------------------

fn parse_ir(api: &LlvmCApi, context: LLVMContextRef, source: &str) -> Result<LLVMModuleRef, CodegenError> {
    let c_source = CString::new(source).map_err(|_| CodegenError::Unsupported {
        backend: "llvm_c_api".into(),
        detail: "LLVM IR source contains null byte".to_owned(),
    })?;
    let name = CString::new("module.ll").unwrap();

    let mem_buf = unsafe {
        (api.create_memory_buffer_with_memory_range_copy)(c_source.as_ptr(), source.len(), name.as_ptr())
    };
    if mem_buf.is_null() {
        return Err(CodegenError::Unsupported {
            backend: "llvm_c_api".into(),
            detail: "failed to create memory buffer".to_owned(),
        });
    }

    let mut module: LLVMModuleRef = ptr::null_mut();
    let mut parse_error: *mut c_char = ptr::null_mut();

    let failed = unsafe { (api.parse_ir_in_context)(context, mem_buf, &mut module, &mut parse_error) };
    unsafe { (api.dispose_memory_buffer)(mem_buf) };

    if failed {
        let err = if !parse_error.is_null() {
            unsafe { CStr::from_ptr(parse_error).to_string_lossy().into_owned() }
        } else {
            "parse failed".to_owned()
        };
        unsafe { (api.dispose_message)(parse_error) };
        return Err(CodegenError::Unsupported {
            backend: "llvm_c_api".into(),
            detail: format!("failed to parse LLVM IR: {}", err),
        });
    }

    Ok(module)
}
