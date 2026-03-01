// Cargo build script — ensures the embedded C runtime files
// trigger a rebuild when they change (include_str! alone is not
// tracked by Cargo's dependency fingerprinting).

fn main() {
    println!("cargo:rerun-if-changed=src/runtime/iris_runtime.c");
    println!("cargo:rerun-if-changed=src/runtime/iris_runtime.h");
}
