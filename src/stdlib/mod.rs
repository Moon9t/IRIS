//! IRIS standard library registry.
//!
//! Stdlib modules are embedded as source strings via `include_str!`.
//! Use `stdlib_source("name")` to retrieve the IRIS source for a module.

/// Returns the IRIS source for the named stdlib module, or `None` if unknown.
pub fn stdlib_source(name: &str) -> Option<&'static str> {
    match name {
        "math" => Some(include_str!("math.iris")),
        "string" => Some(include_str!("string.iris")),
        "fmt" => Some(include_str!("fmt.iris")),
        "set" => Some(include_str!("set.iris")),
        "queue" => Some(include_str!("queue.iris")),
        "heap" => Some(include_str!("heap.iris")),
        "time" => Some(include_str!("time.iris")),
        "path" => Some(include_str!("path.iris")),
        "fs" => Some(include_str!("fs.iris")),
        "json" => Some(include_str!("json.iris")),
        "csv" => Some(include_str!("csv.iris")),
        "http" => Some(include_str!("http.iris")),
        "kv" => Some(include_str!("kv.iris")),
        "table" => Some(include_str!("table.iris")),
        "sql" => Some(include_str!("sql.iris")),
        "dataset" => Some(include_str!("dataset.iris")),
        "dataframe" => Some(include_str!("dataframe.iris")),
        // Phase 105: New stdlib modules
        "iter" => Some(include_str!("iter.iris")),
        "deque" => Some(include_str!("deque.iris")),
        "bitset" => Some(include_str!("bitset.iris")),
        "crypto" => Some(include_str!("crypto.iris")),
        "os" => Some(include_str!("os.iris")),
        "ffi" => Some(include_str!("ffi.iris")),
        "async" => Some(include_str!("async.iris")),
        "testing" => Some(include_str!("testing.iris")),
        "log" => Some(include_str!("log.iris")),
        // ML / AI modules
        "ml" => Some(include_str!("ml.iris")),
        "nn" => Some(include_str!("nn.iris")),
        "tensor" | "tensorx" => Some(include_str!("tensor.iris")),
        // Networking
        "http_server" => Some(include_str!("http_server.iris")),
        _ => None,
    }
}
