//! IRIS standard library registry.
//!
//! Stdlib modules are embedded as source strings via `include_str!`.
//! Use `stdlib_source("name")` to retrieve the IRIS source for a module.

/// Returns the IRIS source for the named stdlib module, or `None` if unknown.
pub fn stdlib_source(name: &str) -> Option<&'static str> {
    match name {
        "math"   => Some(include_str!("math.iris")),
        "string" => Some(include_str!("string.iris")),
        "fmt"    => Some(include_str!("fmt.iris")),
        "set"    => Some(include_str!("set.iris")),
        "queue"  => Some(include_str!("queue.iris")),
        "heap"   => Some(include_str!("heap.iris")),
        "time"   => Some(include_str!("time.iris")),
        "path"   => Some(include_str!("path.iris")),
        "fs"     => Some(include_str!("fs.iris")),
        "json"   => Some(include_str!("json.iris")),
        "csv"    => Some(include_str!("csv.iris")),
        "http"   => Some(include_str!("http.iris")),
        "kv"        => Some(include_str!("kv.iris")),
        "table"     => Some(include_str!("table.iris")),
        "dataset"   => Some(include_str!("dataset.iris")),
        "dataframe" => Some(include_str!("dataframe.iris")),
        _ => None,
    }
}
