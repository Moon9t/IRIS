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
        _ => None,
    }
}
