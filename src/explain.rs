//! Diagnostic error explanations command for IRIS.
//!
//! Provides detailed explanations, examples, and fix instructions for IRIS compiler
//! and interpreter error codes (e.g. E0001, E0100).

pub fn explain(code: &str) -> Option<String> {
    let text = match code {
        "E0001" => {
            r#"
# E0001: Unexpected Character

## Explanation
The lexer encountered a character that is invalid or unrecognized in the IRIS language syntax.

## Example
```iris
def main() -> i64 {
    val x = @hello // Error: '@' is invalid in this context
}
```

## How to Fix
- Remove the unexpected character.
- If you were trying to write a comment, use `//` instead of `#`.
- Avoid using decorators/attributes like `@` unless in allowed `model` DSL contexts.
        "#
        }
        "E0002" => {
            r#"
# E0002: Unterminated String

## Explanation
A string literal was opened with a double quote `"` but was not closed before the end of the line or file.

## Example
```iris
def main() -> str {
    "hello world // Error: missing closing quote
}
```

## How to Fix
- Ensure all string literals have a matching closing double quote `"`.
        "#
        }
        "E0003" => {
            r#"
# E0003: Invalid Escape Sequence

## Explanation
An invalid escape sequence was found inside a string literal. IRIS only supports a specific set of escape sequences.

## Supported Escapes
- `\n` - Newline
- `\t` - Tab
- `\r` - Carriage return
- `\\` - Backslash
- `\"` - Double quote

## How to Fix
- Check escape sequences inside the string literal and replace any unsupported ones.
        "#
        }
        "E0005" => {
            r#"
# E0005: Unexpected Token

## Explanation
The compiler encountered a token that does not match the expected grammatical structure at this point in the program.

## Example
```iris
def main() i64 { // Error: missing '->' arrow
    42
}
```

## How to Fix
- Review the code syntax around the specified span.
- Ensure all functions have proper `-> ReturnType` annotations, and braces or parentheses are balanced.
        "#
        }
        "E0006" => {
            r#"
# E0006: Unexpected End of File

## Explanation
The file ended abruptly while the parser was still expecting more input. This is typically due to unclosed parentheses `()`, brackets `[]`, or braces `{}`.

## How to Fix
- Check your delimiters. Ensure every opening brace `{`, bracket `[`, and parenthesis `(` has a matching closing counterpart.
        "#
        }
        "E0100" => {
            r#"
# E0100: Undefined Variable

## Explanation
The compiler cannot find a variable or function definition with this name in the current scope.

## Example
```iris
def main() -> i64 {
    x + 1 // Error: 'x' is not defined
}
```

## How to Fix
- Check the variable spelling.
- Ensure the variable is declared using `val` or `var` before it is used.
- Ensure any functions you call are defined or brought into scope using `bring`.
        "#
        }
        "E0101" => {
            r#"
# E0101: Type Mismatch

## Explanation
An expression has a different type than what was expected by the surrounding context.

## Example
```iris
def main() -> i64 {
    "hello" // Error: expected i64, found str
}
```

## How to Fix
- Check that types match.
- Use explicit type casting/conversion helpers like `to_i64()`, `to_f64()`, or `to_str()` if needed.
        "#
        }
        "E0102" => {
            r#"
# E0102: Duplicate Function

## Explanation
A function with the same name has already been defined in this scope.

## How to Fix
- Rename one of the functions to avoid name collision.
- If they are in different modules, make sure they are in separate files.
        "#
        }
        "E0402" => {
            r#"
# E0402: Division By Zero

## Explanation
The interpreter or runtime environment attempted a division or modulo operation where the divisor was zero.

## Example
```iris
def main() -> i64 {
    let x = 10 / 0 // Runtime Error
}
```

## How to Fix
- Guard divisions with a conditional check to ensure the divisor is non-zero.
        "#
        }
        "E0403" => {
            r#"
# E0403: Index Out of Bounds

## Explanation
An array or list index operation was attempted with an index that is either negative or greater than or equal to the length of the collection.

## How to Fix
- Check that your index is within the valid range `0 <= index < len(collection)`.
        "#
        }
        "E0406" => {
            r#"
# E0406: Program Panicked

## Explanation
A runtime panic was explicitly triggered by a call to `panic()`, or an assertion failed.

## Example
```iris
def main() -> i64 {
    panic("explicit abort")
}
```

## How to Fix
- Inspect the panic message and stack trace to find why the abort was triggered.
        "#
        }
        _ => return None,
    };
    Some(text.trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explain_valid() {
        let e100 = explain("E0100");
        assert!(e100.is_some());
        assert!(e100.unwrap().contains("Undefined Variable"));

        let e402 = explain("E0402");
        assert!(e402.is_some());
        assert!(e402.unwrap().contains("Division By Zero"));
    }

    #[test]
    fn test_explain_invalid() {
        assert!(explain("E9999").is_none());
        assert!(explain("random").is_none());
    }
}
