import re
import os

book_path = "docs/BOOK.md"
with open(book_path, "r", encoding="utf-8") as f:
    orig_content = f.read()

# Perform simple global version replacements
content = orig_content
content = content.replace("0.2.0", "1.0.0-rc1")
content = content.replace("0.2.x", "1.0.0-rc1")

# Break by '## '
parts = re.split(r'\n## ', content)

# Title is in parts[0]
title = parts[0].strip()

guide = ""
foreword = ""
chapters = []
appendices = []

for part in parts[1:]:
    part_strip = part.strip()
    if part_strip.startswith("A Complete Guide"):
        guide = part
    elif part_strip.startswith("Table of Contents"):
        # We will dynamically generate the new TOC
        pass
    elif part_strip.startswith("Foreword"):
        foreword = part
    elif part_strip.startswith("Chapter "):
        chapters.append(part)
    elif part_strip.startswith("Appendix "):
        appendices.append(part)
    else:
        # Fallback
        chapters.append(part)

print(f"Parsed {len(chapters)} chapters and {len(appendices)} appendices.")

new_chapters = []

# ----------------- Chapter 1: Getting Started -----------------
ch1 = chapters[0] # "Chapter 1: Getting Started"
# Update VS Code extension version in Ch 1
ch1 = ch1.replace("iris-lang-1.0.0-rc1.vsix", "iris-lang-1.0.0-rc1.vsix")

# Add a section on 'iris pkg init' and 'iris test' to Chapter 1, right before 'Try It Yourself'
ch1_insert = """
### 1.6 Project Setup and Testing

Modern IRIS development utilizes the built-in package manager and testing framework:

1. **Initialize a Project**: Run `iris pkg init myproject` to create a standard project layout:
   ```bash
   iris pkg init myproject
   cd myproject
   ```
2. **Standard Structure**: This generates a `src/main.iris` file and a `iris.toml` manifest.
3. **Run Tests**: Check that everything is working:
   ```bash
   iris test
   ```

"""
ch1 = ch1.replace("### Try It Yourself", ch1_insert + "### Try It Yourself")
new_chapters.append(ch1)


# ----------------- Chapter 2: Values and Types -----------------
ch2 = chapters[1] # "Chapter 2: Values and Types"
# Fix float literal default type from f32 to f64
ch2 = ch2.replace("f64 | 64-bit floating-point | `3.14 to f64`, `1.0 to f64` |", "f64 | 64-bit floating-point | `3.14`, `1.0` |")
ch2 = ch2.replace("f32 | 32-bit floating-point | `3.14`, `1.0` |", "f32 | 32-bit floating-point | `3.14 to f32`, `1.0 to f32` |")
ch2 = ch2.replace("Float literals like `3.14` are `f32` by default. To get an `f64`, write `3.14 to f64`.", "Float literals like `3.14` are `f64` by default. To get an `f32`, write `3.14 to f32`.")
ch2 = ch2.replace("Integer literals are `i64` by default. Float literals are `f32` by default.", "Integer literals are `i64` by default. Float literals are `f64` by default.")
ch2 = ch2.replace("val f: f32 = 3.14", "val f: f64 = 3.14")
ch2 = ch2.replace("val big: f64 = f to f64", "val small: f32 = f to f32")
ch2 = ch2.replace("takes an `f32` and returns its square root as `f64`.", "takes an `f64` and returns its square root as `f32`.")
ch2 = ch2.replace("Forgetting that `3.14` is `f32`, not `f64`. If you pass it to a function expecting `f64`, you must write `3.14 to f64`.", "Forgetting that `3.14` is `f64`, not `f32`. If you pass it to a function expecting `f32`, you must write `3.14 to f32`.")

# Expand Ch 2 with missing types
ch2_scalar_insert = """
IRIS has eleven primitive scalar types:

| Type | Description | Example |
|------|-------------|---------|
| `i64` | 64-bit signed integer | `42`, `-7` |
| `i32` | 32-bit signed integer | `42`, `-7` |
| `i8`  | 8-bit signed integer | `42 to i8` |
| `u8`  | 8-bit unsigned integer | `255 to u8` |
| `u32` | 32-bit unsigned integer | `100 to u32` |
| `u64` | 64-bit unsigned integer | `100 to u64` |
| `usize`| Platform pointer size | `10 to usize` |
| `f64` | 64-bit floating-point | `3.14`, `1.0` |
| `f32` | 32-bit floating-point | `3.14 to f32`, `1.0 to f32` |
| `bool` | Boolean | `true`, `false` |
| `str` | String (UTF-8) | `"hello"` |
"""

target_scalar = """IRIS has five primitive scalar types:

| Type | Description | Example |
|------|-------------|---------|
| `i64` | 64-bit signed integer | `42`, `-7`, `0` |
| `i32` | 32-bit signed integer | `42`, `-7` |
| `f64` | 64-bit floating-point | `3.14 to f64`, `1.0 to f64` |
| `f32` | 32-bit floating-point | `3.14`, `1.0` |
| `bool` | Boolean | `true`, `false` |
| `str` | String (UTF-8) | `"hello"` |"""

ch2 = ch2.replace(target_scalar, ch2_scalar_insert.strip())
new_chapters.append(ch2)


# ----------------- Chapter 3: Functions -----------------
ch3 = chapters[2] # "Chapter 3: Functions"
new_chapters.append(ch3)


# ----------------- Chapter 4: Control Flow -----------------
ch4 = chapters[3] # "Chapter 4: Control Flow"
# Add for-each loops, tuple destructuring, and keyword operators (and, or, not)
ch4_insert = """
### 4.8 For-Each Loops

IRIS supports iterating directly over collections like lists, ranges, or arrays:

```iris
def main() -> i64 {
    val items = list();
    push(items, 10); push(items, 20); push(items, 30);
    
    // Iterate over elements directly
    for x in items {
        print(to_str(x))
    }
    0
}
```

### 4.9 Tuple Destructuring

You can bind multiple values at once by destructuring tuples:

```iris
def get_coords() -> (i64, i64) {
    (100, 200)
}

def main() -> i64 {
    val (x, y) = get_coords();
    print(concat("X: ", to_str(x)));
    print(concat("Y: ", to_str(y)));
    0
}
```

### 4.10 Keyword Operators (`and`, `or`, `not`)

In addition to `&&`, `||`, and `!`, IRIS supports readable keyword operators:

```iris
def eligible(age: i64, registered: bool) -> bool {
    age >= 18 and registered
}

def can_enter(has_ticket: bool, is_vip: bool) -> bool {
    has_ticket or is_vip
}

def is_minor(age: i64) -> bool {
    not (age >= 18)
}
```

"""
ch4 = ch4.replace("### Try It Yourself", ch4_insert + "### Try It Yourself")
new_chapters.append(ch4)


# ----------------- Chapter 5: Data Structures -----------------
ch5 = chapters[4] # "Chapter 5: Data Structures"
# Add deque, bitset, and mutex
ch5_insert = """
### 5.10 Deques

Deques are double-ended queues supporting efficient push/pop at both ends:

```iris
def deque_demo() -> i64 {
    val dq = deque_new();
    deque_push_back(dq, 20);
    deque_push_front(dq, 10);
    deque_push_back(dq, 30);
    
    print(to_str(deque_pop_front(dq))); // 10
    print(to_str(deque_pop_back(dq)));  // 30
    0
}
```

### 5.11 BitSets

BitSets provide compact, high-performance bit-array collections:

```iris
def bitset_demo() -> i64 {
    val bs = bitset_new();
    bitset_set(bs, 5, true);
    bitset_set(bs, 10, true);
    
    print(to_str(bitset_get(bs, 5)));   // true
    print(to_str(bitset_get(bs, 7)));   // false
    print(to_str(bitset_count(bs)));    // 2
    0
}
```

### 5.12 Mutexes

Mutexes provide thread-safe mutual exclusion for shared state:

```iris
def mutex_demo() -> i64 {
    val m = mutex(42);
    // Lock and modify inside spawn
    spawn {
        val val_ref = m; // Reference to same mutex
        // Locks are managed safely by built-ins
        0
    };
    0
}
```

"""
ch5 = ch5.replace("### Try It Yourself", ch5_insert + "### Try It Yourself")
new_chapters.append(ch5)


# ----------------- Chapter 6 (NEW): Traits and Generics -----------------
ch6_new = """Chapter 6: Traits and Generics

IRIS features a robust type system that supports traits and generics, allowing for clean code reuse, static polymorphism, and generic programming.

### 6.1 Trait Declarations

A **trait** defines a contract or interface of method signatures that types must satisfy:

```iris
trait Printable {
    def to_string(self: Self) -> str
}

trait Comparable {
    def compare(self: Self, other: Self) -> i64
}
```

The keyword `Self` (capitalized) inside a trait definition represents the type that will implement the trait.

### 6.2 Implementing Traits

Use the `impl` keyword to implement a trait for a concrete record type:

```iris
record Point {
    x: f64,
    y: f64,
}

impl Printable for Point {
    def to_string(self: Point) -> str {
        format("({}, {})", self.x, self.y)
    }
}
```

Once a trait is implemented, its methods can be called on instances of that type:

```iris
def main() -> i64 {
    val p = Point { x: 3.5, y: -2.0 };
    print(p.to_string());
    0
}
```

### 6.3 Generic Functions

Generic functions declare type parameters inside square brackets `[T]`:

```iris
def identity[T](x: T) -> T {
    x
}

def my_max[T](a: T, b: T) -> T {
    if a >= b { a } else { b }
}
```

Generics in IRIS are **monomorphized** at compile time, generating efficient concrete implementations for each type used.

### 6.4 Trait Constraints (`where`)

You can constrain generic parameters using the `where` keyword, enforcing that types must implement specific traits:

```iris
def print_item[T where T: Printable](x: T) -> i64 {
    print(x.to_string());
    0
}
```

### Try It Yourself

1. Define a trait `Area` with a method `area(self: Self) -> f64`.
2. Implement `Area` for `record Circle { radius: f64 }` and `record Rectangle { width: f64, height: f64 }`.
3. Write a generic function `print_area[T where T: Area](shape: T)` that calls `area` and prints the result.

"""
new_chapters.append(ch6_new)


# ----------------- Chapter 7: Closures and Higher-Order Functions -----------------
ch7 = chapters[5] # "Chapter 6: Closures and Higher-Order Functions"
new_chapters.append(ch7)


# ----------------- Chapter 8: String Processing -----------------
ch8 = chapters[6] # "Chapter 7: String Processing"
# Add regex, datetime, and hex literals
ch8_insert = """
### 7.7 Regular Expressions

IRIS features fast, compiled regular expressions built-in:

```iris
def regex_demo() -> i64 {
    val text = "Contact us at sales@example.com or support@example.com";
    val pattern = "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]\\\\.[a-zA-Z]{2,4}";
    
    // Find all matches
    val emails = regex_find_all(text, pattern);
    for email in emails {
        print(email)
    };
    
    // Replace regex match
    val masked = regex_replace(text, pattern, "[redacted]");
    print(masked);
    0
}
```

### 7.8 Date and Time

The datetime built-ins provide high-precision system time operations:

```iris
def datetime_demo() -> i64 {
    val now = datetime_now();
    print(concat("Timestamp: ", to_str(datetime_timestamp())));
    
    val formatted = datetime_format(now, "%Y-%m-%d %H:%M:%S");
    print(concat("Formatted: ", formatted));
    0
}
```

### 7.9 Hexadecimal Literals

You can write integer literals in hexadecimal notation using the `0x` prefix:

```iris
def hex_demo() -> i64 {
    val red = 0xFF0000;
    val green = 0x00FF00;
    val blue = 0x0000FF;
    
    print(concat("Red: ", to_str(red))); // 16711680
    0
}
```

"""
ch8 = ch8.replace("### Try It Yourself", ch8_insert + "### Try It Yourself")
new_chapters.append(ch8)


# ----------------- Chapter 9: Error Handling -----------------
ch9 = chapters[7] # "Chapter 8: Error Handling"
new_chapters.append(ch9)


# ----------------- Chapter 10: Concurrency -----------------
ch10 = chapters[8] # "Chapter 9: Concurrency"
# Add async/await and fix atomic_new -> atomic
ch10 = ch10.replace("atomic_new", "atomic")
ch10_insert = """
### 9.7 Async/Await

In addition to threads and channels, IRIS supports async/await for lightweight, cooperative multitasking:

```iris
async def slow_op() -> i64 {
    // asynchronous operation
    42
}

def main() -> i64 {
    // Calling async def returns an implicit Future/Promise
    val future = slow_op();
    
    // Await pauses execution until the future is ready
    val result = await future;
    print(to_str(result));
    0
}
```

"""
ch10 = ch10.replace("### Try It Yourself", ch10_insert + "### Try It Yourself")
new_chapters.append(ch10)


# ----------------- Chapter 11: Automatic Differentiation -----------------
ch11 = chapters[9] # "Chapter 10: Automatic Differentiation"
# Add reverse-mode AD
ch11_insert = """
### 10.5 Reverse-Mode Automatic Differentiation

For models with thousands of inputs, IRIS provides highly optimized reverse-mode AD (backpropagation) using a taped execution graph:

```iris
def main() -> i64 {
    // 1. Initialize variables on the AD tape
    val x = grad(2.0);
    val y = grad(3.0);
    
    // 2. Perform forward pass
    val z = (x * x) + (x * y);
    
    // 3. Compute gradients backwards
    backward(z);
    
    // 4. Retrieve gradients
    print(concat("dz/dx: ", to_str(grad_of(x)))); // 2*x + y = 7
    print(concat("dz/dy: ", to_str(grad_of(y)))); // x = 2
    0
}
```

This tape-based backpropagation integrates natively with IRIS ML and Tensor subsystems to enable deep learning workflows.

"""
ch11 = ch11.replace("### Try It Yourself", ch11_insert + "### Try It Yourself")
new_chapters.append(ch11)


# ----------------- Chapter 12: Tensors and ML -----------------
ch12 = chapters[10] # "Chapter 11: Tensors and ML"
# Add Model DSL
ch12_insert = """
### 11.9 Model DSL (Neural Network Architectures)

The declarative **Model DSL** simplifies deep learning model definitions:

```iris
model MLP {
    input x: tensor<f32, [batch, 784]>
    layer h1 Linear(x, in_features=784, out_features=128)
    layer a1 ReLU(h1)
    layer h2 Linear(a1, in_features=128, out_features=10)
    output h2
}
```

This model is compiled to highly efficient vector/SIMD instructions and exposes standard weights and biases for training.

### 11.10 Machine Learning Stdlib Modules (`std.ml`, `std.rl`, `std.nn`)

- **`std.nn`**: Offers high-level layers (Linear, Conv2D, RNN), activations (Softmax, Sigmoid), loss functions (CrossEntropy, MSE), and optimizers (SGD, Adam).
- **`std.ml`**: Traditional ML algorithms (k-Means, Gaussian Naive Bayes, k-NN, Linear/Logistic Regression).
- **`std.rl`**: Reinforcement learning framework featuring FIFO replay buffers, experience sampling, stable clipped PPO objectives, and GAE value baselines.

"""
ch12 = ch12.replace("### Try It Yourself", ch12_insert + "### Try It Yourself")
new_chapters.append(ch12)


# ----------------- Chapter 13: Native Compilation -----------------
ch13 = chapters[11] # "Chapter 12: Native Compilation"
new_chapters.append(ch13)


# ----------------- Chapter 14: The Standard Library -----------------
ch14 = chapters[12] # "Chapter 13: The Standard Library"
# Update module count to 34 and add modules
ch14 = ch14.replace("IRIS ships with 25 standard library modules", "IRIS ships with 34 standard library modules")
ch14 = ch14.replace("25 standard library modules", "34 standard library modules")

ch14_table = """| Module | Contents |
|--------|----------|
| `std.math` | `gcd`, `lcm`, `abs_i64`, `is_even`, `is_odd`, `factorial`, `pow_i64` |
| `std.string` | `pad_left`, `pad_right`, `words`, `lines`, `title_case`, `snake_case` |
| `std.fmt` | `sprintf`, `pad_int`, `zero_pad_int`, `format_table` |
| `std.fs` | `read_text`, `write_text`, `path_exists`, `file_lines` |
| `std.json` | `json_stringify`, `json_parse` |
| `std.csv` | `csv_parse_row`, `csv_emit_row` |
| `std.http` | `http_get`, `http_post` |
| `std.time` | `now`, `sleep`, `elapsed` |
| `std.stochastic` | `normal`, `brownian_path`, `gbm_path` |
| `std.crypto` | `sha256`, `uuid`, `hex_encode`, `hex_decode` |
| `std.ffi` | `ffi_open`, `ffi_call_*`, `python_*`, `rust_*` |
| `std.os` | `env_get`, `env_set`, `exec_cmd`, `pid`, `exit_code` |
| `std.testing` | `assert_eq`, `assert_ne`, `assert_true`, `assert_false`, `assert_str_eq` |
| `std.log` | `log_info`, `log_warn`, `log_error`, `log_debug` |
| `std.iter` | `map_list`, `filter_list`, `reduce_list`, `zip_list` |
| `std.set` | Set operations (add, remove, contains, union, intersect) |
| `std.queue` | FIFO queue (enqueue, dequeue, peek) |
| `std.heap` | Priority queue / min-heap (insert, extract_min) |
| `std.deque` | Double-ended queue (push_front, push_back, pop_front, pop_back) |
| `std.kv` | Key-value store (SQLite-backed persistent storage) |
| `std.table` | Tabular data operations |
| `std.dataset` | ML dataset abstraction (load, split, batch) |
| `std.dataframe` | DataFrame-like API |
| `std.path` | Path manipulation (join, parent, extension) |
| `std.async` | Async runtime helpers |
| `std.bitset` | Bit array operations (set, clear, test, count) |
| `std.tensorx` | Advanced tensor manipulation, slices, resizes |
| `std.rl` | Reinforcement learning (ppo_clip_loss, softmax_entropy) |
| `std.ais` | AI Systems and agents interfaces |
| `std.ml` | Traditional ML algorithms |
| `std.nn` | Neural Network layers and optimizers |
| `std.svg` | SVG visualization generation |
| `std.termplot` | Interactive terminal plotting |"""

ch14 = re.sub(r'\| Module \| Contents \|[\s\S]+?### 13.1 `std.math`', ch14_table + "\n\n### 13.1 `std.math`", ch14)

# Expand stub std.svg and std.termplot
ch14_svg_expand = """
### 13.13 `std.svg` & `std.termplot` — Visualizations

The `std.svg` module provides a simple, structured API to generate vector graphics:

```iris
bring std.svg

def generate_chart() -> i64 {
    val canvas = svg.canvas(800, 600);
    svg.rect(canvas, 50, 50, 700, 500, "fill: white; stroke: black; stroke-width: 2");
    svg.circle(canvas, 400, 300, 150, "fill: blue; opacity: 0.5");
    svg.text(canvas, 400, 80, "IRIS Standard Visualization", "font-size: 24px; text-anchor: middle");
    svg.save(canvas, "chart.svg");
    0
}
```

The `std.termplot` module provides instant inline charts directly in your terminal output:

```iris
bring std.termplot

def plot_live() -> i64 {
    val data = list();
    push(data, 1.2); push(data, 2.5); push(data, 3.1);
    push(data, 2.0); push(data, 4.5); push(data, 5.0);
    
    // Plots a clean Unicode line chart
    termplot.line(data, "Performance Profile");
    0
}
```

"""
ch14 = re.sub(r'### 13.13 `std.svg` \& `std.termplot`[\s\S]+?### 13.14', ch14_svg_expand.strip() + "\n\n### 13.14", ch14)
new_chapters.append(ch14)


# ----------------- Chapter 15: Tooling -----------------
ch15 = chapters[13] # "Chapter 14: Tooling"
# Expand Ch 15 with test, bench, profile, explain, fmt, lint, doc
ch15_insert = """
### 14.7 Subcommands & Tooling Suite

IRIS ships with a comprehensive set of developer utilities built directly into the main `iris` compiler binary:

- **`iris test`**: Automated test discovery and execution. Scans code for `@test` decorators or `test_` prefixed functions. Support test filtering:
  ```bash
  iris test --filter math
  ```
- **`iris bench`**: Benchmarking harness. Executes functions tagged with `@bench` multiple times to measure average runtime and memory allocation.
- **`iris profile`**: Runs a program and generates a performance flame graph:
  ```bash
  iris profile main.iris
  ```
- **`iris explain`**: Interactive error and diagnostic catalog. Explains compilation and runtime diagnostic codes with common causes and fixes:
  ```bash
  iris explain E4
  ```
- **`iris fmt`**: Self-contained code formatter. Rewrites `.iris` files to standard, idiomatic layouts.
- **`iris lint`**: Linter that analyzes code structures for performance and naming style issues.
- **`iris doc`**: Automatically extracts doc comments and generates Markdown/HTML API documentation.

"""
ch15 = ch15.replace("### Try It Yourself", ch15_insert + "### Try It Yourself")
new_chapters.append(ch15)


# ----------------- Chapter 16: Building Real Programs -----------------
ch16 = chapters[14] # "Chapter 15: Building Real Programs"
ch16 = ch16.replace("IRIS does not have a built-in build system or package manager", "IRIS includes a robust built-in package manager called `iris pkg`")
new_chapters.append(ch16)


# ----------------- Chapter 17 (NEW): Package Manager -----------------
ch17_new = """Chapter 17: Package Manager

IRIS includes a production-grade package manager and build tool built directly into the CLI as the `iris pkg` subcommand.

### 17.1 Initializing a Project

Create a new structured IRIS package with:

```bash
iris pkg init my_project
```

This creates the standard project layout:
```text
my_project/
├── iris.toml     # Manifest file
├── iris.lock     # Lockfile (generated on build)
└── src/
    └── main.iris # Entry point
```

### 17.2 The `iris.toml` Manifest

The manifest defines package metadata and third-party dependencies:

```toml
[package]
name = "my_project"
version = "1.0.0-rc1"
authors = ["Moon9t"]

[dependencies]
http_utils = { git = "https://github.com/iris-lang/http_utils.git", tag = "v1.2.0" }
json_helper = { path = "../json_helper" }
```

### 17.3 Managing Dependencies

Add dependencies easily using the CLI:

```bash
iris pkg add http_utils --git https://github.com/iris-lang/http_utils.git
```

This automatically downloads, validates, and adds the dependency to your `iris.toml`. 

### 17.4 Package Subcommands

- **`iris pkg build`**: Resolves dependencies, compiles them, and builds the current package.
- **`iris pkg run`**: Compiles and runs the package entry point.
- **`iris pkg update`**: Updates lockfile and checks for newer compatible dependency versions.
- **`iris pkg list`**: Lists all active project dependencies.
- **`iris pkg check`**: Rapidly parses and checks package types without full compilation.

### Try It Yourself

1. Run `iris pkg init calc_project` to initialize a new package.
2. Edit `iris.toml` to set yourself as the author.
3. Build and run it using `iris pkg run`.

"""
new_chapters.append(ch17_new)


# ----------------- Chapter 18: Working with Databases -----------------
ch18 = chapters[15] # "Chapter 16: Working with Databases"
new_chapters.append(ch18)


# ----------------- Chapter 19: Foreign Function Interface -----------------
ch19 = chapters[16] # "Chapter 17: Foreign Function Interface"
new_chapters.append(ch19)


# ----------------- Chapter 20: Networking -----------------
ch20 = chapters[17] # "Chapter 18: Networking"
# Add UDP section
ch20_insert = """
### 18.11 UDP Networking

For low-latency communication, IRIS features high-performance UDP socket support:

```iris
def udp_demo() -> i64 {
    // Open a UDP socket bound to local port 8080
    val socket = udp_open("127.0.0.1:8080");
    
    // Send a datagram
    udp_send(socket, "127.0.0.1:8081", "Ping");
    
    // Receive a datagram (blocks until received)
    val result = udp_recv(socket);
    print(concat("From: ", result.0)); // Sender address
    print(concat("Data: ", result.1)); // Payload
    
    udp_close(socket);
    0
}
```

"""
ch20 = ch20.replace("### Try It Yourself", ch20_insert + "### Try It Yourself")
new_chapters.append(ch20)


# ----------------- Chapter 21 (NEW): Security & Sandboxing -----------------
ch21_new = """Chapter 21: Security & Sandboxing

IRIS provides enterprise-grade runtime sandboxing capabilities to execute untrusted code safely.

### 21.1 The Sandbox Flag

By running the compiler with the `--sandbox` flag, the IRIS C runtime restricts access to operating system capabilities:

```bash
iris run --sandbox untrusted_script.iris
```

### 21.2 Restricted Operations

When running in sandbox mode, the following operations are strictly blocked and cause an immediate runtime panic:

- **Filesystem**: File read/write operations outside designated whitelist directories are rejected.
- **Networking**: Unauthorized outbound TCP/UDP connections or inbound listening sockets are denied.
- **Processes**: System command execution (`exec_cmd`, `pid`) is blocked.
- **FFI**: Foreign Function Interface modules (`std.ffi`, `ffi_open`) are disabled to prevent bypassing sandbox rules.

### 21.3 Customizing Whitelists

You can grant selective access to resources using sandbox flags:

```bash
iris run --sandbox --allow-read ./data/ --allow-net api.example.com script.iris
```

### Try It Yourself

1. Write a script `test_sec.iris` that attempts to read `/etc/passwd` or `C:\\\\Windows\\\\system.ini`.
2. Run it without flags: `iris run test_sec.iris`.
3. Run it with the sandbox flag: `iris run --sandbox test_sec.iris` and observe the sandbox denial panic.

"""
new_chapters.append(ch21_new)


# Let's dynamically renumber all chapters in order!
renumbered_chapters = []
for idx, ch in enumerate(new_chapters):
    # Determine new chapter number (1-based index)
    ch_num = idx + 1
    # Replace the main chapter heading "Chapter X:"
    ch = re.sub(r'^(Chapter|Chapter \d+):', f'Chapter {ch_num}:', ch, flags=re.MULTILINE)
    # Let's replace subheadings like "X.Y" with "ch_num.Y"
    # First, let's parse the actual first heading in the chapter to find its original chapter number, e.g. "### 5.2" or "### 18.4"
    orig_nums = re.findall(r'### (\d+)\.(\d+)', ch)
    if orig_nums:
        orig_ch_num = orig_nums[0][0]
        # Replace all instances of orig_ch_num.sub_num with ch_num.sub_num
        ch = ch.replace(f"### {orig_ch_num}.", f"### {ch_num}.")
    renumbered_chapters.append(ch)

print(f"Renumbered {len(renumbered_chapters)} chapters.")


# ----------------- Appendices Updates -----------------
# We will reconstruct/update the appendices carefully.

app_a_new = """Appendix A: Language Grammar (BNF)

```bnf
module      ::= { top_level }
top_level   ::= function_def
              | record_def
              | enum_def
              | const_def
              | type_alias
              | trait_def
              | impl_def
              | bring_decl
              | extern_def
              | model_def

bring_decl  ::= "bring" bring_path
bring_path  ::= IDENT { "." IDENT }
              | STRING_LIT

function_def ::= [ "pub" ] [ "async" ] "def" IDENT [ type_params ] "(" params ")" "->" type block
type_params  ::= "[" IDENT { "," IDENT } "]"
params       ::= [ param { "," param } ]
param        ::= IDENT ":" type [ "=" expr ]

record_def  ::= [ "pub" ] "record" IDENT "{" field_defs "}"
field_defs  ::= field_def { "," field_def }
field_def   ::= IDENT ":" type

enum_def    ::= [ "pub" ] "choice" IDENT "{" variant_defs "}"
variant_defs ::= variant_def { "," variant_def }
variant_def  ::= IDENT [ "(" type { "," type } ")" ]

const_def   ::= [ "pub" ] "const" IDENT [ ":" type ] "=" expr

type_alias  ::= [ "pub" ] "type" IDENT "=" type

trait_def   ::= "trait" IDENT "{" { trait_method } "}"
trait_method ::= "def" IDENT "(" params ")" "->" type

impl_def    ::= "impl" IDENT "for" IDENT "{" { function_def } "}"

extern_def  ::= "extern" "def" IDENT "(" params ")" "->" type

model_def   ::= "model" IDENT "{" { model_item } "}"
model_item  ::= "input" IDENT ":" type
              | "layer" IDENT IDENT [ "(" layer_args ")" ]
              | "output" IDENT
layer_args  ::= layer_arg { "," layer_arg }
layer_arg   ::= IDENT "=" expr | IDENT

(* Statements *)
block       ::= "{" { stmt } [ expr ] "}"
stmt        ::= let_stmt
              | assign_stmt
              | while_stmt
              | loop_stmt
              | for_stmt
              | par_for_stmt
              | spawn_stmt
              | return_stmt
              | break_stmt
              | continue_stmt
              | expr ";"

let_stmt    ::= "val" IDENT [ ":" type ] "=" expr ";"
              | "var" IDENT [ ":" type ] "=" expr ";"
              | "val" "(" IDENT { "," IDENT } ")" "=" expr ";"
assign_stmt ::= expr "=" expr ";"
while_stmt  ::= "while" expr block
loop_stmt   ::= "loop" block
for_stmt    ::= "for" IDENT "in" expr ".." expr block
              | "for" IDENT "in" expr block
par_for_stmt ::= "par" "for" IDENT "in" expr ".." expr block
spawn_stmt  ::= "spawn" block
return_stmt ::= "return" [ expr ] ";"
break_stmt  ::= "break" ";"
continue_stmt ::= "continue" ";"

(* Expressions — from lowest to highest precedence *)
expr        ::= or_expr
or_expr     ::= and_expr { "||" and_expr }
and_expr    ::= cmp_expr { "&&" cmp_expr }
cmp_expr    ::= add_expr { ( "==" | "!=" | "<" | "<=" | ">" | ">=" ) add_expr }
add_expr    ::= mul_expr { ( "+" | "-" ) mul_expr }
mul_expr    ::= cast_expr { ( "*" | "/" | "%" ) cast_expr }
cast_expr   ::= unary_expr [ "to" type ]
unary_expr  ::= [ "-" | "!" ] postfix_expr
postfix_expr ::= primary { "." IDENT [ "(" args ")" ] | "." INT_LIT | "[" args "]" | "?" }

primary     ::= INT_LIT
              | FLOAT_LIT
              | BOOL_LIT
              | STRING_LIT
              | FSTRING_LIT
              | IDENT [ "::" IDENT ] [ "(" args ")" ]
              | IDENT "{" field_inits "}"
              | "(" expr { "," expr } ")"
              | "[" [ expr { "," expr } ] "]"
              | "|" params "|" expr
              | "if" expr block [ "else" block ]
              | "when" expr "{" when_arms "}"
              | "await" expr
              | block

args        ::= [ expr { "," expr } ]
field_inits ::= [ IDENT ":" expr { "," IDENT ":" expr } ]

when_arms   ::= when_arm { "," when_arm }
when_arm    ::= pattern [ "if" expr ] "=>" expr
pattern     ::= IDENT "." IDENT [ "(" bindings ")" ]
              | "some" "(" IDENT ")"
              | "none"
              | "ok" "(" IDENT ")"
              | "err" "(" IDENT ")"
              | INT_LIT [ "..=" INT_LIT ]
              | BOOL_LIT
              | STRING_LIT
              | "(" pattern { "," pattern } ")"
              | "_"
bindings    ::= [ IDENT { "," IDENT } ]

(* Types *)
type        ::= scalar_type
              | "tensor" "<" scalar_type "," "[" dims "]" ">"
              | "option" "<" type ">"
              | "result" "<" type "," type ">"
              | "channel" "<" type ">"
              | "atomic" "<" type ">"
              | "mutex" "<" type ">"
              | "grad" "<" type ">"
              | "sparse" "<" type ">"
              | "list" "<" type ">"
              | "map" "<" type "," type ">"
              | "[" type ";" INT_LIT "]"
              | "(" type { "," type } ")"
              | "(" [ type { "," type } ] ")" "->" type
              | IDENT  (* named struct/enum/alias *)

scalar_type ::= "i8" | "u8" | "i32" | "u32" | "i64" | "u64" | "usize"
              | "f32" | "f64" | "bool" | "str"

dims        ::= dim { "," dim }
dim         ::= INT_LIT | IDENT
```
"""

app_b_new = """Appendix B: Built-in Functions Reference

IRIS provides a complete catalog of built-in functions available globally in all modules without any imports:

### Math
`sin`, `cos`, `tan`, `exp`, `log`, `log2`, `sqrt`, `abs`, `floor`, `ceil`, `round`, `sign`, `pow`, `min`, `max`, `clamp`, `math_pi`, `math_e`, `math_inf`, `is_nan`, `is_inf`

### String
`len`, `concat`, `contains`, `starts_with`, `ends_with`, `to_upper`, `to_lower`, `trim`, `repeat`, `to_str`, `format`, `split`, `join`, `find`, `slice`, `str_index`, `str_replace`, `str_reverse`, `char_at`, `str_pad_left`, `str_pad_right`, `str_chars`, `str_bytes`, `str_count`

### Bitwise
`band(a, b)`, `bor(a, b)`, `bxor(a, b)`, `shl(a, n)`, `shr(a, n)`, `bitnot(a)`

### I/O
`print`, `read_line`, `read_i64`, `read_f64`

### Collections
- **List**: `list`, `push`, `pop`, `list_get`, `list_set`, `list_len`, `list_pop`, `list_map`, `list_filter`, `list_reduce`, `list_any`, `list_all`, `list_zip`, `list_enumerate`, `list_flatten`, `list_unique`, `list_reverse`, `list_sorted`, `list_sum`, `list_min`, `list_max`
- **Map**: `map`, `map_get`, `map_set`, `map_contains`, `map_remove`, `map_keys`, `map_values`, `map_len`
- **Deque**: `deque_new`, `deque_push_front`, `deque_push_back`, `deque_pop_front`, `deque_pop_back`, `deque_len`, `deque_front`, `deque_back`
- **BitSet**: `bitset_new`, `bitset_set`, `bitset_get`, `bitset_count`, `bitset_clear`

### Reference Cells
`cell(v)`, `cell_get(c)`, `cell_set(c, v)`

### Option & Result
`some`, `none`, `is_some`, `unwrap`, `ok`, `err`, `is_ok`, `is_err`

### Parsing & Regex
`parse_i64`, `parse_f64`, `json_stringify`, `regex_match`, `regex_find_all`, `regex_replace`

### Concurrency
`channel`, `send`, `recv`, `spawn`, `chan_try_recv`, `chan_len`, `select`, `timeout`, `thread_count`, `atomic`, `atomic_load`, `atomic_store`, `atomic_add`

### Date & Time
`datetime_now`, `datetime_timestamp`, `datetime_format`

### OS & System
`cwd`, `list_dir`, `mkdir`, `remove_file`, `path_join`, `env_get`, `env_set`, `exec_cmd`, `pid`, `exit_code`, `type_of`

### Random & Cryptography
`random`, `random_range`, `uuid`, `sha256`, `hash`, `hex_encode`, `hex_decode`, `base64_encode`, `base64_decode`

### TCP & UDP Networking
`tcp_connect`, `tcp_listen`, `tcp_accept`, `tcp_read`, `tcp_write`, `tcp_close`, `udp_open`, `udp_send`, `udp_recv`, `udp_close`

### Terminal Controls
`read_key`, `read_password`, `term_clear`, `term_cursor`, `term_show_cursor`, `term_set_color`, `term_reset`, `term_rows`, `term_cols`
"""

app_c_new = """Appendix C: Type System Reference

### Scalar Types
- **Integers**: `i8` (8-bit signed), `u8` (8-bit unsigned), `i32` (32-bit signed), `u32` (32-bit unsigned), `i64` (64-bit signed), `u64` (64-bit unsigned), `usize` (pointer-sized unsigned)
- **Floats**: `f32` (32-bit single precision), `f64` (64-bit double precision)
- **Booleans**: `bool` (`true`, `false`)
- **Strings**: `str` (UTF-8 immutable sequence)

### Composite Types
- **Tensors**: `tensor<scalar_type, [dimensions]>`
- **Lists**: `list<T>`
- **Maps**: `map<K, V>`
- **Deques**: `deque`
- **BitSets**: `bitset`
- **Mutexes**: `mutex<T>`
- **Channels**: `channel<T>`
- **Reference Cells**: `cell<T>`
- **Automatic Differentiation**: `grad<T>`
- **Sparse Tensors**: `sparse<T>`

### Operator Precedence (highest to lowest)

| Precedence | Category | Operators | Associativity |
|------------|----------|-----------|---------------|
| 1 (highest) | Postfix | `.field` `.method()` `[index]` `?` | Left |
| 2 | Prefix | `-` (negate) `!` (not) | Right |
| 3 | Multiplicative | `*` `/` `%` | Left |
| 4 | Additive | `+` `-` | Left |
| 5 | Cast | `to` | Left |
| 6 | Comparison | `==` `!=` `<` `<=` `>` `>=` | Left, non-chaining |
| 7 | Logical AND | `&&` | Left, short-circuit |
| 8 (lowest) | Logical OR | `||` | Left, short-circuit |
"""

app_d_new = """Appendix D: CLI Reference

The `iris` compiler provides a single, unified CLI with 10 powerful subcommands:

### 10 Subcommands
1. **`build <file.iris>`**: Compiles an IRIS source file into a native binary.
2. **`run <file.iris>`**: Compiles and executes an IRIS program directly.
3. **`repl`**: Starts the interactive REPL shell.
4. **`lsp`**: Launches the background LSP Language Server.
5. **`dap`**: Launches the Debug Adapter Protocol server.
6. **`pkg`**: Package manager operations (init, build, run, add, update, list, check).
7. **`bench <file.iris>`**: Runs benchmarks tagged with `@bench`.
8. **`profile <file.iris>`**: Runs the compiler profiler and outputs execution flame graphs.
9. **`test`**: Discovers and runs test cases in the workspace.
10. **`explain <code>`**: Interactive diagnostic code explanation catalog.

### 14 Emit Kinds (`--emit <kind>`)
Specify intermediate compiler outputs:
- **`eval`**: Direct evaluation in the AST interpreter.
- **`tokens`**: Prints lexical tokens.
- **`ast`**: Prints structural Abstract Syntax Tree.
- **`ir`**: Prints text SSA Intermediate Representation.
- **`ir-opt`**: SSA IR after optimization passes.
- **`llvm`**: Text LLVM Assembly.
- **`bc`**: LLVM Bitcode file.
- **`asm`**: Target assembly code.
- **`obj`**: Compiled object file.
- **`binary`**: Native executable file.
- **`onnx`**: Exported ONNX model graph.
- **`cuda`**: Generated CUDA source code.
- **`simd`**: Vectorized IR output.
- **`graph`**: Generates AST or IR visual dependency dot files.

### Global Flags
- **`--sandbox`**: Strict runtime sandboxing.
- **`--target <triple>`**: Cross-compilation target.
- **`--no-cache`**: Disables AST and LLVM caching.
- **`--dump-ir-after <pass>`**: Dumps compiler state after specific optimizer pass.
"""

app_e_new = """Appendix E: Compiler Error Reference

IRIS has a detailed diagnostic code system cross-referenced directly with the `iris explain` command.

### Diagnostic Code Catalog
- **`E1: Missing else branch`**: Every `if` expression must have a matching `else` block to guarantee a returned value.
- **`E2: Missing semicolon after non-tail statement`**: Semicolons are required to separate non-tail statements in blocks.
- **`E3: Reassigning an immutable binding`**: Attempting to reassign a `val` binding instead of a `var` binding.
- **`E4: Type mismatch in binary operation`**: Operators require both operands to have the same type. IRIS does not perform implicit type casting.
- **`E5: Float literal type`**: Floating-point literal mismatch. Remember that float literals are `f64` by default.
- **`E6: Calling unwrap on none`**: Unsafely calling `unwrap` on an option that contains `none`. Always check with `is_some()`.
- **`E7: Operator precedence with comparison`**: Parsing error because operators like `+` have different precedence relative to comparisons.
- **`E8: find result used as number`**: Attempting to use the `option<i64>` returned by `find` directly in arithmetic.
- **`E9: Function not exported`**: Calling a function from another module that has not been marked with `pub`.
- **`E10: Using % modulo vs / division`**: Diagnostic error checking division operators.
"""


# ----------------- Reconstruct the Book -----------------

# Dynamic Table of Contents Generation
toc_lines = [
    "## Table of Contents",
    "",
    "- [Foreword](#foreword)",
]

# We will extract headings and build the TOC dynamically
for idx, ch in enumerate(renumbered_chapters):
    ch_num = idx + 1
    # Extract main title of the chapter
    lines = ch.strip().split("\n")
    ch_title = lines[0].replace(f"Chapter {ch_num}:", "").strip()
    
    # URL safe anchor
    ch_anchor = "chapter-" + str(ch_num) + "-" + ch_title.lower().replace(" ", "-").replace(":", "").replace("&", "").replace(",", "")
    toc_lines.append(f"- [Chapter {ch_num}: {ch_title}](#{ch_anchor})")
    
    # Sub-headings
    for line in lines:
        if line.startswith("### "):
            sub_title = line.replace("### ", "").strip()
            # E.g. "6.1 Trait Declarations"
            sub_anchor = sub_title.lower().replace(".", "").replace(" ", "-").replace(":", "").replace("(", "").replace(")", "").replace(",", "")
            # Anchor should only keep alphanumeric and dashes
            sub_anchor = re.sub(r'[^a-z0-9\-]', '', sub_anchor)
            # Indent under chapter
            toc_lines.append(f"  - [{sub_title}](#{sub_anchor})")

toc_lines.append("- [Appendix A: Language Grammar (BNF)](#appendix-a-language-grammar-bnf)")
toc_lines.append("- [Appendix B: Built-in Functions Reference](#appendix-b-built-in-functions-reference)")
toc_lines.append("- [Appendix C: Type System Reference](#appendix-c-type-system-reference)")
toc_lines.append("- [Appendix D: CLI Reference](#appendix-d-cli-reference)")
toc_lines.append("- [Appendix E: Compiler Error Reference](#appendix-e-compiler-error-reference)")

toc_content = "\n".join(toc_lines)

# Reassemble everything!
final_book_content = """---
title: "The IRIS Programming Language"
author: "Moon9t"
rights: "GPL-2.0-or-later"
language: "en-US"
toc: true
toc-depth: 2
number-sections: true
geometry: margin=1in
---

# The IRIS Programming Language

## A Complete Guide

\\newpage

## Copyright & License

**The IRIS Programming Language: A Complete Guide**

Copyright \\u00a9 2024-2026 Moon9t. All rights reserved.

This book is licensed under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.

This documentation is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this book. If not, see <https://www.gnu.org/licenses/>.

\\newpage

""" + toc_content + "\n\n## " + foreword + "\n"

for ch in renumbered_chapters:
    final_book_content += "\n## " + ch + "\n"

# Appendices
final_book_content += "\n## " + app_a_new + "\n"
final_book_content += "\n## " + app_b_new + "\n"
final_book_content += "\n## " + app_c_new + "\n"
final_book_content += "\n## " + app_d_new + "\n"
final_book_content += "\n## " + app_e_new + "\n"

# Let's fix the footer version at the very end
final_book_content += """
---

**Version**: Corresponds to IRIS compiler version 1.0.0-rc1
**Platform**: Tested on Windows 10/11, Linux (x86_64), macOS (aarch64) with LLVM 17+ and MinGW ucrt64
**License**: GNU General Public License v2.0 or later — see [LICENSE](LICENSE)
**Source**: [github.com/moon9t/iris](https://github.com/moon9t/iris)
"""

with open(book_path, "w", encoding="utf-8") as f:
    f.write(final_book_content)

print(f"Successfully generated new book.md at {book_path}!")
