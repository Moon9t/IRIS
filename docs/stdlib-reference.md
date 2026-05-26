# IRIS Standard Library Reference

All stdlib modules are imported with `bring std.<module>`:

```iris
bring std.math
bring std.string
bring std.fs
```

---

## math

Numeric utilities implemented in pure IRIS.

```iris
bring std.math

val g = gcd(12, 8)           // 4
val l = lcm(4, 6)            // 12
val a = abs_i64(-5)          // 5
val c = clamp_i64(15, 0, 10) // 10
val e = is_even(4)           // true
val o = is_odd(3)            // true
```

**Functions:** `gcd`, `lcm`, `abs_i64`, `clamp_i64`, `min_i64`, `max_i64`, `sign_i64`, `is_even`, `is_odd`

---

## string

String manipulation utilities.

```iris
bring std.string

val ws  = words("hello world iris")   // list["hello","world","iris"]
val ls  = lines("a\nb\nc")           // list["a","b","c"]
val j   = str_join(ls, ", ")         // "a, b, c"
val e   = is_empty("")               // true
val pl  = pad_left("42", 5, "0")     // "00042"
val pr  = pad_right("hi", 5, ".")    // "hi..."
val rep = str_repeat("ab", 3)        // "ababab"
val tl  = trim_start("  hello")      // "  hello" (approximation)
```

**Functions:** `words`, `lines`, `str_join`, `is_empty`, `pad_left`, `pad_right`, `str_repeat`, `trim_start`, `trim_end`

---

## fmt

Text formatting helpers.

```iris
bring std.fmt

val s = pad_int(7, 4)          // "   7"
val z = zero_pad_int(7, 4)     // "0007"
val l = left_align("hi", 6)    // "hi    "
val r = right_align("hi", 6)   // "    hi"
```

**Functions:** `pad_int`, `zero_pad_int`, `left_align`, `right_align`

---

## fs

File system I/O.

```iris
bring std.fs

val text = read_text("data.txt")          // "" on error
val ok   = write_text("out.txt", "hello") // true on success
val ok2  = append_text("log.txt", "line\n")
val ex   = path_exists("file.iris")       // bool
val lns  = read_lines("data.txt")         // list<str>
val ok3  = copy_file("src.txt", "dst.txt")
```

**Functions:** `read_text`, `write_text`, `append_text`, `path_exists`, `read_lines`, `copy_file`

---

## path

Pure-IRIS path manipulation (no OS calls).

```iris
bring std.path

val b = basename("/home/user/file.txt")  // "file.txt"
val d = dirname("/home/user/file.txt")   // "/home/user"
val e = extension("report.pdf")          // "pdf"
val s = stem("report.pdf")              // "report"
val j = join_path("/home/user", "docs") // "/home/user/docs"
```

**Functions:** `basename`, `dirname`, `extension`, `stem`, `join_path`

---

## time

Timing and stopwatch utilities.

```iris
bring std.time

val ms  = now_ms()              // milliseconds since epoch
val s   = now_s()               // seconds since epoch
val t0  = stopwatch_start()
// ... work ...
val dur = stopwatch_stop(t0)    // elapsed ms
val fmt = format_duration(dur)  // e.g. "1.23s" or "450ms"
sleep(100)                      // sleep 100ms
```

**Functions:** `now_ms`, `now_s`, `sleep`, `elapsed_ms`, `stopwatch_start`, `stopwatch_stop`, `format_duration`

---

## stochastic

Stochastic calculus helpers for Brownian motion and related processes.

```iris
bring std.stochastic

val z = normal()
val incs = brownian_increments(3, 0.1)
val path = brownian_path(3, 0.1)
val gbm = gbm_path(3, 0.1, 1.0, 0.05, 0.2)
```

**Functions:** `normal_pair`, `normal`, `normal_mu_sigma`, `brownian_step`, `brownian_increments`, `brownian_path`, `brownian_motion`, `gbm_path`

---

## tensorx

Dense tensor helpers over flat `list<f64>` storage plus shape metadata.

```iris
bring std.tensorx

val shape = list();
val _ = list_push(shape, 2);
val _ = list_push(shape, 2);
val data = list();
val _ = list_push(data, 1.0);
val _ = list_push(data, 2.0);
val _ = list_push(data, 3.0);
val _ = list_push(data, 4.0);
val t = tensor_from_data(data, shape)
val r = tensor_relu(t)
```

**Functions:** `tensor_numel`, `tensor_full`, `tensor_zeros`, `tensor_from_data`, `tensor_get`, `tensor_set`, `tensor_data`, `tensor_shape`, `tensor_add`, `tensor_relu`, `tensor_sigmoid`, `tensor_matmul2`, `tensor_batch_matmul`

---

## reverse-mode builtins

Interpreter-backed reverse-mode autodiff is available through builtins:

```iris
def main() -> f64 {
    val x = tape(3.0)
    val y = x * x + 2.0 * x
    val _ = backward(y)
    grad(x)
}
```

**Builtins:** `tape`, `backward`, `grad`

---

## testing

Test assertion helpers.

```iris
bring std.testing

def test_basic() -> bool {
    val ok = assert_eq(1 + 1, 2, "addition");
    val ok2 = assert_str_eq("hi", "hi", "strings");
    ok && ok2
}
```

**Functions:** `assert_eq`, `assert_approx_eq`, `assert_str_eq`, `assert_true`, `assert_false`, `assert_some`, `assert_none`, `assert_ok`, `assert_err`, `assert_list_eq`, `fail`

---

## log

Structured logging with severity levels.

```iris
bring std.log

log("server started")
info("Connected to port 8080")
warn("Disk usage above 80%")
debug("req_id=42 path=/api/data")
```

**Constants:** `LOG_DEBUG` (0), `LOG_INFO` (1), `LOG_WARN` (2), `LOG_ERROR` (3)

**Functions:** `debug`, `info`, `warn`, `error`, `log`, `log_at`, `log_kv`

---

## json

Flat-key JSON builder and parser.

```iris
bring std.json

// Build JSON
val s = json_str("hello")           // "\"hello\""
val arr = json_arr(["1","2","3"])   // "[1,2,3]"
val keys = list(); push(keys, "name");
val vals = list(); push(vals, json_str("Alice"));
val obj = json_obj(keys, vals)      // {"name":"Alice"}

// Simple flat object
val doc = json_new()
json_set(doc, "x", "1")
json_set(doc, "y", "2")
val v = json_get(doc, "x")         // some("1")
```

**Functions:** `json_str`, `json_arr`, `json_obj`, `json_new`, `json_set`, `json_get`

---

## iter

Functional list utilities (all operate on `list<i64>`).

```iris
bring std.iter

val nums = list(); push(nums, 3); push(nums, 1); push(nums, 2);
val s  = sum(nums)               // 6
val p  = product(nums)           // 6
val mn = min(nums)               // 1
val mx = max(nums)               // 3
val rv = reverse(nums)           // [2,1,3]
val tk = take(nums, 2)           // [3,1]
val dr = drop(nums, 1)           // [1,2]
val ct = count(nums, 1)          // 1
val idx = index_of(nums, 2)      // 2
```

**Functions:** `sum`, `product`, `min`, `max`, `mean_i64`, `reverse`, `take`, `drop`, `count`, `index_of`, `flatten`, `zip_sum`, `normalize_i64`, `range`

---

## set

Sorted-list-backed set for `i64` values.

```iris
bring std.set

val s = set_new()
set_add(s, 10);
set_add(s, 20);
val has = set_contains(s, 10)    // true
set_remove(s, 10)
val n = set_len(s)               // 1
val u = set_union(s1, s2)
val i = set_intersection(s1, s2)
val d = set_difference(s1, s2)
val l = set_to_list(s)
```

**Functions:** `set_new`, `set_add`, `set_remove`, `set_contains`, `set_len`, `set_union`, `set_intersection`, `set_difference`, `set_to_list`

---

## queue

FIFO queue backed by a list.

```iris
bring std.queue

var q = queue_new()
q = enqueue(q, 10)
q = enqueue(q, 20)
val v = dequeue_val(q)     // 10
q = dequeue_queue(q)
val empty = queue_is_empty(q)
val n = queue_len(q)
```

**Functions:** `queue_new`, `enqueue`, `dequeue_val`, `dequeue_queue`, `queue_peek`, `queue_len`, `queue_is_empty`

---

## heap

Min-heap (sorted-list-backed) for `i64` values.

```iris
bring std.heap

var h = heap_new()
h = heap_push(h, 5)
h = heap_push(h, 3)
h = heap_push(h, 8)
val min = heap_peek(h)    // 3
val v   = heap_pop_val(h) // 3
h = heap_pop_heap(h)
val n = heap_len(h)
```

**Functions:** `heap_new`, `heap_push`, `heap_peek`, `heap_pop_val`, `heap_pop_heap`, `heap_len`

---

## deque

Double-ended queue.

```iris
bring std.deque

val d = deque_new()
deque_push_front(d, 1);
deque_push_back(d, 2);
val f = deque_front(d)      // 1
val b = deque_back(d)       // 2
val pf = deque_pop_front(d) // 1
val pb = deque_pop_back(d)  // 2
val n = deque_len(d)
```

**Functions:** `deque_new`, `deque_push_front`, `deque_push_back`, `deque_pop_front`, `deque_pop_back`, `deque_front`, `deque_back`, `deque_len`

---

## bitset

Fixed-size bitset backed by an `i64`.

```iris
bring std.bitset

val b = bitset_new(64)
bitset_set(b, 3)
val v = bitset_get(b, 3)    // true
bitset_clear(b, 3)
val n = bitset_count(b)     // 0
```

**Functions:** `bitset_new`, `bitset_set`, `bitset_get`, `bitset_clear`, `bitset_count`

---

## os

Operating system utilities.

```iris
bring std.os

val dir = getcwd()             // current directory
val env = getenv("HOME")       // environment variable
setenv("MY_VAR", "value")
val files = readdir(".")       // list<str>
val ok = make_dir("newdir")
val ok2 = exists("file.txt")
val pid = get_pid()
val out = shell("echo hello")  // "hello\n"
exit(0)
```

**Functions:** `getcwd`, `getenv`, `setenv`, `get_pid`, `shell`, `exit`, `readdir`, `make_dir`, `exists`, `cpu_count`

---

## http

HTTP client.

```iris
bring std.http

val body = http_get("https://api.example.com/data")
val resp = http_post("https://api.example.com/post", "{\"x\":1}", "application/json")
```

**Functions:** `http_get`, `http_post`

---

## kv

File-backed key-value store (text format: `key=value\n`).

```iris
bring std.kv

kv_set("store.txt", "name", "Alice")
val v = kv_get("store.txt", "name")   // "Alice"
kv_delete("store.txt", "name")
val keys = kv_keys("store.txt")        // list<str>
```

**Functions:** `kv_get`, `kv_set`, `kv_delete`, `kv_keys`, `kv_read_file`, `kv_write_file`, `kv_pair`

---

## ml

Machine learning algorithms.

```iris
bring std.ml

// Linear regression
val model = linreg_train(X, y, 0.01, 1000)
val pred  = linreg_predict(model, x_new)

// Logistic regression
val clf = logreg_train(X, y, 0.01, 500)
val p   = logreg_predict(clf, x_new)

// k-NN
val label = knn_predict(X_train, y_train, x_query, k)

// k-Means
val centroids = kmeans_train(data, k, 100)

// Naive Bayes
val gnb = gnb_train(X, y, n_classes)
val pred = gnb_predict(gnb, x_new)

// Metrics
val acc = accuracy(y_true, y_pred)
val pr  = precision(y_true, y_pred)
val rc  = recall(y_true, y_pred)
val m   = mse(y_true, y_pred)
val mae_v = mae(y_true, y_pred)
```

---

## nn

Neural network building blocks.

```iris
bring std.nn

val mlp = mlp_create(layer_sizes)
mlp_train(mlp, X, y, lr, epochs)
val pred = mlp_predict(mlp, x)
val cls  = mlp_predict_class(mlp, x)
```

---

## crypto

Hashing and encoding.

```iris
bring std.crypto

val id  = generate_uuid()               // UUID v4 string
val h   = hash_code("hello")            // i64 hash
val b64 = encode_b64("hello")           // base64 string
val raw = decode_b64(b64)               // original string
val hex = to_hex("hello")               // hex-encoded
val dec = from_hex(hex)                 // decoded
```

**Functions:** `generate_uuid`, `hash_code`, `encode_b64`, `decode_b64`, `to_hex`, `from_hex`

---

## ffi

Foreign function interface for calling native C/Rust libraries.

```iris
bring std.ffi

val lib = lib_open("./mylib.so")
val res = lib_call(lib, "my_function")
lib_close(lib)
```

**Functions:** `lib_open`, `lib_call`, `lib_close`, `py_eval`, `py_exec`, `py_call`, `py_version`, `rust_open`

---

## csv

CSV file parser and emitter.

```iris
bring std.csv

val text = read_text("data.csv")
val rows = csv_row_count(text)
val cols = csv_col_count(text)
val row  = csv_get_row(text, 0)       // list<str>
val r    = csv_parse_row("a,b,c")     // list<str>
val line = csv_emit_row(cells)        // "a,b,c"
```

**Functions:** `csv_row_count`, `csv_col_count`, `csv_get_row`, `csv_parse_row`, `csv_emit_row`

---

## rl

Reinforcement learning primitives and experience replay buffers.

```iris
bring std.rl

// Experience tuple
val exp = Experience {
    state: [1.0, 0.0],
    action: 1,
    reward: 1.0,
    next_state: [1.0, 1.0],
    done: false
}

// Experience replay buffer
var buf = replay_buffer_new(100)
buf = replay_buffer_push(buf, exp)
val indices = replay_buffer_sample_indices(buf, 10)

// Q-Learning updates
val q = q_table_new(10, 2)
val val_q = q_get(q, 3, 1, 2)
q_set(q, 3, 1, 2, 0.5)
val updated_q = q_learning_update(q, 3, 1, 1.0, 4, 2, 0.1, 0.99)
val best_action = q_best_action(q, 3, 2)

// Environment step simulations
val step_res = grid_step(3, 1, 10) // (new_state, reward, done)
val reward = bandit_pull(2, 4)
```

**Records:** `Experience`, `ReplayBuffer`

**Functions:** `replay_buffer_new`, `replay_buffer_push`, `replay_buffer_sample_indices`, `q_table_new`, `q_get`, `q_set`, `q_learning_update`, `sarsa_update`, `q_best_action`, `policy_gradient_loss`, `log_prob_softmax`, `grid_step`, `bandit_pull`

---

## ais

Autonomous Intelligent Systems framework for perceptions, decisions, actions, and persistence.

```iris
bring std.ais

// Run autonomous agent loop
val total_steps = agent_loop(
    || sensor_read(), 
    |obs| predict(obs), 
    |act| actuate(act), 
    1000
)

// Decision strategy sampling
val best = argmax([1.0, 2.5, 0.5])         // 1
val a1   = epsilon_greedy([1.0, 2.5], 0.1) // random or best
val a2   = softmax_sample([1.0, 2.0, 0.5])
val a3   = boltzmann_sample([1.0, 2.0], 1.5)

// Reward engineering pipelines
val returns = discount_rewards([0.0, 1.0], 0.99)
val norm_returns = normalize_input(returns)
val clipped = clip_list(norm_returns, 0.0, 1.0)

// Model persistence
val ok = model_save(weights, "weights.txt")
val loaded_weights = model_load("weights.txt")
```

**Functions:** `agent_loop`, `agent_loop_rewards`, `argmax`, `epsilon_greedy`, `softmax_sample`, `boltzmann_sample`, `discount_rewards`, `gae`, `normalize_list`, `normalize_input`, `clip_list`, `model_save`, `model_load`
