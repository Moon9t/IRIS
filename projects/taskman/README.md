# Taskman

A full-featured interactive task manager written entirely in IRIS.

Demonstrates: **multi-file modules**, **KV persistence**, **JSON/CSV export**,
**option/result types**, **formatted output**, **interactive REPL loop**, and
**native binary compilation**.

---

## Project Structure

```bash
projects/taskman/
├── task.iris       Data model: Task record, encode/decode, status helpers
├── storage.iris    Persistence: KV-backed save/load/delete (bring std.kv)
├── report.iris     Reporting: table display, JSON, CSV, statistics
└── main.iris       CLI: interactive REPL loop, command dispatch
```

### Module graph

```bash
main.iris
├── bring "storage.iris"
│   ├── bring "task.iris"   → brings std.time
│   └── bring std.kv
├── bring "report.iris"
│   ├── bring "task.iris"   (cached — loaded once)
│   ├── bring std.json
│   └── bring std.fmt
└── bring std.string
```

---

## Running

### Interpreter (development)

```bash
# From the project root:
iris run projects/taskman/main.iris

# Or from within the taskman directory:
cd projects/taskman
iris run main.iris
```

### Native binary (production)

```bash
iris build projects/taskman/main.iris -o taskman
./taskman
```

The native binary is a self-contained executable with no runtime dependencies.

---

## Commands

| Command        | Description                              |
|----------------|------------------------------------------|
| `add <title>`  | Add a new task                           |
| `list`         | List all tasks with ID and status        |
| `done <id>`    | Mark a task as done `[x]`                |
| `cancel <id>`  | Cancel a task `[-]`                      |
| `rm <id>`      | Permanently remove a task                |
| `stats`        | Show counts and completion percentage    |
| `json`         | Export all tasks as a JSON array         |
| `csv`          | Export all tasks as CSV                  |
| `help`         | Show the command reference               |
| `quit`         | Exit Taskman                             |

---

## Example Session

```bash
  ──────────────────────────────────────────────────
  TASKMAN v1.0.0  —  Task Manager written in IRIS
  Data stored in: .taskman.db
  ──────────────────────────────────────────────────

   ID  Status  Title
  ───  ──────  ──────────────────────────────────────────
     1  [ ]     Buy groceries
     2  [x]     Write IRIS book

command>
add Call the dentist
  Added #3: Call the dentist

command>
done 1
  Done: #1 — Buy groceries

command>
stats

  Total tasks  : 3
  Pending      : 1
  Done         : 2
  Cancelled    : 0
  Completion   : 66%

command>
json
[
  {"id":1,"title":"Buy groceries","status":"done","created_ms":1740000000000,"done_ms":1740000001000},
  {"id":2,"title":"Write IRIS book","status":"done","created_ms":1740000000100,"done_ms":1740000000200},
  {"id":3,"title":"Call the dentist","status":"pending","created_ms":1740000001234,"done_ms":0}
]

command>
csv
id,title,status,created_ms,done_ms
1,"Buy groceries",done,1740000000000,1740000001000
2,"Write IRIS book",done,1740000000100,1740000000200
3,"Call the dentist",pending,1740000001234,0

command>
quit

  Goodbye!
```

---

## Data Storage

Tasks are persisted to `.taskman.db` in the current working directory using
IRIS's built-in KV store (`bring std.kv`).

File format (plain text, human-readable):

```bash
__next_id__=4
t:1=1|1|1740000000000|1740000001000|Buy groceries
t:2=1|1|1740000000100|1740000000200|Write IRIS book
t:3=0|0|1740000001234|0|Call the dentist
```

Task wire format: `id|status|created_ms|done_ms|title`
Status codes: `0` = pending, `1` = done, `2` = cancelled

> **Note:** Task titles must not contain the `|` character.

---

## IRIS Features Used

| Feature | Where |
| --------- | ------- |
| `bring "file.iris"` | Multi-file module system |
| `bring std.kv/time/json/fmt/string` | Standard library |
| `pub record Task { ... }` | Exported data type |
| `pub const STATUS_PENDING: i64 = 0` | Exported constants |
| `option<Task>` with `is_some`/`unwrap` | Safe nullable values |
| `split(s, "\|")` / `parse_i64(s)` | Serialisation |
| `while running { ... }` | Interactive loop |
| `read_line()` | Stdin input |
| `kv_get` / `kv_set` / `kv_delete` | File persistence |
| `pad_int(n, width)` | Formatted table output |
| `json_str(s)` | JSON string quoting |
