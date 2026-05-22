# Taskman

A fast, compiled task manager CLI written in IRIS.
Demonstrates IRIS multi-file module resolution and `std.db` (SQLite) systems programming.

## Building

```bash
iris build projects/taskman/main.iris -o taskman
```

## Usage

```bash
./taskman add "Write documentation"
./taskman list
./taskman complete 1
./taskman list
./taskman delete 1
```
