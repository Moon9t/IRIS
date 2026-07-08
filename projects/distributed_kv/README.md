# Distributed Key-Value Store with Networking

A multifile IRIS project demonstrating a distributed in-memory key-value database
accessible over TCP.

## Features Demonstrated

- **TCP Networking**: Listening server and connecting client via `std.net`.
- **Concurrency**: Client request handling in concurrent background threads spawned via `spawn`.
- **Thread-safe store**: Mutex-guarded operations on a shared `map`.
- **Protocol Parser**: Request command verb and arguments string processing (`split`, `trim`).
- **File Persistence**: Exporting store to a JSON file.
- **Error Handling**: Using `option<T>` for connection status and lookup results.

## Project Structure

```
distributed_kv/
├── README.md        — this file
├── protocol.iris    — request/response string parser
├── store.iris       — in-memory map store operations
├── server.iris      — concurrent TCP server implementation
├── client.iris      — TCP client helpers and scripted test sequence
└── main.iris        — orchestrator and entry point
```

## How to Run

```bash
cargo run -- run projects/distributed_kv/main.iris
```
