# Crypto Ledger — Concurrent Encrypted Transaction Ledger

A multifile IRIS project demonstrating a blockchain-inspired transaction ledger
with concurrent worker threads, SHA-256 hashing, and chain integrity verification.

## Features Demonstrated

| Feature                | Where Used                                  |
|------------------------|---------------------------------------------|
| Records                | `Transaction`, `Block`, `Ledger`            |
| Enums / `choice`       | `TxStatus { Pending, Confirmed, Rejected }` |
| Pattern matching       | `when` on `TxStatus`                        |
| Traits + `impl`        | `Hashable` trait for Transaction and Block  |
| Generics               | `validate[T where T: Hashable](item: T)`   |
| Channels               | Worker → main result collection             |
| `spawn` concurrency    | Parallel transaction generation             |
| Atomics                | Shared worker counter                       |
| File I/O               | Write ledger report to JSON                 |
| SHA-256 hashing        | Block and transaction hashes                |
| UUID generation        | Unique transaction IDs                      |
| JSON serialization     | `json_stringify` for ledger export          |
| Closures               | Amount generator lambda                     |
| `option<T>`            | Safe lookups                                |
| Constants              | `MAX_NONCE`, `BLOCK_SIZE`                   |
| Timestamps             | `now_ms()` on transactions and blocks       |

## Project Structure

```
crypto_ledger/
├── README.md        — this file
├── types.iris       — core data types, traits, and helpers
├── engine.iris      — ledger operations (create, mine, verify)
├── worker.iris      — concurrent transaction processing
└── main.iris        — orchestrator and entry point
```

## How to Run

```bash
cargo run -- run projects/crypto_ledger/main.iris
```

## Architecture

1. **main.iris** creates a fresh ledger and spawns concurrent worker threads
2. **worker.iris** spawns N workers, each generating random transactions via channels
3. **engine.iris** collects pending transactions, mines them into blocks with nonce loops
4. **types.iris** defines all data structures with SHA-256 based `Hashable` trait
5. The chain is verified for integrity, and a JSON report is written to disk
