# Fleet telemetry service

A five-module IRIS system: ingest sensor readings, fuse them into a
trustworthy number, decide whether each device can be believed, and serve the
result over HTTP.

```bash
iris run examples/08_fleet_service/main.iris
```

Every claim the program makes about itself is an `assert`. A run that prints
the summary has verified all of them.

## Modules

| File | Demonstrates |
|---|---|
| `types.iris` | `choice` ADTs, records with defaults, traits with multiple impls, a generic container |
| `fusion.iris` | `option` / `result`, `?` propagation, closures as arguments, range patterns |
| `health.iris` | incremental (Welford) statistics, an adaptive threshold, a safety constitution |
| `api.iris` | JSON construction, HTTP status codes, a routing table, request dispatch |
| `main.iris` | channels + `spawn`, `par for`, atomics, and the end-to-end assertions |

## The decisions worth reading

**A stale device is never "nominal".** `verdict_for` folds the score-based and
age-based assessments with `health_worse`, so a device with a perfect signal
that has not reported in a minute reads Offline. Expressing this as a fold
rather than a rule means no amount of threshold adaptation can erode it.

**Staleness is relative to device class.** A camera silent for two seconds is
dead; a thermal sensor silent for two seconds is fine. `staleness_health`
compares against `class_period_ms`, which is the difference between a useful
alarm and a wall of noise.

**"No data" is not "fine".** `fuse_score` returns `result`, and `assess_device`
maps a device with fewer than two readings to Offline rather than letting a
default score of 1.0 stand in for evidence.

**The threshold adapts, within bounds.** `assess_adapt` widens the drift limit
for a twitchy device and tightens it for a quiet one, clamped to `[1.5, 5.0]`.
Written once; no per-site retuning visit.

## Known limitations hit while building this

Building this surfaced seven previously unrecorded compiler defects, written up
as issues 6–12 in [`docs/known-issues.md`](../../docs/known-issues.md).

**Runs natively.** It did not at first — writing it exposed two codegen bugs
that are now **fixed**, which is the main reason this example earns its place:

- **Issue 6.** `Device` and `Verdict` hold `choice` fields and are returned from
  functions, which emitted invalid LLVM IR (`store ptr` for an `i64` tag).
  Fixed in `MakeStruct` field-store codegen.
- **Issue 12.** `par for` never passed its captures to the loop body, so the
  parallel-counter idiom segfaulted. Fixed by routing captures through a
  closure environment.

Two still shape the code you are reading:

- **`Assessor` inlines its statistics** rather than embedding `std.ais`'s
  `RunningStats`, because a record field typed by a brought module is mangled
  as generic (issue 7).
- **`health_eq` exists** because `==` on two `choice` values fails at runtime
  (issue 8).

Also worked around: `pub bring` not re-exporting types (issue 9, hence every
file brings `types.iris` directly), and effect clauses on trait methods
(issue 10, hence `verdict_detail` as a free function).

None of these were visible from reading the compiler. All seven came from
writing a program that asserts its results and running it — and one of them is
now fixed because of it.
