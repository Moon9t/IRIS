# A self-evolving data-ingestion cleaner

A stream of sensor readings arrives, most good and some corrupt. Nobody writes
the cleaning thresholds. They are encoded in a genome, scored against observed
outcomes, and evolved — and a homeostatic bound stops the result from being
useless.

Run it:

```bash
target/debug/iris.exe --emit eval examples/09_self_evolving/main.iris
IRIS_FORCE_INTERP=1 target/debug/iris.exe --emit eval examples/09_self_evolving/main.iris
```

Both backends produce identical results. Every claim below is asserted in the
file; nothing here is aspirational.

## What actually happens

Measured on this machine, seed `20260816`, 300 readings (80% good near 100,
20% corrupt — half wild spikes, half impossible negatives):

| | balanced reward | naive reward |
|---|---|---|
| evolved band | `lo=63.9, hi=120.0, z=3.24` | `lo=44.2, hi=96.4, z=1.27` |
| score on its own objective | **300 / 300** | 65 |
| accept rate | 0.783 | **0.293** |
| balanced reward | 300 | **6** |
| `homeostat_is_safe` | `true` | **`false`** |

The left column learned to classify perfectly — 300 out of a possible 300 —
without anyone specifying a cut-off. Its accepted band brackets where the good
data lives, and its throughput matches the 80% of the stream that is genuinely
good.

## The part worth reading

The right column is the point.

Every self-optimising cleaner has one dominant failure mode: **it learns to drop
everything.** Zero corrupt records accepted is a perfect score if nothing gets
through. The `reward_naive` function only penalises accepting a bad record —
which is the reward function a careful engineer writes first, because it encodes
the thing they were worried about.

Under it, evolution *succeeded*. It scored 65 on the objective it was given. It
did so by narrowing the accepted band to `[44.2, 96.4]`, which barely overlaps
the region where good readings actually live, and discarding 71% of the stream.
Judged by the reward that reflects what anyone actually wanted, it scores **6
against 300**.

No amount of tuning that reward would have caught this, because the reward is
what caused it.

## The bound that does catch it

`homeostat_is_safe` binds a homeostatic variable to **throughput**, not to
quality:

```iris
h = homeostat_add_var(h, homeovar_new(0.80, 0.12, 0.5));  // 80% ± 12%
```

The learner is free to optimise cleaning however it likes. It is not free to
leave that band. This works precisely because accept-rate is **not a term in the
reward** — there is nothing for the optimiser to trade it against, so no amount
of optimisation pressure argues its way past the gate.

That is the general shape: *let the system evolve inside bounds it cannot
renegotiate.*

## Reproducibility, and why it matters here

`seed()` fixes the entire run — the stream, the initial population, every
mutation, every tournament. Re-running with the same seed reproduces the evolved
policy to the bit, which `check_reproducible` asserts, along with the converse:
a different seed must explore differently, so the reproducibility is not an
implementation quietly ignoring the seed.

This is what makes an evolved system auditable. An operator can ask *"how did it
arrive at these thresholds"* and get an answer by replaying the run rather than
guessing. A self-evolving system that cannot be replayed cannot be reviewed
after an incident.

Before 2026-08-15 this example could not have existed: `random()` was libc
`rand()` natively — 15 bits of resolution, since `RAND_MAX` is 32767 — and a
chained hasher in the interpreter, with no way to seed either. The two backends
produced different sequences for the same program.

## Peripherals are optional

`Source` is a `choice`:

```iris
choice Source { Synthetic, SerialPort, Ros2Topic }
```

The stream here is synthetic so the example runs anywhere. A real deployment
swaps in `std.serial` — an Arduino UNO or ESP32 on a USB port streaming
readings — or `std.ros2`, without touching the policy or the evolution loop.

A `choice` rather than `dyn Trait` because trait objects have no native backend
yet (known-issues #18b). When that lands this becomes a trait and the sources
become independent implementations.

## Honest limits

- The stream is synthetic. The policy shape (a band plus a spread multiplier)
  suits this corruption model; real data needs a richer genome.
- `std.serial`'s protocol layer is asserted, but its **port layer has never been
  run against physical hardware**. Graded Present, not Verified.
- Fitness is evaluated over the whole stream at once. A live pipeline scores
  incrementally, which is a different and harder problem — the homeostat is
  already incremental, but the evolution loop here is not.
