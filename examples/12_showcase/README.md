# Showcase

One program using most of the language at once, because the interesting
question is not whether a feature exists but whether the features compose.

## `order_book.iris`

A warehouse order pipeline. The features are here because the problem needs
them, not to tick boxes:

| What | Where it earns its place |
|---|---|
| `record` + `choice` | model SKUs, line kinds and order size |
| traits | two line kinds, one `subtotal`/`label` interface |
| blanket impl | `render` written once, applies to every `Line` |
| `list<dyn Trait>` | one order holds item lines *and* fee lines |
| `map` | inventory lookup, and per-customer aggregation |
| `option` | a lookup that can miss, without a sentinel |
| `result` + `?` | stock validation, failure propagated from mid-function |
| `when` + guards | classify an order by value |
| generics | a fold that is not tied to one element type |
| closures | the discount rule is supplied at the call site |
| effects | every function declares what it does; one is proven pure |

Run it:

```bash
iris run examples/12_showcase/order_book.iris
```

Then run the version that matters:

```bash
iris --strict-effects --emit eval examples/12_showcase/order_book.iris
```

It passes with no errors and no warnings. That is the compiler agreeing that
every declared effect row matches the whole reachable call graph — and that
`line_tax`, the one function with no `effect` clause, allocates nothing,
performs no I/O and calls nothing external. Add a `println` to it and the build
fails.

## Every assertion is real

There are no `println`-and-hope checks here. Each result is asserted, and the
assertions are chosen so that a plausible wrong implementation fails them:
`total_of` is checked against a value that a vtable stuck on one impl could not
produce, and the `?` propagation test names *which* SKU ran short, so a version
that reported the first failure rather than the real one would not pass.
