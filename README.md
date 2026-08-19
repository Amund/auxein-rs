# Auxein Rust

Small, dependency-free Rust implementation of the [Auxein v0.2.0 engine](https://github.com/Amund/auxein).

This repository targets **causal conformance** with the canonical model in `spec/auxein.md`: the same presentations must induce the same concern decisions, local learning, contextual recursion, material growth/contraction and readout. Floating-point implementations are not required to be bit-for-bit identical when their causal decisions remain identical.

## Workspace

```text
auxein-core/   reusable engine library
auxein/        CLI binary
spec/          canonical v0.2.0 model
```

Both crates use the Rust standard library only. There are no crates.io dependencies. The implementation is MIT licensed.

## Cognitive path

A presentation is a simultaneous logical context. External vectors enter as point kernels `(r, C, V=0)`; internal layers receive at most one contextual kernel `(r, C, V)` per presentation.

```text
presentation
  -> CELL concern / multi-winner allocation
       -> unknown atoms -> private Sigma -> local CELL growth
       -> recognised values -> one recognised-context kernel
  -> next LAYER only when that context has V > 0 and C != 0
```

The vertical path never transmits `x-C` differences. It compresses the distinct values recognised from the frozen layer snapshot. For each atom of mass `r`, its `n` distinct recognised values receive `r/n` in the contextual construction; learning responsibilities do not weight vertical geometry. A singleton recognised value therefore has zero contextual variance and cannot create free depth. An exactly zero-centered context is also silent because Auxein has no canonical vector direction for that symmetric relation.

The same centered-kernel algebra `(W, C, V)` is used at every level. Incoming contextual variance participates in the second concern bound and is preserved by total-variance EMA; the external `V=0` case reduces exactly to the point-input law.

## Production implementation

`auxein-core` keeps the runtime deliberately small:

- `CELL / LAYER / NETWORK` only;
- centered kernels `(W, C, V)`;
- no cognitive matrices;
- EMA state only;
- `f32` or `f64` persistent storage selected at construction;
- all cognitive intermediate calculations in `f64`;
- exact integer material accounting;
- exact duplicate coalescence;
- causal frozen snapshots without replay;
- one all-or-nothing material growth transaction per presentation;
- std-only canonical JSON state import/export.

The production path keeps the episode-5 execution optimizations that are causally invisible. Frozen layer state is moved rather than cloned. Squared norms are cached only in execution memory. EMA targets share flat scratch buffers. Kernels are updated in place and carry transient dirty information only while a mutation is resolved. Canonically sorted centers provide an exact first-coordinate candidate window before the full concern predicate. Sparse CELL support decay is deferred by a per-layer execution clock and replayed with the same persistent projections only when that support becomes observable; `set_eta` materializes pending decay before changing `lambda`. Sigma remains eager because its support participates in promotion every presentation. Readout universe/local-input payloads use immutable shared storage. None of these caches or shortcuts is serialized, budgeted or behaviorally authoritative.

## Build and test

Rust 1.85 or newer:

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo build --release
```

A small in-process benchmark is also included:

```bash
cargo run --release -p auxein-core --example benchmark -- singleton 8 1 100000 1000
cargo run --release -p auxein-core --example benchmark -- pair-context 8 2 100000 1000
```

## Library

```rust
use auxein_core::{Auxein, Budget};

let mut network = Auxein::<f64>::new(
    2,
    50.0,
    1.0,
    Budget::kernels("100"),
    "auxein",
)?;

let report = network.step(&[vec![1.0, 2.0]], false)?;
println!("{:?}", report.readout);
```

For a runtime-selected persistent scalar, use `auxein_core::Network`. Raw material budgets are available with `Budget::units(n)`.

`Recognition` shares its `universe` and `local_input` payloads internally with `Arc`; callers can use `universe()`, `local_input()` and `recognised()` without depending on that storage choice.

### Persistence

```rust
let state = network.export_json();
let restored = Auxein::<f64>::from_json(
    &state,
    Budget::units(network.budget_units()),
    "auxein",
)?;
```

The state schema is `format_version = 2` and is compatible with canonical valid states exported by the Python reference. Budget and `universe` remain execution-environment data and are not serialized.

## CLI

The CLI is a JSONL stream processor. One input line is one external presentation: a non-empty JSON array of vectors. One output line is the corresponding `StepReport`.

```bash
printf '[[2.0]]\n[[2.0]]\n[[2.0]]\n' | \
  cargo run --release -p auxein -- run \
    --dimension 1 \
    --memory 10 \
    --budget 100 \
    --save state.json
```

Reload under an explicitly supplied execution budget:

```bash
printf '[[2.0]]\n' | \
  cargo run --release -p auxein -- run \
    --load state.json \
    --budget 100
```

Useful options:

```text
--scalar f32|f64
--eta RATE
--budget DECIMAL
--budget-units INTEGER
--universe NAME
--detailed
--load FILE
--save FILE
```

## Conformance strategy

The Rust suite locks the canonical boundary cases: local recurrence, internal variance, zero handling, `eta=0`, multi-winner conservation, recognised-context mass/geometry, singleton and zero-centered vertical silence, higher-layer context learning, non-cascade under constant input, duplicate/permutation invariance, material growth, forced solvency, persistence and `f32` projection.

Development additionally uses differential Python↔Rust runs over randomized multi-atom streams and constructed higher-layer cases in both persistent scalar formats. Conformance is causal rather than a requirement that every intermediate binary64 rounding be identical.
