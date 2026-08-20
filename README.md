# Auxein Rust

Dependency-free Rust implementation of the [Auxein v0.3.0 engine](https://github.com/Amund/auxein).


The canonical mathematical/material model lives in [`auxein.md`](https://github.com/Amund/auxein/blob/main/spec/auxein.md). This implementation targets **causal conformance** with the Python reference: the same presentations must induce the same concern decisions, local learning, contextual recursion, adjacent temporal learning, material growth/contraction and readout. Floating-point implementations are not required to be bit-for-bit identical when their causal decisions remain identical.

## Workspace

```text
auxein-core/   reusable engine library
auxein/        JSONL CLI binary
```

Both crates use the Rust standard library only. There are no crates.io dependencies.

## Architecture

Auxein keeps only three architectural levels:

```text
NETWORK
  └─ ordered LAYERs
       ├─ geometric space E
       │    ├─ CELL kernels
       │    └─ private Σ kernels
       │
       └─ temporal space T(E)=E⊕E        [temporal mode]
            ├─ temporal CELL kernels
            ├─ private Σᵀ kernels
            └─ previous recognised context P
```

Every cognitive object is a centered kernel `(W, C, V)`: support, vector center and scalar dispersion. External vectors enter as point kernels `(r, x, 0)`.

The geometric path is:

```text
presentation
  -> CELL concern / multi-winner allocation
       -> unknown atoms -> private Σ -> local CELL growth
       -> recognised values -> one recognised-context kernel
  -> next LAYER only when the context has V > 0 and C != 0
```

In temporal mode, the complete geometric phase runs first. For every layer having recognised contexts at two adjacent external steps,

```text
H(t-1) = (W-, C-, V-)
H(t)   = (W+, C+, V+)
```

the `NETWORK` constructs the direct-product presentation

```text
Xᵀ = (W- W+, C- ⊕ C+, V- + V+)
```

in `T(E)=E⊕E`, then applies the **same** concern/allocation/EMA/detection machinery to temporal `CELL`/`Σᵀ` populations. Geometric and temporal cognition never read or compete with each other. They share only the material economy and the external step readout.

Canonical time is exactly `step-1 -> step`: there is no history window and no `T(T(E))`.

## Modes

```text
geometry   default; geometric cognition only
temporal   geometry + adjacent temporal cognition
```

`mode` is immutable and serialized because it changes the causal state machine. `predictive` is intentionally **not** a v0.3.0 mode.

## Library

Geometry mode remains the default:

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

Temporal mode is explicit:

```rust
use auxein_core::{Auxein, Budget, Mode};

let mut network = Auxein::<f64>::new_with_mode(
    2,
    50.0,
    1.0,
    Mode::Temporal,
    Budget::kernels("100"),
    "auxein",
)?;
```

For a runtime-selected persistent scalar, use `auxein_core::Network` / `Network::new_with_mode`. Raw material budgets are available with `Budget::units(n)`.

### Readout

In `geometry` mode, `StepReport.readout` is `Readout::Geometry` and exposes the same flat conceptual recognitions as before:

```text
[universe, local_input, recognised]
```

In `temporal` mode, `Readout::Temporal` contains two independent lists:

```text
concepts:
  [universe, local_input, recognised]

sequences:
  [
    universe,
    [previous_input, current_input],
    [previous_recognised, current_recognised]
  ]
```

The two lists coexist only at the external causal boundary of the step. No CELL id, layer id, pointer or persistent concept↔sequence relation is created.

### Persistence

The canonical state schema is `format_version = 3`.

Every state serializes:

- `dimension`, `scalar`, `memory`, `eta`, `mode`, `steps_seen`;
- ordered layers;
- geometric `cells` and private `sigma` kernels.

Temporal states additionally serialize, for every layer:

- `temporal_cells` and `temporal_sigma` in dimension `2D`;
- `previous`, the optional recognised context from the immediately preceding external step.

`previous` is causal state, not learned knowledge. It advances even at `eta=0`; a forced material contraction invalidates all previous-context registers so no temporal recognition crosses a knowledge-destruction boundary.

Budget and `universe` remain execution-environment/interface data and are not serialized.

```rust
let state = network.export_json();
let restored = Auxein::<f64>::from_json(
    &state,
    Budget::units(network.budget_units()),
    "auxein",
)?;
```

## Production implementation

`auxein-core` keeps the runtime deliberately small:

- `CELL / LAYER / NETWORK` only;
- centered kernels `(W, C, V)`;
- no cognitive matrices or graph;
- `f32` or `f64` persistent storage selected at construction;
- all cognitive intermediate calculations in `f64`;
- exact integer material accounting;
- exact duplicate coalescence;
- causal frozen snapshots without replay;
- one all-or-nothing material growth transaction per external presentation;
- projected-seed revalidation at the persistent scalar boundary;
- one common economy across geometric and temporal kernels;
- std-only strict canonical JSON import/export.

The causally invisible execution optimizations remain in place. Frozen layer state is moved rather than cloned. Squared norms are cached only in execution memory. EMA targets share flat scratch buffers. Canonically sorted centers provide an exact first-coordinate candidate window before the full concern predicate. Sparse CELL support decay is deferred by execution clocks and materialized with the same persistent projections when it becomes observable. Geometry and temporal populations use **independent decay clocks**, because absence of a temporal presentation is not a zero temporal presentation. No cache or shortcut is serialized, budgeted or behaviorally authoritative.

## Material economy

For persistent scalar size `p` (`4` for f32, `8` for f64):

```text
geometric kernel U_H = (D + 2) p
temporal kernel  U_T = (2D + 2) p
network header   U_N = 34 + 2p
geometry layer   U_L = 16
temporal layer   U_L = 33 + U_H
```

The temporal-layer header includes a fixed material slot for optional `previous`, so merely recognising a context never creates unbudgeted persistent growth.

New geometric seeds, temporal seeds and an optional frontier layer enter **one global growth transaction**. Every seed request is first projected to the persistent scalar format, rechecked against the current `CELL`s in its own space, and exactly coalesced with projected/private clones. Affordability is computed from the resulting net persistent state, so f32 rounding cannot leave a newly persistent `Σ` kernel already covered by a `CELL`.

If forced contraction is already required, private `Σ`/`Σᵀ` work is discarded first, then geometric and temporal `CELL`s share the same exact value ordering

```text
K = ||C||² / (||C||² + V)
```

and equal `K` values live or die together regardless of space.

## CLI

The CLI is a JSONL stream processor. One input line is one non-empty external presentation; one output line is the corresponding `StepReport`.

Geometry mode:

```bash
printf '[[2.0]]\n[[2.0]]\n[[2.0]]\n' | \
  cargo run --release -p auxein -- run \
    --dimension 1 \
    --memory 10 \
    --budget 100 \
    --save state.json
```

Temporal mode:

```bash
printf '[[1.0]]\n[[3.0]]\n[[1.0]]\n[[3.0]]\n' | \
  cargo run --release -p auxein -- run \
    --dimension 1 \
    --memory 10 \
    --mode temporal \
    --budget 100
```

Reloading takes its mode, dimension, memory and scalar from the state:

```bash
printf '[[2.0]]\n' | \
  cargo run --release -p auxein -- run \
    --load state.json \
    --budget 100
```

Useful options:

```text
--scalar f32|f64
--mode geometry|temporal
--eta RATE
--budget DECIMAL
--budget-units INTEGER
--universe NAME
--detailed
--load FILE
--save FILE
```

## Build and test

Rust 1.85 or newer:

```bash
cargo test --workspace --offline
cargo clippy --workspace --all-targets --offline -- -D warnings
cargo build --release --workspace --offline
```

The regression suite covers both modes, including local recurrence, internal variance, zero handling, `eta=0`, multi-winner conservation, recognised-context geometry, vertical silence rules, temporal adjacency/order, temporal recurrence through `Σᵀ`, gap breaking, previous-context persistence, shared growth economics, forced contraction, `0→0` temporal silence, scale invariance, lazy-decay behavior, finite f64 geometric extremes, positive-support underflow and f32 persistent-boundary seed revalidation.

## Benchmark

The in-process benchmark accepts the mode as the sixth positional argument:

```bash
cargo run --release -p auxein-core --example benchmark -- singleton 8 1 100000 1000 geometry
cargo run --release -p auxein-core --example benchmark -- singleton 8 1 100000 1000 temporal
cargo run --release -p auxein-core --example benchmark -- temporal-stable 8 1 100000 1000 temporal
cargo run --release -p auxein-core --example benchmark -- pair-context 8 2 100000 1000 geometry
cargo run --release -p auxein-core --example benchmark -- sparse 8 512 100000 1000 geometry
cargo run --release -p auxein-core --example benchmark -- dense 8 512 100000 1000 geometry
```

`temporal-stable` preloads a known `A→A` temporal `CELL` and exercises the complete geometry + temporal recognition path after warmup.

## Conformance strategy

Rust locks the canonical boundary cases in unit tests and is additionally checked against the Python semantic reference on common deterministic/randomized traces. Conformance is causal: representational optimizations are allowed only when they cannot change a canonical decision.

## License

See [`LICENSE`](LICENSE).
