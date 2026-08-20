# Auxein Rust

Dependency-free Rust implementation of the [Auxein v0.4.0 engine](https://github.com/Amund/auxein).

The canonical mathematical/material model lives in [`spec/auxein.md`](spec/auxein.md).
This implementation targets causal conformance with the Python semantic
reference: the same presentations must induce the same concern decisions,
learning, contextual recursion, adjacent temporal knowledge, predictive
projection, material transactions and readout.

## Workspace

```text
auxein-core/   reusable engine library
auxein/        JSONL CLI binary
spec/          canonical v0.4.0 model
```

Both crates use the Rust standard library only. There are no crates.io
dependencies.

## Architecture

```text
NETWORK
  └─ ordered LAYERs
       ├─ geometric space E
       │    ├─ CELL kernels
       │    └─ private Σ kernels
       │
       └─ temporal space T(E)=E⊕E        [temporal / predictive]
            ├─ temporal CELL kernels
            ├─ private Σᵀ kernels
            └─ previous recognised context P
```

Every cognitive object is a centered kernel `(W, C, V)`: support, vector center
and scalar dispersion. External vectors enter as point kernels `(r, x, 0)`.

The geometric path is:

```text
presentation
  -> CELL concern / multi-winner allocation
       -> unknown atoms -> private Σ -> local CELL growth
       -> recognised values -> one recognised-context kernel
  -> next LAYER only when the context is vertically admissible
```

In temporal and predictive modes, the complete geometric phase runs first. For
every layer having recognised contexts at two adjacent external steps,

```text
H(t-1) = (W-, C-, V-)
H(t)   = (W+, C+, V+)

Xᵀ = (W- W+, C- ⊕ C+, V- + V+)
```

is presented in `T(E)=E⊕E`, where the same concern/allocation/EMA/detection
machinery learns temporal `CELL`/`Σᵀ` populations. Canonical time is exactly
`step-1 -> step`: no history window and no `T(T(E))`.

Predictive mode adds no learned population. Before the current temporal update,
it reads the snapshot of existing temporal `CELL`s. For current recognised
center `C` and temporal center `C- ⊕ C+`, the source projection is concerned iff

```text
||C - C-||² < ||C||²
and
||C - C-||² < ||C-||²
```

holds. Then `C+` is emitted as a known possible immediate successor. Temporal
support and full temporal variance are deliberately ignored: the temporal
quotient stores only `Vᵀ = V- + V+`, so no canonical source variance can be
reconstructed. Multiple successors are all emitted, never ranked, fed back or
chained.

## Modes

There are exactly three cumulative modes:

```text
geometry     geometric cognition only
temporal     geometry + adjacent temporal cognition
predictive   geometry + temporal + immediate predictive readout
```

```text
geometry ⊂ temporal ⊂ predictive
```

`mode` is immutable and serialized. There is no independent predictive flag.

## Library

Geometry remains the default:

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

Temporal or predictive mode is explicit:

```rust
use auxein_core::{Auxein, Budget, Mode};

let mut network = Auxein::<f64>::new_with_mode(
    2,
    50.0,
    1.0,
    Mode::Predictive,
    Budget::kernels("100"),
    "auxein",
)?;
```

For runtime-selected persistent scalar storage, use `auxein_core::Network` /
`Network::new_with_mode`. Raw material budgets are available through
`Budget::units(n)`.

### Readout

`Readout::Geometry` exposes conceptual recognitions:

```text
[universe, local_input, recognised]
```

`Readout::Temporal` contains independent `concepts` and `sequences` lists:

```text
sequence = [
  universe,
  [previous_input, current_input],
  [previous_recognised, current_recognised]
]
```

`Readout::Predictive` adds `predictions`:

```text
prediction = [
  universe,
  current_context,
  recognised_source,
  predicted_successor
]
```

The library exposes `readout.concepts()`, `readout.sequences()` and
`readout.predictions()`. No CELL id, layer id, pointer or persistent relation is
created between these views.

A zero source projection is silent under canonical point concern. A zero target
projection is an explicit valid prediction and remains distinct from no
prediction. Temporal `CELL`s promoted during a step gain predictive authority
only on the following external step.

### Persistence

The canonical state schema is **`format_version = 4`**.

Every state serializes:

- `dimension`, `scalar`, `memory`, `eta`, `mode`, `steps_seen`;
- ordered layers;
- geometric `cells` and private `sigma` kernels.

Temporal and predictive states additionally serialize, for every layer:

- `temporal_cells` and `temporal_sigma` in dimension `2D`;
- `previous`, the optional recognised context from the immediately preceding
  external step.

Predictive mode adds no persistent learned field over temporal mode. For
identical knowledge both modes have identical material cost; only the immutable
mode tag differs.

`previous` is causal state, not learned knowledge. It advances even at `eta=0`.
Forced material contraction invalidates previous-context registers so temporal
recognition cannot cross a knowledge-destruction boundary.

Budget and `universe` are execution/interface data and are not serialized.

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
- `f32` or `f64` persistent storage;
- cognitive intermediate calculations in `f64`;
- exact integer material accounting;
- exact duplicate coalescence;
- causal frozen snapshots without replay;
- one all-or-nothing growth transaction per external presentation;
- projected-seed revalidation at the persistent scalar boundary;
- one common economy across geometric and temporal kernels;
- predictive projection as an ephemeral read-only pass;
- std-only strict canonical JSON import/export.

Causally invisible execution optimizations remain allowed. Frozen layer state is
moved rather than cloned. Squared norms are cached only in execution memory.
EMA targets share scratch buffers. Canonically sorted centers provide an exact
first-coordinate candidate window before the full concern predicate. Sparse
CELL support decay is deferred by independent geometry/temporal execution
clocks and materialized when observable. None of these caches or shortcuts is
serialized, budgeted or behaviorally authoritative.

## Material economy

For persistent scalar size `p` (`4` for f32, `8` for f64):

```text
geometric kernel U_H = (D + 2) p
temporal kernel  U_T = (2D + 2) p
network header   U_N = 34 + 2p
geometry layer   U_L = 16
temporal/predictive layer U_L = 33 + U_H
```

Predictive projections/readout are ephemeral and have no persistent cost.

New geometric seeds, temporal seeds and an optional frontier layer enter one
global growth transaction. Seed requests are projected to the persistent scalar,
revalidated against current `CELL`s in their own space and exactly coalesced
before affordability is decided. If forced contraction is already required,
private `Σ`/`Σᵀ` work is discarded first, then all `CELL`s share the same exact
value ordering:

```text
K = ||C||² / (||C||² + V)
```

Equal `K` values live or die together regardless of space.

## CLI

The CLI is a JSONL stream processor. One input line is one non-empty external
presentation; one output line is the corresponding `StepReport`.

Geometry:

```bash
printf '[[2.0]]\n[[2.0]]\n[[2.0]]\n' | \
  cargo run --release -p auxein -- run \
    --dimension 1 --memory 10 --budget 100
```

Temporal:

```bash
printf '[[1.0]]\n[[3.0]]\n[[1.0]]\n[[3.0]]\n' | \
  cargo run --release -p auxein -- run \
    --dimension 1 --memory 10 --mode temporal --budget 100
```

Predictive:

```bash
printf '[[1.0]]\n[[3.0]]\n[[1.0]]\n[[3.0]]\n[[1.0]]\n' | \
  cargo run --release -p auxein -- run \
    --dimension 1 --memory 10 --mode predictive --budget 100
```

Reloading takes its mode, dimension, memory and scalar from the state:

```bash
printf '[[2.0]]\n' | \
  cargo run --release -p auxein -- run \
    --load state.json --budget 100
```

Useful options:

```text
--scalar f32|f64
--mode geometry|temporal|predictive
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

The regression suite covers geometry, temporal and predictive behavior,
including recurrence, multi-winner conservation, context geometry, temporal
adjacency/recurrence/gaps, causal register persistence, shared economics,
contraction, f32 boundary revalidation, numerical extremes, predictive
center-only projection, branching futures, zero endpoints, next-step authority,
mode round-trip and exact temporal/predictive persistent-trajectory equivalence.

## Benchmark

The in-process benchmark accepts the mode as the sixth positional argument:

```bash
cargo run --release -p auxein-core --example benchmark -- singleton 8 1 100000 1000 geometry
cargo run --release -p auxein-core --example benchmark -- singleton 8 1 100000 1000 temporal
cargo run --release -p auxein-core --example benchmark -- singleton 8 1 100000 1000 predictive
cargo run --release -p auxein-core --example benchmark -- temporal-stable 8 1 100000 1000 temporal
cargo run --release -p auxein-core --example benchmark -- predictive-stable 8 1 100000 1000 predictive
cargo run --release -p auxein-core --example benchmark -- sparse 8 512 100000 1000 geometry
cargo run --release -p auxein-core --example benchmark -- dense 8 512 100000 1000 geometry
```

`temporal-stable` preloads a known `A→A` temporal `CELL` and exercises geometry
+ temporal recognition. `predictive-stable` uses the same persistent knowledge
and additionally exercises the projection/readout pass.

## Conformance strategy

Rust locks canonical boundary cases in unit tests and is additionally checked
against the Python semantic reference on deterministic/randomized traces.
Representational optimizations are allowed only when they cannot change a
canonical causal decision.

## License

See [`LICENSE`](LICENSE).
