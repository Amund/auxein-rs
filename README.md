# Auxein Rust

Auxein is a small unsupervised cognitive engine built from centered kernels,
EMA learning and finite material growth. It has no matrices, labels, target
loss, top-k selection or persistent graph.

This workspace implements the **Auxein v0.5.0** mathematical/material canon in
[`spec/auxein.md`](spec/auxein.md). The core crate is dependency-free and
supports both `f32` and `f64` persistent state.

The central rule is:

> recurrent unknown becomes local knowledge; recognised knowledge is weighted
> by geometric concern and can become higher context; explicit adjacent
> contexts can become private predictive knowledge.

## Workspace

```text
auxein-core/       library implementation and regression tests
auxein/            JSON-lines command-line interface
spec/auxein.md     mathematical/material canon v0.5.0
```

The workspace requires Rust 1.85 or newer.

## Structure

There are only two public modes:

```text
geometry
predictive = geometry + private adjacent succession + future readout
```

Persistent structure:

```text
NETWORK
  └─ ordered LAYERs
       ├─ geometric space E
       │    ├─ CELL kernels
       │    └─ private Σ kernels
       │
       └─ predictive-private T(E)=E⊕E        [predictive only]
            ├─ temporal CELL kernels
            ├─ private Σᵀ kernels
            └─ previous recognised context P
```

Every cognitive object is a centered kernel `(W,C,V)`: support/mass, vector
center and scalar dispersion. Temporal populations are private implementation
state in predictive mode; there is no public `temporal` mode and no sequence
readout.

## Presentations

The canonical boundary is a finite non-empty weighted presentation. In Rust it
is represented by `InputAtom`:

```rust
use auxein_core::InputAtom;

let presentation = vec![
    InputAtom::new(0.25, vec![1.0, 0.0], 0.0),
    InputAtom::new(0.50, vec![0.0, 1.0], 0.2),
    InputAtom::new(0.25, vec![0.0, 0.0], 0.0),
];
```

Weights must be positive and their total mass must lie in `(0,1]`. Exact
`(C,V)` duplicates are coalesced.

A non-empty vector presentation remains available as uniform point-kernel
sugar:

```rust
let presentation = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
// equivalent to two weight-0.5 point kernels
```

A zero-center atom carries causal mass but no cognitive direction. It cannot
concern a `CELL`, feed `Σ`, create a seed or enter a vertical context.

## Recognition and vertical growth

For an atom `X_s=(r_s,c_s,v_s)`, each distinct recognised center `C` receives
knowledge mass from the canonical `CONCERN` gain:

```text
g_C(X_s) = ||c_s||² - ||c_s-C||²

ω_sC = r_s g_C / Σ_D g_D
```

This is deliberately independent of `CELL` support and learning
responsibilities. `ALLOCATE` governs learning; concern gain governs the current
knowledge presentation.

Recognised point kernels are coalesced into `K_L` and merged into one vertical
context `(W,C,V)`. A context rises only when `V>0` and `C!=0`.

## Readout

A responding layer emits its recognised point kernels with knowledge weights,
completed to mass one by a zero remainder when necessary. Persistent `CELL`
variance is not emitted.

The typed readout is:

```rust
pub enum Readout {
    Geometry {
        present: Vec<OutputPresentation>,
    },
    Predictive {
        present: Vec<OutputPresentation>,
        future: Vec<OutputPresentation>,
    },
}
```

Each `OutputPresentation` contains `OutputAtom { weight, center, variance }`.
The variance of recognised/predicted values is `0`.

Conceptually the JSON-compatible shape is:

```text
geometry:
{
  "present": [presentation, ...]
}

predictive:
{
  "present": [presentation, ...],
  "future":  [presentation, ...]
}

presentation:
[
  [weight, center, variance],
  ...
]
```

Each layer presentation owns its own mass universe. Weights from different
layers are never flattened or renormalised together. Distinct future candidates
remain distinct presentations. Predictions are never ranked, fed back or
chained.

## Library

Add `auxein-core` as a dependency, then construct either scalar implementation:

```rust
use auxein_core::{Auxein, Budget, Mode};

fn main() -> auxein_core::Result<()> {
    let mut geometry = Auxein::<f64>::new(
        2,
        20.0,
        1.0,
        Budget::kernels("100"),
    )?;

    let mut predictive = Auxein::<f64>::new_with_mode(
        2,
        20.0,
        1.0,
        Mode::Predictive,
        Budget::kernels("100"),
    )?;

    let report = geometry.step(&[vec![1.0, 0.0]], false)?;
    println!("{:?}", report.readout);

    let _ = predictive.step(&[vec![1.0, 0.0]], false)?;
    Ok(())
}
```

For canonical weighted input use `step_weighted`:

```rust
use auxein_core::InputAtom;

let report = predictive.step_weighted(
    &[
        InputAtom::new(0.25, vec![1.0, 0.0], 0.0),
        InputAtom::new(0.75, vec![9.0, 0.0], 0.0),
    ],
    false,
)?;
```

## Explicit sequences

Causality belongs to an explicit sequence boundary, not to API-call order.

`step()` and `step_weighted()` process **one atomic sequence**. Successive calls
can never learn a temporal relation between one another:

```rust
net.step(&[vec![1.0]], false)?;
net.step(&[vec![10.0]], false)?; // no implicit 1 -> 10 relation
```

Use `sequence()` for a real causal sequence:

```rust
let reports = net.sequence(
    &[
        vec![vec![1.0]],
        vec![vec![10.0]],
    ],
    false,
)?;
```

Weighted causal sequences use `sequence_weighted()`.

For streaming execution:

```rust
net.begin_sequence(false)?;
let r0 = net.sequence_step(&[vec![1.0]], false)?;
let r1 = net.sequence_step(&[vec![10.0]], false)?;
net.end_sequence()?;
```

All previous-context registers are cleared at normal sequence open and close. A
singleton can use already learned temporal knowledge to predict, but it cannot
learn an incoming or outgoing transition.

`begin_sequence(true)` is the explicit opt-in for continuing causal registers
restored from a mid-sequence persistent state. Loading state by itself never
creates causal continuity.

## Direct Auxein → Auxein composition

Only an upstream `present` family is directly composable. `consume()` feeds
each layer presentation downstream, in layer-depth order, as an independent
atomic sequence:

```rust
let upstream_report = upstream.step(&input, false)?;
let downstream_reports = downstream.consume(
    upstream_report.readout.present(),
    false,
)?;
```

This preserves geometry and prevents false temporal links between simultaneous
layer outputs. An empty family still destroys downstream causal continuity.
`future` is never auto-reinjected.

## Predictive mode

For adjacent contexts inside one explicit sequence:

```text
H(t-1) = (W-, C-, V-)
H(t)   = (W+, C+, V+)

Xᵀ = (W-W+, C- ⊕ C+, V- + V+)
```

The private temporal population applies the same centered-kernel laws in
`2D`.

Prediction adds no learned state. From current context center `C` and a frozen
temporal `CELL` center `S⊕T`, the source is concerned by point `CONCERN`:

```text
||C-S||² < ||C||²
and
||C-S||² < ||S||²
```

Each distinct successor `T` becomes its own future presentation. Its local
mass is `W * γ`, where `γ = 1 - ||C-S||² / ||C||²` is the relative source
`CONCERN` gain; the remainder is assigned to zero direction. Distinct futures
are never normalised against one another. If several relations project to the
same exact target, only their maximal `γ` survives. Temporal support and
variance have no predictive authority.

## Persistence

Canonical state uses `format_version=5` and stores:

- `dimension`, scalar, memory, eta, mode and completed-presentation counter;
- ordered layers;
- geometric `cells` and private `sigma`;
- in predictive mode, private `temporal_cells`, `temporal_sigma`, and optional
  `previous` per layer.

Budget is environmental and is not serialized. Forced material contraction
invalidates all causal previous registers.

```rust
let state = net.export_json();
let restored = Auxein::<f64>::from_json(
    &state,
    Budget::kernels("100"),
)?;

// Production hosts can avoid a second full state-sized String:
let mut file = std::fs::File::create("state.json")?;
net.write_json(&mut file)?;
```

The runtime-dispatched `Network` type exposes the same state and stepping
contract when the scalar is chosen dynamically. The CLI also saves state by
streaming to a temporary file and atomically renaming it, avoiding a second
state-sized JSON allocation in the normal persistence path.

## Material economy

The implementation follows the canon's exact packing model. `Budget::kernels`
uses the ergonomic geometric-kernel equivalent, while `Budget::units` accepts
raw material units.

```rust
use auxein_core::Budget;

let ergonomic = Budget::kernels("100.5");
let exact = Budget::units(4096);
```

Budget changes do not mutate cognition immediately. Solvability is restored at
the next presentation boundary. Growth of geometric seeds, private temporal
seeds and a frontier layer is one global all-or-nothing transaction.

## Production hardening

The Rust implementation is designed for long-running hosts rather than only
short benchmark processes. The production paths include:

- lazy CELL and private-Σ decay, with `O(log age)` exponentiation and clock
  rebasing instead of replaying one multiplication per missed epoch;
- sparse Σ maintenance: untouched private kernels are not revisited while the
  CELL geometry is unchanged;
- exact candidate-window pruning plus a safe energy bound before full
  `O(D)` CONCERN evaluation;
- predictive candidate filtering and target deduplication before future
  presentation allocation, with a no-allocation singleton fast path;
- two-phase growth transactions: rejected growth does not clone a future Σ,
  while accepted growth merges already canonical populations linearly;
- incremental forced-contraction accounting instead of rescanning all layers
  after every removal wave;
- homogeneous numerical fallbacks for extreme norms, CONCERN comparisons,
  contraction value and context variance;
- fallible reservations at host-controlled allocation frontiers and structured
  I/O errors on streaming state/report writers;
- bounded JSON nesting before recursive parsing;
- saturating administrative counters and explicit lazy-clock rollover; and
- `#![forbid(unsafe_code)]` in both library and CLI.

Transient high-water scratch can be inspected and explicitly released without
changing persistent cognition:

```rust
let bytes = net.transient_memory_capacity_bytes();
net.compact_transient_memory();
```

Forced material contraction also drops capacity belonging to destroyed private
work instead of retaining it indefinitely.

## CLI

Build or run the `auxein` binary from the workspace:

```bash
cargo run -p auxein -- run \
  --dimension 2 \
  --memory 20 \
  --mode predictive \
  --budget 100
```

The CLI reads one JSON presentation per non-empty stdin line. By default **each
line is an atomic sequence**.

Uniform vector sugar:

```json
[[1.0, 0.0], [0.0, 1.0]]
```

Canonical weighted presentation:

```json
[[0.25,[1.0,0.0],0.0],[0.75,[9.0,0.0],0.0]]
```

Use `--sequence` only when all input lines belong to one explicit causal
sequence:

```bash
printf '%s\n' '[[1.0]]' '[[10.0]]' | \
  cargo run -p auxein -- run \
    --dimension 1 --memory 20 --mode predictive --budget 100 --sequence
```

State can be loaded/saved with `--load FILE` and `--save FILE`; loading uses
canonical `format_version=5` JSON. `summary` reads an existing state:

```bash
cargo run -p auxein -- summary --load state.json --budget 100
```

Run `auxein --help` for all options.

## Build and test

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
```

The regression suite also covers hostile JSON, streaming I/O failure, lazy-clock
rollover, zero-decay memory, f32 representability, subnormal/overflow numeric
closures, large multi-winner mass closure, predictive relative-gain weighting,
independent branches, same-target max envelopes, scratch compaction and direct
composition. For production certification, the workspace is additionally run
with release overflow checks enabled.

## Benchmark

The core crate ships a dependency-free benchmark example. Arguments are:

```text
scenario dimension cells steps warmup mode eta
```

Representative hot paths:

```bash
cargo run --release -p auxein-core --example benchmark -- \
  singleton 8 1 1000000 1000 geometry 0

cargo run --release -p auxein-core --example benchmark -- \
  pair-context 8 2 300000 1000 geometry 1

cargo run --release -p auxein-core --example benchmark -- \
  dense 32 4096 1000 20 geometry 0

cargo run --release -p auxein-core --example benchmark -- \
  predictive-sequence 8 2 200000 1000 predictive 1
```

Adversarial scenarios are intentionally kept in the same benchmark so future
refactors can detect asymptotic regressions:

```bash
# Degenerate first-coordinate index; rejection must happen before D-wide scans.
cargo run --release -p auxein-core --example benchmark -- \
  same-first 64 32768 1000 50 geometry 0

# Large sleeping private population; cost should stay local while Σ is untouched.
cargo run --release -p auxein-core --example benchmark -- \
  sigma-idle 8 32768 100000 100 geometry 0

# Many distinct futures / many relations sharing one future.
cargo run --release -p auxein-core --example benchmark -- \
  predictive-fanout 8 8192 300 20 predictive 0
cargo run --release -p auxein-core --example benchmark -- \
  predictive-duplicate-fanout 8 8192 1000 20 predictive 0

# Temporal population completely outside the current source domain.
cargo run --release -p auxein-core --example benchmark -- \
  temporal-outside 8 8192 100000 100 predictive 0
```

Benchmark numbers are machine-dependent; compare builds on the same host and
alternate run order for microsecond-scale cases. A machine-local certification
snapshot is retained in [`BENCHMARKS.md`](BENCHMARKS.md), with raw runs in
`benchmark_snapshot.json`.

### Endurance / leak probe

A separate long-running harness samples persistent population, transient scratch
and Linux RSS (`null` on systems without `/proc/self/status`):

```bash
cargo run --release -p auxein-core --example endurance -- \
  f64 geometry 2000000 250000

cargo run --release -p auxein-core --example endurance -- \
  f64 predictive 2000000 250000

cargo run --release -p auxein-core --example endurance -- \
  f32 predictive 2000000 250000
```

For a stable learned workload, RSS and scratch should reach a plateau rather
than grow with elapsed steps. Allocator RSS may retain pages after an explicit
scratch compaction even though Auxein has released the corresponding Vec
capacity; the important long-running property is a stable post-peak plateau.
The certified 2M-step runs are summarized in [`ENDURANCE.md`](ENDURANCE.md),
with raw samples in `endurance_snapshot/`.

## Conformance

The Rust implementation is intended to match the Python semantic reference and
`spec/auxein.md`. The regression suite additionally checks `f32`/`f64`
projection, persistence, material accounting, numerical extremes, invariances
and private predictive state.

## License

See [`LICENSE`](LICENSE).
