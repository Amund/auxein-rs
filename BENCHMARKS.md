# Production benchmark snapshot

Machine-local release snapshot. Microsecond cases use medians and alternating baseline/production order. Absolute values are host-dependent; use this file as a regression reference, not a portable throughput promise.

## v0.5.0 baseline vs production

| Scenario | Baseline µs | Production µs | Change |
|---|---:|---:|---:|
| singleton | 0.379 | 0.364 | +3.9% |
| weighted-partial | 0.502 | 0.488 | +2.8% |
| pair-context | 1.000 | 0.891 | +10.9% |
| dense-4096-d32 | 1956.014 | 1475.989 | +24.5% |
| predictive-stable | 0.544 | 0.562 | -3.3% |
| predictive-sequence | 0.612 | 0.617 | -0.8% |

Positive change means production is faster.

## Production adversarial / learning cases

| Scenario | Median µs/presentation |
|---|---:|
| singleton-learn | 0.595 |
| pair-context-learn | 1.423 |
| dense-4096-d32-learn | 2319.676 |
| same-first-32768-d64 | 88.667 |
| sigma-idle-32768 | 0.350 |
| predictive-sequence-learn | 1.115 |
| predictive-fanout-2048 | 265.903 |
| predictive-fanout-8192 | 1217.258 |
| predictive-duplicate-fanout-8192 | 700.346 |
| temporal-outside-8192 | 0.649 |

Raw runs are retained in `benchmark_snapshot.json`.

## Predictive relative-gain correction check

Machine-local release comparison against the immediately preceding v0.5.0
production archive, using five alternating runs per tree after reusing the
already-computed current-context squared norm.

| Scenario | Previous v0.5.0 µs | Corrected v0.5.0 µs | Delta |
|---|---:|---:|---:|
| predictive-stable | 0.528 | 0.512 | -3.0% |
| predictive-fanout-8192 | 1071.573 | 974.450 | -9.1% |
| predictive-duplicate-fanout-8192 | 626.238 | 517.194 | -17.4% |
| temporal-outside-8192 | 0.578 | 0.591 | +2.2% |

Negative delta means the corrected tree is faster. These figures are a local
regression check, not a portable throughput claim. Raw runs are retained in
`predictive_gain_benchmark.json`.
