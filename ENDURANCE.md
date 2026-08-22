# Endurance snapshot

Release-mode long-running snapshot for the production-hardened v0.5.0 Rust implementation. RSS values are Linux `/proc/self/status` samples and are allocator/host dependent.

| Workload | Steps | RSS start KiB | RSS plateau KiB | Scratch plateau bytes | Final persistent populations |
|---|---:|---:|---:|---:|---|
| geometry_f64 | 2000000 | 3008 | 3076 | 320 | L=1, C=1, Σ=0, Cᵀ=0, Σᵀ=0 |
| predictive_f64 | 2000000 | 3068 | 3076 | 320 | L=1, C=2, Σ=0, Cᵀ=2, Σᵀ=0 |
| predictive_f32 | 2000000 | 2992 | 3064 | 320 | L=1, C=2, Σ=0, Cᵀ=2, Σᵀ=0 |

All three stable workloads reached a flat RSS/scratch plateau after initial allocation; no growth proportional to elapsed steps was observed. Raw samples are in `endurance_snapshot/*.jsonl`.
