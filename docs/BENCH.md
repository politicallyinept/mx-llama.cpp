# BENCH.md — Mistral-Large-123B fine-tune-v2 q4_0 on 3x MI50 32GB (gfx906, TheRock ROCm 7.2.3)

All runs: -lm dio, env GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1
HSA_XNACK=0 GPU_MAX_HW_QUEUES=8. Model 64.56GiB. r=1 (smoke) unless noted.

## P0 — smoke, deployed binary (Sep-4 @ 4b9524188), 2026-09-07

| build | sm | fa | pp512 | tg32 |
|---|---|---|---:|---:|
| deployed 4b9524188 | layer | 1 | 71.87 | 7.63 |
| deployed 4b9524188 | tensor | 1 | 90.71 | 16.72 |

**Finding: -sm tensor works on dense Mistral-Large fine-tune (loads, correct-looking perf) and is
+119% tg32 vs layer. Deployed llama-swap config uses layer — 2x decode left on table.**

Bandwidth context: layer tg ceiling ~44 t/s (1 card, 1TB/s, 23.1GB/token measured
bytes/token implied 7.63 -> ~12% eff... TBD with pp/tg detail); tensor tg ceiling
~46 t/s aggregate at 36% achieved. Tuning headroom confirmed.

## P1 — HEAD build (50d5fdeb), 2026-09-07 21:29-21:35

| build | sm | fa | pp512 | pp2048 | tg32 | tg128 |
|---|---|---|---:|---:|---:|---:|
| deployed 4b9524188 | layer | 1 | 71.87 | — | 7.63 | — |
| deployed 4b9524188 | tensor | 1 | 90.71 | — | 16.72 | — |
| HEAD 50d5fdeb | layer | 1 | 71.07 | 135.58 | 7.57 | 7.62 |
| HEAD 50d5fdeb | layer | 0 | 70.94 | — | — | 7.21 |
| HEAD 50d5fdeb | tensor | 1/0 | CRASHED (SIGSEGV) | | | |

**Crash root cause (gdb bt):** ggml_backend_cuda_comm_try_allreduce_nccl ->
librccl.so.1 -> libamdhip64.so.7 SIGSEGV. My build added -DGGML_HIP_RCCL=ON
(FEATURES.md rec); the deployed build has NO RCCL and never enters that path.
TheRock ROCm 7.2.3 RCCL appears broken on gfx906. Fix applied: rebuilt HEAD
without RCCL (custom AR covers our 48KB/layer allreduce). VERIFICATION PENDING —
a llama-server (Mistral-Large fine-tune, -sm layer, -fit on, -lm none, port 8080) took the GPUs
before the re-test could run.

**P5 env toggles: not yet run** (all segfaulted under the RCCL build; re-run on the
no-RCCL build).

fa0 vs fa1 (layer): pp 70.94 vs 71.07, tg128 7.21 vs 7.62 — fa=1 mildly better,
keep fa on.

## Quality gate — PASSED (deployed build, -c 32768, -st)
Layer and tensor both produce correct coherent output; wording differs slightly
(summation-order class, same as fork documents for repack paths). Gen rates:
layer 7.6 t/s, tensor 16.8 t/s with 32k ctx allocated.
