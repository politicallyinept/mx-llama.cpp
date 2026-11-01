# MoE baseline 2026-10-25 -- 3 builds x Qwen3.5-35B-A3B (mi50, TP2 ROCm1+2)

Harness: benchgate (workspace/endpoint-harness), append-only store. All measured requests via `harness.py perf`.
Server config (all legs): TP2 ROCm1+2, -ngl 99 (CV: 999), -fa on, -sm tensor, -c 16384 -b 4096 -ub 1024 -t 16 -np 1, mmap+dio (fork/stock: -lm dio), --host 0.0.0.0.
Workloads: pp840-dec256 / short-512 / tg256 (warm, as specified) + pp840-dec256-cold / short-512-cold (added cold variants -- warm-mode TTFT is prompt-cache-hit latency ~90-125 ms, NOT prefill; cold variants bust the cache and measure real prefill).
Warmup: 5 iterations / 6 measured, per workload (house GDN warm-ramp rule).

## Builds
- stock: rig/cv-h2h-v5/build-stock (mx-llama.cpp @ ), env: w4a4-env.sh + REPACK_Q4_0=1
- fork: rig/gcn-build-merged (perf/int4-next-20261021 @ ), env: w4a4-env.sh + GDN_COLTHREAD=1 GCN_KQ_MOE=1 FA_DENSE_GCN=1; Q4_0: +REPACK_Q4_0=1 W4A4_PROFILE=speed; UD: +REPACK_KQUANT=0 MOE_VEC=1
- cv: rig/cv-h2h-v5/build-cv (castagna-veloce @ ), stock env (no gates), HIP_VISIBLE_DEVICES=1,2, LLAMA_CKPT_LAST_VERIFY=1

## Results (latest GREEN per cell; pp t/s derived = tokens/TTFT_cold, includes ~120 ms request overhead)

| build | model | pp840 (cold TTFT -> t/s) | pp512 (cold TTFT -> t/s) | tg256 decode t/s | warm TTFT ms | run_ids pp840-cold / pp512-cold / tg256 |
|---|---|---|---|---|---|---|
| stock | Q4_0 | 580.4 ms -> 1448 | 523.0 ms -> 979 | 99.42 | 122.9 | f13b / c603 / 935b |
| fork | Q4_0 | 515.9 ms -> 1628 | 481.3 ms -> 1064 | 107.71 | 124.9 | 3669 / 0136 / 6f6a |
| cv | Q4_0 | 453.2 ms -> 1853 | 401.6 ms -> 1275 | 107.03 | 89.0 | e5d5 / fe35 / 3448 |
| stock | UD-Q4_K_XL | 620.9 ms -> 1353 | 559.0 ms -> 916 | 103.42 | 124.4 | 2401 / 1e68 / f2f6 |
| fork | UD-Q4_K_XL (moe-vec ON) | 537.5 ms -> 1563  | 483.0 ms -> 1060  | 107.63 | 124.7 | f67e / f131 / 16f8 |
| cv | UD-Q4_K_XL | 437.0 ms -> 1922 | 389.9 ms -> 1313 | 111.68 | 85.6 | 0b0a / 8928 / 4081 |
| fork | UD-Q4_K_XL (moe-vec OFF, mislabeled first run -- datapoint only) | 498.4 ms -> 1687 | 454.0 ms -> 1128 | 95.00 | 126.3 | dbd8 / b7a3 / d391 |

Full run_ids prefix 20261026T02xx/03xx (see store/index.csv; e.g. f67e = 20261026T030948Z-f67e).
 fork-ud cold runs recorded RED vs a high-water auto-seeded from the moe-vec-OFF runs; per Brad the high-water for
fork-mi50-tp2-q35-35b-ud-q4kxl|{pp840-dec256-cold, short-512-cold} was reset to the ON-config values (537.5/483.0)
with a note; human-directed exception to the never-touch-high-water rule.

## Anomalies / findings
1. Warm-mode TTFT (house workloads) is cache-hit latency (~90-125 ms flat across pp840/pp512/tg256) -- llama-server prompt cache hits on repeated identical prompts. Cold variants added as new workload files to measure real prefill.
2. First workload after model load is systematically contaminated (CV 3.9-23.9%) on fork and CV builds -- capture ramp exceeds the 5-warmup budget on the first shape. Fix: run a throwaway workload first; all final cells GREEN after reorder. Legs re-run for this: fork-q40 (pp840 warm + both cold), cv-q40 (pp840 warm), cv-ud (tg256), stock-ud (short-512 + both cold), fork-ud (full, also for moe-vec fix).
3. fork UD moe-vec OFF vs ON: decode 94.6 -> 106.8 (+13%, matches historical 108-class), but cold prefill TTFT +8% (498->537 pp840, 454->483 pp512), reproduced across 3 windows with CV <= 1.8% -- real, not noise.
4. CV build loads and serves Qwen3.5 fine; stock mx-llama.cpp @ also loads it (no arch failure).
5. Deadlocked foreign lock-holder found at campaign start: gpu-guard.sh (moevec-seal, 18:17 PDT) held rig/gpu.lock while moevec-seal.sh re-flocked the same lock in a child -> self-deadlock, 1 h no progress, empty log. Killed (logged to gpu-booking.log). gpu-guard double-flock bug worth fixing.
6. Q8_0 legs NOT run (wind-down called before optional third leg).

## Wall time
First leg start 02:14:42Z -> wind-down ~03:27Z = ~72 min.
- Plumbing false-starts ~ 10 min: pgrep self-match hang (~5), deadlocked foreign lock (~2), 127.0.0.1 bind fix (~2), profile-name mismatch (~1).
- Re-measure legs (contamination/ramp + moe-vec config fix) ~ 25 min.
- Productive measured legs ~ 35 min (6 combos x ~6 min incl. model load).
