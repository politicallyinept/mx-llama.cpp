# Drift-Ladder Anchor Measurements -- Qwen3.5-35B-A3B on mi50

Date: 2026-10-27 (PDT). Box: mi50 (3x MI50 gfx906), system ROCm 7.2.3, TheRock runtime for fork tiers.

## Tier ladder

| Tier | Engine | Rev | Runtime | Notes |
|---|---|---|---|---|
| **T0** | stock upstream llama.cpp **b10760** | (2nd parent of merge "Merge tag 'b10760'"; tag object not present in mx clone, commit verified upstream) | system ROCm 7.2.3 | built fresh: `rig/build-ladder-t0` from `rig/ladder-t0-src` |
| **T1** | mx fork base | (b10760 + 45 mx commits, merged 2026-09-02) | system ROCm 7.2.3 | built fresh: `rig/build-ladder-t1` |
| **T2** | CV fork | (`rig/castagna-veloce`) | system ROCm, `LLAMA_CKPT_LAST_VERIFY=1 GGML_CUDA_MOE_VEC=1`, `-ngl 999` | `rig/cv-h2h-v5/build-cv` |
| **T3** | INT4 fork pre-CV (**RED LINE**) | (`rig/p0-final`) | w4a4-env (TheRock, ldd-witnessed `libamdhip64.so.7` from `_rocm_sdk_core`), `REPACK_Q4_0=1` on q40 | `rig/build-p0f` |
| **T4** | current stacked | (source tree `rig/stacked-src`, not a git checkout -- rev per campaign record) | w4a4-env + campaign fork gates, `REPACK_Q8_0` unset | `rig/build-stacked` |

Build recipe (T0/T1): `Release GGML_HIP=ON AMDGPU_TARGETS=gfx906 BUILD_SHARED_LIBS=ON GGML_CUDA_FA_ALL_QUANTS=OFF LLAMA_CURL=OFF`, `nice -n 19 -j 8`. Stock b10760 built cleanly on mi50 -- no fallback commit needed.

## Protocol

- Models: `unsloth Qwen3.5-35B-A3B-UD-Q4_K_XL` (ud) and `bartowski Q4_0` (q40) from `bench-quants/`.
- Per tier x quant: one `llama-server` leg, single card (`HIP_VISIBLE_DEVICES=0`, tp1), `-c 16384 -b 4096 -ub 1024 -t 16 -np 1 -fa on`, `-ngl 99` (999 for T2; `-fit off -lm dio` added for T3/T4 fork builds only). Every leg under `flock rig/gpu.lock`, TERM-only teardown, all GPUs verified 0% after campaign.
- 6 fixed prompts (`c1,c2,c3,long-pp-r01..r03`, long-pp ~4.6-4.8 KB ~ ~1.2-1.5k tok, from `rig/c4-attribution/prompts/` -- verbatim copies of the harness long-pp cases).
- Per request: `/v1/chat/completions`, `temperature=0, seed=42, top_k=1, max_tokens=512, logprobs=true, top_logprobs=20`. All legs generated full 512 except q40/c1 (~340, EOS).
- Raw logprobs JSON: `rig/drift-ladder/raw/<tier>-<quant>-<case>.json` (60 files).

## Metrics & approximations

- **KL**: per position, renormalized over the union of both top-20 sets; a token missing from a side's top-20 gets floor prob 1e-8. Top-k truncation makes this an approximation (lower bound on tail mass).
- **KL(prefix)** = mean KL over positions up to and including first top-1 divergence -- the only context-valid region under greedy rollout (after divergence the two models condition on different tokens). **KL(all)** is reported for completeness but is dominated by context mismatch (~10-17 nats) and should not be read as model drift.
- **top1-agree%** = position-wise top-1 token match over the full rollout (post-divergence positions are context-mismatched; low values are expected).
- **first-div-token** = first position where top-1 differs (-1 = never).
- **|Deltalogp| top-1** = mean absolute delta of the top-1 token's logprob (log-softmax proxy for logit delta, not raw logits).

## Ladder table (mean over 6 cases, vs T0)

| Quant | Tier | mean KL(prefix) | top1-agree % | median first-div | mean \|Deltalogp\| top-1 |
|---|---|---:|---:|---:|---:|
| ud | T1 (mx base) | 0.00200 | 22.5 | 78 | 0.0066 |
| ud | T2 (CV) | 0.00340 | 22.2 | 123 | 0.0083 |
| ud | T3 (pre-CV, RED LINE) | 0.00348 | 30.2 | 187 | 0.0086 |
| ud | T4 (stacked) | 0.01140 | 33.4 | 202 | 0.0067 |
| q40 | T1 (mx base) | 0.00910 | 28.6 | 85 | 0.0129 |
| q40 | T2 (CV) | 0.00664 | 29.1 | 76 | 0.0107 |
| q40 | T3 (pre-CV, RED LINE) | 0.00809 | 27.0 | 75 | 0.0135 |
| q40 | T4 (stacked) | 0.00399 | 17.2 | 76 | 0.0122 |

## Per-case detail


quant tier case n firstdiv agree% KL(pre) KL(all) |dlogp|
ud t1 c1 512 78 19.14 0.000410 13.955 0.00328
ud t1 c2 512 17 4.30 0.000603 16.875 0.00599
ud t1 c3 512 295 58.59 0.006029 7.310 0.00372
ud t1 long-pp-r01 512 71 14.26 0.001714 15.130 0.01051
ud t1 long-pp-r02 512 169 33.40 0.002202 11.576 0.01300
ud t1 long-pp-r03 512 22 5.47 0.001028 16.667 0.00283
ud t2 c1 512 78 19.14 0.000230 13.954 0.00286
ud t2 c2 512 17 4.10 0.002548 16.893 0.00586
ud t2 c3 512 187 38.67 0.009660 10.839 0.00347
ud t2 long-pp-r01 512 123 24.02 0.002987 13.404 0.01697
ud t2 long-pp-r02 512 215 42.77 0.002635 9.948 0.01529
ud t2 long-pp-r03 512 22 4.69 0.002321 16.885 0.00537
ud t3 c1 512 383 77.15 0.000449 3.857 0.00322
ud t3 c2 512 17 5.27 0.001899 16.583 0.00261
ud t3 c3 512 187 38.87 0.010427 10.770 0.00292
ud t3 long-pp-r01 512 71 15.23 0.002229 14.790 0.01256
ud t3 long-pp-r02 512 202 39.84 0.003257 10.586 0.01670
ud t3 long-pp-r03 512 22 5.08 0.002610 16.676 0.01352
ud t4 c1 512 383 77.15 0.000319 3.824 0.00262
ud t4 c2 512 17 5.27 0.003269 16.689 0.00664
ud t4 c3 512 299 58.40 0.058758 7.171 0.00501
ud t4 long-pp-r01 512 71 14.45 0.001727 15.131 0.00899
ud t4 long-pp-r02 512 202 39.45 0.002744 10.564 0.01602
ud t4 long-pp-r03 512 22 5.47 0.001576 16.820 0.00111
q40 t1 c1 340 316 94.12 0.026749 0.893 0.00573
q40 t1 c2 512 75 15.23 0.001697 14.832 0.00840
q40 t1 c3 512 91 19.14 0.009431 14.115 0.00730
q40 t1 long-pp-r01 512 62 12.30 0.003342 15.386 0.01450
q40 t1 long-pp-r02 512 85 25.00 0.003095 13.063 0.01747
q40 t1 long-pp-r03 512 22 5.66 0.010267 16.835 0.02404
q40 t2 c1 342 -1 100.00 0.011034 0.011 0.00516
q40 t2 c2 512 75 15.43 0.001303 14.651 0.00642
q40 t2 c3 512 76 16.21 0.014505 14.842 0.00569
q40 t2 long-pp-r01 512 62 12.89 0.001914 15.356 0.00955
q40 t2 long-pp-r02 512 127 25.59 0.003878 12.994 0.01666
q40 t2 long-pp-r03 512 22 4.69 0.007183 16.887 0.02066
q40 t3 c1 342 -1 100.00 0.025554 0.026 0.00764
q40 t3 c2 512 75 15.82 0.004221 14.571 0.01110
q40 t3 c3 512 76 19.92 0.005229 14.086 0.00396
q40 t3 long-pp-r01 512 62 13.28 0.005315 15.080 0.02116
q40 t3 long-pp-r02 512 35 8.01 0.004119 16.149 0.01792
q40 t3 long-pp-r03 512 22 5.08 0.004098 16.874 0.01900
q40 t4 c1 342 80 28.65 0.000346 12.659 0.00368
q40 t4 c2 512 75 15.04 0.002890 14.921 0.00976
q40 t4 c3 512 76 16.21 0.008191 14.835 0.00533
q40 t4 long-pp-r01 512 62 12.70 0.002394 15.042 0.01671
q40 t4 long-pp-r02 512 85 24.80 0.004545 13.098 0.02272
q40 t4 long-pp-r03 512 22 5.86 0.005588 16.678 0.01475


## Findings

1. **No tier is numerically neutral vs stock T0.** Even T1 (mx base = b10760 + 45 mx commits) diverges: median first-div 78-85 tokens, KL(prefix) 0.002-0.009. The 45 mx commits are not bit-preserving on this model.
2. **UD-Q4_K_XL ladder is monotone in KL**: T1 (0.0020) < T2 (0.0034) ~ T3 (0.0035) < T4 (0.0114). The stacked tier adds ~3x the drift of the pre-CV red line on ud -- the largest single step in the ladder.
3. **Q4_0 ladder is NOT monotone**: T4 (0.0040) is closer to stock than T1 (0.0091). T3/T4 q40 legs run the W4A4 repack path (`REPACK_Q4_0=1`), a different kernel family from stock dequant -- kernel-family differences dominate the fork-lineage ordering on q40. Notably on q40/c1, **T2 and T3 reproduce T0's full 342-token rollout exactly** (top1-agree 100%, KL(all) ~ 0.01-0.03) while T4 diverges at token 80 -- the stacked tier changed q40 numerics relative to the red line.
4. **Prompt sensitivity is strong and shared**: `long-pp-r03` diverges at token 22 in every tier x quant pair (a near-tie top-1 there); `c2` at token 17 (ud) / 75 (q40). `c1`/`c3` are far more robust (up to 383 tokens of agreement).
5. **Magnitudes**: within the context-valid prefix, all tiers sit at KL  0.01 nats mean with top-1 logprob deltas  0.02 -- distributions are near-identical pre-divergence; the visible "drift" is mostly greedy rollout amplification of tiny numerical differences, not large per-token distribution shift.
6. **Red-line status (T3)**: T3's drift vs stock is in the same band as T1/T2 -- the pre-CV W4A4+P0 lineage does not add measurable distributional drift beyond the fork-base + kernel-repath level on this model. The ud-side step from T3->T4 (0.0035->0.0114) is attributable to the stacked (post-CV) changes.

## Artifacts

- Raw logprobs: `rig/drift-ladder/raw/<tier>-<quant>-<case>.json` (60 files)
- Per-case metrics: `rig/drift-ladder/metrics.json`
- Server logs + runtime witnesses (env, cmdline, resolved libamdhip64 per leg): `rig/drift-ladder/logs/`
- Scripts: `rig/drift-ladder/run_leg.sh`, `rig/drift-ladder/analyze.py`; builds `rig/build-ladder-t0`, `rig/build-ladder-t1` (sources `rig/ladder-t0-src`, `rig/ladder-t1-src` -- git worktrees of `/mx-llama.cpp`)

Caveats: KL is top-20-truncated with 1e-8 floor renormalization (approximation); |Deltalogp| is a log-softmax proxy, not raw logits; T4 source rev not verifiable from the build tree (`rig/stacked-src` is not a git repo) -- binary + campaign env witnessed instead; a concurrent tp2 campaign shared `rig/gpu.lock` (serialized cleanly, its `tp2-*` files also landed in `raw/` -- excluded from analysis).
