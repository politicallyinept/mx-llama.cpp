# TIER3-DECODEV2 -- TP1-UD decode kernel-efficiency teardown v2 (fork build-final2 vs CV)

**Date:** 2026-10-31 (PDT) - **Box:** mi50, GPU0 only - **Model:** Qwen3.5-35B-A3B UD-Q4_K_XL, tp1
**Trees:** fork `rig/anchor-final-src` @ (binary `rig/build-final2`, tier3-final2 tp1 envs exactly) vs CV `rig/cv-h2h-v5/build-cv` (tp1-decode-teardown envs exactly).
**ANALYSIS-ONLY: no builds, no gate flips, no commits.**

## VERDICT

**The "same kernels, slower on fork" hypothesis is REFUTED.** Bare gap reproduced: fork 105.03 t/s (9.52 ms/step) vs CV 109.09 (9.17) = **+0.35 ms/step (-3.9%)**. But traced GPU-busy: **fork 13.93 vs CV 14.21 ms/step -- fork busy is 0.28 ms/step LOWER**, and on every shared kernel the fork is equal or faster (main `mul_mat_vec_q<q8_0,1,f,f,f>` -6.9% avg, `topk_moe_sorted_gcn` -20.4%, `rms_norm_f32_narrow` -8..-12%, `gated_delta_net` -6% vs CV's decode kernel). Only three same-kernel regressions exist and all are <=0.03 ms/step (gcn_f32_mv +7.6% = +0.021; flash_attn_tile +15% = +0.030; mmv_q<8,1,f,t,f> +4.4% = +0.011).

**The entire remaining gap is CPU-side / launch-structure: fork +71.4 launches/step (1373.9 vs 1302.5).** In-graph gaps are ~0 (median 0.15 us both engines); the traced >1 ms gap class (CPU stalls between graph replays) is +2.1 ms/step worse on fork. Wall gap 0.35 + busy advantage -0.28  +0.63 ms/step of extra CPU-side time ~ 8.8 us x 71 extra graph nodes/step.

**Root cause: the anchor-final `attn_qk_fused` default-ON flip is DEAD CODE on this model.** The fork trace contains **zero** `attn_qk_fused` kernels (CV: 10/step) and no `[attn-qk-fused] ACTIVE` banner -- the matcher at rope.cu:1101 rejects the chain before launch. Mode 40 = `GGML_ROPE_TYPE_IMROPE` (8|32) **passes** the mode gate (`mode & MROPE(8)` true, kernel has `a.imrope` support), so a different predicate rejects -- prime suspect: the 8-node window shape (only one MUL_MAT allowed at slot n==3; Qwen3.5 decode puts the v-projection MUL_MAT and/or the sigmoid attn-gate node inside the window) or the `pq->src[1]->ne[0]==4*T` pos-tensor requirement. The anchor's "byte-clean 0/28" witness was **vacuous** (nothing fired), same failure mode as the fusion-matcher lane (fusion-gpu report: "Launch-count delta 0").

**Launch-count exoneration is invalid.** The fusion-matcher A/B that "proved" launches perf-neutral never fired (zero MATCH events, zero launches removed) -- it tested nothing. This trace re-convicts launch structure as the sole remaining gap.

## Launch-delta composition (+71.4/step)

| Deltalaunch | fork | CV | note |
|---:|---:|---:|---|
| +40 | rope_multix20 + k_set_rowsx20 + rms_norm_f32<256>x10 + unary sigmoidx10 (60) | attn_qk_fusedx10 + hc_attn_gatex10 (20) | **attn-qk chain unfired** |
| +10 | quantize_q8_1 162 | 152 | CV's hc_attn_gate absorbs the k-cache quantize (signature takes block_q8_1*) |
| +10 | rms_norm_f32_narrow<true> 40 | 30 | absorbed by CV attn_qk_fused |
| +10 | cpy_scalar 40 | 30 | absorbed by CV attn_qk_fused |
| +2.4 | mul_mat_vec_q<q8_0,4,f,f> 2.4 | 0 | fork-only 4-row path (last-verify) |
| +1.1/-3 | k_bin_bcast +1.1, flash variants, misc | | |

## Per-op table (traced, ms/step, fork vs CV; top by |Delta|)

| fork ms/st | CV ms/st | Delta | f/cnt | c/cnt | f avg us | c avg us | kernel |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4.280 | 4.607 | -0.327 | 180.6 | 181.0 | 23.7 | 25.5 | mul_mat_vec_q<q8_0,1,f,f,f> -- fork -6.9% |
| 1.379 | 1.372 | +0.007 | 39.0 | 39.0 | 35.4 | 35.2 | moe_coal_kq_ls<16,1> -- parity |
| 0.895 | 0.858 | +0.037 | 162.0 | 152.0 | 5.5 | 5.6 | quantize_q8_1 (fork <32>) -- +10 launches |
| 0.780 | 0.789 | -0.009 | 39.0 | 39.0 | 20.0 | 20.2 | moe_coal_kq_ls<8,1> -- parity |
| 0.647 | 0.679 | -0.033 | 48.6 | 50.0 | 13.3 | 13.6 | mul_mat_vec_q GLU variant -- fork faster |
| 0.557 | 0.568 | -0.012 | 101.1 | 100.0 | 5.5 | 5.7 | k_bin_bcast<add> -- parity |
| 0.457 | 0.503 | -0.047 | 81.0 | 81.0 | 5.6 | 6.2 | rms_norm_f32<1024> -- fork faster |
| 0.437 | 0.441 | -0.004 | 79.0 | 79.0 | 5.5 | 5.6 | moe_vec_quantize_gather -- parity |
| 0.366 | 0.460 | -0.094 | 40.0 | 40.0 | 9.2 | 11.5 | topk_moe_sorted_gcn<256> -- fork -20.4% |
| 0.346 | 0.366 | -0.020 | 30.0 | 30.0 | 11.5 | 12.2 | gated_delta_net_cuda<128,..,4> vs CV gated_delta_net_decode -- fork faster |
| 0.334 | 0.313 | +0.021 | 39.7 | 40.0 | 8.4 | 7.8 | gcn_f32_mv<1,4> -- **fork +7.6%** |
| 0.325 | 0.352 | -0.027 | 60.0 | 60.0 | 5.4 | 5.9 | rms_norm_f32_narrow<false> -- fork faster |
| 0.307 | 0.322 | -0.015 | 40.0 | 40.0 | 7.7 | 8.0 | shexp_gate_add -- parity |
| 0.305 | 0.295 | +0.011 | 39.7 | 40.0 | 7.7 | 7.4 | mul_mat_vec_q<8,1,f,t,f> -- fork +4.4% |
| 0.242 | 0.212 | +0.030 | 9.9 | 10.0 | 24.4 | 21.2 | flash_attn_tile -- **fork +15%** |
| 0.230 | 0.195 | +0.035 | 40.0 | 30.0 | 5.8 | 6.5 | rms_norm_f32_narrow<true> -- +10 launches |
| 0.221 | 0.175 | +0.046 | 40.0 | 30.0 | 5.5 | 5.8 | cpy_scalar -- +10 launches |
| 0.165 | -- | +0.165 | 20.0 | 0.0 | 8.3 | -- | rope_multi (unfused qk chain) |
| 0.163 | 0.173 | -0.010 | 29.8 | 30.0 | 5.5 | 5.8 | concat_cont vs CV concat_dim0_small -- parity |
| 0.110 | -- | +0.110 | 20.0 | 0.0 | 5.5 | -- | k_set_rows (unfused) |
| 0.057 | -- | +0.057 | 10.0 | 0.0 | 5.7 | -- | rms_norm_f32<256> (unfused q norm) |
| 0.055 | -- | +0.055 | 10.0 | 0.0 | 5.5 | -- | unary_gated sigmoid (attn gate, unfused) |
| 0.050 | -- | +0.050 | 2.4 | 0.0 | 20.8 | -- | mul_mat_vec_q<8,4,f,f> fork-only |
| -- | 0.076 | -0.076 | 0.0 | 10.0 | -- | 7.6 | CV attn_qk_fused<256> |
| -- | 0.054 | -0.054 | 0.0 | 10.0 | -- | 5.4 | CV hc_attn_gate (absorbs k-quantize) |
| -- | 0.127 | -0.127 | 0.0 | 1.5 | -- | 86.5 | CV ggml_cuda_copy_bytes (staged copy) |

Totals: fork busy 13.93, CV 14.21 ms/step; fork 1373.9, CV 1302.5 launches/step.

## Top-5 gaps ranked (fork - CV)

| rank | gap | busy Delta ms/step | % of 9.52 ms wall | launches | state |
|---|---|---:|---:|---:|---|
| 1 | **attn-qk chain unfired** (rope_multi/k_set_rows/rms_norm<256>/sigmoid + absorbed narrow<true>/cpy_scalar/quantize vs attn_qk_fused+hc_attn_gate) | +0.40 (core chain +0.26) | 4.2% (core 2.7%) | +70 | **ported, default-ON, NEVER FIRES** -- matcher rejects (mode 40 passes; shape/pos predicate fails) |
| 2 | mul_mat_vec_q<q8_0,4,f,f> fork-only 4-row path | +0.050 | 0.5% | +2.4 | fork-side verify-path dispatch differs from CV |
| 3 | flash_attn_tile +15% avg | +0.030 | 0.3% | 0 | fork dense-gcn variant vs CV 2-extra-template variant -- geometry retune |
| 4 | gcn_f32_mv +7.6% avg | +0.021 | 0.2% | 0 | she-fuse ported kernel, minor geometry delta |
| 5 | mul_mat_vec_q<8,1,f,t,f> +4.4% avg | +0.011 | 0.1% | 0 | trivial |

## Fix proposals (NOT authored, NOT flipped)

**F1 -- Make `attn_qk_fused` actually fire on Qwen3.5 decode (the whole ballgame).**
Add per-gate rejection debug to `ggml_cuda_attn_qk_fused` (rope.cu:1101) and dump the decode graph node sequence; identify the failing predicate. Mode 40/IMROPE passes the mode gate and the kernel already implements `a.imrope`, so the reject is structural: prime suspects (a) the 8-node window -- matcher allows exactly one MUL_MAT at slot n==3, but Qwen3.5 decode likely places the v-projection MUL_MAT and/or the sigmoid attn-gate node inside the window; (b) `pq->src[1]->ne[0]==4*T` pos-tensor shape under IMROPE. Extend the matcher window (allow 2 middle MUL_MATs / gate node outside) or the pos check accordingly.
**Expected recovery:** -40..-70 launches/step + busy -0.26..-0.40 ms/step  wall **~0.25-0.45 ms/step ~ +2.6-4.7%** -- closes most or all of the -3.9% deficit.
**Risk:** ordered -- the anchor's byte-clean witness was vacuous (kernel never ran); first real firing changes accumulation order of norm+rope+set_rows  expect greedy-seal churn, plan re-seal. Gates: seal suite, TP2 leg, graph-capture witness.

**F2 -- Close the flash_attn_tile +15% decode regression.**
Diff fork `fattn-tile.cu` dispatch vs CV (CV template carries 2 extra params -- dense-gcn/IDX knobs); retune decode tile geometry for head-256 single-token tiles. **Expected:** +0.03 ms/step (~0.3%). Only worth spending if F1 under-delivers; same for gcn_f32_mv +7.6% (+0.021).

## Method

- Bare legs: warmupx2 + 3xtg128, fork 105.20/105.03/104.98 (median 105.03, CV 0.1%), CV 108.80/109.09/109.38 (median 109.09) -- reproduces the reference -3.9%.
- Traced legs: `rocprofv3 --kernel-trace` (outer env), warmupx4 (8 tok) -> 3 s gap -> 128-token measured leg; decode cluster = last >10k-kernel cluster (`analyze.py`, tp1-decode-teardown convention, +avg_us per launch).
- Gap decomposition: in-graph gaps ~0 (median 0.15 us); >1 ms CPU-stall gaps: fork 909 (61.9 ms/step) vs CV 640 (60.7) under trace.
- Cell envs: fork = tier3-final2 tp1 driver envs exactly (no attn-qk env -- defaults are the point); CV = tp1-decode-teardown cv cell exactly.
- Artifacts: `traces/{fork,cv}-tp1-ud/*_kernel_trace.csv`, `raw/*-summary.json`, `raw/delta-fork-cv.txt`, `raw/*-bare*.log`, `launch.sh`, `leg.sh`, `bare.sh`, `driver.sh`, `analyze.py`, `delta.py`.

## Box state

All servers TERM-torn-down; gpu.lock free; GPUs 0% (verified post-run); no stray processes; nothing built, no gates flipped, no commits.
