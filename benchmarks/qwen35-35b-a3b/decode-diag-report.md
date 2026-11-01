# DECODE-DIAG -- fork (build-p0f) vs CV, TP2, UD-Q4_K_XL kernel-level attribution

**Date:** 2026-10-26 (PDT) - **Box:** mi50 - **Cards:** ROCm0+ROCm1 only (GPU2 never touched -- verified 0% util throughout) - **Model:** `bench-quants/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`

## VERDICT

**The 4-5% UD decode gap belongs to CV: CV 113.4 t/s vs fork 108.5-109.2 t/s bare (CV +4.1%).** The rocprofv3 trace attributes fork's deficit to four ranked causes: (1) fork's shared-expert/dense-f32 path is ~2x CV's fused equivalent, (2) fork's W4A4 `q8_0_repacked` mmv is slower per-step than CV's plain q8_0 mmv on this quant, (3) fork launches ~440 more kernels/step (extra per-layer sigmoid/add/mul gating ops CV fuses), (4) the shared MOE_VEC kernels fire identically on both engines but run slower on fork. The MoE kernel family itself is NOT the difference -- same kernels, same counts per step on both sides.

## Method

- **Harness:** `llama-server` + completion requests (NOT llama-bench -- CV bench historically unreliable and TP+repack pool bug; server protocol used for both engines, stated per task).
- **Legs:** `rocprofv3 --kernel-trace` around full server run; warmup x4 (8 tok) -> 3 s gap -> **measured leg 128 decode steps** -> 3 s gap -> TERM teardown. Decode cluster isolated by >2 s gap clustering, last-big-cluster rule (`rig/decode-diag/analyze.py`, extends `tp2-decode-check/parse-trace.py`).
- **fork env** (`diag-launch.sh`, = `tp2-decode-check/launch.sh` witness): `w4a4-env.sh` + `GGML_GDN_COLTHREAD=1 GGML_CUDA_GCN_KQ_MOE=1 GGML_CUDA_REPACK_KQUANT=0 GGML_CUDA_MOE_VEC=1 GGML_CUDA_MOE_FUSE=1 GGML_CUDA_MOE_DEC_MAX_TOKENS=8 GGML_CUDA_FA_DENSE_GCN=1`; flags `-ngl 99 -fa on -fit off -sm tensor --device ROCm0,ROCm1 -lm dio -c 16384 -b 4096 -ub 1024 -t 16 -np 1`. Gate banners confirmed in `raw/fork-ud-server.log`.
- **CV env** (launch2.sh cv case, retargeted to cards 0,1): system ROCm, `HIP_VISIBLE_DEVICES=0,1`, `LLAMA_CKPT_LAST_VERIFY=1`, `GGML_CUDA_MOE_VEC=1`; same core flags, no `--device`/`-lm dio`/`-fit off` (per launch2.sh cv).
- **Discipline:** `flock rig/gpu.lock` per leg; PID-capture + process-group TERM/KILL teardown; `rocm-smi` 0% verified after every leg. GPU2 untouched.
- **Attempt history:** attempt 1 died to our own 9-min readiness timeout during the traced repack (rocprofv3 was tracing fine -- 2.2 GB CSV produced; the known-good tp2-q40 traced load also takes ~11 min). Attempt 2 with 20-min readiness succeeded. One orphaned server PID from attempt 1 was found and killed; GPUs verified clean before relaunch.
- **Q4_0 second arm: SKIPPED (budget).** Note: a fork Q4_0 TP2 trace already exists (`rig/tp2-decode-check/traces/tp2-q40/`) but has no CV counterpart, so no delta possible without a CV-q40 leg.

### Reference bare (untraced) decode, same day
| engine | tg128 bare | source |
|---|---|---|
| fork build-p0f | 108.5-109.2 t/s (9.22 ms/tok) | tp2-decode-check report (ROCm1,2) |
| CV | **113.4 t/s (8.82 ms/tok)** | this lane, ROCm0,1, `raw/cv-ud-bare.log` |

### Traced wall-clock (NOT comparable across engines -- see caveats)
fork 187.5 ms/token traced vs CV 19.2 ms/token traced. The asymmetry is fork's custom-AR **spin-wait** kernel (`k_broadcast_reduce<2>`, 160 launches/step whose traced duration is peer waiting, not work -- same artifact flagged in tp2-decode-check). All comparisons below **exclude TP-comm kernels**.

## Per-engine top kernels (measured cluster, ms/step, 128 steps, both cards summed)

**fork-ud** (423,606 kernels, 40 distinct, busy 186.2 ms/step; **23.7 ms/step excl. spin-AR**):
| ms/step | /step | kernel |
|---|---|---|
| 162.445 | 160.0 | `k_broadcast_reduce<2>` (custom-AR spin -- EXCLUDED) |
| 5.366 | 456.4 | `mul_mat_vec_q8_0_repacked<16,16,false,64,false>` |
| 1.741 | 158.8 | `mul_mat_vec_f<float,1,128>` |
| 1.630 | 324.0 | `quantize_q8_1<32>` |
| 1.539 | 300.0 | `k_bin_bcast<op_add>` |
| 1.343 | 78.0 | `moe_coal_kq_ls<false,false,true,16,1>` |
| 0.997 | 2.0 | `mul_mat_vec_q<q8_0,1>` |
| 0.882 | 162.0 | `rms_norm_f32<1024,true,false,false,false>` |
| 0.867 | 78.0 | `moe_coal_kq_ls<true,true,false,8,1>` |
| 0.822 | 160.5 | `k_bin_bcast<op_mul>` |
| 0.809 | 158.0 | `moe_vec_quantize_gather` |
| 0.743 | 80.0 | `topk_moe_sorted_gcn<256>` |
| 0.705 | 140.0 | `unary_op_kernel<op_sigmoid>` |
| 0.632 | 120.0 | `rms_norm_f32_narrow<false>` |
| 0.559 | 79.4 | `mul_mat_vec_q8_0_repacked<16,16,false,64,true>` |

**cv-ud** (367,192 kernels, 33 distinct, busy 20.5 ms/step; **15.8 ms/step excl. p2p-AR**):
| ms/step | /step | kernel |
|---|---|---|
| 4.742 | 160.0 | `ggml_cuda_ar_p2p_kernel` (CV AR -- EXCLUDED) |
| 3.724 | 382.0 | `mul_mat_vec_q<q8_0,1,false,false,false>` |
| 1.315 | 304.0 | `quantize_q8_1` (non-templated) |
| 0.965 | 78.0 | `moe_coal_kq_ls<false,false,true,16,1>` |
| 0.883 | 220.0 | `k_bin_bcast<op_add>` |
| 0.726 | 162.0 | `rms_norm_f32<1024,true,false>` |
| 0.672 | 158.0 | `moe_vec_quantize_gather` |
| 0.666 | 78.0 | `moe_coal_kq_ls<true,true,false,8,1>` |
| 0.645 | 80.0 | `topk_moe_sorted_gcn<256>` |
| 0.546 | 80.0 | `mul_mat_vec_q<q8_0,1,true,false,false>` |
| 0.517 | 120.0 | `rms_norm_f32_narrow<false>` |
| 0.400 | 80.0 | `mul_mat_vec_q<q8_0,1,false,true,false>` |
| 0.398 | 80.0 | `gcn_f32_mv<1,4>` |
| 0.393 | 80.0 | `shexp_gate_add` |
| 0.350 | 80.0 | `k_bin_bcast<op_mul>` |

## Delta table (fork - CV, ms/step; full table in `raw/delta-fork-cv-ud.txt`)

| Delta ms/step | fork | CV | f/cnt /step | class | kernel / role |
|---|---|---|---|---|---|
| +162.4 | 162.4 | -- | 160/160 | mechanism-diff | fork spin-AR `k_broadcast_reduce` vs CV `ar_p2p` (-4.7) -- traced spin is wait, not work; NOT a port target |
| +1.26 | 5.93 | 4.67 | 456/542 | same-role pair, fork slower | fork `mul_mat_vec_q8_0_repacked` (W4A4 u4) vs CV `mul_mat_vec_q<q8_0>` variants |
| +0.95 | 1.74 | 0.79 | 159/160 | CV-only fusion (port target) | fork `mul_mat_vec_f` vs CV `gcn_f32_mv`+`shexp_gate_add` (shared-expert path) |
| +0.58 | 2.21 | 1.63 | 78/78 | same-kernel-slower | `moe_coal_kq_ls` x2 variants -- identical kernel+count, fork slower |
| +0.46 | 0.71 | 0.25 | 140/60 | kernel-count diff | `unary_op_kernel<op_sigmoid>` -- fork launches 80 extra sigmoids/step (per-layer gating unfused) |
| +0.32 | 1.63 | 1.32 | 324/304 | same-kernel-slower | `quantize_q8_1<32>` (fork) vs `quantize_q8_1` (CV) |
| +0.25 | 0.35 | 0.10 | 60/60 | same-kernel-slower | `k_get_rows_float_vec` (fork) vs `k_get_rows_float` (CV) |
| +0.16 | 0.88 | 0.73 | 162/162 | same-kernel-slower | `rms_norm_f32` template variant |
| +0.14 | 0.48 | 0.34 | 20/20 | same-kernel-slower | `flash_attn_tile` variant (attention decode) |
| +0.10 | 0.74 | 0.65 | 80/80 | same-kernel-slower | `topk_moe_sorted_gcn<256>` (router -- identical kernel, fork slower) |
| +0.09 | 0.35 | 0.26 | 60/60 | same-kernel-slower | `gated_delta_net_cuda<128>` (fork) vs `gated_delta_net_decode` (CV) |
| +0.04 | 0.31 | 0.26 | 60/60 | same-kernel-slower | `concat_cont` (fork) vs `concat_dim0_small` (CV) |
| +0.66 | 1.54 | 0.88 | 300/220 | kernel-count diff | `k_bin_bcast<op_add>` -- fork +80 add-bcasts/step |
| +0.47 | 0.82 | 0.35 | 160/80 | kernel-count diff | `k_bin_bcast<op_mul>` -- fork 2x CV mul-bcasts/step |

**Kernel-count structure:** fork 3,309 launches/step vs CV 2,869 (+440). The excess is concentrated in per-layer gating ops (sigmoid +80, add-bcast +80, mul-bcast +80) that CV fuses into `shexp_gate_add`/router path.

**MOE_VEC family verification (task-critical):** `moe_coal_kq_ls` (both template variants), `moe_vec_quantize_gather`, `topk_moe_sorted_gcn` fire on BOTH engines at IDENTICAL counts/step (78/78, 158/158, 80/80). The MoE decode kernel family is shared -- it is not the source of the gap; the gap is in the dense/shared-expert path, the W4A4 mmv, launch structure, and small-kernel efficiency.

## Ranked port targets (CV -> fork), estimated savings

Bare gap is 0.40 ms/token; traced serialized deltas are upper bounds (trace serializes dispatches, inflating small-kernel cost). Ranked by traced Delta:

1. **Shared-expert dense path fusion** -- port CV `gcn_f32_mv<1,4>` + `shexp_gate_add` to replace fork's `mul_mat_vec_f` + separate gate add/mul/sigmoid ops. **Est. -0.9 to -1.9 ms/step traced** (mul_mat_vec_f Delta0.95 + most of the +1.6 gating-launch excess). Biggest structural win; also removes ~240 launches/step.
2. **W4A4 mmv on K-quant models** -- fork `mul_mat_vec_q8_0_repacked` (5.93 ms/step) vs CV plain q8_0 mmv (4.67). On UD (REPACK_KQUANT=0) the repacked-u4 mmv path is net-slower than CV's q8_0 path. **Est. -1.3 ms/step traced** if the repack path is retuned or bypassed for K-quant models (check whether `REPACK_Q4_0` admission should be gated off for UD-class files).
3. **`moe_coal_kq_ls` same-kernel-slower (+0.58)** -- identical kernel, identical counts, fork 36% slower. Investigate launch config / graph shape / clock state differences vs CV. **Est. -0.6 ms/step.**
4. **`quantize_q8_1` variant (+0.32)** -- fork's `<32>`-templated variant slower than CV's plain one at similar counts. **Est. -0.3 ms/step.**
5. **Small same-kernel deltas (+0.5 total)** -- `rms_norm_f32` template (+0.16), `topk_moe_sorted_gcn` (+0.10), `flash_attn_tile` (+0.14), `k_get_rows` (+0.25), `gated_delta_net` (+0.09). Individually small; batch-audit launch shapes.

## Caveats

- Single traced leg per engine; traced times are serialized GPU busy time (no dispatch overlap) -- good for attribution ranking, not for absolute ms/token.
- fork spin-AR (`k_broadcast_reduce`) traced duration is peer-wait (162 ms/step) and is excluded from all work comparisons; in bare execution fork's custom AR is not a deficit.
- Trace overhead is asymmetric (fork 187.5 vs CV 19.2 ms/token traced wall) purely due to that spin kernel; bare numbers are the ground truth for the gap.
- Q4_0 arm not run (budget); fork-q40 trace exists in tp2-decode-check but lacks a CV counterpart.

## Artifacts (on mi50)

- `traces/fork-ud/fork-ud_kernel_trace.csv` (6.2 GB), `traces/cv-ud/cv-ud_kernel_trace.csv` (178 MB)
- `raw/fork-ud-summary.json`, `raw/cv-ud-summary.json` -- full per-kernel aggregates (all 40/33 kernels)
- `raw/delta-fork-cv-ud.txt` -- full delta table + category rollup
- `raw/*-server.log`, `raw/cv-ud-bare.log` -- server witnesses incl. gate banners and bare timings
- `analyze.py` (gap-cluster isolation), `delta.py` (delta+classification), `diag-launch.sh`, `diag-leg.sh`
