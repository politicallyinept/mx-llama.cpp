# Q81-CACHE -- CV `mul_mat_vec_q_multi` + shared q8_1 activation cache, ported behind `GGML_CUDA_Q81_CACHE`

**Date:** 2026-10-27 (PDT) - **Box:** mi50 - **Branch:** `q81-cache` in `rig/q81cache-src` (baseline = stacked-src snapshot; commits port + test-only knobs, **not pushed**) - **Patch:** `rig/q81cache.patch` (366 lines) - **Build:** `rig/build-q81cache` (standard gfx906 recipe, fresh dir)

## Verdict line

**Byte-neutral PASS (both models, gate on + off, vs build-stacked) and prefix-KL 0.000000; kernel bit-identity additionally proven dynamically (FORCE mode, 400 real multi launches, byte-identical 32/256-token greedy output). But the gate is VACUOUS on Qwen3.5-35B-A3B: the graph never exposes consecutive MUL_MATs sharing src1 (fused attn_qkv + DFS-interleaved consumer chains), so tp1/tp2 perf delta is 0.00%. Adoption: keep the port, keep the gate OFF (default); redirect the candidate-#2 quantize-gap claim to #5c (quantize block-size retune).**

## 1. Mechanism diff summary

CV (`rig/castagna-veloce`, mmvq.cu/.cuh + ggml-cuda.cu):
- `mul_mat_vec_q_multi<type,ncols>`: N matrices of one type/K sharing one activation in one segmented-grid launch (block b in segment s for `blk0[s] <= b < blk0[s+1]`). K loop, row clamp, cross-warp shared-mem sum and `warp_reduce_sum` order are `mul_mat_vec_q`'s verbatim -> bit-identical per row by construction. **Ported verbatim** (kernel body diff-clean vs CV; only CV's `GGML_CUDA_MMVQ_MULTI_READ` "results are wrong" bandwidth experiment intentionally dropped).
- `ggml_cuda_try_fuse_mmvq_multi`: graph matcher -- consecutive MUL_MATs, same src1 ptr, same weight type in {q8_0,q4_K,q5_K,q6_K}, 1-4 cols, `should_use_mmvq` true, run shrinks before a member the next node consumes (protects gate/up GLU + bias fusions), dst/src overlap audit. Ported with two fork adaptations: defers to `GGML_CUDA_GCN_KQ_MV` when that gate is on; placed in `ggml_cuda_try_fuse` after the gdn-ab-dec/she-fuse chain checks, before the MUL_MAT_ID shared-expert block.
- `ggml_cuda_q8_1_activation` (CV's shared cache): **not ported -- the fork already has an equivalent**, `ggml_cuda_q8_1_cache_acquire` (default ON, keyed by tensor ptr + data ptr + variant + shape/strides, 2-entry LRU with try_alloc pressure handling, reset per graph eval, wired into mmvq/mmq/q8_repack). The multi launcher reuses it (variant 0), so the group's activation is quantized once per graph eval either way. CV's unlimited/any-type variant makes no quantize-count difference under per-layer locality.

Gates: `GGML_CUDA_Q81_CACHE=1` master (default **OFF** -- all new paths unreachable), `GGML_CUDA_Q81_CACHE_TRACE=1` per-launch line + per-node try/rej1..rej5 witnesses, `GGML_CUDA_Q81_CACHE_FORCE=1` **test-only** (run scan steps over pure view nodes + skips the shrink rule, which protects sibling fusions, not correctness), CV's `GGML_CUDA_MMVQ_MULTI_KQ` sub-knob kept.

## 2. Byte-match -- PASS

Greedy temp-0 seed-42 32-token `/completion` probe (TCP/UDP prompt), tp1 GPU0, fork gate bundle, vs `rig/build-stacked`:

| model | gate-off vs stacked | gate-on vs stacked |
|---|---|---|
| UD-Q4_K_XL | **IDENTICAL** | **IDENTICAL** |
| bartowski Q4_0 | **IDENTICAL** | **IDENTICAL** |

## 3. Prefix-KL spot -- 0.000000

logprobs top-20, 2 long-pp prompts (c4-attribution `long-pp-r01/r02`), 512 tokens, gate-off vs gate-on UD, drift-ladder KL math (FLOOR=1e-8, common prefix incl. first divergence):

| case | n | first_div | KL(prefix) |
|---|---|---|---|
| long-pp-r01 | 512 | none | **0.000000** |
| long-pp-r02 | 512 | none | **0.000000** |

## 4. Perf A/B (interleaved off/on, llama-bench `-r 3` per run, 2 cycles, UD)

**tp1 (GPU0), pp840 / tg256 t/s:**

| cycle | gate-off | gate-on |
|---|---|---|
| 1 | 2023.99 +/- 21.75 / 94.16 +/- 0.58 | 2014.94 +/- 13.14 / 94.07 +/- 0.41 |
| 2 | 2020.50 +/- 18.34 / 94.41 +/- 0.49 | 2015.12 +/- 20.16 / 94.13 +/- 0.38 |
| **median** | **2022.2 / 94.29** | **2015.0 / 94.10** |

**tp2 (`-tps 2`, TheRock runtime), pp840 / tg256 t/s:**

| cycle | gate-off | gate-on |
|---|---|---|
| 1 | 2076.47 +/- 8.89 / 87.17 +/- 0.39 | 2073.45 +/- 11.22 / 88.24 +/- 0.38 |
| 2 | 2081.30 +/- 12.23 / 83.68 +/- 5.00 | 2071.45 +/- 6.50 / 86.84 +/- 0.20 |
| **median** | **2078.9 / 85.4** | **2072.5 / 87.5** |

Deltas are run-order/thermal noise -- the gate is provably vacuous on this model (5), so the true delta is **0.00%**.

## 5. Why the gate is vacuous on Qwen3.5-35B-A3B (the real finding)

TRACE witness: **0 multi launches** across full UD and Q4_0 sessions (load, warmups, probes, 2x512-token logprobs). Rejection witnesses show the matcher evaluating every MUL_MAT and correctly rejecting:

- Dense attention uses the **fused `blk.N.attn_qkv.weight`** projection (verified in GGUF tensor table) -> one MUL_MAT, no q/k/v run.
- GDN layers: `wqkv` mm -> RESHAPE -> `wqkv_gate` mm -- separated by the qkv split/GDN consumer chain in graph order (DFS post-order interleaves each projection's consumer chain before the next projection is reached), not adjacent.
- Shared-expert gate/up pair IS adjacent (DFS adds both leaves consecutively) but the next node (GLU) consumes `up` -> the shrink rule correctly rejects it, preserving the GLU fusion (CV's own design).
- Router / o_proj / down_proj: single MUL_MATs; routed experts are MUL_MAT_ID (out of scope).
- Even FORCE-mode view-skipping finds no run on this graph (0 fires) -- the separation is by real compute nodes, not views.

So CV's win #2 as scoped for our workload ("removes per-layer redundant q8_1 requantization + batches projection GEMVs") does not apply to Qwen3.5-35B-A3B: the fork's existing default-on q8_1 cache already dedupes per-activation quantization, and there are no adjacent projection runs to batch. The decode-trace quantize-family gap (1.63 vs 1.32 ms/step) is attributable to candidate **#5c** (fork's gfx906 `quantize <32>` block size vs CV's plain block size), not to this mechanism. **Teardown correction: candidate #2 "fires-on-ours" = no for Qwen3.5-35B-A3B.**

## 6. Dynamic kernel bit-identity proof (FORCE mode)

Since the production matcher can't fire on this model, `GGML_CUDA_Q81_CACHE_FORCE=1` (test-only) was used to exercise the kernel on real weights/activations: it lets the run form across view nodes and skips the shrink rule, so the shexp gate/up pairs batch. UD session with FORCE: **400 multi launches** (`[q81-cache] multi n=2 type=8 K=2048 nc=2 ...`, q3_K shexp gate/up, incl. the nc=2 MTP-verify path and CUDA-graph capture/replay of the multi kernel):

| probe | gate-off vs FORCE-on |
|---|---|
| greedy 32-token | **IDENTICAL** (112 chars) |
| greedy 256-token | **IDENTICAL** (1222 chars) |

Combined with gate-off  build-stacked (2), the multi kernel output is byte-identical to the per-layer mmvq path on real UD data -- the teardown's "bit-identical by construction" claim holds empirically; no numerics were tolerated, the byte-match is exact.

## 7. Adoption call

- **Do not enable** `GGML_CUDA_Q81_CACHE` on the Qwen3.5-35B-A3B fleet: zero effect (vacuous), zero risk taken. Default stays OFF.
- **Keep the port** on branch `q81-cache`: byte-neutral by construction and verified both statically (kernel verbatim vs CV; fork `mul_mat_vec_q` K loop verbatim vs mx base) and dynamically (FORCE proof). It becomes live value for models whose graphs place sibling projections adjacently (CV's GLM-5-Next KDA class).
- **Redirect the #2 perf claim to #5c** (quantize block-size retune) -- that is where the fork's quantize-family decode gap actually is.
- Optional follow-up (not in scope, deviation from CV): a matcher variant that batches projections separated by pure-view nodes AND whose last member's consumer is a fusion sibling -- would unlock shexp gate/up batching on UD, but it trades the GLU fusion for the multi batching; needs its own perf A/B before anyone adopts it.

## Artifacts & hygiene

- Source: `rig/q81cache-src` (branch `q81-cache`, commits + , not pushed); patch `rig/q81cache.patch`
- Build: `rig/build-q81cache` (logs `rig/q81cache-build{,2,3,4,5}.log`)
- Raw: `rig/q81cache/raw/` (probe32/probe256/logprobs JSON per arm, bench-tp1.txt, bench-tp2.txt, server logs with trace witnesses); drivers `rig/q81cache-battery.sh`, `rig/q81cache-force.sh`, `rig/q81-envtest.sh`; analysis `rig/q81cache/analyze.py`; campaign log `rig/q81cache/campaign.log`
- All GPU legs ran under `flock rig/gpu.lock` (queued behind the bisect/she-fusion/c4 agents, never forced, TERM-only teardown). **Boxes checked clean after: all three cards ~10 MB VRAM idle, no q81cache servers/processes left.**
