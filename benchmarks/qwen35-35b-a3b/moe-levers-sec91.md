## 91. LEVER 2: gcn_kq_moe MoE launcher wired into mul_mat_id + Q3_K admission: +26-30% MoE prefill, decode untouched (2026-10-24, build-tp2, rig/moe-levers/)

The expert-grouped v4 MoE GEMM launcher (gcn-q8-gemm.cu, byte-equivalent to
CV's) was dead code in our tree - CV hooks it in ggml_cuda_mul_mat_id before
the MMQ branch and fuses gate/up+GLU at their ggml-cuda.cu:8039. Ported the
dispatch hook (after the MMVQ decode branch, before MMQ, CV pattern) + Q3_K
admission (UD routed gate/up are q3_K in 39/40 blocks; the v4 template
already supported it - dense path admits it). Env gates:
GGML_CUDA_GCN_KQ_MOE (default 1), GGML_CUDA_GCN_KQ_MOE_Q3K (default 1),
routing witness GGML_CUDA_GCN_KQ_MOE_TRACE=1 (first 12 launches, off by
default - traces OFF in measurement legs per standing policy). CV gate
predicates inherited verbatim: MIN_AVG=12 tokens/expert (decode stays on
MMVQ), BN=32 tiles when avg<=64 else 64, K%QK_K==0, alignment checks.
NOT ported (follow-up): CV's gate/up+SwiGLU graph fusion (launch-count cut).

MEASUREMENT (same-window same-binary env arms, server protocol, MTP off,
3 discarded warmups, medians of 3, UD-Q4_K_XL = Qwen3.5-35B-A3B):
| leg | pp512 OFF | pp512 ON | delta | tg128 OFF | tg128 ON |
|---|---|---|---|---|---|
| TP2 | 1315.4 | 1708.4 | +29.9% | 90.06 | 90.03 (neutral) |
| 1 GPU | 1040.5 | 1310.8 | +26.0% | 73.52 | 73.44 (neutral) |
Decode neutral as designed (MMVQ keeps decode; the gate only takes prefill).
UD TP2 pp512 1708 now EXCEEDS the Q4_0-u4 arm from section 90's vehicle class
(1620) on the K-quant file - the K-quant gap vs our own Q4_0 path is now
mostly closed on prefill.

CORRECTNESS GATES:
- 17x23 probe PASS on every leg (Q4_0 canon/u4, UD off/on, 1G + TP2).
- Q4_0 (hook must NOT fire): new binary byte-identical to pre-patch binary on
 4 temp-0 500-tok prompts, 0 hook launches witnessed. PASS.
- UD greedy: prompts with prefill < MIN_AVG gate byte-identical off vs on
 (gate correctly declines); 512-token prefill (avg 16/expert) diverges as
 expected for a numerics-changing kernel port; all outputs coherent.
- PPL spot check (single engine, same shape arms): llama-perplexity does NOT
 exercise this path - MoE MUL_MAT_ID in perplexity never passes the MIN_AVG
 gate at any tested shape (c512/ub512 and c2048/ub2048, n_seq 2/4: identical
 PPL off vs on, 0 MoE launches traced while the dense kq4 witness fires in
 the same process). Numerics evidence = greedy divergence + coherence +
 substantive long-prompt A/B + probe (documented, same bar as prior ports).
- Routing witness: 12 launches witnessed server-side: q3_K gate/up (K=2048
 M=512) + q4_K down (K=512 M=2048), experts=256, tokens=508, avg=16, bn=32.
Build note: box tree carried a pre-existing uncommitted lane2 hunk (dense
iq2_s mask, inert for all models tested); committed source excludes it.

