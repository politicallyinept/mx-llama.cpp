# Task E6 Ledger -- E4 re-scoped step 1c: HAS_IDS (mul_mat_id) Q8_0 expert GEMM bit-safety

Date: 2026-11-11 (PST). Worktree /root/int4-noquant-wt, branch INT4-noquant @ 67a0fbd571.
Measurement only: no kernel code was modified. Closes E1 residual check #3
(taskE1-LEDGER.md "Implications for step 1") and E4 plan Risk 3 (HAS_IDS un-proven).

## Question
The repacked HAS_IDS (MoE expert grouped) GEMM shares the repacked GEMM implementation
with the dense path per the E4 plan, but E1's 171-shape grid covered only the dense
launch. MoE serving runs experts at routed-token widths (9..31 = spec-decode verify
class). Does the repacked mmq_gemm_repacked*<true,...> expert GEMM diverge from the
stock mmq-id path (mul_mat_q with ids_dst/expert_bounds) on identical inputs?

## Does HAS_IDS even route through the repack implementation? YES.
Trace (this tree):
- ggml-cuda.cu:3026-3028 -- `ggml_cuda_mul_mat_id` fires the repack path first:
  `if (ggml_cuda_repack_mul_mat_should_fire(src0)) { ggml_cuda_mul_mat_id_repacked(...); return; }`
  (should_fire = tensor on a repack buffer, repack-common.cu:285; Q8_0 admission is the
  GGML_CUDA_REPACK_Q8_0 gate at load time).
- q8_repack/mul-mat-id.cu:80 `ggml_cuda_mul_mat_id_repacked` dispatch by routed-token width:
  - n_tokens == 1: repacked mat-vec (mul_mat_vec_q8_0_repacked / mul_mat_vec_rp), :320+
  - 2 <= n_tokens <= 8 (MMQ_RP_Q8_MOE_MMV_MAX_TOKENS=8, repack-common.cuh:24):
    mul_mat_vec_repacked_id1 mat-vec, :207+
  - n_tokens > 8: repacked GEMM -- repack_tile_off<BN> + mmq_gemm_repacked_w32<true,...>
    (:468) or mmq_gemm_repacked<true,...> (:486); variant pick use_w32 = n_assign <
    2*BN_ID*ne02 (:450) -- w32 for every width in this grid.
- Stock side (gate off / canonical tensors), ggml-cuda.cu:3050-3080: ne2 <= 8 -> mmvq-id
  (mul_mat_vec_q); else gcn_kq_moe is consulted but Q8_0 is NOT in its type set
  (gcn-q8-gemm.cu:2289 type_ok = K-quant/i-quant only), so Q8_0 experts at width > 8 take
  ggml_cuda_mul_mat_q with ids (ggml-cuda.cu:3079) -> the mmq-id HAS_IDS branch
  (mmq.cu:382-510: mm_ids_helper + compact-order q8_1 + mul_mat_q with ids_dst/
  expert_bounds + MoE tile map, mmq.cuh:1505-1528).
=> The runtime stock baseline for Q8_0 expert GEMMs IS mmq-id, and the widths this leg
   proves (9..32) are exactly the widths the runtime routes to the repacked GEMM.
   Widths 1..8 route to mat-vec arms on BOTH sides (not this leg; the id1 mat-vec arm is
   covered by the section-92 MoE gate and E2's nc 6-8 work).

## Method
Standalone microbench: ggml/src/ggml-cuda/q8_repack/mb_hasids_test.cu (same scaffolding
as E1's mb_e4_gemm_test.cu).
- Stock arm: exact mirror of ggml_cuda_mul_mat_q's HAS_IDS launch for gfx906 Q8_0 --
  J sweep with ncols_opt=ne12 and the ids J>64 skip (mmq.cu:246), GCN short-K MoE J=32
  override (K<2048 && n_assign<=16*ne02, mmq.cuh mul_mat_q_switch_J), non-stream-K
  launch with the MoE tile map (mmq_build_tile_map, default ON), ids_dst padded by J-1
  for the kernel's unconditional shared load (mirrors mmq.cu:419).
- Repacked arm: BOTH GEMM variants at every shape (w32 + 64-wide), repack_tile_off +
  weights via repack_q8_0_host per expert at repack_gcn_nbytes stride.
- Identical input bytes: src1 fp32 quantized ONCE via the host bit-mirror of
  quantize_mmq_q8_1<DS_LAYOUT_D4>; stock arm fed the COMPACT-order buffer (row a =
  q8_1 of src1 column ids_src1[a], what quantize_mmq_q8_1_cuda with ids_src1 produces),
  repacked arm the ORIGINAL-order buffer (quantize with ids=nullptr, mul-mat-id.cu:441)
  which the kernel indexes through ids_src1 (sCol[]). ids_src1/ids_dst/expert_bounds
  computed on the host with mm_ids_helper semantics (mmid.cu: stable counting sort,
  write_inverse=false forward map).
- Grid: n_tokens in {1,2,4,8,9,16,31,32} x n_expert_used in {1,2,4} x n_expert in
  {1,2,8} (1 = dense form: all assignments one expert; >1 = expert-scatter incl. empty
  experts) x K in {1024,1152,2048,2176,4096,5120,6144} (1152/2176: K%256=128 class)
  x M in {1024,2048} = 1008 shapes, + 6 real Qwen3-30B-A3B shapes (gate/up K=2048
  M=768, down K=768 M=2048, 64 experts, 4 used -- exercises the short-K J=32 override).
  J values exercised: 8/16/32 (the full set the runtime picks at these widths; gfx906
  MoE never takes J>64).
- Per-output fp32 bit-diff stock-vs-repacked (both variants) and w32-vs-64wide; each arm
  launched twice (launch determinism); independent ascending-K fmaf CPU reference on
  every shape with M*n_assign*K <= 2^26.
- Build/run: TheRock 7.15 (source /root/w4a4-env.sh), hipcc -O3 --offload-arch=gfx906,
  GPU2 under flock -w 600 /root/gpu.lock. Full output: docs/E6-hasids-results.csv.

## Results
- 1014/1014 shapes: stock mmq-id vs repacked HAS_IDS GEMM per-output fp32 bit-diff = 0
  (both repacked variants), including every K%256!=0 shape (288 rows), the dense form
  (n_expert=1), expert-scatter with empty experts, and the real 64-expert shapes.
- 1014/1014: all three arms launch-deterministic (two launches bit-identical).
- 548/548 shapes with CPU reference: all three kernels bit-identical to the independent
  ascending-K fmaf reference (ref_maxrel = 0) -- the outputs are real computed values,
  not untouched buffers.
- w32 vs 64-wide repacked variants: bit-identical at every shape (0 diff rows).

## Verdict
HAS_IDS routes through the repacked implementation (ggml-cuda.cu:3026 ->
q8_repack/mul-mat-id.cu), and the repacked Q8_0 expert GEMM is BIT-IDENTICAL to the
stock mmq-id path on identical input bytes at every tested routed-token width
(1..32, incl. the 9..31 spec-decode verify class), every expert-count/scatter pattern tested
(1/2/8/64 experts, empty experts included), and every K class tested (incl. K%256!=0
row padding and the short-K J=32 override). E4 plan Risk 3 (HAS_IDS un-proven -> MoE
un-gating not bit-safe) is CLOSED for the GEMM half: with E1's dense 171/171 and this
1014/1014, the repacked Q8_0 GEMM is bit-safe for MoE expert matmuls at the kernel
level under REPACK_Q8_0.

## Residual (NOT closed by this leg)
- Widths 1..8 route to mat-vec arms at runtime (repacked id1 mat-vec vs stock mmvq-id);
  this leg proved the GEMM kernels at those widths (superset) but not the runtime-
  selected mat-vec pair. E2 validated nc 6-8; the section-92 MoE gate covered decode
  end-to-end; a dedicated mmv-id bit-diff remains open if full default-ON needs it.
- Byte-generation checks from E1 (GPU repack_q8_0_kernel vs host permutation; repack vs
  gcn_kq family for dense) are separate legs; gcn_kq_moe does not serve Q8_0 (type set
  is K-quant/i-quant only), so no gcn_kq-vs-repack obligation arises for Q8_0 experts.
- Bit-identity here is kernel-level on identical activation bytes; end-to-end greedy
  bit-id on a real MoE model (validation-plan leg 3) is still the ship gate.

## Files
- ggml/src/ggml-cuda/q8_repack/mb_hasids_test.cu  (microbench, this commit)
- docs/E6-hasids-results.csv                      (1014-row grid output)
- taskE6-hasids-LEDGER.md                         (this file)
