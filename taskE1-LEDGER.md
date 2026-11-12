# Task E1 Ledger -- E4 Step 0: TRUE mechanism of stock vs repacked Q8_0 prefill GEMM

Date: 2026-11-11 (PST). Worktree /root/int4-noquant-wt, branch INT4-noquant.
Plan: docs/E4-GEMM-restyle-plan.md, step 0 (ground truth before any kernel code).
Measurement only: no kernel code was modified.

## Question
The fork divergence map attributes the stock-vs-repacked Q8_0 prefill-GEMM divergence to
"stock splits K across tile lanes + shared reduce". That wording does not match the dp4a
MMQ path gfx906 actually runs (mmq.cuh:912, mmq-vec-dot.cuh:137-148: per-thread fp32
register accumulator, serial K, no shared reduce). Step 0 pins the true mechanism
empirically before any restyle is designed.

## Method
(a) Standalone microbench: ggml/src/ggml-cuda/q8_repack/mb_e4_gemm.cu
    - Stock arm: exact mirror of the ggml_cuda_mul_mat_q host launch for gfx906 dense
      Q8_0 (J sweep with CU-occupancy gate + shared-mem gate, fallback geometry for
      M<128; stream_k is always false on GCN -> non-stream-K launch).
    - Repacked arm: mmq_gemm_repacked_w32 (N<128) / mmq_gemm_repacked (N>=128), weights
      via the repack_q8_0_host byte permutation.
    - Identical input bytes: activations quantized ONCE via
      quantize_mmq_q8_1<DS_LAYOUT_D4> (the same kernel the runtime uses on both paths,
      quantize.cu:908); Q8_0 weight blocks shared between arms; zero tail mirrors pool
      padding (stock reads up to 8 blocks past the last kb0 row).
    - Per-output fp32 bit-diff; each kernel launched twice (launch determinism);
      independent CPU reference replicating the exact fp32 term sequence
      (ascending 32-blocks, acc = fmaf((float)d_w * d_x, (float)idot, acc)).
    - Grid: N(tokens) in {9,16,31,32,33,64,127,128,256} x K in {1024,2048,4096,5120,6144}
      x M in {2048,4096,5120} (135 shapes) + divergence-class grid M in {100,192,2048}
      x K in {1024,1152,2176,4096} (K%256=128 class) x N in {9,32,128} (36 shapes).
      J values exercised: 16/32/40/64/128; both repacked kernels; fallback geometry.
    - Build/run: TheRock 7.15 (source /root/w4a4-env.sh), hipcc -O3 --offload-arch=gfx906,
      GPU2 under flock -w 600 /root/gpu.lock. Full output: docs/E4-step0-results.csv.
(b) ISA forensics: carve .hip_fatbin (llvm-objcopy) -> clang-offload-bundler gfx906 code
    object -> llvm-objdump disassembly (gnu objdump emits EMPTY AMDGPU disasm; use the
    llvm one). Evidence: docs/E4-step0-isa-evidence.txt.

## Results
- 171/171 shapes: stock vs repacked per-output fp32 bit-diff = 0. NO diverging shape
  exists on the grid, including both divergence classes the map names (M<128 fallback
  geometry, K%256!=0 row padding).
- 171/171: both arms launch-deterministic (two launches bit-identical).
- 81/81 shapes with CPU reference: BOTH kernels bit-identical to the independent
  ascending-K fmaf reference (ref_maxrel = 0). The non-contracted reference form
  (mul+add instead of fma) deviates up to 2.8e-2 relative on cancellation-prone
  near-zero outputs -- exactly the FMA-contraction signature, shared by both kernels.
- ISA: both kernels emit the identical per-term pattern
      v_cvt_f32_i32 idot -> v_mul_f32 scaleprod(d_w,d_x) -> v_fmac_f32 acc, scaleprod, idot_f
  with an 8:1 v_dot4_i32_i8 : fp32-term ratio (stock J=64: 512 dot4 / 64 cvt / 64 mul /
  64 fma; repacked w32: 64 / 8 / 8 / 8). ZERO permlane/dpp instructions in either kernel:
  no cross-lane reduction anywhere. Stock's writeback stores the very v_fmac_f32
  accumulator registers (v49-v54) straight to global memory: per-thread fp32 register
  accumulator, serial ascending K, no shared reduce.

## Verdict
The divergence map's mechanism wording is REFUTED for the dp4a MMQ path. Stock MMQ and
the repacked Q8_0 GEMM are bit-identical on identical input bytes at every shape tested,
because both implement the same fp32 arithmetic: per-output per-thread fp32 accumulator,
serial ascending 32-value blocks, term update acc = fmaf((float)d_w * d_x, (float)idot,
acc), no K-split across lanes, no shared reduce, identical FMA contraction. The map's
"stock K-split + shared reduce" describes a geometry that does not exist in the code
gfx906 runs.
=> There is nothing to restyle at the GEMM-arithmetic level. Any runtime divergence seen
   in gate batteries must originate UPSTREAM of the GEMM (byte generation) or from a
   different kernel family -- not from the repack epilogue.

## Implications for step 1 (kernel-fix design)
- Repack admission can be widened on the bit-identity assumption at the GEMM level,
  conditional on the byte-generation checks below.
- Residual byte-generation checks (NOT GEMM arithmetic):
  1. repack_q8_0_kernel (GPU, q8_repack/buffer.cu:137) vs repack_q8_0_host byte
     equivalence: the microbench used the host permutation; runtime uses the GPU kernel.
     Comments claim a match (buffer.cu:136) but it is not directly verified here.
  2. The runtime stock baseline on non-yield shapes may be the gcn_kq GEMM family
     (gcn-q8-gemm.cu, own 8x4 register-tile geometry). If a gate battery's "stock"
     config runs gcn_kq, bit-identity requires repack == gcn_kq, not repack == mul_mat_q.
     Step 1 should bit-diff repack vs gcn_kq on the gate-battery shapes.
  3. HAS_IDS (mul_mat_id) expert GEMM path is out of scope for this grid (plan already
     marks it unproven).
- The E4 plan's epilogue-restyle work item should be re-scoped or dropped; replace it
  with the byte-generation checks above.

## Files
- ggml/src/ggml-cuda/q8_repack/mb_e4_gemm.cu  (microbench, this commit)
- docs/E4-step0-results.csv                   (171-row grid output)
- docs/E4-step0-isa-evidence.txt              (ISA forensics excerpt)
- taskE1-LEDGER.md                            (this file)
