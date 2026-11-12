# Task E5 Ledger -- E4 step-1b re-scope: runtime stock baseline on non-yield shapes + repack-vs-gcn_kq bit-diff

Date: 2026-11-11 (PST). Worktree /root/int4-noquant-wt, branch INT4-noquant @ 67a0fbd571.
Measurement only: no kernel code was modified. GPU2 only, every leg under
`flock -w 600 /root/gpu.lock`, TheRock 7.15 (source /root/w4a4-env.sh), mi50-doctor ALL CLEAR.

## Question (re-scoped from task E1 residual check #2)
E1 proved the repacked Q8_0 GEMM is bit-identical to stock MMQ (mul_mat_q) at 171/171
kernel-level shapes. Open question: on NON-yield shapes, is the runtime "stock" competitor
actually the gcn_kq GEMM family (gcn-q8-gemm.cu)? If serving comparisons measured
repack-vs-gcn_kq, "bit-identical to stock" would require repacked == gcn_kq, which the
different register-tile geometry makes doubtful. Deliverables: (1) dispatch witness of
which kernel serves a repack-admissible Q8_0 matmul per shape class; (2) kernel-level
bit-diff repacked vs gcn_kq on identical inputs; (3) verdict for the un-gating gate.

## Method
(a) ggml-level dispatch harness: ggml/src/ggml-cuda/q8_repack/e5_gcnkq_bitdiff_test.cu.
    One dense MUL_MAT (Q8_0 src0 K x M, f32 src1 K x N) through the REAL ggml-cuda
    dispatch, dst fp32 dumped per arm:
      S  stock          (no repack env, cuda buft)        -> gcn_kq if admitted, else MMQ
      M  stock MMQ       GGML_CUDA_GCN_KQ=0               -> mul_mat_q on every shape
      R1 repack yield ON REPACK_Q8_0=1 STOCK_ACC=0       -> yields gcn_kq-admitted shapes
      R0 repack yield OFF + REPACK_Q8_0_YIELD=0          -> repacked GEMM on every shape
    Grid: 23 shapes = 10 gcn_kq-admitted (N in {32,33,64,128,256} x (K,M) in
    {(2048,2048),(4096,2048)}, (1024,4096) x {32,128}) + 7 non-yield N in {9,16,31} +
    3 M<128 fallback (M=100, N in {9,32,128}) + 3 K%256!=0 (K=1152, N in {9,32,128}).
    92 legs, per-output fp32 bit-diff per pair (docs/E5-gcnkq-results.csv).
(b) Kernel-launch witness: scripts/e5_kname_shim.c, an LD_PRELOAD interposer on
    hipLaunchKernel resolving the launch stub address against the ELF .symtab of
    libggml-hip.so (stubs are local STT_OBJECT symbols; dladdr/dynsym give nothing).
    Combined with GGML_GCN_KQ_TRACE / GGML_Q8_YIELD_TRACE / GGML_CUDA_RP_DISPATCH_TRACE.
    Witness logs: /root/e5-work/witness/ (host scratch, not committed).

## Witness results (which kernel serves what)
- Stock arm, N>=32 admitted (e.g. K=2048 M=2048 N=32): gcn_q8x_quantize +
  gcn_kq4_gemm<GGML_TYPE_Q8_0,...> + gcn_q8_splitk_reduce. gcn_kq v4 serves it.
- Stock arm, N=9..31: [GCN-KQ] intercept logged then DECLINED (hard admission
  `N < 32` -> false, gcn-q8-gemm.cu:2020; also M<min_rows(128), K%QK_K!=0, Q8_0 v4-only)
  -> quantize_mmq_q8_1 + mul_mat_q (stock MMQ). gcn_kq NEVER serves N<32.
- Repack arm yield ON, N=9..31: [rp-dispatch] -> gemm; quantize_mmq_q8_1 +
  mmq_gemm_repacked_w32 (repacked GEMM). No gcn_kq intercept: the yield gate
  (q8_repack/mul-mat.cu:127) requires ne11>=32 before consulting ggml_cuda_gcn_kq_gemm.
- Repack arm yield ON, N>=32 admitted: [Q8-YIELD] gcn_kq=1 with canon = data + rec
  (canonical block_q8_0 copy, canon%4=0) -> gcn_q8x_quantize + gcn_kq4_gemm<Q8_0> +
  splitk_reduce: the SAME kernel family and input bytes as the stock arm.
- Repack arm yield OFF, N>=32: mmq_gemm_repacked_w32 (the un-gated arm).
- MoE note: ggml_cuda_gcn_kq_moe_supported does NOT admit Q8_0 experts (type_ok is
  K-quant/i-quant only), so expert GEMMs never run gcn_kq either; stock expert arm is
  MMVQ (<=8) / MMQ.

## Bit-diff results (docs/E5-gcnkq-results.csv, 23 shapes x 4 pairs)
- R0_vs_M (repacked GEMM vs stock MMQ): 23/23 bit-identical, 0 differing fp32.
  E1's kernel-level result re-confirmed through the real runtime dispatch, including
  the yield shapes (the repacked GEMM matches MMQ even where stock runs gcn_kq).
- R1_vs_S (production repack arm vs stock): 23/23 bit-identical, 0 differing fp32.
  Includes all 10 yield shapes -> the canonical copy is byte-exact and the yield
  reproduces the stock regime exactly (dense level; HAS_IDS experts NOT covered).
- R0_vs_S (un-gated repack vs stock): diverges ONLY on the 10 gcn_kq-admitted shapes:
  41-82% of outputs differ per shape (e.g. 53615/65536 at N=32; 344703/524288 at
  N=256), max_abs_diff 7.6e-6..3.8e-5, max_rel_diff up to 6.2e-2 (cancellation-prone
  near-zero outputs). All 13 non-yield shapes: 0 diff.
- S_vs_M (gcn_kq vs MMQ, the stock regime split): IDENTICAL diff counts to R0_vs_S at
  every shape (totals equal: 1566221 differing fp32). The repacked GEMM contributes
  ZERO divergence of its own: repack-vs-stock-with-yield-off IS exactly gcn_kq-vs-MMQ.
- Mechanism of the repack-vs-gcn_kq gap: gcn_kq v4 quantizes activations differently
  (gcn_q8x_quantize: int8 values + per-32 fp32 scale pairs, no min-offset) than the
  block_q8_1 quantization both MMQ and the repacked GEMM consume, and accumulates on
  64x64 v4 tiles with split-K fp32 partial reduce (gcn_q8_splitk_reduce). Different
  input bytes AND different fp32 geometry: not an epilogue-restyle-recoverable gap.

## Verdict
1. The E1 open question is REFUTED for dense Q8_0: on non-yield shapes (N 9..31, M<128,
   K%256!=0) the runtime stock baseline is mul_mat_q (stock MMQ), never gcn_kq - the
   gcn_kq admission hard-rejects N<32 (and M<128, K%256!=0, non-v4). The repack arm
   serves those shapes with mmq_gemm_repacked_w32, which is bit-identical to MMQ
   (23/23 here, 171/171 E1 kernel-level).
2. repack-vs-gcn_kq is only measurable on gcn_kq-admitted shapes (N>=32, M>=128,
   K%256==0, v4-aligned) and is NOT bit-identical: 41-82% of outputs differ, max rel
   ~6e-2. The divergence is fully explained by gcn_kq's different activation
   quantization + tile/split-K geometry (S_vs_M == R0_vs_S element-for-element).
3.   For the un-gating gate definition: "bit-identical to stock" must be defined
   per-shape-class against the kernel stock actually runs: repacked == MMQ on
   non-yield shapes (HOLDS, both kernel-level and runtime-dispatch level), and the
   gcn_kq yield must STAY on gcn_kq-admitted shapes (removing it makes bit-identity
   impossible without also replicating gcn_kq's activation quantization - out of
   scope and pointless: the yield already gives bit-identity there at ~1% pp cost).
   The shipped-yield (R1) arm is bit-identical to stock on every dense shape tested.
4. Consequence for the section-93 hotfix rationale: the claim that the un-restyled
   repacked GEMM was "not bit-safe on the shapes the yield declines (9..31-token
   prefill etc)" is NOT supported - on exactly those shapes it is bit-identical to
   stock MMQ. Any observed repack-vs-stock serving divergence (KL 0.0035 class) cannot
   come from dense non-yield GEMMs; remaining candidates are the HAS_IDS expert GEMMs
   (un-proven, plan risk #3), the nc width 6-8 gap (closed by task E2), or a leg that
   ran with the yield off. The dense-GEMM half of the un-gating evidence is now closed.

## Files
- ggml/src/ggml-cuda/q8_repack/e5_gcnkq_bitdiff_test.cu  (dispatch harness, *_test.cu so the
  ggml-hip CMake glob excludes it; build: hipcc -O2 -std=c++17 -I ggml/include -I ggml/src
  <file> -o build/bin/e5_gcnkq_bitdiff_test -L build/bin -Wl,-rpath,<build/bin> -lggml -lggml-base -ldl;
  run from build/bin so the backend registry finds libggml-hip.so)
- scripts/e5_kname_shim.c  (LD_PRELOAD kernel-launch witness; build:
  hipcc -O2 -x c++ -D__HIP_PLATFORM_AMD__ -shared -fPIC -I/opt/rocm-7.2.3/include <file> -o kname_shim.so -ldl)
- docs/E5-gcnkq-results.csv  (92-row pairwise bit-diff table)
- taskE5-gcnkq-LEDGER.md  (this file)
