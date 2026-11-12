# Task E4 bytegen LEDGER -- step 1a: GPU repack kernel vs host permutation (byte equivalence)

Date: 2026-11-11. Branch INT4-noquant @ 67a0fbd571 + this commit. Box mi50, GPU2,
flock /root/gpu.lock, TheRock toolchain via /root/w4a4-env.sh.

## Question (E1 residual check #1, re-scoped step 1a)
E1 proved stock MMQ vs repacked Q8_0 GEMM bit-identical at 171/171 shapes using the
HOST permutation (repack_q8_0_host, repack-common.cu:350) for the repacked arm. The
runtime async admission path instead repacks on the GPU (repack_q8_0_kernel,
buffer.cu:137, launched at buffer.cu:493). The byte-equivalence of the two was
comment-claimed (buffer.cu:136) but never verified. If they diverged, that would be
the item-#5 defect (byte generation, not GEMM arithmetic).

## Method
Harness: ggml/src/ggml-cuda/q8_repack/mb_e4_bytegen_test.cu (named *_test.cu so the
ggml-hip CMake glob excludes it). Verbatim copies of repack_q8_0_host and
repack_q8_0_kernel; layout helpers (repack_qs_row_stride, repack_gcn_nbytes) are the
real inline ones from repack-common.cuh. Both paths consume IDENTICAL deterministic
PRNG canonical Q8_0 bytes (random f16 scale bits + random qs bytes). Launch
parameters mirror the runtime async path exactly: qs_str = repack_qs_row_stride
(Q8_0, ne0), qs_len = ne1*qs_str, src_str = ggml_nbytes/ne2 (contiguous:
ne1*n_blocks*34), dst_str = repack_gcn_nbytes(Q8_0, ne0, ne1), n_out =
ne2*ne1*n_blocks, block 256, grid min(ceil(n_out/256), 65535).

Per shape:
  A. host record vs GPU kernel into a ZERO-filled dst -> full-record byte compare.
  B. host record vs GPU kernel into a 0xA5-filled dst -> every differing byte
     classified qs-live / qs-gap / d-plane (qs-gap = the 16 B de-alias bump
     repack_qs_stride adds when ne0%128==0).
  C. GPU kernel twice into zero-filled dst -> launch determinism.
Plus per-record FNV-1a 64 hashes of both records (printed; equal on every shape).

Shapes (26): the 15 dense E1 grid points K x M, K in {1024,2048,4096,5120,6144} x
M in {2048,4096,5120}; MoE expert tensors (2048x2048x64, 4096x2048x64,
2048x4096x64, 4096x2048x8 -- the 64-expert 4096x2048 case has n_out = 16777216 >
65535*256, so it exercises the runtime grid cap and the grid-stride tail);
no-gap K shapes K in {1056,4128,5152} (K%32==0, K%128!=0 -> qs_str == n_blocks*32);
tiny edges (32x32x1, 64x96x2, 128x128x1); big 3D (6144x5120x2).

## Result: BYTE-EQUIVALENT -- 26/26 shapes, zero live-byte differences
docs/E4-step1a-bytegen-results.csv. Every shape: A = MATCH (full record identical
when the GPU dst starts zeroed), B = gap-only (all diffs vs the 0xA5-filled dst are
exactly the qs-gap bytes, count = ne2*ne1*16 where the bump fires, 0 on no-gap
shapes), C = launch-deterministic, FNV hashes equal. The 65535-grid-cap shape
matches, so the grid-stride tail is covered.

## The one real difference found: stride-gap bytes (cosmetic, not the defect)
repack_q8_0_host memsets the whole record including the 16 B per-row de-alias gap
that repack_qs_stride adds when ne0%128==0 (true for EVERY grid K: 1024..6144).
repack_q8_0_kernel never writes the gap, and the runtime does not zero tensor->data
before the async repack kernel (no memset in buffer.cu), so gap bytes in the
runtime record are uninitialized. This is NOT a correctness defect: the consumer
kernels iterate sub-blocks with sb < n_sub (or sb_hi <= n_sub under K-split,
repack-kernels.cuh:776, :1001-1005), so gap bytes are never read -- same contract
the MXFP4 path documents. But the kernel comment at buffer.cu:136 ("Padding
sub-blocks ... are zeroed, matching repack_q8_0_host()") is misleading: the kernel
zeroes nothing; the host's zeros come from its memset. Recommend a comment fix in
the wiring commit; no code change required.

## Verdict / consequences
- The item-#5 defect is NOT in GPU-vs-host byte generation. The runtime stock-vs-
  repack divergence cannot come from repack_q8_0_kernel vs repack_q8_0_host.
- Residual byte-generation suspects narrow to E1 ledger items #2 (runtime "stock"
  baseline may be the gcn_kq GEMM family -- repack == gcn_kq is the open question)
  and #3 (HAS_IDS expert GEMM coverage). Siblings own those.

## Repro
  source /root/w4a4-env.sh
  hipcc -O3 -std=c++17 --offload-arch=gfx906 -DGGML_USE_HIP \
    -Iggml/include -Iggml/src -Iggml/src/ggml-cuda \
    ggml/src/ggml-cuda/q8_repack/mb_e4_bytegen_test.cu -o mb_e4_bytegen
  flock -w 600 /root/gpu.lock env HIP_VISIBLE_DEVICES=2 ./mb_e4_bytegen
Exit 0 = all shapes byte-identical.

## Files
- ggml/src/ggml-cuda/q8_repack/mb_e4_bytegen_test.cu (harness, this commit)
- docs/E4-step1a-bytegen-results.csv (26-row grid output, this commit)
- taskE4-bytegen-LEDGER.md (this file)
