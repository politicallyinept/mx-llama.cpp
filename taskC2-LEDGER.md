# Task C2 LEDGER — i8narrow KSPLIT>1 reducer + shape-2 occupancy evaluation

Worktree /root/wt-ksplit, branch INT4-noquant-ksplit (base 058e420c76).
Scope discipline: edits confined to (a) the KSPLIT>1 dispatch path + new reducer
kernel in w4a4-i8narrow.cu/.cuh, (b) mb harness additions. The T=1..8 KSPLIT=1
kernel code paths are byte-identical (git-diff verified; wt-t16fix owns the
T=16 bug in the same file - merge later).

## Task 1 - reducer design
`mul_mat_i8narrow_reduce_q40(ksplit_ws, y, ne1, T, ys, ksplit)`:
one thread per output element i = row*T + c (coalesced reads in the pass-1 ws
layout [z][row*T + c]); sums ws[z*ne1*T + i] for z = 0..KSPLIT-1 in FIXED ORDER
starting from 0.0f; writes y[c*ys + row]. No fp32 atomics anywhere ->
deterministic by construction (pass-1 wave reduce is a fixed-order LDS tree;
pass-2 is a fixed-order sequential sum). Grid ceil(N*T/256) x 256; ws read
<= 2.6 MB at ks=8/N=5120/T=16 - negligible vs the 44.6 MB weight stream.
VERIFIED: consecutive launches are BIT-identical at every T<=8 cell (harness
memcmp check), KSPLIT=1 and >1 alike.

## Task 2 - dispatch
`mul_mat_i8narrow_launch` gains an overload `(…, stream, int ksplit, float * ksplit_ws)`;
the old 11-arg signature is now a wrapper calling it with (1, nullptr), so the
KSPLIT=1 path is unchanged (same grid, same default-stream launch, nullptr ws).
ksplit in {2,4,8}: pass 1 grid=(ceil(ne1/64), 1, ksplit) on the caller stream
(NWAVES=4 waves cooperate on one 64-row group; the wave-level K split + LDS
reduce already in the kernel does the rest), then the reducer on the same
stream. KSPLIT stays a template parameter (the kernel derives its K segmentation
from it at compile time; grid.z must equal it) -> nested T x KSPLIT switch;
unsupported ksplit -> GGML_ABORT. Workspace [ksplit][ne1][T] floats is
CALLER-ALLOCATED (mb harness hipMallocs ks_max*N*TMAX*4 bytes).

## Task 3 - microbench (TheRock amdclang++ per toolchain rule, flock /root/gpu.lock,
wall-clock median of 30 after 5 warmup; logs /root/mb_ksplit_run{1,2,3}.log)
Oracles: KSPLIT=1 cells use the ORIGINAL cpu_ref (unchanged). KSPLIT>1 cells use
cpu_ref_grouped - a CPU mirror of the kernel's fixed-order grouping (z segments
ceil(nkb/ks), wave sub-segments ceil(len/4), fixed w-then-z order). KSPLIT
changes the fp32 summation ORDER, so the KSPLIT=1 sequential sum is the wrong
oracle for ks>1 cells; the vs-ks1-delta column documents the order-delta class
vs the KSPLIT=1 DEVICE output.

### Shape 2 sweep (K=17408 N=5120) - medians ms (run3; run1 within 2%)
| T \ KSPLIT |  1    |  2    |  4    |  8    |
|------------|-------|-------|-------|-------|
|  1 | 0.449 | 0.349 | 0.406 | 0.291 |
|  2 | 0.486 | 0.367 | 0.445 | 0.337 |
|  4 | 0.653 | 0.451 | 0.501 | 0.380 |
|  8 | 0.898 | 0.467 | 0.524 | 0.424 |
| 16 | 1.572 | 0.778 | 0.723 | 0.676 |
T8/T1: ks1 2.00x (reproduces the 0.450/0.899 baseline exactly) / ks2 1.34x /
ks4 1.29x / ks8 1.46x. BAR MET: best KSPLIT=4 T8/T1 = 1.29x <= 1.5x.
KSPLIT=8 has the best ABSOLUTE times at every width (T1 0.291, T8 0.424);
KSPLIT=4 the flattest ratio. KSPLIT=1 T=16 1.572 ms is inflated by the known
T=16 instantiation bug.

### Correctness (all cells, run3)
- KSPLIT>1 vs the KSPLIT-matched grouped oracle: maxrel <= 0.0059 (fp32
  contraction class) at every cell, EXCEPT one cancellation-exempt element:
  i=20346 (c=2, |ref| = 2.74e-4 while summed terms are ~1e-2), absdiff
  <= 5.8e-6 - the RELATIVE 1e-2 bar is meaningless on a near-zero output; it
  exceeds rel 1e-2 at KSPLIT=1 TOO (maxrel 0.021, absdiff 5.75e-6) -> pre-existing,
  not introduced by KSPLIT. Harness bad-count = rel>1e-2 AND abs>1e-3 (documented
  cancellation escape; the T=16 corruption, absdiff ~1e6, still fails).
- vs-ks1-delta (order-delta class, KSPLIT>1 vs KSPLIT=1 device output):
  max rel 0.045 (that same cancellation element), max ABS 1.14e-4 everywhere -
  pure fp32 reordering noise.
- determinism: bit-match ok at every T<=8 cell, every KSPLIT.
- T=16: shape 2 det-bitmatch FAILS at KSPLIT=1 TOO (no reducer involved) ->
  manifestation of the known T=16 instantiation bug (wt-t16fix owns it; shape 1
  T=16 KSPLIT=1 reproduces the known 87040/278528 corruption unchanged).

### Shape 1 regression (K=5120 N=17408, KSPLIT=1) - UNCHANGED
| T | run1 | run3 | prior (RESULTS-microbench-addendum) |
|---|------|------|--------------------------------------|
| 1 | 0.327 | 0.331 | 0.330 |
| 2 | 0.355 | 0.356 | 0.356 |
| 4 | 0.347 | 0.356 | 0.354 |
| 8 | 0.369 | 0.403 | 0.368 |
T8/T1 1.13x (run1) / 1.22x (run3), prior 1.12x - PASS either way. verify maxrel
0.0059 = prior exactly. The T8 +9% run2/3 drift is environmental (a production
llama-server on GPU1/2 shifts clocks; the KSPLIT=1 kernel code is byte-identical
per git diff, and shape 2 KSPLIT=1 reproduces its baseline 0.450/0.899/2.00x
exactly in the same runs).

## Verdict
KSPLIT>1 fixes the shape-2 occupancy failure: T8/T1 2.00x -> 1.29x (KSPLIT=4),
bar <= 1.5x MET, correctness preserved (KSPLIT-matched oracle + determinism
bit-match), KSPLIT=1 path regression-free. Follow-ups (not in scope): production
KSPLIT auto-pick policy in the mul-mat.cu glue (KSPLIT=8 best absolute, KSPLIT=4
best ratio at N=5120); T=16 bug is wt-t16fix's.
