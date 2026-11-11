
## GPU microbench + ISA census results (2026-11-11, INT4-noquant)

ISA census (carved.elf from TheRock build; llvm-readobj --notes + objdump):
- vgpr per T=1/2/4/8/16: 46/46/59/62/62 - ALL <= 64 (4 waves/SIMD feasible at every width)
- sgpr: 32/44/64/94/98 (T=8/16 near the 104 limit - noted, legal)
- spills: 0 everywhere; dot8 count = exactly 8*T per kernel (two-accumulator scheme as designed)

GPU microbench (mb_i8narrow_test, one idle card, wall-clock median of 30, warmup 5):
- shape K=5120 N=17408: T1 0.330 / T2 0.356 / T4 0.354 / T8 0.368 ms -> T8/T1 = 1.12x PASS
  (verify widths 2..8 are nearly free on the wide-N orientation - bandwidth-bound, thesis confirmed)
- shape K=17408 N=5120: T1 0.450 / T2 0.487 / T4 0.653 / T8 0.899 ms -> T8/T1 = 2.00x FAIL
  (one-lane-per-row layout underfills the GPU at N=5120: only 20 blocks on 64 CUs; the
   narrow-N orientation needs a row/K split variant before it can serve down-projections)
- T=16 on shape 1: 87040/278528 outputs garbage (5 of 16 token planes corrupted) - REAL BUG
  in the T=16 instantiation, found by the microbench (the skeleton was never compiled before).
  DFlash2 verify widths are <= 8, so this blocks only the small-batch (9..16) use case.
- correctness: integer path EXACT vs CPU reference (maxrel 0.0059 on 3-16 of 17408 outputs
  = fp32 contraction class: device FMA vs host mul-add; the earlier "maxrel 0 ok" was VACUOUS -
  the microbench harness itself had (a) separate qs/d device allocations where the kernel
  expects ONE contiguous [qs|d] buffer (weight d read at wbase+dplane_off), and
  (b) ad[...] = b->d half->uint16_t truncation zeroing every sub-0.5 scale. Both fixed;
  vacuous-pass lesson mirrors the seal-input-validation rule.

Structural fixes to the skeleton (commit c3b7fcb1f5 + this commit):
- body-level arch guard (gcn-u4-gemm pattern): the __global__ definition is now unconditional
  with the gfx906 code behind #if inside the body - the previous whole-TU gate made the
  host pass see only GGML_ABORT stubs, i.e. the lib's i8narrow path was unrunnable from host.
- kernel stays `static __global__` (non-static + template head triggered a clang pass
  shadow/nesting error suite; static is census-proven).
- templated launcher replaced by a non-template dispatch fn (runtime T switch over the five
  instantiations); KSPLIT>1 second-pass reducer still TODO (only KSPLIT=1 dispatchable).
- microbench harness lessons: contiguous [qs|d] device buffer (dplane_off layout), bit-copy
  the f16 scales (never assign half through integer types), explicit stream + wall-clock
  timing (hipEventElapsedTime returned 0 on this build), verify-vs-CPU tolerance 1e-2 rel
  documented as contraction class.

Open items: T=16 plane-corruption bug; shape-2 occupancy variant (row/K split);
stock-nc-kernel head-to-head arm; KSPLIT>1 reducer.
