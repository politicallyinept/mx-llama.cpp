
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


## DRAFTER-LEVER EXPERIMENT (F4q) + np4 PROBE — 2026-11-11 (late session)

### Cell C2 correction (supersedes the cell-C finding above)
The earlier "np4 + DFlash2 d7 soft-hangs the server at the slot-orchestration layer"
claim is RETIRED - it was a DRIVER artifact, not a server defect (analysis doc:
docs/ANALYSIS-np4-dflash-hang.md, commit de84e63b):
1. leg_b_prof.sh read prompts.json key 'prompt'; the file uses 'text_pp' -> P1/P2
   were empty -> 9 of 12 request bodies were malformed JSON (400 parse errors).
2. The cell-C-only bare `wait` at leg_b_prof.sh:87 waited for ALL background jobs -
   including the setsid'd llama-server child (setsid does not fork from a
   non-interactive shell) -> blocked forever. Slots 0/1/2 never received work.
CORRECTED PROBE (fixed key + explicit curl pids): np4 + DFlash2 d7 COMPLETED in
54 s - 4 concurrent requests, slots launched and finished, acceptance 0.59/request,
clean teardown. np4 concurrency with the draft path WORKS; latent hazard H2
(deferred-encode flush, speculative.cpp:1055-1061/1253-1355) not exercised.

### F4q: Q4_0-requantized drafter (the drafter-bandwidth lever)
Drafter: z-lab DFlash2 Q8_0 (2.06 GB) -> Q4_0 (1.09 GB, sha 92d2e2d5...;
all 49 Q8_0 tensors -> Q4_0, norms stayed F32, arch dflash + all 51 KV preserved;
selector_predecessor/successor/hidden weights ALSO requantized - acceptance risk noted).
Result (np1, d4, replicated, 8-prompt suite, same protocol as F1/F2):
- tg256 median 36.6 t/s vs F1/F2 37.8/38.0 -> -3%, WITHIN NOISE, NOT A WIN.
- acceptance UNCHANGED: pooled 2.396 vs 2.39; per-pos (0.620,0.394,0.268,0.184)
  vs (0.624,0.388,0.269,0.185) - the 4-bit selector did NOT hurt acceptance.
- phase-level (statistics dur): draft gen 43094 ms vs F1 51066 ms = -15.7%
  per-call 14.13 -> 11.92 ms - the drafter IS faster, exactly tracking the
  -46% drafter bytes at partial bandwidth sensitivity.
CONCLUSION: the drafter lever is phase-real but END-TO-END IRRELEVANT at d4:
the drafter is only ~14% of the step (11.9 of ~90 ms), so -16% drafter = ~-2%
step. THE VERIFY PASS (~60% of step) REMAINS THE ONLY LEVER THAT MATTERS -
the i8narrow kernel line (items 3-4) is confirmed as the program's path by
elimination. A further drafter win (f16 selector, better drafter arch) has
<= 2% end-to-end ceiling at d4 and is NOT worth pursuing.
