# taskV2 R7 LEDGER — wave-split chunk combine in the stock chunked fattn path

Lane: R7 subagent (GPU pair 2,3; gpu.lock shared, flock -w 2700).
Date: 2026-11-14 (PST). Branch: int4-w-combine @ 77175a3f6b (from tip 0340ba0a1e). NOT pushed.
Build: /root/build-combine (mi50-build.sh, TheRock-pinned amdclang++, ccache;
BUILD-PROVENANCE src_sha=77175a3f6b, dirty=0). Baseline binary: /root/build-qmerge.
Artifacts: /root/taskV2-r7/ (driver /root/r7-drive.sh, chain /root/r7-chain.sh, agg /root/r7-agg.py).

## What was built (ggml/src/ggml-cuda/fattn-common.cuh, +125/-6, commit 77175a3f6b)
- New kernel flash_attn_combine_results_split<D>: port of rq_fa_combine's block/wave-split
  design (taskJ1) into the STOCK chunked path. 4 waves x 64 lanes per block; chunk weights
  exp(m_l - m) computed once per block in LDS (expf, not __expf, to minimize drift);
  per-channel numerator = 4 strided partial sums + tree; denominator same. Reads the
  exact same parts/meta layout as stock (grid.x = linear (seq,col,head), grid.y = D/64
  channel groups) - format-independent (f16/q8_0/q4_0 KV all served identically).
- Env gate GGML_CUDA_FA_COMBINE_SPLIT=1, DEFAULT OFF, inert unless cc==gfx906 (VEGA20).
  Gate-off call site is the untouched stock code path. Engagement banner [FA-COMBINE-SPLIT]
  prints from the real call site (once per template instantiation).
- Numeric delta vs stock: combine reorders fp32 adds only when parallel_blocks >= 5
  (for pb<=4 each wave holds <=1 chunk and the tree sum equals the stock serial order -
  bit-identical by construction; consistent with byte-identical short-ctx probes).

## Seal results
1. Gate-off byte-identity: greedy probe (temp0 seed42 n=64) byte-identical across
   build-qmerge vs build-combine(gate off) vs build-combine(gate on, short ctx):
   sha256[:16]=77841cb6a251235d on all three (also serves as determinism control).
2. test-backend-ops -o FLASH_ATTN_EXT: gate-off 4001/4001 pass; gate-ON 4001/4001 pass
   with [FA-COMBINE-SPLIT] banner x69 (kernel genuinely engaged at test shapes).
3. KLD vs gate-off (taskS -b 8 protocol): Mean KLD = 0.005357 +/- 0.001566,
   Maximum KLD 12.45 (heavy-tail: a few near-tie argmax flips dominate; teacher-forced,
   no trajectory divergence). Same benign reorder class as the shipped FA_DOT8 gate
   (taskS measured 0.0035/0.0018). Banner hits confirm engagement.
   DEVIATION: 64k-context KLD is infeasible on this 40GB host - llama-perplexity needs
   >=2*ctx tokens; 133k tokens x ~65k vocab x 4B = ~35GB logits file drove the box to
   0GB avail / 25GB swap (PSI 48) before being killed. Sealed at -c 16384 (33k-token
   wikitext slice): -b 8 at 16k KV drives parallel_blocks well past 5 = same wave-split
   reorder path as 64k. 64k numeric evidence: greedy probes + ops NMSE + KLD-at-16k.
4. 64k greedy probes (doc64 tail + question, temp0 seed42, n=64): off vs split
   byte-IDENTICAL (sha256[:16]=a74bf852242a0fc7), split banner x3 = kernel engaged
   during the run - reorder produced zero argmax flips on this sample.
   DFlash2 d4 lc-probe: q1/q3 texts byte-identical, q2 diverged (one near-tie flip
   mid-answer) - expected reorder class, matches KLD structure.

## Perf results (27B Q8_0 TP2 f16 KV, pair 2,3, single reps, arms ~30min apart)
Engagement witness at 64k: banner shows parallel_blocks=20 at verify/decode shapes
(D=256 head; TP2 halves the TP4 chunk count - taskS saw 114 chunks at TP4 q4_0).

- DFlash2 d4 @64k, verify attention graph (DFLASH-PROF ne11=5 nodes=628):
  off med=592us (n=244) vs split med=590us (n=246)  =>  -2us / -0.3%  (noise floor)
- DFlash2 d4 @64k e2e tg (lc-probe, identical-text phases q1/q3):
  off 30.22/30.11 vs split 29.77/29.44  =>  -1.5% / -2.2%  (single reps; sign-consistent)
  acceptance: off 258/483=0.534 vs split 257/486=0.529 (unchanged)
- Plain decode @64k (primed 64.7k KV, no draft, q1-q3):
  off 28.64/28.57/28.74 vs split 28.73/28.66/28.80  =>  +0.3% (noise)
- llama-bench -p 65536 (tg leg runs at EMPTY context - does NOT exercise combine;
  pp sanity only): 469.5 vs 468.9 t/s = -0.1%.

## Verdict
NEGATIVE at these shapes. The wave-split combine is correct (ops 4001/4001, byte-identical
gate-off, KLD 0.0054 same class as shipped gates) but does NOT beat stock at TP2 64k
f16-KV: at pb~20 the combine is a few us of a 590us verify step, and the measured
verify-graph delta (-0.3%) sits inside noise while e2e tg shows a possible small
REGRESSION (-1.5..-2.2%, single-rep, plausibly the 4x-block/LDS overhead of the split
design dominating at low chunk counts). The taskS-context win (17-20% of step at TP4,
114 chunks) needs ~100+ chunks per GPU to materialize; TP2 halves the chunk count and
the win vanishes. Recommendation: keep the gate default OFF (as implemented); do not
adopt at TP2. If revisited: TP4/q4_0-KV stacks (pb>100), or a pb-threshold on the gate
(engage only when parallel_blocks >= ~64), and a paired-interleaved rerun to beat noise.

## Issues encountered (for the lane)
- 64k llama-perplexity KLD wedges RAM on this host (35GB logits) - do not retry at 64k.
- llama-bench -p/-n tg leg decodes at empty context: useless for long-context kernels;
  use primed lc-probe instead.
- llama-perplexity token-count abort exits rc=0 (guards must check outputs, not rc).
- 262KB prompt as argv hits MAX_ARG_STRLEN (128KB) - pass long prompts via file.
