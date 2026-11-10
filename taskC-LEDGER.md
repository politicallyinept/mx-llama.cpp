# taskC LEDGER - i8narrow: narrow-batch Q4_0 kernel, exact 8-bit activations (non-GPU)

Worktree /root/int4-noquant-wt, branch INT4-noquant @ fb18241213 (verified via
git rev-parse). Box state at start: RAM 37/40 GB used, GPU1/GPU2 busy serving
llama-swap fleet. HARD RULES honored: no GPU work, no cmake/make builds; only
single-file CPU compiles (g++ -O2) were run.

## 2026-11-10 (evening) - read phase DONE
- Narrow window: MMQ_RP_Q8_MMV_MAX_TOKENS=8 (repack-common.cuh:20); rp_narrow_max
  env-clamped [1,8] (mul-mat.cu:320-330); window ne11 in [1, rp_narrow_max]
  (mul-mat.cu:343); tiled GEMM starts mul-mat.cu:509. Width 16 is OUTSIDE the
  window - dispatch change needed (design §2).
- Census verified: narrow nc kernel = 8 dot8 in a 186-instr hot loop, 44 VGPR,
  0 spills (w4a8-census.md:35); "doubling the dot count is nearly free" for
  mat-vec (census:42). Wide-GEMM NO-GO (census:44) NOT re-opened.
- Q4_0 pairing verified from CODE: byte j = {val j low, val j+16 high}
  (ggml-quants.c:137-146); activation side same order, signed (q-8)&0xF
  (quantize.cu:601-650; host ref repack-common.cu:789-794). The WARNING comment
  at repack-common.cuh:206-211 is STALE for the host ref (fixed 2026-10-22).
- Exact identity derived: S = 16*D_hi + D_lo + 8*SUMW == sum((c-8)*a) == stock
  integer partial (ilo+ihi) - 8*SA. SUMW must be SIGNED (3b class).

## 2026-11-10 - seal DONE (CPU-only)
- seals/seal_i8split.cc authored (single file, g++ -O2 -std=c++17, no deps):
  regime coefficient asserts (9 regimes incl. degenerate ones, vacuity-guarded),
  per-nibble-lane sign sweep (32 lanes x h=-8..7), wrong-pairing detector
  ({2k,2k+1} distinguished 64/64), adversarial scale boundary (stock q8_1 never
  emits -128; -128/+127 mixes exercised anyway), 200k-block fuzz
  i8split==stock==direct, fp32 epilogue vs double ref (1e-3 bar, observed 2.3e-4),
  int32 headroom (worst |S| 7860 vs bound 32512).
- Local build+run: 16 invariants, 0 FAIL, exit 0. mi50 run recorded below.

## 2026-11-10 - design + skeleton DONE
- docs/DESIGN-i8-narrow-kernel.md: full design (layout, register plan, identity,
  pairing spec, K-split, barriers, wave/offset policy, shapes, GPU bar).
- ggml/src/ggml-cuda/q8_repack/w4a4-i8narrow.cuh/.cu: HIP skeleton, NOT COMPILED
  (pause point). SIZEOF_PROBE idiom + paired sizeof/alignof static_asserts on
  block_i8split (36 B / align 2).

## GPU-QUEUED (do NOT run until an idle card + gpu.lock + RAM gate clear)
1. Build: normal TheRock/cmake build of the fork with the two new files added to
   the ggml-cuda target (build gate: >= 12 GB free RAM - currently NOT met).
2. ISA census before timing (method docs/w4a8-census.md:6-12):
   llvm-objdump the new kernel; bars: 0 VGPR spills, <= 64 VGPR at T<=8,
   dot8 count per hot loop == 8*T.
3. Microbench (standalone hip-event harness, one idle card, flock gpu.lock):
   shapes (ne00=5120, ne01=17408) and (ne00=17408, ne01=5120); widths 1/2/4/8/16;
   KSPLIT 1 and 4 on the N=5120 shape.
   PASS bars: 8-token time <= 1.5x 1-token time (same kernel/shape);
   >= 95% of current narrow stock nc kernel at widths 1-8; beat it at 4-16.
   Cross-ref ruquant w4a4_skinny on the same card. llama-bench DIAGNOSTIC ONLY;
   authoritative claims via harness/server protocol later.
4. Device-vs-host bit-match: dump device integer partials, compare to
   seal_i8split emulation (extend harness with a device dump path).
5. Correctness gate before any perf claim: greedy seal via harness protocol.

## Status: PAUSED at the GPU/build boundary. All non-GPU deliverables committed on
INT4-noquant (see git log). No pushes.

## 2026-11-10 - mi50 seal run (CPU-only, g++ -O2 -std=c++17, actual output)
$ ssh mi50 'cd /root/int4-noquant-wt && g++ -O2 -std=c++17 -o /root/seal_i8split_build seals/seal_i8split.cc && /root/seal_i8split_build'
seal_i8split: exact-8-bit nibble-dot path for narrow-batch Q4_0

== per-regime coefficient asserts ==
INVARIANT regime const-0 (coeffs + exactness)                              PASS
INVARIANT regime const-127 (coeffs + exactness)                            PASS
INVARIANT regime const-128 (coeffs + exactness)                            PASS
INVARIANT regime const-8 (coeffs + exactness)                              PASS
INVARIANT regime const-minus8 (coeffs + exactness)                         PASS
INVARIANT regime const-1 (coeffs + exactness)                              PASS
INVARIANT regime const-minus1 (coeffs + exactness)                         PASS
INVARIANT regime varying-lows (coeffs + exactness)                         PASS
INVARIANT regime varying-mixed (coeffs + exactness)                        PASS
INVARIANT per-nibble-lane sign sweep (32 lanes x h=-8..7)                  PASS
  wrong-pairing detection: 64/64 randomized blocks distinguished
INVARIANT pairing check: canonical {i,i+16} exact, wrong {2k,2k+1} detected PASS
INVARIANT scale boundary: stock q8_1 never emits -128, reaches +/-127      PASS
INVARIANT scale boundary: -128/+127 mix exact vs direct and stock          PASS
INVARIANT fuzz: i8split == stock == direct (200000 blocks)                 PASS
  fp32 epilogue worst relative error: 2.264e-04 (bar 1e-3: fp32 fold-over-blocks class, same as stock)
INVARIANT fp32 epilogue within 1e-3 rel (2000 rows x 160 blocks)           PASS
  worst |S| observed: 7860 (bound 32512)
INVARIANT int32 headroom: |S| <= 32512 per 32-block                        PASS

16 invariants, 0 FAIL
SEAL_EXIT=0

## 2026-11-10 - committed
- 42983d62d7 on INT4-noquant: design doc + seal + HIP skeleton + this ledger. No push.
- Next: GPU-QUEUED section above; resume only when RAM >= 12 GB free and a card is idle under gpu.lock.
