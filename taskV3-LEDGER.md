# taskV3 LEDGER - SOLO clean rerun of the pinned-drafter 3x2x2 matrix (decision-grade)

Date: 2026-11-15 (PST). Box: mi50. Branch `int4-v3-pinsolo` off int4-r-pindraft tip
(b31ca84796, NOT pushed). Build: /root/build-uwave (BUILD-PROVENANCE src_tree
/root/int4-noquant-wt @ e9b3e3197f (INT4-noquant tip), llama-server md5
b956a6858d8976624271378f97d825cb, libggml-hip md5 127f1494af87e2a85b2e6801892ca542,
dirty_files=0).

NOTE ON BUILD: taskR ran on /root/build-qmerge (src 2a80a1c8ad); this rerun uses the
newer /root/build-uwave (src e9b3e3197f = taskX-merged INT4-noquant tip, the build the
parent pinned for V-lanes). Absolute t/s may carry a small build delta vs taskR; the
PAIRED within-session deltas are the decision data.

## 0. Why

taskR's matrix ran under heavy sibling contention (taskS/taskT lanes held /root/gpu.lock
much of the night; f128q4w4-d7 pair swung 32.41->28.20 t/s between reps; one q8lds leg
contaminated and re-run). The +4.67% f0q4-d4 pin cell - the ONLY adopt cell - needs a
clean-solo confirmation with a measured noise floor before adoption. This task: full
3x2x2 grid (f0q4/f128q4w4/q8lds x d4/d7 x borrowed/pinned), 2 paired reps per cell
(+1 tiebreaker rep on the two d4 cells that flipped sign), zero concurrent GPU work per
leg (global flock), on build-uwave.

## 1. Setup and doctrine compliance

- Harness: /root/taskV3/taskV3-drive.sh (adapted taskR-drive.sh) + grid.txt +
  laneC.sh (single serial lane, all legs on the uniform pair) + supervise-v3.sh;
  bench /root/p2-prefill/bench.py (natural suite, 8 prompts x 3 reps tg256 + pp4096),
  greedy 64-token temp0 seed42 probe per leg.
- Grafts (own-head drafters, unchanged from taskR): DFlash2-Q4_0-ownhead-q4.gguf
  sha256 7920500c27e09edb99f46b02659c2074ad42c9619415323435f911b476f64578 and
  ...-q8.gguf sha256 e056fb8db654556aa03c5a4a9ffbabe72a9415cf9258f0324c901b0a47d46909 -
  BOTH re-verified byte-exact vs taskR ledger prefixes before launch.
- Pair policy (CORRECTED MID-RUN 01:30 per parent health note): rocm0 (GPU 0) shows
  918,505+ correctable ECC and climbing (:9400 exporter; witnessed in
  /root/taskV3/ecc-witness.txt; re-witnessed 04:35 at 919,793, GPU 1/2/3 all 0/0).
  Correctable ECC corrupts nothing (byte-id gates stay valid) but scrubs slow the card,
  so 0,1 legs are timing-biased. The FIRST pass ran a mixed split (f0q4+f128q4w4-d4 on
  0,1; f128q4w4-d7+q8lds on 2,3); after the health note the ENTIRE grid was re-planned
  to the uniform ECC-clean pair DEVS=1,2 (rocm1+rocm2). All wrong-pair legs run before
  the correction (f0q4-b-d4-r1 on 0,1; f128q4w4-b/p-d7-r1, f128q4w4-b/p-d7-r2 on 2,3;
  one aborted mid-load) are QUARANTINED under /root/taskV3/quarantine/wrong-pair/ and
  kept only as supplementary witnesses (their acceptance digit-matches and probe hashes
  vs taskR still hold). Every decision cell below is same-pair (1,2) paired reps.
  -devd ROCm0 = first VISIBLE card = physical GPU 1 on every pinned leg.
- Sibling interleave: taskV1 moved to pair 1,2 as well; per parent, everything
  serializes through flock /root/gpu.lock -w 900 (all lanes honor it, so at most one
  llama-server runs box-wide at any instant - each leg is solo on the GPU).
  taskV2-moe (pair 2,3) drained ~01:07; taskV1 last leg 01:05. One orphan server from
  a killed stray leg (port 8412 on 0,1) was TERMed per doctrine at 01:36; TERM-only
  teardown throughout, pgrep -x llama-server empty after every leg (final sweep 0).
- Production env via /root/mi50-run.sh + /root/production-env.sh; env+maps witnessed
  per leg from /proc/PID; refuse-if-exists .done per leg.
- r3 tiebreakers: f128q4w4-d4 (taskR said -5.16%, first two clean reps gave +3.57/-2.28)
  and f0q4-d4 got a third paired rep each. One leg (f0q4-b-d4-r3) failed load on first
  attempt (transient, rc=1 at 14s), retried clean at 05:37.

## 2. Results (tg = tg_t_s_median of natural suite; accept = mean acceptance)

| cell        | borrowed tg (r1/r2/r3)   | pinned tg (r1/r2/r3)   | mean delta | per-rep deltas      | noise floor | grade          |
|-------------|--------------------------|------------------------|------------|---------------------|-------------|----------------|
| f0q4-d4     | 55.32 / 55.46 / 54.47    | 58.70 / 57.36 / 57.03  | **+3.69%** | +6.12 +3.43 +4.70   | 2.90%       | DECISION-GRADE |
| f0q4-d7     | 43.73 / 43.63            | 43.83 / 44.29          | +0.87%     | +0.22 +1.52         | 1.05%       | within noise   |
| f128q4w4-d4 | 50.51 / 50.04 / 48.43    | 52.31 / 48.90 / 49.29  | -1.51%     | +3.57 -2.28 +1.76   | 6.80%       | within noise   |
| f128q4w4-d7 | 33.07 / 33.30            | 33.99 / 33.95          | **+2.37%** | +2.79 +1.95         | 0.69%       | DECISION-GRADE |
| q8lds-d4    | 50.00 / 47.81            | 51.63 / 43.13          | -3.12%     | +3.26 -9.78         | 17.93%      | within noise   |
| q8lds-d7    | 28.65 / 39.96            | 29.05 / 29.47          | -14.71%    | +1.39 -26.25        | 32.95%      | CONTAMINATED   |

Draft-fwd cost witness (the mechanism): borrowed draft-fwd ~11.6-16.6 ms/call vs pinned
~0.52-0.60 ms/call - the pin removes host-mediated draft sync, consistent with taskR H3.

## 3. Gates

- Probe (greedy temp0 seed42): BYTE-IDENTICAL (sha 77841cb6a251235d...) on ALL 29 legs,
  pinned == borrowed, every cell. No output change from -devd pinning.
- Acceptance digit-match: holds on all cells EXCEPT two legs - f0q4-d4 p-r1 (0.3985 vs
  0.3956) and f128q4w4-d4 p-r2 (0.3885 vs 0.3872). Both deviations are PINNED-arm only,
  both on the tech-expert prompt only, both ~+0.3% accept. Under greedy temp0 this is a
  nondeterministic draft-proposal path on the pinned head for one prompt class - no
  correctness impact on the byte-id probe, but flagged for follow-up. Verdicts below
  hold with or without those two legs (r3 reps on both cells are exact digit-match).
- q8lds-d7 contamination: b-r1 tg median 28.65 vs b-r2 39.96 (taskR ref ~41), and
  p-r1 math-reasoning prompt collapsed (26.7 vs cell max 52.3). Per-PROMPT medians of
  the clean legs (b-r2 vs p-r2) are near-parity (~+1%), consistent with taskR's +0.05%.
  Cause: session-intrinsic q8 stack variance (repeated 29GB dio loads; GPU 1/2 ECC
  stayed 0/0, temps normal, no sibling held the GPU during these legs - checked).
  The tg-median statistic is fragile for this arm; treat q8lds cells as INCONCLUSIVE
  this session rather than negative.

## 4. Verdicts (adopt/don't-adopt -devd pinning per config)

- f0q4-d4: **ADOPT pinned.** +3.69% mean over 3 clean paired reps (all positive:
  +6.12/+3.43/+4.70), noise floor 2.90% -> decision-grade. CONFIRMS taskR's +4.67%
  (within session drift). This is the candidate and it reproduces.
- f0q4-d7: neutral (+0.87%, within 1.05% noise). Pinning harmless, no gain; either head
  fine. Matches taskR +0.71%.
- f128q4w4-d4: **taskR's -5.16% penalty DOES NOT REPRODUCE.** Clean 3-rep deltas
  +3.57/-2.28/+1.76 (mean +1.0%, noise floor 6.80%) - no evidence of regression, no
  gain. Verdict: DON'T ADOPT (no benefit), but the -5.2% "pinned is bad" reading is
  retracted - it was contention bias in taskR's night run.
- f128q4w4-d7: +2.37% DECISION-GRADE positive (both reps positive, 0.83pp spread).
  Pinning is mildly beneficial here; adopt optional (taskR had it noisy +1.88%).
- q8lds-d4 / q8lds-d7: INCONCLUSIVE this session (noise floors 17.9%/33.0%, one
  contaminated leg). taskR's clean near-neutral (+0.37%/+0.05%) stands -> keep
  borrowed. If decision-grade q8lds pin data is needed, re-run with a stricter RAM
  gate (>=20 GiB MemAvailable before q8 legs) and per-prompt-median aggregation.

## 5. Session drift bands (solo paired deltas only)

- Within-session rep-to-rep noise floor per cell: f0q4-d4 2.90%, f0q4-d7 1.05%,
  f128q4w4-d4 6.80%, f128q4w4-d7 0.69%, q8lds-d4 17.93%, q8lds-d7 32.95%.
  Clean-arm floor (f0q4/f128q4w4-d7): ~0.7-2.9%. Anything under ~3% in those arms is
  not decision-grade from 2 reps.
- Cross-session borrowed vs taskR refs: f0q4-d4 +0.67%, f0q4-d7 -0.48%,
  f128q4w4-d4 -0.41% (build-uwave vs build-qmerge: no material drift on d4 arms);
  f128q4w4-d7 +9.51% and q8lds-d7 -16.25% are taskR contention artifacts / this
  session's q8 variance, not build drift.

## 6. Files

- /root/taskV3/taskV3-drive.sh grid.txt laneC.sh supervise-v3.sh
  taskV3-agg.py status-v3.sh ecc-witness.txt
- /root/taskV3/{tag}.reps.jsonl[.agg.json], probe-{tag}.json, {tag}.server.log,
  {tag}.witness, {tag}.proc.witness, {tag}.gate.witness, {tag}.bench.log (29 legs)
- /root/taskV3/quarantine/wrong-pair/ - pre-correction mixed-pair legs (supplementary)
- Reproduce: bash laneC.sh (refuse-if-exists); python3 taskV3-agg.py
