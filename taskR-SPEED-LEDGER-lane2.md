# taskR SPEED LEDGER - pinned own-head drafter (-devd) vs borrowed head on the 27B TP2 stacks

Date: 2026-11-14 (PST). Box: mi50 (4x MI50 gfx906, LXC on PVE). No pushes; local commit only.
Question (from both reviews): does taskH3's drafter pinning win (+14.2% d4 / +7.3% d7 on the
35B Q8_0 MoE TP2) compose with the new 27B repack stacks? The pinned best-config numbers
become the llama-swap adoption candidates.

Build: /root/build-qmerge @ 2a80a1c8ad, TP2, DFlash2-Q4_0 drafter. Natural suite, tg256
median (8 prompts x 3 reps, temp-0), harness /root/p2-prefill/bench.py - same suite as
taskQ/taskL so the unpinned baselines are directly comparable. GGML_DFLASH_PROF=1
step-phase witness on every leg. Driver /root/taskR/taskR-drive.sh + matrix.sh
(interleaved b/p, 2 reps, refuse-if-exists). All legs flock /root/gpu.lock, TERM-only
teardown, production env witnessed via mi50-run.sh. rocm0 (bus 12:00.0) pre-checked:
dmesg shows NO uncorrectable ECC events, temps 56C edge / 82C junction - cleared for use;
the gate auto-picked DEVS per leg (0,1 or 2,3 depending on sibling occupancy; -devd ROCm0
means first VISIBLE card, so pinning semantics are identical on either pair).

Contention note: three lanes shared /root/gpu.lock this session (this taskR matrix +
taskS + taskT). 3 legs hit LOCKFAIL at the 900s window mid-run; after the second round
the flock wait was raised to 2700s (drive.sh atomic swap + drive2.sh for the rerun pass)
- same lock discipline, longer patience; no LOCKFAIL after the swap. One leg
(q8lds-b-d4-r1) was quarantined after its first attempt showed a contention artifact
(math-reasoning 6.8 t/s vs 55.8 on the paired leg); it was re-run clean (50.56, matches
baseline). f128q4w4 ran session-depressed overall (b drift -7.5%/-10.8% vs taskQ) - the
within-session paired deltas below remain valid, absolute numbers do not.

Stacks (arm env per drive.sh):
  f0q4     = 27B Q4_0 + floor-0 + REPACK_Q8_0 pair + RP_Q8_NC_LDS=8 + RP_Q4_NC_LDS=8
  f128q4w4 = 27B Q4_0 + REPACK_Q8_0 pair + RP_Q8_NC_LDS=8 + RP_Q4_NC_LDS=4
  q8lds    = 27B Q8_0 + REPACK_Q8_0 pair + RP_Q8_NC_LDS=8
Head arms: b = borrowed-head stock drafter (Qwen3.8-27B-DFlash2-Q4_0.gguf);
p = own-head grafted drafter (draft/DFlash2-Q4_0-ownhead-q{4,8}.gguf via graft_ownhead.py)
+ -devd ROCm0 (H3-proven placement). No code change: GGUF-level head graft.

## 1. Feasibility (probes)

prb-f0q4-p-d4 / prb-q8lds-p-d4: own-head graft + -devd ROCm0 loads and drafts on both
model sizes (short-prompt acceptance 0.875/0.828, mean len 4.50/4.20). All 24 bench legs
loaded and completed rc=0.

## 2. A/B numbers (tg256 median; per-rep tg, medians in bold sense)

| arm | depth | b rep1/rep2 | b med | p rep1/rep2 | p med | p vs b |
|---|---|---|---|---|---|---|
| f0q4     | d4 | 55.48 / 54.57 | 55.02 | 57.65 / 57.54 | 57.60 | +4.67% |
| f0q4     | d7 | 43.90 / 43.88 | 43.89 | 44.20 / 44.20 | 44.20 | +0.71% |
| f128q4w4 | d4 | 50.46 / 50.05 | 50.25 | 48.59 / 46.72 | 47.66 | -5.16% |
| f128q4w4 | d7 | 32.41 / 28.20 | 30.30 | 33.82 / 27.93 | 30.87 | +1.88% |
| q8lds    | d4 | 50.56 / 50.66 | 50.61 | 49.85 / 51.74 | 50.80 | +0.37% |
| q8lds    | d7 | 40.87 / 41.06 | 40.96 | 40.83 / 41.14 | 40.98 | +0.05% |

f128q4w4 d7 is the noisiest cell (r1 pair +4.4%, r2 pair -1.0%, both pairs internally
consistent; session degraded between reps). f0q4 and q8lds cells are tight (rep spread
<0.5 t/s within head).

## 3. Gates (agg-r.py)

GATES: ALL-PASS (6/6 cells). Acceptance DIGIT-MATCH between b and p in every cell
(e.g. f0q4 d4 0.3956/2.5822 both heads) - the own-head graft is numerics-preserving, the
delta is pure placement. Greedy probe BYTE-IDENTICAL across all 24 legs
(sha256[:16] 77841cb6a251235d).

## 4. Composition vs unpinned baselines (taskQ f0q4/f128q4, taskL q4d)

| arm | depth | unpinned base | b drift vs base | p vs base |
|---|---|---|---|---|
| f0q4     | d4 | 56.04 | -1.8% | +2.8% |
| f0q4     | d7 | 44.30 | -0.9% | -0.2% |
| f128q4w4 | d4 | 54.30 | -7.5% | -12.2% |
| f128q4w4 | d7 | 33.96 | -10.8% | -9.1% |
| q8lds    | d4 | 50.44 | +0.3% | +0.7% |
| q8lds    | d7 | 40.61 | +0.9% | +0.9% |

## 5. Step-phase (DFLASH-PROF medians, us/step; b -> p)

| arm | depth | draft-fwd | inject | gather | tgt-decode |
|---|---|---|---|---|---|
| f0q4     | d4 | 11531 -> 523 (-95%) | 870 -> 318 | 27 -> 25 | 33013 -> 33582 (+1.7%) |
| f0q4     | d7 | 15470 -> 566 (-96%) | 1035 -> 438 | 262 -> 257 | 42684 -> 43600 (+2.1%) |
| f128q4w4 | d4 | 11673 -> 545 (-95%) | 945 -> 361 | 33 -> 34 | 39026 -> 39824 (+2.0%) |
| f128q4w4 | d7 | 15849 -> 601 (-96%) | 1201 -> 579 | 269 -> 270 | 50961 -> 51934 (+1.9%) |
| q8lds    | d4 | 12081 -> 554 (-95%) | 868 -> 395 | 27 -> 35 | 37754 -> 38636 (+2.3%) |
| q8lds    | d7 | 16370 -> 559 (-97%) | 1053 -> 456 | 267 -> 267 | 48592 -> 49741 (+2.4%) |

The device-side signature reproduces H3 exactly: pinned draft-fwd collapses ~95-97%
(11.5-16.4 ms -> 0.52-0.62 ms). But the net tg gain does NOT: derived step time for
f0q4 d4 is 46.9 -> 44.8 ms/step (-2.1 ms net out of -11.0 ms device savings). The
host-mediated gather/inject loop of the pinned arm gives back ~8-9 ms/step on this engine
(worse than H3's ~4 ms give-back), and the pinned draft sharing a card with the target's
half costs +1.7-2.4% tgt-decode. Three-lane CPU contention this session plausibly
inflated the host-sync phases of the pinned arms specifically (they are host-mediated;
the borrowed arm runs device-graph); a quiet-box revalidation would bound that residual.

## 6. Verdict + adoption candidates

VERDICT: H3's +14.2% pinning win does NOT compose with the 27B repack stacks. Measured
p-vs-b: +4.7% (f0q4 d4), +0.7% (f0q4 d7), -5.2% (f128q4w4 d4), +1.9% (f128q4w4 d7,
noisy), +0.4% (q8lds d4), +0.0% (q8lds d7). The draft-fwd collapse is real and identical
to H3, but on the 27B TP2 the repack-optimized target leaves the pinned arm's host
round-trip + same-card contention almost no net margin. H3's 35B MoE won because its
replicated draft cost was per-layer allreduce across 6 drafter layers on a slower target
path; the 27B stacks' faster target path shrinks the relative win to noise.

llama-swap adoption candidates (pinned best-config numbers, this session):
  d4: f0q4 + pinned drafter = 57.60 t/s  [ADOPT: +4.7% vs borrowed same-session,
      +2.8% vs the 56.04 unpinned baseline; acceptance bit-equal, probe byte-ident]
  d7: f0q4 + pinned drafter = 44.20 t/s  [NEUTRAL: +0.7%/-0.2% vs baseline - adopt only
      for uniformity with d4; borrowed f0q4 d7 43.89 is equivalent]
  f128q4w4 / q8lds: do NOT switch to pinned (neutral-to-negative).
Serving rule update to H3's "pin at both depths": on the 27B stacks pin only where the
stack is f0q4 AND depth is 4; elsewhere keep the borrowed-head drafter.

## 7. Artifacts

/root/taskR/: taskR-drive.sh (flock wait now 2700s), drive2.sh (rerun variant),
matrix.sh, rerun.sh, rerun2.sh, graft_ownhead.py, agg-r.py, phase-r.py, final-r.py,
draft/DFlash2-Q4_0-ownhead-q{4,8}.gguf, {arm}-{b,p}-d{4,7}-r{1,2}.{server.log,
reps.jsonl,reps.jsonl.agg.json,bench.log,probe-*.json,witness files,.done},
quarantine/q8lds-b-d4-r1.* (contaminated first attempt), matrix.log, rerun2.log,
prb-{f0q4,q8lds}-p-d4.* (feasibility probes).
