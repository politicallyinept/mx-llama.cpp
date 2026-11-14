# taskR LEDGER - own-head pinned drafter (-devd) for DFlash2 on the 27B stacks

Date: 2026-11-14 (PST). Box: mi50. Branch `int4-r-pindraft` off INT4-noquant @ 2a80a1c8ad
(NOT pushed). Build: /root/build-qmerge (TheRock-pinned via mi50-build.sh, src_sha
2a80a1c8ad, llama-server md5 3d639a60266023fd78821f4e0369c86a). Harness:
/root/taskR/taskR-drive.sh + matrix.sh + supervise-r.sh (LOCKFAIL retry), bench
/root/p2-prefill/bench.py (natural suite, 8 prompts x 3 reps tg256 + pp4096), all legs
flock /root/gpu.lock -w 900, TERM-only teardown, production env via mi50-run.sh,
refuse-if-exists per-leg files under /root/taskR/.

## 0. Why

H3 showed on the 35B-A3B MoE that pinning the DFlash2 drafter to one card with -devd is
worth +14.2% (d4) / +7.3% (d7), but requires a drafter GGUF carrying its OWN
token_embd.weight + output.weight (the stock 27B DFlash2-Q4_0 drafter is borrowed-head:
81 tensors, no head/embeddings; under -sm tensor those live on the target's Meta device
and -devd aborts). This task: graft an own-head 27B drafter and measure the pinning delta
on the NEW best env configs (taskQ f0q4/f128q4w4, taskL q8lds), both depths, both stacks.

## 1. Graft (numerics-preserving, GGUF byte surgery, no code change)

/root/taskR/graft_ownhead.py (adapted from /root/taskH3/graft_ownhead.py: parameterized,
dtype table extended to this fork's on-disk enum {0=F32,2=Q4_0,3=Q4_1,8=Q8_0,14=Q6_K},
gap-method cross-check of every tensor size vs the dtype table on both inputs).
Streamed raw-byte copies of the TARGET's token_embd.weight + output.weight appended to a
copy of the drafter; norms untouched (F32 dtype-0 preserved); n_tensors 81->83.

* Q4_0 stack: graft from Qwen3.8-27B-Q4_0.gguf (token_embd Q4_0 715,161,600 B +
  output Q6_K 1,042,944,000 B) -> /root/taskR/draft/DFlash2-Q4_0-ownhead-q4.gguf
  (2,852,452,064 B, sha256 7920500c27e09edb...). FULL byte-compare of both grafted
  tensors vs target source: PASS; originals unmoved; spot-compares PASS.
* Q8_0 stack: graft from Qwen3.8-27B-Q8_0.gguf (token_embd + output, Q8_0
  1,350,860,800 B each) -> /root/taskR/draft/DFlash2-Q4_0-ownhead-q8.gguf
  (3,796,068,064 B, sha256 e056fb8db654556a...). Same verifies: PASS.

Feasibility witnesses (f0q4 d4 / q8lds d4 short probes):
* borrowed drafter + -devd ROCm0: ABORTS - ggml-backend.cpp:1461 "pre-allocated tensor
  (output.weight) in a buffer (Meta(ROCm0,ROCm1)) that cannot run the operation (NONE)"
  (prb-f0q4-borrowed-devd; H3's constraint reproduces on the 27B).
* grafted drafter + -devd ROCm0: LOADS AND DRAFTS on BOTH stacks (prb-f0q4-p-d4,
  prb-q8lds-p-d4; acceptance 0.875/4.50 and 0.828/4.20 on the 64-token probe).
  THE GRAFT IS THE FIX: VERIFIED.
* Placement note: -devd ROCm0 = first VISIBLE card (HIP_VISIBLE_DEVICES=$DEVS), the
  H3-proven placement; the brief's "ROCm1" is read as "pin to one device".

## 2. A/B matrix (24 legs: 3 env x 2 depth x 2 head x 2 reps, interleaved)

Stacks/envs (drafter DFlash2-Q4_0 everywhere; b = borrowed stock, p = own-head + -devd):
  f0q4     = 27B Q4_0 TP2 + floor-0 + REPACK_Q8_0 pair + RP_Q8_NC_LDS=8 + RP_Q4_NC_LDS=8
  f128q4w4 = 27B Q4_0 TP2 + REPACK_Q8_0 pair + RP_Q8_NC_LDS=8 + RP_Q4_NC_LDS=4
  q8lds    = 27B Q8_0 TP2 + REPACK_Q8_0 pair + RP_Q8_NC_LDS=8

tg256 medians (natural suite; per-rep values in /root/taskR/*.reps.jsonl.agg.json):

| env        | d | borrowed tg (r1/r2) | pinned tg (r1/r2) | delta  |
|------------|---|---------------------|-------------------|--------|
| f0q4       | 4 | 55.48 / 54.57 = 55.02 | 57.65 / 57.54 = 57.60 | +4.67% |
| f0q4       | 7 | 43.90 / 43.88 = 43.89 | 44.20 / 44.20 = 44.20 | +0.71% |
| f128q4w4   | 4 | 50.46 / 50.05 = 50.25 | 48.59 / 46.72 = 47.66 | -5.16% |
| f128q4w4   | 7 | 32.41 / 28.20 = 30.30 | 33.82 / 27.93 = 30.87 | +1.88% (noisy) |
| q8lds      | 4 | 50.56 / 50.66 = 50.61 | 49.85 / 51.74 = 50.80 | +0.37% |
| q8lds      | 7 | 40.87 / 41.06 = 40.96 | 40.83 / 41.14 = 40.98 | +0.05% |

Baseline reproduction (borrowed arms vs prior ledgers, same build class): f0q4-b d4 55.02
vs taskQ 56.04 (-1.8% session drift), d7 43.89 vs 44.30 (-0.9%); q8lds-b d4 50.61 vs
taskL 50.44 (+0.3%), d7 40.96 vs 40.61 (+0.9%). Acceptance identical to the digit on
every cell (see gates), so the borrowed legs are the same compute as taskQ/taskL.

Data-quality notes: f128q4w4-d7 legs ran during heavy sibling contention (taskS/taskT
chains + fleet held gpu.lock much of the night; several legs re-queued via the
supervisor); r1 pair 32.41->33.82 (+4.4%), r2 pair 28.20->27.93 (-1.0%) - treat the d7
f128q4w4 delta as noise-bounded. q8lds-b-d4-r1 first attempt read 38.35 under
contention; its done-marker vanished and the supervisor re-ran it (50.56, consistent
with taskL); the table uses the clean re-run. Interleaving of that leg's reps is
inverted (r2 before r1) - noted for honesty.

## 3. Mandatory gates: ALL PASS (all 24 legs)

* Greedy 64-token temp-0 seed-42 probe: sha16 77841cb6a251235d on EVERY leg (both heads,
  all envs/depths) - byte-identical borrowed vs pinned (the canonical taskJ2/Q/L hash).
  The graft is numerics-preserving end-to-end.
* Acceptance digit-match (bench agg, draft_n/draft_n_accepted summed over the suite):
  f0q4 d4 0.3956/len 2.5822; f0q4 d7 0.2374/2.6615; f128q4w4 d4 0.3872/2.5489;
  f128q4w4 d7 0.2105/2.4737; q8lds d4 0.3544/2.4174; q8lds d7 0.2349/2.6445 -
  IDENTICAL between heads in every cell (and equal to taskQ/taskL references).

## 4. Drafter fwd (GGML_DFLASH_PROF, us/step medians)

draft-fwd collapses ~95-96% on every cell: f0q4 d4 11530 -> 523; f0q4 d7 15470 -> 566;
f128q4w4 d4 11673 -> 545; f128q4w4 d7 15849 -> 601; q8lds d4 12081 -> 554;
q8lds d7 16370 -> 559. draft-inject also drops (0.87 -> 0.32 ms at f0q4-d4);
tgt-decode rises slightly on the pinned arm (+0.6 ms f0q4-d4, +1.2 ms q8lds-d7 - the
pinned draft shares the card with the target's half).

## 5. Verdict vs the +14.2% prediction

H3's +14.2% DOES NOT TRANSFER to the dense 27B. Measured pinning deltas: +4.67% (f0q4 d4),
+0.71% (f0q4 d7), -5.16% (f128q4w4 d4), +1.88% (f128q4w4 d7, noisy), +0.37% (q8lds d4),
+0.05% (q8lds d7). Mean +0.4%.

Mechanism (step-time arithmetic, f0q4 d4): borrowed step = len/tg = 46.9 ms ~= tgt-decode
33.0 + draft-fwd 11.5 + inject 0.9 (the borrowed TP-split draft is nearly serial on the
MoE 35B too, but there its device cost was the allreduce storm H3 removed). Pinned step
= 44.8 ms while its device events sum to only ~34.5 ms: the -devd path (meta wrapper
dropped, split_mode LAYER) pays ~10 ms/step of host-mediated draft orchestration
(loop draft-phase sync - H3 saw the same effect at 4.6-6.4 ms), which eats most of the
device-side win on this stack. Net device-time saved (11 ms) ~= net wall-clock saved
(2.1 ms) only at f0q4-d4; on f128q4w4-d4 the card-sharing cost (tgt-decode +0.6 ms on a
slower verify path) exceeds the overlap gain => -5.2%.

## 6. Best serving config per stack

* 27B Q4_0, floor-0 serving (f0q4): OWN-HEAD PINNED drafter at d4: +4.7% same-session
  (57.60 vs 55.02; scales to ~58.7 in taskQ-session terms) - new best d4 cell. At d7
  pinned is +0.7% (44.20 vs 43.89, consistent in both reps) - small but never negative;
  pinned is fine at d7 too. Gates prove it is numerics-free, so adoption is a pure
  placement decision.
* 27B Q4_0, quality-neutral serving (f128q4w4): KEEP BORROWED head (-5.2% at d4; d7
  noise-bounded). Pinning does not pay on the floor-ON affine arm.
* 27B Q8_0 (q8lds): KEEP BORROWED head (neutral: +0.4% d4 / +0.05% d7, inside rep noise).

Net: the graft is a safe, numerics-preserving option that unlocks -devd on the 27B, and
is worth adopting ONLY for the f0q4 d4 flagship cell. The big H3 pinning win was an
MoE-35B-specific effect (replicated-draft allreduce storm); on dense TP2 stacks the
borrowed TP-split draft already overlaps enough that pinning nets ~0 outside floor-0-d4.

## 7. Files

* /root/taskR/graft_ownhead.py            - parameterized graft + verifies (committed)
* /root/taskR/draft/DFlash2-Q4_0-ownhead-q4.gguf / ...-q8.gguf - grafted drafters (not committed, 2.9/3.8 GB)
* /root/taskR/taskR-drive.sh matrix.sh supervise-r.sh agg-r.py - driver/matrix/retry/agg (committed)
* /root/taskR/{tag}.reps.jsonl[.agg.json], probe-{tag}.json, {tag}.server.log,
  {tag}.witness, {tag}.gate.witness - per-leg data (24 legs + 3 probes)
* Reproduce: bash /root/taskR/matrix.sh (refuse-if-exists); python3 /root/taskR/agg-r.py

Honest limits: single session (drift vs taskQ/taskL ~2%, paired deltas are within-session);
f128q4w4-d7 contention-noisy; tg medians over 8x3 bench reps per leg; no KLD legs (not
required - acceptance + greedy probe are exact-equivalence gates here).
