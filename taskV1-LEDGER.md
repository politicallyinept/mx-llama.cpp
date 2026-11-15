# taskV1 ledger - confirmation legs on the 27B Q4_0 stack (build-uwave): d7 champion r2, f0q4kv 27B-KLD cell, pinned f0q4-d4 solo

Date: 2026-11-15 (PST). Box: mi50, TP2 pair A GPU0+GPU1 (HIP_VISIBLE_DEVICES=0,1 pinned; sibling
lanes taskV3-pinsolo + taskV2-moe shared /root/gpu.lock on pair 2,3). Branch `int4-v1-confirm`
off INT4-noquant @ e9b3e3197f (NOT pushed).
Build: /root/build-uwave, BUILD-PROVENANCE src_sha=e9b3e3197f dirty_files=0 (built 2026-11-14T12:56).
NOTE: the dispatch brief cited tip "e9b581387f" - no such object exists in the repo; the live tip
at leg time was e9b3e3197f and the build matches it exactly (verified per the brief's own
"verify git log -1 first" instruction). rocm0/1 cleared: doctor ALL CLEAR, ECC/RAS clean at start.

Stack = taskU/taskQ protocol exactly (pure delta vs those ledgers): Qwen3.8-27B-Q4_0.gguf TP2 +
DFlash2-Q4_0 drafter, -ngl 99 -fa on -sm tensor --fit off -c 8192 -ub 1024 -t 16 -np 1 -lm dio,
production bundle via mi50-run.sh, HSA_KERNARG_POOL_SIZE=67108864, bench.py natural suite
(8 prompts x 3 reps tg256), GGML_DFLASH_PROF, greedy 64-tok temp0 seed42 probe per leg.
Driver /root/taskV1-drive.sh (clone of taskU-drive.sh with BIN/BASE swapped + taskR f0q4 arm),
chain /root/taskV1-chain.sh, data /root/taskV1/. Ops: flock /root/gpu.lock -w 900 per leg (brief),
TERM-only teardown, pgrep -x llama-server empty after every leg (chain.log: zero WARN/STUCK),
refuse-if-exists. One LOCKFAIL round (00:06-00:21, sibling lanes held the lock past the 900s
window); retry #2 of the chain's 3x retry loop acquired it - no leg lost to contention.

## 1. Leg 1 - f0ldsq4kv-d7 r2 (the +52% d7 champion, solo confirmation) + same-session control

| leg (build-uwave, pair 0,1)  | tg256 med | pp     | accept | mean len | probe byte-id |
|---|---|---|---|---|---|
| ctrl-f0lds-d7-r1 (control)   | 34.82   | 544.6  | 0.2374 | 2.6615   | PASS |
| ctrl-f0lds-d7-r2 (control)   | 34.68   | 547.1  | 0.2374 | 2.6615   | PASS |
| f0ldsq4kv-d7-r2 (champion)   | **52.09** | 541.1 | 0.3301 | 3.3105  | PASS |

* Control-to-control spread r1 vs r2: 0.40% - the same-session base is tight; both controls
  digit-match taskQ's acceptance (0.2374/2.6615) and sit +1.9..+2.3% on taskQ's 34.04 tg
  (same-day band). Binary drift vs taskQ/taskU era: NONE observable at this resolution.
* Champion r2 52.09 vs taskU r1 51.69 (single rep, under contention): **+0.78% - REPRODUCES**.
  Acceptance digit-identical to r1 (0.3301/3.3105).
* Champion vs same-session control mean (34.75): **+49.9%** (taskU quoted +51.8% vs the inherited
  taskQ 34.04; the honest same-session number is +49.9%).
* VERDICT: the d7 champion cell CONFIRMED on the current tip: f0ldsq4kv = 52.09 t/s solo.

## 2. Leg 2 - f0q4kv 27B-KLD battery (the taskU-starved supplementary; completes floor-0 x q4KV)

taskQ narrow-window protocol (llama-perplexity -b 8 -c 512, 64KB wikitext, vs the sealed f16-KV
base logits /root/taskQ/q-f0lds.logits), on build-uwave:

| arm (vs q-f0lds f16-KV base) | Mean KLD          | Max KLD  | PPL ratio | verdict |
|---|---|---|---|---|
| f0q4kv   (floor-0, q4KV, LDS OFF) | 0.055359 +/- 0.001736 | 6.125607 | 1.005980 (+0.60%) | AMBER |
| f0ldsq4kv (floor-0, q4KV, LDS ON) re-cert | 0.055151 +/- 0.001696 | 5.394522 | 1.000352 (+0.035%) | AMBER (digit-identical to taskU/build-qmerge) |

* The f0q4kv cell is now MEASURED: 0.055359 - statistically identical to f0ldsq4kv's 0.055151.
  The floor-0 x q4KV KL shift is **LDS-pair-independent**: the interaction is floor-0 x q4KV
  itself, not LDS-mediated. Both corners of the floor-0 x q4KV cell now carry KLD; the taskU
  AMBER flag extends to the LDS-off corner. ~8x the floor-ON q4KV delta (0.0069), exceeds the
  J1 0.011223 envelope precedent on this window; PPL cost small (+0.60%), greedy probe
  byte-identical - same numerics-aggressive class as the d7 champion, as taskU flagged.
* The f0ldsq4kv re-cert reproduces taskU's build-qmerge value to the LAST DIGIT
  (0.055151 +/- 0.001696, max 5.394522) - independent cross-binary confirmation that the
  2a80a1c8ad..e9b3e3197f delta (taskV2-R7 env-gated default-OFF + taskX T=16 i8narrow restore)
  is numerically inert for this stack, corroborated by the 5/5 probe byte-identity below.
* Acceptance: N/A on KLD legs (llama-perplexity, no drafter). Arm acceptance on record from
  taskU bench legs: f0q4kv d4 0.3668/2.4671, d7 0.2433/2.7031.

## 3. Leg 3 - pinned-drafter f0q4-d4 solo confirmation (taskR adoption cell hardened)

Quiet-box solo reps on build-uwave, same-session pair (b = borrowed-head control run first):

| leg (f0q4 arm: floor-0 + pair + RP_Q8_NC_LDS=8 + RP_Q4_NC_LDS=8) | tg256 med | accept | mean len | probe byte-id |
|---|---|---|---|---|
| f0q4-b-d4-r1 (borrowed head, quiet control) | 55.47 | 0.3956 | 2.5822 | PASS |
| f0q4-p-d4-r1 (own-head graft + -devd ROCm0, pinned) | **57.09** | 0.3956 | 2.5822 | PASS |

* Pinned solo 57.09 vs taskR contended median 57.60: **-0.89% - REPRODUCES** within noise.
  Acceptance digit-identical to taskR (0.3956/2.5822) and bit-equal between heads (graft is
  numerics-preserving); all 8 per-prompt medians positive for p over b (+2.7%..+4.0%).
* The quiet-box same-session delta is **+2.90%** (57.09/55.47), NOT taskR's contended +4.67%:
  the borrowed control rose under quiet (55.02 -> 55.47) while the pinned arm dipped slightly.
  taskR's own suspicion (host-mediated gather/inject inflated by 3-lane CPU contention) is
  CONFIRMED - the +4.7% was contention-inflated. Hardened adoption number: **+2.9% same-session
  quiet** (+1.9% vs the 56.04 unpinned baseline). Direction holds at d4; ADOPT stands, the
  ledger number to quote is now 57.09/+2.9%, not 57.60/+4.7%.
* KLD gate for this arm: kld27-f0q4 vs q-f0lds.logits = Mean KLD 0.000000 +/- 0.000000,
  Max 0.000071, PPL +0.028% - the RP_Q4_NC_LDS=8 lever is bit-identical class vs the f16-KV
  base on the uwave binary (taskQ's prediction confirmed).

## 4. Mandatory gates - all GREEN

* Greedy 64-tok probe (temp0 seed42): **5/5 bench legs byte-identical to the canonical sealed
  bytes sha16 77841cb6a251235d** (recomputed from taskU's sealed probe file; the brief's
  "77841cb6a251d4" is a corrupted copy of this canonical - first 11 hex agree). Includes both
  q4_0-KV legs, both floor-0 arms, and the pinned-drafter leg.
* KLD vs f16-KV base: sec 2 (both floor-0 x q4KV corners) + sec 3 (f0q4 arm, ~0). The
  taskI1-protocol 35B envelope gate is already on record for this binary lineage (taskU
  reproduced 0.011223 to the digit on build-qmerge; the digit-identical 27B re-cert above
  covers the uwave delta).
* Acceptance reported per bench leg (tables above); deterministic within-arm, digit-matches the
  taskU/taskQ references in every control.
* Teardown: TERM-only, `pgrep -x llama-server` empty after every leg (zero WARN/STUCK lines).
* Production env: /root/production-env.sh asserted by mi50-run.sh on every leg; proc.witness
  carries the full env (6 levers + HSA_FORCE_FINE_GRAIN_PCIE=1, LD_LIBRARY_PATH therock-first,
  HSA_KERNARG_POOL_SIZE=67108864); zero RUNTIME-FAIL. Gate banners witnessed per leg
  (REPACK_Q8_0 ACTIVE, RP_Q8_NC_LDS/RP_Q4_NC_LDS ACTIVE, MIN_TOKENS floor, -ctk/-ctv and
  -devd ROCm0 present in the server cmdlines). DFLASH-PROF counters non-zero on every bench
  leg (path engaged).
* Lock: flock /root/gpu.lock -w 900 per leg per brief; pair A (0,1) pinned; one LOCKFAIL round
  recovered by chain retry. llama-swap untouched.

## 5. Verdicts

1. d7 champion CONFIRMED: f0ldsq4kv-d7 = 52.09 t/s solo on the current tip (+49.9% vs
   same-session f0lds-d7 control; acceptance deterministic). taskU's single-rep 51.69 stands.
2. floor-0 x q4KV cell COMPLETE: f0q4kv 27B KLD 0.055359 AMBER, LDS-independent - the KL
   interaction is floor-0 x q4KV itself. Serving guidance unchanged from taskU: the floor-0 +
   q4KV class is numerics-aggressive at both LDS settings.
3. Pinned-drafter d4 adoption HARDENED: 57.09 t/s solo (-0.9% vs recorded 57.60), quiet-box
   delta +2.9% (the contended +4.7% was inflated by sibling CPU contention; direction holds,
   magnitude corrected). Acceptance bit-equal, probe byte-identical, arm KLD ~0.

## 6. Artifacts

* /root/taskV1/: {ctrl-f0lds-d7-r1,ctrl-f0lds-d7-r2,f0ldsq4kv-d7-r2,f0q4-b-d4-r1,f0q4-p-d4-r1}
  .{server.log,reps.jsonl,reps.jsonl.agg.json,bench.log,probe-*.json,witness,proc.witness,
  gate.witness,prof-count,run.out,.done}; kld27-{f0q4kv,f0ldsq4kv,f0q4}.log(+.done); chain.log.
  NOTE: /root/taskV1 pre-existed from the Nov-11 w4a4-profile campaign (kld-control/repack/stock
  logs etc.); no tag collisions - this task's files are the ones above.
* /root/taskV1-drive.sh, /root/taskV1-chain.sh, /root/taskV1-agg.py.
* This ledger: taskV1-LEDGER.md on branch int4-v1-confirm (worktree /root/wt-v1). NOT pushed.
