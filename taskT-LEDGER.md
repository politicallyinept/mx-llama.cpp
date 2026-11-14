# taskT ledger - adaptive-depth v2: explore-then-stick controller (LLAMA_DFLASH_ADAPTIVE=2)

Date: 2026-11-14. Branch `int4-t-adaptv2` off INT4-noquant @ 2a80a1c8ad (NOT pushed; parent merges).
Code commit: abf83967af - explore-then-stick depth controller in common/speculative.cpp,
env-gated, default OFF. Build: /root/build-t-adaptv2 (mi50-build.sh, TheRock pinned,
BUILD-PROVENANCE src_sha=abf83967af dirty=0). Driver /root/taskT-drive.sh, chain
/root/taskT-chain.sh, mt sim /root/taskT-mt.py, agg /root/taskT-agg.py, depth analyzer
/root/taskT-depth.py, data /root/taskT/ (AGG.txt = full aggregation).

## 0. Design (fixes the three H1 flaws by construction)

v1 (taskH1, LLAMA_DFLASH_ADAPTIVE=1): sticky per-request depth from (a) keyword classifier
(b) saturation memory keyed on the FULL-prompt hash. Flaws: (a) growing multi-turn prompts
never repeat the hash so sat never fires in real serving; (b) fixed sat threshold 4.4 is
toolchain-sensitive; (c) classifier missed math on one toolchain.

v2 (this task, LLAMA_DFLASH_ADAPTIVE=2): explore-then-stick per STABLE prompt identity.
* Stable identity: FNV-1a over the FIRST N prompt tokens (LLAMA_DFLASH_KEYTOK, default 64)
  = system + first-user-turn prefix. A growing multi-turn conversation maps to ONE key.
  (Bonus on the natural suite: text_pp and text_tg of one doc share their first 64 tokens,
  so pp and tg requests of one prompt share a key - observed and correct.)
* Per-key LRU memory (LLAMA_DFLASH_V2CAP, default 4096):
  phase 0 (first sight): run SHALLOW -> record mean accept len (baseline).
  phase 1 (second request): run DEEP -> record mean accept len (probe).
  phase 2 (decide, freeze): deep iff deep_mean_len >= LLAMA_DFLASH_RATIO * shallow_mean_len
  (ratio default 1.35 = the H1 separation rule; toolchain-adaptive by construction - no
  fixed thresholds, no keyword classifier). Sticks for the key thereafter.
* Depth still switches ONLY at begin() (request start): block width is baked into the draft
  graph (~40 ms rebuild on MI50). st.depth is written only in begin() (grep: lines 1458/1480,
  both inside begin()); the draft loop reads it per step -> mid-request switches impossible
  by construction (sticky).
* Server flag --spec-draft-n-max stays at the DEEP bound (7) so output/graph sizing covers
  the widest block (taskH1 sizing rule).
* v1 path preserved byte-for-byte under LLAMA_DFLASH_ADAPTIVE=1 (mode branch; v1 code moved
  verbatim into the else arm). Default OFF: gate-off path untouched (dctl.on false short-circuits).
* Telemetry: existing per-step accept() counters (drafts/accepted per request) -> mean_len
  = 1 + accepted/drafts, same estimator as v1. Requests with no draft steps (n_predict=1)
  record nothing and do not advance the phase (witnessed: probe_pp warmups stay phase 0).

## 1. Stack + arms

27B Q4_0 stack (taskQ best quality-neutral config, "f128q4w4"): Qwen3.8-27B-Q4_0.gguf TP2
+ DFlash2-Q4_0 drafter, env = REPACK_Q8_0 pair (REPACK_Q8_0=1 STOCK_ACC=0) + RP_Q8_NC_LDS=8
+ RP_Q4_NC_LDS=4, floor-ON (W4A4 MIN_TOKENS default 128), production bundle via mi50-run.sh,
flock gpu.lock, TERM-only, env witnessed per leg (/proc/PID/environ).
Arms: fixed-d4 (flag 4), fixed-d7 (flag 7), v2 (flag 7 + ADAPTIVE=2 SHALLOW=4),
v1 (flag 7 + ADAPTIVE=1, multi-turn flaw demo), gateoff (build-qmerge binary, flag 7).
Traffic: natural suite tg256 (bench.py, 8 prompts x 3 reps) + multi-turn sim (mt.py:
8 conversations x 5 turns = seed + 4 growing follow-ups, temp0 seed42, 128 tok/turn,
cache_prompt false). 2 reps per arm, interleaved d4/d7/v2. Acceptance digit-identical
across reps on every arm (deterministic).

## 2. Gates - ALL PASS

* Greedy probe byte-identical across controller states and binaries: the same 3-probe
  sequence (temp0 seed42 64tok) on every leg -> sha16 77841cb6a251235d on ALL 9 legs
  (d4 x2, d7 x2, v1, v2 x3, gateoff-on-build-qmerge). Matches the taskQ/taskN reference
  probe sha (77841cb6a251235d) = the stack's canonical greedy bytes. Depth never changes
  the verified output; 3/3 identical within every leg (determinism).
* Gate-off cross-binary identity: gateoff-r1 (build-qmerge @ 2a80a1c8ad, flag 7, no adaptive
  env) probe == build-t-adaptv2 arms == reference sha. The v2 code is inert when the env is unset.
* Sticky / no mid-request switch: st.depth written only in begin() (code-structural); depth
  log shows exactly one decision line per request; v2 trajectories show constant depth per
  request (see sec 4).
* v1 flaw demo on multi-turn (v1-r1): across the 40 growing conversation turns sat NEVER
  fires (sat=-1.000 on every turn line); the only sat lookups that hit are byte-identical
  repeats (warmup 3.493, probe 4.200) and both sit under the fixed 4.4 threshold, so sat
  never triggers deep anywhere in the leg. v2 on the same traffic: 8/8 conversations get
  both observations and a DECIDE (stable hash persists across all 5 growing turns - the
  flaw-(a) fix witnessed directly).

## 3. Single-turn results (natural suite tg256, median over 2 reps x 3 internal reps)

| prompt            | d4    | d7    | v2    | v2 decision (ratio)      |
|-------------------|-------|-------|-------|--------------------------|
| code-discussion   | 86.07 | 89.50 | 89.96 | DEEP  (1.375) -> d7      |
| dialogue          | 54.64 | 27.36 | 54.67 | shallow (0.652)          |
| econ-reasoning    | 54.47 | 39.13 | 53.97 | shallow (0.942)          |
| list-structured   | 39.43 | 26.14 | 39.06 | shallow (0.870)          |
| math-reasoning    | 71.44 | 67.99 | 70.57 | shallow (1.226)          |
| narrative-mystery | 31.50 | 22.59 | 31.50 | shallow (0.940)          |
| narrative-scifi   | 35.25 | 28.38 | 35.00 | shallow (1.047)          |
| tech-explain      | 46.60 | 43.62 | 46.44 | shallow (1.241)          |
| MEAN              | 52.43 | 43.09 | 52.65 |                          |

v2 = d4 +0.4%, d7 +22.2%. v2 captures the only deep winner (code, +4.5% over d4, matching
the d7 value) and keeps every prose prompt at the d4 number. The ratio rule called every
prompt correctly in hindsight on THIS toolchain - including math (1.226 -> shallow; d7 math
67.99 < d4 71.44 here, the mirror of H1's classifier-misses-math flaw: v2 needs no
toolchain-specific knowledge). Note math is NOT a deep winner on this 27B Q4_0 stack
(unlike H1's build-h1 35B toolchain) - v2 discovered that automatically.

## 4. Multi-turn results (8 conversations x 5 growing turns, tg mean over 2 reps)

| conversation      | d4    | d7    | v1    | v2    | v2 mt decision (ratio) |
|-------------------|-------|-------|-------|-------|------------------------|
| code-discussion   | 61.23 | 57.04 | 57.10 | 55.80 | shallow (1.134)        |
| dialogue          | 54.55 | 52.25 | 53.97 | 54.37 | shallow (1.334)        |
| econ-reasoning    | 61.58 | 55.33 | 61.69 | 59.58 | shallow (0.825)        |
| list-structured   | 55.63 | 47.17 | 56.05 | 56.47 | DEEP (1.353) -> d7     |
| math-reasoning    | 60.56 | 49.38 | 60.75 | 60.82 | shallow (1.059)        |
| narrative-mystery | 51.27 | 46.57 | 51.67 | 54.99 | shallow (1.189)        |
| narrative-scifi   | 59.68 | 54.57 | 59.56 | 57.68 | shallow (0.841)        |
| tech-explain      | 55.35 | 51.02 | 50.56 | 53.66 | shallow (0.880)        |
| OVERALL           | 57.48 | 51.66 | 56.42 | 56.67 |                        |

v2 = d7 +9.7%, d4 -1.4%. v1 = d4 -1.9%: the classifier forces deep on code/tech mt where
deep LOSES in mt context (tech mt v1 50.56 vs d4 55.35) and sat never fires to help anything.

Mean tg by turn: turn 2 is the deep-probe turn for v2 (46.77 vs d4 54.82 = the exploration
tax, -14.7% on one turn of five); turns 3-5 recover to d4 parity or better (turn 5 v2 61.79
> d4 54.83, driven by the list-structured deep-stick turn-5 spike 90.7 t/s).

Trajectory witness (v2-r1, identical in r2): every conversation shows
`p0/d4 -> p1/d7 -> DECIDE -> p2/dX p2/dX p2/dX` under ONE stable hash across all 5 growing
turns. Single-turn bench shows the same per prompt (6 requests: pp/tg warmups + 3 tg reps).

NEW FINDING - draft-KV width-history effect: after a deep probe, later shallow turns in the
SAME conversation show different acceptance than pure-shallow (code mt turns 4-5: mean len
2.79 vs d4's 3.48; tg 50.4 vs 61.0) although prompt, depth and final text are identical
(prompt_n identical across arms; probe bytes identical). The persistent draft-context KV
carries the width history of earlier requests (block-width interference, taskH1 mechanism,
now visible across requests); it perturbs draft PROPOSALS (speed), never the verified
output. This is most of v2's -1.4% vs d4 on mt: the probe tax (turn 2) + post-probe
proposal perturbation roughly cancel the deep-stick gains on this 5-turn suite. Longer
conversations amortize the probe; the effect is an argument for the ratio rule being
per-key, not per-request.

## 5. Verdict + serving recommendation

1. v2 works as designed and passes every gate: stable-identity keying fixes flaw (a) (sat-
   never-fires on growing traffic demonstrated for v1: 48 misses / 0 useful hits; v2 decides
   8/8 conversations), the ratio rule fixes flaws (b)/(c) (no fixed threshold, no classifier;
   correct calls on this toolchain including math).
2. Single-turn: v2 ~= fixed-d4 (+0.4%) and +22.2% vs fixed-d7; it captures the deep winner
   on repeated prompts at zero prose cost. fixed-d4 remains a fine default for pure
   single-turn traffic; ADAPTIVE=2 is safe to enable (gate-off byte-identical, probe-identical
   when on).
3. Multi-turn/chat: v2 is the only arm that adapts per conversation: +9.7% vs fixed-d7,
   -1.4% vs fixed-d4 net on this suite (probe tax + draft-KV history perturbation). v1 is
   NET-NEGATIVE vs d4 (-1.9%) and should NOT be used for chat. RECOMMENDATION: serve mixed/
   chat traffic with LLAMA_DFLASH_ADAPTIVE=2 LLAMA_DFLASH_SHALLOW=4 at flag n-max 7 on this
   stack; expect d4-class throughput on prose conversations and deep-class on structured
   ones, with a one-turn probe cost per conversation. Do not ship v1.
4. Follow-ons (not built): multi-observation exploration (decide after k>=2 pairs to cut the
   single-observation noise - dialogue mt decided at 1.334 vs threshold 1.35, borderline);
   draft-KV rollback of mask rows after deep probes (would remove the post-probe
   perturbation); per-width wave selection for RP_Q4_NC_LDS (taskQ follow-on) unchanged here.

Honest limits: n=8 prompts/conversations, one toolchain (TheRock-pinned build), 2 reps;
sibling taskS legs shared the box mid-campaign (rep CV small, acceptance deterministic,
tg carries a few % contention noise on rep-2 legs); mt suite is synthetic follow-ups on
natural seeds. Full data: /root/taskT/AGG.txt + per-leg jsonl/logs.
