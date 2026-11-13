# taskK ledger — 2x2 combine: q4_0 KV x LDS narrow matvec on 27B Q8_0 DFlash2 d7

Date: 2026-11-13 (PST). Box: mi50 (3x MI50 gfx906, TheRock runtime). Branch int4-k-combine created
from a63c468d60 exactly (INT4-noquant tip at task start; tip later moved to 15196ec20f via parent
re-merge — branch NOT created from the moved tip, per instruction). Build under test:
/root/build-jmerge (merged tree containing GGML_CUDA_FA_DOT8 [left OFF on every leg] and
GGML_CUDA_RP_Q8_NC_LDS). No pushes; local commits only.

## 1. Question

J1: q4_0 KV flips 27B DFlash2-on ABOVE plain decode on the Q4_0 stack (53.7 vs 47.14, n4).
J2: LDS narrow matvec flips DFlash2-on above plain on the Q8_0 stack (39.71 vs 37.06, d7).
Does the stack COMBINE on the Q8_0 stack: is q4_0KV + LDS > max(39.71, q4_0KV-alone)?

## 2. Protocol

Driver /root/taskK-drive.sh (modeled on taskJ2 drive.sh), outputs /root/taskK/.
M=Qwen3.8-27B-Q8_0.gguf (unsloth), drafter D=Qwen3.8-27B-DFlash2-Q8_0; server:
--spec-type draft-dflash --spec-draft-n-max 7 -ngl 99 -fa on -sm tensor --fit off -c 8192
-ub 1024 -t 16 -np 1 -lm dio; q4_0 arms add -ctk q4_0 -ctv q4_0; LDS arms add
GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0 GGML_CUDA_RP_Q8_NC_LDS=8.
Legs (interleaved): q40-r1, q40lds-r1, ldsctrl-r1 (drift control), q40-r2, q40lds-r2; then KLD.
Each leg: bench.py natural suite (8 prompts x 3 reps tg256) + GGML_DFLASH_PROF + greedy 64-tok
temp0 seed42 probe. f16-KV cells carry over from J2 (gate-off byte-identity was proven).
Numeric gates for q4_0-KV on this stack: greedy probe vs f16-KV base probe, acceptance delta,
KLD re-score vs /root/taskJ2/j2-base-f16.logits (taskJ2 harness: llama-perplexity -b 8 -c 512,
64KB wikitext subset).
Ops: flock /root/gpu.lock per leg (wait raised 900->2700s after sibling-agent legs caused LOCKFAIL
churn, same finding as taskJ1 ops note), TERM-only teardown, pgrep -x llama-server empty after each
leg, production env + TheRock runtime witnessed per leg. A llama-swap-owned production server
(Huihui Q8_0, GPU1+2) held the GPUs 03:00-04:05; never touched per ops rule; legs waited for its
ttl expiry. A sibling agent's legs shared the lock throughout (04:06-04:18, 04:46-04:55,
05:04-05:12, 05:23-05:36 waits) — adds cross-rep t/s noise, acceptance unaffected.

## 3. Results

### 3.1 The 2x2 (tg256 natural-suite medians, DFlash2 d7)

| cell                 | reps t/s        | accept | acc_len | source            |
|----------------------|-----------------|--------|---------|-------------------|
| f16 KV, LDS off      | 25.14 / 25.13   | 0.2224 | 2.557   | J2 base (carried) |
| f16 KV, LDS on       | 39.71 / 39.61   | 0.2327 | 2.629   | J2 lds (carried)  |
| q4_0 KV, LDS off     | 21.69 / 23.47   | 0.2211 | 2.548   | NEW q40           |
| q4_0 KV, LDS on      | 38.47 / 39.58   | 0.2565 | 2.795   | NEW q40lds        |
| LDS control (jmerge) | 38.06           | 0.2327 | 2.629   | NEW ldsctrl-r1    |

Plain decode reference: 37.06 (J2 s5a). Acceptance is digit-identical across reps within each arm
(greedy verify deterministic).

Drift control: ldsctrl-r1 on build-jmerge = 38.06 vs build-j2 39.71/39.61 (-4.1pct t/s), but
acceptance 0.2327 identical to the digit and greedy probe byte-identical -> numerically identical
path; the t/s gap is environmental (sibling-agent contention), not build drift.

### 3.2 Step-phase (GGML-DFLASH-PROF medians, us/call)

| arm       | draft-fwd | tgt-decode w8 | vs J2 ref          |
|-----------|-----------|---------------|--------------------|
| q40-r1    | 19262     | 79880 (n=3306)| base: 19180 / 79503|
| q40lds-r1 | 19344     | 49003 (n=3025)| lds:  18908 / 48317|
| ldsctrl-r1| 19296     | 48533 (n=3337)| lds:  18908 / 48317|

Draft fwd unchanged on every arm (gate does not touch the M=1 draft path, as designed). LDS cuts
verify-w8 by -39pct regardless of KV type; q4_0 KV adds ~+1pct to verify-w8. The q40 arm's -10pct
e2e vs base is NOT visible in the w8 verify median — it sits in the per-token attention/dequant
slice outside the verify graph (consistent with J1: stock q4_0-KV dequant path costs ~12pct tg).

### 3.3 Numeric gates (q4_0-KV on the Q8_0 stack) — GREEN

* Greedy 64-token probe (temp0 seed42, d7): q40-r1/r2, q40lds-r1/r2, ldsctrl-r1 ALL byte-identical
  to the f16-KV baseline (sha16 77841cb6a251235d, same as J2 base/lds probes). q4_0 KV does not
  flip the greedy argmax on the probe.
* Acceptance delta: q40 0.2211 vs base 0.2224 (-0.6pct rel, noise); q40lds 0.2565 vs ldsctrl
  0.2327 (+10.2pct rel). q4_0 KV raises draft agreement when the LDS kernel is active; the raw
  dequant-path cost cancels it at d7.
* KLD vs f16 base logits (taskJ2 harness, llama-perplexity -b 8 -c 512, 64KB wikitext subset):
  - q40 arm:    Mean KLD 0.004022 +/- 0.000114, Max KLD 0.354321, PPL ratio 1.001826 (+0.18pct),
    Max dp 14.520pct.
  - q40lds arm: Mean KLD 0.004025 +/- 0.000126, Max KLD 0.554895, PPL ratio 1.001076 (+0.11pct),
    Max dp 48.943pct.
  Both well inside the J1 q4_0-KV envelope precedent (0.011223 PASS on the 35B prefill protocol);
  the LDS delta is invisible at this window (0.004025 vs 0.004022 — J2's own LDS KLD was 4.95e-4,
  same accumulation-noise band, sub-additive here). Max-dp tails are rare-token artifacts that do
  not move the greedy argmax (probes byte-identical). GREEN both arms.

## 4. Verdict

THE STACK DOES NOT COMBINE AT d7. q4_0KV+LDS = 39.02 (median of 38.47/39.58) vs the LDS-alone
official 39.71 (-1.7pct) and vs the same-build LDS control 38.06 (+2.5pct, inside the 2.9pct
rep-to-rep spread). No meaningful gain over LDS alone; the +10pct relative acceptance gain of the
combined arm is exactly eaten by the stock q4_0-KV dequant attention cost at d7.

Cell-by-cell: q4_0 KV ALONE is a -10pct LOSS on this stack at d7 (22.58 vs 25.14) — unlike the
Q4_0 stack at n4 where it was +42pct (J1). The q4_0-KV lever is stack- and width-specific; it
does not transfer to Q8_0-main at 8-token verify.

DELIVERABLE — projected best standard-quant serving config for 27B DFlash2 d7:
  27B Q8_0 + DFlash2-Q8_0 drafter d7, TP2, production env,
  GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0 GGML_CUDA_RP_Q8_NC_LDS=8,
  f16 KV (NO q4_0 KV), FA_DOT8 OFF  ->  ~39.7 t/s natural tg256 (+7pct over plain decode 37.06).
q4_0 KV adds nothing on this stack/width; keep f16 KV. (J1's q4_0-KV win lives on the Q4_0 stack at
n4 — a different axis; the sibling n4 legs on this stack cover the width question.)

## 5. Secondary: Q4_0-stack LDS tensor-map check

Qwen3.8-27B-Q4_0.gguf tensor map: ssm_out is MIXED — 24 of 48 ssm_out tensors are Q8_0 (the LDS
gate applies to those after repack), the rest Q4_0; lm_head (output.weight) is Q6_K, NOT Q8_0 —
the upstream claim's lm_head half does not exist on this file. Partial justification for one LDS
leg on the Q4_0 stack (53.7 baseline with q4_0-KV at n4). Leg q4stack-r1 NOT RUN: queued last in
the chain, starved by sibling-agent lock contention past the time budget; the partial tensor map
(half ssm_out Q8_0, lm_head Q6_K) predicts a sub-half-strength effect and the n4-vs-d7 axis is
already covered by sibling n4 legs on the Q8_0 stack. Driver support (arm q4stack) is in
/root/taskK-drive.sh if the parent wants the leg.

## 6. Artifacts

* /root/taskK/: drive.sh, chain.log, per-arm server logs, reps.jsonl(+agg), probes, witnesses,
  kld-q40.log, kld-q40lds.log.
* /root/taskK-drive.sh (driver), /root/taskK-chain.sh, /root/taskK-agg.py.
* This ledger: taskK-LEDGER.md on branch int4-k-combine (worktree /root/wt-k). NOT pushed.
