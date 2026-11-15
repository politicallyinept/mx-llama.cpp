# taskW3 ledger - fill the two missing matrix cells (f128q4kv x pinned drafter; TP3 on 1,2,3)

Date: 2026-11-15 (PST). Box: mi50. Branch `int4-w3-cells` off INT4-noquant tip c0dd09e4c9
(NOT pushed; worktree /root/wt-w3). Build: /root/build-uwave, llama-server md5
b956a6858d8976624271378f97d825cb - the e9b3e3197f..c0dd09e4c9 delta is LEDGER-files-only
(git diff --stat), so build-uwave is tip-equivalent; no rebuild needed.
GPU0 QUARANTINED per parent health note (919k correctable ECC era); every leg ran
HIP_VISIBLE_DEVICES=1,2 or 1,2,3 - device 0 never touched. The :9400 exporter read
0/0 on all four cards at session start and end (counters appear re-armed since the
taskV3-era 919,793 reading); quarantine kept regardless per brief.
NOTE on the brief's canonical probe string "77841cb6a235d": the sealed sha16 on the box
(taskU/taskV1/taskV3 ledgers, probe json sha256[:16]) is 77841cb6a251235d - the brief
string is a truncation typo of the same hash; all byte-identity gates below use the
sealed value.

## 1. Questions (the two cells both external reviews flagged)

1. **f128q4kv + PINNED drafter on the 27B** - pinning was only ever measured on
   f0q4/q8lds/f128q4w4 (taskR/taskV3); never on the f128q4kv quality-clean champion
   (taskU 56.43 d4). Paired reps x2, pinned-vs-borrowed delta with noise floor,
   adopt/don't verdict.
2. **Champion configs at TP3 (GPUs 1,2,3)** - every published comparison is TP2;
   run f128q4kv-d4 and pinned f0q4-d4 at TP3, compare vs their TP2 numbers, verdict
   per config.

## 2. Protocol

Stack = taskU/taskV3 exactly (pure delta vs those ledgers): Qwen3.8-27B-Q4_0.gguf +
DFlash2-Q4_0 drafter (borrowed) or DFlash2-Q4_0-ownhead-q4.gguf (pinned, -devd ROCm0 =
first VISIBLE card = physical GPU 1), graft sha256 7920500c... re-verified byte-exact
vs taskV3 record before launch. Server: -ngl 99 -fa on -sm tensor --fit off -c 8192
-ub 1024 -t 16 -np 1 -lm dio; f128q4kv adds -ctk q4_0 -ctv q4_0 with NO W4A4 env levers
(floor-ON default); f0q4 = MIN_TOKENS=0 + REPACK_Q8_0/STOCK_ACC=0 + RP_Q8_NC_LDS=8 +
RP_Q4_NC_LDS=8, f16 KV. Production bundle via mi50-run.sh + production-env.sh,
HSA_KERNARG_POOL_SIZE=67108864, env+maps witnessed per leg from /proc/PID.
bench.py natural suite (8 prompts x 3 reps tg256 + pp4096), GGML_DFLASH_PROF, greedy
64-tok temp0 seed42 probe per leg. Refuse-if-exists per leg.
Arms/legs (driver /root/taskW3-drive.sh, chain /root/taskW3-chain.sh, data /root/taskW3/):
  TP2 pair 1,2: f128q4kv {b,p} d4 r1,r2 (cell 1); f0q4 p d4 r1,r2 (TP2 ref for TP3).
  TP3 GPUs 1,2,3: f128q4kv b d4 r1,r2; f0q4 p d4 r1,r2.
  KLD (config changes numerics): taskQ narrow-window protocol (llama-perplexity -b 8
  -c 512, 64KB wikitext) - capture same-config TP2 logits (w3-<arm>-tp2.logits), then
  score the TP3 run against them. Cell 1 needs no KLD: the own-head graft is a byte-copy
  of the target head and pinning is numerically neutral - proven per-leg by probe
  byte-identity + acceptance digit-match below (taskH3/taskV3 precedent).
Ops: flock /root/gpu.lock per leg behind taskW1 (1,2) + taskW2 (2,3); TERM-only teardown,
pgrep -x llama-server empty after every leg (final sweep 0). One LOCKFAIL round at
12:14 behind a sibling hold >45 min; per the shared-lock doctrine the flock wait was
raised 900->2700 s via atomic swap at 13:02 (new invocations get the new file; running
leg kept the old inode). At 13:27 TERMed a FOREIGN orphan server (taskW1's build-w1,
port 8452, owner driver dead, 40 min at 101% CPU / 0% GPU spin) that was blocking the
pair-free gate - logged in /root/taskW3/foreign-orphan-action.log; died on first TERM.
First chain pass died instantly on kldcap (self-referencing `local` under set -u - the
known taskV2-moe trap); fixed by splitting the local lines, chain restarted 13:49,
refuse-if-exists made the restart free. 14/14 legs DONE rc=0, zero contamination reruns.

## 3. Cell 1 - f128q4kv-d4, pinned vs borrowed (TP2 pair 1,2, paired reps)

| leg                | tg256 med | pp    | accept | mean len | draft-fwd us/step | probe byte-id |
|--------------------|----------:|------:|-------:|---------:|------------------:|---------------|
| f128q4kv-b-d4-r1   | 56.29     | 539.9 | 0.4377 | 2.7508   | 11781             | PASS          |
| f128q4kv-p-d4-r1   | 58.33     | 545.2 | 0.4377 | 2.7508   | 515               | PASS          |
| f128q4kv-b-d4-r2   | 55.93     | 542.8 | 0.4377 | 2.7508   | 11676             | PASS          |
| f128q4kv-p-d4-r2   | 58.06     | 544.5 | 0.4377 | 2.7508   | 541               | PASS          |

* **Pin delta: mean +3.72% (per-rep +3.63% / +3.80%), noise floor 0.63% -> DECISION-GRADE.**
  Both reps positive, spread 0.17pp. Matches the f0q4-d4 pin cell (+3.69%, taskV3) to
  the tenth of a point - the pin gain is drafter-placement, not arm-specific.
* Mechanism witness: borrowed draft-fwd ~11.7 ms/step -> pinned ~0.52 ms/step (the pin
  removes the host-mediated draft sync + replicated allreduce, taskH3/H3).
* Acceptance DIGIT-IDENTICAL across all four legs (0.4377 / 2.7508) and equal to the
  taskU reference - pinning is numerically neutral; no KLD required for this cell.
* Borrowed cross-session check: 56.11 median vs taskU 56.43 = -0.57% (no build drift).
* VERDICT: **ADOPT pinned drafter for f128q4kv-d4.** New envelope-clean champion cell:
  f128q4kv-p-d4 = 58.20 t/s median (pair 1,2, solo). Quality envelope unchanged (q4KV
  KLD certified by taskU 0.006855 + taskI1 re-score; pin adds zero numerics).

## 4. Cell 2 - TP3 (GPUs 1,2,3) vs TP2 (pair 1,2), same session

| config        | TP2 tg (r1/r2) | TP3 tg (r1/r2) | mean delta | per-rep        | noise floor | grade          |
|---------------|----------------|----------------|-----------:|----------------|-------------|----------------|
| f128q4kv-b-d4 | 56.29 / 55.93  | 17.29 / 17.27  | **-69.2%** | -69.28 -69.12  | 0.63%       | DECISION-GRADE |
| f0q4-p-d4     | 57.92 / 57.47  | 50.04 / 50.51  | **-12.9%** | -13.62 -12.10  | 0.95%       | DECISION-GRADE |

TP2 same-session refs: f0q4-p 57.69 median vs taskV1 57.09 (+1.06%) and taskV3
58.70/57.36/57.03 - consistent, no drift. f0q4-p TP2 reps differ at the 4th decimal
(0.3956 vs 0.3946) - the taskV3-flagged pinned-arm accept nondeterminism on one prompt
class recurs here; no byte impact (probes identical), verdicts unaffected.

**f128q4kv-b-d4 at TP3: BROKEN, do not adopt.** Deterministic garbage, both reps:
greedy probe = "3333..." (sha16 79bd7d7fd684b399, NOT canonical), acceptance 0.0000
(0 accepted / 1010+ generated per leg), tg pinned at ~17.3 on EVERY prompt (uniform,
not contention - reps identical to 0.02 t/s). Server log: `internal AllReduce init
failed (n_devices != 2?); falling back to meta-backend butterfly` - the fork's custom
AR is TP2-only. Attribution: the TP3 q4KV numerics themselves are FINE in isolation -
the perplexity KLD leg (TP3 vs same-config TP2 logits, no drafter) scored Mean KLD
0.005879 +/- 0.000159, PPL ratio 1.000053 (same class as taskU's q4KV delta 0.0069).
So the corruption is the SERVING path: TP3 tensor-split x borrowed DFlash drafter
(borrows target head via ctx_other) x q4KV - the target logits themselves go garbage
(spec decode is lossless, so garbage OUTPUT = target-path corruption, not draft noise).
File as bug: do not ship q4KV + borrowed-drafter on TP3 on this fork; a pinned
(own-head) f128q4kv TP3 leg would isolate drafter-vs-KV further (not in scope).

**f0q4-p-d4 at TP3: healthy but LOSES, do not adopt.** Probe BYTE-IDENTICAL to the
canonical sealed bytes on both TP3 legs (greedy output unchanged); acceptance
0.3946/0.3956 (TP2) -> 0.3487 (TP3) - a real but modest numerics shift, quantified by
KLD vs same-config TP2 logits: Mean 0.053374 +/- 0.001615, Max 4.190319, PPL ratio
0.9987 (same floor-0 KL class as taskV1's 0.055359). tg -12.9% decision-grade: the
published "TP3 wins tg on some classes" does NOT transfer to this stack - TP3 server
logs show repeated `tg-capture ok=0 ... shape permanently on per-subgraph dispatch`
(draft-width graph shapes fail capture under the 3-way meta split -> eager dispatch),
and the 3-way allreduce costs more per verify step than the narrower shards save.
VERDICT per config: **TP2 stays the serving topology for both champions.**

## 5. Gates summary

* Greedy 64-tok probe (temp0 seed42): byte-identical to canonical 77841cb6a251235d on
  10/10 TP2 legs and both healthy f0q4-p TP3 legs; the two f128q4kv TP3 legs diverge
  deterministically (garbage - reported above, not a gate dodge).
* Acceptance: reported for all 10 bench legs (sec 3/4); digit-match holds within every
  same-config pair except the broken TP3 f128q4kv arm (0.0) and the TP3 f0q4 shift
  (0.3487 vs 0.3946/0.3956) - both are the TP3 numerics readout itself.
* KLD where numerics change: TP3-vs-TP2 per config (taskQ narrow window, same-binary,
  same window, only topology differs): f128q4kv 0.005879 (GREEN, perplexity path);
  f0q4 0.053374 (floor-0 class, matches taskV1 precedent). Cell 1: N/A by design
  (numerically neutral graft, proven by probe + acceptance identity).
* Teardown: TERM-only, 0 stuck, 0 residual; final pgrep -x llama-server = 0; lock free.

## 6. Files

* Driver /root/taskW3-drive.sh (taskV3 clone + f128q4kv arm + kv flags + tp3 mode +
  kldcap/kldscore), chain /root/taskW3-chain.sh, aggregator /root/taskW3/taskW3-agg.py.
* Data /root/taskW3/: 14 x (reps.jsonl + agg.json + server.log + bench.log + probe +
  proc/gate witnesses), kldcap/kld logs, w3-{f128q4kv,f0q4}-tp2.logits, chain.log,
  ecc-witness-start.txt, foreign-orphan-action.log.
* Ledger committed on branch int4-w3-cells (this file), NOT pushed.
