# taskU ledger - fill the two dangling Q4_0-stack composition cells (q4stack LDS x q4_0-KV x floor-0)

Date: 2026-11-14 (PST). Box: mi50, TP2 pair GPU0+GPU1 (lane U2 on 2,3; shared /root/gpu.lock,
hold -w 2700 per leg). Branch `int4-u-compcells` off INT4-noquant @ 0340ba0a1e (NOT pushed).
Build: /root/build-qmerge (src_sha 2a80a1c8ad, dirty=0). The 2a80a1c8ad..0340ba0a1e delta is
common/speculative.cpp only (taskT adaptive-depth, env-gated default OFF, gate-off byte-identity
proven by taskT's own cross-binary leg) - build-qmerge is numerically tip-equivalent for every
arm here; no rebuild needed. dmesg: no uncorrectable ECC on rocm0/1 at task start.

## 1. Question

Two cells dangle on the 27B Q4_0 serving table (taskQ/taskN baselines):
1. **q4stack** - taskK's starved arm: the Q8_0-tensor LDS pair (REPACK_Q8_0=1 STOCK_ACC=0 +
   RP_Q8_NC_LDS=8) on the Q4_0 stack with q4_0 KV (taskK-drive.sh arm q4stack: Q4_0 main +
   repack pair + LDS8 + -ctk/-ctv q4_0). On the Q8_0 stack the pair is worth +58% (taskJ2);
   the Q4_0 GGUF tensor map (24/48 ssm_out Q8_0 = 5.1% of verify bytes, lm_head Q6_K) predicts
   a sub-half-strength effect. Measure d4/d7 vs the taskQ baselines (f128lds 49.09/30.08).
2. **floor-0 x q4_0-KV cross-product** - J1's q4_0-KV win (+42% DFlash-on, +14% over plain,
   53.74 at n4) was measured WITHOUT the stack env (no LDS pair, floor default, Q8_0 drafter,
   -c 65536). Does it stack with f0lds-class env on the taskQ stack? With and without the
   LDS pair, floor-0 and floor-ON: the full 2x2x2 (floor x LDS x KV-type) per width.

## 2. Protocol

Stack = taskQ's exactly (every new cell is a pure delta vs the taskQ baselines):
Qwen3.8-27B-Q4_0.gguf TP2 + DFlash2-Q4_0 drafter, server -ngl 99 -fa on -sm tensor --fit off
-c 8192 -ub 1024 -t 16 -np 1 -lm dio, production bundle via mi50-run.sh, HSA_KERNARG_POOL_SIZE
=67108864. NOTE vs taskK-drive.sh's literal q4stack arm (Q8_0 drafter, build-jmerge): rebased
onto the taskQ Q4_0-drafter stack for apples-to-apples vs f128lds/f0lds; the LDS tensor-map
effect is a main-model property (ssm_out), drafter-independent.
Arms: ctrl-f128lds / ctrl-f0lds = taskQ controls (drift + gate-off byte-identity on THIS
binary); f128ldsq4kv = ctrl-f128lds + `-ctk q4_0 -ctv q4_0` (cell 1); f0ldsq4kv = ctrl-f0lds +
q4KV (cell 2a); f0q4kv = floor-0 + q4KV, LDS pair OFF (cell 2b); f128q4kv = floor-ON + q4KV,
LDS pair OFF (4th corner). Widths d4 (verify w5) and d7 (verify w8). bench.py natural suite
(8 prompts x 3 reps tg256), GGML_DFLASH_PROF, greedy 64-tok temp0 seed42 probe per leg.
Numeric gates per arm: (a) greedy probe vs the canonical sealed bytes sha16 77841cb6a251235d;
(b) acceptance digit-match - controls vs taskQ references, and across reps; q4KV on/off shifts
are reported as numerics readouts (spec decode is lossless, so acceptance = distribution shift);
(c) KLD: taskI1 protocol re-score (35B Q8_0 TP2, wikitext 16ch x2048, vs
/root/taskI1/i1-base-f16.logits - certifies the shipped q4_0-KV envelope on THIS binary) PLUS
27B-local narrow-window scoring (taskQ harness -b 8 -c 512, 64KB wikitext; q4KV arms vs the
taskQ f16-KV base logits q-f128lds.logits / q-f0lds.logits - isolates the q4_0-KV delta on the
serving stack itself).
Ops: flock gpu.lock -w 2700 per leg, pair pinned 0,1 (VRAM-free gate on GPU0+1), load<16 gate,
RAM>=12GiB gate, TERM-only teardown + pgrep -x llama-server empty after every leg, refuse-if-
exists. Driver /root/taskU-drive.sh, chains /root/taskU-chain{,2}.sh, data /root/taskU/.

## 3. Results - the completed grid (tg256 natural-suite medians | accept | mean acc len)

Carried baselines: taskN d4 f128 39.68/0.3489/2.3955, f0 46.15/0.3320/2.3280;
taskQ d4 f128lds 49.09, f0lds 54.98, f0q4 56.04 (RP_Q4_NC_LDS lever - different axis);
taskQ d7 f128lds 30.08/0.2105/2.4737, f0lds 34.04/0.2374/2.6615, f0q4 44.30, f128q4w4 33.96.

### 3.1 d4 (verify width 5)

| floor | LDS pair OFF, f16 KV | LDS pair OFF, q4_0 KV | LDS pair ON, f16 KV | LDS pair ON, q4_0 KV |
|---|---|---|---|---|
| floor-ON  (MIN_TOKENS=128) | 39.68 (taskN) | **56.43** /0.4377/2.7508 NEW | 48.32 ctrl (taskQ 49.09) | 47.08 /0.3913/2.5652 NEW |
| floor-0   (MIN_TOKENS=0)   | 46.15 (taskN) | 47.77 /0.3668/2.4671 NEW | 54.83 ctrl (taskQ 54.98) | 52.45 /0.3817/2.5269 NEW |

### 3.2 d7 (verify width 8)

| floor | LDS pair OFF, f16 KV | LDS pair OFF, q4_0 KV | LDS pair ON, f16 KV | LDS pair ON, q4_0 KV |
|---|---|---|---|---|
| floor-ON  | (not measured, out of scope) | 38.20 /0.2604/2.8231 NEW | 30.08 (taskQ) | 37.67 /0.2552/2.7867 NEW |
| floor-0   | (not measured, out of scope) | 34.73 /0.2433/2.7031 NEW | 34.04 (taskQ) | **51.69** /0.3301/3.3105 NEW |

### 3.3 Marginal effects (the composition answer)

* **Cell 1 (q4stack, floor-ON + LDS + q4KV): 47.08 d4 / 37.67 d7.** vs f128lds: -2.6% d4,
  +25.2% d7. The d7 gain comes from q4KV, not the LDS pair: the LDS pair's marginal effect
  UNDER q4KV at floor-ON is NEGATIVE at both widths: -16.6% d4 (56.43 -> 47.08),
  -1.4% d7 (38.20 -> 37.67). At floor-ON the LDS pair and q4_0-KV are SUBSTITUTES -
  both attack the same verify-width bandwidth bottleneck; combining them buys nothing
  (d4: 47.08 < max(56.43, 48.32)). At width 5 the LDS repack path costs more than it
  gives once q4KV has already cheapened the verify pass.
* **Cell 2 (floor-0 x q4KV): does J1's win stack on f0lds-class env? d4 NO, d7 YES - big.**
  f0ldsq4kv = 52.45 d4 (-4.3% vs f0lds 54.83) but **51.69 d7 (+51.8% vs f0lds 34.04)** -
  the new d7 champion, beating taskQ's f0q4 (44.30) by +16.7%. Without the LDS pair
  (f0q4kv): 47.77 d4 (+3.5% vs taskN f0), 34.73 d7.
* **Three-way interaction floor-0 x LDS x q4KV at width 8**: the LDS pair is worth +48.8%
  on top of floor-0+q4KV at d7 (34.73 -> 51.69) but -1.4% on top of floor-ON+q4KV. The
  taskK tensor-map prediction ("sub-half-strength vs +58%") HOLDS at floor-ON (~0-22%
  depending on KV type) and FAILS at floor-0+d7 (+48.8%). The composite is dominated by
  this interaction, not by the Q8_0-tensor byte share alone; acceptance corroborates the
  numerics shift (f0lds-d7 0.2374 -> f0ldsq4kv-d7 0.3301, len 2.6615 -> 3.3105).
* q4KV marginal vs J1: f128q4kv-d4 56.43 vs taskN f128 39.68 = +42.2% - J1's +42% DFlash-on
  jump REPRODUCES on the taskQ stack (Q4_0 drafter, -c 8192); acceptance 0.4377 vs J1's
  0.4463 (different drafter/context, same shape).

## 4. Numeric gates

* Greedy 64-tok probe (temp0 seed42): **10/10 legs byte-identical to the canonical sealed
  bytes sha16 77841cb6a251235d** (taskQ/taskN/taskT reference). No arm flips the greedy
  argmax on the probe - including all four q4_0-KV arms and both floor-0 arms.
* Drift controls on build-qmerge: ctrl-f128lds 48.32/0.3872/2.5489 vs taskQ 49.09/0.3872/
  2.5489; ctrl-f0lds 54.83/0.3956/2.5822 vs taskQ 54.98/0.3956/2.5822. Acceptance/len
  DIGIT-IDENTICAL both, tg within -1.6%/-0.3% (same-session noise band). Binary drift: NONE.
* taskI1 protocol re-score (35B Q8_0 TP2, q4_0-KV vs f16 base logits, THIS binary):
  Mean KLD **0.011223 +/- 0.000237**, Max 1.351633, PPL ratio 1.006371 (+0.64%) - reproduces
  taskI1/J1 to the last digit. Shipped q4_0-KV envelope: GREEN on build-qmerge.
* 27B-local narrow-window KLD (q4KV arm vs same-env f16-KV taskQ base logits, -b 8 window):
  - f128q4kv    vs q-f128lds: Mean KLD 0.006855 +/- 0.000217, Max 0.978106, PPL +0.43%. PASS
  - f128ldsq4kv vs q-f128lds: Mean KLD 0.007062 +/- 0.000265, Max 1.070654, PPL +0.33%. PASS
    (LDS-pair-independent, as expected: same q4KV delta with or without the pair)
  - f0ldsq4kv   vs q-f0lds:   Mean KLD **0.055151 +/- 0.001696**, Max 5.394522, PPL +0.035%.
    AMBER: the floor-0 x q4KV interaction is ~8x the floor-ON q4KV delta and exceeds the
    J1 0.011223 envelope precedent on this window. PPL impact is negligible and the greedy
    probe is byte-identical, but the distribution shift is real - flag the d7 champion
    (f0ldsq4kv) as numerics-aggressive (floor-0 KL class + q4KV envelope + interaction).
  - f0q4kv      vs q-f0lds:   NOT RUN - supplementary leg starved by sibling lane
    taskV2-r7 (66k-context logits dump held load AND RAM gates past the retry budget;
    chain2 terminated at task budget). The MANDATORY taskI1-protocol envelope gate above
    covers all q4_0-KV arms on this binary; the f0ldsq4kv leg (same floor-0 base logits)
    is the stack-side readout for the floor-0 x q4KV interaction.
* Acceptance digit-match: deterministic within-arm (greedy verify); controls match taskQ to
  the digit (sec 4). q4KV acceptance shifts are systematic numerics readouts: d7 f128 pair
  0.2105 -> 0.2552 (+21% rel), f0 pair 0.2374 -> 0.3301 (+39% rel); d4 mixed (-2.6% f0,
  +1.1% f128). r2 reproduction legs starved by sibling contention (see f0q4kv note).
* Gate witnesses: every leg carries proc.witness (cmdline incl. -ctk/-ctv where applicable,
  /proc/PID/environ, lib maps - TheRock only) + gate.witness (REPACK_Q8_0/RP_Q8_NC_LDS/
  MIN_TOKENS banners). This fork logs no KV-type banner at load (same as taskK); the KLD
  legs above are the engagement proof - an inert -ctk flag would score KLD ~0, not 0.0069.

## 5. Verdict - final Q4_0-stack serving-config table (tg256 natural suite, TP2, DFlash2-Q4_0)

| config (all on production bundle)                                   | d4     | d7     | numerics cost |
|---|---|---|---|
| floor-ON, no LDS, f16 KV (sealed default)                           | 39.68  | -      | none (reference) |
| floor-ON + LDS pair (f128lds)                                       | 48.32  | 30.08  | none vs floor-ON generic (taskJ2-class) |
| floor-ON + RP_Q4_NC_LDS (f128q4/f128q4w4, taskQ)                   | 54.30  | 33.96  | none (bit-identical kernels) |
| floor-0 + LDS pair (f0lds)                                          | 54.83  | 34.04  | floor-0 KL class |
| **floor-ON + q4_0 KV (f128q4kv) - NEW BEST d4**                     | **56.43** | 38.20 | q4_0-KV envelope (KLD 0.0112 35B / 0.0069 27B, PPL +0.4-0.6%) |
| floor-0 + LDS + q4_0 KV (f0ldsq4kv) - NEW BEST d7                   | 52.45  | **51.69** | floor-0 class + q4KV + interaction (27B KLD 0.055 AMBER) |
| floor-ON + LDS + q4_0 KV (q4stack, cell 1)                          | 47.08  | 37.67  | q4KV envelope; NO gain over max(f128q4kv, f128lds) |

RECOMMENDATIONS:
1. d4 serving: adopt **f128q4kv** (floor-ON, NO LDS pair, -ctk q4_0 -ctv q4_0): 56.43, the
   best d4 cell, beats f0lds without paying floor-0's quality class; KLD inside the shipped
   q4_0-KV envelope; probe byte-identical. Do NOT add the LDS pair on top at floor-ON
   (substitutes, -16.6%).
2. d7 serving: **f0ldsq4kv** (floor-0 + LDS pair + q4_0-KV) = 51.69 (+52% vs f0lds-d7, +17%
   vs f0q4) IF the 0.055 narrow-window KLD interaction is acceptable for the workload
   (PPL +0.035%, greedy probe byte-identical). Conservative alternative: f128q4kv 38.20
   (envelope-clean, KLD 0.0069). Quality-neutral floor: f128q4w4 33.96 (taskQ).
3. The q4_0-KV lever is stack- AND width-dependent (taskK verdict stands for Q8_0-main): on
   the Q4_0 stack it is worth +42% d4 / +25..+52% d7 depending on floor/LDS; it substitutes
   for the LDS pair at floor-ON and multiplies with it at floor-0/width-8.

Honest limits: single rep per cell (bench.py internal 3 reps; r2 spot-check legs starved by
sibling contention, not run); d7 f16-KV LDS-off cells not measured (out of scope); 27B KLD
window is a 64KB wikitext subset at -b 8 (narrow-window protocol, taskJ2/taskQ class);
sibling lanes (U2 + taskV2-r7) shared the box - tg cells carry a few % contention noise,
acceptance deterministic; f0ldsq4kv-d7 51.69 is a single-rep champion number resting on its
deterministic acceptance (0.3301/3.3105) - an r2 leg is the obvious confirmation leg.

## 6. Artifacts

* /root/taskU/: per-arm server logs, reps.jsonl(+agg.json), probes, proc/gate witnesses,
  kld-i1-q40.log, kld27-*.log, chain.log, chain2.log.
* /root/taskU-drive.sh (arms/gates/kld), /root/taskU-chain.sh, /root/taskU-chain2.sh,
  /root/taskU-agg.py.
* This ledger: taskU-LEDGER.md on branch int4-u-compcells (worktree /root/wt-u). NOT pushed.
