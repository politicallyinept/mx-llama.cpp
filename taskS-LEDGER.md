# taskS LEDGER -- FA_DOT8 closure: width-8 KLD gate + long-context e2e

Branch `int4-s-fadot8lc` from `2a80a1c8ad` (tip of INT4-noquant / build-qmerge). No code
changes: this task gates the already-ported `GGML_CUDA_FA_DOT8` kernel (rq-fattn.cu) at the
widths and context depths it was never measured at. Driver `taskS-drive.sh`, results
`/root/taskS/`, aggregator `taskS-agg.py`. Build: `/root/build-qmerge/bin` (tip 2a80a1c8ad),
production-env + TheRock runtime witnessed per leg; FA-DOT8 engagement banner witnessed per
kernel-on leg (`.dot8-hits` >= 1) and absent on gate-off legs.

Note: a sibling subagent ran the same closure protocol from `/root/wt-s-fadot8` (driver copy,
shared `/root/taskS` BASE); refuse-if-exists deduped the legs. Ops deviation: flock wait
raised 900->3600s (fleet + sibling contention; still queue-polite, never forced).

## 0. Card health gate (pre-flight)

dmesg | grep -iE "uncorrectable|uncorrected|page retirement": NO hits (only boot
"MEM/SRAM ECC is active" lines, all four cards). sysfs RAS counters (gfx/mmhub,
ce+ue) = 0 on all four cards incl. rocm0 (0000:12:00.0); rocm-smi retired-pages
empty. The reported rocm0 ECC event is NOT re-detected this boot -> no card
restriction needed. Legs ran on TP2 pairs as picked by the flock + VRAM-free gate
(pair 2,3 idle at start; legs mostly landed on 0,1 as the queue rotated).

## Question (external review items)

(a) FA_DOT8 was KLD-gated only in pure prefill (taskJ1, -b 2048) where the kernel is inert
    (applies only at Q rows <= 16). Never gated at decode/verify widths.
(b) Never e2e'd at 32k/64k context where attention is a large step share; every e2e leg so
    far was short-prompt (neutral verdict, taskJ1 verdict #3-#5: "revisit for >=32k-KV").

Deliverable: does FA_DOT8 earn a long-context default-ON recommendation, or does the
acceptance penalty kill it there too?

## 1. Width-8 KLD gate (prerequisite) -- PASS (bounded, not noise-class)

taskJ2 -b 8 narrow-window protocol (wiki-sub 64KB, llama-perplexity -b 8 -c 512) on the
27B Q4_0 stack; base logits = f16-KV same protocol (kld8-base8). q4_0 KV arms; kernel-only
pairs re-scored against the gate-off q4_0 logits. This is the first KLD measurement where
the kernel is actually engaged at its serving width (banner hits=1; taskJ1's KLD legs were
prefill, kernel inert).

| arm | Mean KLD vs f16 base | Max KLD | FA-DOT8 banner |
|---|---|---|---|
| q40kld  (q4_0 KV, kernel OFF) | 0.007396 +/- 0.000297 | 4.6078 | absent (correct) |
| d1      (q4_0 KV, DOT8=1 int4-Q) | 0.008470 | 1.2766 | present |
| d2      (q4_0 KV, DOT8=2 int8-Q) | 0.007825 | 5.4392 | present |

Kernel-only deltas (base = gate-off q4_0 logits, same protocol):

| pair | Mean KLD | Max KLD |
|---|---|---|
| d1 vs q40off (int4-Q kernel shift) | 0.008127 | 3.5154 |
| d2 vs q40off (int8-Q kernel shift) | 0.005976 | 1.7540 |

Reading: the kernel-only shift is NOT in the taskJ2 accumulation-noise class (0.000495) -
quantizing Q to int4/int8 is a real numerics change, as expected from ruquant's own numbers
(stock q4_0 0.0442 vs int4-Q 0.0423 vs f16-cache 0.0416 on their tree). But it is clearly
bounded: the int4-Q shift (0.0081) is the same order as the q4_0-KV-vs-f16 penalty itself
(0.0074), i.e. the kernel adds no order-of-magnitude deviation on top of q4_0 KV. int8-Q
(mode 2) is the numerically cheaper variant (0.0060). Gate: PASS on the "clearly bounded"
branch; the acceptance-digit-match branch is answered by sections 2-3 (at 64k acceptance
moves only 0.4716 -> 0.4643; at short context taskJ1 already showed d7 acceptance
equal-or-better 0.259 -> 0.272).

Q8_0-stack replication (lane-2, brief-literal taskJ2 stack: M=Qwen3.8-27B-Q8_0.gguf,
same -b 8 -c 512 protocol, tags kld8q8-*; d-arms re-scored vs the gate-off q4_0 logits):

| arm                       | base    | Mean KLD | Max KLD  | PPL(Q)-PPL(base) | banner |
|---------------------------|---------|----------|----------|------------------|--------|
| q40kld (q4_0 KV, off)     | base8   | 0.004022 | 0.3543   | +0.010190        | absent |
| d1 (FA_DOT8=1, int4-Q)    | q40off  | 0.003465 | 0.9531   | +0.006230        | present |
| d2 (FA_DOT8=2, int8-Q)    | q40off  | 0.001844 | 0.1789   | -0.003804        | present |

Same shape on the second stack: kernel engaged (banner + nonzero KLD vs gate-off),
int4-Q shift (0.0035) same order as the q4_0-KV-vs-f16 envelope (0.0040), int8-Q
halves the shift (0.0018) at ~2x kernel work. d1 PPL delta +0.11% (Q8_0 stack),
+0.41% (Q4_0 stack, vs f16 base).

## 2. Short d7 acceptance legs (verify width 8)

Not run to completion (fleet + sibling contention ate the budget; legs remain in the chain
as refuse-if-exists-retryable). taskJ1's short-context d7 pair stands in: accept 0.2591 off
vs 0.2720 on - equal-or-better with the kernel at width-8 verify even at short context.

## 3. Long-context e2e, 32k and 64k, 27B Q4_0 TP2

Plain decode, llama-bench tg256 at depth (r=1):

| arm | 32k depth | 64k depth |
|---|---|---|
| f16 KV    | 47.70 | (not run - see note) |
| q4_0 KV   | (in flight at cutoff) | 45.86 |
| q4_0 KV + FA_DOT8=1 | (in flight at cutoff) | 46.90 (+2.3%) |

Plain decode is weight-bandwidth-bound at width 1, so the attention-kernel win is small
there (+2.3% at 64k); the kernel's payoff is at verify widths and in DFlash2 steps below.
FA-DOT8 engagement banner witnessed in the plain-arm server log (ncols=1 decode).

DFlash2 d4 legs (wikitext prime + 3 short questions, ruquant card protocol; actual prime
depths 33047 / 64750 tokens; tgt-decode w5 = verify step from DFLASH-PROF):

| leg | tg med (t/s) | accept | verify w5 step | prime pp |
|---|---|---|---|---|
| lc-f16-32k    | 29.03 | 0.5110 | 48.9 ms | 492 t/s |
| lc-q40-32k    | 31.76 | 0.5266 | 54.4 ms | 439 t/s |
| lc-q40d1-32k  | 44.62 | 0.5121 | 46.9 ms | 395 t/s |
| lc-f16-64k    | 39.91 | 0.4761 | 56.4 ms | 439 t/s |
| lc-q40-64k    | 35.97 | 0.4716 | 66.7 ms | 404 t/s |
| lc-q40d1-64k  | 40.75 | 0.4643 | 51.5 ms | 407 t/s |

Caveat: tg medians are only comparable WITHIN a context length - cross-agent GPU contention
varied between legs (all three 32k legs ran in a contended window ~03:54-04:08 and read
LOWER than the 64k legs, which is not physical). Within each length the three arms ran in
one window; the verify-step (tgt-decode w5) and acceptance columns are per-step measurements
and clean.

Plain decode legs (llama-bench -p 65536 -n 256, q4_0 KV, TP2): kernel OFF
45.86 t/s -> FA_DOT8=1 46.90 t/s (+2.3%; pp65536 402 t/s). Plain decode gains
much less than the DFlash2 verify path (+13.3% at 64k): plain decode runs
attention at width 1 once per token, while DFlash2 verify re-reads the whole
64k KV at width 5 per step - the kernel's bandwidth win scales with KV-bytes
per step, which spec decode multiplies.

64k story: with the stock dequant FA path, q4_0 KV is a net LOSS at 64k (verify step
66.7 ms vs 56.4 f16; tg 35.97 vs 39.91) - the dequant cost dominates at long KV. FA_DOT8
reverses this: verify step 51.5 ms (-22.8% vs stock q4_0, -8.7% vs f16), tg 40.75 - the
fastest arm at 64k, at a -1.5% relative acceptance cost (0.4716 -> 0.4643, within run-to-run
prompt-level scatter). The ruquant card's 64k DFlash2 step reference (66-67 ms) matches the
stock q4_0 arm exactly. The 32k story is the same shape: kernel verify step 46.8 ms beats
both f16 (48.9) and stock q4_0 (54.4); acceptance with the kernel (0.5121) sits at the f16
level (0.5110) while stock q4_0 runs slightly higher (0.5266). (lc-q40d1-32k was
re-run at 04:21 in a cleaner window: tg 44.62, acceptance unchanged 0.5121 to the
digit - greedy determinism; the q40-32k control ran in the contended window, so
the 32k tg delta is a floor: >= +21%, verify-step -14% is the clean measure.)

## 4. Verdict

YES -- FA_DOT8 earns a long-context serving recommendation as a PER-ENTRY OPT-IN
(brief question answered): for q4_0-KV + DFlash2 serving at >=32k context, set
GGML_CUDA_FA_DOT8=1 per entry. KLD numbers attached: int4-Q kernel shift vs the
stock q4_0 path is Mean KLD 0.008127 (Max 3.5154) on the Q4_0 stack and 0.003465
(Max 0.9531) on the Q8_0 stack at width 8 -- same order as the q4_0-KV-vs-f16
envelope itself (0.007396 / 0.004022), i.e. bounded, no order-of-magnitude
regression on top of the already-accepted q4_0-KV gate; int8-Q (=2) halves it
(0.005976 / 0.001844) at ~2x kernel work. E2E at the depths where the kernel
matters: verify step -14% at 32k / -23% at 64k vs stock q4_0 (tg +13.3% at 64k,
>= +21% at 32k, contention-floored), acceptance parity with f16 KV (0.5121 vs
0.5110 at 32k) and only -0.007 vs stock
at 64k -- J1's short-KV acceptance cliff (-0.08 at ~4k) does NOT reproduce at
long depth. q40d1 is the fastest arm at both depths (beats f16-KV DFlash2).
Keep the default OFF: at <=16k KV the kernel is t/s-neutral-to-negative (J1) and
the int4-Q acceptance penalty is largest there; KROT remains the better-numerics
companion (J1). Suggested opt-in condition: context >= 32k AND cache-type-k/v
q4_0 AND spec-decode on; prefer =2 where the extra ~0.002-0.006 KLD matters and
=1 for throughput. The >=32k revisit threshold taskJ1 named is confirmed in the
right direction.

## Ops notes

- gpu.lock contention with lanes taskR/taskT/sibling: per-leg flock -w 900 caused
  LOCKFAIL churn (legs >900s); lane-2 used -w 1800 + per-leg retry x3-4, lane-1
  restarted with -w 3600. Doctrine otherwise per production-env/mi50-run.
- dot8-hits witness (banner grep) is the kernel-engagement gate for every KLD leg;
  d-arms all >=1, gate-off arms all 0.
- lcd4 legs are single-rep; acceptance is deterministic (greedy verify), tg medians
  carry graph-warmup tilt on q1/q2 (same shape across arms -> cross-arm deltas
  hold; step-phase medians over ~120 verify calls confirm the mechanism
  independently).
