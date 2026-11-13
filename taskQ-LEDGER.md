# taskQ LEDGER — Q4_0 LDS-staged narrow matvec (branch int4-q-q4lds)

Date: 2026-11-13. Branch `int4-q-q4lds` off INT4-noquant @ 3d122c72b5 (NOT pushed; parent merges).
Code commit: 6d1c30a8f2 — `rp_q4_nc_lds<T,W,U4ACT,SUMW>` (repack-kernels.cuh) + dispatch gate
`GGML_CUDA_RP_Q4_NC_LDS=<2|4|8|16>` (q8_repack/mul-mat.cu), DEFAULT 0 = OFF. Build:
/root/build-q-q4lds (mi50-build.sh, TheRock pinned). Microbench: /root/mb_q4lds_test
(ggml/src/ggml-cuda/q8_repack/mb_q4lds_test.cu). Driver /root/taskQ-drive.sh, data /root/taskQ/.

## 0. Why (external review consensus, highest-priority lane)

The taskJ2 LDS-staging pattern (rp_q8_nc_lds, +57.8% e2e on the 27B Q8_0 DFlash2 d7 stack)
attacks the DFlash verify-width window (T=4-8). On the Q4_0-main stack the current best config
(f0lds = floor-0 forced-W4A4 + REPACK pair + Q8 LDS, 55.27 t/s) pays floor-0's quality cost
(narrow prefill KL 0.864 nats, acceptance len -2.8%). A quality-neutral Q4_0 LDS kernel either
(a) stacks on f0lds (U4ACT arm) or (b) replaces the floor-0 component (affine arm at floor-ON).

## 1. Design

Port of rp_q8_nc_lds to the Q4_0 nc family: same lanes (64/row), same block order, same
products/sums/reduction as `mul_mat_vec_repacked_nc<2,1,T,2,64,Q4_0,U4ACT,false,false,false,SUMW>`
=> BIT-IDENTICAL to that shape per decode arm (the claim the microbench seals). Each step's
activation blocks are staged in LDS once per W-wave workgroup (2 rows/lane shared).
Two decode arms, both gated by the same env:
  * affine (U4ACT=false): stock q8_1-activation decode (load_w nibble split, {d,8d} epilogue,
    sx = activation sum from the DS4 slot). Serves floor-ON / no-floor-0 serving at widths 4-8.
  * U4ACT=true (+SUMW threaded): dp8a decode over raw nibbles (floor-0 arm), activations
    block_q8_1_u4 staged as qs uint4 + {d4,m} floats. sw summed over RAW codes pre-XOR,
    signed before the float cast (the 3b class), SUMW plane honored to match the generic
    dispatch instantiation bit-for-bit.
Widths 2-3 keep the existing shapes (staging does not amortize below 4, taskJ2 finding).
Gate-off: the dispatch-site branch is never taken; every existing path byte-identical.

## 2. Coverage (honest tensor map, Qwen3.8-27B-Q4_0.gguf, gguf-py dump)

Verify-width weight bytes per full target pass (TP2 splits rows; shares unchanged):

| class | tensors | GB/pass | share | served by |
|---|---|---|---|---|
| Q4_0  | 369 (attn/ffn/ssm linears) | 12.78 | 83.1% | THIS kernel (both arms) |
| Q6_K  | output.weight 5120x248320 (6.7%) + 6x attn_q | 1.44 | 9.4% | NOT covered (no Q6_K LDS) |
| Q8_0  | 24x ssm_out 6144x5120 | 0.78 | 5.1% | taskJ2 Q8 LDS gate (in the pair) |
| Q4_1  | 8x ffn_down 17408x5120 | 0.38 | 2.5% | NOT covered; not repack-admitted at DOT8=3 anyway (stock path) |

lm_head is Q6_K on this file (taskK finding): upstream lm_head claims assume a Q8_0 lm_head
this GGUF does not have. The Q4 LDS gate covers 83.1% of verify-width weight bytes; the
composite e2e delta is bounded by that share (the Q6_K lm_head half of the verify pass is
untouched — a Q6_K LDS twin is the obvious follow-on lane, NOT in scope here).
Draft tensors: DFlash2-Q4_0 sidecar is NOT repack-admitted (use_extra_bufts=false, taskN S4),
so the drafter forward never reaches the nc window; the gate is target-verify-only by construction.

## 3. Microbench (mb_q4lds_test.cu, one MI50 under flock, clean run)

Bars ALL PASS at every shape: rp_q4_nc_lds (W=8 and W=4) == generic
`mul_mat_vec_repacked_nc<2,1,T,2,64,Q4_0,...>` BIT-EXACT for BOTH decode arms
(affine + U4ACT/SUMW), affine pair == CPU oracle (oracle_bad=0), launch-deterministic.
dx/sx staged as DISTINCT random halves so a swap bug could not pass.

us/call medians, clean run (/root/taskQ/mb-clean.txt; % = speedup vs generic):

| shape (K x N)        | arm | T=4           | T=5           | T=8            |
|----------------------|-----|---------------|---------------|----------------|
| 5120x6144 (ssm/gate) | aff | +26.9 / +34.7 | +23.0 / +4.0  | +8.7 / +33.6   |
| 5120x17408 (ffn_up)  | aff | +14.5 / +13.6 | (see below)   | +2.7 / +27.6   |
| 5120x10240 (qkv)     | aff | +16.7 / +15.5 | (see below)   | +4.7 / +25.2   |
| 17408x5120 (ffn_down)| aff | +16.2 / +17.5 | (see below)   | +6.9 / +34.5   |
| 5120x6144            | u4  | -0.7 / +2.7   | +1.2 / -2.8   | +71.3 / +51.8  |
| 5120x17408           | u4  | -9.6 / -10.2  | -6.0 / -11.8  | +72.5 / +51.4  |
| 5120x10240           | u4  | (n/a)         | -5.5 / -10.7  | +70.3 / +49.1  |
| 17408x5120           | u4  | (n/a)         | +6.2 / +2.0   | +68.1 / +51.1  |

(aff T=5 ffn/qkv/ffn_down cells were measured in the first under-load run: +26.0/+17.5/+16.0
at lds8; the clean run covers ssm T=5 aff + all T=5 u4.)

Reading: AFFINE arm wins broadly at T=4-5 (+13..+35%) and at T=8 with W=4 (+25..+35%);
W=8 at T=8 is flat (+2.7..+8.7%). U4ACT arm is flat-to-negative at T=4-5 but +68..+73%
at T=8 with W=8 - the dp8a decode is compute-dense enough that the LDS win only lands
when 8 activation streams share the stage. Wave choice by width: T=4-5 -> W=8 (affine)
or either (u4, flat); T=8 -> W=4 affine, W=8 u4. The e2e legs below confirm this split.
T=1/T=2 reference (no LDS arm): gen-affine 72.6/80.1 us at 5120x17408.

## 4. Numeric gates

* Gate-OFF byte-identity PROVEN: f0lds-d4-r1 on the NEW binary (RP_Q4_NC_LDS unset) vs
  taskN f0lds reference: greedy 64-token probe byte-identical (sha16 77841cb6a251235d),
  acceptance digit-identical (0.3956 / len 2.5822), tg 54.98 vs 55.31/55.22 (same-session
  noise band). The dispatch-site branch is never taken when the env is unset.
* Gate-ON quality neutrality (by construction + witnessed): the LDS kernels are bit-identical
  to the generic nc shapes (section 3), so the gate-on arms must not move acceptance or the
  greedy argmax. END-TO-END CONFIRMED at d4: f0q4 acceptance 0.3956/2.5822 == f0lds control
  to the digit, probe byte-identical; f128q4 acceptance 0.3872/2.5489 == f128lds reference
  (taskN) to the digit, probe byte-identical.
* KLD narrow-window legs DONE (taskJ2 protocol, llama-perplexity -b 8 window, pair-isolated
  vs same-env base logits, uint16 f16 logits):
  - f0q4 vs f0lds:     Mean KLD 0.000000 +- 0.000000, max 0.000071, Same-top-p 99.975%.
  - f128q4 vs f128lds: Mean KLD -0.000000 +- 0.000000, max 0.000059, Same-top-p 100.000%.
  The max values are the uint16 logit-quantization floor of the harness (taskJ2 seal class),
  i.e. the kernels are numerically identical to their generic paths end-to-end, as the mb
  bit bars predict. Quality-neutral: CONFIRMED.
* Gate witnesses: every gate-on leg logs "GGML_CUDA_RP_Q4_NC_LDS=8 ACTIVE" (*.gate.witness).

## 5. End-to-end A/B (27B Q4_0 TP2 DFlash2 d4 + d7)

Natural-suite tg256 medians (bench.py 7 reps, interleaved chain, one rep/arm so far),
DFlash2-Q4_0 drafter, production bundle on every leg:

| arm      | env delta vs f0lds/f128lds        | d4 tg  | accept | len    |
|----------|-----------------------------------|--------|--------|--------|
| f0lds    | (control, Q4 gate OFF)            | 54.98  | 0.3956 | 2.5822 |
| f0q4     | + RP_Q4_NC_LDS=8 (u4 LDS arm)     | 56.04  | 0.3956 | 2.5822 |
| f128lds  | (control, floor-ON, Q4 gate OFF)  | 49.09  | 0.3872 | 2.5489 |
| f128q4   | + RP_Q4_NC_LDS=8 (affine LDS arm) | 54.30  | 0.3872 | 2.5489 |

Paired deltas (same-session):
* f0q4 vs f0lds:  +1.9% tg, acceptance/len/probe IDENTICAL (bit-identical kernels).
* f128q4 vs f128lds: +10.6% tg, acceptance/len IDENTICAL, probe byte-identical.
* f128q4 vs f0lds: -1.2% tg WITHOUT floor-0 - i.e. the Q4 LDS affine arm recovers nearly
  all of floor-0's verify-width win at ZERO quality cost (floor-0's narrow KL 0.864 /
  acceptance -2.8% class is not paid: f128q4 is bit-identical to the floor-ON path).
pp4096 flat (538-556) all arms, as expected (gate is narrow-width only).

d7 (8-token verify), same protocol:

| arm      | d7 tg  | accept | len    |
|----------|--------|--------|--------|
| f0lds    | 34.04  | 0.2374 | 2.6615 |
| f0q4     | 44.30  | 0.2374 | 2.6615 |
| f128lds  | 30.08  | 0.2105 | 2.4737 |
| f128q4   | 30.77  | 0.2105 | 2.4737 |
| f128q4w4 | 33.96  | 0.2105 | 2.4737 |

* f0q4 vs f0lds at d7: +30.2% tg, acceptance/len IDENTICAL, probes byte-identical
  (sha16 77841cb6a251235d on all d7 legs). The u4 LDS arm reproduces the J2
  Q8_0-style verify-width win on the Q4_0 stack at width 8 (mb predicted +69..73% on
  the Q4_0 matvecs; composite lands +30% with the uncovered Q6_K lm_head + attention).
* f128q4 (W=8) vs f128lds at d7: only +2.3% - EXACTLY as the microbench predicts
  (affine T=8 W=8 flat). With W=4 (f128q4w4): +12.9% and 33.96 ~= f0lds-d7 34.04 -
  the AFFINE arm at W=4 MATCHES floor-0's d7 number with ZERO quality cost, no floor-0.
  Wave count is per-width: W=8 optimal for u4@T8, W=4 for affine@T8 (mb); a per-width
  W default inside the gate is a cheap follow-on, deliberately not baked in here.
* f0q4-d7 (44.30) remains the best cell overall: floor-0 + u4 LDS stacks +30% on f0lds.

DFLASH-PROF step-phase medians (tgt-decode verify width, us->ms; draft-fwd unchanged
on every arm - the gate never touches the M=1 draft path):

| leg pair (d4)          | tgt w5: base -> gate-on | delta |
|------------------------|-------------------------|-------|
| f0lds -> f0q4          | 33.0 -> 32.5 ms         | -1.5% |
| f128lds -> f128q4      | 40.5 -> 35.3 ms         | -12.8% |
| leg pair (d7)          | tgt w8: base -> gate-on |       |
| f0lds -> f0q4          | 58.6 -> 42.3 ms         | -27.8% |
| f128lds -> f128q4      | 60.1 -> 58.1 ms         | -3.3%  (W=8) |

## 6. Verdict

SUCCESS BAR MET, both readings:

1. Quality-neutral tg at f0lds-class: f128q4-d4 = 54.30 vs f0lds 54.98 (-1.2%) and
   f128q4w4-d7 = 33.96 vs f0lds-d7 34.04 (-0.2%) - floor-0-class verify-width speed with
   ZERO quality change (bit-identical to the floor-ON generic path: KLD 0.000000,
   acceptance digit-identical, probes byte-identical). floor-0's KL 0.864 / acceptance
   -2.8% cost is NOT paid by these arms.
2. Clearly beats plain-LDS without floor-0: +10.6% (d4, W=8) and +12.9% (d7, W=4) over
   f128lds, quality-neutral by construction.
3. Composition is real: f0q4 (floor-0 + u4 LDS) = 56.04 d4 / 44.30 d7 - the u4 LDS arm
   stacks +1.9%/+30.2% on top of f0lds, bit-identical to f0lds outputs. New best
   Q4_0-stack numbers on this branch: 56.04 (d4), 44.30 (d7).

Recommended deployment knobs (per verify width, both arms env-gated, default OFF):
* d4 (width 5): GGML_CUDA_RP_Q4_NC_LDS=8 (affine +23..+26% mb; u4 flat, still bit-safe).
* d7 (width 8): floor-ON serving -> =4 (affine +25..+35%); floor-0 serving -> =8 (u4 +68..+73%).
  A single value cannot be optimal for both arms at width 8; per-width wave selection is a
  one-line follow-on if both arms must share one env value.

Honest limits: covers 83.1% of verify-width weight bytes (Q4_0 only); Q6_K lm_head
(6.7% of bytes but the largest single matvec) and Q4_1 ffn_down are untouched - the
composite +30% at d7 already prices that in. Single-rep e2e cells (7 internal bench reps
each); control reproduced taskN's f0lds within noise (54.98 vs 55.31/55.22). Gate default
OFF; nothing merged; branch int4-q-q4lds awaits parent review.
