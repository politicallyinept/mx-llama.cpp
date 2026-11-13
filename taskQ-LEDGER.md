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

## 3. Microbench (mb_q4lds_test.cu, one MI50 under flock)

Bars ALL PASS at every shape: rp_q4_nc_lds (W=4, W=8) == generic
`mul_mat_vec_repacked_nc<2,1,T,2,64,Q4_0,...>` BIT-EXACT for BOTH decode arms
(affine + U4ACT/SUMW), affine pair == CPU oracle (oracle_bad=0), launch-deterministic.
dx/sx staged as DISTINCT random halves so a swap bug could not pass.

us/call medians (7x30 iters; first run under build CPU load, clean re-run in flight):

| shape (K x N)        | arm  | T=4            | T=5            | T=8            |
|----------------------|------|----------------|----------------|----------------|
| 5120x6144 (ssm/gate) | aff  | +24.2 (W8) / +32.5 (W4) | +26.0 / +4.6 | +7.4 / +30.1 |
| 5120x17408 (ffn)     | aff  | +12.3 / +13.5  | -              | +2.1 / +26.9 |
| 5120x10240 (qkv)     | aff  | +17.5 / +16.1  | -              | +5.8 / +25.9 |
| 17408x5120 (ffn_down)| aff  | +16.0 / +15.0  | -              | +6.4 / +34.3 |
| 5120x6144            | u4   | -0.2 / +1.0    | TBD            | +72.6 / +52.2 |
| 5120x17408           | u4   | -9.7 / -10.6   | TBD            | +73.4 / +51.2 |
| 5120x10240           | u4   | -              | TBD            | +71.4 / +48.7 |
| 17408x5120           | u4   | -              | TBD            | +68.8 / +51.5 |

Reading: the AFFINE arm wins broadly at T=4-5 (+12..+34%) and splits at T=8 (W=8 nearly
flat, W=4 +26..+34% - LDS pressure at T=8 favors the smaller workgroup). The U4ACT arm is
flat-to-negative at T=4 but +69..+73% at T=8 (W=8) - the dp8a decode is more compute-dense
so the LDS win only lands once 8 activation streams share the stage; T=5 (the d4 verify
width) is the open question the added cells answer. W=4 is the safer affine width at T=8,
W=8 the safer u4 width; the e2e A/B picks (gate accepts 2/4/8/16).
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
  (taskN) to the digit, probe byte-identical. KLD narrow-window legs (taskJ2 -b 8 protocol,
  pair-isolated vs same-env base logits) in flight - expect the 0.000000/6.6e-05 uint16 floor.
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

d7 (8-token verify) legs in flight - the u4 LDS arm measured +69..+73% at T=8 in the
microbench, so f0q4-d7 is the headline cell.

## 6. Verdict

TBD.
