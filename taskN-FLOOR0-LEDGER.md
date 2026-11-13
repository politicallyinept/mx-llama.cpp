# taskN LEDGER — floor-0 forced-W4A4 adoption review (branch int4-n-floor0)

Date: 2026-11-13. Branch `int4-n-floor0` off INT4-noquant @ 93818795f7 (NOT pushed; parent merges).
Code commit: 5fb06b0c81 (GGML_CUDA_DRAFT_REPACK gate, default OFF).
Review build: /root/build-jmerge (src a63c468d60 = tree tip, BUILD-PROVENANCE verified, dirty=0;
J1 v_dot8 + J2 LDS kernels present, both default-OFF). Drafter-scoping build: /root/build-n-floor0
(tree /root/n-floor0-wt @ 5fb06b0c81, mi50-build.sh clean).
Stack: 27B Qwen3.8 Q4_0 TP2 + DFlash2-Q4_0 drafter, d4, -c 8192 -ub 1024, production bundle
enforced by mi50-run.sh on every leg. Data: /root/taskN/. Drivers: taskN-drive/kld/kld1/s2/drv.sh.

## What is reviewed

F5 (2026-11-11): GGML_CUDA_W4A4_MIN_TOKENS=0 on the sealed W4A4 repack stack = runtime int4
activations at ALL widths (floor normally falls back to stock q8_1 below width 128).
Historical claim (RESULTS-microbench-addendum.md F5 entry): +20-24% spec tg at -3% acceptance;
quality record: narrow prefill KL vs Q8 ref 0.684->1.357, floor-ON vs floor-0 direct KL 0.864,
anchor KLD unchanged 0.0656 (wide ladder blind), era-1 runtime KLD 0.0756 vs 0.0499 (+51%).

## Arms (env-only deltas on the sealed default; production bundle constant on every arm)

| arm | env delta | meaning |
|---|---|---|
| f128    | (none) | sealed default baseline (speed profile DOT8=3 MIN_TOKENS=128 KSPLIT=1 SUMW=1) |
| f0      | MIN_TOKENS=0 | F5 forced 4-bit activations |
| f128lds | REPACK_Q8_0 pair + RP_Q8_NC_LDS=8 | the 50.44-config env pair on the Q4_0 stack (control) |
| f0lds   | MIN_TOKENS=0 + same pair | floor-0 stacked with the LDS pair |
| dr0/dr1f/dr1 | GGML_CUDA_DRAFT_REPACK gate (n-floor0 build) | drafter-scoping change (section 4) |

2 interleaved reps/arm (rep2 reverse order), natural-suite tg256 + acceptance, greedy temp0
seed42 probe per leg, gate witnesses per leg (*.gate.witness: floor ACTIVE [explicit] vs
[PROFILE=speed default], repack banners). BINARY AUDIT: every leg proc.witness shows
build-jmerge (dr*: build-n-floor0); the build-ship-20261112 server seen mid-campaign was
llama-swap's PRODUCTION Huihui entry (ppid=llama-swap, port 10005, ttl 3600, active traffic,
ROCm1/2) - not a leg leftover; left untouched per contract. It forced the KLD ladder to a
TP1 shape on the free GPU0 (see section 2 note) and blocked dr1f at campaign end.

## 1. Perf A/B (tg256 medians, 27B Q4_0 TP2 d4, natural suite)

| arm | r1 | r2 | median | accept_rate | mean_accept_len |
|---|---|---|---|---|---|
| f128    | 38.64 | 40.71 | 39.68 | 0.3489 | 2.3955 |
| f0      | 46.28 | 46.02 | 46.15 | 0.3320 | 2.3280 |
| f128lds | 49.16 | 48.01 | 48.59 | 0.3872 | 2.5489 |
| f0lds   | 55.31 | 55.22 | 55.27 | 0.3956 | 2.5822 |

Paired deltas (per-rep, interleaved):
- f0 vs f128:       +19.8% / +13.0% -> mean +16.3% tg; accept -4.8% rel; len 2.3955->2.3280 = -2.8%
- f0lds vs f128lds: +12.5% / +15.0% -> mean +13.7% tg; accept +2.2% rel; len +1.3%
- f0lds vs f128:    +39.3% total composition; pp4096 flat all arms (551-556).
- f0lds 55.27 EXCEEDS the Q8_0-stack best (50.44, taskL) by +9.6% - different main-quant class.

CLAIM RE-VERDICT: direction HOLDS, magnitude SOFTER: +13-20% (mean +16.3%) vs historical
+20-24%. The historical delta was measured against a Q8_0-drafter baseline (F1/F2 37.8-38.0);
the current f128 baseline already carries the shipped Q4_0 drafter (+5%), absorbing part of
the delta. Acceptance cost reproduces EXACTLY: len -2.8% vs historical -3% (2.396->2.328).

Greedy probe (64-token temp0 seed42): ALL arms byte-identical (sha 77841cb6a251235d) - the
probe prompt is margin-robust; floor-0's greedy change is prompt-dependent (section 3).

## 2. KLD re-score, taskI1 protocol (27B Q4_0, wikitext 16ch ctx2048, base = floor-ON f16-KV logits)

Shape note: TP1 on GPU0 (base + all arms same shape) because the llama-swap production server
held ROCm1/2 with live traffic for the whole KLD window; the floor-ON-vs-candidate differential
is internally valid, absolute anchor comparability carries the TP1 note.

| arm | Mean KLD vs floor-ON base | Max KLD |
|---|---|---|
| f128  | 0.000000 +/- 0.000000 | 0.000066 (uint16 logprob floor) |
| f0    | 0.000000 +/- 0.000000 | 0.000066 |
| f0lds | 0.000000 +/- 0.000000 | 0.000066 |

Re-verification of F5 finding #1 on the current build: the anchor ladder is STRUCTURALLY BLIND
to floor-0 - at 2048-token chunk width both arms run the same wide u4 path, KLD delta exactly
zero. f0lds also 0.000000: on the Q4_0 stack the repack pair is bit-identical at wide width
(its divergence lives at narrow/verify widths - exactly where acceptance moved, cf. taskK
0.0040 on the Q8_0-main stack where most tensors are Q8_0).
The historical 0.0756-vs-0.0499 era-1 runtime KLD is a NARROW-width protocol number; it is
NOT reproducible on the wide ladder by construction - the narrow witness is section 3.

## 3. Narrow-width endpoint KLD (S2 re-verify: floor-ON ref GPU0:8620 vs floor-0 cand GPU1:8621)

Harness run 20261113T163644Z-c73c: prefill row KL 0.86405 nats, greedy divergence at row 1
(ref ':' vs cand ' the'). REPRODUCES the historical S2 (0.86405, same divergence row and
tokens) DIGIT-EXACT on the current build. RED by design (quant-class change, same class as
q4_0-KV-era seals across quant boundaries); this is the quality-cost witness, not a seal failure.

## 4. Drafter scoping (taskL no-op resolution)

Root cause: common/speculative.cpp loads sidecar drafts with use_extra_bufts=false ("the
tensor-parallel splitter cannot mix repacked and canonical layouts in one group"), so draft
q4_0 tensors never enter repack admission and MIN_TOKENS=0 is a process-global no-op for the
draft forward (taskL witness).
IMPLEMENTED (5fb06b0c81, default OFF): GGML_CUDA_DRAFT_REPACK=1 admits the draft to repack
admission, with witness banner. Gates: dr0 (gate OFF) byte-identical to f128 (probe sha +
accept 0.3489/len 2.3955 identical) - default-off byte-identity PASS. dr1 load witness:
"draft repack admission ON" + draft tensors admitted (3D attn_conv_base kept canonical per
the existing dense-repack guard); TP2 tensor-split load survives (mixed-layout concern not
triggered when the target is also repack-admitted).
RESULT: dr1 (DRAFT_REPACK=1 + MIN_TOKENS=0) = 43.42 t/s vs f0 46.15 = -5.9%; accept
0.329/len 2.3158 vs 0.332/2.328 (-0.9%). The scoped drafter u4 path is a NET LOSS - confirms
the F5-era phase attribution ("the drafter-only variant would win ~nothing - the lever is the
TARGET's verify batches"): canonical q4_0 mmvq already beats repacked-u4 at draft width.
dr1f (repack ON, floor 128 - layout-vs-u4act attribution) BLOCKED by the production server at
campaign end; chain3 (taskN-chain3.sh) runs it if ROCm1/2 free up. Attribution does not
change the verdict: the arm is net-negative as configured.
VERDICT on drafter scoping: code change SHIPPED default-OFF as a research knob; do NOT enable.

## 5. Bundle interaction

The 27B dense stack's equivalent bundle IS production-env.sh (csAR trio, RPB1=2, SKIP_XBAR,
MOE_VEC/FUSE, GDN_COLTHREAD, KMAX=8); mi50-run.sh hard-sources it on EVERY arm, so the
floor-0 deltas above are bundle-inclusive by construction. MOE_VEC/MOE_FUSE (the taskM 35B
movers, +13%) are MoE-path and inert on the dense 27B; the dense-relevant subset (csAR/RPB1)
touches allreduce/graph plumbing, orthogonal to the u4 GEMM path, and is constant across arms
- no separable bundle x floor-0 interaction term. A bundle-OFF control arm was NOT run:
launching outside the production bundle violates the ops contract.

## 6. VERDICT: MODIFY-THEN-ADOPT (per-entry opt-in; NOT a fleet default)

Numbers, plainly:
- Throughput: floor-0 = +16.3% tg mean (+13-20% paired) on the current merged build,
  acceptance -2.8% len (reproduces the historical -3%); pp flat. Stacked with the LDS pair
  (f0lds) it is the new best standard-quant tg on this box: 55.27 (+39.3% over sealed default,
  +13.7% over the LDS pair alone).
- Quality cost: ZERO at wide width (anchor KLD 0.000000 vs floor-ON, re-verified); at the
  served narrow operating point the KL to the floored baseline is 0.864 nats prefill and the
  greedy first-decode token flips on at least some prompts (S2 row 1; reproduced digit-exact).
  The honest quality label is the era-1 u4act class: ~+51% runtime KLD (0.0756 vs 0.0499).
  The 64-token greedy probe did NOT diverge on our probe prompt - the cost is prompt/margin
  dependent, not universal.
- Composition: floor-0 composes with the LDS pair on the Q4_0 stack (+13.7% on top). It is a
  NO-OP on the Q8_0-main 50.44 config (taskL: no q4_0 main tensors, drafter not admitted) -
  floor-0 only makes sense on repack-admitted q4_0 mains.
- Drafter scoping: implemented + gated + byte-identity-proven; measured -5.9% -> ship the
  knob default-OFF, do not enable.

Recommendation: adopt floor-0 as a DOCUMENTED PER-ENTRY OPT-IN env flag (reversible per fleet
entry, as the F5 revisit note anticipated) for spec-decode-heavy Q4_0 entries where +16-39%
tg justifies the u4act narrow-width quality class; do NOT flip the sealed default - entries
whose acceptance bar is greedy-identity with current served output must stay floored (S2 shows
first-token divergence across the boundary). Pair with REPACK_Q8_0 pair + LDS=8 for the full
+39% composition. Owner's Pareto call reduces to: is ~0.86 nats narrow-width KL (below the
sibling W4A4 engine's own 1.455) acceptable for the tg gain - the numbers for that call are
sections 1-3, reproduced on the current merged build.

## Ops notes
- flock /root/gpu.lock -w 900 every leg; TERM-only teardown; pgrep -x llama-server empty after
  each leg (one WARN at dr1 teardown = the llama-swap production server starting, not ours).
- Production env witnessed per leg (proc.witness environ + TheRock libs). llama-swap untouched.
- KLD driver bugs fixed mid-campaign (arm_env base case; llama-perplexity exits 0 on missing
  base logits - now validated by file/line checks; TP2->TP1 fallback for the blocked window).
