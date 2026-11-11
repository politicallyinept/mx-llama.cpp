
## taskF KLD ladder session 2026-11-11 (gpu-guard owner taskF:kld-ladder)

Protocol notes:
- Two ladders in one session: (A) endpoint KLD via endpoint-harness `kld` (frozen
  kld-prompt.txt, topk 64, prefill row + 16 decode rows, temp0 seed42 top_k1,
  api=completions), ref = Q8_0 on fork engine ROCm0 port 8620, candidates on
  ROCm1 port 8621; (B) anchor-protocol ladder: llama-perplexity 16ch ctx2048 TP2
  scored against the existing anchor base /root/ruquant-bench/p2logs/q8_0-16.kld
  (same base that produced the 0.0499/0.0653 anchors).
- Caveat: endpoint KLD saturates across quantization boundaries (greedy seal
  breaks at row 1 for any Q4-vs-Q8 pair; post-divergence per-row KL ~30 nats is
  trajectory artifact, not quality). Ladder A is only meaningful as an
  L1-vs-L2 differential; ladder B is the anchor-comparable ladder.
- Driver bug: wait_vram checked both cards while ref legitimately held GPU0 ->
  300 s WARN per wait point (~20 min wasted, no correctness impact).

### L1 fork floor-ON (sealed default)
- server: /root/int4-noquant-wt/build/bin llama-server, Q4_0 + DFlash2-Q4_0 drafter,
  --spec-type draft-dflash --spec-draft-n-max 4, env w4a4-env + ALLREDUCE=internal
  REPACK_KQUANT=0 GDN_COLTHREAD=1, MIN_TOKENS unset.
- witness (log): "W4A4_PROFILE=speed (unset -> default) resolved: DOT8=3 MIN_TOKENS=128
  KSPLIT=1 SUMW=1" + "MIN_TOKENS=128 floor ACTIVE".
- harness kld: run 20261111T204203Z-6a36 RED - prefill KL 0.68420, token divergence
  at row 1 (ref ' in' vs cand ':').
- raw 17-row: prefill 0.684202; decode row1 KL 0.640016 (pre-divergence);
  post-divergence rows saturate ~25-30 nats (expected artifact); tokens_identical 1/17.

### L2 fork floor-0 (F5 forced 4-bit activations) - THE number
- same as L1 + GGML_CUDA_W4A4_MIN_TOKENS=0.
- witness (log): "resolved: DOT8=3 MIN_TOKENS=0 (explicit) KSPLIT=1 SUMW=1" +
  "DOT8=3 gate ACTIVE ... MIN_TOKENS=0 [explicit]".
- harness kld: run 20261111T205639Z-dae5 RED - prefill KL 1.35724, token divergence
  at row 1 (ref ' in' vs cand ' the').
- raw 17-row: prefill 1.357237; decode row1 1.39752 (pre-divergence); post-divergence
  saturated (trajectory artifact). tokens_identical 1/17.
- differential vs L1 on the SAME reference: prefill KL 0.684 -> 1.357 (+0.673 nats, ~2x).
  The frozen prompt is ~55 tokens, below the 128 floor, so L1 prefill ran stock q8_1
  while L2 prefill ran u4act: this differential IS the forced-W4A4 cost at narrow width.

### L3 rq engine (W4A4 + DFlash2 d7)
- server: /root/ruquant-src/build/bin llama-server, W4A4-OST-GPTQ-wg64-rb4 +
  DFlash2-W4A4H-drafter-rb4, --spec-type draft-dflash --spec-draft-n-max 7
  -bs --samplers temperature, env TheRock-first + HSA fine-grain + META_TOKEN_GRAPH=0
  META_SPIN=2000000 DISABLE_GRAPHS=1 (their recipe).
- harness kld: run 20261111T211024Z-0f08 RED - prefill KL 1.45523; greedy seal held
  rows 0-10 (11 identical tokens), divergence at row 11.
- raw 17-row: prefill 1.455229; decode pre-divergence (10 rows): mean 0.0735,
  max 0.2855, 4/10 over 0.05 ceiling; tokens_identical 11/17.
- caveat: ref server is the fork engine, so L3 carries a cross-engine numerics
  component that L1/L2 do not.

### Anchor-protocol perplexity ladder (16ch ctx2048 TP2 vs q8_0-16.kld, same base as anchors)
- PPL-L3 (rq W4A4): Mean KLD 0.065291 +/- 0.004052 - BIT-EXACT reproduction of the
  0.0653 anchor (kld-w4a4.log: 0.065291 +/- 0.004052). Protocol calibration PROVEN.
- PPL-L1 (fork sealed, MIN_TOKENS=128 default): Mean KLD 0.065577 +/- 0.003905.
  Witnesses: REPACK_Q4_0 gate ACTIVE + "q8_repack: repacking q4_0 tensors on device 0".
- PPL-L2 (fork MIN_TOKENS=0 explicit, witnessed): Mean KLD 0.065577 +/- 0.003905 -
  BYTE-IDENTICAL to PPL-L1. At 2048-token chunk width both arms are above the 128
  floor, so the wide-width ladder is structurally BLIND to the F5 delta. F5's cost
  lives at narrow width only (see endpoint ladder).
- ANCHOR PROVENANCE FINDING: the 0.0499 anchor (kld-q4_0.log 0.049850) is plain Q4_0
  scored through the RUQUANT engine with no fork gates - NOT a fork-engine sealed
  measurement. The era-1 table's "fork sealed = 0.0499" label is a carryover.
  The fork engine's sealed W4A4 repack path actually measures 0.0656 on this protocol.
- Controls added (separate window taskF:ppl-controls):
  A0  = plain Q4_0 via rq engine (anchor repro, expect 0.049850)
  L1n = fork engine Q4_0 with GGML_CUDA_REPACK_Q4_0=0 (isolates repack-path cost)

### Controls (window taskF:ppl-controls)
- A0  plain Q4_0 via rq engine: Mean KLD 0.049850 +/- 0.003539 - BIT-EXACT anchor
  reproduction (kld-q4_0.log: 0.049850 +/- 0.003539). Protocol calibration doubled.
- L1n fork engine Q4_0 REPACK_Q4_0=0: Mean KLD 0.048993 +/- 0.003538 - fork engine
  without the u4 repack path matches the anchor class (0.0490 vs 0.0499, within noise).
- Decomposition at wide width (2048-token chunks):
    rq-engine plain Q4_0        0.0499  (anchor)
    fork-engine, repack OFF     0.0490  (L1n)
    fork-engine, sealed repack  0.0656  (L1)   <- fork u4 repack/GEMM costs +0.0166
    fork-engine, MIN_TOKENS=0   0.0656  (L2)   <- F5 adds ZERO at wide width
    rq W4A4 format              0.0653  (L3, anchor repro)

### S2 direct leg (window taskF:s2-direct-retry): KL(floor-ON || floor-0)
- ref = fork floor-ON server (ROCm0:8620, witness MIN_TOKENS=128 default),
  cand = fork floor-0 server (ROCm1:8621, witness MIN_TOKENS=0 explicit).
  Same engine, same Q4_0 weights, same DFlash2-Q4_0 drafter, same flags.
- harness kld: run 20261111T212656Z-6fc7 RED - prefill KL 0.86405, token divergence
  at row 1 (ref ':' vs cand ' the'). F5 changes the greedy first-decode token.
- raw: prefill 0.864051; row1 0.675094 (pre-div); tokens_identical 1/17.
- first s2 attempt failed on a script bug (health probe arg order), retried once.
