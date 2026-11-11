# KLD ladder: quality cost of forced 4-bit activations (F5)

Date: 2026-11-11. Same-session ladder, one gpu-guard window per phase, reference
held up throughout the endpoint phase. Two complementary ladders were run against
the SAME reference (local Q8_0 GGUF of the served 27B dense target):

- Ladder A (endpoint KLD, endpoint-harness `kld`): live servers, frozen prompt
  (~55 tokens), topk 64, prefill row + 16 decode rows, temp 0 / seed 42 / top_k 1,
  api=completions. Reference: Q8_0 on the fork engine, ROCm0 port 8620.
  Candidates on ROCm1 port 8621.
- Ladder B (anchor protocol): llama-perplexity, 16 chunks, ctx 2048, TP2, scored
  against the existing anchor base q8_0-16.kld -- the same base that produced the
  2026-11-05 anchors (0.0499 / 0.0653 / 0.0185).

## Ladder A: endpoint KLD vs Q8_0 reference (harness verdicts all RED by design)

Any Q4-class candidate vs a Q8 reference breaks the greedy seal within the first
rows; post-divergence per-row KL (~30 nats) is trajectory artifact, not quality.
The quotable numbers are the prefill row and the pre-divergence decode rows.

| leg | config | prefill KL (nats) | decode rows before divergence | decode mean/max (pre-div) | verdict |
|---|---|---|---|---|---|
| L1 | fork floor-ON (sealed default, MIN_TOKENS=128) | 0.6842 | 1 (diverged row 1) | 0.640 / 0.640 | RED (seal row 1) |
| L2 | fork floor-0 (F5 forced 4-bit activations)    | 1.3572 | 1 (diverged row 1) | 1.398 / 1.398 | RED (seal row 1) |
| L3 | sibling-engine W4A4 + DFlash2 d7              | 1.4552 | 11 (diverged row 11) | 0.074 / 0.285 (4/10 over 0.05) | RED (seal row 11) |

Harness run ids: L1 20261111T204203Z-6a36, L2 20261111T205639Z-dae5,
L3 20261111T211024Z-0f08.

L1 vs L2 is the F5 differential on the served model: same engine, same weights,
same drafter, only GGML_CUDA_W4A4_MIN_TOKENS differs (128 default vs 0). The
frozen prompt is below the 128 floor, so L1 ran stock q8_1 accumulation at these
widths while L2 ran the u4 activation path: prefill KL doubles, 0.684 -> 1.357.

Supplementary direct leg S2 (floor-ON server as reference, floor-0 server as
candidate; same engine, same weights, same drafter, only the env differs):
prefill KL 0.8641 nats, greedy diverges at row 1 (run 20261111T212656Z-6fc7).
F5 does not just shift the distribution at narrow width, it changes the served
model's greedy first-decode token.

## Ladder B: anchor-protocol mean KLD vs Q8_0 base (comparable to 0.0499/0.0653)

| arm | engine / config | mean KLD | +/- |
|---|---|---|---|
| A0  | plain Q4 weights, sibling engine (anchor repro) | 0.049850 | 0.003539 |
| L1n | fork engine, repack gate OFF (control)          | 0.048993 | 0.003538 |
| L1  | fork engine, sealed W4A4 repack (floor 128)     | 0.065577 | 0.003905 |
| L2  | fork engine, MIN_TOKENS=0 (F5)                  | 0.065577 | 0.003905 |
| L3  | sibling engine W4A4 format (anchor repro)       | 0.065291 | 0.004052 |

Calibration: A0 reproduces the 0.0499 anchor bit-exactly (0.049850) and L3
reproduces the 0.0653 anchor bit-exactly (0.065291). The protocol is the same
protocol; the numbers land beside the anchors.

Provenance finding: the 0.0499 anchor is plain Q4 weights scored through the
sibling engine with no fork gates. The era-1 table's "fork sealed = 0.0499"
label is a carryover, not a fork-engine measurement. The fork engine's sealed
repack path actually costs 0.0656 at wide width (L1n -> L1: +0.0166 attributable
to the fork u4 repack/GEMM path).

## Decomposition

Wide width (2048-token chunks, ladder B):
  plain Q4 (either engine, repack off)  ~0.049-0.050
  fork sealed repack                    0.0656   (+0.017 vs repack-off)
  fork F5 (MIN_TOKENS=0)                0.0656   (byte-identical to L1)
  sibling W4A4 format                   0.0653

Narrow width (endpoint ladder A, T=1 decode and sub-floor prefill):
  fork floor-ON   prefill KL 0.684 vs Q8 ref
  fork floor-0    prefill KL 1.357 vs Q8 ref   (F5 doubles the divergence)
  sibling W4A4    prefill KL 1.455 vs Q8 ref

## Verdict

1. F5 (GGML_CUDA_W4A4_MIN_TOKENS=0) is FREE at wide width: ladder B is
   byte-identical with and without it, because the 2048-token chunks sit above
   the 128 floor in both arms. The anchor ladder is structurally blind to F5.
2. F5 is NOT free at narrow width: on the endpoint ladder the divergence from
   the Q8 reference roughly doubles (prefill 0.684 -> 1.357 nats; first decode
   row 0.640 -> 1.398). That is the quality cost of forcing 4-bit activations
   at T=1 decode, bought for +20-24 percent spec tg at -3 percent acceptance.
3. Where it sits vs the sibling engine: at wide width the fork sealed config
   (0.0656) and the sibling W4A4 format (0.0653) are statistically tied; both
   pay roughly +0.016-0.017 over their own repack-off/plain baselines (~0.049).
   At narrow width, F5's endpoint divergence (1.357) is just below the sibling
   engine's (1.455) and both are roughly 2x the fork floor-ON arm (0.684).
4. Is F5 qualitatively worse than the sealed baseline? At the anchor protocol:
   no (identical). At the served narrow-width operating point: yes, measurably
   -- the KL to the Q8 reference doubles. Whether that is acceptable is a
   throughput/quality trade call: the delta is concentrated in decode-path
   accumulation error, it does not move the wide-width anchor number at all,
   and it stays in the same band as the sibling W4A4 engine's own cost.

## Protocol deviations and caveats

- The endpoint harness verdicts are RED for all three legs (greedy seal breaks
  across quantization boundaries); numbers recorded anyway per campaign rules.
- L3 carries a cross-engine component: the Q8 reference server ran the fork
  engine, so L3's endpoint KL includes engine numerics differences that L1/L2
  do not. Ladder B removes this by scoring each arm with its own engine against
  the shared Q8 logits base.
- Driver bug: the VRAM drain check required BOTH cards idle while the reference
  legitimately held GPU0, adding a 300 s warning wait per checkpoint (no
  correctness impact).
- The first s2 (direct floor-ON vs floor-0) launch failed on a script argument
  bug (health probe checked the wrong pid); retried once per campaign rules.
