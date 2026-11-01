# DOT8=4 correctness seal -- VERDICT: FAIL

Date: 2026-10-26 - Box: mi50 - GPU0 only (HIP_VISIBLE_DEVICES=0; GPU1 h2h never touched, GPU2 never used)
Build: rig/build-p0f (llama-perplexity target built during this session -- was missing)
Model: bench-quants/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf (dense, 8x Q4_1 ffn_down blk.0-7)
Env per leg: rig/w4a4-env.sh + GGML_CUDA_W4A4_PROFILE=speed + fork gate bundle
(GDN_COLTHREAD=1 GCN_KQ_MOE=1 REPACK_KQUANT=0 MOE_VEC=1 MOE_FUSE=1 MOE_DEC_MAX_TOKENS=8 FA_DENSE_GCN=1),
DOT8 set per arm; gate banner "DOT8=3/4 (explicit)" confirmed in every server/bench log.
flock rig/gpu.lock per leg; servers torn down with kill -TERM on the captured server PID
(pgrep -f 'llama-server -m .*--port <p>'), GPU0 verified ~10.8 MB idle at every swap and at end.

## Protocol (harness: benchgate, local workstation workspace/endpoint-harness)

Critical finding during setup: llama-server greedy output is NOT byte-stable across the
first-sight/warm boundary (cold prefill vs prefix-cache/token-graph replay state). The first
baseline attempt (plain suite, reps against one long-lived server) went RED on the DOT8=3 arm
against itself for this reason -- protocol flaw, fixed, old baseline kept as
seal-dot84.json.superseded-cold-vs-warm.

Fixed protocol: 23 rep suites x 17 cases; every prompt carries a unique per-rep cache-bust
prefix "[eval rNN]" and short tasks are context-padded to >=128-token prefills so every leg
exercises the DOT8 gate (MIN_TOKENS=128 under PROFILE=speed) cold on BOTH arms.
Arm A = 23x `promote gen` (DOT8=3 baselines, 391 legs). Arm B = 23x `gen` (391 legs) against
those baselines on a fresh DOT8=4 server. A-vs-A control (fresh DOT8=3 server, rep r01):
GREEN 17/17 -- protocol sound, divergence is attributable to DOT8.

## (a) Greedy seal suite -- FAIL (96/391 = 24.6% byte-identical; requirement >= 99%)

Per-case mismatch count (of 23 reps): p01 16, p02 20, p03 21, p04 22, p05 20, p06 15, p07 16,
p08 22, p09 20, p10 20, p12 12, p13 22, p14 22, p16 22, p17 21, pp512 0, pp840 4.
Long-prefill prompt-echo cases (pp512/pp840, 32-token robust continuations) are mostly stable;
the 64-token reasoning-shaped cases diverge in 52-96% of reps.

Exact first-divergence token positions (cold-vs-cold, rep r01, generated-token index incl.
model prompt echo): p01-capital token 4 (A "1]\n\nParis." vs B "2] Reference..."),
p03-tensor-def token 42, p04-secs-conv token 48. See raw/token-divergence-exact.json.

## (b) KLD (DOT8=3 ref vs DOT8=4 cand, frozen kld-prompt, topk 64 via /v1/completions) -- RED

- Prefill row KL = 0.190285 nats -- within the 0.5 ceiling, but clearly non-zero:
 DOT8=4 measurably changes prefill logits.
- Token divergence at row 1: ref=' the' cand=' in' -- harness rule: divergence is a finding,
 not retryable; decode rows not evaluated against the 0.05 ceiling.
- Sequential capture driver (dot84_kld_sequential.py) reuses benchgate client + kld math;
 needed because two 16 GB servers cannot co-reside on one 32 GB GPU0. No harness code edited.

## (c) Tensor-def hash comparison -- NOT SUPPORTED

benchgate has no tensor-level divergence feature; skipped per task ("if the harness supports
it"). The KLD prefill-row KL is the closest available numeric-divergence signal.

## (d) PPL pair (llama-perplexity, wikitext-2 wiki.test.raw, -c 512 --chunks 32) -- FAIL

- DOT8=3: PPL = 5.7326 +/- 0.15104
- DOT8=4: PPL = 5.7852 +/- 0.15254
- Relative change: +0.92% (> 0.5% fail threshold)
- Determinism control: DOT8=3 repeat run reproduced 5.7326 exactly -- the delta is DOT8, not noise.

## Verdict table

| Leg | Result | Number | Gate |
|---|---|---|---|
| Greedy seal 391 legs | FAIL | 96/391 = 24.6% identical | >= 99% |
| KLD prefill | pass-on-KL | 0.1903 nats | <= 0.5 |
| KLD decode | RED | token divergence row 1 | identity + <= 0.05 |
| Tensor-def hash | n/a | harness unsupported | -- |
| PPL pair | FAIL | 5.7326 -> 5.7852 (+0.92%) | < 0.2% pass / > 0.5% fail |
| A-vs-A control | GREEN | 17/17 | protocol soundness |

## VERDICT: FAIL -- do NOT adopt DOT8=4 in the F config on the strength of the +23.7% pp gain.

DOT8=4 changes numerics on the dense Q4_0/Q4_1 bulk path (non-zero prefill KL, +0.92% PPL,
75% greedy divergence on reasoning-shaped outputs). The earlier 32-token spot-check passed
only because prompt-echo-style continuations are robust to the perturbation; pp512/pp840
stability here is the same robustness effect, not evidence of unchanged numerics.
Owner decision needed: the +22-24% prefill gain now has a measured accuracy cost
(+0.92% PPL) and breaks the byte-identity seal.

Artifacts: this file; raw/ (server logs on box, per-rep baselines, kld rows, ppl logs,
divergence reports); harness store (local workstation) holds all 46 gen/promote run records
with profile+suite SHA-256.
