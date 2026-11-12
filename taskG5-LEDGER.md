# taskG5 LEDGER - DFlash2 depth decomposition: n_max=4 vs n_max=7 on the 35B Q8_0 MoE (TP2)

Date: 2026-11-12 (PST). Box: mi50 (3x MI50 gfx906, LXC on PVE). No pushes; local commit only.
Build: /root/int4-noquant-wt/build/bin/llama-server (tree d496c12154-era, binary stamp
d67cd96f0f = nc 6-8 1-warp fix present; same binary as taskV2/taskG4 fork legs).
Stack: target Qwen_Qwen3.6-35B-A3B-Q8_0.gguf (35.2 GiB, TP2 GPU1+2) + drafter
/root/bench-quants/dflash/Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf (community ft, same base,
6 layers, dflash.block_size=8 -> n_max 7 is the trained full width, no clamp;
target_layers [2,7,12,17,23,28,33,38]).
Serving flags: w4a4-env + ALLREDUCE=internal REPACK_KQUANT=0 GDN_COLTHREAD=1,
-ngl 99 -fa on -sm tensor --fit off -c 16384 -ub 1024 -t 16 -np 1 -lm dio,
--spec-type draft-dflash --spec-draft-n-max {4,7}. Harness /root/p2-prefill/bench.py
(natural suite, tg256 median of 8 prompts x 3 reps, temp-0). Driver /root/taskG5/drive.sh,
interleaved d4,d7,d4,d7 then instrumented d4-p,d7-p (GGML_DFLASH_PROF=1
LLAMA_SERVER_TIMING=1 GGML_CUDA_GRAPH_STATS=1 --verbose). All legs flock -w 600
/root/gpu.lock, TERM-only teardown, TheRock runtime witnessed, no foreign servers,
all 6 legs OK first attempt, GPUs drained after every leg.

## 1. Headline (clean arms, median of r1/r2; per-leg spread <2.3%)

| arm | tg256 t/s | mean accept len | accept rate | step ms (len/t_s) |
|---|---|---|---|---|
| n_max=4 | 70.58 (70.72/70.44) | 2.492 | 0.373 | 35.31 |
| n_max=7 | 59.13 (58.47/59.79) | 2.654 | 0.236 | 44.88 |

Depth delta on this stack: -16.2% tg (-4% on the 27B Q4_0 stack of taskG4; the MoE Q8_0
stack punishes fixed depth harder). Step cost +9.57 ms (+27.1%) buys +0.162 tokens/step
(+6.5%): marginal cost 59.1 ms/token vs 14.2 ms/token base average = 4.2x overpriced.
Break-even at d7 would need mean len >= 3.16 (actual 2.65) or step <= 37.6 ms (actual 44.9).

## 2. Per-step decomposition (instrumented legs d4-p/d7-p; steady decode windows)

| phase | d4 ms/step | d7 ms/step | delta | share of delta |
|---|---|---|---|---|
| target verify fwd+sync (tgt-decode width 5 / 8) | 22.40 (p50 20.85) | 28.89 (p50 27.35) | +6.49 | 68% |
| drafter fwd (draft-fwd, width 5 / 8, one block-diffusion pass) | 8.94 | 10.34 | +1.40 | 15% |
| drafter inject (fused encode+inject, width 1+n_acc) | 1.05 | 1.05 | ~0 | 0% |
| drafter gather | 0.03 | 0.17 | +0.14 | 1% |
| verify/accept bookkeeping | 0.65 | 0.75 | +0.10 | 1% |
| host residual (batch prep, sampler, idle) | ~2.2 | ~3.3 | +1.1 | 12% |
| step total (clean-arm derived) | 35.31 | 44.88 | +9.57 | |

Cross-checks: tgt-decode width histogram matches LLAMA_SERVER_TIMING per-width buckets
(w5 22.40 vs 22.48; w8 28.89 vs 29.00). Phase sums + residual reproduce clean-arm step
time within 1.5 ms on both arms. Profiling overhead negligible (d4-p 69.59 vs clean 70.58;
d7-p 59.76 vs 59.13).

Verify kernel: width 5 = 22.4 ms vs plain 1-token decode ~9.2 ms (taskV2 stock tg) ->
verify at width 5 already costs 2.4x a single token on this MoE; width 8 costs 3.1x
(28.9 ms). Per verify token: 4.48 ms (w5) -> 3.61 ms (w8): sublinear but far from the
bandwidth-bound flat cost that makes depth pay. Drafter per draft token: 2.24 ms (n4)
-> 1.48 ms (n7) - the block-diffusion forward scales sublinearly, so the drafter is NOT
the depth problem; the verify pass is.

## 3. Acceptance curve (end-of-leg cumulative stats, natural suite)

- d4: #acc rate/pos = (0.647, 0.394, 0.298, 0.248), sum 1.587, mean len 2.59
- d7: #acc rate/pos = (0.635, 0.344, 0.237, 0.189, 0.153, 0.125, 0.107), sum 1.790, mean len 2.79
- Tail positions 5-7 yield +0.385 tokens/step; positions 1-4 LOSE 0.182 tokens/step at
  block width 8 vs 5 (block-width interference: the trained 8-wide block degrades early
  positions relative to the shorter 5-wide block).
- Tail acceptance at 5-7 is 0.15/0.13/0.11 on natural prose - the parent-cited ruquant
  0.45-0.37 tail figures come from their acc_bench mix (mean len 4.25); on the SAME
  natural suite ruquant pooled mean len is 2.62 (taskG4) vs our 2.65 = parity. Our
  drafter is not worse; natural prose tails just do not pay in either engine.

Per-prompt (r1 legs, tg256 median):

| prompt | d4 t/s | d7 t/s | delta% | d4 len | d7 len |
|---|---|---|---|---|---|
| code-discussion | 133.2 | 156.6 | +17.5 | 4.88 | 7.56 |
| tech-explain | 123.6 | 127.3 | +3.0 | 4.49 | 5.91 |
| narrative-scifi | 58.9 | 66.9 | +13.5 | 1.90 | 2.77 |
| dialogue | 49.9 | 44.3 | -11.2 | 1.60 | 1.81 |
| econ-reasoning | 71.0 | 53.0 | -25.3 | 2.26 | 2.15 |
| list-structured | 60.5 | 45.4 | -25.0 | 1.97 | 1.85 |
| narrative-mystery | 70.3 | 50.9 | -27.5 | 2.33 | 2.06 |
| math-reasoning | 115.4 | 74.5 | -35.5 | 4.01 | 3.18 |

Depth wins only where the chain runs near-saturated (code len 7.56, tech 5.91); on 5 of 8
prompts mean len at d7 is LOWER than at d4 - the interference signature.

## 4. Ruled out (measured, not assumed)

- Graph/orchestration: token graph healthy both arms (tg capture_ok 64/56, fail 0,
  invalidate 0, not-eligible 0, cuda-graph-unstable 0); owner-off per-subgraph counts
  identical shape class both arms (~54k, by design under meta token graph). No
  width-churn capture storm at depth 7.
- KV/memory: gather 0.03-0.17 ms/step; VRAM ~24 GB/GPU, no pressure; no -c effects.
- Scheduling/idle: host residual 2.2 vs 3.3 ms/step, not depth-structural; host submit
  inside decode 1.7-2.2 ms (d4) vs 1.6-6.1 (d7, prefill-interleaved windows only).
- Drafter throughput: +1.4 ms/step only (15% of delta).

## 5. Root cause

Cost-side depth scaling, not acceptance collapse. The verify pass on this Q8_0 MoE is
compute/expert-bound at decode widths (2.4x single-token cost already at width 5), so
widening 5->8 adds +6.5 ms/step (68% of the regression) while the acceptance yield is
+0.16 tokens/step - and that yield is itself suppressed by block-width interference
(width-8 block costs -0.18 tokens/step at positions 1-4). ruquant's engine scales with
depth because its width-8 step is cheap (their design point: W4A4 kernels + pinned
drafter, ~9.5 ms/token at len 4.25 on their mix); our engine pays 16.9 ms/token at
len 2.65. Same acceptance, different cost curve.

## 6. Fix candidates (ranked by expected tg recovery at fixed n_max=7 = 59.1 t/s baseline)

1. Adaptive draft depth (per-request sticky n_max, or oracle per-prompt-class): mixed
   natural traffic recovers to >= d4 level (+16-19%); code/tech keep the depth win
   (+17.5%/+3%). Cheapest lever; p_min chain truncation exists but does not fix the
   interference (block still 8 wide); a length controller must switch n_max itself
   (sticky, per mtp-natural controller notes: graph rebuild ~40 ms on MI50).
2. Drafter block-width interference fix: if width-8 positions 1-4 matched width-5
   acceptance, d7 len -> 2.97 -> +12% tg at d7. Investigate lattice selector chain and
   mask-block context at trained width.
3. Verify width 6-8 kernel work: nc 6-8 runs 1 warp (bit-match-to-stock constraint,
   d67cd96f0f); A/B 2-warp nc at width 8 (needs a bit-id policy call: make 1-warp
   canonical instead). Every 1 ms saved at width 8 = +2.2% tg at d7.
4. Drafter placement: pin drafter to one GPU instead of TP-replicating (their docs price
   replication at 19%; our draft-fwd is 10.3 ms/step = 23% of d7 step). Est +5-9%.
5. Serving rule (measured, stands): fork DFlash2 at n_max=4 for mixed traffic;
   n_max=7 only for code/tech-heavy traffic. Do not raise the fork design point to 7.

## 7. Artifacts

/root/taskG5/: drive.sh, agg.py, perprompt.py, d{4,7}-r{1,2}.{log,server.log,witness,
env-witness,reps.jsonl,reps.jsonl.agg.json}, d{4,7}-p.{same + agg-prof.json}, driver.out.
