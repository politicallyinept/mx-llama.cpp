# taskW1 LEDGER — root-cause of the ~10 ms/step host draft-orchestration cost (pinned -devd DFlash2 path)

Date: 2026-11-15 (PST). Box: mi50. Branch `int4-w1-hostorch` off INT4-noquant tip c0dd09e4c9
(NOT pushed). NOTE: taskP2 fast-forwarded this branch to 44302fb494 (env-gated i8narrow
strip, probe-sealed) at 11:23 before my build; build-w1 = 44302fb494 + instrumentation.
Build: /root/build-w1 (md5s in /root/build-w1/BUILD-PROVENANCE.txt).
Pair: GPUs 1,2 (rocm0 quarantined, 919k+ correctable ECC; exporter re-zeroed but quarantine stands).
Baseline: pinned f0q4-d4 = 57.09 t/s (taskV1 solo, pair 0,1); taskV3 pair 1,2 reps 58.70/57.36/57.03.
Canonical probe sha16: 77841cb6a251235d (the brief's "77841cb6a235d" is a corrupted copy;
first 10 hex match — same convention as taskV1 §4).

## 0. Why

taskR §5: pinned step = 44.8 ms wall while device events sum ~34.5 ms → ~10 ms/step host
draft-orchestration. draft-fwd timer (llama_process wall) collapsed 11.5→0.52 ms but the
step only improved ~2 ms. Both external reviews rank this #1 residual.

## 1. Free data mined from taskV1 f0q4-p-d4-r1 log (build-uwave, pair 0,1, GGML_DFLASH_PROF)

- spec dur(g) bucket: 34,304 ms / 3,421 draft() calls = **10.03 ms/call wall** — the whole
  ~10 ms lives INSIDE common_speculative_draft(), not in tgt sync/accept.
- draft-fwd (llama_process launch window): 0.63 ms mean — async launch only; the device
  wait hides in llama_get_embeddings_nextn() → ctx->synchronize() (uninstrumented).
- Steady-state draft forward = ONE graph_compute, dev0, ne11=5, **nodes=690**, and it was
  ALREADY a captured+replayed whole-draft HIP graph in taskV1 too: nodes=690 mode
  histogram = replay 3,291 / capture 61 / direct 69 (identical in f0q4-p-d4-r1 and in
  my w1-meas-p-d4 leg; the 69 directs are per-request warmups, ~1.25 ms host each).
- Target verify width-5 subgraphs also run mode=direct on both devs in steady state
  (token graph tg-replay only 40/3,394 steps; owner-off 12,384 + direct 8,648 @ ne11=5).
- cuda-graph capture churn: 5,636 captures @ ne11=5 (~1.65/step, 280 µs each ≈ 0.46
  ms/step) — spread through steady-state decode (target-side graph resets, not draft-side;
  sits inside the verify bucket).
- 1,148 cuda-graph-unstable permanent disables (resets=4).
- ckpt-update-tgt 80 calls × 73 ms = per-request prompt checkpoints, NOT per-step. ckpt-load: 0.
- gather 0.03 ms, tgtsync 0.05 ms, inject 0.36 ms, accept 0.13 ms per decode call — all small.

## 2. Instrumentation (this task)

common/speculative.cpp draft(): GGML_DFLASH_PROF-gated host-phase timers →
`[DFLASH-PROF] draft-host call=N total_us= flush_us= batch_us= fwd_us= readnx_us= walk_us=`
+ `draft-fwd-begin` marker for log-order attribution of cuda-graph lines to the draft window.
Gate-off = byte-identical (fprintf only when env set).

## 3. Measurement leg (build-w1, pair 1,2, w1-meas-p-d4, 11:40-11:47)

- tg256 median **57.94 t/s** (24 reps, natural suite) — instrumented build tracks the
  57.09/58.70 baseline band; acceptance per-prompt matches taskV1/V3 references.
- Greedy probe: **sha16 77841cb6a251235d CANONICAL-MATCH** (instrumentation inert).
- Teardown clean (TERM-only, pgrep empty; no WARN/STUCK/LOCKFAIL).

Per-step phase table (n=3,421 draft() calls, means):

| phase (inside draft())                    | ms/step | note |
|-------------------------------------------|---------|------|
| flush_pending                             | 0.000   | empty during decode |
| batch build + sampler reset               | 0.003   | |
| llama_process (launch: sched re-split + stable-uid hash + HIP graph replay launch) | 0.605 | graph_compute host time only 0.081; replay mode 3,291/3,421 |
| **llama_get_embeddings_nextn -> ctx->synchronize (DRAFT DEVICE TIME)** | **9.260** | waits the 690-kernel draft HIP graph |
| selector lattice walk (host)              | 0.001   | |
| residual loop bookkeeping                 | 0.001   | |
| **draft() total**                         | **9.870** | matches server t_draft bucket 9.64-9.88 |

Step accounting: 44.8 ms step = 34.4 verify device (tgt-decode w5 34.37 inner 34.31)
+ 9.6 draft phase + ~0.7 inject/accept/other. Device sum 34.4+9.3 = 43.7 ≈ wall:
**the GPU is NOT idle during the "missing" 10 ms — it is executing the drafter.**

## 4. Root cause

The ~10 ms/step is the **DFlash2 drafter's own forward device time**, not host
orchestration. The pinned draft forward is already a single whole-draft HIP graph
(690 nodes, dev0, replayed 96% of steps, host launch 8-84 µs). Its device execution is
9.26 ms. GGUF tensor census of the drafter (DFlash2-Q4_0-ownhead-q4, 2.65 GiB total):
**head 0.97 GiB (37%), FFN 0.72 GiB (27%), attn 0.15 GiB, other 0.80 GiB** — the
embedding table is gather-only (~25 KB/step), so streamed weight bytes per draft step
≈ 1.95 GiB => **~226 GB/s effective per GPU** — identical to the target verify's
per-GPU rate (14.5 GiB streamed / 34.4 ms TP2 = 226 GB/s/GPU). Both phases sit at the
same achieved Q4_0 decode bandwidth floor; the drafter is bandwidth-bound, with the
256k-vocab own-head alone costing ~4.5 ms of the 9.26 ms.

Why it was mistaken for host time: the draft-fwd timer wraps llama_process, which is an
ASYNC launch (returns after submission). Borrowed path: llama_process was synchronous
(TP-split, 11.5 ms wall) so the same device time showed up "in" draft-fwd; pinning moved
the wait into llama_get_embeddings_nextn()'s ctx->synchronize() (uninstrumented), which
read as "0.5 ms device-side + 10 ms host". taskR §5's own arithmetic (device saved 11 ms
vs wall saved 2.1 ms) is fully explained: both paths pay ~9-11 ms of drafter device time;
pinning only removed ~1.5-2 ms of TP-split/allreduce overhead + host sync churn.

Host-phase residual (the actual host orchestration): batch build 0.003 + walk 0.001 +
launch/scheduler 0.605 = **0.61 ms/step**, not 10. The brief's premise is refuted with
numbers; the ms sum to ~10 as: 9.26 drafter device + 0.60 launch/scheduler + 0.004
batch/walk + 0.006 residual ≈ 9.87 draft() wall.

## 4b. Ranked residual causes (post root-cause)

1. Drafter forward device time 9.26 ms (bandwidth-bound 2.85 GB Q4_0 forward at width 5;
   serial with verify by data dependency: draft N+1 needs the token sampled at N).
2. llama_process host overhead 0.60 ms/step (scheduler re-split + stable-uid hash over
   ~700 nodes/leafs every step + replay launch 0.08 ms).
2b. Target-side capture churn ~0.46 ms/step (1.65 re-captures/step @ ne11=5 × 280 µs,
   spread through decode; target-ctx graph instability + 1,148 permanent disables).
   Largest remaining host item but OUTSIDE the draft path — belongs to the verify lane.
3. Per-request draft-graph re-warm: ~61 captures + 69 directs over 58 requests (~1.8 ms
   per request start, amortized <0.05 ms/step).
4. Everything else (batch, walk, flush, ckpt-per-request, inject, accept): <1 ms/step.

## 4c. rocprofv3 kernel trace (w1-diag, 45 steady-state draft windows)

Segmented by GPU2-idle gaps (verify is TP2 on both devs; the draft is GPU1-only), so each
window = accept + draft-graph + inject as seen on GPU1:

- GPU1 kernel-busy per window: **10.84 ms** — matches readnx 9.26 ms + inject ~1 ms +
  boundary slack. The sync wait IS kernel execution; there is no launch-gap pool.
- Composition (ms/step, x launches/step):
  - `mul_mat_vec_q` **8.68 ms x58** — 80% of draft device time. Weight-streaming matvecs
    (5 layers x ~11 + 256k-vocab head + selector scores) => bandwidth floor, confirmed
    at kernel level.
  - glue pool ~2.2 ms: k_bin_bcast 0.55 x109, copyBufferRect 0.31 x60 (KV/conv-state
    moves), flash_attn 0.23, quantize_q8_1 0.20 x33 (mmvq activation quant), rms_norm
    0.21, pad/copy 0.22, rope 0.08, set_rows 0.09, **top_k_radix_* 0.15** (selector
    top-k — negligible, refutes the per-position-selector-cost worry at kernel level).
- Fusion ceiling if ALL glue vanished: ~2.2 ms/step = 4.9% of step; realistic fusion
  gain ~1 ms = 2.2% — kernel-program territory, below this task's env-gate remit.

## 5. Fix verdict

**No env-gated host-side fix is justified — the premise ("~10 ms host time, GPUs idle")
is false.** The ms accounted: 9.26 ms is the drafter's HIP-graph device execution
(HBM-bound: ~1.95 GiB streamed / 9.26 ms = 226 GB/s/GPU — exactly the rate the target
verify achieves per GPU (14.5 GiB / 34.4 ms TP2), i.e. both phases are at the fleet's
achieved Q4_0 decode bandwidth floor; there is no launch-overhead or gap pool to
harvest). The brief's conditional for prototyping a whole-draft HIP graph capture —
"if launch-count-bound" — is refuted: the whole-draft capture already exists via the
per-subgraph graph cache (single 690-node graph, replayed 3,291/3,421 = 96% of steps,
host launch 8-84 µs; the remaining 4% are per-request re-warms, ~1.8 ms each).

Levers considered and rejected with numbers:
- TP2-split the drafter across both idle-during-draft GPUs: this IS the borrowed path —
  measured worse (11.5 ms vs 9.26 ms; allreduce storm on 690 tiny kernels, taskR H3).
- Overlap draft with verify: impossible — draft N+1 consumes the token sampled at N and
  the verify-N hidden states (data dependency, not scheduling).
- Adaptive draft depth (taskH1 dctl): device time is weight-bandwidth-bound, ~independent
  of width (d7 vs d4 measured neutral, taskV3) — narrower drafts save nothing.
- Shave the 0.60 ms launch bucket (scheduler re-split + stable-uid hash over ~700
  nodes/step): 1.3% of step; the hash is what makes replay-against-moved-memory
  impossible (documented HSA aperture violations); halving it lands under the 2.9% noise
  floor. Not decision-grade, risk-negative.
- Repack admission for drafter tensors: already active (GGML_CUDA_REPACK_Q4_0=1 admits
  the drafter's 2-D contiguous Q4_0; RP_Q4_NC_LDS=8 routes its width-5 nc matvecs to the
  LDS-staged kernel). The 226 GB/s is the repacked path's achieved rate.

## 6. Gates (applied to the instrumentation commit; no numerics change to A/B)

- Gate-off byte-identity: additions are fprintf-only behind the GGML_DFLASH_PROF static;
  with the env unset no code path changes. With the gate ON the greedy probe is
  byte-identical to canonical sha16 77841cb6a251235d (w1-meas-p-d4 probe, 64 tok temp0
  seed42) — a strictly stronger witness than gate-off.
- KLD vs f16-KV base: numerically inert by construction (no compute-path edit). The f0q4
  arm KLD stands on record from taskV1 §3: 0.000000 +/- 0.000000, max 0.000071.
- e2e tg: instrumented build 57.94 t/s median (pair 1,2, 24 reps) vs baseline 57.09
  (taskV1 pair 0,1) and taskV3 pair 1,2 reps 58.70/57.36/57.03 — inside the session band,
  instrumentation overhead not resolvable.
- Per-phase before/after: taskR-era view "step 44.8 = 34.5 device + ~10 host" ->
  corrected "step 44.8 = 34.4 verify device + 9.26 draft device + 0.61 draft host +
  ~0.5 inject/accept/other". The host residual to attack, if ever, is 0.61 ms, not 10.

## 7. What would actually move the 9.26 ms (for the kernel program, not this task)

1. Fewer weight bytes per draft step: a smaller-quant or distilled DFlash2 drafter
   (every GiB removed ~ 4.7 ms/step at the achieved 226 GB/s rate).
2. Faster-than-repack Q4_0 decode kernels for the drafter's shape mix (5120-dim widths
   4-8 + the 256k-vocab head matvec) — the head alone is 0.97 GiB/step (~4.5 ms, 37% of
   the draft phase). A pruned draft vocabulary (as most MTP drafters ship) is the single
   biggest structural lever visible from the tensor census.
3. MTP-style fusion of the drafter into the target forward (the 144 t/s reference's
   structural advantage: the draft rides the verify's weight reads at ~0 marginal cost).

## 8. Files

- /root/taskW1/: w1-meas-p-d4.{server.log,reps.jsonl,bench.log,probe-*.json,witness,
  proc.witness,gate.witness,prof-count,run.out,.done}; w1-diag.sh + rocprof-draft/
  (rocprofv3 --kernel-trace CSV: k_kernel_trace.csv 239,753 records, 45 draft windows;
  segmented by /root/w1-kernel-seg.py). Diag teardown lessons: (a) TERM must reach the
  llama-server PID itself — TERMing the `timeout` wrapper is NOT forwarded and orphans
  the server (twice observed); (b) TERMing rocprofv3 before its child exits yields only
  an opaque .dat, no CSV — graceful server exit first is what triggers CSV generation.
- /root/w1-leg.sh, /root/w1-chain.sh, /root/w1-parse.py, /root/w1-mine.sh, /root/w1-mine2.sh
- /root/build-w1 (BUILD-PROVENANCE.txt: 44302fb494 + instrumentation, md5s recorded)
- Instrumentation commit c0e27e217f on int4-w1-hostorch; this ledger on the same branch.
  NOT pushed.
- ECC health: amd_gpu_ecc_errors_total all-zero at 11:53, 13:24 and 13:39 (exporter on
  localhost:9400; GPU0 quarantine stands regardless). All GPU legs flock'd on
  /root/gpu.lock, production-env witnessed, TERM-only teardown, pgrep -x llama-server
  empty at close (13:40).
