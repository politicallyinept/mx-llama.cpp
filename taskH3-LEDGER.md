# taskH3 LEDGER - drafter placement for DFlash2 on the 35B Q8_0 MoE (TP2): replicated vs pinned

Date: 2026-11-12 (PST). Box: mi50 (3x MI50 gfx906, LXC on PVE). No pushes; local commit only.
Build: /root/int4-noquant-wt/build/bin/llama-server @ tree 5a918c5576 (same binary as
taskG5/taskV2 fork legs). Stack: target Qwen_Qwen3.6-35B-A3B-Q8_0.gguf (35.2 GiB, TP2
GPU1+GPU2) + drafter Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf (570 MB, 6 layers, block_size 8).
Serving flags per taskG5: w4a4-env + GGML_CUDA_ALLREDUCE=internal REPACK_KQUANT=0
GDN_COLTHREAD=1, -ngl 99 -fa on -sm tensor --fit off -c 16384 -ub 1024 -t 16 -np 1 -lm dio,
--spec-type draft-dflash --spec-draft-n-max {4,7}. Harness /root/p2-prefill/bench.py
(natural suite, tg256 median of 8 prompts x 3 reps, temp-0) - same suite/build as taskG5 so
the replicated arms are directly comparable; per-arm refuse-if-exists files. Driver
/root/taskH3/drive.sh + supervise.sh (orphan-reaping, resumable). All legs flock -w 600
/root/gpu.lock, TERM-only teardown, TheRock runtime witnessed, interleaved with sibling H1.

## 1. Pinning feasibility (code cite + flag test)

VERDICT: pinning is FEASIBLE, but not via -tpsd. The supported path is `-devd <one device>`
plus a drafter GGUF that carries its OWN token_embd.weight + output.weight (own-head copy).

- `-tpsd` / `--spec-draft-tensor-parallel-size` EXISTS in the build (common/arg.cpp:2855 ->
  params.speculative.draft.tensor_parallel_size; --help witness in probeT.witness). But it is
  a TP GROUP WIDTH, not a pin: in llama_prepare_model_devices/draft_subset_devs
  (src/llama.cpp:164-186) an explicit tpsd is a device-SUBSET request only when it does NOT
  divide the device count. On our 2-device TP group tpsd=1 divides 2, so it resolves to
  tps=1 x n_stages=2: the draft is PIPELINE-split across both cards (a cross-device hop per
  drafted token - the flag text itself warns this is why the draft does not inherit -tps).
  Flag test probeT (-tpsd 1, borrowed-head drafter): loads and drafts fine (acceptance
  0.952, mean len 4.70) - it cannot pin, because a pinned borrowed-head draft would have
  aborted like probeA. -tpsd 1 is NOT the pinning knob on TP2.
- Pinning knob = `-devd` / `--spec-draft-device` (common/arg.cpp:4271); a single-device draft
  list drops the meta wrapper (common/speculative.cpp:3481-3486: split_mode -> LAYER).
- The external review's constraint is CONFIRMED by crash: the stock DFlash2 drafter ships no
  head or embeddings (96 tensors, none matching output/token_embd) and borrows the target's
  via ctx_other (src/models/dflash.cpp:684-691 tok_embd, 809-820 output). Under -sm tensor
  those tensors live on the target's Meta device, so `-devd ROCm0` with the stock drafter
  aborts at draft-context init:
  `ggml-backend.cpp:1461: pre-allocated tensor (output.weight) in a buffer
  (Meta(ROCm0,ROCm1)) that cannot run the operation (NONE)` (probeA-devd-borrowed, rc=3).
- The fork already anticipates the own-head-copy fix: dflash.cpp:170-173 creates output as
  TENSOR_NOT_REQUIRED with the comment "a draft with its own embeddings + head references no
  target tensors and can run on devices the target does not use (e.g. -devd with a
  tensor-split target)"; tok_embd likewise optional at dflash.cpp:117.
- Implementation (GGUF-level, no code change): /root/taskH3/graft_ownhead.py byte-copies the
  target's token_embd.weight + output.weight (Q8_0 [2048, 248320], 540,344,320 B each) into a
  copy of the drafter -> /root/taskH3/draft/DFlash2-Q8_0-ownhead.gguf (1,651,157,280 B).
  Verified: header re-parse (n_tensors 96->98, originals unmoved), FULL byte-compare of both
  grafted tensors vs the target source, spot-compare of original tensors.
  probeB (grafted drafter + -devd ROCm0): loads, serves, drafts with acceptance 0.95238 /
  mean len 4.70 - IDENTICAL to probeT on the same short prompt: the graft is numerics-
  preserving (same Q8_0 head bytes), and every A/B leg below shows bit-equal acceptance.

## 2. A/B numbers (same-session, interleaved; natural suite; tg256 median)

| arm | rep tg256 | pin tg256 | pin-vs-rep | mean accept len | accept rate |
|---|---|---|---|---|---|
| n_max=4 | 69.18 / 70.22 (med 69.70) | 79.54 / 79.69 (med 79.62) | +14.2% | 2.4919 both | 0.373 both |
| n_max=7 | 59.89 / 59.90 (med 59.90) | 64.26 / 64.30 (med 64.28) | +7.3%  | 2.6535 both | 0.236 both |

Drift control: rep arms vs taskG5 same-build numbers (d4 70.58, d7 59.13): -1.2% / +1.3% -
session drift inside taskG5's own 2.3% leg spread; pin arms are paired within-session anyway.
Acceptance is IDENTICAL between arms at both depths (len 2.4919/2.6535 to 4 dp) - the delta
is pure placement, not draft quality.

Derived step time (len/tg): d4 35.75 -> 31.30 ms (-4.45 ms, -12.4%); d7 44.30 -> 41.28 ms
(-3.02 ms, -6.8%).

### Drafter fwd (GGML_DFLASH_PROF instrumented legs)

| phase | rep d4 (taskG5 d4-p) | pin d4 (d4-p) | rep d7 (taskG5 d7-p) | pin d7 (d7-p) |
|---|---|---|---|---|
| draft-fwd event mean ms/step | 8.94 | 0.63 | 10.34 | 0.76 |
| draft-inject p50 | 1.05 | 0.26 | 1.05 | 0.50 |
| draft-gather p50 | 0.03 | 0.02 | 0.17 | 0.18 |
| tgt-decode w5/w8 mean | 22.40 | 23.35 | 28.89 | 30.85 |

The pinned drafter forward collapses ~93% (8.94 -> 0.63 at d4, 10.34 -> 0.76 at d7): the
replicated draft's cost was per-layer allreduce + meta orchestration across the 6 drafter
layers, not compute. The win is partly given back by (a) host-mediated gather/inject sync on
the pinned arm (loop draft-phase ~4.6-6.4 ms/iter vs the ~0.9 ms of device events) and (b)
the pinned draft sharing GPU1 with the target's half: tgt-decode +0.95/+1.96 ms vs taskG5.

## 3. Delta vs the 19% prediction

ruquant docs price TP replication at 19%; taskG5 candidate-4 estimate was +5-9%. Measured:
+14.2% at n_max=4 (near the 19% ceiling, above the estimate), +7.3% at n_max=7 (inside the
estimate). The residual gap to 19% is explained by the host round-trip the pinned draft pays
on our engine (gather-to-host + inject-to-device sync per step) and same-card contention with
the target - ruquant's engine hides those. d7 gains less because the step is dominated by the
width-8 verify pass (taskG5 root cause), which pinning does not touch.

## 4. Serving rule

DFlash2 TP2 on this rig: pin the drafter (own-head graft + -devd) at both depths; the win is
larger at the recommended design point n_max=4 (+14.3%). The graft is a GGUF file swap - no
code shipped (none needed). A server-level alternative (replicate output into the draft's
meta graph) would avoid the host round-trip but is a code change out of scope here.

## 5. Artifacts

/root/taskH3/: graft_ownhead.py, drive.sh, supervise.sh, cleanup.sh, analyze.py,
draft/DFlash2-Q8_0-ownhead.gguf, probe{A,T,B}-*.{log,server.log,witness,env-witness},
{rep,pin}-d{4,7}-r{1,2}.{log,server.log,witness,env-witness,reps.jsonl,reps.jsonl.agg.json},
d{4,7}-p.{same}, driver.out.
