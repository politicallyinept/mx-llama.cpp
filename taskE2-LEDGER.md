# Task E2: repacked nc verify widths 6-8 at 1 warp (E4 plan step 3)

Date: 2026-11-11 (PST). Branch INT4-noquant, commit d67cd96f0f (base 4fa0ef66c0).
GPU legs: GPU2 only, flock /root/gpu.lock held, TheRock env (/root/w4a4-env.sh),
TERM teardown PID-verified on every server leg.

## Problem
The restyle-gated nc dispatch launched `mul_mat_vec_q8_0_repacked8_nc<2, NC>` for
every width 2..8 (q8_repack/mul-mat.cu, RP_R8NC_LAUNCH). The stock GCN mmvq table
(`calc_nwarps`, mmvq.cu:469-486) runs nwarps=2 for ncols_dst 1-4 and 5 (Q8_0
special case) but nwarps=1 for 6-8. The warp count sets
`blocks_per_iter = vdr*nwarps*warp_size/qi` (32 vs 16): the kbx->thread
partition, the per-thread fp32 accumulation order, and whether a cross-warp
shared partial exists. So widths 6-8 diverged from stock; width 8 = dflash
verify, blocking full default-ON.

## Change (commit d67cd96f0f)
- `ggml/src/ggml-cuda/q8_repack/mul-mat.cu`: RP_R8NC_LAUNCH takes NWARPS; the
  restyle-gated switch selects 2 warps for widths 2-5 (unchanged) and 1 warp
  for 6-8. The kernel template was already generic over NWARPS - no kernel body
  change, no new instantiations beyond <1,6>,<1,7>,<1,8>.
- `ggml/src/ggml-cuda/q8_repack/repack-kernels.cuh`: comment correction (launch
  geometry note).
- `ggml/src/ggml-cuda/q8_repack/mb_nc68_test.cu`: new bit-match harness
  (precedent mb_i8narrow_test.cu).
- Q8_0 GEMM dispatch sites (mul-mat.cu:1039/:1149 area) NOT touched (sibling lane).

## Validation (a): microbench bit-match
Build: `hipcc -O2 -std=c++17 --offload-arch=gfx906 -DGGML_USE_HIP
-ffp-contract=off` (host+device share one strict IEEE binary32 model; production
may contract `acc += dw*dx*(float)sumi` into an FMA but does so identically for
stock mmvq and repacked8_nc - same source shape, same toolchain; the same
argument the widths 2-5 production bit-id gate rests on).
Oracle: CPU mirror of the stock sequence - per-thread ascending-kbx fp32 chain
(kbx = tid/4 stride 16*nwarps, kqs = 2*(tid%4)), warp0 adds warp1's partial at
nwarps=2, 64-lane shfl-XOR butterfly.
Grid: K in {2048,4096,5120} x N in {16,100} x NC in {2..8} = 42 cells
(log /root/mb_nc68_run1.log, binary /root/mb_nc68_test):
- NC 6-8: <1,NC> == stock oracle bit-exact (bad 0) - THE FIX.
- NC 2-5: <2,NC> == stock oracle bit-exact (bad 0) - shipped arm unchanged.
- NC 6-8: old <2,NC> arm diverges (bad 64..581/cell) - reproduces the bug.
- <1,NC> vs <2,NC> differ at every width (geometry load-bearing); every launch
  launch-deterministic. 42/42 PASS.

## Validation (b): end-to-end greedy equivalence
Model: Qwen3-VL-8B Q8_0 tp1 (dense Q8_0 8.10 GiB; mode-1 forced admission -
witnessed "q8_repack: repacking q8_0 tensors on device 0" in gate-arm logs,
absent in stock-arm logs). Raw /completion, temp 0 seed 42.

### Concurrent burst (8+6 slots, identical prompts)
Gate-ON (with fix) vs stock defaults: all 28 responses content-identical
(sha 3b292dab779efd55). Raw-JSON sha differs only via timing stats.
Logprob-level comparison is NOT valid in concurrent runs: two identical slots
WITHIN one arm differ on 11-17/17 tokens (intra-arm control) - slot scheduling
makes mixed prefill+decode ubatches of width >=9, which hit the repacked Q8_0
GEMM (the known unfixed E1 divergence class). Batch-composition noise, not the
nc toggle.

### Deterministic single-slot isolation (the acceptance test)
Single slot, prompts sized so prefill ubatch width lands exactly on 6/7/8
(prompt_n verified per probe) -> nc fires with fully deterministic batch
composition; decode width 1 (shipped bit-identical MMV). Arms:
- OLD  = gate ON, nc 6-8 reverted to 2-warp (toggle build, same tree otherwise)
- FIX  = gate ON, committed 1-warp dispatch
- STOCK= defaults (repack suppressed)
Expected: FIX == STOCK bit-identical (logprob-level hash over content +
completion_probabilities); OLD != STOCK (bug demonstrated, test non-vacuous).
Result (prompt_n verified 6/7/8; OLD on GPU2 pre-contention, FIX+STOCK on GPU0):
- FIX vs STOCK: 0/16 differing tokens per width, max logprob delta 0.000e+00
  at ALL THREE WIDTHS - bit-identical end-to-end at logprob level. PASS.
- FIX vs OLD: 16/16 tokens differ at every width (max delta 3.3e-2 / 1.6 / 7.8e-2)
  - the old 2-warp arm diverged end-to-end; the fix corrects it. Non-vacuous.
- Greedy content: FIX == STOCK at all widths; OLD diverges at w7 content.
- (Whole-file sha256 hashes HASH OLD e4ce2710f3bf799b / FIX aba8a1265134cef0 /
  STOCK 4c4a5dfce1297938 differ only because the hash seeds include the
  filename tag; the per-token payload comparison above is the verdict.)

## Ops notes
- Build-graph collision (transient): a sibling's untracked q8_repack/mb_e4_gemm.cu
  (own main(), bad include path) got globbed into libggml-hip by
  `file(GLOB ... q8_repack/*.cu)` (the exclusion regex only covers *_test.cu)
  and broke the restore build. Parked it (dotfile rename) for the build window,
  reconfigured, built, restored it byte-identical (md5 9f1e789bd44b3db5ef9c99336e12fc1b verified).
  Sibling should rename to *_test.cu or add an exclusion.
- GPU legs: OLD ran on GPU2 before sibling contention; FIX/STOCK moved to GPU0
  per orchestrator steering (GPU2 held by sibling's measurement campaign);
  the FIX-vs-STOCK verdict pair ran on the SAME GPU (GPU0). All legs flock'd.
- pgrep self-match trap hit once: a wait-loop `pgrep -f llama-server` matched the
  chain's own command line (contains "--target llama-server"); use pgrep -x.

## Notes
- Gate-ON vs stock divergence on wider shapes (prefill GEMM >=9 tokens) is the
  pre-existing E1 item-#5 defect class, out of scope here; the single-slot test
  keeps every ubatch <=8 so that path never fires.
- Perf: the 1-warp arm at 6-8 matches stock's own geometry (stock runs 1 warp
  there), so no regression vs stock is expected; perf A/B out of scope (plan
  validation item 6 is greedy-coverage only).
- Full default-ON still needs the GEMM restyle + tensor-split-only admission
  (plan risk 5); this task closes only the nc 6-8 closure.
- Artifacts on mi50: /root/mb_nc68_test{,.log}, /root/e2-smoke-*, /root/e2-toggle-*,
  /root/e2-single-* (+ logs), scripts /root/e2-smoke.sh /root/e2-toggle.sh /root/e2-single.sh.
