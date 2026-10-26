# Upstream Merge Plan: ggml-org master -> gfx906-production-123b

**Date:** 2026-09-26 · **Status:** PLANNED (not started)
**Base:** gitea/gfx906-production-123b @ d8fe612c3 (prod binary aa6b4c5cb, deployed + smoke-verified)
**Target:** ggml-org/llama.cpp master (445 commits past merge-base 0f3a71be1, 2026-09-02)
**Precedent:** efd6b9976 (2026-09-19 vendor sync, clean, no regressions). This is the first ggml-org sync since 09-02.

---

## 1. What we are getting (ranked by MI50 value)

| Commit | What | Why it matters here |
|---|---|---|
| c8edceb06 (#27841) | GCN MMQ launch-config table (`mmq-config-gcn.cuh`, 281 lines) | **Governs MI50 decode**: upstream classifies gfx906 as GCN (cc < CDNA1). Affects MMQ models: Cydonia Q8_0, Precog Q4_0, all non-toggled entries, DFlash draft. NOT the two W4A4-toggled models (repack path bypasses MMQ). No upstream gfx906 numbers -> must A/B. |
| 2f3fd0252 (#28549) | CUDA graph for MTP draft | Spec-decode base improvement (perf only; correctness bug NOT fixed, see 1b) |
| 5cdd3d1da (#28630) | MTP kv-cache allocation fix (deepseek2/glm4moe/...) | Draft-model correctness fix |
| 415e909d8 (#28390) | Single-device drafter meta backend wrapper | Drafter robustness |
| e9f824d8c (#24364) | `llama_prec_policy` + NVFP4 W4A4/W4A8 (Blackwell-only) | **Zero MI50 runtime effect** (gated on `blackwell_mma_available(cc)` + FP4 types). Matters as (a) the main conflict driver, (b) the upstream precision-metadata convention our future paths should adopt. |
| 08618ff8e (#27530) | K/V + recurrent state cleanup after failed restores | General correctness |
| fc343a84b (#24669) | `llama_batch_ext` | API modernization our fork will need eventually |

### 1b. What this merge does NOT fix
`set_sampler` backend-sampling guard (src/llama-context.cpp:1566-1577) is **byte-identical upstream**. MTP + DFlash remain lossless-blocked. Do not expect spec-decode correctness from this merge. (DFlash2 verdict d8fe612c3 stands: perf +10.6%, gate FAILED.)

## 2. Pre-flight (CPU-only, no GPU)

1. Fresh worktree from gitea/gfx906-production-123b. `git merge --no-ff upstream/master`.
2. Trial-merge (2026-09-26) predicts **14 conflicted files**. Resolution intent:

| File | Fork side | Upstream side | Resolution |
|---|---|---|---|
| .github/actions/windows-setup-cuda/action.yml | none of value | CI update | take upstream |
| common/speculative.cpp | DSpark/DFlash touches | drafter changes | manual union, keep dflash paths compiling |
| ggml/src/ggml-backend-meta.cpp | PLX meta sched copies | meta backend wrapper | manual, HIGH CARE (sched core) |
| ggml/src/ggml-cuda/ggml-cuda.cu | custom AR dispatch + W4A4 dispatch | allreduce dispatcher + NVFP4 dispatch | keep BOTH gates; custom-AR gate must stay ahead of generic allreduce in dispatch order |
| ggml/src/ggml-cuda/mmq.cu | repack interplay | GCN table selector + NVFP4 tiles | manual; repack short-circuit must precede MMQ config lookup |
| src/llama-arch.cpp/.h | gemma4/wordsmith arch | prec-policy refactor (52 lines) | union |
| src/llama-context.cpp/.h | set_sampler guard + MTP opt | prec plumbing + batch_ext | keep fork guard EXACTLY; do not "simplify" |
| src/llama-graph.cpp/.h | W4A4 hints | prec policy +9 lines | union |
| src/llama-model.cpp | fork deltas | prec policy +53 lines | union |
| src/models/{mimo2,qwen35,qwen4exp}.cpp | check fork deltas first | model updates | prefer upstream, re-apply fork deltas if present |
| tools/llama-bench/llama-bench.cpp | bench flags if any | upstream bench | upstream + re-add fork flags |

3. **Invariants grep (must all pass post-resolve):**
   - `getenv("GGML_CUDA_W4A4_DOT8")` in `mul-mat.cu` + `mul-mat-id.cu` (default OFF)
   - `GGML_TP_AR_MTP_SAFE` sticky lock intact
   - `set_sampler` guard block identical to pre-merge
   - custom AR env gates (`GGML_TP_AR_SKIP_XBAR`, threshold 131072) intact
   - `q8_repack/` tree untouched by upstream (auto-merged clean in trial)

4. **Full clean rebuild** of build-w4a4 (ggml-impl.h + backend ABI changed -> no incremental reuse). Prod build-rccl only after gates pass.
5. `test-backend-ops` on HIP (upstream added NVFP4 cases; must pass).

## 3. GPU validation (flock /root/gpu.lock, all runs same-day interleaved)

### Phase A - GCN MMQ table A/B (the new variable, isolated)
- A = current prod binary (pre-merge), B = merged build
- Models: **Cydonia-R1-24B-v4f** (Q8_0 pure MMQ) and **Precog** (Q4_0 repack-off path)
- Matrix: tg128 + pp2048, 3 runs each, interleaved A/B/A/B
- **Gate: any tg128 regression > 2% -> stop, bisect mmq-config-gcn table** (worst case: pin the old config as a local patch; the file is self-contained)

### Phase B - standard gauntlet (BENCH.md §35+ protocol)
- Mistral-class-prod: tg128/pp2048 at W4A4 gate=0/1/2 + PPL@2k/8k + greedy primes @64k
- Ornith 2rk spot check (MoE; shares the allreduce path)
- TP AR nranks-3 twoshot behavior unchanged

### Phase C - prod cutover
- Rebuild build-rccl from merged branch; **rename old prod binary, do not delete** (rollback path)
- llama-swap restart off-hours; post-cutover smoke: Mistral-class-prod + production-128B live-gen + one non-toggled model
- Monitor 24h for the 175W/stability watch

## 4. Risks

| Risk | Mitigation |
|---|---|
| GCN MMQ table regresses gfx906 decode | Phase A gate + table is a self-contained file, pinning is trivial |
| ABI break -> stale-objects build | clean rebuild mandated |
| Dispatcher fight: custom AR vs upstream allreduce | explicit dispatch-order check in invariants step |
| MTP graph enable changes draft perf | re-bench MTP tg post-merge (low priority; still lossless-gated) |
| Merge widens while MXFP4 work waits | do this merge BEFORE starting MXFP4 imatrix path |

## 5. Sequencing + labor

1. Subagents (CPU): per-file-group conflict resolution + invariants grep + clean build (est. half day)
2. Brad/me: Phase A-C on GPU (est. half day; Phase A alone ~1h)
3. After gates: rebase the three D3 upstream-PR candidates (nranks-3 twoshot, Q4_0 repack entry, dense-MXFP4 mode) onto the merged tree and submit - the prec-policy discussion shows upstream moving fast; submit before they restructure again
