# MI50 W4A4/Speculative Program - Current State
**Date:** 2026-09-29 evening - **Author:** Hermes (deploy + Flash-Next program session)
**Repo:** mx-llama.cpp, branch `gfx906-w4a4` (production docs) + `perf/w4a4-dot8-gfx906` (all W4A4 work)
**Rig:** 3x MI50 (gfx906), PLX 88096 P2P, 175W/card power cap, TheRock nightly 7.15.0a runtime
**Authoritative data:** BENCH.md sections 30-43 on Gitea. Everything below the STATUS UPDATE header is
the 2026-09-26 snapshot, kept for history; the update supersedes it.

## STATUS UPDATE (2026-09-29)

### Deployed since 09-26
- **D1 DONE**: dense W4A4 gate=1 IS in production. Prod binary rebuilt 09-26 12:04 at aa6b4c5cb
  (W4A4 code merged into gfx906-w4a4, commit 2a8de9d9d), llama-swap restarted onto it 13:03.
  `GGML_CUDA_W4A4_DOT8=1` set per-entry (toggle) on Mistral-Large-123B fine-tune-v2 + TheDrummer-Mistral-Large-123B fine-tune-v3 only.
- **D2 DECLINED** by user: gate=2 quality mode not adopted (pp -37% not worth it for RP use).
- **D4 SOLVED**: second PSU installed; no power events since. 175W cap retained as belt-and-braces.
- **D3 PENDING**: upstream PR trio (nranks-3 twoshot, Q4_0 repack entry, dense-MXFP4 quantize mode) - user wants open paths explored first.

### New program: RVN-Qwen3.8-Flash-Next (qwen4exp MoE) - see BENCH section 42-43
- Model deployed in llama-swap (RVN-Flash-Next-MTP): target Q4_0 + MTP Q8_0 draft, -b/-ub 4096,
  GGML_TP_AR_MTP_SAFE=1 + LLAMA_PLE_PREFAULT=1. PLE n-gram table (27.5G) demand-paged from disk, zero VRAM.
- Prod stack verified active on it: Q4_0 repack + custom AR twoshot (+10% decode vs AR-off).
- MoE W4A4 REJECT reproduced (decode -5.4%); W4A4 PREFILL is a +7-25% win on this arch -> gate mode 3
  implemented (batch-gated dispatch, commit 3b1079b84 on perf/w4a4-dot8-gfx906): W4A4 for multi-token
  GEMMs, stock path for decode. Decode fully protected (-0.1%). Not enabled on any entry yet (MoE pp
  gain dilutes to +2.5%; natural fit is dense entries).
- MTP draft: +14.7% tg avg (above the +10% bar), SAFE guard ~free here; NOT lossless (same set_sampler
  fork bug as 2c below) - deployed as-is per user call, same regime as other MTP entries.
- Levers landed: LLAMA_PLE_PREFAULT=1 (+23% cold prefill, kills TTL-reload first-request penalty),
  -ub 4096 (+6-9% warm prefill).

### Open bugs under investigation (2026-09-29, source analysis in flight)
1. qwen4exp warmup HANG with GGML_CUDA_REPACK_Q4_0=0 under -sm tensor (section 42 finding 5) - deterministic; repack currently functionally required for this arch.
2. set_sampler MTP fork bug (below, 2c) unchanged - fix design being drafted; upstream status being checked.

### 09-30 addendum: investigations closed
- MTP sampler guard T1 fix IMPLEMENTED + SEALED (fix/mtp-sampler-guard-t1, f83423ec2): capability test
  (device sampling iff logits MIRRORED or TP width 1). Correct + perf win, but seals REFUTED the guard
  as the losslessness blocker: RVN MTP + DFlash2 diverge with device sampling engaged and AR off ->
  upstream #25618 batch-invariance class (BENCH 44a).
- qwen4exp stock-load "hang" RESOLVED (BENCH 44b + addendum): mmap-loader page-cache thrash in the meta
  set_tensor 2D-copy path (per-call GART pin + stream sync, 26x read amplification). Mitigation deployed:
  -lm dio on the RVN swap entry. Fix design documented, not implemented.
- DFlash2 retest CLOSED: superseded - the 44a seals show DFlash2 losslessness is #25618-class, and MTP
  won the scheme shootout (43c).
- u4_quant_check.hip harness: superseded by mode-3 first-token seals; CLOSED.

### MXFP4 imatrix - CORRECTION: already done, REJECT (2026-09-28, branch perf/mxfp4-imatrix)
The 09-26 snapshot listed this as untried; it was completed on 09-28 (commits b0cd9cbb7 M1 quantizer,
aa11d00de M2 CLI, a531a28a9 M3 Ornith validation):
- imatrix closes 56% of the MXFP4-vs-Q4_0 PPL gap (2.4418 -> 2.1626, Q4_0 ref 1.9455)
- residual = E2M1 mantissa poverty (8 levels vs Q4_0 16): the FORMAT is the wall, not the quantizer
- ADOPTION BAR FAILED (bar was PPL <= Q4_0); STOP per protocol. Results: RESULTS-mxfp4-imatrix.md
  on perf/mxfp4-imatrix. MXFP4 stays experimental on gfx906.

### Still open (true remaining items)
- Upstream PR trio (nranks-3 twoshot, Q4_0 repack entry, dense-MXFP4 quantize mode) + upstream discussion
  pieces now ready: T1 guard fix + mmap meta-copy pathology (with fix design).
- MXFP4 frontier beyond the E2M1 wall (only if wanted): no quantizer-side path remains; the wall is
  format-level. Options: accept, or exotic (NVFP4 E4M3 scales - scale-rich but same 2-bit mantissa,
  unlikely to clear; GEMM-side per-8 epilogue rework was also REJECT on perf).
- W4A4 gate mode 3 enablement on dense entries (optional, ~neutral for dense since mode1==mode3 prefill).

---

## ARCHIVE: 2026-09-26 snapshot
**Date:** 2026-09-26 · **Author:** Hermes (session audit + full program state)
**Repo:** mx-llama.cpp, branch `gfx906-w4a4` (production docs) + `perf/w4a4-dot8-gfx906` (all W4A4 work)
**Rig:** 3x MI50 (gfx906), PLX 88096 P2P, 175W/card power cap (upp, host-persistent), TheRock nightly 7.15.0a runtime
**Authoritative data:** BENCH.md sections 30-39 on Gitea. All pre-175W numbers (§30-34 absolute values) are superseded; §35+ is the live baseline.

## 0. STATUS UPDATE (2026-09-26 evening)

- **UPSTREAM MERGE DEPLOYED (2026-09-26 ~23:10)**: ggml-org master (2b129ccfa, 445 commits) merged into gfx906-w4a4; prod branch = e4f26e4cb. All 14 conflict files hand-resolved (plan: docs/upstream-merge-plan-20260926.md), invariants gate green, 3 build-fix rounds (bf16 dedupe in common.cuh, qwen35 Kcur/Vcur + deepseek41 accessor, dflash pos_pinned). VALIDATION: 7-model same-harness A/B all within +-0.9% (table below); primes@32k gate=1 EXACT; coherence + DFlash draft smokes OK. Cutover: build-rccl/bin swapped in place (libggml 0.22->0.25.3), rollback = bin-premerge-aa6b4c5cb/ rename. Live llama-swap smoke: "merge cutover ok".

### Merge A/B matrix (llama-bench -r 3, same session, prod vs merge)
| Model | tg128 | pp2048 |
|---|---|---|
| Cydonia Q8_0 | 26.59 -> 26.59 (+0.00%) | 774.4 -> 773.1 (-0.16%) |
| Precog Q4_0 | 9.14 -> 9.07 (-0.77%) | 134.9 -> 135.3 (+0.25%) |
| Mistral-Large fine-tune W4A4 g=1 | 21.25 -> 21.29 (+0.19%) | 238.7 -> 240.8 (+0.88%) |
| Ornith MoE (gate off) | - | 2696 -> 2702 (+0.21%) |
GCN MMQ table (#27841) = wash on gfx906 (bandwidth-bound at these batch sizes). NOTE: same-harness deltas are much smaller than vs BENCH.md §35 absolute values (ggml lib generation changed 0.22->0.25.3); §35 numbers remain the historical record. W4A4 gate=1 is env-toggled per-model: GGML_CUDA_W4A4_DOT8=1 on Mistral-Large-123B fine-tune-v2 + TheDrummer-Mistral-Large-123B fine-tune-v3 entries ONLY (flag is getenv-gated, default off). Verified: gate symbols in shipped libggml-hip.so; llama-swap journal shows env delivered at model load. Branch: perf/w4a4-dot8-gfx906 merged into gfx906-w4a4 (2a8de9d9d) - prod branch now matches prod binary. Live-generation smoke test verified post-reboot 17:29: Mistral-Large fine-tune via llama-swap returned the exact greedy response with gate=1 env active. Mistral-Large fine-tune not individually exercised (same binary, same env mechanism).
- **D2 DECLINED** (user call): not for RP use. Correction on record: gate=2 was +0.5% tg with dPPL +0.019@8k (better than gate=1); declined on the -37% prefill cost.
- **D3 DEFERRED** (user call): upstreaming deferred until open paths are explored.
- **D4 SOLVED**: second PSU installed 2026-09-26, no rail-brownout events since. INCIDENT same day ~15:56: all-3-card SMC/powerplay wedge (message 0x2a/0x2b response 0xfc on every card, /dev/kfd open EINVAL, rocm-smi cannot initialize, reset attempts hung D-state). ROOT CAUSE IDENTIFIED: llama-cli kill -9 during model load (DFlash retest debug) - NOT spontaneous, NOT power-related. Recovery: host-side LXC reboot (mi50 is an LXC, /sys read-only from inside; PVE host action). Lesson: never kill -9 llama-cli mid-model-load on this stack. Watch only for a spontaneous (uninduced) recurrence.
- **DFlash retest v2 COMPLETE (2026-09-26 17:34)**: perf legs: base 54.2/53.3 t/s, dflash 55.2/63.7 t/s (means 53.75 vs 59.45, +10.6% for dflash, n=2 high variance). Greedy lossless gate FAILED: outputs diverge within the first thinking tokens (base "We need answer user:" vs df "We need to answer user:"), same prompt, --temp 0 --top-k 1. DFlash2 verify is not strict-greedy-equivalent under -sm tensor - same fork-bug family as MTP (llama-context.cpp:1566-1577 sampler guard). VERDICT: do-not-adopt on correctness; the marginal perf win is moot until the sampler bug is fixed. Both DFlash and MTP now gated on the same upstream/fork fix.

---

## 1. PRODUCTION STATE (what is running right now)

- llama-swap: 26+ entries, serving from `/root/mx-Mistral-Large fine-tune/build-rccl/bin/llama-server` (RCCL-linked, hybrid AR gate ON)
- Binary: rebuilt 2026-09-26 12:04 from aa6b4c5cb (dense W4A4 code included, default-off). W4A4 active ONLY where the per-model env toggle is set (Mistral-Large fine-tune, Mistral-Large fine-tune).
- Stack: nightly-first LD_LIBRARY_PATH, 175W/card, perf high, hybrid AR gate (prefill->RCCL, decode->custom AR, threshold ne<131072 @3rk confirmed optimal in Phase 1 sweep)
- Per-model: `GGML_CUDA_REPACK_Q4_0=1` global; `GGML_TP_AR_NO_GATE=1` removed (Ornith now uses the Huihui imatrix Q4_0 quant, +28.6% vs old Q4_K_M)

## 2. COMPLETED PROGRAMS (results, all measured, all documented)

### 2a. Dense W4A4 (the main program) — ADOPT-CANDIDATE, not yet deployed
Branch `perf/w4a4-dot8-gfx906`, commits: a4643ebc8 (M1 dp8a+u4 block type), cd96ad6475 (M2a quantize kernel), f5325c1e6 (M2b decode kernel), 1ec20cd54 (M3 dispatch), 88bbc3572 (M4 fix), ae6ea504b (G2 GEMM), 05bae03af (G3 GEMM dispatch), 2e8499bb6 (M4 GEMM per-8), 60c627733 (step2 spill fix), aa6b4c5cb (MTP guard).

| Config | tg128 (Mistral-Large fine-tune) | pp2048 | PPL@2k | PPL@8k |
|---|---:|---:|---:|---:|
| gate=0 stock | 20.47 | 157.7 | 1.0564 | 1.0751 |
| gate=1 per-32 | 21.20 (+3.6%) | 208.0 (+32%) | 1.0564 (0.000) | 1.1572 (+0.082) |
| gate=2 per-8 GEMM | 20.85 (+0.5%) | 98-117 (-37%) | 1.0564 (0.000) | 1.0943 (+0.019) |

- Quality: gate=1 dPPL=0.0000@2k but +0.082@8k; gate=2 dPPL=+0.019@8k (75% recovery)
- Perf: gate=1 = the speed mode; gate=2 = quality mode (-37% pp, the per-8 epilogue is FMA-bound, NOT spills — step2 fixed those and recovered 19% but the structural -26% remains)
- All quality-sealed: greedy primes exact at 64k depth on gate=1 and gate=2

### 2b. MoE W4A4 — REJECT
Milestone 4 (ad123ecdc): correct implementation (primes passed, dPPL=0) but -5.3% tg on Ornith@2rk. Mechanism: single-token MoE consumes the SUMW recompute once per block, no amortization. Env flag stays OFF for MoE entries.

### 2c. MTP speculative decode — FORK-BUG-BLOCKED
Root cause (verified in source): `llama_context::set_sampler` (src/llama-context.cpp:1566-1577) has a hard guard: backend sampling refused under SPLIT_MODE_TENSOR -> CPU sampler fallback -> verify not strict-greedy-equivalent -> greedy MTP output differs from no-MTP (both Qwen3.6 Native-MTP and Ornith MTPv2, same signature). NOT model-specific.
- MTP throughput on Qwen3.6: +7.5% median (below +10% bar) — sub-threshold even before the correctness issue
- Safety guard shipped: `GGML_TP_AR_MTP_SAFE=1/2` (commit aa6b4c5cb) — env-gated sticky lock, disables custom AR for the context on first width transition. Verified lossless byte-identical. Default OFF.
- Deployed: per-entry env on the two production draft-mtp entries (Ornith-Agents-A1, Qwen3.6-Fable-Fusion-MTP) — makes MTP lossless at -10.4% tg cost vs fast no-MTP
- MTP accepted as fork-bug-blocked, not scheme-blocked. Revisit if upstream fixes the offload path (their MTP fix commits may address it — untested) or if a Q4_0 quant of a native-MTP model makes the W4A4xMTP stacking test valid.

### 2d. Other closed programs
- Upstream sync: 32 commits merged, no regression (verified by interleaved same-day A/B — the apparent -4% was day drift)
- Speculative: native MTP > DFlash (both blocked by the same fork bug). DFlash retest v2 ran 2026-09-26 on the current binary (build-w4a4 @ aa6b4c5cb): base x2 + dflash x2 valid, greedy gate blocked by script bug (run() flag collision); greedy rerun + numbers extraction pending
- Perf pin: WASH (auto already ramps to those clocks)
- Gate threshold: 131072 confirmed optimal-enough
- Spec decode: native MTP > DFlash
- NCCL knobs: exhausted, defaults optimal

## 3. DECISIONS LEFT TO MAKE

| # | Decision | Depends on |
|---|---|---|
| D1 | **Deploy dense W4A4 gate=1** into production (merge perf/w4a4-dot8-gfx906 into build-rccl, add GGML_CUDA_W4A4_DOT8=1 per-model on Q4_0 dense entries) | Your call. Data is complete: +3.6% tg, +32% pp, PPL cost +0.082@8k (Mon) / +0.73@8k (Qwen). RP models (Mistral-Large fine-tune/Mistral-Large fine-tune) = tolerable; Qwen at 131k = significant |
| D2 | **Use gate=2 as quality mode?** (only if D1 lands) — env-selectable, not default; useful only where the -37% pp is acceptable and the +0.019 dPPL matters | D1 first |
| D3 | **Upstream PRs** — the nranks-3 auto-twoshot fix, the Q4_0 repack entry, and the dense-MXFP4 quantize mode are all upstream-quality (the author already shipped Q5_1, the Q4_0 twin) | Your time to submit |
| D4 | **Crash mitigation** — 2 hard power events (log-stop signature, NanoKVM USB cycling = rail brownout symptom); second PSU ordered by you; 175W cap in place. Verify stability under sustained long-context load after PSU install | New PSU arrival |

## 4. PATHS LEFT TO EXPLORE

| Path | Expected gain | Effort | Notes |
|---|---|---|---|
| **MXFP4 with imatrix-aware quantizer** | +3% decode, PPL cost reduced from +5.9% to near-zero (the current MXFP4 mode has no imatrix hook: quantize_row_mxfp4_ref ignores imatrix entirely) | Medium — quantizer patch + re-quant + full gauntlet | The M1 study's per-8 insight applies here too |
| **GEMM W4A4 optimization** (per-8 epilogue restructure) | Recover part of the -37% pp loss on gate=2 | High — the epilogue is FMA-bound, needs algebra change, not spill fix | Only worth it if you need the quality mode at speed |
| **W4A4xMTP stacking** (needs Q4_0 native-MTP quant) | Unmeasured — could be significant if MTP acceptance is high | Low if MTP bug fixed upstream | Blocked on: MTP fix (D: fork bug) + a Q4_0 native-MTP quant |
| **DFlash2 retest on current binary** | Unknown — the "loses to baseline" verdict predates the upstream merge and the MTP fixes | Low — the test script exists (`/root/dflash-retest.sh`, needs re-launch) | Same lossless-gate requirement |
| **BF16-wire AR** | Was a win pre-PLX, -1.2% post-PLX | — | Closed, don't re-test without a topology change |
| **The 3 GPU crashes** | — | — | 2 hard power events confirmed (log-stop, no kernel panic); 175W cap = the mitigation; second PSU ordered |
| **Long-context accuracy on RP models** | Mistral-Large fine-tune +0.078@8k, Mistral-Large fine-tune +0.19@8k with gate=1 | — | User-tolerable per your framing (RP models don't need deep precision); Qwen/Ornith are the critical ones |

## 5. OPEN BUGS / INCOMPLETE ITEMS

1. **MTP fork bug**: draft sampler falls back to CPU under -sm tensor (src/llama-context.cpp:1566-1577, the hard guard). Verify not strict-greedy-equivalent. Fix = either condition the guard or fix the offload path. Upstream may have addressed it.
2. **The 2.76x GEMM per-8 slowdown** is structural (epilogue FMA count), not spills. Would need algebra change to recover.
3. **MXFP4 imatrix hook**: quantize_row_mxfp4_ref ignores imatrix entirely (verified: no importance-matrix arg in its signature). The fix is a quantizer patch.
4. **DFlash2 retest**: driver script exists, needs re-launch post-PSU.
5. **2a's deferred harness re-run**: /tmp/u4_quant_check.hip multi-channel verification (was interrupted by the GPU pause, never re-run).

## 6. INFRASTRUCTURE NOTES

- 3 build trees on mi50: `/root/mx-Mistral-Large fine-tune/build` (perf/mxfp4-quant, W4A4 dev), `/root/mx-Mistral-Large fine-tune/build-rccl` (PRODUCTION, aa6b4c5cb since 2026-09-26 12:04), `/root/build-w4a4` (W4A4 testing)
- Standalone reference build: `/root/mx-premerge` (pre-upstream-merge, for day-drift control)
- All raw data: `/root/*-gauntlet*.txt`, `/root/m4-gauntlet-logs/`, `/root/ppl-logs/`, `/root/w4a4-profile*`
- Local workspace: `/home/brad/workspace/mx-Mistral-Large fine-tune/` (docs, BENCH.md, design docs)
- Gitea: root/mx-llama.cpp, branches gfx906-w4a4 (production docs) + perf/w4a4-dot8-gfx906 (all W4A4 work) + perf/mxfp4-quant
- Power: 175W/card via upp (persists reboots); perf high pinned; second PSU ordered
- Cron: `mxfp4-overnight-verdict` job fired and completed (job removed)
