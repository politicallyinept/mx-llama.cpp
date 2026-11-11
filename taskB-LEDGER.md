# taskB LEDGER - DFlash2 step-cost instrumentation (non-GPU authoring)

Worktree: /root/int4-noquant-wt, branch INT4-noquant, base fb18241213.
Hard rules honored: no GPU work, no cmake/make build, no touching /llama-swap.

## 2026-11-10 (this task)
- [done] Mapped dflash/spec-decode path (full file:line map in
  PROF-PLAN-dflash-step.md section 0). Key sites:
  - drafter forward: common/speculative.cpp:1583 draft(), llama_process(ctx_dft) ~1633
  - fused encode+inject + gather + tgtsync: process() ~1440-1580
  - verify batch build: tools/server/server-context.cpp:604-640; target forward decode() 4308/4348
  - accept/rollback + ckpt restore: server-context.cpp:4580-4700
  - snapshot buffers: common/common.h:1278, common/common.cpp:2424-2492
  - HIP per-subgraph graphs: ggml-cuda.cu:6884 entry, key 3911-3925, unstable-disable 6915-6960
  - token graph: ggml-backend-meta.cpp:5410-5760 (tg_warm_needed=2, permanent disable on fail)
  - nc narrow kernel widths 6-8: ggml-cuda/norm.cu:444-533
- [done] Authored GGML_DFLASH_PROF=1 instrumentation (default-off, function-local
  static getenv, [DFLASH-PROF] banner): draft-fwd/draft-inject/draft-gather,
  tgt-decode width witness, cuda-graph per-device ne11 witness + mode + us,
  cuda-graph-unstable, tg-capture/tg-replay/tg-not-eligible/tg-invalidate,
  ckpt-update/load with ptr + distinct-pointer churn counters.
  Files touched: common/speculative.cpp (4 hunks), common/common.cpp (4),
  tools/server/server-context.cpp (1), ggml/src/ggml-cuda/ggml-cuda.cu (3),
  ggml/src/ggml-backend-meta.cpp (7). 177 insertions, 1 deletion.
- [done] Syntax verification WITHOUT building the project: g++ -fsyntax-only
  passes on common/common.cpp, common/speculative.cpp,
  tools/server/server-context.cpp, ggml/src/ggml-backend-meta.cpp.
  ggml-cuda.cu hunks hand-reviewed only (HIP TU, not compiled per pause rule).
- [done] PROF-PLAN-dflash-step.md written (map, preconditions, build, cells,
  greps, cross-checks, teardown, exit criteria).
- [done] Committed on INT4-noquant (see below), no push.

## GPU-QUEUED (do NOT run until gates pass)
1. Build: see PROF-PLAN section 2. Gate: MemAvailable >= 12 GB, flock /root/gpu.lock,
   one model load in flight, PID-verified TERM teardown.
2. Run cells A-D per PROF-PLAN section 3, structured + prose prompts, >=3 requests/op.

## State at handoff
- Working tree clean except untracked taskB-LEDGER.md.
- No processes started, no GPU touched.

## Commit SHAs
- df1c818125 instrumentation (GGML_DFLASH_PROF)
- a28b5143ef PROF-PLAN-dflash-step.md
- branch INT4-noquant, not pushed
