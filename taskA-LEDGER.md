# Task A ledger - ruquant spec-decode redo preparation (non-GPU)

Rules: append, never rewrite. Long children die to infra restarts; this file is the state.

## 2026-11-10 (evening) - session 1

- Branch verified: git -C /root/int4-noquant-wt rev-parse --abbrev-ref HEAD = INT4-noquant.
- Box state at start: MemAvailable ~40 GB, loadavg 0.28; llama-swap fleet UP (PID 316605,
  port 8181, GPU1/GPU2 busy). No GPU work run. No builds run.
- Asset audit complete (all paths/sizes/sha256 recorded in RUN-CARD section 1):
  * rq W4A4 target + CONVERTED W4A4H drafter present under /root/bench-quants/ruquant/,
    both sha256-match published SHA256SUMS. Converted drafter carries ruquant.drafter KV.
  * stock Q4_0 (unsloth) + z-lab DFlash2 Q8_0 drafter present (fork arm ready except build).
  * /root/ruquant-src built @ 7a08316; /root/ruquant-repo @ 4959bd7; rig/final_mi50.sh,
    rig/acc_bench.py read; exact flags + -devd semantics extracted from docs/source.
  * /root/int4-noquant-wt/build ABSENT -> GAP G1 (build queued, command in card F0).
- Root-cause verdicts on the 49.5 t/s run (evidence in card section 2):
  (a) runtime mismatch: RESOLVED (TheRock-first in launchers; 7.15.0a20260702 installed)
  (b) drafter unpinned: CONFIRMED (cell_p2.sh has no -devd -> replicated -> -19%)
  (c) unconverted drafter: REFUTED (rq-spec.server.log:24 loads the converted W4A4H file)
  (d) missing sampling flags: CONFIRMED (no -bs --samplers temperature in cell_p2.sh)
  (e) NEW: 0.230/1.46 acceptance = bench.py artifact (called with draft_n_max=2 vs a
      draft-7 server); server logs show 0.636/5.43 and 0.307/3.11. W5 = authoritative.
- RUN-CARD-ruquant-spec-redo.md authored + predicates verified on box (real outputs pasted
  in Appendix A). scripts/rq_gguf_kv.py added (rotation-metadata witness tool).
- Committed run-card + ledger + kv script (see commit lines below). No pushes.
