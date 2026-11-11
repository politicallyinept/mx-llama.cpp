# RESULTS: ruquant speculative-decode redo (2026-11-11, mi50 TP2)

Session: post-SMC-power-cycle box (uptime 21 min at start), GPUs 0,1 (DEVS=0,1),
llama-swap fleet unloaded, RAM >= 40 GB avail throughout. Drivers:
scripts/leg_t.sh / leg_r.sh / leg_f.sh per RUN-CARD-ruquant-spec-redo.md
(gates G1-G5, witnesses W1-W7, TERM-only PID-verified teardown, flock per leg).
Logs + raw reps: /root/ruquant-bench/redo/leg{R,F}-r{1,2}.* and legT-*.

## Verdict

1. ruquant's published speculative number REPRODUCES on our rig with their recipe
   plus the two fixes (-devd ROCm0, -bs --samplers temperature):
   their acc_bench: 4.25 tok/step, 94.2 t/s, 45.1-45.2 ms/step (r1 and r2)
   vs published 89.5 t/s @ 4.27 acc / 47.7 ms. Reproduced at ~105%.
   The historical 49.5 t/s / 0.230-acc figure decomposed into: (a) missing -devd
   (documented 19% drafter-replication penalty), (b) missing -bs/--samplers,
   (c) bench.py derived-acceptance math dividing by DRAFT_N_MAX=2 against a d7
   server (the server's own lines always showed healthy acceptance, e.g.
   0.636/5.43 in the 2026-11-05 run).
2. Acceptance is workload-mix-dependent, not a hardware gap: code 6.7 acc /
   math 5.85 vs narrative 1.8-1.9 on the natural-prose suite. Published
   aggregates assume their benchmark mix (acc ~4.3).
3. Our fork's DFlash2 (stock Q4_0 + z-lab Q8_0 drafter, d4, replicated) does NOT
   reach their stock TP2 baseline: 37.8/38.0 t/s vs published 60.5 @ 3.68.
   Acceptance is PARITY on the same suite (fork 2.47/2.39 vs rq 2.42 pooled) -
   the whole gap is step cost.
4. Step-cost decomposition (server statistics, same ~3.8k verify calls/session):
   verify pass: fork 44.6 ms/call at width 5 vs rq 33.8 ms/call at width 8
   (per drafted token ~2x worse; cross-checks leg T: ours 55.0 ms @ w8,
   rq 33.6 ms @ w8 = their published 35.1 REPRODUCED).
   draft pass: fork 3.56 ms/draft-token vs rq 0.88 (4x worse; replicated Q8_0
   drafter vs their pinned W4A4H drafter).
   => TWO levers, both format-free: (i) the narrow/verify kernel (items 3-4 of
   the external brief; DESIGN-i8narrow-kernel.md + CPU seal already on this
   branch), (ii) the drafter itself (4-bitize or pin; ~4x per-token headroom)
   which the brief did not isolate.

## Numbers (medians of r1/r2; full per-prompt tables in the agg files)

| arm | pp4096 | tg256 median | pooled acc | notes |
|---|---|---|---|---|
| rq W4A4 + DFlash2 d7 (-devd, -bs) | 653.1 | 50.8 | 2.42 | their harness: 4.25 acc / 94.2 t/s |
| fork Q4_0 + DFlash2 d4 (replicated) | 553.5 | 37.9 | 2.39 | fills the missing stock cell |
| rq published stock Q4_0 d4 | - | 60.5 | 3.68 | RESULTS.md TP2 column |
| 8-token verify forward (diag) | ours 55.0 ms | rq 33.6 ms | | published 62.8 / 35.1 |

NOTE: the external brief's "62.6 t/s two cards" stock baseline remains
untraceable in ruquant's repo; published figure is 60.5 and is what the fork
cell is measured against.

## Leg-level facts

- rq r1/r2 tg256 medians 50.8/50.7, pp 653.1/652.7; acc_bench all-arm identical
  (4.25 / 94.2 / 45.1-45.2 ms) - engine and config fully deterministic across reps.
- fork r1/r2 tg256 medians 37.8/38.0, pp 553.7/553.4; server acc 2.47/2.39.
- rq p6 llama-bench aborted once at -r 8 inside ggml_backend_cuda_comm_allreduce_tensor
  (their engine) and WEDGED THE SMC (host power cycle required, documented in
  references/foreign-fork-h2h.md of the gfx906-w4a4-program skill). Standing rule
  adopted: no higher -r reruns of their engine without a fresh single-cell probe;
  any abort in their comm path ends the session for a wedge check.
- Fork build used: INT4-noquant @ c2ec8b678a (incl. GGML_DFLASH_PROF instrumentation;
  profiling env was NOT set for these legs - clean baseline).
