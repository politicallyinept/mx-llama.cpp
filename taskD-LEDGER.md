# Task D: root-cause np4 + draft-dflash d7 soft-hang (legB cell C)

Date: 2026-11-11 (PST). Read-only analysis: log mining + branch source review.
No GPU, no builds, no process actions. Branch INT4-noquant, HEAD 21770741eb.

## Verdict
The server never hung. Full write-up: docs/ANALYSIS-np4-dflash-hang.md.

- Only slot 3 ever launched tasks (0, 54, 108); all three completed with full
  print_timing + release. Slots 0/1/2 received ZERO tasks: 3 of the 4 concurrent
  volley bodies were malformed JSON (driver read prompts.json key "prompt",
  file uses "text_pp" -> empty $P1/$P2 -> 400 parse_error.101 col 11, 9x).
- After the last release (t=0.34.669.345) the log is silent for 4357 s until an
  external SIGTERM; teardown meta-prof: compute=0.1% of 4356.9 s wall -> idle.
- The cell hung in the driver: cell-C-only bare `wait` (scripts/leg_b_prof.sh:87)
  also waits on the setsid server child (PID $SRV, :59-63), which never exits.
  Cells A/B/D skip line 87 and tear down cleanly with the same broken prompts.
- np4 + DFlash2 concurrency was therefore never exercised; no server-side defect
  demonstrated. Latent np>1 review: per-slot spec state is clean
  (server-context.cpp:605-647, 4637-4677; per-slot checkpoints common.cpp:2443-2547);
  the only shared cross-slot drafter state is the deferred prompt-encode queue
  (speculative.cpp:1055-1061, 1253-1355), which fails soft, not hard.

## Probes proposed (config-only)
P0 driver fix + same cell-C server config (primary); P1 np4/n-max 3 :8617;
P2 np2/n-max 7 :8618; P3 np4 MTP d2 :8619. Exact lines in the analysis doc sec 7.
