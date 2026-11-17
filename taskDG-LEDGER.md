# taskDG LEDGER — verify-path graph replay: root cause of the verify token-graph "no-warm" claim + the 1,148 permanent disables

Date: 2026-11-16 (PST). Box: mi50. Branch `int4-dg-graphreplay` off INT4-noquant tip 27ca445502 (NOT pushed).
Commits: 7323699746 + 6f38be6cfe (GGML_GRAPH_DIAG instrumentation), 90b8caada1 (GGML_META_TG_INNER_PER_UID fix gate).
Build: /mnt/old/build-dg (mi50-build.sh, provenance 90b8caada1+scope-fix, TheRock amdclang++ pinned).
Pair: GPUs 1,3 (GPU0 burn-in untouched; GPU2 light-only). Champion baseline: f128q4kv-p-d4 58.20/58.33 (taskW3, pair 1,2), probe 77841cb6a251235d, acceptance 0.4377/2.7508 (3888/8883).

## 0. Why

taskW1 §1/§4b-2b ranked the verify-side graph pathology as the largest host residual: "token-graph
tg-replay only 40/3,394 steps; owner-off 12,384 + direct 8,648 @ ne11=5; 1,148 cuda-graph-unstable
permanent disables; 5,636 captures @ ne11=5 ≈ 1.65 re-captures/step × 280 µs ≈ 0.46 ms/step".
This lane: find why the verify token graph never warms at the constant verify shape, attribute every
disable, and ship the minimal gate if small.

## 1. PREMISE CORRECTION: the verify token graph DOES warm and replay

The W1 "tg-replay 40" is a rate-limit artifact: the `[DFLASH-PROF] tg-replay` line prints only the
first 8 + every 100th replay (ggml-backend-meta.cpp dprof block). The last line in the W1 log reads
`tg-replay uid=71046 count=3200` — the token graph replayed ~3,200 of 3,394 decode steps (94%).
Re-mining the W1 log (w1-meas-p-d4.server.log, 131,715 DFLASH-PROF lines):

- 56 `tg-capture ok=1` events (129/129 subgraphs covered, 1 run x 2 lanes, ~17 ms each), one per
  request; 40 tg-replay LINES but count reached 3,200; only 2 `tg-capture ok=0` permanent failures.
- The per-subgraph ne11=5 capture bursts are clusters of exactly 129 (= subgraphs per token) located
  at REQUEST boundaries (line-position buckets), not spread uniformly through decode. The "1.65
  re-captures/step" average is real arithmetic over a bursty, per-request phenomenon.
- mode histogram @ ne11=5 (dev0): replay 8,508 (incl. draft 3,291), outer 6,966 (= 56 captures x
  124 non-empty subgraphs), owner-off 6,192 (warm windows before the latch), direct 4,359, capture
  2,849. dev1 mirrors minus the draft.

So the target verify path is NOT "essentially no graph replay" — it replays 94% of steps. The real
costs are per-request re-warm/re-capture churn and the per-subgraph cache pathology below.

## 2. Mechanisms (code-level, all witnessed in the W1 log or read in-tree)

M1 — per-request token-graph re-capture (~17 ms x 56 ≈ 0.95 s ≈ 0.6 % of wall):
  `GGML_SCHED_STABLE_UID` split-uid hash (ggml_backend_sched_stamp_split_uids) pins EVERY node/leaf
  struct ptr, data ptr, buffer, ne/nb, full op_params, src edges and scratch-gen. Each request's
  prefill perturbs the allocator, the decode-shape graphs re-address, the hash changes, uid is new,
  `tg_lookup` misses -> re-warm (tg_warm_needed=2 runs, owner-off direct) + re-capture (~17 ms).
  `tg_cache_max=8` with FIFO eviction guarantees the decode entry cannot survive the per-request
  shape rotation (prefill chunks + decode widths) anyway.

M2 — INNER_ON_FAIL latch -> wasted warm-window captures (5,611 x ~249 us ≈ 1.4 s ≈ 0.9 % of wall):
  LATCH WITNESS (W1 log, line-indexed): before the first capture failure (line 54115) the ne11=5
  warm-window histogram is owner-off 12,384 / capture 25 (draft only); after it, owner-off 0 /
  capture 5,611, bursty at request boundaries and continuing to the last line of the run. The two
  chunked-prefill uids (34288, 52361; ne11=513/514, fast fail us≈500) tripped the latch ~40 % into
  the run. `inner_graphs_reenable()` latches per META CONTEXT (inner_graphs_reenabled),
  not per shape: from the first failure on, EVERY request's 2-run warm window dispatches through the
  per-subgraph CUDA graph cache — run 1 direct (warmup), run 2 CAPTURE (~280 us/subgraph) — and those
  captures never replay, because the token graph takes the shape over on run 3. The keys churn with
  the allocator (key = nodes[0] ptr + n_nodes + ne[1] + flags), so a fresh generation of ~129 keys
  is born at every allocator perturbation; each generation captures in its warm windows until it
  burns its 4-reset budget a few requests later: 975 distinct unstable keys, ~5.8 captures each.

M3 — the 1,148-class permanent disables (consequence of M2; direct cost ≈ 0):
  Per-subgraph entries see: uid changes every request (M1) -> the uid fast-path in
  `ggml_cuda_graph_update_required` misses -> full property compare -> cross-request pointer churn ->
  properties_changed -> reset. 4 resets within 4-7 checks trips `unstable_disabled` (permanent).
  W1 witness: 1,150 disable lines, all resets=4, checks in {4,5,6,7,9,20}, 975 distinct keys, bursty
  at request boundaries. After disable the warm window runs direct — which is exactly what the
  owner-off path would have done. The disables are the smoke; M2 is the fire.

M4 — the 2 failed prefill shapes: token-graph ineligible/capture-refused fast (us≈500, i.e. NOT
  ELIGIBLE or an early capture refusal, not a full 17 ms attempt). Exact refusal reason is captured
  by the diag leg (GGML_META_TG_DEBUG=1 prints REFUSED code / NOT ELIGIBLE / capture_begin failures).

## 3. Instrumentation (commit 7323699746, +6f38be6cfe; gate GGML_GRAPH_DIAG=1, default OFF)

- ggml-cuda.cu graph_compute: one `[GRAPH-DIAG] dev uid ne11 nodes mode reason resets checks` line
  per call; reason in {outer, owner-off, incompat, warmup-first, warmup-change, reset,
  unstable-disable, unstable, capture, replay, mem-fit, gpu-arch}. The first-changed-node attribution
  reuses the existing GGML_CUDA_GRAPH_DIFF=N probe.
- ggml-backend.cpp stamp_split_uids: `[GRAPH-DIAG] uidstamp ... changed=<class>` — whole-graph hash
  changes attributed to {scratch-gen, counts/flags, struct-ptrs, data-ptrs, shape, params+src}.
- ggml-backend-meta.cpp: tg-new / tg-evict / tg-warm / tg-replay lifecycle lines (evict exposes the
  tg_cache_max=8 FIFO pressure).
Gate-off = fprintf-only-when-on; 141 insertions, 2 deletions (one semantically-identical condition
restructure). Greedy probe gate-off/on identity is checked by the legs.

## 4. Fix A (commit 90b8caada1): GGML_META_TG_INNER_PER_UID=1 (default OFF)

Makes the lanes' per-subgraph cache state track the CURRENT shape's token-graph outcome
(`inner_cache_set(!tge->failed)` per token) instead of latching on the first failure anywhere:
warm-window dispatches of a shape the owner has not given up on run owner-off (no per-subgraph
capture churn, no warmup resets, no unstable-disable burn); shapes whose token graph failed keep
the cache (replay instead of ~0.5 us/node direct). Gate-off collapses to the old latching path
byte-identically. Kills M2 + M3. Does NOT touch M1 (the 17 ms/request re-capture) — that is Fix B.

## 5. Measurements

STATUS at ledger close (14:30): pair 1,3 held box-wide by the llama-swap serving model (user's
daemon, ignores gpu.lock, ttl not firing under health-check traffic; contract forbids co-scheduling
and the legs' VRAM<2GB gate refuses). No DG GPU leg ran in-session. The root cause above is
established from the W1 log (real witness, different config f0q4-p-d4, same tree/graph stack) +
code reading; the diag leg is confirmatory, not load-bearing.

Resumable watcher /root/dg-watch.sh (alive, window to ~16:47) auto-runs, when GPUs 1+3 drain <2GB:
dg-diag1 (champion f128q4kv-p-d4 pair 1,3, GGML_GRAPH_DIAG+DIFF+STATS+TG_DEBUG ON, full bench) ->
/root/dg-parse.py -> dg-diag1.parsed.txt (decision histogram, capture-us, uidstamp changed-classes,
reset-diff attribution, tg lifecycle incl. evictions); then paired-interleaved e2e legs
dg-off1/on1/off2/on2 -> /root/dg-e2e.py (tg medians vs 58.20, paired deltas, acceptance bit-equal
0.4377/2.7508 = 3888/8883, probe 77841cb6a251235d). Legs: /root/dg-leg.sh <tag> /mnt/old/build-dg
<diag|off|on>, flock -w 2700, production-env witnessed, TERM-only teardown, SKIP-EXISTS markers.

Honest ceiling: the addressable churn is ~0.46-0.9 ms/step (1-2 %) per the brief; Fix A removes the
per-subgraph capture churn (M2, ~0.9 % of wall concentrated where fresh keys churn) but NOT the
17 ms/request token-graph re-capture (M1). If the e2e delta does not clear the paired noise floor,
the verdict is sub-noise and Fix A ships as a hygiene gate (removes ~1.4 s startup churn +
hipGraphInstantiate memory pressure per session), with Fix B as the perf lever.

## 6. Fix B design (next lane): cross-request token-graph reuse (attacks M1, ~0.5-1 %)

B1 (preferred): key tg_entry on a shape hash EXCLUDING data pointers; on uid change with matching
   shape, patch the captured graph exec's baked pointers (hipGraphExecKernelNodeSetParams walk, the
   same trick ggml_cuda_graph_update_executable already uses per-subgraph) instead of re-capturing.
   Saves the ~17 ms/request capture and the 2-run warm window. Risk: pointer sets must be patched
   per-node (src data ptrs), and scratch-gen invalidation must stay a hard re-capture.
B2 (cheap probe first): raise tg_cache_max 8->64 + LRU. Only pays off together with B1 or allocator
   stability — alone it just keeps more cold entries.
B3: allocator-perturbation isolation — LLAMA_GRAPH_RES_PER_SHAPE already buckets arenas per shape;
   find why the decode arena re-addresses across requests (prefill arena growth vs pool coalescing)
   and pin it. The uidstamp `changed=` classes from the diag leg localize this.

## 7. Files

- /root/wt-dg-graphreplay (branch int4-dg-graphreplay): 3 commits on 27ca445502.
- /mnt/old/build-dg (BUILD-PROVENANCE.txt), /root/build-dg.log.
- /root/taskDG/: legs artifacts (witnesses, server logs, reps, agg, parsed), watch.log.
- /root/dg-{leg,watch,parse,agg,e2e}.py|sh; patchers /root/dg-patch.py, /root/dg-fixA.py.
- All GPU legs: flock /root/gpu.lock -w 2700, production-env witnessed, TERM-only teardown,
  pair 1,3 pinned, GPU0 untouched.

## §8 e2e verdict (parent, 2026-11-17): Fix A measured-NEUTRAL — gate stays available, NOT adopted
- Legs: off1/on1/off2/on2 paired-interleaved on pair 1,3 (post-reboot clean cards), champion f128q4kv-p-d4.
- tg256: OFF median 57.34 vs ON median 57.31 = -0.06% — SUB-NOISE (arm spreads 0.03/0.05%). The
  ~0.5-1% mechanism-level ceiling did not translate to e2e; capture churn is real but its wall
  share is smaller than modeled.
- Probes: canonical 77841cb6a251235d all 4 legs (Fix A gate-off-identical confirmed e2e).
- Acceptance: 3/4 legs bit-equal 0.4377/2.7508; off2 shows 0.4380/2.752 (3889/8879) — matches the
  KNOWN pinned-arm one-off acceptance nondeterminism class (K1/V3 flagged, tech-exp prompt), and
  it occurred in an OFF arm, so it is not attributable to Fix A. Logged, not new.
- DISPOSITION: GGML_META_TG_INNER_PER_UID stays shipped default-OFF, available for entries with
  long-lived sessions where churn dominates; not in the serving config. Fix B (exec-pointer
  patching) remains the designed follow-on if the M1 per-request re-capture ever matters.
