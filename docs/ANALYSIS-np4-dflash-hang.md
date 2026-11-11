# ANALYSIS: np4 + draft-dflash d7 "soft-hang" (legB cell C)

Date: 2026-11-11 (PST). Analysis-only task: read-only log/code inspection, no GPU,
no builds. Branch INT4-noquant at /root/int4-noquant-wt, HEAD 21770741eb (verified
via git rev-parse). Data: /root/ruquant-bench/redo/legB-C.server.log (44,719 lines,
44,053 of them [DFLASH-PROF]), compared against legB-A/-B/-D.server.log (same build).

## 1. Bottom line

The server did NOT hang. Every task the server ever received completed cleanly, and
the server sat idle for 72 minutes until an external SIGTERM. The cell-C-only
"concurrency volley" never produced concurrent slots: 3 of its 4 request bodies were
malformed JSON (empty prompt field) and were rejected with HTTP 400 at the HTTP
layer, so slots 0/1/2 never received a task. The cell hung in the DRIVER, not the
server: the bare `wait` at scripts/leg_b_prof.sh:87 waits for ALL background children
of the script, including the setsid'd llama-server itself (launched at
scripts/leg_b_prof.sh:59, `SRV=$!` at :63), which never exits on its own.

Answer to the posed question "why does np4 + dflash d7 stall while np1 works": it is
not np4 and not DFlash2. Cells A/B/D (np1) skip line 87 entirely (the volley block is
`if [ "$CELL" = C ]`); cell C is the only cell that executes a bare `wait`, and that
is the only difference that matters. The np4 + DFlash2 concurrent path was never
exercised by this capture.

## 2. Per-slot timeline (legB-C.server.log, line numbers in file)

Log timestamps are elapsed H.MM.SS.mmm since server start (0.34.669 = 34.669 s;
72.54.318 = 72 min 54 s).

    t=0.17.849  listening on :8615 (n_slots = 4, n_ctx_slot = 1024)
    t=0.20.058  W srv operator(): got exception: 400 parse_error.101 col 11 (x5, to 0.20.081)
    t=0.20.086  L322-323  slot get_availabl: id 3 | selected slot by LRU, t_last = -1
                          slot launch_slot_: id 3 | task 0
    t=0.25.389  L15111-15117  print_timing id 3 task 0 (256 tok, 58.84 t/s, draft acc 0.58960)
                          slot release: id 3 | task 0 | stop processing: n_tokens = 301
    t=0.25.395  1x 400 parse error
    t=0.25.605  L15119-15120  launch_slot_: id 3 | task 54
    t=0.30.139  L29907-29913  print_timing + release id 3 task 54
    t=0.30.145  3x 400 parse error (the malformed members of the "volley")
    t=0.30.146  L29917-29918  launch_slot_: id 3 | task 108
    t=0.34.669  L44707-44713  print_timing + release id 3 task 108
                          "slot      release: id  3 | task 108 | stop processing: n_tokens = 301, truncated = 0"
    ---- 4357 s of ZERO log lines ----
    t=72.54.318 L44714  "srv    operator(): operator(): cleaning up before exit..." (SIGTERM)
    t=72.54.xxx L44715-44718  meta-prof + graph_stats teardown dumps

Slot/task census (all four cells, same method):

    cell  np  slots ever launched   tasks (launch->release)          400 parse errors
    A     1   slot 0 only           0, 54            both complete  6
    B     1   slot 0 only           0, 78            both complete  6
    C     4   slot 3 ONLY           0, 54, 108       all complete    9
    D     1   slot 0 only           0, 98            both complete   6

Slots 0/1/2 in cell C have ZERO lines: no get_availabl, no launch_slot_, no
print_timing, no release. They never "failed to complete" - they were never given
work. Slot 3 did start second and third requests (tasks 54, 108) and completed both.

Instrumentation counters agree with three solo requests and nothing else:
tgt-decode call=159, draft-gather call=159, draft-inject call=159 (last:
`draft-inject call=159 n_tokens=1 us=723 rc=0`, L44706), draft-fwd call=150,
ckpt-update-tgt call=1..3 all `seq=3` (one per request, size=156894364, 70-150 ms
each, L589/L15384/L30182). The last three decode steps of task 108 are visible:
`tgt-decode call=157 width=8`, `call=158 width=4`, `call=159 width=1` (L44155,
L44436, L44701) - a normal wind-down to the final token, then print_timing/release.

Idle proof: between the release (t=0.34.669.345) and the SIGTERM cleanup
(t=72.54.318) there is not one line of output, although the build emits a
[DFLASH-PROF] cuda-graph line for every graph compute (43,410 such lines during the
~15 s of actual work). The teardown meta-prof states "compute=0.1% sync=0.0% of
4356.9 s wall" (L44715): the process spent 99.9% of wall time not computing. That is
an idle event loop sleeping in its wait, consistent with 0% CPU / S-state and with
the clean SIGTERM shutdown (queued statistics printed, graph_stats dumped, exit).

## 3. Why the volley never reached the scheduler

Driver: /root/int4-noquant-wt/scripts/leg_b_prof.sh.

- :77-78 extract prompts as `ps[0]['prompt']` / `ps[3]['prompt']` from
  /root/mtp-natural/prompts.json. The file's entries use the key `text_pp` (verified
  live: keys are name/category/text_pp/...), so python raises KeyError, stdout is
  empty, and $P1/$P2 are empty strings.
- :80 builds the body as `{"prompt":$1,"n_predict":256,...}`. With empty $1 the body
  is `{"prompt":,"n_predict":...}` -> json parse error at line 1 column 11, exactly
  what the 9 "got exception" lines report (logged at tools/server/server-http.cpp:196).
- Request accounting: 8 sequential + 4 concurrent = 12 requests = 3 valid (P3 only:
  tasks 0, 54, 108) + 9 malformed (6 sequential + 3 of the 4 concurrent). The volley
  at :87 (`req "$P1" & req "$P2" & req "$P3" & req "$P1" & wait`) therefore launched
  exactly ONE server task (108) and three instant 400s (t=0.30.145.732/.742/.878).

Side note on the task-id gaps (0 -> 54 -> 108): the id counter is shared and every
HTTP request consumes ids, including the per-scrape METRICS tasks that /metrics
generates at tools/server/server-context.cpp:5419. External monitoring scraping the
port at ~10-20 Hz accounts for the ~53-54 id gap per ~5 s request; no requests were
lost to the gap. The tg-debug `uid` values (up to 7302) are cgraph uids from
ggml/src/ggml-backend-meta.cpp:5471, unrelated to requests.

## 4. Where the cell actually hung: the bare `wait`

- scripts/leg_b_prof.sh:59-63: `setsid nohup env ... llama-server ... &` then
  `SRV=$!`. In a non-interactive shell (no job control) the backgrounded child is not
  a process-group leader, so setsid(1) does NOT fork: llama-server runs as the direct
  child with PID $SRV. The script's own `kill -0 "$SRV"` health loop and the
  `kill -TERM "$SRV"` teardown (which worked cleanly in cells A/B/D) confirm $SRV is
  the server process itself.
- scripts/leg_b_prof.sh:87 ends with a bare `wait`. Bash `wait` with no arguments
  waits for ALL active background jobs of the script - the four curls AND the server.
  The last curl returned at t=0.34.669 (task 108 response sent: send_final_response
  runs before release, tools/server/server-context.cpp:4582-4584), but the server
  child never exits, so `wait` blocked forever.
- Wall-clock corroboration: legB-C.log shows start 02:32:24, DONE 03:45:24 (73 min);
  the server's cleanup line is at +72:54. The two "draft acceptance" grep lines and
  the VRAM lines appear in legB-C.log only after +72 min - i.e., the script resumed
  exactly when the server died. The SIGTERM at +72:54 came from outside the driver
  (campaign supervisor/operator); the driver's own `kill -TERM "$SRV"` was then a
  no-op and the teardown loop exited immediately with a clean DONE.
- Cells A/B/D never reach line 87, so they tore down normally ~10 s after the last
  request despite the same empty-prompt breakage.

## 5. np>1 spec-decode orchestration review (for the record)

Read on this branch (all file:line at HEAD 21770741eb):

- Batch build: each generating slot appends its sampled token plus its own draft
  tokens to the ONE shared batch and records row indices in its own spec_i_batch
  (tools/server/server-context.cpp:605-647, handle_last_sampled_token; spec_i_batch
  is a per-slot vector, asserted empty before refill).
- Draft dispatch: the pre-decode loop collects every slot needing a fresh draft into
  `drafting` (server-context.cpp:3462 generating, :3514 drafting.push_back) and calls
  common_speculative_draft() ONCE for all of them (:3525). Per-slot draft params are
  passed via common_speculative_get_draft_params(spec, slot.id) (:3503-3513), keyed
  by slot id.
- speculative.cpp draft(): flushes the shared deferred-prompt queue first (:1605-1609,
  on failure it clears dp.drafting for ALL params and skips drafts - a soft skip, not
  a block), then builds ONE ctx_dft decode containing every drafting sequence's noise
  block (i_block_beg/n_block per seq, :1613-1650), then reads the DFlash2 selector
  lattice per seq (:1685+). is_dflash2 = selector_top_k > 0 (:1034, :1101). There is
  no np>1-specific DFlash2 branch; the multi-seq path is the normal path.
- Accept/rollback is per slot: n_rollback computed per slot (:4637), checkpoint
  restore via slot.spec_ckpt.load_tgt/load_dft keyed by slot.id
  (server-context.cpp:4646-4663, common_speculative_accept at :4677). Checkpoints are
  per-slot (slot.spec_ckpt; wrappers llama_state_seq_get/set_data_ext at
  common/common.cpp:2443/2478/2513/2547), keyed by seq id = slot id. The
  ckpt-update-tgt lines (seq=3, distinct ptr per request) confirm per-slot capture.
- release() (server-context.cpp:648) only touches the releasing slot's state; with
  async checkpoints it first llama_synchronize(ctx_tgt). Slot 3's release could not
  strand slots 0/1/2 - they had no pending spec state because they had no tasks.
- The only genuinely SHARED cross-slot state found in the DFlash2 drafter is the
  deferred prompt-encode queue `pending` (common/speculative.cpp:1055-1061), a single
  vector of {seq_id, p0, n, epoch} chunks appended at :1458 and replayed in insertion
  order by flush_pending_upto (:1253-1355). Its window-trimming optimization is
  explicitly guarded to `n_seq == 1` (:1274), showing the authors treated multi-seq
  pending as a special case. A multi-slot volley interleaves pending chunks across
  seqs; the lag-one flush gate (:1502) can then defer replays until draft() forces a
  full flush (:1605). This is the one place with a plausible np>1 hazard, but it fails
  soft (skip drafts / LOG_ERR), and it was never exercised here.

## 6. Mechanism hypothesis (ranked)

H1 (primary, proven by the evidence above): driver-side hang. Slot 0/1/2 block on
nothing - they were never launched because 3 of the 4 volley bodies were invalid JSON
(empty $P1/$P2 from the wrong prompts.json key). The condition that "never becomes
true" is bash's `wait` condition "all background children have exited": the setsid'd
server child (PID $SRV, scripts/leg_b_prof.sh:59-63) never exits, so the script
blocks at :87 forever; the server is idle-but-healthy and dies cleanly to the
supervisor's TERM.

H2 (latent, unproven, NOT exercised by this capture): a real np>1 DFlash2 defect in
the shared deferred encode path (speculative.cpp:1055-1061, 1253-1355, 1458, 1502) or
in multi-seq noise-block drafting (:1613-1650) when several slots draft/verify
concurrently. Cannot be confirmed or refuted from this log because true concurrency
never happened.

H3 (rejected): response-delivery stall (slot released but curl never got the body).
send_final_response precedes release (server-context.cpp:4582-4584); the sequential
requests demonstrably returned ~6 ms after release (release 0.25.389.524 -> next
request 0.25.395.742); and curl -m 900 caps any client-side hang at 15 min, not 72.

H4 (rejected): allreduce/SMC wedge. 0% CPU S-state, zero GPU use, zero log output,
and a clean responding SIGTERM are the opposite signature.

Evidence that would DISPROVE H1:
- Any log line (slot, [DFLASH-PROF], graph_stats) between 0.34.669.345 and
  72.54.318 - there are none.
- launch_slot_ lines for slots 0/1/2 with no matching release - none exist.
- Proof the +72:54 TERM was the driver's own kill (i.e., `wait` returned on its own);
  the ordering in legB-C.log (grep output appears only after the server died) points
  the other way.
- A rerun where setsid forks (e.g., under an interactive/job-control shell) so $! is
  a short-lived intermediate and `wait` returns at +34.7 s - if the cell then still
  stalls, look again at the server.
- Well-formed volley bodies (e.g., prompts extracted via the text_pp key) that still
  produce slots 0/1/2 launches that never complete - that would be H2 surfacing.

## 7. Cheapest confirming GPU probes (config-only, no code changes)

All probes reuse the leg_b_prof.sh pattern verbatim: G2 load gate, `exec 9>/root/gpu.lock;
flock -w 1800 9`, MemAvailable >= 12 GiB gate, HIP_VISIBLE_DEVICES from the free-GPU
probe, same env exports, setsid nohup launch, TERM-only teardown with the 24x5 s wait.
$M and $D are the target/draft model paths at scripts/leg_b_prof.sh:12-13; the exact
cell-C command line is preserved in /root/ruquant-bench/redo/legB-C.witness.

P0 (cheapest, directly tests H1 - driver fix only, same server config as cell C):
  fix prompt extraction to the real key (`ps[i]['text_pp']`) and replace the bare
  `wait` at :87 with explicit curl pids:
      req "$P1" & c1=$!; req "$P2" & c2=$!; req "$P3" & c3=$!; req "$P1" & c4=$!; wait $c1 $c2 $c3 $c4
  Server line unchanged (cell C witness). Expect: four launch_slot_ lines on slots
  0-3 (or fewer slots if prompts are short), cell DONE in ~2 min, no TERM needed.
  If it completes -> H1 confirmed and the np4 DFlash2 concurrent path is healthy.
  If slots launch and then stall -> H2 is real; capture /proc/<pid>/task/*/stack.

P1 (width threshold, np4 + n-max 3):
      $BIN/llama-server -m "$M" -md "$D" --host 0.0.0.0 --port 8617 -ngl 99 -fa on \
        -sm tensor --fit off -c 4096 -ub 512 -t 16 -np 4 -lm dio \
        --spec-type draft-dflash --spec-draft-n-max 3
P2 (slot-count threshold, np2 + n-max 7):
      $BIN/llama-server -m "$M" -md "$D" --host 0.0.0.0 --port 8618 -ngl 99 -fa on \
        -sm tensor --fit off -c 4096 -ub 512 -t 16 -np 2 -lm dio \
        --spec-type draft-dflash --spec-draft-n-max 7
P3 (drafter-family control, np4 + MTP d2):
      $BIN/llama-server -m "$M" --host 0.0.0.0 --port 8619 -ngl 99 -fa on \
        -sm tensor --fit off -c 4096 -ub 512 -t 16 -np 4 -lm dio \
        --spec-type draft-mtp --spec-draft-n-max 2
Each probe must be driven with the P0 volley (valid prompts, explicit-pid wait);
otherwise it reproduces the driver hang instead of testing the server.

## 8. Key evidence quotes

    L317    0.20.058.510 W srv operator(): got exception: {"error":{"code":400,
            "message":"[json.exception.parse_error.101] parse error at line 1, column 11:
            syntax error while parsing value - unexpected ','; ...
    L323    0.20.086.669 I slot launch_slot_: id  3 | task 0 | processing task, is_child = 0
    L15117  0.25.389.524 I slot      release: id  3 | task 0 | stop processing: n_tokens = 301, truncated = 0
    L29918  0.30.146.151 I slot launch_slot_: id  3 | task 108 | processing task, is_child = 0
    L44701  [DFLASH-PROF] tgt-decode call=159 width=1 us=59990 inner_us=59946
    L44706  [DFLASH-PROF] draft-inject call=159 n_tokens=1 us=723 rc=0
    L44713  0.34.669.345 I slot      release: id  3 | task 108 | stop processing: n_tokens = 301, truncated = 0
    L44714  72.54.318.444 I srv    operator(): operator(): cleaning up before exit...
    L44715  [meta-prof] ... totals: compute=0.1% sync=0.0% of 4356.9 s wall
