# task M4 LEDGER - bench-infra incidental defects (M2 findings #5 and #6)

Branch: INT4-noquant @ c4bc5c172f (worktree /root/int4-noquant-wt)
Date: 2026-11-11. Scope: root-cause + fix defect A (post-TERM teardown GP
fault) and defect B (mtp_bench.py 'w'-mode silent overwrite). No pushes.

## Defect A: llama-server GP fault in libc during post-TERM teardown

### Root cause (PROVEN, reproduced)
The bench drivers start the server as `setsid nohup timeout 14400 llama-server
...` and tear down with `pkill -TERM -f "llama-serve[r].*--port $port"`. The
-f pattern matches BOTH the `timeout` wrapper (its cmdline embeds the
llama-server path) and the server itself; `timeout` also forwards the TERM it
received to its child. Result: llama-server gets TWO SIGTERMs within
milliseconds. Every crashed M2 arm log ends with the tell-tale line
"Received second interrupt, terminating immediately." (server.cpp signal_handler
second-hit branch).

The second-hit branch called exit(1) FROM THE SIGNAL HANDLER while the main
thread was still in teardown (35B model free takes seconds). exit() is NOT
async-signal-safe: __run_exit_handlers then runs reentrantly on the interrupted
thread and races the main thread's own teardown/exit -> corrupted atexit
(exit_function_list) nodes.

Fault site symbolication: dmesg reports ip at offset 0x1f98b from the exec-VMA
base; the exec segment maps at file vaddr 0x28000, so the true libc vaddr is
0x4798b = __run_exit_handlers+0xeb (local symbol; between exported
erand48_r@0x478a0 and exit@0x47ba0 in libc6 2.39-0ubuntu8.7). Disassembly at
the fault ip is the exit_function_list walk (32-byte nodes, ef_on/ef_is/
ef_thread flavor tests 2/3/4) dying on `mov 0x10(%rax),%rcx` with a wild node
pointer.

Two observed failure modes: (1) GP fault in libc (dmesg signature identical
across all M2 arms and both of my GPU repro hits); (2) futex deadlock - handler
interrupts a thread holding the exit/malloc lock, exit(1) re-acquires -> process
hangs in futex_wait holding VRAM (2 of 12 repro trials; required SIGQUIT to
clear, TERM no longer serviced). Kernel teardown reclaims VRAM either way,
which is why bench data was unaffected.

### Repro (GPU2, under flock, tiny model - no 35B needed)
/root/m4-repro2.sh + /root/m4-repro3.sh + /root/m4-repro4.sh: stories15M /
tiny-Q4_0 served on GPU2 with the production-env bundle, torn down with the
harness pattern (pkill double-match) or double `kill -TERM` at 1-80 ms spacing.
Hit 2x GP fault (dmesg [3552.6], [3835.0], same ip offset 0x1f98b) + 2x futex
deadlock in ~12 trials. Single TERM never crashes (m4-repro.sh: clean rc=0).

### Not a branch regression
The identical signal_handler/exit(1) code exists in the stock tree
(/root/upstream-llama.cpp/tools/server/server.cpp:28-37). Upstream wart
amplified by the harness double-TERM, not an INT4-noquant/gfx906 defect.

### Fix (committed on INT4-noquant)
tools/server/server.cpp: second-hit branch exit(1) -> _exit(1)
(async-signal-safe; matches the "terminate immediately" intent; kernel reclaims
mappings + GPU buffers via driver release). Incremental rebuild with
/root/w4a4-env.sh toolchain: bin md5 895219cb739b4867411a383acf096eae (M2-
recorded) -> df6c4ffacb1cb4d78e0131e53ad74788.

### Verification
/root/m4-verify.sh (CPU-only stories15M, same libc exit path): 12 double-TERM
trials (spacing 1-80 ms): 4/12 logged "second interrupt" (fixed path exercised)
-> all exited immediately; 0 GP faults, 0 stuck processes, single-TERM path
unchanged (clean rc=0 in m4-repro.sh).

### Harness-side recommendation (not committed - drivers are campaign artifacts)
- m2_i8narrow_ab_v3.sh:100 / cell.sh:108: pkill/pgrep -f "llama-serve[r]..."
  also matches the `timeout` wrapper -> double TERM. Match the binary path
  only, or use `timeout --foreground`, or signal the recorded child PID.
- Retry loops that re-run mtp_bench.py on the same jsonl (v3:79-85,
  gpu_ab_36.sh:69-73, m2_i8narrow_ab.sh:66-70) must `rm -f` the partial file
  before retrying now that mtp_bench refuses to clobber (fail-loudly is the
  intended new behavior).

## Defect B: mtp_bench.py silent jsonl overwrite

### Fix
/root/mtp-final/mtp_bench.py (backup: mtp_bench.py.pre-M4fix.bak):
open(out_path, "w") -> open(out_path, "x") (atomic O_EXCL: REFUSE if the path
exists, with a clear REFUSING message) plus an explicit --append opt-in for
intentional accumulation. Same fix applied to the identical copy at
/root/harnessfix/mtp_bench.py (backup alongside).

### Verification
Stubbed-HTTP harness ran the real script 3x: run1 fresh path -> 6 records;
run2 same path -> REFUSED, existing data untouched; run3 --append -> 6 more
records appended (12 total, both cells intact).

### Sweep for the same 'w'-mode pattern
- /root/moe-nextn/gpu_ab.sh:34 and m2_i8narrow_ab_v2.sh:79: ALL arms share one
  jsonl (ab-results.jsonl / m2-results.jsonl) - the exact flagged collision;
  now REFUSE loudly via the fixed mtp_bench (v3 already moved to per-arm paths).
- /root/harnessfix/mtp_bench.py:69 - same script, fixed (above).
- /root/compare-1101/bench.py:91, analyze.py:74, /root/moe-nextn/
  verify_graft.py:134, scripts/compare-logprobs.py:54/116,
  scripts/server-bench.py:66, scripts/create_ops_docs.py:180,
  scripts/sync_vendor.py:124, scripts/w4a4-outlier-analysis.py:127 - single-run
  outputs/snapshots or per-port logs, not multi-arm result files; left as-is.
  scripts/tool_bench.py:242 already has an explicit append flag;
  compare-1101/bench.py:84 uses tmp+os.replace (atomic); gpu_ab_36.sh:102 is a
  deliberate aggregation step. No change needed.

## Rule compliance
GPU touches (3 repro drivers) each inside one `flock -w 600 /root/gpu.lock`
window, GPU2 only (HIP_VISIBLE_DEVICES=2), tiny/small models only, curl /health
polling, no bare wait, TERM-only teardown (SIGQUIT used ONLY on two
already-deadlocked repro processes of this task that could no longer service
TERM and were holding 2.3GB VRAM; kill -9 never used). Post-fix verification is
CPU-only (-ngl 0), no GPU touched. Another agent's 35B server on GPU2 (port
8545) was never touched. Rebuild via sourced /root/w4a4-env.sh, incremental.
ASCII commit message, no push.
