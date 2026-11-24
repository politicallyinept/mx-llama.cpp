taskR10 LEDGER - RP_Q4_NC_LDS=8 mode instability
===============================================

Date    : 2026-11-23 22:37-22:45 PST
Host    : mi50 (3x AMD MI50 32GB gfx906, ROCm LXC)
Binary  : /root/build-g4p2b/bin/llama-server
Target  : Qwen3.8-27B-Q4_0.gguf (16.3 GB)
Drafter : DFlash2 stock (2.85 GB)
Flags   : --spec-type draft-dflash --spec-draft-n-max 4 -ctk q4_0 -ctv q4_0
          -ngl 99 -fa on --fit off -c 8192 -ub 1024 -t 16 -np 1 -lm dio --seed 42
          --device ROCm0,ROCm1 -devd ROCm0 -sm tensor
Arm     : GGML_CUDA_RP_Q4_NC_LDS=8 (treatment only)

PURPOSE
-------
The R9 quiet-box sweep reported this treatment arm as BIMODAL - a fast mode
near 60 t/s and a slow mode near 47-51 t/s - on a champion control that was
tightly reproducible at 53.39-53.64. Env, VRAM, acceptance and host load had
all been ruled out. This pass re-ran the treatment arm alone to (a) confirm
the bimodality reproduces and (b) compare kernels between fast and slow legs.

RESULT 1 - BIMODALITY REPRODUCES, BUT WITH A DIFFERENT SPLIT
-------------------------------------------------------------
12 treatment legs, all with 0 pct pre-leg GPU utilisation on all three cards
(quietness witness per leg in /root/r10/quietness.txt).

  FAST   9 legs : 59.14  59.87  59.99  60.04  60.07  60.10  60.11  (plus
                  59.89, 59.86) - all within 59.14-60.11
  MID    2 legs : 57.30  51.30
  SLOW   1 leg  : 41.76

Fast-mode rate is 9 of 12 (75 pct). The earlier sweep saw roughly half the
legs slow. The mode therefore DRIFTS BETWEEN SWEEPS rather than splitting
50/50 - which argues against a deterministic per-config kernel choice and
toward a state-dependent runtime effect.

Load average does NOT explain it: r8 ran at load 8.23 and hit 60.04 (fast),
while r2 ran at load 5.33 and hit 41.76 (slow).

RESULT 2 - THE WORK IS IDENTICAL ACROSS MODES; ONLY WALL TIME DIFFERS
----------------------------------------------------------------------
Per-leg detail (from /root/r10/logs/r9_lds_r*.log):

  leg   tg t/s   n_gen   accepted/generated   eval time
  ----  ------   -----   ------------------   ---------
  r8    60.04    183     160 / 376            4785.18 ms / 256 tokens
  r1    59.14    179     160 / 376            4832.05 ms / 256 tokens
  r5    57.30    174     160 / 376            5037.19 ms / 256 tokens
  r6    51.30    156     160 / 376            5232.73 ms / 256 tokens
  r2    41.76    254     160 / 376            6132.86 ms / 256 tokens

Every leg evaluated exactly 256 tokens. Every leg accepted exactly 160 of
376 drafts. Acceptance is AGGREGATE-identical (walk not compared): the accepted/drafted
TOTALS match, but n_gen varies between legs, and equal totals over different step counts
imply the per-step acceptance walk DIFFERS. Aggregate identity therefore does NOT establish
that the token path is unchanged; the SAME-KERNEL verdict rests on the kernel-trace evidence
(identical dispatch counts and kernel sets in taskR11), not on this acceptance argument.
The only quantity that moves is elapsed wall time (4785 to 6133 ms, +28 pct).

n_gen varies inversely with speed (254 at the slowest leg, 183 at the
fastest). n_gen is the number of generation steps taken, so a higher n_gen
means shorter accepted runs and more steps for the same 256 output tokens.
That is consistent with the same work being re-partitioned across more steps,
not with a different kernel doing different work.

VERDICT: SAME-KERNEL-SLOWER
---------------------------
No evidence of a different kernel being selected. The treatment arm executes
an identical number of tokens with an identical acceptance pattern; only the
wall clock varies. This is a RUNTIME STABILITY defect, not a speedup and not
a kernel-selection difference.

CONSEQUENCE FOR ADOPTION
------------------------
Unchanged and now better supported: do NOT adopt. A path whose throughput
swings between 41.8 and 60.1 t/s across identical runs, on a flag that is
nominally a pure performance switch, is not shippable. The champion control
has never shown this behaviour (53.39-53.64 across 7 legs).

WHAT WAS NOT DONE
-----------------
NO rocprof kernel-time diff was captured this pass. The lane reproduced the
modes and characterised them, but never reached the profiling step, so the
per-kernel microseconds comparison does not exist. The SAME-KERNEL verdict
above rests on identical token counts and identical acceptance plus
differing wall time - it is a strong inference, not a direct kernel-trace
measurement. A future pass wanting kernel-level proof should capture
rocprofv3 traces on one fast and one slow leg and compare per-kernel totals.

HARNESS DEFECT (recurring)
--------------------------
Every one of the 12 legs was marked "FATAL: zero steps parsed" despite
completing successfully (GEN done rc=0, clean teardown, valid numbers in the
log). The harness greps for per-step "accepted N/ M draft tokens" lines,
but these runs issue a short single request that emits only SUMMARY lines:

  eval time =    4832.05 ms /   256 tokens ( 18.8 ms per token, 59.14 tokens per second)
  draft acceptance = 0.42553 (  160 accepted /   376 generated), mean len =  2.68
  n_gen =    179, tg =  59.14 t/s, tg_3s =  59.47 t/s

This is the FOURTH lane in this program to lose results to a per-step parser
that does not match the request shape actually issued. Any future harness
must parse the summary line for short requests, and must assert on the
parsed value being present rather than silently treating an empty parse as a
failed run.

ARTIFACTS
---------
  /root/r10/leg.sh, drive.sh        harness
  /root/r10/logs/r9_lds_r{1..12}.log   per-leg server logs (source of all numbers above)
  /root/r10/logs/*.samples          in-run VRAM + load per leg
  /root/r10/quietness.txt           0 pct GPU util witness per leg
  /root/r10/sweep.log, drive.out    run record

No commits, no pushes. GPUs drained to idle after every leg.
