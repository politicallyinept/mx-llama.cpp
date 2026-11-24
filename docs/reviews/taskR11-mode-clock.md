taskR11 - RP_Q4_NC_LDS=8 MODE INSTABILITY: KERNEL TRACES, TP BROADCAST STALL, CLOCK TELEMETRY
=============================================================================================
Date    : 2026-11-23 22:57 - 2026-11-24 PST (merged revision)
Host    : mi50 (3x AMD MI50 32GB gfx906, ROCm 7.2.3, rocprofv3 1.1.0, ROCm LXC)
Binary  : /root/build-g4p2b/bin/llama-server
Target  : Qwen3.8-27B-Q4_0.gguf (16.3 GB)   Drafter: DFlash2 stock (2.85 GB)
Flags   : --spec-type draft-dflash --spec-draft-n-max 4 -ctk q4_0 -ctv q4_0 -ngl 99 -fa on --fit off -c 8192 -ub 1024 -t 16 -np 1 -lm dio --seed 42 --device ROCm0,ROCm1 -devd ROCm0 -sm tensor
Arm     : GGML_CUDA_RP_Q4_NC_LDS=8
Method  : launch-under-rocprofv3 kernel trace (--attach is broken on this host)

REVISION NOTE
-------------
Replaces an earlier quick pass that concluded "the mode is selected by GPU CLOCK STATE". That is RETRACTED. All six clock legs were FAST (60.00-60.17 t/s), so the clock data never contained a slow leg and showed only that the DPM governor moves. The measured mechanism is a tensor-parallel broadcast barrier with a heavy p99 tail. Clock state is retained as a SECONDARY, UNQUANTIFIED hypothesis. Two first-pass numbers are corrected: n_gen is not monotone, and trace row counts are not constant.

QUESTION
--------
R9/R10 established this arm is BIMODAL (fast ~60 t/s, slow ~41-51 t/s) on a tightly reproducible control, and that env, VRAM, acceptance and host load do NOT select the mode. This pass asked: is the slow mode a DIFFERENT KERNEL, or the SAME KERNEL running slower?

VERDICT: SAME-KERNEL-SLOWER, but NOT REPRODUCED under profiling as a separable fast/slow mode, and the extra time does NOT live in the LDS kernel. Two claims are supported: (a) NO DIFFERENT-KERNEL EVIDENCE. Name sets and per-kernel dispatch counts are identical across profiled pairs (69 vs 69 kernels, q11_1 vs q11_2; shared dispatch counts equal, median ratio 1.0000). Slow mode selects no different code. (b) WHERE THE TIME GOES. ggml_cuda_tp::k_broadcast_reduce<2> is 54.7 pct of kernel time in p11_2 (20,181 ms of 36,875 ms summed over both agents) and it is a WAIT, not compute: mean 1,355.8-1,421.2 us/dispatch on Agent 1 (GPU0) versus 25.6-26.0 us/dispatch on Agent 2 (GPU1), a 52-55x asymmetry for the SAME kernel. On Agent 1 the slowest 1 pct of its calls are 62.2-64.2 pct of that agent's time in the kernel - a pooled both-agents count gives 70.1 pct. Spin-barrier signature; it absorbs per-step jitter. The unprofiled fast/slow mode did NOT reproduce: 28 fresh legs gave 0 slow legs and no slow leg was captured under profiling. That is the explicit limit of this ledger (L1).

PART 1 - KERNEL COMPOSITION IS INVARIANT (stands from the first pass)
---------------------------------------------------------------------
From /root/r10/trace/<tag>/k_kernel_trace.csv. Columns: n_gen | tg(prof) | acc | rows:

  t1    252 | 10.15 | 160/376 | 1,267,902    504-char prompt
  t2    252 | 10.14 | 160/376 | 1,267,902    504-char prompt
  t3    247 | 10.12 | 160/376 | 1,267,902    504-char prompt
  t4    230 | 10.15 | 160/376 | 1,267,902    504-char prompt
  t5    249 | 10.04 | 160/376 | 1,267,902    504-char prompt
  p11_1 252 | 10.33 | 160/376 | 1,159,063    TRUNCATED trace
  p11_2 254 | 10.37 | 160/376 | 1,267,902    629-char prompt
  q11_1 243 | 10.08 | 160/376 | 1,267,902    629-char prompt
  q11_2 230 | 10.38 | 160/376 | 1,267,902    629-char prompt
  q11_3 249 | 10.18 | 160/376 | 1,219,262    629-char prompt
  v11_1 247 |  9.61 | 153/404 | 1,295,340    REJECTED, wrong prompt

Full trace p11_2 vs t1: 88 vs 88 kernel signatures, dispatch-count differences ZERO on every shared signature. Rows are 1,267,902 on 8 traces, 1,219,262 on q11_3, 1,159,063 on p11_1 (truncated); range across all 11 traces 1,159,063-1,295,340. rp_q4_nc_lds<5,8> dispatches 79,188 on p11_2, q11_1, q11_2, t1..t5; 72,179 on q11_3; 62,686 on p11_1.

VERDICT PART 1: SAME-KERNEL-SLOWER. No kernel is selected differently. n_gen varies (230-254) while the kernel mix does not, so n_gen is not evidence of a different path. CAVEAT: rocprofv3 adds ~6x overhead (10.0-10.4 t/s vs 59-60 unprofiled). These traces are valid for KERNEL COMPOSITION only, never for wall-clock comparison with unprofiled legs.

PART 2 - WHERE THE TIME ACTUALLY GOES (p11_2, full trace, both agents)
----------------------------------------------------------------------
36,875 ms of kernel time over 1,267,902 rows, 89 distinct kernels:

  k_broadcast_reduce<2> (TP sync)      20,181 ms  54.7 pct  d=28,928
  rp_q4_nc_lds<5,8,false,false>         4,238 ms  11.5 pct  d=79,188
  gcn_u4_gemm<false>                    3,931 ms  10.7 pct  d=1,428
  __amd_rocclr_copyBuffer               2,787 ms   7.6 pct  d=745,315
  mul_mat_vec_q<(ggml_type)14,5>          756 ms   2.0 pct  d=324
  mul_mat_vec_f<float,float,5,128>        735 ms   2.0 pct  d=20,544

TENSOR-PARALLEL-BROADCAST BOUND: 54.7 pct is TP sync, ~62 pct with its dependent copies. The LDS kernel this effort has been tuning is 11.5 pct - no LDS optimisation can address more than half the time. (The first pass quoted 56.0/63 pct from the same trace using a wider broadcast subtree; 54.7 pct single-kernel is the conservative figure, used throughout.)

PART 3 - THE 52-55x BROADCAST ASYMMETRY (the measured mechanism)
-----------------------------------------------------------------
k_broadcast_reduce<2> by Agent_Id (A1 = GPU0, A2 = GPU1), mean us/dispatch, via /root/r10/kernelstats.py. Columns: A1 mean | A1 max | A2 mean | A2 max | ratio | A1 slowest 1 pct:

  p11_1  1,355.8 | 125,672 | 25.8 | 182 | 52.5x | 62.8 pct
  p11_2  1,369.4 | 312,671 | 25.8 | 183 | 53.1x | 64.2 pct
  q11_1  1,412.5 | 263,213 | 25.7 | 189 | 55.0x | 62.9 pct
  q11_2  1,349.1 | 135,648 | 26.0 | 188 | 51.9x | 62.4 pct
  q11_3  1,421.2 | 594,162 | 25.7 | 179 | 55.3x | 63.7 pct
  t1     1,376.5 | 122,504 | 25.6 | 180 | 53.8x | 62.2 pct
  t4     1,394.7 | 694,117 | 25.9 | 182 | 53.9x | 64.0 pct

GPU0 records the long "wait for peer" side, GPU1 the short "arrive" side. Agent-1 duration distribution (14,464 calls, p11_2): p50 62.9-63.7 us | p75 74.1-84.2 | p90 187.8-199.4 | p95 8,945.9-9,523.0 | p99 10,180-10,591 | p99.9 118,213-122,072 us | max per table. The cliff is between p90 and p95. On Agent 1 alone the slowest 1 pct of calls (144 of 14,464) carry 64.1 pct of that kernel's Agent-1 time, and calls at or below p50 carry ~1.7 pct. NOTE ON THE HEADLINE 'slowest 1 pct = 70.1 pct': that figure is computed over BOTH agents pooled (289 of 28,928 calls, which is 2.0 pct of Agent 1's own call count), so it is a pooled number, not a per-agent 1 pct. The per-agent statistic is 64.1 pct (p11_2). Both say the same thing: a small tail dominates. On Agent 2 the same kernel is flat: p99 41-47 us, max 168-191 us, slowest 1 pct only 2.1-2.5 pct of its time. INTERPRETATION: the dominant cost is a tensor-split synchronisation barrier whose tail absorbs per-step jitter. A leg can run ~60 t/s most of the time and still average 41 because the barrier occasionally stalls for hundreds of ms. The stall is in the barrier, not in any matvec kernel. Target this; it is not yet proven to be the mode selector (L1).

PART 4 - THE LDS KERNEL IS INVARIANT (new evidence)
---------------------------------------------------
rp_q4_nc_lds<5,8> per-dispatch mean, 7 traces (GPU0 us/disp / GPU1 us/disp, disp per agent):
  p11_1 48.96 / 60.91 (30,629)   p11_2 48.96 / 61.05 (38,520)   q11_1 48.84 / 59.22 (38,520)
  q11_2 48.92 / 60.92 (38,520)   t1 48.60 / 59.94 (38,520)      t2 49.19 / 60.64 (38,520)
  t3 49.28 / 61.41 (38,520)
GPU0 48.60-49.28 us (spread 1.4 pct), GPU1 59.22-61.41 us (spread 3.7 pct): a constant, not a variable. A fixed ~10.7-12.8 us GPU0->GPU1 offset (GPU1 consistently slower) is present in every trace. Per-call maxima 111,520-160,160 us; sole outlier a 7,679 us hiccup on q11_1/GPU1. This kernel is ruled out as a mode selector.

PART 5 - CLEANEST PROFILED PAIR, EQUAL-WINDOW
---------------------------------------------
Windows are the last N seconds of each trace because rocprofv3 truncates at the last finalised dispatch, so raw totals are not comparable (an early raw comparison produced phantom "kernels only in one arm", all prefill: gcn_u4_gemm, rocBLAS Cijk_*, p2p). (a) q11_1 vs q11_2, last 18.0 s: 69 vs 69 distinct kernels; kernel time 18,537,529 vs 18,793,616 us (+1.38 pct); dispatches 310,481 vs 319,627; 69 shared kernels, median dispatch-count ratio 1.0000 (17 exactly equal). rp_q4_nc_lds<5,8> 54.32 vs 55.23 us/disp (-1.6 pct); k_broadcast_reduce 739.27 vs 719.20 us/disp (+2.8 pct). Everything within +-3 pct except the broadcast, which moves upward. (b) p11_1 vs p11_2, last 19.0 s: kernel time 19,785,416 vs 19,754,072 us (-0.16 pct); dispatches 331,206 vs 333,346; k_broadcast_reduce 13,363,595 vs 13,572,586 us (+1.6 pct), 18,459 vs 18,687 disp; rp_q4_nc_lds<5,8> 55.06 vs 55.49 us/disp (+0.7 pct). Name counts 75 vs 69 - the extra 6 are window-edge prefill entries. Both windows agree in sign and magnitude, so the conclusion does not hinge on the choice.

PART 6 - LEG CLASSIFICATION AND THE n_gen PREDICTOR (CORRECTED)
---------------------------------------------------------------
tg_ts = pure decode rate (draft excluded), PRIMARY; tg_incl includes draft; never mixed. 40 unprofiled legs (12 historical R10 + 28 fresh), same arm/binary/prompt/seed; every one reports acc = 160/376 (0.42553, mean len 2.68) - the WORK is bit-identical in every leg.
  SLOW: r9_lds_r2 n_gen 254 tg_ts 41.58 | r9_lds_r6 n_gen 156 tg_ts 48.73 | r9_lds_r5 n_gen 174 tg_ts 50.62
  fast (edge): r9_lds_r1 n_gen 179 tg_ts 52.77 | r9_lds_rs11b_10 n_gen 179 tg_ts 53.00
  other 35: n_gen 183, tg_ts 53.05-53.39
FAST population 37 legs; SLOW 3 legs (41.58-50.62). Perfect separation on this set.
n_gen AS A PREDICTOR - TESTED, NOT ASSUMED, CORRECTED FROM THE FIRST PASS: on the R10 12-leg set, n_gen in {179..183} => all fast; outside => 41.58/48.73/50.62. The band is a FLAG, NOT MONOTONE. Two of three slow legs have FEWER steps than the fast band: r6 slow at 156, r5 slow at 174. Only r2 (254) matches "more steps". The first pass implied monotonicity; that is wrong in direction for 2 of 3 slow legs. Use n_gen only as an out-of-band classifier: {179..183} = fast, outside = suspect. It did NOT hold under profiling (230-254 on all 9 profiled legs, one speed population), so it is correlational. 28 unprofiled legs this session produced ZERO slow; slow mode is genuinely rare and not load-triggered.

PART 7 - HARNESS DEFECTS FOUND AND FIXED (4)
--------------------------------------------
D1. leg.sh / drive.sh "FATAL: zero steps parsed" on EVERY successful leg - they grepped for per-step lines ("accepted N/ M draft tokens") that this single-request shape never emits. Every R10 leg returned rc=1 despite succeeding, so the 12-leg R10 sweep had NO parsed metrics. FIX: /root/r10/parse_leg.py parses summary lines (n_gen, tg, tg_3s, acceptance, timings) and asserts each field exists (empty parse => HARNESS-FAIL exit 2). All 12 historical logs now parse, all 160/376, no mismatches.
D2. trace-v2.sh TERM'd "$RPID" = the rocprofv3 WRAPPER, tearing rocprofv3 down before it finalises k_kernel_trace.csv. This DESTROYED the CSV on every leg run through it (229 MB of scratch, no CSV). FIX: /root/r10/taskR11-trace.sh TERMs the llama-server CHILD, waits for the CSV, then unwinds. CSV written on every run since.
D3. trace-v2.sh shipped a 504-char prompt vs the canonical 629 in leg.sh and traceF.sh. Measured: a 504-char leg gave acceptance 153/404 (0.37871, mean len 2.50) instead of 160/376 (0.42553, mean len 2.68) - a DIFFERENT trajectory, breaking acceptance comparability. All t1..t5 and v11_1 inherit this and are flagged; they are used only for prompt-independent claims (broadcast asymmetry, LDS invariance), both of which reproduce on the 629-char legs. FIX: taskR11-trace.sh uses 629 chars; all 5 taskR11 profiled legs reproduced 160/376.
D4. LEAKED "timeout 900" wrapper - under rocprofv3 the inner "timeout 900 llama-server" ignores TERM and spins in state R with zero children and zero VRAM while holding /root/gpu.lock through the inherited fd, stalling the next leg. FIX: cleared only after /proc proves it is /usr/bin/timeout with 0 children; llama-server is NEVER -9'd (that wedges the SMC). Verified: q11 loop ran 3 legs, exited, lock free.
Concurrency: another agent's R10 job (traceF.sh t1..t5) and a clockleg.sh queue were live holding /root/gpu.lock on arrival. Not disturbed; this session serialised behind the lock. Their traceF.sh leaked resident servers holding ~13.7 GB; cleared by TERM only.

PART 8 - CLOCK TELEMETRY: SECONDARY, UNQUANTIFIED HYPOTHESIS
-------------------------------------------------------------
Six unprofiled legs, per-second sampling (/root/r10/clock/logs/clk_r{1..6}):
  clk_r1 60.17 | clk_r2 60.02 | clk_r3 60.08 | clk_r4 60.01 | clk_r5 60.00 | clk_r6 60.17
  n_gen = 183 and acceptance 160/376 (0.42553, mean len 2.68) on every leg.
ALL SIX ARE FAST. rocm-smi encodes level+MHz (71606 = level 7 at 1606 MHz). Distinct samples:
  clk_r1: 0925 81850 71606   clk_r2: 0925 81850 71606   clk_r3: 0925 41282 71606
  clk_r4: 0925 61485 71606 81850   clk_r5: 0925 61485 71606   clk_r6: 0925 71606
The governor moves between 925, 1282, 1485, 1606 and 1850 MHz within a single leg under nominally identical load (clk_r1 steps 925 -> 1850 -> 1606 within 3 seconds). WHAT THIS DOES AND DOES NOT SHOW: it shows the governor moves during a leg. It does NOT show that clock state causes the slow mode, because all six legs ran at 60.00-60.17 t/s - the data contains ZERO slow legs and cannot separate fast from slow. The first pass's "mode is selected by GPU clock state" was inference from a moving governor, not measurement of a slow leg, and is retracted. TELEMETRY LIMIT: every sample also records pwr0=0 jt0=0 use0=0 - power, junction temperature and utilisation all read zero. This box has the blind-telemetry condition, so sclk is the ONLY usable observable and throttle cause (thermal vs power vs ECC) cannot be attributed here.

LIMITS - READ BEFORE CITING
---------------------------
L1. NO GENUINE SLOW LEG WAS EVER CAPTURED UNDER PROFILING. 28 fresh unprofiled legs gave 0 slow legs; all 9 profiled legs are ONE population (tg 10.02-10.38, spread 3.6 pct). The fast-vs-slow comparison in PART 5 is FASTEST versus SLOWEST legs WITHIN the profiled baseline, NOT the 41.58 t/s mode. Do not read PART 5 as evidence about that mode; it is evidence that no code-path difference exists to explain it. The 52-55x asymmetry (PART 3) is likewise measured on one population - a mechanism CANDIDATE for the mode, not a measurement of the mode itself.
L2. Profiled tg (10.0-10.4 t/s) is ~6x slower than unprofiled (59-60) from rocprofv3 overhead. Profiled and unprofiled figures are never compared here.
L3. Equal-window totals are window-sensitive; 18.0 s and 19.0 s agree in sign and magnitude (kernel time within 1.4 pct, dispatch within 2.9 pct).
L4. t1..t5 and v11_1 used the 504-char prompt (153/404); flagged, used only for prompt-independent claims.
L5. n_gen is NOT monotone (r5/r6 slow with FEWER steps). Band {179..183} = fast only. Corrected from the first pass.
L6. Multiple agents were on the box (traceF t1..t5, clockleg x5-6). All runs serialised behind /root/gpu.lock; no leg overlapped another on the GPU. Clock/DPM state was sampled (clocks.pre/mid/post) but is NOT used to attribute cause.
L7. Single-prompt, single-seed, single-configuration evidence. The broadcast asymmetry is a property of -sm tensor on ROCm0,ROCm1 and must not be generalised to other splits.
L8. Trace rows are NOT constant: 1,159,063 (p11_1, truncated) to 1,295,340 (v11_1), with 1,267,902 on 8 traces and 1,219,262 on q11_3. The first pass's "1,267,902 rows in every trace" came from comparing only t1 vs t4 and is corrected here.

CONCLUSION
----------
  - Same kernels always execute; dispatch counts invariant (PART 1, PART 5).
  - ~55 pct of kernel time is TP broadcast; ~11.5 pct is the LDS kernel (PART 2).
  - The broadcast is a WAIT: 52-55x costlier per dispatch on GPU0 than GPU1; on GPU0 the slowest 1 pct of its calls are 62-64 pct of that agent's time (70.1 pct pooled over both agents) (PART 3).
  - The LDS kernel is a constant (48.60-49.28 us GPU0 / 59.22-61.41 us GPU1 across 7 traces) and is ruled out as a mode selector (PART 4).
  - Clock state moves within fast legs but was never observed on a slow leg; it remains an UNQUANTIFIED secondary hypothesis, not the demonstrated selector (PART 8).
ADOPTION POSITION: DO NOT ADOPT. A path whose throughput varies 41-60 t/s is not a performance switch. The reason is now understood to be a tensor-parallel synchronisation barrier with a heavy tail - NOT a defect in the LDS kernel, and NOT (proven) a clock-state effect. Pinning clocks would not by itself make this arm adoptable; it only removes one uncontrolled variable.

RECOMMENDED NEXT STEPS (not done here)
--------------------------------------
1. Pin clocks for all future measurement (rocm-smi --setperflevel high, or a fixed sclk level). This removes a known uncontrolled variable even though clock state is not yet proven causal, and any future A/B without pinning measures the governor too.
2. If throughput is the goal, target k_broadcast_reduce (54.7 pct of kernel time, with a 62-64 pct self-concentrated p99 tail), not the quantised matmul path (11.5 pct). The tail is where a leg's wall clock is lost.
3. Resolve the blind telemetry (pwr/jt/use all zero) before any thermal or power attribution.
4. Capture ONE genuine slow leg under profiling. Until that exists, every fast-vs-slow statement here is within-population.

ARTIFACTS
---------
  /root/r10/trace/{p11_1,p11_2,q11_1,q11_2,q11_3,t1..t5,v11_1}/  k_kernel_trace.csv, server.log, window.markers, gate.witness, clocks.*, temp.*
  /root/r10/logs/r9_lds_*.json  40 unprofiled legs   /root/r10/clock/logs/clk_r*  six fast legs
  /root/r10/clockleg.sh, traceF.sh (504-char), taskR11-trace.sh (629-char, TERM child)
  /root/r10/parse_leg.py, kernelstats.py, agg_trace.py, final_table.py

No commits, no pushes from this pass.
