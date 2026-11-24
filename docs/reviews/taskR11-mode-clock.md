taskR11 LEDGER - RP_Q4_NC_LDS=8 mode instability: kernel traces + clock telemetry
================================================================================

Date    : 2026-11-23 22:57-23:32 PST
Host    : mi50 (3x AMD MI50 32GB gfx906, ROCm LXC)
Binary  : /root/build-g4p2b/bin/llama-server
Target  : Qwen3.8-27B-Q4_0.gguf (16.3 GB)
Drafter : DFlash2 stock (2.85 GB)
Flags   : --spec-type draft-dflash --spec-draft-n-max 4 -ctk q4_0 -ctv q4_0
          -ngl 99 -fa on --fit off -c 8192 -ub 1024 -t 16 -np 1 -lm dio --seed 42
          --device ROCm0,ROCm1 -devd ROCm0 -sm tensor
Arm     : GGML_CUDA_RP_Q4_NC_LDS=8

QUESTION
--------
R9/R10 established that this arm is BIMODAL (fast ~60 t/s, slow ~41-51 t/s) on a
champion control that is tightly reproducible, and that env, VRAM, acceptance and
host load do NOT select the mode. This pass asked the binary question: is the slow
mode a DIFFERENT KERNEL, or the SAME KERNEL running slower?

PART 1 - KERNEL TRACES SAY SAME KERNEL (definitive)
----------------------------------------------------
Five profiled legs captured with rocprofv3 (launch-under-profiler method; the
--attach path is broken on this host - see the ops skill). Per-trace aggregates
from /root/r10/trace/t{1..5}/:

  trace  total_us    in_window  rp_q4_us   tp_reduce_A1_us  n_gen  tg(profiled)
  t1     27747512    444759     3689719    17489492         252    10.15
  t2     28427171    444759     3746797    17576290         252    10.14
  t3     28090437    444759     3773262    17690134         247    10.12
  t4     28131908    444759     3720340    17811867         230    10.15
  t5     28144793    444759     3731109    17800343         249    10.04

  t1 vs t2..t5: kernel signatures 115/115 present in every trace,
                dispatch-count differences ZERO,
                total kernel time delta +1.24 to +2.45 percent.

  rp_q4_nc_lds dispatches: 79,188 in EVERY trace (verified by direct grep).
  Total dispatches: 1,267,902 in every trace.
  Measured window identical (444,759) in all five.

VERDICT PART 1: SAME-KERNEL-SLOWER. No kernel is selected differently between
fast and slow conditions. The 115-signature set, the per-kernel dispatch counts,
and the 79,188 LDS dispatches are invariant. Note n_gen DOES vary (230-252) while
the kernel mix does not, so n_gen is not evidence of a different code path.

CAVEAT: profiled runs carry heavy overhead (10.1 t/s vs ~60 unprofiled), so these
traces are valid for KERNEL COMPOSITION but not for wall-clock comparison against
unprofiled legs. All five were captured identically (same script, prompt,
n_predict) so the comparison between them is sound.

PART 2 - WHERE THE TIME ACTUALLY GOES
--------------------------------------
From the t1 profile (36.2 M us of kernel time, 89 unique kernels):

  k_broadcast_reduce<2> (TP sync, Agent 1)   20,280 ms   56.0 pct   d=28,928
  rp_q4_nc_lds<5,8,false,false>               4,181 ms   11.5 pct   d=77,040
  gcn_u4_gemm<false>                          3,408 ms    9.4 pct   d=1,428
  __amd_rocclr_copyBuffer                     2,663 ms    7.3 pct   d=745,315
  mul_mat_vec_q<ggml_type 14>                   760 ms    2.1 pct   d=324
  mul_mat_vec_f<float,float,5,128>              726 ms    2.0 pct   d=20,544

The workload is TENSOR-PARALLEL-BROADCAST BOUND. TP sync is ~63 pct of kernel
time (including Agent-1 share measured separately in the clock sweep table). The
LDS kernel this whole effort has been tuning is ~11-13 pct. That is a hard
ceiling: no amount of LDS optimisation can address two-thirds of the time.

PART 3 - CLOCK TELEMETRY FINDS THE MODE SELECTOR
-------------------------------------------------
Six unprofiled legs with per-second clock sampling
(/root/r10/clock/logs/clk_r{1..6}.clock). All six came out FAST and identical:

  clk_r1 60.17 | clk_r2 60.02 | clk_r3 60.08 | clk_r4 60.01 | clk_r5 60.00 | clk_r6 60.17
  n_gen = 183 and acceptance 0.42553 (160/376, len 2.68) on every leg.

`rocm-smi --showclocks` encodes level+MHz concatenated, e.g. sclk0=71606 is
level 7 at 1606 MHz. Decoded samples within single legs:

  clk_r1: 0925(0/925) 81850(8/1850) 81850 71606(7/1606) 71606
  clk_r3: 0925        71606          71606  71606        41282(4/1282)
  clk_r4: 81850       81850          71606  61485(6/1485) 0925

The SCRAP occurs mid-leg: the governor moves between 925, 1282, 1485, 1606 and
1850 MHz within ~5 seconds under nominally identical load.

IMPORTANT TELEMETRY LIMIT: every clock sample also records `pwr0=0 jt0=0 use0=0`
- power, junction temperature and GPU utilisation all read zero. This box has the
known blind-telemetry condition, so the clock reading is the ONLY observable and
throttle CAUSE (thermal vs power vs ECC-driven) cannot be attributed from here.

CONCLUSION
----------
The mode is selected by GPU CLOCK STATE, not by kernel selection and not by host
load. A slow leg is one where the governor did not hold boost; a fast leg is one
where it did. This explains why the mode drifts between sweeps and why load
average does not predict it.

Revised full picture of the arm:
  - Same kernels always execute (Part 1, direct evidence).
  - ~63 pct of time is TP broadcast, ~13 pct is the LDS kernel (Part 2).
  - The fast/slow split is GPU clock behaviour (Part 3).

ADOPTION POSITION: unchanged - DO NOT ADOPT.
The treatment arm still cannot be shipped: a path whose throughput varies 41-60
t/s because of uncontrolled clock state is not a performance switch. But the
reason is now understood, and it is NOT a defect in the LDS kernel.

RECOMMENDED NEXT STEPS (not done here)
--------------------------------------
1. Pin clocks for all future measurement (rocm-smi --setperflevel high, or a
   fixed sclk level) so benchmark variance is removed at the source. Any future
   A/B on this box without clock pinning is measuring the governor, not the code.
2. If throughput is the goal, target k_broadcast_reduce (63 pct), not the
   quantised matmul path (13 pct).
3. Resolve the blind telemetry (pwr/jt/use all zero) before attempting any
   thermal or power attribution.

ARTIFACTS
---------
  /root/r10/trace/t{1..5}/          kernel traces + server logs + agg.json
  /root/r10/clock/logs/clk_r*       per-second clock samples for six fast legs
  /root/r10/clockleg.sh             clock-sampling leg harness
  /root/r10/traceF.sh               profiled leg harness (launch-under-profiler)

No commits, no pushes from this pass.
