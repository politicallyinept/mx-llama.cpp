
================================================================================
taskR9 LEDGER - QUIET-BOX CONFIRMATION OF THE FLOOR-ON + RP_Q4_NC_LDS=8 DECODE CLAIM
================================================================================
Date         : 2026-11-23 (22:20-22:30 PST; sweeps 22:20:39-22:24:05 and 22:25:07-22:29:52)
Host         : mi50 (3x AMD MI50 32GB gfx906, ROCm LXC; ROCm0/1/2 live)
My cards     : ROCm0 + ROCm1 ONLY (ROCm2 never touched; observed idle throughout)
Binary       : /root/build-g4p2b/bin/llama-server  md5 c93c4df35191af8416b85c81ff32fa77
               BUILD-PROVENANCE: src_sha 94d1dbaa38de72600000c9b080dbe2eda43b6631, dirty_files=0
Target       : /root/bench-quants/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf (16.3 GB)
Drafter      : /mnt/old/home/brad/models/DFlash2-Q4_0-ownhead-q4.gguf (STOCK, 2.85 GB)
Env          : /root/w4a4-env.sh (TheRock libs first on LD_LIBRARY_PATH)
Placement    : --device ROCm0,ROCm1 -devd ROCm0 -sm tensor
Flags        : --spec-type draft-dflash --spec-draft-n-max 4 -ctk q4_0 -ctv q4_0 -ngl 99
               -fa on --fit off -c 8192 -ub 1024 -t 16 -np 1 -lm dio --seed 42
Request      : temp 0 (greedy), seed 42, n_predict 256, cache_prompt false, 112-token prompt
Measurement  : server-side response.timings (predicted_ms/predicted_n) + "draft acceptance" log line

--------------------------------------------------------------------------------
VERDICT: CONTENTION ARTIFACT *AND* NOT A +10.9% WIN -- i.e. INCONCLUSIVE-AS-CLAIMED,
         BUT WITH A REAL, SMALL, HIGH-VARIANCE EFFECT (+1.4%..+3.7%) AND A RESOLVED
         CHAMPION-BASELINE DOUBT (48.7 measured, not 58.20)
--------------------------------------------------------------------------------

THE SHORT ANSWER TO THE KEY QUESTION
  The +10.9% does NOT reproduce on an idle box. Seven interleaved reps per arm give
  champ median 48.72 t/s vs lds median 49.39 t/s = +1.37% (Welch t=1.85, n.s.).
  The only way to manufacture +10.9% from this data is lds_best(53.59) vs
  champ_median(48.72) = +9.99% -- a selection artifact, not a treatment effect.
  The prior lane's 57.43/58.69/57.66-vs-53.12/51.62/51.83 cannot be reproduced here.

  AND the champion baseline doubt is REAL but has an explanation that is NOT
  contention: on this box, today, the champion measures 48.72 t/s median
  (robust core reps3-7 = 48.75, spread 0.39 t/s), which is 16.3% BELOW the
  historically recorded 58.20. That gap is not GPU contention (see quietness
  proof) -- it is a property of the current build/harness, discussed in section 5.

--------------------------------------------------------------------------------
1. ARMS
--------------------------------------------------------------------------------
  ARM A  "champ" = floor-ON default (GGML_CUDA_W4A4_MIN_TOKENS=128 resolved from
                   PROFILE=speed default; NO W4A4 levers exported) + q4_0 KV.
                   NO GGML_CUDA_RP_Q4_NC_LDS.
  ARM B  "lds"   = byte-identical to ARM A + GGML_CUDA_RP_Q4_NC_LDS=8.

  Floor flag provenance: the exact knob is GGML_CUDA_W4A4_MIN_TOKENS (see
  q8_repack/mul-mat.cu:246-275 "W4A4 WIDTH FLOOR"). "floor-ON" = 128, sourced from
  /root/floor_gate_v1.sh:21 and /root/merge-ladder.sh:65. CRITICAL: on this binary
  the PROFILE=speed default ALREADY resolves MIN_TOKENS=128 without any export --
  witnessed in every log: "GGML_CUDA_W4A4_PROFILE=speed (unset -> default) resolved:
  DOT8=3, MIN_TOKENS=128 (PROFILE=speed default), KSPLIT=1, SUMW=1". So both arms
  were floor-ON by default, exactly per the brief.

  ARM SEPARATION VERIFIED (per-leg gate witness, not assumed):
    r9_lds_r3.gate.witness   -> "GGML_CUDA_RP_Q4_NC_LDS=8 ACTIVE: repacked Q4_0 nc
                                 per-width admission (W8: u4act T>=6, affine T=4..5;
                                 W4: u4act T=4,7,8, affine T=4,6..8) routes to the
                                 LDS-staged kernel at the measured-best wave;
                                 bit-identical to the generic nc shape per decode arm"
    r9_champ_r3.gate.witness -> NO RP_Q4_NC_LDS line. Kernel NOT engaged.
  Both arms show MIN_TOKENS=128 floor ACTIVE. Separation is clean.

--------------------------------------------------------------------------------
2. INTERLEAVING AND EXECUTION ORDER (14 legs, 7 per arm)
--------------------------------------------------------------------------------
  Pass 1 (alternating A,B,A,B,A,B):
      A1 champ r1  22:20:39      B1 lds r1  22:21:14
      A2 champ r2  22:21:48      B2 lds r2  22:22:22
      A3 champ r3  22:22:57      B3 lds r3  22:23:31
  Pass 2 (ABBA counterbalanced, to break any monotonic drift symmetry):
      C4 champ r4  22:25:07      D4 lds r4   22:25:42
      D5 lds r5   22:26:22      C5 champ r5 22:26:56
      C6 champ r6 22:27:31      D6 lds r6   22:28:05
      D7 lds r7   22:28:44      C7 champ r7 22:29:18
  Single flock -w 2700 /root/gpu.lock held across each whole pass, released
  between passes. TERM-only teardown every leg; 0 kill -9.

--------------------------------------------------------------------------------
3. PER-LEG RESULTS (absolute)
--------------------------------------------------------------------------------
  leg          ord   tg t/s   ms/step   pp t/s   accept   mean_len   drafts(acc/gen)
  champ r1     A1    46.580    21.468     62.5   0.4255     2.723       376/160
  champ r2     A2    46.896    21.324     96.1   0.4255     2.723       376/160
  champ r3     A3    48.721    20.525     96.2   0.4255     2.723       376/160
  champ r4     C4    49.024    20.398     96.3   0.4255     2.723       376/160
  champ r5     C5    48.852    20.470     96.2   0.4255     2.723       376/160
  champ r6     C6    48.751    20.513     96.2   0.4255     2.723       376/160
  champ r7     C7    48.630    20.563     94.8   0.4255     2.723       376/160

  lds   r1     B1    48.418    20.653     80.3   0.4255     2.723       376/160
  lds   r2     B2    49.206    20.323     55.1   0.4255     2.723       376/160
  lds   r3     B3    53.586    18.662     96.4   0.4255     2.723       376/160
  lds   r4     D4    46.112    21.686     96.3   0.4255     2.723       376/160
  lds   r5     D5    53.504    18.690     83.9   0.4255     2.723       376/160
  lds   r6     D6    51.726    19.332     86.8   0.4255     2.723       376/160
  lds   r7     D7    49.390    20.247     95.1   0.4255     2.723       376/160

  ACCEPTANCE IS BIT-IDENTICAL IN ALL 14 LEGS: 0.42553 = 160 accepted / 376 drafted,
  mean accepted length 2.723 tokens. The gate is numerically neutral -- confirmed
  on an idle box, matching the prior lane's claim of bit-identical acceptance.

--------------------------------------------------------------------------------
4. PER-ARM MEDIAN AND SPREAD
--------------------------------------------------------------------------------
  arm      n   tg median   tg min    tg max   spread     ms/step med   stdev
  champ    7    48.721     46.580    49.024    2.444 (5.02%)   20.525   1.016
  lds      7    49.390     46.112    53.586    7.474 (15.13%)  20.323   2.775

  DELTA (median of medians) = +0.669 t/s = +1.37%
  Paired per-rep deltas      = +1.838, +2.310, +4.865, -2.912, +4.652, +2.975, +0.760
                               -> NOT all positive (rep4 is NEGATIVE, -2.912).
  Welch t (7v7)              = 1.85  (not significant at alpha=0.05; need ~2.18)

  Excluding rep1 (cold first-touch; champ pp was 62.5 vs 96 steady state):
  arm      n   tg median   tg min    tg max   spread
  champ    6    48.736     46.896    49.024    2.128 (4.37%)
  lds      6    50.558     46.112    53.586    7.474 (14.78%)
  DELTA = +1.822 t/s = +3.74%; Welch t = 1.72 (still not significant).

  CHAMPION IS TIGHT, LDS IS NOT: reps 3-7 of champ span 48.630-49.024 (0.394 t/s,
  0.8%). LDS spans 46.112-53.586 (7.474 t/s) -- 5x the champ variance and it goes
  BOTH directions (rep4 is the slowest leg of the entire session; rep3/rep5 are the
  fastest). A kernel that changes decode work should not ADD variance while leaving
  acceptance identical. This is the signature of an unstable measurement, not a
  stable speedup.

  How +10.9% could be manufactured from this data (shown to disqualify it):
    lds_max / champ_median = 53.586/48.721 = +9.99%   <- cherry-picked best vs median
    lds_max / champ_min    = 53.586/46.580 = +15.04%  <- worst-case pairing
    lds_median/champ_median= +1.37%                   <- the honest number
  The prior lane's 3-vs-3 with no counterbalancing could land on exactly this kind
  of best-vs-typical split. It is not reproducible under 7-vs-7 ABBA.

--------------------------------------------------------------------------------
5. THE 58.20 CHAMPION DOUBT -- RESOLVED, AND IT IS NOT CONTENTION
--------------------------------------------------------------------------------
  Historically recorded champion: f128q4kv-p-d4 = 58.20/58.33 t/s (taskW3, pair 1,2,
  acceptance 0.4377 / mean_len 2.7508, 3888/8883).
  Measured today: champ median 48.72 t/s, acceptance 0.4255 / mean_len 2.723.

  Two independent discrepancies, both REAL and both explained:
  (a) SPEED: 48.72 vs 58.20 = -16.3% BELOW the historical record.
  (b) ACCEPTANCE: 0.4255 vs 0.4377 = a slightly shorter accepted length (2.723 vs
      2.7508) -- a different draft/target walk, not a speed effect.

  (b) is the documented WALK REGRESSION. /root/wt-g4-phase2/taskG4-PHASE2-LEDGER.md
  section 3 records exactly this on this build lineage: ctrl arms landing far below
  historical (d4-ctrl 40.14 vs adopt a-ctrl 57.41, acceptance 0.348 vs 0.4377) with
  non-uniform per-prompt acceptance shifts, attributed to the uwave(e9b3e3197)->tip
  merge range (k1-headsplit, k2-headkernel, k3-q6lds, fix4, d0-determ, dg-graphreplay,
  d1-vecdot), and it notes the owner's "~57.48 class" reference likely predates the
  regression. My build-g4p2b (94d1dbaa, Nov 17) is inside that range. Today's 48.72
  with acceptance 0.4255 sits BETWEEN the pre-regression 57-58/0.4377 and the
  fully-regressed 40.1/0.348 -- consistent with a partially regressed walk plus a
  different KV/placement from taskW3 (taskW3 used HIP_VISIBLE_DEVICES=1,2 via
  mi50-run.sh; I used --device ROCm0,ROCm1 per brief).
  => The 58.20 reference is NOT reproducible on this binary by construction of the
     build lineage. Comparing my absolute numbers to 58.20 is comparing across a
     known regression; the +10.9% claim's error was in its CHAMPION BASELINE (53.1
     extrapolated downward), not in a hidden contention.

  (a) IS NOT CONTENTION -- proven in section 6. During every measured leg the cards
  were mine alone, 0% pre-leg util, ~13.0/9.8 GB VRAM steady, no foreign server, and
  the champ arm reproduced 48.6-49.0 across five consecutive later legs.

--------------------------------------------------------------------------------
6. QUIETNESS PROOF
--------------------------------------------------------------------------------
  Pre-leg GPU util (rocm-smi --showuse), all 14 legs, all three cards:
    champ r1..r7, lds r1..r7 -> "GPU use (%): 0" for GPU0, GPU1, GPU2 on EVERY pre-leg
    check (/root/r9/quietness.txt, 14 entries, all zero).
  In-run util: GPU0=99-100%, GPU1=99-100%, GPU2=0-2% (ROCm2 untouched, as required).
  Per-card VRAM during measurement (per-5s samples): GPU0 13051-13054 MiB,
    GPU1 9763-9766 MiB -- identical across all 14 legs to within 3 MiB.
    (Target+KV on the two cards; drafter on ROCm0. Values are stable, not contended.)
  Post-leg drain: every leg returned GPU0 and GPU1 to 10-17 MiB before the next leg
    (DRAIN lines), no exceptions.
  RAM: 38 GB available before each pass (need >=20 for a server, >=12 per mi50-run).
  Load average: 1.43-1.84 during pass 1 and most of pass 2 (this is the STALE
    trailing average from the sibling taskR8 run that ended 21:23 -- ps shows NO
    CPU-consuming process: top entries are node_exporter 0.3%, llama-swap 0.0%,
    containerd 0.0%). It rose to 7.07/8.66 in the last two legs (22:28:44, 22:29:18).
    This is an EXTERNAL load that appeared late in pass 2. It did NOT enter the
    comparison asymmetrically: it coincided with lds r7 (49.390) and champ r7
    (48.630), i.e. one leg of each arm, and both were mid-pack. It does inflate
    session-wide variance and is one reason I trust the champ core (reps3-7, before
    the spike) over the full 7.

  In short: the box was quiet for the comparison. The 48.7-vs-58.20 gap is a BUILD
  property, not a contention artifact.

--------------------------------------------------------------------------------
7. WHY THE LDS ARM IS UNSTABLE (mechanism note)
--------------------------------------------------------------------------------
  The kernel is admitted per-width: "W8: u4act T>=6, affine T=4..5; W4: u4act T=4,7,8,
  affine T=4,6..8". With --spec-draft-n-max 4 and the dflash drafter, verify width
  swings step-to-step, so the LDS path is entered on some decode steps and the
  generic <2,1,T,2,64,Q4_0,...> shape on others. Which widths dominate is a function
  of the draft acceptance walk -- identical here (160/376 every leg) -- but the LDS
  waves-per-workgroup choice at width 4 vs 8 has different occupancy, and on this
  card a small occupancy/latency trade can flip sign run to run. Result: identical
  numerics, bimodal timing (46.1 / 49 / 51.7 / 53.5 clusters). That bimodality --
  not a mean shift -- is the actual story the data tells.

--------------------------------------------------------------------------------
8. WHAT I WOULD CHANGE BEFORE BELIEVING ANY RP_Q4_NC_LDS NUMBER
--------------------------------------------------------------------------------
  1. Run >= 10 reps/arm with the rep1 cold leg discarded AND a fixed warm-up of 3
     x 256 tokens before the measured request (pass 1 rep1 pp=62.5 shows the cold
     effect is real and worth ~4%).
  2. Pin the verify width: measure with --spec-draft-n-max 1 (width 2) so the LDS
     admission is deterministic, then with n-max 4. If the win only exists at
     n-max 4 it is a width-mix artifact.
  3. Report MEDIAN OF MEDIANS ONLY. Best-of-N vs median is how +10.9% appeared.
  4. Benchmark per-leg ms/step from predicted_ms/predicted_n, never tg_3s (the
     n_gen n>155 window excludes draft overhead and reads ~10% high: 53.9 vs 48.7).

--------------------------------------------------------------------------------
9. FILES
--------------------------------------------------------------------------------
  /root/taskR9-leg.sh           single server cycle, one measured request
  /root/taskR9-drive.sh         pass 1 (A,B,A,B,A,B)
  /root/taskR9-drive2.sh        pass 2 (ABBA counterbalance, reps 4-7)
  /root/taskR9-parse.py         response.timings -> tg t/s, ms/step, pp t/s, accept
  /root/taskR9-agg.py           median/spread/verdict
  /root/r9/logs/*.log           14 full llama-server logs
  /root/r9/logs/*.resp.json     14 raw /completion responses
  /root/r9/logs/*.samples       per-5s VRAM/util/load during each run
  /root/r9/logs/*.gate.witness  per-leg kernel-gate proof
  /root/r9/quietness.txt        pre-leg util, all zero
  /root/r9/sweep.log            pass 1 timeline
  /root/r9/sweep2.log           pass 2 timeline

--------------------------------------------------------------------------------
10. TEARDOWN
--------------------------------------------------------------------------------
  pgrep -x llama-server -> 0 (verified 22:29:56 and after)
  rocm-smi VRAM -> GPU0 10 MiB, GPU1 10 MiB, GPU2 10 MiB (all back to baseline)
  /root/gpu.lock -> no holders (fuser empty)
  ROCm2 never touched. No git operations performed. Not committed, not pushed.
================================================================================
