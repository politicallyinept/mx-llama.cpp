# taskL3b ledger - GPU0 (rocm0) burn-in rerun #2 after owner's further physical changes: verdict FAIL (ECC, dramatically worse than #1); thermal no better than #1

Date: 2026-11-16 (PST). Box: mi50. Branch `int4-l3b-burn2` off INT4-noquant tip 27ca445502
(worktree /mnt/old/wt-l3b). Subject: GPU0 = rocm0, PCI 12:00.0, HIP device 0.
History: burn-in #1 (taskL3, same day) = thermal PASS (plateau 75.5C vs old 88C) / ECC FAIL
(50.7/min correctable flat at the healthy temp, 62,311 fresh in 2 h + 56k first-touch burst,
0 uncorrectable). Owner then made further unspecified physical changes to the card. This rerun
re-tests both axes with the stricter bar: near-zero correctable climb at load temp; ANY
uncorrectable = immediate STOP + FAIL.

## 1. Question

Can rocm0 return to the measurement pool after the owner's latest physical changes?
(a) THERMAL: plateau should sit at or below burn-in #1's 75.5C junction.
(b) ECC: correctable RATE at load temp should collapse toward zero; #1's 50.7/min flat = fail.

## 2. Timeline forensics (what is observable about the "changes")

- `who -b` / `last reboot`: last boot 2026-11-16 03:28. Burn-in #1's leg ran 04:07:53-06:08:17
  AFTER that boot. There has been NO reboot since burn-in #1.
- ECC counters at pre-flight read exactly 62,311 on GPU0 = the end-of-leg value of burn-in #1,
  and had not moved in the ~4.7 h of idle between #1's teardown and this session's pre-flight.
  Counters are volatile and reset on reboot, so the exact match proves: same card, no power
  cycle since #1, and zero correctable accumulation at idle for hours after load stopped.
- Consequence: the physical changes made after burn-in #1 did NOT involve a reboot/power cycle
  of this box. Idle state is indistinguishable from #1's pre-flight: junction 29C, edge 26C,
  mem 28C, idle sclk 930 MHz / mclk 350 MHz, same as all siblings. No observable idle-state
  improvement from the changes.

## 3. Measurement setup (protocol identical to taskL3)

- ECC + temp authority: Prometheus exporter :9400 (`amd_gpu_ecc_errors_total`,
  `amd_gpu_temp_celsius`), scraped every 60 s -> /root/taskL3b/telemetry.csv (all 4 cards,
  plus vram/power/busy/mem-activity). Container dmesg/journal/RAS are blind here.
- Decision metric = climb RATE (counters are continuous with #1, not reset).
- Load: llama-server (build-jmerge @ a63c468d60, same binary as #1) serving
  /root/bench-quants/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf (16.35 GB), TP1 on HIP
  device 0 (HIP_VISIBLE_DEVICES=0, HSA_XNACK=0, -ngl 99 -fa on -c 16384 -b 4096 -ub 1024
  -t 16 -np 1 -lm dio, port 8215). Runtime resolves libamdhip64/libhsa from /opt/rocm-7.2.3
  (same launch path as #1; TP1, no cross-rank AR path).
- Workload: back-to-back /completion, 4083-token prompt + 256-token decode, cache_prompt=false
  (full re-prefill every cycle, no idle gaps). Same body file as #1 (body1.json).
- Lock: flock /root/gpu.lock -w 2700 held for the whole leg (acquired 11:07:12 first attempt,
  released 13:09:07). Sibling lanes untouched; a sibling server on ROCm1,2,3 (port 10023) ran
  concurrently on other cards; GPU1-3 ECC stayed 0/0 throughout.
- Watchdog: collector TERMs the server on ANY gpu0 uncorrectable. Never fired (0 uncorrectable).

## 4. Baseline (idle, pre-load, 11:07)

| card | edge | hotspot | mem | ecc corr | ecc uncorr |
|------|------|---------|-----|----------|------------|
| gpu0 (rocm0) | 26 | 29 | 28 | 62,311 (cont. from #1) | 0 |
| gpu1 | 29 | 32 | 30 | 0 | 0 |
| gpu2 | 28 | 31 | 32 | 0 | 0 |
| gpu3 | 28 | 32 | 33 | 0 | 0 |

## 5. Load window (leg1: 11:07:12 -> 13:09:07, 120 min sustained, 316 cycles)

GPU0 before/after:

| metric | idle baseline | sustained-load plateau | max |
|--------|--------------|------------------------|-----|
| hotspot (junction) | 29C | **79.5C mean** (drifted 77 -> 81 over the run) | 83C |
| edge | 26C | 53.9C | 55C |
| mem (HBM) | 28C | 59.5C | 66C |
| socket power | 21 W | 174 W mean | 216 W |
| busy | 0% | 99.2% mean | 100% |
| VRAM used | ~10 MB | 16.44 GiB peak | - |
| ecc correctable | 62,311 | 1,036,042 end-of-load (+973,731) | - |
| ecc uncorrectable | 0 | **0** | 0 |
| cycle wall | - | 23.1 s mean (316 cycles) | 24 s |

Rate-vs-temp curve (10-min buckets, GPU0):

```
 t_min  time   hotspot  edge  mem  watts  busy  ecc_corr  d/min
     2  11:08    77.0  51.7   57    175    99    141237    8770
    12  11:18    78.8  53.3   59    176   100    219730    8721
    22  11:28    78.0  53.1   59    175   100    298916    8798
    32  11:38    78.6  53.0   59    172   100    381322    9156
    42  11:48    79.9  54.3   59    183   100    458954    8626
    52  11:58    79.4  53.9   60    176    98    541364    9157
    62  12:08    79.7  54.1   59    175    99    623774    9157
    72  12:18    80.0  54.0   60    177   100    703463    8854
    82  12:28    79.8  54.2   60    168   100    782962    8833
    92  12:38    79.9  54.3   59    178   100    863657    8966
   102  12:48    81.2  54.9   61    169    98    945185    9059
   112  12:58    81.3  55.4   62    168    98   1028090    9212
   122  13:08    81.0  55.0   58    174   100   1036042    7952
```

Post-load cooldown: 81C -> 34C within 60 s of teardown, 32C by +3 min (thermal contact still
good both ways). ECC counter flat at 1,043,464 after load stopped (no idle climb).

## 6. ECC analysis

- First-touch burst: +63,457 in the first ~8 min (model load + first cycles) - same class and
  magnitude as #1's 56k burst.
- Steady-state (post-burst, 112 min): +903,015 = **8,062.6/min**, first half 8,012/min vs
  second half 8,113/min - perfectly linear, no decay, no ramp.
- vs burn-in #1: 50.7/min -> 8,063/min = **159x WORSE** at essentially the same load temp.
  The rate is flat across the whole 77 -> 81C drift (no temp dependence), and zero at idle
  when cold: hard/marginal cells in the working set, not thermal soft errors.
- Structural difference from #1: in #1 the storm was confined to the first-touch burst and
  inference-time reads were nearly clean (50/min). Here the inference reads themselves
  (weights + KV re-touched every cycle) generate ~8k/min - the bad-cell population now sits
  inside the per-cycle working set. The HBM degradation progressed severely between #1 and
  this rerun (hours apart); the physical changes did not help and may have hurt (mount
  pressure on HBM is the obvious suspect, but that is the owner's domain).
- Uncorrectable: 0 across the whole run, all cards. No STOP condition fired. No perf cliff
  (cycle wall 23.1 s mean vs #1's 22.9 s).
- Extrapolation: 973,731 fresh correctables in 2 h = ~11.6M/day at this duty cycle (#1 was
  ~740k/day).

## 7. Thermal analysis

- Plateau 79.5C junction mean (range 77.0-81.3, max 83C) vs #1's 75.5C (73.8-76.8, max 79C):
  **+4C WORSE than #1**, with a slow upward drift over the 2 h (+3C from first to last third).
- Still far below the pre-repaste 88C and cooldown is fast, so mounting contact is not the
  story; but the owner's changes did NOT lower the plateau as expected - they raised it.
- Edge 53.9 vs 50.6, mem 59.5 vs 58.1, watts 174 vs 171 W: consistent small regression.

## 8. Verdict: FAIL - rocm0 stays quarantined; RMA recommendation stands, strengthened

- ECC axis: FAIL, catastrophically. The stricter bar was "near-zero climb at load temp";
  the card delivers 8,063/min steady linear - 159x the #1 rate that already failed. The
  rate did not just fail to collapse after the owner's changes; it exploded. The card's HBM
  is severely degraded and progressing. RMA / replace is the recommendation.
- Thermal axis: no improvement over #1 (79.5C vs 75.5C plateau). Servable temps in isolation,
  but the changes did not achieve the expected lower plateau.
- Pair (0,1) is NOT restored; rocm0 does NOT return to the pool; no probation path - the
  pass bar is not remotely met. Benchmark lanes stay on 1,3; 4-card work stays gated.
- Leg 2 (hotter push) NOT run: extension is conditional on a clean leg-1; leg-1 was not
  clean (same gating as #1).
- Watch-item for the owner: GPU1-3 stayed 0/0 throughout, so this is card-local, not a box
  or PSU or scrubber artifact.

## 9. Operational notes

- Pre-flight note for the parent brief: the brief said the box was rebooted once after #1;
  the evidence says otherwise (boot 03:28 predates #1's 04:07 leg; counters continuous).
  Rate-based comparison was used regardless, so the verdict is unaffected.
- burn.sh/collect.py reused verbatim from taskL3 (paths -> /root/taskL3b), port 8215 to avoid
  the #1 port. Lock acquired first attempt, held 122 min, released clean; server TERM drained
  normally (no wedge; HIP stack healthy after).
- Artifacts on box: /root/taskL3b/ (telemetry.csv raw 60 s x 4 cards, leg1.log cycle record,
  burn.sh/collect.py/analyze.py). telemetry.csv copy committed alongside this ledger.
