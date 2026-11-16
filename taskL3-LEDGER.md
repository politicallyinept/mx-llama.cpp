# taskL3 ledger — GPU0 (rocm0) burn-in after remount+repaste: verdict FAIL (ECC), PASS (thermal)

Date: 2026-11-16 (PST). Box: mi50. Branch `int4-l3-gpu0burn` off INT4-noquant tip e5c91714e3
(worktree /root/wt-l3-burnin). Subject: GPU0 = rocm0, PCI 12:00.0, HIP device 0.
History: 919k+ correctable ECC climbing, 0 uncorrectable; 88°C junction hot card; owner
remounted + repasted and rebooted the box (box uptime 16 min at session start).

## 1. Question

Does the repasted rocm0 return to the measurement pool? Two independent axes:
(a) THERMAL — repaste quality: sustained-load junction plateau should sit well under the
old 88°C. (b) ECC — correctable rate at the lower temp should collapse toward zero;
ANY uncorrectable = immediate STOP + FAIL.

## 2. Measurement setup

- ECC + temp authority: Prometheus exporter :9400 (`amd_gpu_ecc_errors_total`,
  `amd_gpu_temp_celsius`), scraped every 60 s -> /root/taskL3/telemetry.csv (also
  vram/power/busy/mem-activity, all 4 cards). Container dmesg/journal/RAS are blind here.
- NOTE: the owner's reboot reset the volatile ECC counters — all four cards read 0 at
  pre-flight. The 919k+ figure is pre-reboot history; the decision metric is therefore
  the climb RATE under load, not the absolute count.
- Load: llama-server (build-jmerge @ a63c468d60) serving
  /root/bench-quants/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf (16.35 GB, GGUF magic
  verified), TP1 on HIP device 0 (HIP_VISIBLE_DEVICES=0, clean shell + HSA_XNACK=0,
  -ngl 99 -fa on -c 16384 -b 4096 -ub 1024 -t 16 -np 1 -lm dio, port 8213).
- Workload: back-to-back /completion requests, 4083-token prompt + 256-token decode,
  cache_prompt=false (full re-prefill every cycle, no idle gaps). 315 cycles in 120 min,
  mean wall 22.9 s/cycle, busy mean 97.5% (100% in steady buckets).
- Lock: flock /root/gpu.lock held for the whole leg (acquired 03:57 first attempt,
  04:07:53 real leg, released 06:08:17). Sibling lanes untouched; cards 1,3 idle through
  my window except brief sibling legs after my release.
- Watchdog: collector TERMs the server on any gpu0 uncorrectable. Never fired.

## 3. Baseline (idle, pre-load, 03:55-03:57)

| card | edge | hotspot | mem | ecc corr | ecc uncorr |
|------|------|---------|-----|----------|------------|
| gpu0 (rocm0) | 26 | 29 | 29 | 0 | 0 |
| gpu1 | 28 | 31 | 29 | 0 | 0 |
| gpu2 | 27 | 31 | 32 | 0 | 0 |
| gpu3 | 28 | 31 | 33 | 0 | 0 |

## 4. Load window (leg1: 04:07:53 -> 06:08:17, 120 min sustained)

GPU0 before/after:

| metric | idle baseline | sustained-load plateau | max |
|--------|--------------|------------------------|-----|
| hotspot (junction) | 29°C | **75.5°C** | 79°C |
| edge | 26°C | 50.6°C | 53°C |
| mem (HBM) | 29°C | 58.1°C | 64°C |
| socket power | 24 W | 171 W mean | 232 W |
| busy | 0% | 97.5% mean | 100% |
| VRAM used | ~21 MB | 16.27 GiB peak | — |
| ecc correctable | 0 | 62,311 end-of-leg | — |
| ecc uncorrectable | 0 | **0** | 0 |

Sustained-load curve (10-min buckets, GPU0):

```
 t_min  time   hotspot  edge  mem  watts  busy  ecc_corr  d/min
     2  04:10    76.2  51.3   58    174   100     56782     50
    12  04:20    73.8  50.8   58    155    90     57285     50
    22  04:30    75.7  51.1   57    164    98     57794     51
    32  04:40    74.5  50.7   58    173    96     58290     50
    42  04:50    75.9  50.9   56    176    95     58821     53
    52  05:00    75.2  50.2   58    173    99     59319     50
    62  05:10    76.8  51.5   57    171   100     59836     52
    72  05:20    75.8  50.9   58    179   100     60338     50
    82  05:30    75.8  50.7   57    175    96     60859     52
    92  05:40    75.5  50.7   59    173    97     61365     51
   102  05:50    75.7  50.8   57    169   100     61868     50
   112  06:00    75.3  50.2   58    183   100     62311     49
```

Plateau holds flat for the full 2 h: hotspot 73.8-76.8°C across all buckets, ECC rate
rock-steady 49-53/min. Post-load cooldown: 76°C -> 32°C within 60 s of teardown (healthy
thermal contact both ways).

## 5. ECC analysis

- First-touch burst: counters went 0 -> 56,224 during model load + first cycles
  (03:57-04:04, ~7 min incl. the 16.3 GB VRAM fill). ~3.4 correctable errors per MB of
  first-touched memory — a large population of marginal HBM cells surfacing on first read.
- Steady-state (post-burst, 04:11 -> 06:08, 117 min): +5,930 = **50.7/min**, first half
  50.8/min vs second half 50.6/min — no decay, no ramp, perfectly linear accumulation.
- Uncorrectable: 0 across the whole run (all cards). No STOP condition, no page-retirement
  storm visible, no perf cliff (cycle wall 22.9 s mean, stable).
- Comparison to pre-repaste state: ~70/min at 88°C -> ~51/min at 76°C. The rate dropped
  only ~27% for a 12°C plateau reduction. If the errors were thermally-driven soft
  errors the rate should have collapsed much harder; the flat rate at the lower temp plus
  the 56k first-touch burst says the array damage is real (hard/marginal cells), not a
  temperature artifact.

## 6. Verdict: FAIL — rocm0 stays quarantined

- THERMAL axis: PASS. Repaste is good: 75.5°C sustained junction plateau vs the old 88°C
  (-12.5°C), max 79°C, fast cooldown. The card no longer runs hot.
- ECC axis: FAIL. Correctable rate stays high (~51/min) at the now-healthy temperature —
  exactly the task's stated fail condition ("correctable rate stays high at low temp").
  62,311 fresh correctables in a 2 h window extrapolates to ~740k/day at this duty cycle.
- Per protocol the hotter leg-2 push was NOT run (extension is conditional on a clean
  leg-1; leg 1 was not clean).
- Consequence: pair (0,1) is NOT restored; benchmark lanes stay on 1,3; 4-card work stays
  gated. The repaste fixed the symptom (heat) but the HBM degradation is independent of it
  and continues. Recommend: keep rocm0 out of the measurement pool; treat as replace/RMA
  candidate. If it must ever return, only after page-retirement data (amdgpu bad-pages)
  shows the failing region is containable — and note ECC counters reset on every reboot,
  so track RATE, not counters.
- GPU2 watch-item: no anomaly during this run — one brief sibling leg (~1 min, 63 W,
  hotspot 37°C) and the post-release lanes; ECC stayed 0/0 throughout.

## 7. Operational notes

- The first leg-1 attempt (pp2640 body, 03:57) was restarted to hit the spec'd pp4096
  (body re-tokenized to 4083 tokens). Its llama-server took >4 min to exit on SIGTERM
  (stuck in "cleaning up before exit" while the driver kept sending requests); a second
  TERM after killing the driver exited it cleanly. No wedge; HIP stack healthy after
  (rocminfo/rocm-smi normal, later legs unaffected).
- Chain watcher for leg-2 died at launch: `pgrep -f "chain.sh" || setsid ... chain.sh`
  self-matched the ssh wrapper cmdline and short-circuited the launch (known trap). The
  verdict decision was manual anyway; no impact on data.
- Artifacts on box: /root/taskL3/ (telemetry.csv raw 60 s x 4 cards, leg1.log cycle
  record, burn.sh/collect.py drivers, analysis.json). telemetry.csv copy committed
  alongside this ledger.
