# taskP2 — review item 5: KSPLIT dead-path strip + R7 combine A/B-only marker

Date: 2026-11-15. Base: c0dd09e4c9. Review source: external audit 9.11 (option b).

## Changes
1. w4a4-i8narrow.cu: KSPLIT>1 code (12-arg overload dispatch tail + second-pass
   reducer mul_mat_i8narrow_reduce_q40) now compiles ONLY when
   GGML_I8NARROW_KEEP_KSPLIT is defined; serving builds get a GGML_ABORT stub in
   the unreachable path. mb_i8narrow_test.cu defines the flag itself (line 1,
   before the .cu include) so the taskC2 KSPLIT sweep harness is unchanged.
   NOT touched: GGML_RP_KSPLIT (different mechanism, K-sliced w32 GEMM), the
   10-arg overload, the serving dispatch at mul-mat.cu:439.
2. fattn-common.cuh: GGML_CUDA_FA_COMBINE_SPLIT gate marked A/B-ONLY per the
   taskV2-r7 verdict (negative at TP2 64k; ruquant-class win needs TP4 or
   pb>=64 engagement). Comment only; code untouched, still default OFF.

## Verification
- Build: /root/build-p2strip via /root/mi50-build.sh (TheRock pinned), rc=0.
  (First attempt failed legitimately: a manual cmake configure poisoned the
  build dir with the wrong HIP compiler; the script guard caught it — FATAL:
  libggml-hip not produced. rm -rf + rerun, green.)
- Gate-off byte-identity seal: f0q4 arm (exercises the 10-arg i8narrow dispatch
  whose KSPLIT>1 tail was stripped), 27B Q4_0 TP2 DEVS=1,2 (GPU0 quarantined),
  production env witnessed, flock, TERM-only teardown, pgrep empty.
  PROBE-SHA=77841cb6a251235d == canonical → SEAL-PASS (/root/taskP2/).
