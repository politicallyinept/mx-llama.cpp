# taskJ1 ledger — v_dot8 q4_0 KV attention port (ruquant -> INT4-noquant fork)

Date: 2026-11-12 (PST). Branch int4-j1-vdot8 (worktree /root/wt-j1, base INT4-noquant @ 70b650a459).
Build: /root/build-j1 via /root/mi50-build.sh (TheRock amdclang++ pinned, gfx906, --tests).
NOTE: branch name int4/j1-vdot8 was impossible (refs/heads/int4 exists as a branch ref) -> int4-j1-vdot8.

## 1. Port (commits 62aa0f43ef, 017f00df32)

Source: /root/ruquant-src branch ruquant, files ggml/src/ggml-cuda/rq-fattn.cu + .cuh at tip
(= state after e509cee + 047d127 + b9bf750; tip commits 3164cf1/82c73b5 do not touch rq-fattn).
Kernel mechanics ported (HIP/gfx906 branch only; NVIDIA mma.sync branch dropped - fork is gfx906-only):
- Q.K straight from q4_0 nibbles: Q quantized per-32 to int4 (q4_0 rounding, largest->-8), K words
  fed to v_dot8_i32_i4 after one xor (0x88888888); K prepared once per workgroup at tile load
  (b9bf750), int8-Q mode (=2) as two int4 planes.
- P.V: lanes = 256 channels, two channels per lane (nibble pairs of one byte), V decoded per lane
  once per 8 tokens, v_dot2_f32_f16 into f32; lazy softmax rescaling (delta > 8), DPP row
  reductions (row_shr/row_bcast sum, 7 DPP + readlane), P at 16B-aligned LDS for ds_read_b128.
- KV sequence split over runtime-occupancy chunks (64-token tiles through LDS), online softmax,
  rq_fa_combine (own chunk combine: rows x channel quarters over blocks, chunks over waves).
- Template dispatch RMAX {6,12,24,48,96} x NGR/NGT per ruquant (decode R<=6 splits tile tokens
  across wave pairs so V decodes once; verify widths split rows).

Fork deltas vs source (all deliberate):
- Env gate GGML_CUDA_FA_DOT8: DEFAULT OFF (fork convention for numerics-changing levers; ruquant
  defaults ON). 1 = int4 Q, 2 = int8 Q. GGML_CUDA_FA_DOT8_CHUNKS overrides chunk count.
- ggml_cuda_rq_fattn_q4_applies(dst, device) takes the device (used from get_alloc_size where the
  current device may differ); extra guards mask->ne[0] >= K->ne[1] && mask->ne[1] >= Q->ne[1]
  (fork's compact-mask n_kv_max feature).
- Wired in fattn.cu after the dense_gcn hook: dispatch, get_alloc_size (no f16 K/V buffers when the
  kernel applies), supported() returns true when it applies.
- One-time [FA-DOT8] engagement banner on first use (fork witness convention, cf. [MOE-FUSE]).
- test-backend-ops: test_flash_attn_ext_rq_q4 (1e-3 NMSE vs CPU - int4 Q lands ~4-5.5e-4 against
  the CPU's q8_0 Q) at (256,256,4,{6,1}) nb {1,2,8,16} x kv {256,4096}; perf cases q4_0 KV at the
  27B TP2 per-GPU shape (256,256,2,{12,1}) kv {16384,65536} x nb {1,8} + (256,256,4,{6,1}) 65536.

Gate coherence with LLAMA_KV_KROT (taskI1): the kernel is rotation-agnostic - it reads whatever
q4_0 is in the cache and the Q the graph hands it (the graph rotates Q/K/V before the cache when
LLAMA_KV_KROT/LLAMA_KV_KROT_V are on, and de-rotates the attention output for V). Rotation changes
the cache numerics (taskI1 KLD gate: raw q4_0 0.011223, +KROT 0.009821 vs f16), the kernel changes
the attention math at decode/verify widths. The two gates compose orthogonally; both arms validated.

## 2. Protocol

Driver /root/taskJ1-drive.sh, outputs /root/taskJ1/. All legs: flock /root/gpu.lock -w 900,
load/RAM/GPU-free gates, TERM-only teardown + pgrep -x llama-server check, production-env.sh,
TheRock runtime witnessed per leg (no /opt/rocm-7.2.3/lib in maps).
- micro: test-backend-ops perf+test, -o FLASH_ATTN_EXT -p type_K=q4_0, GGML_CUDA_FA_DOT8 0/1/2,
  CUDA0 single GPU.
- KLD (MANDATORY numeric gate): taskI1 protocol re-score - 35B Q8_0 TP2, wikitext 16ch x2048,
  vs existing f16 base logits /root/taskI1/i1-base-f16.logits; arms q40+dot8, q40krot+dot8.
  NOTE: the protocol is pure prefill (ubatch 2048 > 16) so the kernel is INERT there by design -
  these legs certify (a) the shipped q4_0-KV numerics envelope is unchanged by the port and
  (b) the new binary reproduces taskI1's numbers (prefill-path non-regression). The kernel's own
  numerics are gated by test-backend-ops NMSE vs CPU + greedy probe + acceptance (below).
- greedy probe: raw /completion temp0 seed42 n64 fixed prompt per 27B leg vs f16-KV baseline.
- e2e 27B Q4_0 TP2 (unsloth Q4_0 + DFlash2-Q8_0 drafter, server per taskF5/taskI2 config:
  -ngl 99 -fa on -sm tensor --fit off -c 65536 -ub 1024 -t 16 -np 1 -lm dio): natural suite
  (bench.py tg256 x3 + acceptance) arms: off-f16 | on-f16 n4 | on-q40 n4 | on-q40dot8 n4 |
  on7-q40 / on7-q40dot8 (n_max 7) | on-q40krotdot8 n4 | rep-2 of the core trio, interleaved.
  GGML_DFLASH_PROF=1 on all DFlash legs -> draft-fwd / tgt-decode step-phase decomposition.
- llama-bench pp512/tg256 -r 2 per KV arm (f16 | q40 | q40dot8 | q40krotdot8).

## 3. Results

### 3.1 Microbench (test-backend-ops perf, ROCm0 = 1 MI50, gate off vs on)

Correctness first: test-backend-ops test -o FLASH_ATTN_EXT -p type_K=q4_0: 359/359 PASS with
GGML_CUDA_FA_DOT8=0, =1 and =2 (incl. the 8 new rq_q4 cases at 1e-3 NMSE vs CPU). Engagement
banner witnessed: "[FA-DOT8] ... engaged: mode=1 ncols=1 gqa=12 nkvh=2 n_kv=4096".

us/call, single MI50, q4_0 KV (27B per-GPU TP2 shape = 2 KV heads x 12 query heads, D=256):
| shape | gate off (stock vec/tile) | gate on (v_dot8) | speedup |
|---|---|---|---|
| decode nb=1 kv=4096  |  97.5 us |  32.7 us | 3.0x |
| decode nb=1 kv=16384 | 308.9 us |  75.1 us | 4.1x |
| decode nb=1 kv=65536 | 1201.4 us | 246.6 us | 4.9x |
| verify nb=8 kv=16384 | 650.1 us | 355.3 us | 1.83x |
| verify nb=8 kv=65536 | 2517.3 us | 1268.3 us | 1.98x |
| decode nb=1 kv=65536, 4 KV heads (whole model on 1 GPU) | 1233.6 -> 324.6 us | 3.8x |
| verify nb=8 kv=65536, 4 KV heads | 3716.3 -> 1250.7 us | 3.0x |

1.03 -> 6.53 TFLOPS decode at 64k; 5.1 -> 10.2 TFLOPS verify. int8-Q mode (=2) is ~10-25% slower
than int4 (decode 64k 273.8 us, verify 1590 us - 2x the v_dot8 work) but still 4.4x/1.6x over
stock; kept as the higher-accuracy option, not the perf arm. Raw files
/root/taskJ1/micro-perf-dot8{0,1,2}.txt.

### 3.2 KLD gate (taskI1 protocol re-score, 35B Q8_0 TP2, f16 base logits) - PASS

| arm | Mean KLD vs f16 | Max KLD | PPL(Q)-PPL(f16) | taskI1 ref (kernel off) |
|---|---|---|---|---|
| q40 + DOT8=1      | 0.011223 +/- 0.000237 | 1.351633 | +0.0375 (+0.64%) | 0.011223 / 1.35 / +0.64% |
| q40krot + DOT8=1  | 0.009821 +/- 0.000204 | 1.289141 | +0.0243 (+0.41%) | 0.009821 / 1.29 / +0.41% |

Both arms reproduce taskI1's numbers to the last digit (deterministic pp scoring). This is the
expected shape: the protocol is pure prefill (ubatch 2048 > the kernel's 16-token width cap), so
the kernel is inert there - the legs certify (a) the shipped q4_0-KV numerics envelope is
unchanged by the port and (b) the new binary reproduces the pre-fork-gate prefill numerics
exactly (non-regression control). The kernel fired only in the startup warmup graph
([FA-DOT8] ncols=2 n_kv=256 banner), which does not touch scored logits.
The kernel's own numerics are gated by: test-backend-ops 359/359 vs CPU (int4-Q NMSE < 1e-3,
section 3.1), the 64-token greedy probes vs f16-KV baseline (3.3), and DFlash2 acceptance (3.3 -
spec decode is lossless, so acceptance is a distribution-shift readout of the verify path).
Rotation-vs-raw ordering preserved (0.0098 < 0.0112): the KROT arm remains the recommended
companion to the kernel, matching taskI1's verdict.

### 3.3 27B end-to-end A/B

llama-bench plain (27B, TP2, pp512/tg256, r=2):

| arm | pp512 | tg256 |
|---|---|---|
| f16 KV (this binary)      | 586.06 | 44.17 |
| q4_0 KV, kernel OFF       | 525.77 | 38.98 |
| q4_0 KV, DOT8=1           | queued | queued |
| q4_0 KV + KROT, DOT8=1    | queued | queued |

q4_0 KV costs ~12% tg vs f16 in the stock dequant path - the gap the v_dot8 kernel targets
(microbench 3.1 says it should beat f16 at long context, be roughly even at short).

Natural-suite DFlash2 A/B legs (bench.py 8 prompts x 3 reps tg256, GGML_DFLASH_PROF step-phase,
greedy 64-token temp0 seed42 raw-/completion probe per leg): arms off-f16, on-f16(n4),
on-q40(n4), on-q40dot8(n4), on-q40krotdot8(n4), on7-q40(n7), on7-q40dot8(n7), r2 reps of
on-f16/on-q40/on-q40dot8. (taskI2 context, same model/box: natural tg 46.99 off / 38.10 on.)

Interim results (tg256 median t/s, natural suite, KV ~4.3k per prompt):

| arm                          | tg t/s | accept | acc_len | verify tgt-decode w=5 |
|---|---|---|---|---|
| off-f16 (plain decode)       | 47.14  | -      | -       | -                     |
| on-f16  (DFlash2 n4)         | 37.88  | 0.3482 | 2.393   | (see below)           |
| on-q40  (kernel OFF)         | 53.74  | 0.4463 | 2.785   | 41.8 ms               |
| on-q40dot8 (kernel ON)       | 42.68  | 0.3659 | 2.464   | 41.6 ms               |
| on7-q40dot8 (n7)             | 30.62  | 0.2720 | 2.904   | -                     |

Reproduction control: on-f16 37.88/0.3482 vs taskI2 38.10/0.3482 - same box, same suite.

Step-phase (GGML_DFLASH_PROF, on-q40dot8 n4): verify tgt-decode w=5 med 41.6 ms, draft-fwd
(n=5) 13.8 ms, draft-inject 0.84 ms, gather 0.03 ms -> ~56 ms/step x 2.46 tok/step = 43.6 t/s
(AGG 42.68, consistent). Kernel-off control: tgt-decode w=5 med 41.8 ms - the kernel changes
verify wall-time by ~0 at ~4k KV even though the isolated microbench says 1.8-3x on the
attention call alone: at this context the attention slice is a few % of the 27B step.

Surprise: q4_0 KV *without* the kernel (on-q40) is the big lever on this suite - DFlash2-on
jumps 37.9 -> 53.7 (+42%, and +14% over plain decode), with acceptance rising 0.348 -> 0.446.
The kernel arm loses acceptance vs the dequant path (0.446 -> 0.366) at equal step time, i.e.
int4-Q attention numerics shift verify logits enough to hurt draft agreement at ~4k KV.
Long-context probe (benchlong: llama-bench pp16384/tg256, kernel off vs on) added to test
where the microbench win (3-5x at 16k-64k) should show up in full-model decode.
Ops notes: first driver chain died mid-leg ~21:32 (no OOM; sibling taskJ2 pkill collateral);
kld() unbound-$NAME fix; env-prefix must use `env VAR=..` (expanded VAR=x words are not
assignments in bash); load-avg gate removed (starved under sibling 16-thread servers - flock +
GPU-free picker suffice); lock now held once per phase, not per leg (sibling legs >900s caused
per-leg LOCKFAIL churn); gate failures exit the phase for the retry loop.
