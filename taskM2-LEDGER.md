# taskM2-LEDGER: GGML_CUDA_I8NARROW empirical verification on Qwen3.5-35B-A3B-UD-Q4_K_XL-nextn

Date: 2026-11-11 (PST). Agent: M2 subagent. Box: mi50 (fresh boot 19:12, all GPUs idle at start).

## Verdict

REFUTED (as expected): the env-gated i8narrow dispatch (GGML_CUDA_I8NARROW=1) does
NOTHING for this model. The gate can never fire because admission requires a repacked
Q4_0 src0 and the model contains zero Q4_0 tensors. Measured tg/pp deltas are within
base-to-base noise. The gate mechanism itself is proven alive by a positive-control
arm (below): when the dispatch site IS reached, the witness banner fires, yet the
i8narrow branch is still never taken because no tensor has type Q4_0.

## Build under test

- Worktree /root/int4-noquant-wt, branch INT4-noquant @ 7129719aa5 (post-history-rewrite).
- Binary: /root/int4-noquant-wt/build/bin/llama-server (M1's candidate build dir;
  the anticipated build-hip/ path never appeared, so the completed same-tree build
  was used read-only; no build was run by M2).
- llama-server --version: "0.6.0-dev (build 11881, commit 7129719aa5)" - stamped with
  the current HEAD (M1 rebuilt the dir between the pre-run probe and leg start).
- md5 stable across all legs (bin-md5-before == bin-md5-after):
  llama-server   895219cb739b4867411a383acf096eae
  libggml-hip.so 496dde34b8da7f4c7c51b80d855d69c9
- Gate compiled in: `strings build/bin/libggml-hip.so | grep -n I8NARROW` ->
  witness string at line 3490, env name at 5638.

## Static admission analysis (source)

ggml/src/ggml-cuda/q8_repack/mul-mat.cu:383 (inside ggml_cuda_mul_mat_repacked, the
DENSE repacked entry point, reached only when ggml_cuda_repack_mul_mat_should_fire
holds, i.e. src0 lives on a cuda-repack buffer):

    if (rp_i8narrow_on() && src0->type == GGML_TYPE_Q4_0 &&
            ggml_cuda_repack_mul_mat_should_fire(src0) &&
            (ne11 == 1 || ne11 == 2 || ne11 == 4 || ne11 == 8 || ne11 == 16))

Admission therefore requires a Q4_0 tensor admitted to the repack layout
(GGML_CUDA_REPACK_Q4_0=1 opt-in, repack-common.cu:103-112). The witness banner is
emitted from rp_i8narrow_on() (a static lambda) at the first call of that dispatch
site, i.e. banner absence means the repacked narrow dispatch site was never reached.

## GGUF tensor-type histogram (gguf-py GGUFReader, tensor-info block only)

File: /root/moe-nextn/Qwen3.5-35B-A3B-UD-Q4_K_XL-nextn.gguf (23139907520 B)
arch qwen35moe, 41 blocks, 256 experts, 753 tensors total:

    Q8_0  323   (dense attn/embedding/out + shared expert)
    F32   308   (norms, router ffn_gate_inp, ...)
    Q4_K   78   (ffn_gate_exps / ffn_up_exps 3D expert tensors)
    Q5_K   41   (ffn_down_exps)
    BF16    2   (blk.40 router tensors, nextn graft)
    Q6_K    1   (blk.10 ffn_down_exps)
    Q4_0    0   <-- ZERO. i8narrow admission statically impossible.

## Live A/B (GPU2 only, single GPU, production-env bundle, one flock window)

Driver: /root/moe-nextn/m2_i8narrow_ab_v3.sh (v2 at m2_i8narrow_ab_v2.sh),
artifacts in /root/moe-nextn/m2-i8narrow-v3/ (v2: /root/moe-nextn/m2-i8narrow/).
Server flags: -ngl 99 -fa on -c 16384 -b 4096 -ub 4096 -t 16 -np 1 -lm dio -fit off
--device ROCm2 --spec-type draft-mtp --spec-draft-n-max 2 (mirrors the Nov-2 ab-ud
serving config, single-GPU per hard rule). Bench: mtp_bench.py pp4096 x3 + tg256 x3.
Arms interleaved off/on/off on the same binary; MTP acceptance 170/170 on every tg rep.

Witness (literal greps of the server logs):

1. gate-ON production arm (environ proves GGML_CUDA_I8NARROW=1 reached pid):
   $ grep -n "I8NARROW" gate-on-1.server.log
   (absent)
   $ grep "\[rp-dispatch\]" gate-on-1.server.log | grep -oE "type=[0-9]+" | sort | uniq -c
   (empty; type=2 count: 0)
   Load banners show why: under production-env, GGML_CUDA_REPACK_KQUANT=0 keeps
   Q4_K/Q5_K/Q6_K canonical, Q8_0 is suppressed by GGML_CUDA_REPACK_Q8_0_STOCK_ACC,
   and the Q8_0 shape-hint admission is off for non-tensor split -> NO tensor is on
   the repack buft -> the repacked dispatch site is never reached -> banner absent.

2. Positive control q8-ctl (GGML_CUDA_I8NARROW=1 + GGML_CUDA_REPACK_Q8_0=1 forced
   admission + STOCK_ACC=0, witness-only arm, NOT a perf arm):
   $ grep -n "I8NARROW" q8-ctl.server.log
   17:0.12.413.067 W [I8NARROW] gate ACTIVE: repacked Q4_0 narrow-batch window (T in
   {1,2,4,8,16}) routes to mul_mat_i8narrow_launch (exact 8-bit split activations,
   [qs|d] planes)
   $ grep "\[rp-dispatch\]" q8-ctl.server.log | grep -oE "type=[0-9]+" | sort | uniq -c
        60 type=8
   $ grep "\[rp-dispatch\]" q8-ctl.server.log | grep -c "type=2"
   0
   => banner mechanism works when the site is reached; the i8narrow branch is STILL
   never taken (60 dispatches, all type=8 Q8_0, zero Q4_0).

Perf (medians of 3 reps; off = median of the two off-arm medians):

    arm          pp4096 t/s      tg256 t/s
    gate-off-1   2101.34         161.34
    gate-on-1    2099.51         161.17
    gate-off-2   2097.63         161.24
    ---------------------------------------------------------
    gate-on vs off:  pp +0.001%   tg -0.074%
    base spread (off-1 vs off-2): pp 0.177%  tg 0.062%

Both deltas are at or below the base-to-base spread: no effect, as predicted by the
static analysis (the branch is a constant false for this model).

## Findings

1. CONFIRMED: the gate NEVER fires on Qwen3.5-35B-A3B-UD-Q4_K_XL-nextn. Admission
   requires repacked Q4_0 src0; the model has 0 Q4_0 tensors (UD-Q4_K_XL uses
   Q4_K/Q5_K/Q6_K experts + Q8_0 dense). Witness banner absent in the gate-ON
   production arm even though environ proves the env reached the process.
2. The witness mechanism is functional: the forced-Q8_0-admission control arm fires
   the exact banner while the dispatch trace shows 60 repacked narrow dispatches,
   all type=8, zero type=2 - the i8narrow branch is correctly never taken.
3. tg/pp gate-ON vs gate-OFF on the same build: pp +0.001%, tg -0.074%, both inside
   the 0.06-0.18% base spread. No perf effect, no regression.
4. Under the production bundle this model has ZERO tensors on the repack buft
   (KQUANT=0, STOCK_ACC on, shape-hint off for non-tensor split), so the entire
   q8_repack dense dispatch is inert for it regardless of I8NARROW.
5. Incidental: llama-server on this build segfaults in libc during post-TERM
   teardown on every arm (dmesg: general protection fault in libc.so.6). VRAM
   drained cleanly after each (next arm started normally); bench data unaffected.
   Flag for the branch owner - teardown-path regression, not a load/dispatch bug.
6. Incidental: /root/mtp-final/mtp_bench.py opens its out path with "w" - multiple
   arms sharing one jsonl path silently overwrite each other; use per-arm paths.
7. Process note: the anticipated build-hip/bin/llama-server never appeared within
   the polling window; M1's completed same-tree build dir was used instead (md5
   bracketed before/after the legs). No M2 build was run; no race into a build dir.

## Rule compliance

All GPU touches (rocm-smi, serving) inside one flock -w window on /root/gpu.lock per
run; GPU2 only; -lm dio; TERM-only teardown (never kill -9); curl /health polling;
RAM guard >=15GB; no GGUF data reads (tensor-info block only via gguf-py).
