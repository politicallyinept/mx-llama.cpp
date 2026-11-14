# taskX LEDGER -- T=16 i8narrow determinism re-proof + G5 fresh-dir reproduction

Branch: int4-x-verify (from INT4-noquant tip 0340ba0a1e). Box: mi50.
Date: 2026-11-14 (PST). No pushes; local commits only. Two verification
debts from the review cycle, one lane.

## Part 1 -- T=16 determinism re-proof: VERDICT PASS, width 16 RESTORED

Method (taskA2, /root/wt-t16fix/taskA2-LEDGER.md): standalone single-TU
build of the fixed harness mb_i8narrow_test.cu (the *_test.cu glob filter
excludes it from the lib build, the e9bac2e8 pattern), TheRock amdclang++
only, GPU runs under flock /root/gpu.lock on a free pair.

Build (worktree /root/wt-x-verify @ 0340ba0a1e):
  $TR/bin/amdclang++ -x hip --offload-arch=gfx906 -O3 -DNDEBUG -std=gnu++17
    -DGGML_USE_HIP -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1
    -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -DGGML_HIP_GRAPHS -DGGML_HIP_NO_VMM
    -I ggml/src -I ggml/include -isystem /opt/rocm/include
    -isystem /opt/rocm-7.2.3/include mb_i8narrow_test.cu -lamdhip64
  ($TR = /root/therock-venv/lib/python3.12/site-packages/_rocm_sdk_devel)
The tip harness already carries the taskA2 fixes (d-plane uploaded in BYTES
+ device readback proof, d_y sentinel pre-fill, NaN-safe bad count) plus the
task C2 KSPLIT sweep and per-launch det-bitmatch check.

Runs: 3x (r1/r2/r3, DEVS=0,1 each), logs /root/wt-x-verify/mb-r{1,2,3}.log,
all MB-RC=0:
- d_ad readback mismatches=0/2560 and 0/8704 (both shapes) -- upload-in-
  bytes proof holds on-device.
- 0 VERIFY FAIL lines; NaN-safe bad count 0 in every cell (T=1..16, both
  shapes, KSPLIT 1/2/4/8 on shape 2).
- det-bitmatch ok in every cell (double-launch bit-match inside the run).
- Shape 1 (K=5120 N=17408): T=16 verify ok, maxrel 0.00647, median 0.508 ms
  -- matches the taskA2 validation numbers exactly; T8/T1 1.11x PASS.
- Shape 2 (K=17408 N=5120): T=16 KSPLIT=1 maxrel 0.021 (atol class,
  cancel-exempt 1) ok, 1.570 ms; KSPLIT-BAR best KSPLIT=4 T8/T1 1.31x PASS.
  KSPLIT=1 T8/T1 1.99x FAIL is the known ungated shape-2 underfill
  characteristic (taskA2 section 5.5), not a regression.
- Cross-run determinism: all 25 verify lines (maxrel, cancel-exempt,
  det-bitmatch, vs-ks1-delta) byte-identical across r1/r2/r3; only the
  timing medians differ.

Action: width 16 RESTORED to the admitted set {1,2,4,8,16} in
ggml/src/ggml-cuda/q8_repack/mul-mat.cu (admission condition, banner text,
admission comment). Single-TU compile of the edited mul-mat.cu clean (only
the two PRE-EXISTING mmq_gemm_repacked waves-per-eu warnings, per taskB2
section 4). Gate GGML_CUDA_I8NARROW stays DEFAULT OFF -- default serving
unaffected. Trail notes appended to taskB2-LEDGER.md section 6 and
taskP-CLEANUP-LEDGER.md item 4.

## Part 2 -- G5 fresh-dir reproduction: VERDICT PASS, bit-exact

Fresh build: /root/build-x-g5 via /root/mi50-build.sh /root/int4-noquant-wt
/root/build-x-g5. All guards passed: clean tree at 0340ba0a1e, post-configure
TheRock-compiler guard OK (CMAKE_HIP_COMPILER =
/root/therock-venv/.../_rocm_sdk_devel/bin/amdclang++), 4 ROCm devices
enumerated. BUILD-PROVENANCE.txt: llama_server_md5=9b7b23c70004cf99cd0a8ddfd4ae0349,
libggml_hip_md5=19ce6af4226879d4814d5b9d862a2ab6. Built from tip 0340ba0a1e
BEFORE the taskX width-16 restore commit (the restore is gate-off dead code
for this leg; kept out to test the tip exactly).

Leg d4-x1: G5 d4 control -- 35B Qwen_Qwen3.6-35B-A3B-Q8_0 TP2 +
DFlash2-Q8_0 drafter, --spec-draft-n-max 4, natural suite
(/root/p2-prefill/bench.py, temp-0, 8 prompts x 3 reps, probe_pp + 3
warmups). Driver /root/taskX-g5-drive.sh = clone of /root/taskG5/drive.sh
leg() (clean arm, prof off), flock -w 2700, TERM-only teardown. DEVS=1,2.
Witnesses in /root/taskX-g5/d4-x1.witness: model sha256s, server cmdline,
environ, loaded libs = therock runtime (no apt libamdhip64 -- W4 check ok).

Comparison vs canonical /root/taskG5/d4-r1 (and d4-r2):
- ALL 51 "#acc rate/pos" lines byte-identical; final line =
  (0.647, 0.394, 0.298, 0.248) -- the canonical per-pos acceptance, exactly.
- ALL 35 per-task "draft acceptance" lines byte-identical (e.g. 0.31486,
  142 accepted / 451 generated, mean len 2.26).
- Final cumulative counters identical: #calls(b,p,g,a) = 57 3615 3446 3446,
  #gen drafts 3446, #acc drafts 2229, #gen tokens 13700, #acc tokens 5468.
- AGG: accept_rate 0.373, mean_accept_len 2.4919 (canonical 0.373 / 2.492);
  per-prompt mean_accept_len matches the canonical per-prompt table on all
  8 prompts.
- Timing (not the gate): tg_t_s_median 69.51 vs canonical 70.58 (-1.5%,
  inside the canonical legs' own <2.3% spread).

Verdict: a BRAND-NEW build dir from the current tip via mi50-build.sh
reproduces the canonical G5 d4 acceptance counters bit-exactly. The
toolchain pin (TheRock amdclang++) is validated for fresh dirs.

## Files changed (this branch)

- ggml/src/ggml-cuda/q8_repack/mul-mat.cu -- width 16 restored to the
  i8narrow admitted set {1,2,4,8,16} (condition + banner + comment)
- taskB2-LEDGER.md -- section 6 T=16 bullet: re-proof CLOSED trail
- taskP-CLEANUP-LEDGER.md -- item 4: re-proof LANDED trail
- taskX-LEDGER.md (this file)

## Artifacts on box (not committed)

- /root/wt-x-verify/{mb_i8narrow_test, mb-r{1,2,3}.log, run-mb.sh}
- /root/taskX-g5/{d4-x1.*, driver.out}, /root/taskX-g5-drive.sh
- /root/build-x-g5/ (fresh G5 build dir + BUILD-PROVENANCE.txt)
