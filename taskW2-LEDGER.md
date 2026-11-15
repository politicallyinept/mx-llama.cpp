# taskW2 LEDGER - re-seal of the 35B MoE DFlash2 spec-decode configs under REPACK_Q8_0 admission

Date: 2026-11-15 (PST). Box: mi50. Branch `int4-w2-moeseal` off tip c0dd09e4c9
(worktree /root/wt-w2-moeseal). NOT pushed. Driver /root/taskW2/drive.sh +
legs-extra.sh + retry-d7.sh, artifacts /root/taskW2/.

## 0. Question

R8 (taskV2-LEDGER taskV2-moe section) ADOPTED GGML_CUDA_REPACK_Q8_0=1 +
GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0 on the 35B MoE after a bit-identical plain-decode
seal (greedy 10/10 BITID, KLD 0.000000 control-identical, tg256 +1.82pct). The
DFlash2/spec-decode configs on this model were sealed BEFORE that adoption and were
never re-sealed - the gate is live on an unsealed config class. This lane re-seals
the DFlash2 d4 serving config (taskG5/taskH3, ~70 t/s class) with the pair ON vs OFF:
greedy probe byte-identity (text + logprob-payload sha), KLD vs stock, acceptance per
leg; plus the nc 6-8 / RP_Q4_NC_LDS engagement check. Plain decode is already sealed
by taskV2/taskV2-moe - not redone here.

## 1. Setup

Build: /root/build-uwave (BUILD-PROVENANCE src_sha=e9b3e3197f dirty_files=0; llama-server
md5 b956a6858d8976624271378f97d825cb, libggml-hip 127f1494af87e2a85b2e6801892ca542).
Tip c0dd09e4c9 differs from e9b3e3197f ONLY in ledger .md files (git diff --stat:
taskV1/V2/V3 ledgers, 0 code files) - the binary is tip-valid.

Config under seal (BOTH arms, taskG5/taskH3 serving line verbatim): target
Qwen_Qwen3.6-35B-A3B-Q8_0.gguf (35.2 GiB, qwen35moe) + drafter
Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf (570 MB, 6 layers, block_size 8); env
w4a4-env + GGML_CUDA_ALLREDUCE=internal GGML_CUDA_REPACK_KQUANT=0 GGML_GDN_COLTHREAD=1,
HIP_VISIBLE_DEVICES=2,3 (pair B TP2; GPU0 quarantined, never device 0),
HSA_KERNARG_POOL_SIZE=67108864; flags -ngl 99 -fa on -sm tensor --fit off -c 16384
-ub 1024 -t 16 -np 1 -lm dio --spec-type draft-dflash --spec-draft-n-max 4.

Arms (env-only isolation, pair is the ONLY delta):
* stock:  pair unset. Witness: "REPACK_Q8_0 gate inactive (env mode 0)" + "STOCK_ACC
  gate ACTIVE: Q8_0 stays canonical" + MOE_CANONICAL ACTIVE + draft-dflash n_max=4
  block_size=8 banners; /proc/pid/environ carries no pair vars.
* repack: GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0 (YIELD default ON).
  Witness: "gate ACTIVE (env mode 1)" + RESTYLE ACTIVE + YIELD ACTIVE + functional
  "q8_repack: repacking q8_0 tensors on device 0" AND "device 1" + /proc environ.
MOE_CANONICAL keeps the 35.08 GB of 3D expert Q8_0 canonical on BOTH arms; the admitted
repacked set is the 260 dense 2D Q8_0 tensors (2.615 GB) of the target PLUS - under
mode 1 (forced-for-all) - the whole Q8_0 drafter (mode 1 admits every model in the
process, so the drafter's tensors are repacked too; this is exactly the surface the
re-seal had to cover).

Ops: doctor ALL CLEAR pre-launch; flock /root/gpu.lock -w 900 per leg (6 retries); RAM
guard >=12 GB; GPU2/3 drain <2 GB gates before launch and after teardown; TERM-only
teardown (PID-targeted); refuse-if-exists per leg; pgrep -x llama-server empty after
every teardown; llama-swap untouched. TheRock runtime witnessed per leg (therock
libamdhip64 in /proc/maps; apt-runtime check = pass).

## 2. Greedy seal - THE GATE (temp-0 seed-42, logprobs=1, 64 tok, via the DFlash2 server)

Suite: 10 probes p0008..p4128 (same calibration suite as the taskV2-moe plain-decode
seal: declined 9..31 GEMM class, >=32 yield class, chunked prefill, N=1 repacked MMV
decode), run THROUGH the spec-decode server so every generated token passes the
width-5 DFlash2 verify path. Comparison = text sha256 + logprob-payload sha256 per probe.

| comparison                          | result                                            |
|-------------------------------------|---------------------------------------------------|
| stock-r1 vs stock-r2 (determinism)  | 10/10 BITID                                       |
| stock-r1 vs repack-r1 (THE GATE)    | 10/10 BITID, tokdiff 0, max_logprob_delta 0.000e+00 |
| stock-r2 vs repack-r2 (gate r2)     | 10/10 BITID                                       |
| stock-r1 vs repack-r2 (cross)       | 10/10 BITID                                       |
| repack-r1 vs repack-r2 (repack det.)| 10/10 BITID                                       |

Canonical sealed bytes (stock-d4-r1, sha16 text/logprob-payload):
p0008 0ff3ca8cea8f/24016f385369  p0012 fa4c8c1c9e76/fdbb097c8594
p0024 53ae05672d8c/623cd879a17d  p0030 3e6da07e12e5/f32906e6495e
p0033 3e6da07e12e5/b68dfacf6588  p0064 2b204ac6bc07/207c2177f5bb
p0512 04c6c7cbbc19/f20629d7bd25  p2048 6e4b1312af16/aec1eb688f78
p4112 04c6c7cbbc19/fe4b97b04d19  p4128 4f6634be0d8b/84fe6bb36fb1
All four d4 legs byte-identical to these bytes. GREEDY-BIT-ID: PASS.

## 3. KLD vs stock (target-model envelope, taskV2-moe protocol)

llama-perplexity on wiki.test.raw -c 2048 -b 2048 --chunks 16 (32768 positions), TP2
pair B, same arm envs (no drafter: perplexity has no spec path; the spec-path numerics
are covered by the greedy gate in sec 2).

* repack vs stock dump: Mean KLD 0.000000 +/- 0.000000, Maximum +0.000069, Minimum
  -0.000060, 99.9pct 0.000051, Mean PPL(Q) 5.886370 +/- 0.110208, PPL ratio 1.000657.
* CONTROL (stock vs its own dump): IDENTICAL to the last digit (mean 0.000000, max
  +0.000069, min -0.000060, ratio 1.000657) -> the +/-7e-5 band is the uint16
  base-logprob quantization floor, not a logit delta. KLD 0.000000 class CONFIRMED
  on the spec-config process shape (reproduces taskV2-moe's +/-6e-5 floor).

## 4. Acceptance + tg/pp per leg (natural suite, bench.py 8 prompts x 3 reps tg256)

| leg           | tg256 med t/s | pp4096 med | accept rate | mean accept len |
|---------------|---------------|------------|-------------|-----------------|
| stock-d4-r1   | 68.13         | 1777.0     | 0.373       | 2.4919          |
| repack-d4-r1  | 68.55         | 1776.5     | 0.373       | 2.4919          |
| stock-d4-r2   | 68.85         | 1780.6     | 0.373       | 2.4919          |
| repack-d4-r2  | 68.04         | 1785.8     | 0.373       | 2.4919          |

* Acceptance BIT-EQUAL across all four legs and digit-matches the taskG5/taskH3 d4
  reference (0.373 / 2.4919) - the repack pair does not move draft acceptance.
* tg256 repack-vs-stock: +0.62pct (r1) / -1.18pct (r2), interleaved -> NEUTRAL on the
  spec stack. The plain-decode +1.82pct repack gain does NOT transfer to DFlash2 d4:
  the width-5 verify step dominates the step (taskG5: w5 verify 22.4 ms vs 9.2 ms plain
  decode) and the repacked nc-5 kernel is bit-identical but not faster at that shape.
  pp4096 neutral both arms (yield path).
* Session note: tg ~68 vs taskG5 70.58 / taskH3 69.70 (-2..-3pct session drift under
  sibling CPU contention); same-session interleaved pairing makes the arm delta valid.
* VRAM at end of measurement: repack 22.40/22.37 GB vs stock 21.62/21.59 GB per GPU
  (GPU2/GPU3) -> dual-layout cost +0.78 GB/GPU on the spec stack, ample headroom on
  32 GiB cards (matches taskV2-moe's +0.79 on plain decode).

## 5. nc 6-8 / RP_Q4_NC_LDS engagement check

The DFlash2 MoE config does NOT engage GGML_CUDA_RP_Q4_NC_LDS:
* the lever is not set in the taskG5/taskH3 serving env (grep of both drivers: 0 hits)
  and absent from every taskW2 leg env-witness;
* the lever routes repacked Q4_0 nc widths, and neither the Q8_0 target (383 Q8_0 +
  368 F32 norms + 2 BF16, ZERO Q4_0 tensors - taskV2-moe inventory) nor the Q8_0
  drafter carries Q4_0 tensors for it to act on;
* zero "RP_Q4_NC_LDS" banner lines in all four d4 server logs.
The step-3 condition ("if the DFlash2 config engages RP_Q4_NC_LDS on the MoE") is FALSE;
no width-window spot-check is required by that clause.

What the repacked nc path DOES see on this config: verify width 5 (d4) -> repacked
Q8_0 nc <2,5> arm (E2-shipped, bit-matched) - sealed end-to-end by sec 2. Verify
width 8 (d7) -> repacked Q8_0 nc <1,8> arm (the E2 1-warp fix, previously
microbench-only on the MoE) - spot-checked by the d7 greedy pair in sec 6.

## 6. d7 width-8 spot-check + pinned-drafter d4 pair

[PENDING - legs re-queued behind taskW1 GPU2 hold via retry-d7.sh]

## 7. Verdicts

1. DFlash2 d4 (replicated drafter, ~70 t/s class): SEAL PASS. Greedy 10/10 BITID on
   both interleaved pairs + both determinism controls, KLD 0.000000 control-identical
   (uint16 floor), acceptance bit-equal to the pre-adoption reference. No config change
   required; the pair is numerics-safe on this spec entry.
2. Speed posture: the pair is tg-NEUTRAL on the spec stack (the +1.82pct plain-decode
   class does not transfer to the width-5 verify step). Adoption on spec entries is a
   numerics-neutral, speed-neutral no-op - the serving gain stays on the plain-decode
   entry. No reason to restrict the pair for speed; none needed for correctness.
3. Fleet posture (=2, shape-gated) for spec entries: SAFE. Mode 2 admits per-model by
   shape hint (3D experts + dense Q8_0 >= 2 GiB floor + tensor split): the MoE target
   qualifies (3D experts, 2.615 GB dense Q8_0, -sm tensor); the 570 MB DFlash2 drafter
   is BELOW the 2 GiB dense floor and is never admitted by the main model's hint
   (per-tensor hint, repack-common.cu:41-65) - it stays canonical. The =1 seal above
   repacked a SUPERSET of the =2 admitted set (target + drafter) bit-identically, so
   the =2 subset is covered. =1 also remains bit-identical on this config, but =2 stays
   the shipping posture (taskV2-moe caveat 1: =1 forces admission for every model in
   the process).
4. [PENDING d7/pinned verdicts]

## 8. Ops notes

* All 7 core legs (4x d4 server + 3x KLD) OK on first attempt; zero bench retries.
* taskW1 sibling contention: one taskW1 leg held GPU2 (10.7 GB) for ~50 min while I
  held gpu.lock - its server started 13 s AFTER my lock acquisition, i.e. that lane
  ran without the lock (worth flagging to the fleet owner). My drain gate correctly
  refused co-scheduling (a 24 GB d7 leg + 10.7 GB sibling > 32 GB card = OOM risk);
  the d7 legs were re-queued via retry-d7.sh rather than run dirty.
* GPU0 never used (quarantined); pair B GPU2+3 per dispatch.

## 9. Artifacts

/root/taskW2/: drive.sh, legs-extra.sh, retry-d7.sh, greedy_probe.py, compare.py,
driver.log, retry.log, arm-status.txt, bin-md5.txt; per-leg
{server.log, witness.txt, bench.log, bench-*.jsonl(.agg.json), greedy-*.jsonl,
greedy-*.log, accept.txt, .done}; kld-{stock-dump,repack,control}.log(+.rc).
