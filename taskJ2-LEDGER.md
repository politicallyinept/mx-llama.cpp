# taskJ2 LEDGER - LDS-staged Q8_0 narrow mat-vec port (ruquant 82c73b5), env-gated default OFF

Date: 2026-11-12 (PST). Box: mi50 (3x MI50 gfx906, TheRock runtime). No pushes; local commits only.
Worktree: /root/wt-j2 branch int4-j2-lds (branch name uses a hyphen: refs/heads/int4 exists in the
shared repo, so the refnamespaced int4/j2-lds cannot be created). Build: /root/build-j2 via
/root/mi50-build.sh.

## 1. Question + provenance (parent correction applied)

Port ruquant 82c73b5's LDS-staged Q8_0 narrow mat-vec (rq_q8_nc) into the fork as an env-gated,
default-OFF implementation. Claim axis (verified from the commit message): Qwen3.8-27B MAIN MODEL
at Q8_0, kernel mul_mat_vec_repacked_nc, proven shape = ssm_out (5120x6144, 33 MB weights,
~140 MB/call activation re-reads at 8 tokens, 90 -> 70 us) + lm_head at verify widths 4-8.
Upstream claims: one-GPU DFlash2 91 -> 93.3, TP2 105.3 -> 107.4 (+2.0%), TP4 119.7 -> 120.5;
8-token forward 38.6 -> 37.4 ms one-GPU.

## 2. Port design (mechanics, not the diff blindly)

Their kernel sits next to a generic mul_mat_vec_repacked_nc that is the SAME lineage as ours
(rp_traits/load_w/scale/rp_slot_keep/rp_warp_reduce_sum identical semantics). Mapping:

* Placement: kernel rp_q8_nc_lds in q8_repack/repack-kernels.cuh (house style: kernels there,
  dispatch in mul-mat.cu; also includable by the standalone mb harness). Gate + dispatch in
  q8_repack/mul-mat.cu ggml_cuda_mul_mat_repacked_nc_t.
* Bit-identity scope: rp_q8_nc_lds is bit-identical to OUR generic mul_mat_vec_repacked_nc
  <2,1,T,2,64,Q8_0> (same lanes/order/reduction) - verified bit-exact in microbench at every
  shape, W=4 and W=8. Our DEFAULT nc path is the RESTYLE kernel (different accumulation
  granularity), so enabling the gate changes numerics vs the repack-pair default -> KLD gate run.
* Gate: GGML_CUDA_RP_Q8_NC_LDS=<2|4|8|16> waves per workgroup, DEFAULT 0 = OFF (upstream's
  RQ_Q8_NC_WAVES defaults 8 there; fork convention is opt-in). Widths 4-8 only.
* Tall/lm_head: upstream's RQ_Q8_NC_TALL knob has no counterpart here - our tree has no 16-lane
  shape at widths 4-8; the LDS kernel covers tall matrices with the same bit-identity to the
  generic shape (upstream's tall arm replaced a different kernel and was not bit-identical).

## 3. Microbench (mb_q8lds_test.cu, committed; one MI50 under flock)

Bars ALL PASS: rp_q8_nc_lds (W=4, W=8) == generic <2,1,T,2,64,Q8_0> bit-exact at every shape,
both == CPU oracle (oracle_bad=0), launch-deterministic, differs from RESTYLE default as designed.

us/call median (7x30 iters): ssm 5120x6144 T=4 67.2->59.5 (+13.0%), T=5 76.5->64.3 (+18.9%),
T=8 101.7->79.8 (+27.4%; upstream ratio 90->70 = 1.27x reproduced). lm_head 5120x151936 T=4
+14.6%, T=8 +64.9%. 2048x6144 T=8 +32.5%. RESTYLE default ~2x slower than generic at widths 4-8
(ssm T=8: 249.3 vs 101.7 us). W=8 best at width 8; serving A/B used LDS=8.

## 4. Numeric gates

* Greedy 64-token probe (temp0 seed42, DFlash2 d7): ALL ARMS BYTE-IDENTICAL (base, rp, oldrp,
  lds - sha16 77841cb6a251235d, 327 chars). Gate-on does not flip the greedy argmax on the probe.
* Gate-OFF byte-id PROVEN: rp-r1 (new binary, repack pair, LDS unset) vs oldrp-r1 (old binary
  @70b650a459, repack pair): probe byte-identical AND tg256 medians identical to 2dp (23.67 /
  23.67). Dispatch-site never taken when GGML_CUDA_RP_Q8_NC_LDS unset.
* Suite acceptance (all 24 tg256 runs): lds 3777/16233 accepted vs base 3708/16674 (+4.6% rel) -
  accumulation-order drift flips a few accepts; output remains target-model distribution.
* KLD (taskI1 protocol, llama-perplexity -b 8 pp batches exercising the nc width-8 window,
  64KB wikitext subset): base leg DONE rc=0 (j2-base-f16.logits, 3.9 GB f16, kld-base.log).
  lds leg queued behind sibling GPU traffic (follow.sh retry loop, /root/taskJ2/chain4.log);
  result will land in kld-lds.log as "Mean KLD / Maximum KLD" lines. Given the greedy probe is
  byte-identical and suite acceptance differs by <5% relative, expect KLD in the same band as
  taskI1's kernel-reorder precedents; the acceptance-count delta (s4) confirms logits differ
  slightly by design (accumulation order), which is exactly what this protocol bounds.

## 5. End-to-end A/B: 27B Qwen3.8-27B Q8_0 TP2, DFlash2 n_max 7 (8-token verify)

Natural suite tg256 medians (bench.py, 24 runs/arm, interleaved reps):
| arm  | config                                   | tg256 med t/s |
|------|------------------------------------------|---------------|
| base | production env (canonical Q8_0)          | 25.14 / 25.13 |
| rp   | + REPACK_Q8_0 pair, LDS gate OFF         | 23.67         |
| oldrp| old binary + repack pair (gate-off ref)  | 23.67         |
| lds  | rp + GGML_CUDA_RP_Q8_NC_LDS=8            | 39.71 / 39.61 |

DELTA lds vs base: +57.8%. rp vs base: -5.8% (repack pair with RESTYLE nc is a net LOSS on this
stack; the LDS gate is what makes repack pay). rp->lds: +67.7%.

GGML_DFLASH_PROF step phases (median us/call):
| arm  | draft-fwd (M=1) | tgt-decode w8 | tgt w4 | tgt w5 |
|------|-----------------|---------------|--------|--------|
| base | 19180           | 79503         | 83493  | 93932  |
| lds  | 18908           | 48317         | 70122  | 72711  |
Deliverable: drafter fwd ms/step UNCHANGED (~19.0 vs 19.2 ms, -1.4% noise - the gate does not
touch the M=1 draft path, as designed). Verify forward at width 8: -39% (79.5 -> 48.3 ms).
Step-time arithmetic checks out: base 100.3 ms/step vs lds 65.4 ms/step (delta ~ tgt-decode delta).

Why +58% here vs upstream's +2%: their engine SHIPS repack-generic as the default baseline, so
their delta is generic->LDS only (~+2% e2e). Our base arm is the canonical stock path; our lds arm
adopts repack AND LDS. The kernel-level ratio (1.27x on ssm_out) reproduces exactly; the e2e
delta is stack-position-dependent, not a contradiction.

## 5a. Plain decode reference (llama-bench pp512/tg256, TP2, same envs)

First attempt failed on a llama-bench arg bug (-c is not a bench flag); fixed in drive2.sh,
queued in follow2.sh (chain5.log). Results land in plain-base-r1.jsonl / plain-lds-r1.jsonl
(agg.py prints pp512/tg256 medians). Context: on the Q4_0 stack (taskI2) plain decode was 46.99
vs DFlash-on 38.10; on this Q8_0 stack DFlash2-on lds already runs 39.66 t/s, so the port moves
DFlash2-on decisively toward (and likely past) plain Q8_0 decode, whose bandwidth floor is
~29 GB/step / ~1.2 TB/s TP2 aggregate ~= 24 t/s.

## 5b. ruquant engine same-protocol (parent item 3)

rq arm = their engine @82c73b5 (build-82c73b5), same 27B Q8_0 + our DFlash2 Q8_0 drafter, TP2
tensor, their recipe env (META_TOKEN_GRAPH=0, META_SPIN=2e6, DISABLE_GRAPHS=1). CRASHED at model
load: ggml_abort in ggml_backend_sched_backend_id_from_cur <- split_graph <- resolve_fused_ops
during the DRAFTER ctx sched_reserve (main ctx reserved fine). Their repack posture is
unconditional (no env gate in their tree), so their Q8_0 repack + TP meta split + dflash drafter
path aborts on this config on our box. Retry with RQ_FUSE_*=0: see below. Consequence: their
published 91-107 t/s DFlash2 numbers are NOT reproducible on this box with their own engine at
Q8_0 main + Q8_0 drafter; our fork at lds reaches 39.7 t/s on the same protocol (their 60.5
published figure is the Q4_0 stack per taskI2 s4/s5 - different axis).

## 6. Third item survey: W4A4 skinny-kernel perf (b9543c0)

Per-wave K-order rotation (RQ_SKINNY_ROT=7) vs DRAM channel conflicts in their W4A4 verify-width
kernel; bit-exact (integer partial sums); 40.4->39.2 ms one-GPU verify. NOT FEASIBLE as direct
port: our fork has no W4A4/W4A4H dtypes (GGML_TYPE_COUNT=43). Transferable idea (rotate per-wave
K-group walk order in our repacked/dp8a narrow kernels) targets a different kernel family here;
medium effort, own A/B needed - NOT trivial, not implemented per instruction.

## 7. Verdict

PORT LANDED, GATES GREEN, LARGE WIN ON THIS STACK:
* Env gate GGML_CUDA_RP_Q8_NC_LDS default OFF; gate-off byte-identical to old binary (proven).
* Kernel bit-exact vs our generic repacked nc shape; greedy probe byte-identical base vs lds.
* 27B Q8_0 DFlash2 8-token verify: +57.8% end-to-end (25.13 -> 39.66 t/s), verify fwd -39%,
  draft fwd unchanged. Toward plain decode: see s5a (lds 39.66 vs Q8_0 plain floor ~24 t/s).
* Upstream kernel ratio reproduces (1.27x ssm_out); upstream e2e +2% is their baseline position
  (repack-generic default), ours measures canonical->repack+LDS.
* rq engine itself crashes on this config (their bug, backtrace in rq-r1.server.log).

## 8. Artifacts

* Branch int4-j2-lds in /root/wt-j2: kernel + gate (repack-kernels.cuh, q8_repack/mul-mat.cu),
  mb_q8lds_test.cu microbench, this ledger. NOT pushed (parent merges).
* /root/taskJ2/: drive.sh, chain.log, chain2.log, per-arm server logs, reps.jsonl, probes,
  agg.py, kld logs, plain-bench jsonl.
* /root/mb_q8lds_test binary; /root/build-j2 (new binary @ int4-j2-lds).
