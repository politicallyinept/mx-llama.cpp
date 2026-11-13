# taskM ledger - 35B Q8_0 MoE regression + q4_0-KV extension on the merged J build

Date: 2026-11-13 (PST). Branch int4-m-moe35 @ 15196ec20f (INT4-noquant tip at
branch time; worktree /root/wt-m-moe35). Build under test: /root/build-jmerge
(BUILD-PROVENANCE src_sha a63c468d60 = pre-reword twin of 15196ec20f;
`git diff a63c468d60 HEAD` empty, so the binary carries the merged tree).
All three J-wave gates verified present in the binaries (strings:
GGML_CUDA_FA_DOT8 + GGML_CUDA_RP_Q8_NC_LDS in bin/libggml-hip.so.0.26.0,
LLAMA_KV_KROT in bin/libllama.so.0.6.0) and default OFF; every leg's
env-witness shows no gate env set on regression arms.

Stack: Qwen3.6-35B-A3B Q8_0 (37.8 GB) TP2, DFlash2-Q8_0 drafter
(replicated /root/bench-quants/dflash/..., pinned own-head
/root/taskH3/draft/DFlash2-Q8_0-ownhead.gguf via -devd ROCm0).
Natural suite /root/p2-prefill/bench.py (8 prompts x 3 reps tg256 + pp4096).
Regression arms replicate the baseline drivers byte-for-byte except BIN:
plain decode = taskI2 config (production-env.sh, -c 65536);
DFlash2 arms = taskH3 config (w4a4-env.sh + ALLREDUCE=internal
REPACK_KQUANT=0 GDN_COLTHREAD=1, -c 16384 -ub 1024).
Driver /root/taskM-drive.sh (+ drive2 supplementary), outputs /root/taskM-moe35/.
Doctrine: flock /root/gpu.lock -w 900 per leg with retry loop, TERM-only
teardown, pgrep -x llama-server empty after each leg, TheRock runtime
witnessed per leg, refuse-if-exists + DONE markers.

## 1. REGRESSION: merged build at default gates vs baselines - NEUTRAL

| arm | this build | baseline (taskG5/H3/I2) | delta | acceptance |
|---|---|---|---|---|
| plain decode r1/r2   | 104.81 / 106.73 | 106.81 / 106.83 | -1.9% / -0.1% | n/a |
| DFlash2 d4 replicated| 69.33  | 69.18-70.72 | within band | 0.373 / 2.4919 BIT-EXACT |
| DFlash2 d4 pinned r1/r2 | 79.71 / 79.77 | 79.54-79.69 | +0.1% | 0.373 / 2.4919 BIT-EXACT |

Acceptance is bit-exact to the baseline ledgers (draft_n 9789 / accepted 3651,
per-prompt acceptances identical to 4 dp), so the verify-path numerics are
unchanged by the merge. tg deltas sit inside the baselines' own spread
(the plain-decode baseline itself contains an 88.57 contention rep; the two
clean reps bracket 106.8 and my r2 lands at 106.73, -0.1%). pp medians:
plain 2643.96 vs 2641 baseline (+0.1%), d4 1915-1945 vs 1907-1945 (band).

KLD re-score (mandatory gate, taskI1 protocol: llama-perplexity, wikitext
16ch x 2048, TP2, vs existing f16 base logits /root/taskI1/i1-base-f16.logits,
q4_0 KV arm): Mean KLD 0.011223 +/- 0.000237, Max 1.351633,
PPL ratio 1.006371 (+0.64%). Reproduces taskI1/taskJ1 references to the last
digit. The shipped q4_0-KV numerics envelope is untouched by the merge.

Greedy probes (64-token temp0 seed42 raw /completion, fixed prompt): byte-
identical (sha 445cc9d6d4fa...) across plain, d4 rep/pin f16-KV, AND both
q4_0-KV arms. Caveat: the protocol prompt degenerates into a repetition loop
that is insensitive to the KV numerics shift; the probe passes but is a weak
discriminator - acceptance + KLD carry the numeric weight.

VERDICT: the J-wave merge is NEUTRAL for 35B MoE serving at default gates.

## 2. EXTENSION: q4_0 KV (-ctk q4_0 -ctv q4_0) on the 35B DFlash2 stack

All extension arms h3 bundle (same as baselines); draft inherits q4_0 KV.

| arm | tg | vs f16-KV same placement | accept / len | vs plain decode 106.8 |
|---|---|---|---|---|
| d4 pinned + q4_0 KV | 92.74 | +16.4% (79.62-79.77) | 0.4136 / 2.6543 | -13.2% |
| d4 replicated + q4_0 KV | 87.43 | +26.1% (69.33) | 0.4136 / 2.6543 | -18.1% |
| d7 pinned + q4_0 KV | 58.27 | -9.3% (64.28) | 0.2218 / 2.5528 | -45.4% |

Placement invariance holds under q4_0 KV: rep and pin acceptance are
bit-identical (0.4136/2.6543), same as f16-KV.

J1 transfer verdict: PARTIAL. The direction transfers - q4_0 KV raises MoE
acceptance (0.373 -> 0.4136, cf. 27B 0.348 -> 0.446) and lifts DFlash2 d4 by
+16.4% (pin) / +26.1% (rep). What does NOT transfer: the flip to net-positive
over plain decode. The MoE plain decode (106.8) is far above the 27B ratio;
92.74 is still -13.2% short. At d7 the q4_0-KV numerics hurt acceptance
(0.2362 -> 0.2218) and the arm is -9.3% vs f16-KV d7. Net: on the 35B MoE,
q4_0 KV + DFlash2 d4 is the best spec-config but plain decode remains the
fastest serving mode; per Brad's standing order the q4_0-KV DFlash2 config is
NOT "positive" vs plain on this model (it is positive vs f16-KV DFlash2).

Cross-check: the prod-bundle replicated q4_0 arm (76.23 / 0.3508 / 2.4031)
reproduces taskI1's q40 leg (76.50 / 0.3508 / 2.4031) - bundle-dependent,
deterministic per bundle.

## 3. LDS arm: GGML_CUDA_RP_Q8_NC_LDS=8 is INERT on the 35B MoE

Tensor map: 753 tensors, 383 Q8_0 (output/token_embd [2048,248320], ssm_out
[4096,2048], attn_qkv/attn_gate all Q8_0; 137 non-expert Q8_0). The LDS
kernel routes only REPACKED Q8_0 nc widths 4-8, and Q8_0 repack admission is
OFF by default: log shows "REPACK_Q8_0 gate inactive (env mode 0)" plus the
default-ON "REPACK_Q8_0_STOCK_ACC ... repack suppressed for Q8_0". With
RP_Q8_NC_LDS=8 alone: no LDS banner, kernel unreachable. With
REPACK_Q8_0=2 + RP_Q8_NC_LDS=8 (lds2-d4-pin-r1): admission still inactive
(mode 2 shape hint overridden by STOCK_ACC suppression), no banner,
acceptance bit-identical 0.373/2.4919, tg 79.42 (within pin noise).
The 27B +58pct LDS effect does NOT transfer: engaging it on MoE would require
forcing Q8_0 repack (REPACK_Q8_0=1 + STOCK_ACC off), a separate and much
larger numerics lever than the LDS gate itself.

## 4. Incidental finding: the 35B DFlash2 stack is bundle-sensitive

A driver env leak (see 5) produced two arms running the production bundle
(MOE_VEC/MOE_FUSE/CSAR/RPB1/KMAX) instead of the taskH3 bundle:
pin d4 f16-KV moved 79.7 -> 90.1 (+13%) with acceptance 0.373 -> 0.3581.
The 35B DFlash2 baseline is bundle-sensitive; taskH3/G5 baselines are only
comparable within the h3 bundle (as done here).

## 5. Ops notes

- Sibling contention (taskK + taskL chains) held /root/gpu.lock nearly
  continuously 04:00-06:45; several 900 s LOCKFAILs, absorbed by the per-leg
  retry loop (no leg lost). All legs DEVS=1,2 (baselines' plain-decode legs
  ran 0,1; DFlash2 baselines ran 1,2).
- Bug fixed mid-run: leg() used `shift 8` with 7 params; under set -u the
  shift failed and the label leaked into the server argv (invalid argument,
  legs died at load). Fixed to shift 7; failed legs re-ran clean.
- Env leak: serve()/bundle() export into the driver shell, so production-bundle
  vars from the plain-decode r2 leg leaked into two later arms (classic
  multi-cell export leak). Caught via per-leg env-witness; supplementary legs
  re-run in fresh shells (drive2). The affected arms are labeled and excluded
  from headline claims.
- TERM-only teardown throughout; pgrep -x llama-server empty after every leg;
  no kill -9, no wedges.

## 6. Files

- /root/taskM-drive.sh, /root/taskM-drive2.sh (drivers, also in this commit)
- /root/taskM-moe35/ (per-leg logs, server logs, witnesses, reps.jsonl,
  greedy probes, kld-q40.log)
- Branch int4-m-moe35: this ledger + drivers. NOT pushed.
