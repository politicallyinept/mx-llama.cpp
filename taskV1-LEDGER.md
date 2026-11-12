# taskV1 LEDGER - end-to-end un-gating gate: REPACK_Q8_0 vs STOCK on the real 35B MoE

Date: 2026-11-11 (PST). Branch INT4-noquant (tip 68a757a411 at run time; binary built
at d496c12154). Artifacts: /root/taskV1/. No pushes.

## Verdict

UN-GATING BIT-SAFE. REPACK (GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0,
yield default-ON) is end-to-end bit-identical to STOCK defaults on the real
Qwen3.6-35B-A3B Q8_0 MoE: greedy 10/10 probes BITID (text sha + per-token logprob sha,
0 differing tokens, max logprob delta 0.000e+00), KLD mean -0.000000 +/- 0.000000 with
a stock-vs-stock control proving the per-token +/-6e-5 spread is the uint16
base-logprob quantization floor, not a logit delta.

## Setup

- Model: /root/bench-quants/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf
  (37,812,647,552 B = 35.2 GiB; qwen35moe, 41 blocks, 256 experts / 8 used;
  Q8_0 37.70 GB = dense 2.48 GB + expert 35.22 GB; F32 0.10 GB).
- Build: /root/int4-noquant-wt/build/bin (llama-server md5 c4fd0ba7e28290cfdb8b10c2524b69ce,
  libggml-hip.so b535d1b4f1d1dbe13aa681d502dfd281 - identical before and after all legs).
- TOPOLOGY DEVIATION (recorded): serving on TP2 GPU1+GPU2 (-sm tensor -fit off,
  HIP_VISIBLE_DEVICES=1,2), NOT GPU2-only. The 35.2 GiB weight file exceeds one 32 GiB
  card; GPU2-only serving of this model is physically impossible. Sibling taskV2 made
  the identical call (68a757a411). All legs under flock -w 600 /root/gpu.lock, RAM guard
  >=12 GB, VRAM drain <2 GB/card gates, TERM-only teardown (PID-targeted, no double-TERM,
  no kill -9).
- Serving line (both arms): -ngl 99 -fa on -sm tensor -fit off -c 16384 -b 4096
  -ub 4096 -t 16 -np 1 -lm dio, host 127.0.0.1, production-env.sh bundle +
  GGML_TP_AR_MTP_SAFE=1. No spec/MTP flags (mirrors taskM1 off-control config).
- Arms: STOCK = bundle defaults (REPACK_Q8_0 unset -> mode 0, STOCK_ACC default-ON);
  REPACK = + GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0 (YIELD unset =
  default-ON, per the E5 requirement "un-gating gate must keep the yield").
- Engagement witnesses (server logs):
  * repack arm: "GGML_CUDA_REPACK_Q8_0 gate ACTIVE (tensor 'blk.0.attn_qkv.weight',
    env mode 1)" + functional witness "q8_repack: repacking q8_0 tensors on device 0"
    AND "device 1" (both TP2 ranks) + "GGML_CUDA_REPACK_Q8_0_YIELD gate ACTIVE" +
    RESTYLE ACTIVE; no STOCK_ACC banner.
  * stock arm: "GGML_CUDA_REPACK_Q8_0 gate inactive (env mode 0)" + STOCK_ACC ACTIVE
    banner; no repacking witness.
  * /proc/<pid>/environ witnesses in stock.env-witness / repack.env-witness.

## Leg 1: greedy probes (temp 0, seed 42, max_tokens 64, /v1/completions logprobs=1)

Suite (v1_greedy.py; prompts calibrated to EXACT token counts via /tokenize):
p0008 p0012 p0024 p0030 p0033 p0064 p0512 p2048 p4112 p4128.
Class coverage: 12/24/30 = declined class 9..31 (repack GEMM); 33/64/512/2048 =
yield class >=32 (dense prefill yields to gcn_kq canonical copy); 4112/4128 =
chunked prefill (4096 + 16/32, last chunk lands declined/yield respectively);
every probe's 64 decode steps exercise the N=1 repacked MMV path; MoE routing
exercises HAS_IDS repacked expert GEMMs at every width.
Results (v1_compare.py, verdict.txt):
- Determinism control: stock r1 vs stock r2 = 10/10 BITID (harness valid).
- THE GATE: stock r1 vs repack r1 = 10/10 BITID. text sha EQ, logprob-payload sha EQ,
  tokdiff=0, max_logprob_delta=0.000e+00 on every probe. No empty captures (guard on).
- DETERMINISM: PASS. GREEDY-BIT-ID: PASS.

## Leg 2: KLD ladder (llama-perplexity anchor, taskM1 protocol)

wiki.test.raw, -c 2048 -b 2048 --chunks 16 (32768 positions). Stock run dumps
--save-all-logits v1-base.logits; repack run --kl-divergence-base vs that dump.
- repack vs stock: Mean KLD -0.000000 +/- 0.000000; Maximum +0.000063; Minimum -0.000061;
  Mean PPL(Q) 5.891720 +/- 0.110374 (stock dump run: PPL 5.8917).
- CONTROL (stock env vs its OWN dump, v1_kld_control.sh): Mean KLD -0.000000 +/- 0.000000;
  Maximum +0.000063; Minimum -0.000061; Mean PPL(Q) 5.891720 +/- 0.110374;
  Mean ln(PPL(Q)/PPL(base)) 0.000667 - IDENTICAL statistics to the repack leg.
Interpretation: the fork stores base log-probs as uint16 (scale*grid + min, see
tools/perplexity/perplexity.cpp log_softmax); the +/-6e-5 per-token spread and the
0.000667 ln-PPL offset are that quantization floor and appear identically in a
self-comparison. The candidate PPL(Q) is bit-identical to the base run's PPL at 6 dp,
i.e. the repack arm's logits equal stock's. KLD 0.000000 class CONFIRMED; the 0.05
real-quant reference class is not approached.

## Not covered by this leg (scope notes)

- nc verify widths 6-8 (E2 class) are not exercised end-to-end here: no spec-decode
  flags in the serving config. E2's microbench (42/42) + 8B e2e remain the evidence.
- VRAM: repack arm measured ~19.2 GB/GPU at peak vs ~18.4 stock; the dual-layout
  (repacked record + canonical copy) cost is bounded to the 2.48 GB dense Q8_0 subset
  (experts carry no canonical copy: ggml_cuda_q8_yield_eligible requires ne[2]==1).

## Ops notes

- This fork's /tokenize takes "content", NOT "prompt" (server-context.cpp
  post_tokenize); a "prompt" body silently returns {"tokens":[]} and a calibration
  loop on it hangs forever (first driver attempt died this way; guard added).
- A python co-process launched inside the flock window inherits fd 9 and holds the
  lock after the driver exits if not closed: launch with 9>&- (same class as the
  known llama-server fd-9 inheritance trap).
- Cold dio load of the 37.8 GB file: ~26 s to health on this NVMe; llama-perplexity
  anchor leg ~2 min. v1-base.logits (~20 GB) deleted after the control leg.
