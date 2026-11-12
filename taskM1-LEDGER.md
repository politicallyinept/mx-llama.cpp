# task M1 LEDGER - INT4-noquant gate-OFF neutrality vs CV-port base (35B MoE, mi50)

Date: 2026-11-11 (PST). Branch INT4-noquant @ 7129719aa5 (post-history-rewrite;
compiled code identical to the pre-rewrite validated tree - the rewrite touched
docs/*.md only).

## Question
Is the merged INT4-noquant branch, gate OFF / default env (GGML_CUDA_I8NARROW and
GGML_DFLASH_PROF unset), NEUTRAL vs the CV-port base on
/root/moe-nextn/Qwen3.5-35B-A3B-UD-Q4_K_XL-nextn.gguf?

## Binaries (same toolchain, same cmake flags)
- base:     /root/cvport-base-wt/build/bin/llama-server @ 1062001ac8 (CV-port tip),
            built 2026-11-11 19:32 with TheRock amdclang++ HIP compiler,
            f0_build.sh flag set (GGML_HIP, gfx906, BACKEND_DL, HIP_GRAPHS,
            HIP_NO_VMM, HIP_RCCL, Release).
- int4off:  /root/int4-noquant-wt/build/bin/llama-server @ 7129719aa5, same flags,
            incremental rebuild confirmed current.
- /root/build-cvport-rebase REJECTED as baseline: --version reports commit
  5113c4cc5 (35 files behind CV-port tip) and it was configured with the apt
  /opt/rocm-7.2.3 HIP compiler (violates the TheRock-only rule).

## Method
- Serving A/B: /root/m1_neutral_ab.sh. GPU2 only (HIP_VISIBLE_DEVICES=2), single
  flock window on /root/gpu.lock, RAM guard >=12GB, VRAM drain on GPU2, timeout
  14400, TERM-only teardown, health polled via curl. Env: production-env.sh +
  GGML_TP_AR_MTP_SAFE=1 (gpu_ab_36.sh precedent). Server flags mirror the
  ab-ud-off control arm (no spec flags): -ngl 99 -fa on -c 16384 -b 4096
  -ub 4096 -t 16 -np 1 -lm dio -fit off. Client: /root/mtp-final/mtp_bench.py
  (pp4096 x3 + tg256 x3, temp 0 seed 42, 2 warmups).
- Gate-OFF witness: both server logs show NO [I8NARROW] banner and 0
  DFLASH-PROF lines; identical w4a4 resolved banner (DOT8=3 MIN_TOKENS=128
  KSPLIT=1 SUMW=1).
- KLD spot-check: /root/m1_kld_spot.sh, kld_leg.sh pattern - CV-port
  llama-perplexity saves logits (--kl-divergence-base), fork llama-perplexity
  scores them (--kl-divergence). -c 2048 -b 2048 --chunks 16, wikitext
  wiki.test.raw, GPU2 only.

## Results (median of 3 reps per cell)
| arm     | pp4096 ms | pp t/s   | tg256 ms | tg t/s  |
|---------|-----------|----------|----------|---------|
| base    | 1951.038  | 2099.395 | 2385.523 | 106.895 |
| int4off | 1950.236  | 2100.259 | 2392.478 | 106.584 |

Delta pp: +0.04% (int4off marginally faster). Delta tg: -0.29%. Both inside the
rep-to-rep spread (<=0.3% within each arm).

KLD (INT4-off vs base logits): Mean KLD -0.000000 +/- 0.000000, Median -0.000000
- logits identical at fp16 logits-file storage precision. Reference point: the
kld-ladder A0 control (a real quant change) reads 0.0499. PPL(Q)/PPL(base) =
1.000285 +/- 0.000215 (fp16 storage artifact of the base file, not a model delta).

## Verdict
NEUTRAL. Gate-OFF INT4-noquant is behaviorally identical to the CV-port base
(KLD 0.000000) and statistically within timing noise (pp +0.04%, tg -0.29%).

## Artifacts
- /root/moe-nextn/m1-neutral/  (m1-base.jsonl, m1-int4off.jsonl, server logs, witnesses)
- /root/moe-nextn/m1-kld/      (kld-base.log, kld-int4off.log, moe-base.kld 8.1GB - reclaimable)
- scripts: /root/m1_neutral_ab.sh, /root/m1_kld_spot.sh, /root/m1_cvport_build.sh
