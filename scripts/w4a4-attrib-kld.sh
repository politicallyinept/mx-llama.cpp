#!/usr/bin/env bash
# w4a4-attrib-kld.sh — per-tensor-class KLD attribution for W4A4 (perf/w4a4-attrib).
# Legs: control (gate off), attn-only u4act, ffn-only u4act, all (gate1).
# Single GPU ROCm1, -c 512 -b 512 -ub 512 (n_seq=1 clean path), 128 chunks.
set -u
source /root/w4a4-env.sh
BIN=/root/build-w4a4/bin
Q40=/mnt/old/home/brad/models/qwen35-9b-upstream/Qwen_Qwen3.5-9B-Q4_0.gguf
WT=/root/mx-production-123b/scripts/wikitext-2-raw/wiki.test.raw
LOGITS=/mnt/old/home/brad/qwen35-bench/f16-c512-ch128.logits
LOGD=/root/w4a4-attrib-logs
mkdir -p "$LOGD"
OUT=/root/w4a4-attrib-kld.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$OUT"; }

run_kld(){ # $1=name [extra env assignments...]
  local name=$1; shift
  log "=== KLD leg: $name ($*) ==="
  # shellcheck disable=SC2086
  timeout --signal=TERM 1200 flock /root/gpu.lock env "$@" \
    "$BIN/llama-perplexity" -m "$Q40" -f "$WT" -c 512 -b 512 -ub 512 -ngl 999 \
      -lm dio --device ROCm1 --kl-divergence-base "$LOGITS" --kl-divergence --chunks 128 \
      > "$LOGD/kld-$name.log" 2>&1
  local rc=$?
  log "rc=$rc | $(grep -E 'Mean    KLD|99.9%|99.0%|Median  KLD|Maximum' "$LOGD/kld-$name.log" | tr '\n' ' ')"
}

log "start; build=$("$BIN/llama-perplexity" --version 2>/dev/null | head -1)"
run_kld control  GGML_CUDA_REPACK_Q4_0=1
run_kld attnonly GGML_CUDA_REPACK_Q4_0=1 GGML_CUDA_W4A4_DOT8=1 GGML_W4A4_ONLY=attn
run_kld ffnonly  GGML_CUDA_REPACK_Q4_0=1 GGML_CUDA_W4A4_DOT8=1 GGML_W4A4_ONLY=ffn
run_kld all      GGML_CUDA_REPACK_Q4_0=1 GGML_CUDA_W4A4_DOT8=1
log "ALL KLD LEGS DONE"
