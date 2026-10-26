#!/usr/bin/env bash
# w4a4-attrib-dump.sh — activation dumps for outlier localization.
# Each leg: gate1 W4A4 perplexity, 16 chunks; the GGML_W4A4_DUMP hook in
# quantize_mmq_q8_1_u4_cuda writes the raw fp32 activations of the named
# tensor's first 16 prefill calls to /root/w4a4-actdump.bin.
set -u
source /root/w4a4-env.sh
BIN=/root/build-w4a4/bin
Q40=/mnt/old/home/brad/models/qwen35-9b-upstream/Qwen_Qwen3.5-9B-Q4_0.gguf
WT=/root/mx-production-123b/scripts/wikitext-2-raw/wiki.test.raw
LOGD=/root/w4a4-attrib-logs
DUMP=/root/w4a4-actdump.bin
mkdir -p "$LOGD"
rm -f "$DUMP"
OUT=/root/w4a4-attrib-dump.log
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$OUT"; }

dump_leg(){ # $1=tensor-substring
  local t=$1
  log "=== dump leg: $t ==="
  timeout --signal=TERM 600 flock /root/gpu.lock env \
    GGML_CUDA_REPACK_Q4_0=1 GGML_CUDA_W4A4_DOT8=1 \
    GGML_W4A4_DUMP="$t" GGML_W4A4_DUMP_N=16 GGML_W4A4_DUMP_FILE="$DUMP" \
    "$BIN/llama-perplexity" -m "$Q40" -f "$WT" -c 512 -b 512 -ub 512 -ngl 999 \
      -lm dio --device ROCm1 --chunks 16 > "$LOGD/dump-$t.log" 2>&1
  log "rc=$? size=$(du -h "$DUMP" | cut -f1)"
}

dump_leg blk.10.ffn_gate
dump_leg blk.10.ffn_down
dump_leg blk.10.attn_qkv
dump_leg blk.11.attn_q
dump_leg blk.20.ffn_gate
dump_leg blk.5.ffn_gate
log "ALL DUMP LEGS DONE"
