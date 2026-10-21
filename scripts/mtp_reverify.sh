#!/bin/bash
# MTP lossless re-verification on current binary (build-w4a4, post-M4-fix 2e8499bb6)
# STEP 1: reproduce original failing comparison (tensor MTP vs noMTP) + env arms
set -u
source /root/w4a4-env.sh
BIN=/root/build-w4a4/bin/llama-server
MODEL=${MODEL:?set MODEL to the 27B community MTP-preserved finetune GGUF path on the box}
PORT=8094
LOGDIR=/root/mtp-reverify-logs
OUT=/root/mtp-reverify.txt
mkdir -p "$LOGDIR"
SRV_PID=""
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
start_server(){ local leg="$1"; shift
  log "START $leg: $*"
  nohup "$BIN" -m "$MODEL" -c 32768 -b 512 -ub 512 -fa on -sm tensor -tps 3 "$@" \
    --host 127.0.0.1 --port "$PORT" > "$LOGDIR/$leg.out" 2>&1 &
  SRV_PID=$!
  for i in $(seq 1 240); do
    curl -s -m 3 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q '"ok"' && { log "up $leg"; return 0; }
    kill -0 $SRV_PID 2>/dev/null || { log "DIED $leg"; tail -30 "$LOGDIR/$leg.out"; return 1; }
    sleep 5
  done
  log "TIMEOUT $leg"; return 1
}
stop_server(){ [ -z "$SRV_PID" ] && return 0
  curl -s -m 8 -X POST http://127.0.0.1:$PORT/shutdown >/dev/null 2>&1
  for i in $(seq 1 30); do kill -0 $SRV_PID 2>/dev/null || { SRV_PID=""; return 0; }; sleep 2; done
  kill $SRV_PID 2>/dev/null; sleep 3; SRV_PID=""; }
trap 'stop_server' EXIT
BODY='{"model":"qwen","messages":[{"role":"user","content":"List the first ten prime numbers, then one sentence about tides."}],"max_tokens":400,"temperature":0,"top_k":1}'
greedy(){ curl -s -m 900 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "$BODY" \
  | jq -r '[(.choices[0].message.content // ""), "||R||", (.choices[0].message.reasoning_content // "")] | join("\n")' > "$1"; }
warn(){ local c; c=$(grep -c "backend offload failed" "$1" 2>/dev/null) || c=0; echo "$c"; }
cmpf(){ if diff -q "$2" "$3" >/dev/null 2>&1; then echo "$1: IDENTICAL" | tee -a "$OUT";
  else echo "$1: DIFFERS" | tee -a "$OUT"; diff "$2" "$3" | head -8 | tee -a "$OUT"; fi; }

(
  flock -n 200 || { log "GPU LOCK BUSY"; exit 2; }
  log "=== lock acquired ==="
  echo "binary: $($BIN --version 2>&1 | head -1)" | tee -a "$OUT"
  echo "tree: $(git -C /root/mx-production-123b log --oneline -1)" | tee -a "$OUT"
  echo "perflevel: $(cat /sys/class/drm/card*/device/power_dpm_force_performance_level 2>/dev/null | tr '\n' ' ')" | tee -a "$OUT"

  # Legs 1-2: noMTP x2 determinism control (original env: CUSTOM_AR=1 SKIP_XBAR=1)
  start_server nomtp1 || exit 1
  greedy "$LOGDIR/nomtp1.txt"; echo "nomtp1 offload_warnings=$(warn "$LOGDIR/nomtp1.out")" | tee -a "$OUT"; stop_server
  start_server nomtp2 || exit 1
  greedy "$LOGDIR/nomtp2.txt"; echo "nomtp2 offload_warnings=$(warn "$LOGDIR/nomtp2.out")" | tee -a "$OUT"; stop_server
  cmpf "CONTROL nomtp1-vs-nomtp2" "$LOGDIR/nomtp1.txt" "$LOGDIR/nomtp2.txt"

  # Legs 3-4: MTP greedy x2, original env (SKIP_XBAR=1)
  start_server mtp_skip1 --spec-type draft-mtp --spec-draft-n-max 6 || exit 1
  greedy "$LOGDIR/mtp_skip1.txt"; echo "mtp_skip1 offload_warnings=$(warn "$LOGDIR/mtp_skip1.out")" | tee -a "$OUT"; stop_server
  start_server mtp_skip2 --spec-type draft-mtp --spec-draft-n-max 6 || exit 1
  greedy "$LOGDIR/mtp_skip2.txt"; echo "mtp_skip2 offload_warnings=$(warn "$LOGDIR/mtp_skip2.out")" | tee -a "$OUT"; stop_server
  cmpf "MTP-vs-noMTP (SKIP_XBAR=1) leg1" "$LOGDIR/nomtp1.txt" "$LOGDIR/mtp_skip1.txt"
  cmpf "MTP-vs-noMTP (SKIP_XBAR=1) leg2" "$LOGDIR/nomtp1.txt" "$LOGDIR/mtp_skip2.txt"
  cmpf "MTP-vs-MTP determinism (SKIP_XBAR=1)" "$LOGDIR/mtp_skip1.txt" "$LOGDIR/mtp_skip2.txt"

  # Leg 5: MTP with SKIP_XBAR flipped to 0 (without-skip arm)
  export GGML_TP_AR_SKIP_XBAR=0
  start_server mtp_noskip --spec-type draft-mtp --spec-draft-n-max 6 || exit 1
  greedy "$LOGDIR/mtp_noskip.txt"; echo "mtp_noskip offload_warnings=$(warn "$LOGDIR/mtp_noskip.out")" | tee -a "$OUT"; stop_server
  cmpf "MTP-vs-noMTP (SKIP_XBAR=0)" "$LOGDIR/nomtp1.txt" "$LOGDIR/mtp_noskip.txt"

  # Leg 6: MTP with custom AR fully disabled (H3 decisive arm)
  export GGML_ENABLE_CUSTOM_AR=0
  start_server mtp_noAR --spec-type draft-mtp --spec-draft-n-max 6 || exit 1
  greedy "$LOGDIR/mtp_noAR.txt"; echo "mtp_noAR offload_warnings=$(warn "$LOGDIR/mtp_noAR.out")" | tee -a "$OUT"; stop_server
  cmpf "MTP-vs-noMTP (CUSTOM_AR=0)" "$LOGDIR/nomtp1.txt" "$LOGDIR/mtp_noAR.txt"

  sha256sum "$LOGDIR"/*.txt >> "$OUT"
  log "=== STEP1 COMPLETE ==="
) 200>/root/gpu.lock
exit $?
