#!/bin/bash
# STEP 2 fix-test: MTP_SAFE guard — lossless re-test + throughput cost
set -u
source /root/w4a4-env.sh
BIN=/root/build-w4a4/bin/llama-server
MODEL=${MODEL:?set MODEL to the 27B community MTP-preserved finetune GGUF path on the box}
PORT=8094
LOGDIR=/root/mtp-fixtest-logs
OUT=/root/mtp-fixtest.txt
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
CREATIVE='{"model":"qwen","messages":[{"role":"user","content":"Write a short story of about 200 words about a lighthouse keeper who finds a bottle washed ashore containing a letter that describes tomorrow in exact detail. Make the ending unexpected and vivid."}],"max_tokens":200,"temperature":0.1}'
greedy(){ curl -s -m 900 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "$BODY" \
  | jq -r '[(.choices[0].message.content // ""), "||R||", (.choices[0].message.reasoning_content // "")] | join("\n")' > "$1"; }
tg(){ local f="$LOGDIR/tg_$1_$2.json"; curl -s -m 900 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "$CREATIVE" > "$f"
  local t; t=$(jq -r '.timings.predicted_per_second // "null"' "$f"); echo "TG leg=$1 run=$2 tg=$t" | tee -a "$OUT"; }
cmpf(){ if diff -q "$2" "$3" >/dev/null 2>&1; then echo "$1: IDENTICAL" | tee -a "$OUT";
  else echo "$1: DIFFERS" | tee -a "$OUT"; diff "$2" "$3" | head -8 | tee -a "$OUT"; fi; }

(
  flock -n 200 || { log "GPU LOCK BUSY"; exit 2; }
  log "=== lock acquired ==="
  export GGML_TP_AR_MTP_SAFE=2   # 2 = log transitions for evidence

  # Lossless legs with guard ON
  start_server S_nomtp || exit 1
  greedy "$LOGDIR/S_nomtp.txt"; stop_server
  start_server S_mtp --spec-type draft-mtp --spec-draft-n-max 6 || exit 1
  greedy "$LOGDIR/S_mtp.txt"; stop_server
  cmpf "MTP-vs-noMTP (MTP_SAFE=2)" "$LOGDIR/S_nomtp.txt" "$LOGDIR/S_mtp.txt"
  echo "guard_transition_logs=$(grep -c 'MTP_SAFE' "$LOGDIR/S_mtp.out")" | tee -a "$OUT"
  grep "MTP_SAFE" "$LOGDIR/S_mtp.out" | head -4 >> "$OUT"

  # Throughput cost of the guard: MTP safe vs MTP unsafe vs baseline (creative x3)
  unset GGML_TP_AR_MTP_SAFE
  start_server P_base || exit 1
  for r in 1 2 3; do tg base_noguard $r; done; stop_server
  start_server P_mtp || exit 1
  for r in 1 2 3; do tg mtp_noguard $r; done; stop_server
  export GGML_TP_AR_MTP_SAFE=1
  start_server Q_mtp_safe --spec-type draft-mtp --spec-draft-n-max 6 || exit 1
  for r in 1 2 3; do tg mtp_safe $r; done; stop_server
  start_server Q_base_safe || exit 1
  for r in 1 2 3; do tg base_safe $r; done; stop_server

  sha256sum "$LOGDIR"/*.txt >> "$OUT"
  log "=== FIXTEST COMPLETE ==="
) 200>/root/gpu.lock
exit $?
