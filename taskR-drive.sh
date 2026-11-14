#!/bin/bash
# taskR driver: own-head pinned drafter (-devd) A/B vs borrowed-head on the 27B stacks.
# Stacks (build-qmerge @ 2a80a1c8ad, TP2, DFlash2-Q4_0 drafter):
#   f0q4     = 27B Q4_0 + floor-0 + REPACK_Q8_0 pair + RP_Q8_NC_LDS=8 + RP_Q4_NC_LDS=8  [best d4 56.04 / d7 44.30]
#   f128q4w4 = 27B Q4_0 + REPACK_Q8_0 pair + RP_Q8_NC_LDS=8 + RP_Q4_NC_LDS=4            [quality-neutral d4 54.30 / d7 33.96]
#   q8lds    = 27B Q8_0 + REPACK_Q8_0 pair + RP_Q8_NC_LDS=8                             [taskL q4d d4 50.44 / d7 40.61]
# Head arms: b = borrowed-head stock drafter (no -devd); p = own-head grafted + -devd ROCm0
#            (first visible card; H3-proven placement).
# Doctrine: flock /root/gpu.lock -w 900, TERM-only teardown, pgrep -x llama-server empty
# after each leg, production env witnessed via mi50-run.sh, refuse-if-exists.
set -u -o pipefail
BASE=/root/taskR
BIN=/root/build-qmerge
M4=/mnt/old/home/brad/models/Qwen3.8-27B-Q4_0.gguf
M8=/mnt/old/home/brad/models/Qwen3.8-27B-Q8_0.gguf
D=/mnt/old/home/brad/models/Qwen3.8-27B-DFlash2-Q4_0.gguf
D4P=/root/taskR/draft/DFlash2-Q4_0-ownhead-q4.gguf
D8P=/root/taskR/draft/DFlash2-Q4_0-ownhead-q8.gguf
GPROMPT="The engineer opened the notebook and began to write about the machine."
mkdir -p "$BASE"

ARMS="GGML_CUDA_W4A4_MIN_TOKENS GGML_CUDA_REPACK_Q8_0 GGML_CUDA_REPACK_Q8_0_STOCK_ACC GGML_CUDA_RP_Q8_NC_LDS GGML_CUDA_RP_Q4_NC_LDS"

arm_env() {
  case "$1" in
    f0q4)     echo "GGML_CUDA_W4A4_MIN_TOKENS=0 GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0 GGML_CUDA_RP_Q8_NC_LDS=8 GGML_CUDA_RP_Q4_NC_LDS=8" ;;
    f128q4w4) echo "GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0 GGML_CUDA_RP_Q8_NC_LDS=8 GGML_CUDA_RP_Q4_NC_LDS=4" ;;
    q8lds)    echo "GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0 GGML_CUDA_RP_Q8_NC_LDS=8" ;;
    *) return 1 ;;
  esac
}
arm_model()    { case "$1" in f0q4|f128q4w4) echo "$M4" ;; q8lds) echo "$M8" ;; *) return 1 ;; esac; }
arm_drafter()  { case "$1" in f0q4|f128q4w4) echo "$D4P" ;; q8lds) echo "$D8P" ;; *) return 1 ;; esac; }

gates() { # sets DEVS, holds fd 9
  local i
  for i in $(seq 1 40); do
    awk '{exit !($1<16)}' /proc/loadavg && break
    [ "$i" = 40 ] && { echo "GATE-LOAD-FAIL"; return 97; }
    sleep 30
  done
  exec 9>/root/gpu.lock
  flock -w 2700 9 || { echo "LOCKFAIL"; return 1; }
  awk '/MemAvailable/{exit !($2>=12582912)}' /proc/meminfo || { echo "GATE-RAM-FAIL"; flock -u 9; return 99; }
  for i in $(seq 1 40); do
    DEVS=$(python3 -c '
import subprocess
o=subprocess.run(["rocm-smi","--showmeminfo","vram"],capture_output=True,text=True).stdout
free=[]
for l in o.splitlines():
    m=__import__("re").match(r"GPU\[(\d+)\].*Used Memory \(B\):\s*(\d+)",l)
    if m and int(m.group(2))<2*1024**3: free.append(m.group(1))
print(",".join(free[:2]) if len(free)>=2 else "")')
    [ -n "$DEVS" ] && break
    echo "GPU-BUSY retry $i $(date -Is) foreign=$(pgrep -x llama-server | tr '\n' ' ')"
    sleep 30
  done
  [ -n "$DEVS" ] || { echo "GATE-GPU-FAIL"; flock -u 9; return 96; }
  echo "QUIET total-used-B=$(rocm-smi --showmeminfo vram | grep 'Used Memory' | awk -F': ' '{s+=$2} END {print s}') foreign=$(pgrep -c -x llama-server || true)"
}

teardown() { # pid
  kill -TERM "$1" 2>/dev/null
  local i
  for i in $(seq 1 60); do kill -0 "$1" 2>/dev/null || break; sleep 5; done
  kill -0 "$1" 2>/dev/null && { echo "TEARDOWN-STUCK pid=$1"; return 90; }
  sleep 5
  [ -z "$(pgrep -x llama-server)" ] || echo "WARN llama-server present after teardown: $(pgrep -af -x llama-server | head -2)"
  flock -u 9
}

witness() { # $1 pid $2 file
  tr '\0' ' ' < /proc/$1/cmdline > "$2"; echo >> "$2"
  tr '\0' '\n' < /proc/$1/environ | grep -E '^(GGML_|HSA_|HIP_|LD_LIBRARY|RQ_|LLAMA_)' | sort >> "$2"
  grep -oE '/[^ ]*lib(amdhip64|rccl|rocblas)[^ ]*' /proc/$1/maps | sort -u >> "$2"
  grep -q "/usr/lib.*libamdhip64" "$2" && echo "RUNTIME-FAIL apt-libamdhip64" >> "$2"
}

measure() { # $1 TAG $2 LOG $3 depth
  local TAG=$1 LOG=$2 N=$3
  python3 /root/p2-prefill/probe_pp.py "$PORT" 6 >/dev/null 2>&1
  local i
  for i in 1 2 3; do curl -s -m 600 "http://127.0.0.1:$PORT/completion" -d \
    '{"prompt":"Warmup pass for the decode shape.","n_predict":256,"temperature":0,"seed":42,"cache_prompt":false,"ignore_eos":true}' >/dev/null; done
  timeout 5400 python3 /root/p2-prefill/bench.py "$PORT" "$TAG" "$BASE/$TAG.reps.jsonl" "$N" > "$BASE/$TAG.bench.log" 2>&1
  local rc=$?
  curl -s -m 300 "http://127.0.0.1:$PORT/completion" -d \
    "{\"prompt\":\"$GPROMPT\",\"n_predict\":64,\"temperature\":0,\"seed\":42,\"ignore_eos\":true}" \
    > "$BASE/probe-$TAG.json"
  grep -E 'draft acceptance = ' "$LOG" | tail -3 >> "$BASE/$TAG.bench.log"
  grep -c "DFLASH-PROF" "$LOG" > "$BASE/$TAG.prof-count"
  return $rc
}

leg() { # $1 arm(f0q4|f128q4w4|q8lds) $2 head(b|p) $3 depth $4 rep $5 port
  local arm=$1 HEAD=$2 N=$3 REP=$4 PORT=$5 TAG DR LOG SPID rc ENVV M DFLAGS
  TAG="$arm-$HEAD-d$N-r$REP"; LOG=$BASE/"$TAG".server.log
  [ -e "$BASE/$TAG.done" ] && { echo "$TAG SKIP-EXISTS"; return 0; }
  ENVV=$(arm_env "$arm") || { echo "BAD-ARM $arm"; return 2; }
  M=$(arm_model "$arm"); DR=$(arm_drafter "$arm")
  DFLAGS=()
  if [ "$HEAD" = p ]; then DFLAGS=(-devd ROCm0); else DR="$D"; fi
  gates || return $?
  echo "$TAG start $(date -Is) DEVS=$DEVS arm=$arm head=$HEAD depth=$N env[$ENVV]"
  sha256sum "$M" "$DR" > "$BASE/$TAG.witness"
  source /root/production-env.sh
  export HSA_KERNARG_POOL_SIZE=67108864
  unset $ARMS
  [ -n "$ENVV" ] && export $ENVV
  export HIP_VISIBLE_DEVICES=$DEVS GGML_DFLASH_PROF=1
  MI50_LOG="$LOG" MI50_TIMEOUT=5400 /root/mi50-run.sh "$BIN" "$M" "$PORT" \
    -md "$DR" --spec-type draft-dflash --spec-draft-n-max "$N" "${DFLAGS[@]}" \
    -ngl 99 -fa on -sm tensor --fit off -c 8192 -ub 1024 -t 16 -np 1 -lm dio \
    > "$BASE/$TAG.run.out" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "LEG-LOADFAIL rc=$rc"; tail -20 "$BASE/$TAG.run.out"; flock -u 9; return $rc; fi
  SPID=$(tail -1 "$BASE/$TAG.run.out")
  witness "$SPID" "$BASE/$TAG.proc.witness"
  grep -E "REPACK_Q8_0 gate|STOCK_ACC|RP_Q8_NC_LDS|RP_Q4_NC_LDS|repacking q8_0|repacking q4_0|MIN_TOKENS|floor ACTIVE|W4A4|draft model|dflash|devd|draft.*device|Meta" "$LOG" | head -20 > "$BASE/$TAG.gate.witness"
  measure "$TAG" "$LOG" "$N"; rc=$?
  unset GGML_DFLASH_PROF HIP_VISIBLE_DEVICES $ARMS
  teardown "$SPID"
  [ $rc = 0 ] && touch "$BASE/$TAG.done"
  echo "$TAG DONE rc=$rc $(date -Is)"
}

# short probe leg: load + one short tg + acceptance (feasibility witness, no bench)
probe() { # $1 arm $2 head $3 depth $4 port
  local arm=$1 HEAD=$2 N=$3 PORT=$4 TAG DR LOG rc ENVV M DFLAGS
  TAG="prb-$arm-$HEAD-d$N"; LOG=$BASE/"$TAG".server.log
  [ -e "$BASE/$TAG.done" ] && { echo "$TAG SKIP-EXISTS"; return 0; }
  ENVV=$(arm_env "$arm"); M=$(arm_model "$arm"); DR=$(arm_drafter "$arm")
  DFLAGS=(); [ "$HEAD" = p ] && DFLAGS=(-devd ROCm0)
  gates || return $?
  echo "$TAG start $(date -Is) DEVS=$DEVS" | tee "$BASE/$TAG.log"
  source /root/production-env.sh
  export HSA_KERNARG_POOL_SIZE=67108864
  unset $ARMS; [ -n "$ENVV" ] && export $ENVV
  export HIP_VISIBLE_DEVICES=$DEVS GGML_DFLASH_PROF=1
  MI50_LOG="$LOG" MI50_TIMEOUT=5400 /root/mi50-run.sh "$BIN" "$M" "$PORT" \
    -md "$DR" --spec-type draft-dflash --spec-draft-n-max "$N" "${DFLAGS[@]}" \
    -ngl 99 -fa on -sm tensor --fit off -c 8192 -ub 1024 -t 16 -np 1 -lm dio \
    > "$BASE/$TAG.run.out" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "PROBE-LOADFAIL rc=$rc" | tee -a "$BASE/$TAG.log"; tail -30 "$BASE/$TAG.run.out" | tee -a "$BASE/$TAG.log"; flock -u 9; return $rc; fi
  SPID=$(tail -1 "$BASE/$TAG.run.out")
  tr '\0' ' ' < /proc/$SPID/cmdline >> "$BASE/$TAG.log"; echo >> "$BASE/$TAG.log"
  curl -s -m 300 "http://127.0.0.1:$PORT/completion" -d \
    "{\"prompt\":\"$GPROMPT\",\"n_predict\":64,\"temperature\":0,\"seed\":42,\"ignore_eos\":true}" \
    > "$BASE/probe-$TAG.json"
  grep -E 'draft acceptance = ' "$LOG" | tail -2 | tee -a "$BASE/$TAG.log"
  grep -E "Abort|abort|GGML_ASSERT|cannot run the operation" "$LOG" | head -5 | tee -a "$BASE/$TAG.log"
  teardown "$SPID"
  touch "$BASE/$TAG.done"
  echo "$TAG DONE rc=$rc $(date -Is)" | tee -a "$BASE/$TAG.log"
}

case "${1:-}" in
  leg) shift; leg "$@" ;;
  probe) shift; probe "$@" ;;
  *) echo "usage: taskR-drive.sh probe <arm> <b|p> <depth> <port> | leg <arm> <b|p> <depth> <rep> <port>"; exit 2 ;;
esac
