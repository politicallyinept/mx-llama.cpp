#!/bin/bash
# taskL driver: Q4_0 drafter A/B vs Q8_0 drafter on 27B Q8_0 TP2, LDS-alone config
# (production env + REPACK_Q8_0 pair + RP_Q8_NC_LDS=8), build-jmerge @ INT4-noquant tip.
# Arms (all dflash, depths 4 and 7, 2 interleaved reps):
#   q8d  = DFlash2-Q8_0 drafter (control)
#   q4d  = DFlash2-Q4_0 drafter (the requant)
#   q4w0 = q4d + GGML_CUDA_W4A4_MIN_TOKENS=0 (F5 floor-0; with a Q8_0 main model the
#          only q4_0 tensors in the process are the drafter's, so the forced-W4A4 u4
#          path is scoped to the drafter forward by dtype composition)
# Doctrine: flock /root/gpu.lock -w 900, TERM-only teardown, pgrep -x llama-server empty
# after each leg, production env witnessed via mi50-run.sh, refuse-if-exists.
set -u -o pipefail
BASE=/root/taskL
BIN=/root/build-jmerge
M=/root/bench-quants/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
D8=/mnt/old/home/brad/models/Qwen3.8-27B-DFlash2-Q8_0.gguf
D4=/mnt/old/home/brad/models/Qwen3.8-27B-DFlash2-Q4_0.gguf
GPROMPT="The engineer opened the notebook and began to write about the machine."
mkdir -p "$BASE"

arm_drafter() { case "$1" in q8d) echo "$D8" ;; q4d|q4w0) echo "$D4" ;; esac; }

gates() { # sets DEVS, holds fd 9
  local i
  for i in $(seq 1 40); do
    awk '{exit !($1<16)}' /proc/loadavg && break
    [ "$i" = 40 ] && { echo "GATE-LOAD-FAIL"; return 97; }
    sleep 30
  done
  exec 9>/root/gpu.lock
  flock -w 900 9 || { echo "LOCKFAIL"; return 1; }
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
    '{"prompt":"Warmup pass for the decode shape.","n_predict":256,"temperature":0,"cache_prompt":false,"ignore_eos":true}' >/dev/null; done
  timeout 5400 python3 /root/p2-prefill/bench.py "$PORT" "$TAG" "$BASE/$TAG.reps.jsonl" "$N" > "$BASE/$TAG.bench.log" 2>&1
  local rc=$?
  curl -s -m 300 "http://127.0.0.1:$PORT/completion" -d \
    "{\"prompt\":\"$GPROMPT\",\"n_predict\":64,\"temperature\":0,\"seed\":42,\"ignore_eos\":true}" \
    > "$BASE/probe-$TAG.json"
  grep -E 'draft acceptance = ' "$LOG" | tail -3 >> "$BASE/$TAG.bench.log"
  grep -c "DFLASH-PROF" "$LOG" > "$BASE/$TAG.prof-count"
  return $rc
}

leg() { # $1 arm $2 depth $3 rep $4 port
  local arm=$1 N=$2 REP=$3 PORT=$4 TAG DR LOG SPID rc
  TAG="$arm-d$N-r$REP"; LOG=$BASE/"$TAG".server.log
  [ -e "$BASE/$TAG.done" ] && { echo "$TAG SKIP-EXISTS"; return 0; }
  DR=$(arm_drafter "$arm")
  gates || return $?
  echo "$TAG start $(date -Is) DEVS=$DEVS arm=$arm depth=$N"
  sha256sum "$M" "$DR" > "$BASE/$TAG.witness"
  local BINX=$BIN
  source /root/production-env.sh
  export HSA_KERNARG_POOL_SIZE=67108864
  export GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0 GGML_CUDA_RP_Q8_NC_LDS=8
  unset GGML_CUDA_W4A4_MIN_TOKENS
  [ "$arm" = q4w0 ] && export GGML_CUDA_W4A4_MIN_TOKENS=0
  export HIP_VISIBLE_DEVICES=$DEVS GGML_DFLASH_PROF=1
  MI50_LOG="$LOG" MI50_TIMEOUT=5400 /root/mi50-run.sh "$BINX" "$M" "$PORT" \
    -md "$DR" --spec-type draft-dflash --spec-draft-n-max "$N" \
    -ngl 99 -fa on -sm tensor --fit off -c 8192 -ub 1024 -t 16 -np 1 -lm dio \
    > "$BASE/$TAG.run.out" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "LEG-LOADFAIL rc=$rc"; tail -20 "$BASE/$TAG.run.out"; flock -u 9; return $rc; fi
  SPID=$(tail -1 "$BASE/$TAG.run.out")
  witness "$SPID" "$BASE/$TAG.proc.witness"
  grep -E "REPACK_Q8_0 gate|STOCK_ACC|RP_Q8_NC_LDS|repacking q8_0|repacking q4_0|MIN_TOKENS|W4A4|draft model|dflash" "$LOG" | head -12 > "$BASE/$TAG.gate.witness"
  measure "$TAG" "$LOG" "$N"; rc=$?
  unset GGML_DFLASH_PROF HIP_VISIBLE_DEVICES GGML_CUDA_W4A4_MIN_TOKENS
  teardown "$SPID"
  [ $rc = 0 ] && touch "$BASE/$TAG.done"
  echo "$TAG DONE rc=$rc $(date -Is)"
}

case "${1:-}" in
  leg) shift; leg "$@" ;;
  *) echo "usage: taskL-drive.sh leg <q8d|q4d|q4w0> <4|7> <rep> <port>"; exit 2 ;;
esac
