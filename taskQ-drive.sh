#!/bin/bash
# taskQ driver: Q4_0 LDS-staged narrow matvec (GGML_CUDA_RP_Q4_NC_LDS) A/B on the
# 27B Q4_0 stack. Stack: Qwen3.8-27B Q4_0 TP2 + DFlash2-Q4_0 drafter, build-q-q4lds.
# Arms (env-only deltas on top of the production bundle, enforced by mi50-run.sh):
#   f0lds    = floor-0 + REPACK_Q8_0 pair + RP_Q8_NC_LDS=8  [taskN control, Q4 gate OFF:
#              reproduces 55.27 AND proves gate-off byte-identity on the new binary]
#   f0q4     = f0lds + GGML_CUDA_RP_Q4_NC_LDS=8             [U4ACT LDS arm at verify width]
#   f128lds  = REPACK_Q8_0 pair + RP_Q8_NC_LDS=8            [floor-ON control, no floor-0]
#   f128q4   = f128lds + GGML_CUDA_RP_Q4_NC_LDS=8           [affine LDS arm, no floor-0]
# Doctrine: flock /root/gpu.lock -w 900, TERM-only teardown, pgrep -x llama-server empty
# after each leg, production env witnessed via mi50-run.sh, refuse-if-exists.
set -u -o pipefail
BASE=/root/taskQ
BIN=${BIN:-/root/build-q-q4lds}
M=/root/bench-quants/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf
D=/mnt/old/home/brad/models/Qwen3.8-27B-DFlash2-Q4_0.gguf
GPROMPT="The engineer opened the notebook and began to write about the machine."
WIKI=/root/mx-int-rebase/scripts/wikitext-2-raw/wiki.test.raw
WSUB=$BASE/wiki-sub.txt
mkdir -p "$BASE"
[ -e "$WSUB" ] || head -c 65536 "$WIKI" > "$WSUB"

ARMS="GGML_CUDA_W4A4_MIN_TOKENS GGML_CUDA_REPACK_Q8_0 GGML_CUDA_REPACK_Q8_0_STOCK_ACC GGML_CUDA_RP_Q8_NC_LDS GGML_CUDA_RP_Q4_NC_LDS"

arm_env() {
  case "$1" in
    f0lds)   echo "GGML_CUDA_W4A4_MIN_TOKENS=0 GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0 GGML_CUDA_RP_Q8_NC_LDS=8" ;;
    f0q4)    echo "GGML_CUDA_W4A4_MIN_TOKENS=0 GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0 GGML_CUDA_RP_Q8_NC_LDS=8 GGML_CUDA_RP_Q4_NC_LDS=8" ;;
    f128lds) echo "GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0 GGML_CUDA_RP_Q8_NC_LDS=8" ;;
    f128q4)  echo "GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0 GGML_CUDA_RP_Q8_NC_LDS=8 GGML_CUDA_RP_Q4_NC_LDS=8" ;;
    f128q4w4) echo "GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0 GGML_CUDA_RP_Q8_NC_LDS=8 GGML_CUDA_RP_Q4_NC_LDS=4" ;;
    f0q4w4)  echo "GGML_CUDA_W4A4_MIN_TOKENS=0 GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0 GGML_CUDA_RP_Q8_NC_LDS=8 GGML_CUDA_RP_Q4_NC_LDS=4" ;;
    *) return 1 ;;
  esac
}

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
  local arm=$1 N=$2 REP=$3 PORT=$4 TAG DR LOG SPID rc ENVV
  TAG="$arm-d$N-r$REP"; LOG=$BASE/"$TAG".server.log
  [ -e "$BASE/$TAG.done" ] && { echo "$TAG SKIP-EXISTS"; return 0; }
  ENVV=$(arm_env "$arm") || { echo "BAD-ARM $arm"; return 2; }
  gates || return $?
  echo "$TAG start $(date -Is) DEVS=$DEVS arm=$arm depth=$N env[$ENVV]"
  sha256sum "$M" "$D" > "$BASE/$TAG.witness"
  source /root/production-env.sh
  export HSA_KERNARG_POOL_SIZE=67108864
  unset $ARMS
  [ -n "$ENVV" ] && export $ENVV
  export HIP_VISIBLE_DEVICES=$DEVS GGML_DFLASH_PROF=1
  MI50_LOG="$LOG" MI50_TIMEOUT=5400 /root/mi50-run.sh "$BIN" "$M" "$PORT" \
    -md "$D" --spec-type draft-dflash --spec-draft-n-max "$N" \
    -ngl 99 -fa on -sm tensor --fit off -c 8192 -ub 1024 -t 16 -np 1 -lm dio \
    > "$BASE/$TAG.run.out" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "LEG-LOADFAIL rc=$rc"; tail -20 "$BASE/$TAG.run.out"; flock -u 9; return $rc; fi
  SPID=$(tail -1 "$BASE/$TAG.run.out")
  witness "$SPID" "$BASE/$TAG.proc.witness"
  grep -E "REPACK_Q8_0 gate|STOCK_ACC|RP_Q8_NC_LDS|RP_Q4_NC_LDS|repacking q8_0|repacking q4_0|MIN_TOKENS|floor ACTIVE|W4A4|draft model|dflash" "$LOG" | head -16 > "$BASE/$TAG.gate.witness"
  measure "$TAG" "$LOG" "$N"; rc=$?
  unset GGML_DFLASH_PROF HIP_VISIBLE_DEVICES $ARMS
  teardown "$SPID"
  [ $rc = 0 ] && touch "$BASE/$TAG.done"
  echo "$TAG DONE rc=$rc $(date -Is)"
}

kld() { # $1 arm ; taskJ2 narrow-window protocol: -b 8 pp batches exercise the nc width-8
  # window. Pairing: f0q4 scored vs f0lds logits, f128q4 scored vs f128lds logits -
  # each pair isolates the RP_Q4_NC_LDS gate delta (expect ~0: bit-identical kernels).
  local arm=$1; local out=$BASE/kld-$arm.log
  [ -e "$out.done" ] && { echo "kld-$arm SKIP-EXISTS"; return 0; }
  local ENVV PARENT; ENVV=$(arm_env "$arm") || { echo "BAD-ARM"; return 2; }
  case "$arm" in
    f0lds)   PARENT="" ;;
    f128lds) PARENT="" ;;
    f0q4|f0q4w4)    PARENT=f0lds ;;
    f128q4|f128q4w4) PARENT=f128lds ;;
    *) return 2 ;;
  esac
  gates || return $?
  echo "kld-$arm start $(date -Is)"
  ( source /root/production-env.sh
    export GGML_TP_AR_MTP_SAFE=1 HIP_VISIBLE_DEVICES=$DEVS HSA_KERNARG_POOL_SIZE=67108864
    unset $ARMS
    [ -n "$ENVV" ] && export $ENVV
    if [ -z "$PARENT" ]; then
      timeout 3600 "$BIN/bin/llama-perplexity" -m "$M" -f "$WSUB" -b 8 -c 512 -ngl 99 -fa on -sm tensor -fit off -lm dio \
        --save-all-logits "$BASE/q-$arm.logits" > "$out" 2>&1 < /dev/null 9>&-
    else
      timeout 3600 "$BIN/bin/llama-perplexity" -m "$M" -f "$WSUB" -b 8 -c 512 -ngl 99 -fa on -sm tensor -fit off -lm dio \
        --kl-divergence-base "$BASE/q-$PARENT.logits" --kl-divergence > "$out" 2>&1 < /dev/null 9>&-
    fi
    rc=$?; echo "rc=$rc" >> "$out"; exit $rc )
  local rc=$?
  grep -E "Mean    KLD|Maximum KLD|Mean PPL\(Q\)|Mean    PPL|rc=" "$out" | tail -6
  flock -u 9
  [ "$rc" = 0 ] && touch "$out.done"
  echo "kld-$arm DONE rc=$rc $(date -Is)"
}

case "${1:-}" in
  leg) shift; leg "$@" ;;
  kld) shift; kld "$@" ;;
  *) echo "usage: taskQ-drive.sh leg <f0lds|f0q4|f128lds|f128q4> <4|7> <rep> <port> | kld <f0lds|f0q4|f128lds|f128q4>"; exit 2 ;;
esac
