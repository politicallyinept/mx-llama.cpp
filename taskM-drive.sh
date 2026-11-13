#!/bin/bash
# taskM driver: 35B Q8_0 MoE regression + q4_0-KV extension on merged build
# /root/build-jmerge (tip 15196ec20f, all J-wave gates default OFF).
# Regression arms replicate taskI2 (plain) / taskH3 (DFlash2 rep+pin) configs
# byte-for-byte except BIN. Extension arms = taskH3 pin config + q4_0 KV.
# Doctrine: flock /root/gpu.lock -w 900 per leg (retry loop on gate fail),
# TERM-only teardown, pgrep -x llama-server check, env witnessed per leg,
# refuse-if-exists + DONE markers (resumable).
set -u -o pipefail
BASE=/root/taskM-moe35
T=/root/build-jmerge/bin/llama-server
BIN=/root/build-jmerge/bin
M=/root/bench-quants/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf
D=/root/bench-quants/dflash/Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf
D2=/root/taskH3/draft/DFlash2-Q8_0-ownhead.gguf
GPROMPT="The engineer opened the notebook and began to write about the machine."
WIKI=/root/mx-int-rebase/scripts/wikitext-2-raw/wiki.test.raw
BASELOGITS=/root/taskI1/i1-base-f16.logits
PFLAGS="-c 2048 -b 2048 --chunks 16 -ngl 99 -fa on -sm tensor -fit off -lm dio"
mkdir -p "$BASE"

gates() { # $1 port ; sets DEVS, holds fd 9
  exec 9>/root/gpu.lock
  flock -w 900 9 || { echo "LOCKFAIL"; return 1; }
  awk '/MemAvailable/{exit !($2>=12582912)}' /proc/meminfo || { echo "GATE-RAM-FAIL"; flock -u 9; return 99; }
  local i
  for i in $(seq 1 20); do
    DEVS=$(python3 -c '
import subprocess,re
o=subprocess.run(["rocm-smi","--showmeminfo","vram"],capture_output=True,text=True).stdout
free=[]
for l in o.splitlines():
    m=re.match(r"GPU\[(\d+)\].*Used Memory \(B\):\s*(\d+)",l)
    if m and int(m.group(2))<2*1024**3: free.append(m.group(1))
if "1" in free and "2" in free: print("1,2")
else: print(",".join(free[:2]) if len(free)>=2 else "")')
    [ -n "$DEVS" ] && break
    echo "GPU-BUSY retry $i $(date -Is)"
    sleep 30
  done
  [ -n "$DEVS" ] || { echo "GATE-GPU-FAIL"; flock -u 9; return 96; }
  ss -ltn | grep -qE ":$1 " && { echo "GATE-PORT-FAIL $1"; flock -u 9; return 95; }
  QUIET=$(rocm-smi --showmeminfo vram | grep "Used Memory" | awk -F': ' '{s+=$2} END {print s}')
  echo "QUIET-CHECK total-used-B=$QUIET foreign-servers=$(pgrep -c -x llama-server || true)"
}

teardown() { # $1 pid
  kill -TERM "$1" 2>/dev/null
  local i
  for i in $(seq 1 36); do kill -0 "$1" 2>/dev/null || break; sleep 5; done
  if kill -0 "$1" 2>/dev/null; then echo "TEARDOWN-STUCK pid=$1 - STOP" >&2; return 90; fi
  local STRAY; STRAY=$(pgrep -x llama-server || true)
  [ -n "$STRAY" ] && echo "NOTE: foreign llama-server still running: $STRAY"
  sleep 5
  flock -u 9
}

bundle() { # $1 h3|prod ; called inside serve/kld subshell context
  case "$1" in
    h3)   source /root/w4a4-env.sh
          export GGML_CUDA_ALLREDUCE=internal GGML_CUDA_REPACK_KQUANT=0 GGML_GDN_COLTHREAD=1
          export HSA_KERNARG_POOL_SIZE=67108864 GPU_MAX_HW_QUEUES=8 ;;
    prod) source /root/production-env.sh
          export HSA_KERNARG_POOL_SIZE=67108864 ;;
  esac
}

serve() { # $1 label $2 port $3 bundle $4 extra-env-string $5.. draft/KV flags ; sets SRV LOG
  local LABEL=$1 PORT=$2 B=$3 XENV=$4; shift 4
  LOG=$BASE/$LABEL
  bundle "$B"
  local XARR=(); [ -n "$XENV" ] && XARR=($XENV)
  setsid nohup env HIP_VISIBLE_DEVICES=$DEVS "${XARR[@]}" $T -m "$M" "$@" \
    --host 0.0.0.0 --port "$PORT" -ngl 99 -fa on -sm tensor --fit off \
    -c 16384 -ub 1024 -t 16 -np 1 -lm dio > "$LOG.server.log" 2>&1 < /dev/null &
  SRV=$!
  echo "SRV_PID=$SRV" >> "$LOG.witness"
  local i
  for i in $(seq 1 120); do
    kill -0 "$SRV" 2>/dev/null || { echo "SERVERDIED" | tee -a "$LOG.log"; tail -40 "$LOG.server.log" >> "$LOG.log"; return 3; }
    curl -s -m 5 "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && return 0
    sleep 10
  done
  echo "HEALTHFAIL" | tee -a "$LOG.log"; return 4
}

serve_plain() { # $1 label $2 port ; taskI2 plain config (-c 65536, prod bundle)
  local LABEL=$1 PORT=$2
  LOG=$BASE/$LABEL
  bundle prod
  setsid nohup env HIP_VISIBLE_DEVICES=$DEVS $T -m "$M" \
    --host 0.0.0.0 --port "$PORT" -ngl 99 -fa on -sm tensor --fit off \
    -c 65536 -ub 1024 -t 16 -np 1 -lm dio > "$LOG.server.log" 2>&1 < /dev/null &
  SRV=$!
  echo "SRV_PID=$SRV" >> "$LOG.witness"
  local i
  for i in $(seq 1 120); do
    kill -0 "$SRV" 2>/dev/null || { echo "SERVERDIED" | tee -a "$LOG.log"; tail -40 "$LOG.server.log" >> "$LOG.log"; return 3; }
    curl -s -m 5 "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && return 0
    sleep 10
  done
  echo "HEALTHFAIL" | tee -a "$LOG.log"; return 4
}

witness() { # $1 label ; arm identity + gate-off proof + runtime
  tr '\0' ' ' < /proc/$SRV/cmdline >> "$LOG.witness"; echo >> "$LOG.witness"
  tr '\0' '\n' < /proc/$SRV/environ | grep -E '^(LD_LIBRARY_PATH|GGML_|HSA_|HIP_|LLAMA_)' | sort > "$LOG.env-witness"
  grep -oE '/[^ ]*lib(amdhip64|rccl|rocblas)[^ ]*' /proc/$SRV/maps | sort -u >> "$LOG.witness"
  grep -q "/opt/rocm-7.2.3/lib/libamdhip64" "$LOG.witness" && { echo "W4-FAIL apt-runtime" | tee -a "$LOG.log"; return 5; }
  grep -E "loading draft model|draft-dflash|n_max=|block_size=|dev=|split mode|KV|cache type" "$LOG.server.log" | head -20 >> "$LOG.witness"
  grep -E "\[FA-DOT8\]|RP_Q8_NC_LDS|KROT" "$LOG.server.log" | head -5 >> "$LOG.witness" || true
  return 0
}

greedy() { # $1 label $2 port : 64-token temp0 seed42 probe
  curl -s -m 300 "http://127.0.0.1:$2/completion" -d \
    "{\"prompt\":\"$GPROMPT\",\"n_predict\":64,\"temperature\":0,\"seed\":42,\"ignore_eos\":true}" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("content",""))' > "$BASE/$1.greedy.txt"
  sha256sum "$BASE/$1.greedy.txt" | cut -c1-16 >> "$BASE/$1.log"
}

leg() { # $1 label $2 kind(plain|dflash) $3 port $4 depth $5 bundle $6 extra-env $7 draft-model $8.. kv/draft flags
  local LABEL=$1 KIND=$2 PORT=$3 N=$4 B=$5 XENV=$6 DR=$7; shift 7
  local LOGF=$BASE/$LABEL SRV RC
  grep -q "$LABEL DONE" "$LOGF.log" 2>/dev/null && { echo "$LABEL SKIP-EXISTS"; return 0; }
  local attempt rc
  for attempt in 1 2 3 4 5 6; do
    rc=0
    gates "$PORT" || rc=$?
    [ $rc -ne 0 ] && { echo "$LABEL gate-fail rc=$rc attempt=$attempt $(date -Is)" | tee -a "$LOGF.log"; sleep 60; continue; }
    echo "$LABEL start $(date -Is) DEVS=$DEVS kind=$KIND depth=$N bundle=$B xenv='$XENV'" | tee "$LOGF.log"
    sha256sum "$M" > "$LOGF.witness"
    if [ "$KIND" = plain ]; then
      serve_plain "$LABEL" "$PORT"; RC=$?
    else
      sha256sum "$DR" >> "$LOGF.witness"
      serve "$LABEL" "$PORT" "$B" "$XENV" -md "$DR" --spec-type draft-dflash --spec-draft-n-max "$N" "$@"; RC=$?
    fi
    if [ $RC -ne 0 ]; then teardown $SRV; echo "$LABEL SERVE-FAIL rc=$RC attempt=$attempt" | tee -a "$LOGF.log"; sleep 60; continue; fi
    witness "$LABEL" || { teardown $SRV; echo "$LABEL WITNESS-FAIL" | tee -a "$LOGF.log"; return 5; }
    python3 /root/p2-prefill/probe_pp.py "$PORT" 6 >/dev/null 2>&1
    local i; for i in 1 2 3; do curl -s -m 600 "http://127.0.0.1:$PORT/completion" -d \
      '{"prompt":"Warmup pass for the decode shape.","n_predict":256,"temperature":0,"cache_prompt":false,"ignore_eos":true}' >/dev/null; done
    timeout 5400 python3 /root/p2-prefill/bench.py "$PORT" "$LABEL" "$LOGF.reps.jsonl" "$N" 2>&1 | tail -2 | tee -a "$LOGF.log"
    greedy "$LABEL" "$PORT"
    grep -E 'draft acceptance = ' "$LOG.server.log" | tail -3 | tee -a "$LOGF.log"
    teardown $SRV || return $?
    echo "$LABEL DONE $(date -Is)" | tee -a "$LOGF.log"
    return 0
  done
  echo "$LABEL GAVE-UP after attempts" | tee -a "$LOGF.log"; return 1
}

kld() { # $1 arm(q40) $2 port
  local ARM=$1 PORT=$2
  local out=$BASE/kld-$ARM.log
  [ -e "$out.done" ] && { echo "kld-$ARM SKIP-EXISTS"; return 0; }
  local attempt rc
  for attempt in 1 2 3 4 5 6; do
    rc=0; gates "$PORT" || rc=$?
    [ $rc -ne 0 ] && { echo "kld-$ARM gate-fail rc=$rc attempt=$attempt"; sleep 60; continue; }
    echo "kld-$ARM start $(date -Is) DEVS=$DEVS" | tee -a "$out"
    ( bundle prod
      export GGML_TP_AR_MTP_SAFE=1 HIP_VISIBLE_DEVICES=$DEVS
      timeout 3600 "$BIN/llama-perplexity" -m "$M" -f "$WIKI" \
        --kl-divergence-base "$BASELOGITS" --kl-divergence \
        --cache-type-k q4_0 --cache-type-v q4_0 $PFLAGS >> "$out" 2>&1 < /dev/null 9>&-
      rc=$?; echo "rc=$rc" >> "$out"; exit $rc )
    local krc=$?
    flock -u 9
    grep -E "Mean    KLD|Maximum KLD|Mean PPL\(Q\)|Mean    PPL|rc=" "$out" | tail -6
    if [ "$krc" = 0 ]; then touch "$out.done"; echo "kld-$ARM DONE $(date -Is)"; return 0; fi
    echo "kld-$ARM FAIL rc=$krc attempt=$attempt"; sleep 60
  done
  echo "kld-$ARM GAVE-UP"; return 1
}

# ---- regression (all gates default OFF) ----
leg reg-plain-r1    plain  8761 1 prod "" ""
leg reg-d4-rep-r1   dflash 8762 4 h3   "" "$D"
leg reg-d4-pin-r1   dflash 8763 4 h3   "" "$D2" -devd ROCm0
# ---- extension: q4_0 KV on pinned DFlash2 ----
leg ext-d4-pin-q40  dflash 8764 4 h3   "" "$D2" -devd ROCm0 --cache-type-k q4_0 --cache-type-v q4_0
leg ext-d7-pin-q40  dflash 8765 7 h3   "" "$D2" -devd ROCm0 --cache-type-k q4_0 --cache-type-v q4_0
# ---- mandatory numeric gate: KLD re-score vs taskI1 f16 base logits ----
kld q40 8766
# ---- second reps (noise band) ----
leg reg-d4-pin-r2   dflash 8767 4 h3   "" "$D2" -devd ROCm0
leg reg-plain-r2    plain  8768 1 prod "" ""
# ---- cross-check: replicated drafter + q4_0 KV vs taskI1 76.50/0.3508 ----
leg ext-d4-rep-q40  dflash 8769 4 h3   "" "$D" --cache-type-k q4_0 --cache-type-v q4_0
# ---- optional: LDS arm ----
leg lds-d4-pin-r1   dflash 8770 4 h3   "GGML_CUDA_RP_Q8_NC_LDS=8" "$D2" -devd ROCm0
echo "DRIVER-ALL-DONE $(date -Is)"
