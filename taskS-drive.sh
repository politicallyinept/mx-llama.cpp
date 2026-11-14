#!/bin/bash
# taskS driver: FA_DOT8 closure - (1) width-8 KLD gate on the 27B Q4_0 stack (taskJ2 -b 8
# narrow-window protocol, kernel ACTIVE at its real serving widths), (2) short d7 acceptance
# legs, (3) 32k/64k long-context e2e: plain decode + DFlash2 d4, arms f16-KV / q4_0-KV /
# q4_0-KV+FA_DOT8. Doctrine: flock /root/gpu.lock -w 900, TERM-only teardown, pgrep clean,
# production-env + TheRock runtime witnessed, refuse-if-exists.
set -u -o pipefail
BASE=/root/taskS
BIN=/root/build-qmerge
M=/root/bench-quants/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf
D=/mnt/old/home/brad/models/Qwen3.8-27B-DFlash2-Q8_0.gguf
WIKI=/root/mx-int-rebase/scripts/wikitext-2-raw/wiki.test.raw
WSUB=/root/taskJ2/wiki-sub.txt
DOC32=$BASE/doc32.txt
DOC64=$BASE/doc64.txt
GPROMPT="The engineer opened the notebook and began to write about the machine."
mkdir -p "$BASE"
[ -e "$WSUB" ] || head -c 65536 "$WIKI" > "$WSUB"
[ -e "$DOC32" ] || head -c 137626 "$WIKI" > "$DOC32"   # ~32k tokens at ~4.2 ch/tok
[ -e "$DOC64" ] || head -c 275252 "$WIKI" > "$DOC64"   # ~64k tokens

arm_kv() { case "$1" in
  f16) echo "" ;;
  q40|q40d1|q40d2|q40krotd1) echo "--cache-type-k q4_0 --cache-type-v q4_0" ;;
esac; }
arm_env() { case "$1" in
  f16|q40) echo "" ;;
  q40d1) echo "GGML_CUDA_FA_DOT8=1" ;;
  q40d2) echo "GGML_CUDA_FA_DOT8=2" ;;
  q40krotd1) echo "GGML_CUDA_FA_DOT8=1 LLAMA_KV_KROT=1" ;;
esac; }

gates() { # sets DEVS, holds fd 9
  local i
  for i in $(seq 1 40); do
    awk '{exit !($1<16)}' /proc/loadavg && break
    [ "$i" = 40 ] && { echo "GATE-LOAD-FAIL"; return 97; }
    sleep 30
  done
  exec 9>/root/gpu.lock
  flock -w 1800 9 || { echo "LOCKFAIL"; return 1; }
  awk '/MemAvailable/{exit !($2>=12582912)}' /proc/meminfo || { echo "GATE-RAM-FAIL"; flock -u 9; return 99; }
  for i in $(seq 1 20); do
    DEVS=$(python3 -c '
import subprocess
o=subprocess.run(["rocm-smi","--showmeminfo","vram"],capture_output=True,text=True).stdout
free=[]
for l in o.splitlines():
    m=__import__("re").match(r"GPU\[(\d+)\].*Used Memory \(B\):\s*(\d+)",l)
    if m and int(m.group(2))<2*1024**3: free.append(m.group(1))
print(",".join(free[:2]) if len(free)>=2 else "")')
    [ -n "$DEVS" ] && break
    echo "GPU-BUSY retry $i $(date -Is)"
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
  [ -z "$(pgrep -x llama-server)" ] || echo "WARN llama-server present after teardown"
  flock -u 9
}

witness() { # $1 pid $2 file
  tr '\0' ' ' < /proc/$1/cmdline > "$2"; echo >> "$2"
  tr '\0' '\n' < /proc/$1/environ | grep -E '^(GGML_|HSA_|HIP_|LD_LIBRARY|LLAMA_)' | sort >> "$2"
  grep -oE '/[^ ]*lib(amdhip64|rccl|rocblas)[^ ]*' /proc/$1/maps | sort -u >> "$2"
  grep -q "/usr/lib.*libamdhip64" "$2" && echo "RUNTIME-FAIL apt-libamdhip64" >> "$2"
}

# ---------------- phase 1: width-8 KLD gate (taskJ2 -b 8 protocol, 27B Q4_0) ----------------
kld8() { # $1 name $2 envv $3 kvflag $4 base-logits(""=none) $5 save(0|1)
  local NAME=$1 ENVV=$2 KV=$3 BASEL=$4 SAVE=$5
  local out=$BASE/kld8-$NAME.log
  [ -e "$out.done" ] && { echo "kld8-$NAME SKIP-EXISTS"; return 0; }
  gates || return $?
  echo "kld8-$NAME start $(date -Is) DEVS=$DEVS ENV='$ENVV' KV='$KV' BASE='$BASEL'"
  ( source /root/production-env.sh
    export GGML_TP_AR_MTP_SAFE=1 HIP_VISIBLE_DEVICES=$DEVS HSA_KERNARG_POOL_SIZE=67108864
    [ -n "$ENVV" ] && export $ENVV
    local ARGS=( -m "$M" -f "$WSUB" -b 8 -c 512 -ngl 99 -fa on -sm tensor -fit off -lm dio )
    [ -n "$KV" ] && read -r -a KVA <<< "$KV" && ARGS+=( "${KVA[@]}" )
    [ -n "$BASEL" ] && ARGS+=( --kl-divergence-base "$BASEL" --kl-divergence )
    [ "$SAVE" = 1 ] && ARGS+=( --save-all-logits "$BASE/kld8-$NAME.logits" )
    timeout 3600 "$BIN/bin/llama-perplexity" "${ARGS[@]}" > "$out" 2>&1 < /dev/null 9>&-
    rc=$?; echo "rc=$rc" >> "$out"; exit $rc )
  local rc=$?
  grep -E "Mean    KLD|Maximum KLD|Mean PPL\(Q\)|Mean    PPL|rc=" "$out" | tail -6
  grep -c "FA-DOT8" "$out" > "$out.dot8-hits"
  # success guards: KLD run must have produced a KLD line; save run must have written logits
  # (this llama.cpp aliases --save-all-logits == --kl-divergence-base: never pass both)
  if [ "$rc" = 0 ] && [ -n "$BASEL" ] && ! grep -q "Mean    KLD" "$out"; then echo "KLD-GUARD-FAIL: no KLD in $out"; rc=1; fi
  if [ "$rc" = 0 ] && [ "$SAVE" = 1 ] && [ ! -s "$BASE/kld8-$NAME.logits" ]; then echo "SAVE-GUARD-FAIL: no logits"; rc=1; fi
  flock -u 9
  [ "$rc" = 0 ] && touch "$out.done"
  echo "kld8-$NAME DONE rc=$rc dot8-hits=$(cat "$out.dot8-hits") $(date -Is)"
}

kld8all() {
  kld8 base8   "" "" "" 1
  kld8 q40off  "" "--cache-type-k q4_0 --cache-type-v q4_0" "" 1
  kld8 q40kld  "" "--cache-type-k q4_0 --cache-type-v q4_0" "$BASE/kld8-base8.logits" 0
  kld8 d1      "GGML_CUDA_FA_DOT8=1" "--cache-type-k q4_0 --cache-type-v q4_0" "$BASE/kld8-base8.logits" 0
  kld8 d2      "GGML_CUDA_FA_DOT8=2" "--cache-type-k q4_0 --cache-type-v q4_0" "$BASE/kld8-base8.logits" 0
  kld8 d1vsoff "GGML_CUDA_FA_DOT8=1" "--cache-type-k q4_0 --cache-type-v q4_0" "$BASE/kld8-q40off.logits" 0
  kld8 d2vsoff "GGML_CUDA_FA_DOT8=2" "--cache-type-k q4_0 --cache-type-v q4_0" "$BASE/kld8-q40off.logits" 0
}

# ---------------- phase 2: short d7 acceptance legs (verify width 8) ----------------
measure7() { # $1 TAG $2 LOG $3 PORT : warmup + bench.py suite (draft 7) + greedy probe
  local TAG=$1 LOG=$2 PORT=$3
  python3 /root/p2-prefill/probe_pp.py "$PORT" 6 >/dev/null 2>&1
  local i
  for i in 1 2 3; do curl -s -m 600 "http://127.0.0.1:$PORT/completion" -d \
    '{"prompt":"Warmup pass for the decode shape.","n_predict":256,"temperature":0,"cache_prompt":false,"ignore_eos":true}' >/dev/null; done
  timeout 5400 python3 /root/p2-prefill/bench.py "$PORT" "$TAG" "$BASE/$TAG.reps.jsonl" 7 > "$BASE/$TAG.bench.log" 2>&1
  local rc=$?
  curl -s -m 300 "http://127.0.0.1:$PORT/completion" -d \
    "{\"prompt\":\"$GPROMPT\",\"n_predict\":64,\"temperature\":0,\"seed\":42,\"ignore_eos\":true}" \
    > "$BASE/probe-$TAG.json"
  grep -E 'draft acceptance = ' "$LOG" | tail -3 >> "$BASE/$TAG.bench.log"
  return $rc
}

shortleg() { # $1 arm $2 port
  local TAG=s-$1 PORT=$2
  [ -e "$BASE/$TAG.done" ] && { echo "$TAG SKIP-EXISTS"; return 0; }
  gates || return $?
  echo "short-$TAG start $(date -Is) DEVS=$DEVS arm=$1"
  sha256sum "$M" "$D" > "$BASE/$TAG.witness"
  source /root/production-env.sh
  export HSA_KERNARG_POOL_SIZE=67108864
  local ENVV; ENVV=$(arm_env "$1"); [ -n "$ENVV" ] && export $ENVV
  export HIP_VISIBLE_DEVICES=$DEVS GGML_DFLASH_PROF=1
  local KV; read -r -a KVA <<< "$(arm_kv "$1")"
  MI50_LOG="$BASE/$TAG.server.log" MI50_TIMEOUT=5400 /root/mi50-run.sh "$BIN" "$M" "$PORT" \
    -md "$D" --spec-type draft-dflash --spec-draft-n-max 7 "${KVA[@]}" \
    -ngl 99 -fa on -sm tensor --fit off -c 8192 -ub 1024 -t 16 -np 1 -lm dio \
    > "$BASE/$TAG.run.out" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "LOADFAIL rc=$rc"; tail -20 "$BASE/$TAG.run.out"; flock -u 9; return $rc; fi
  local SPID; SPID=$(tail -1 "$BASE/$TAG.run.out")
  witness "$SPID" "$BASE/$TAG.proc.witness"
  grep -m2 "FA-DOT8" "$BASE/$TAG.server.log" >> "$BASE/$TAG.witness" 2>/dev/null
  measure7 "$TAG" "$BASE/$TAG.server.log" "$PORT"; rc=$?
  unset GGML_DFLASH_PROF HIP_VISIBLE_DEVICES
  teardown "$SPID"
  [ $rc = 0 ] && touch "$BASE/$TAG.done"
  echo "short-$TAG DONE rc=$rc $(date -Is)"
}

shortall() {
  shortleg q40 8711
  shortleg q40d1 8712
  shortleg q40d2 8713
}

# ---------------- phase 3: plain long-context decode (llama-bench) ----------------
plainbench() { # $1 arm $2 p-length
  local TAG=plain-$1-p$2
  [ -e "$BASE/$TAG.done" ] && { echo "$TAG SKIP-EXISTS"; return 0; }
  gates || return $?
  echo "plain-$TAG start $(date -Is) DEVS=$DEVS"
  source /root/production-env.sh
  export HSA_KERNARG_POOL_SIZE=67108864 HIP_VISIBLE_DEVICES=$DEVS
  local ENVV; ENVV=$(arm_env "$1"); [ -n "$ENVV" ] && export $ENVV
  local KV; read -r -a KVA <<< "$(arm_kv "$1")"
  timeout 5400 "$BIN/bin/llama-bench" -m "$M" -ngl 99 -fa on -sm tensor -lm dio "${KVA[@]}" \
    -p "$2" -n 256 -r 1 -o jsonl > "$BASE/$TAG.jsonl" 2> "$BASE/$TAG.log" < /dev/null 9>&-
  local rc=$?
  echo "rc=$rc" >> "$BASE/$TAG.log"
  tail -3 "$BASE/$TAG.log"
  flock -u 9
  [ "$rc" = 0 ] && grep -q "tg256" "$BASE/$TAG.jsonl" && touch "$BASE/$TAG.done"
  echo "plain-$TAG DONE rc=$rc $(date -Is)"
}

plainall() {
  for a in f16 q40 q40d1; do plainbench "$a" 32768; done
  for a in f16 q40 q40d1; do plainbench "$a" 65536; done
}

# ---------------- phase 4: DFlash2 d4 long-context legs ----------------
lcd4() { # $1 arm $2 len(32k|64k) $3 port
  local TAG=lc-$1-$2 PORT=$3 CTX DOC
  case "$2" in 32k) CTX=33792; DOC=$DOC32 ;; 64k) CTX=66560; DOC=$DOC64 ;; esac
  [ -e "$BASE/$TAG.done" ] && { echo "$TAG SKIP-EXISTS"; return 0; }
  gates || return $?
  echo "lcd4-$TAG start $(date -Is) DEVS=$DEVS arm=$1 ctx=$CTX"
  sha256sum "$M" "$D" > "$BASE/$TAG.witness"
  source /root/production-env.sh
  export HSA_KERNARG_POOL_SIZE=67108864
  local ENVV; ENVV=$(arm_env "$1"); [ -n "$ENVV" ] && export $ENVV
  export HIP_VISIBLE_DEVICES=$DEVS GGML_DFLASH_PROF=1
  local KV; read -r -a KVA <<< "$(arm_kv "$1")"
  MI50_LOG="$BASE/$TAG.server.log" MI50_TIMEOUT=5400 /root/mi50-run.sh "$BIN" "$M" "$PORT" \
    -md "$D" --spec-type draft-dflash --spec-draft-n-max 4 "${KVA[@]}" \
    -ngl 99 -fa on -sm tensor --fit off -c "$CTX" -ub 1024 -t 16 -np 1 -lm dio \
    > "$BASE/$TAG.run.out" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "LOADFAIL rc=$rc"; tail -20 "$BASE/$TAG.run.out"; flock -u 9; return $rc; fi
  local SPID; SPID=$(tail -1 "$BASE/$TAG.run.out")
  witness "$SPID" "$BASE/$TAG.proc.witness"
  grep -m2 "FA-DOT8" "$BASE/$TAG.server.log" >> "$BASE/$TAG.witness" 2>/dev/null
  timeout 3600 python3 /root/int4-noquant-wt/taskS-lc-probe.py "$PORT" "$DOC" "$BASE/$TAG.q.jsonl"; rc=$?
  grep "DFLASH-PROF" "$BASE/$TAG.server.log" > "$BASE/$TAG.prof" 2>/dev/null
  wc -l < "$BASE/$TAG.prof" > "$BASE/$TAG.prof-count" 2>/dev/null
  unset GGML_DFLASH_PROF HIP_VISIBLE_DEVICES
  teardown "$SPID"
  [ $rc = 0 ] && touch "$BASE/$TAG.done"
  echo "lcd4-$TAG DONE rc=$rc $(date -Is)"
}

lcd4all() {
  for a in f16 q40 q40d1; do lcd4 "$a" 32k 8721; done
  for a in f16 q40 q40d1; do lcd4 "$a" 64k 8724; done
}

krotall() { # KROT arms if acceptance moved
  shortleg q40krotd1 8714
  lcd4 q40krotd1 64k 8727
}

case "${1:-}" in
  kld8) kld8all ;;
  short) shortall ;;
  plain) plainall ;;
  lcd4) lcd4all ;;
  krot) krotall ;;
  shortleg) shortleg "${2:-}" "${3:-8719}" ;;
  lcd4leg) lcd4 "${2:-}" "${3:-64k}" "${4:-8729}" ;;
  plainleg) plainbench "${2:-}" "${3:-32768}" ;;
  kld8leg) kld8 "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-0}" ;;
  *) echo "usage: $0 kld8|short|plain|lcd4|krot|shortleg <arm> <port>|lcd4leg <arm> <32k|64k> <port>|plainleg <arm> <p>|kld8leg <name> <envv> <kv> <baselogits> <save>"; exit 2 ;;
esac
