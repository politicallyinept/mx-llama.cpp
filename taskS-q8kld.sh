#!/bin/bash
# taskS lane-2: width-8 KLD gate on the 27B Q8_0 stack (brief-literal taskJ2 -b 8
# protocol: Q8_0 model, wiki-sub 64KB, -b 8 -c 512; kernel ACTIVE at width 8).
# Doctrine: flock /root/gpu.lock -w 900, refuse-if-exists, production-env, TheRock
# runtime via production-env (perplexity is synchronous, no server teardown needed).
# Load-avg gate omitted per taskJ1 ops note (starves under sibling 16-thread servers;
# flock + RAM + GPU-free picker suffice).
set -u -o pipefail
BASE=/root/taskS
BIN=/root/build-qmerge
M=/root/bench-quants/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf
WSUB=/root/taskJ2/wiki-sub.txt
KV40="--cache-type-k q4_0 --cache-type-v q4_0"

gates() { # sets DEVS, holds fd 9
  exec 9>/root/gpu.lock
  flock -w 1800 9 || { echo "LOCKFAIL"; return 1; }
  awk '/MemAvailable/{exit !($2>=12582912)}' /proc/meminfo || { echo "GATE-RAM-FAIL"; flock -u 9; return 99; }
  local i
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
    echo "GPU-BUSY retry $i $(date -Is)"
    sleep 30
  done
  [ -n "$DEVS" ] || { echo "GATE-GPU-FAIL"; flock -u 9; return 96; }
  echo "QUIET total-used-B=$(rocm-smi --showmeminfo vram | grep 'Used Memory' | awk -F': ' '{s+=$2} END {print s}') foreign=$(pgrep -c -x llama-server || true)"
}

kld8q8() { # $1 name $2 envv $3 kvflag $4 base-logits(""=none) $5 save(0|1)
  local NAME=$1 ENVV=$2 KV=$3 BASEL=$4 SAVE=$5
  local out=$BASE/kld8q8-$NAME.log
  [ -e "$out.done" ] && { echo "kld8q8-$NAME SKIP-EXISTS"; return 0; }
  gates || return $?
  echo "kld8q8-$NAME start $(date -Is) DEVS=$DEVS ENV='$ENVV' KV='$KV' BASE='$BASEL'"
  ( source /root/production-env.sh
    export GGML_TP_AR_MTP_SAFE=1 HIP_VISIBLE_DEVICES=$DEVS HSA_KERNARG_POOL_SIZE=67108864
    [ -n "$ENVV" ] && export $ENVV
    local ARGS=( -m "$M" -f "$WSUB" -b 8 -c 512 -ngl 99 -fa on -sm tensor -fit off -lm dio )
    [ -n "$KV" ] && read -r -a KVA <<< "$KV" && ARGS+=( "${KVA[@]}" )
    [ -n "$BASEL" ] && ARGS+=( --kl-divergence-base "$BASEL" --kl-divergence )
    [ "$SAVE" = 1 ] && ARGS+=( --save-all-logits "$BASE/kld8q8-$NAME.logits" )
    timeout 3600 "$BIN/bin/llama-perplexity" "${ARGS[@]}" > "$out" 2>&1 < /dev/null 9>&-
    rc=$?; echo "rc=$rc" >> "$out"; exit $rc )
  local rc=$?
  grep -E "Mean    KLD|Maximum KLD|Mean PPL\(Q\)|Mean    PPL|rc=" "$out" | tail -6
  grep -c "FA-DOT8" "$out" > "$out.dot8-hits"
  if [ "$rc" = 0 ] && [ -n "$BASEL" ] && ! grep -q "Mean    KLD" "$out"; then echo "KLD-GUARD-FAIL: no KLD in $out"; rc=1; fi
  if [ "$rc" = 0 ] && [ "$SAVE" = 1 ] && [ ! -s "$BASE/kld8q8-$NAME.logits" ]; then echo "SAVE-GUARD-FAIL: no logits"; rc=1; fi
  flock -u 9
  [ "$rc" = 0 ] && touch "$out.done"
  echo "kld8q8-$NAME DONE rc=$rc dot8-hits=$(cat "$out.dot8-hits") $(date -Is)"
}

retry() { local t; for t in 1 2 3 4; do "$@" && return 0; echo "RETRY-$t fail: $* $(date -Is)"; sleep 60; done; return 1; }
retry kld8q8 base8    "" "" "" 1
retry kld8q8 q40off   "" "$KV40" "" 1
retry kld8q8 q40kld   "" "$KV40" "$BASE/kld8q8-base8.logits" 0
retry kld8q8 d1       "GGML_CUDA_FA_DOT8=1" "$KV40" "$BASE/kld8q8-q40off.logits" 0
retry kld8q8 d2       "GGML_CUDA_FA_DOT8=2" "$KV40" "$BASE/kld8q8-q40off.logits" 0
retry kld8q8 d1vsbase "GGML_CUDA_FA_DOT8=1" "$KV40" "$BASE/kld8q8-base8.logits" 0
retry kld8q8 d2vsbase "GGML_CUDA_FA_DOT8=2" "$KV40" "$BASE/kld8q8-base8.logits" 0
echo "Q8KLD-CHAIN-END $(date -Is)"
