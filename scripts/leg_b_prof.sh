#!/bin/bash
# Task B profiling leg: GGML_DFLASH_PROF step attribution for DFlash2 on the fork.
# Usage: leg_b_prof.sh <A|B|C|D>   (cells per PROF-PLAN-dflash-step.md section 3)
#   A: n_max 7 np1 (width 8)   B: n_max 3 np1 (width 4)
#   C: n_max 7 np4 (wide)      D: MTP d2 baseline (no drafter file)
set -u -o pipefail
CELL=$1
BASE=/root/ruquant-bench/redo
LOG=$BASE/legB-$CELL
mkdir -p "$BASE"
BIN=/root/int4-noquant-wt/build/bin
M=/root/bench-quants/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf
D=/mnt/old/home/brad/models/Qwen3.8-27B-DFlash2-Q8_0.gguf
# G2 load wait (outside the lock)
for i in $(seq 1 40); do
  awk '{exit !($1<16)}' /proc/loadavg && break
  [ "$i" = 40 ] && { echo "GATE-LOAD-FAIL $(cut -d' ' -f1 /proc/loadavg)"; exit 97; }
  sleep 30
done
exec 9>/root/gpu.lock
flock -w 1800 9 || { echo "LOCKFAIL B-$CELL"; exit 1; }
awk '/MemAvailable/{exit !($2>=12582912)}' /proc/meminfo || { echo "GATE-RAM-FAIL"; exit 99; }
probe_gpus() {
python3 -c '
import subprocess,re
o=subprocess.run(["rocm-smi","--showmeminfo","vram"],capture_output=True,text=True).stdout
free=[]
for l in o.splitlines():
    m=re.match(r"GPU\[(\d+)\].*Used Memory \(B\):\s*(\d+)",l)
    if m and int(m.group(2))<2*1024**3: free.append(m.group(1))
print(",".join(free[:2]) if len(free)>=2 else "")'
}
probe_leak() {
python3 -c '
import subprocess,re
o=subprocess.run(["rocm-smi","--showmeminfo","vram"],capture_output=True,text=True).stdout
free=[]
for l in o.splitlines():
    m=re.match(r"GPU\[(\d+)\].*Used Memory \(B\):\s*(\d+)",l)
    if m and int(m.group(2))<25*1024**3: free.append(m.group(1))
print(",".join(free[:2]) if len(free)>=2 else "")'
}
# G3: WAIT for two free GPUs (fleet/other-agent co-tenants), max 60x60s, then leak-tolerated fallback
DEVS=""
for gi in $(seq 1 60); do
  DEVS=$(probe_gpus)
  [ -n "$DEVS" ] && break
  if [ $gi -ge 2 ] && rocm-smi --showpids 2>/dev/null | grep -q "No KFD PIDs"; then
    DEVS=$(probe_leak)
    [ -n "$DEVS" ] && { echo "LEAK-TOLERATED: $DEVS"; break; }
  fi
  [ "$gi" = 60 ] && { echo "GATE-GPU-FAIL after wait"; exit 96; }
  sleep 60
done


case "$CELL" in
  A) NMAX=7; NP=1; SPEC='--spec-type draft-dflash --spec-draft-n-max 7'; PORT=8613; MD="-md $D" ;;
  B) NMAX=3; NP=1; SPEC='--spec-type draft-dflash --spec-draft-n-max 3'; PORT=8614; MD="-md $D" ;;
  C) NMAX=7; NP=4; SPEC='--spec-type draft-dflash --spec-draft-n-max 7'; PORT=8615; MD="-md $D" ;;
  D) NMAX=2; NP=1; SPEC='--spec-type draft-mtp --spec-draft-n-max 2';   PORT=8616; MD="" ;;
  *) echo "BAD-CELL $CELL"; exit 2 ;;
esac
echo "legB $CELL start $(date -Is) DEVS=$DEVS nmax=$NMAX np=$NP port=$PORT" | tee "$LOG.log"

source /root/w4a4-env.sh
export GGML_CUDA_ALLREDUCE=internal GGML_CUDA_REPACK_KQUANT=0 GGML_GDN_COLTHREAD=1
export HIP_VISIBLE_DEVICES=$DEVS HSA_KERNARG_POOL_SIZE=67108864 GPU_MAX_HW_QUEUES=8
# Profiling instruments (attribution run; t/s from this leg is NOT comparable to clean legs)
export GGML_DFLASH_PROF=1 GGML_META_PROFILE=1 GGML_META_TG_DEBUG=1
export GGML_CUDA_GRAPH_STATS=1 LLAMA_SERVER_TIMING=1
setsid nohup env HIP_VISIBLE_DEVICES=$DEVS $BIN/llama-server -m "$M" $MD \
  --host 0.0.0.0 --port $PORT -ngl 99 -fa on -sm tensor --fit off \
  -c 4096 -ub 512 -t 16 -np $NP -lm dio $SPEC \
  > "$LOG.server.log" 2>&1 < /dev/null &
SRV=$!
echo "SRV_PID=$SRV" | tee -a "$LOG.log"
OK=0
for i in $(seq 1 90); do
  kill -0 "$SRV" 2>/dev/null || { echo "SERVERDIED B-$CELL"; tail -25 "$LOG.server.log" | tee -a "$LOG.log"; exit 3; }
  curl -s -m 5 "http://127.0.0.1:$PORT/health" | grep -q '"ok"' && { OK=1; break; }
  sleep 10
done
[ "$OK" = 1 ] || { echo "HEALTHFAIL B-$CELL"; kill -TERM "$SRV"; exit 4; }
# Engagement witnesses
grep -iE "draft-dflash|draft-mtp|n_max|speculative" "$LOG.server.log" | head -5 | tee -a "$LOG.log"
tr '\0' ' ' < /proc/$SRV/cmdline | tee -a "$LOG.witness"

# Prompts: 2 prose from the natural set + 1 structured JSON-ish
P1=$(python3 -c "import json;d=json.load(open('/root/mtp-natural/prompts.json'));ps=d['prompts'] if isinstance(d,dict) else d;e=ps[0];print(json.dumps(e.get('text_pp') or e.get('prompt') or e.get('text','')))")
P2=$(python3 -c "import json;d=json.load(open('/root/mtp-natural/prompts.json'));ps=d['prompts'] if isinstance(d,dict) else d;e=ps[3];print(json.dumps(e.get('text_pp') or e.get('prompt') or e.get('text','')))")
P3='"Write a JSON config for a web service: port 8443, 4 worker threads, TLS cert paths, a cache block with 3 backends and per-backend timeouts, then explain each field in one sentence."'
req() { curl -s -m 900 "http://127.0.0.1:$PORT/completion" -d "{\"prompt\":$1,\"n_predict\":256,\"temperature\":0,\"cache_prompt\":false,\"ignore_eos\":true}" > /dev/null; }
# warmups (2x same shape)
req "$P1"; req "$P1"
# measured: 3x prose, 3x structured
req "$P1"; req "$P2"; req "$P1"; req "$P3"; req "$P2"; req "$P3"
# np4 cell: fire 4 concurrent mid-run too
if [ "$CELL" = C ]; then
  req "$P1" & c1=$!; req "$P2" & c2=$!; req "$P3" & c3=$!; req "$P1" & c4=$!
  wait $c1 $c2 $c3 $c4
fi
grep -E 'draft acceptance' "$LOG.server.log" | tail -2 | tee -a "$LOG.log"
grep -E 'server_loop_timing|t_dec_' "$LOG.server.log" | tail -3 | tee -a "$LOG.log"
kill -TERM "$SRV" 2>/dev/null
for i in $(seq 1 24); do kill -0 "$SRV" 2>/dev/null || break; sleep 5; done
kill -0 "$SRV" 2>/dev/null && { echo "TEARDOWN-STUCK pid=$SRV - STOP"; exit 90; }
sleep 5; rocm-smi --showmeminfo vram | grep Used | head -3 | tee -a "$LOG.log"
echo "legB $CELL DONE $(date -Is)" | tee -a "$LOG.log"
