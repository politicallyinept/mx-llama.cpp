#!/bin/bash
# Leg T: 8-token forward timing, BOTH engines (RUN-CARD section 7; DIAGNOSTIC-ONLY).
# Usage: leg_t.sh <rq|fork> <tag>
set -u -o pipefail
ENG=$1; TAG=$2; PLIST=${3:-1,6,8}; REPS=${4:-8}
BASE=/root/ruquant-bench/redo
LOG=$BASE/legT-$ENG-$TAG
mkdir -p "$BASE"
# G2 load: WAIT (outside the lock) until 1-min loadavg < 16, max 40x30s; then flock.
for i in $(seq 1 40); do
  awk '{exit !($1<16)}' /proc/loadavg && break
  [ "$i" = 40 ] && { echo "GATE-LOAD-FAIL $(cut -d' ' -f1 /proc/loadavg)"; exit 97; }
  sleep 30
done
exec 9>/root/gpu.lock
flock -w 1800 9 || { echo "LOCKFAIL T-$ENG"; exit 1; }
# G1 RAM
awk '/MemAvailable/{exit !($2>=12582912)}' /proc/meminfo || { echo "GATE-RAM-FAIL"; exit 99; }
echo "start load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$LOG.log"
# G3 two free GPUs
DEVS=$(python3 -c '
import subprocess,re
o=subprocess.run(["rocm-smi","--showmeminfo","vram"],capture_output=True,text=True).stdout
free=[]
for l in o.splitlines():
    m=re.match(r"GPU\[(\d+)\].*Used Memory \(B\):\s*(\d+)",l)
    if m and int(m.group(2))<2*1024**3: free.append(m.group(1))
print(",".join(free[:2]) if len(free)>=2 else "")')
if [ -z "$DEVS" ] && rocm-smi --showpids 2>/dev/null | grep -q "No KFD PIDs"; then
  # leaked-VRAM tolerated mode: no live kfd tenant on the box; use cards whose
  # remaining free VRAM still fits the leg (used < 25 GB of 32 GB).
  DEVS=$(python3 -c '
import subprocess,re
o=subprocess.run(["rocm-smi","--showmeminfo","vram"],capture_output=True,text=True).stdout
free=[]
for l in o.splitlines():
    m=re.match(r"GPU\[(\d+)\].*Used Memory \(B\):\s*(\d+)",l)
    if m and int(m.group(2))<25*1024**3: free.append(m.group(1))
print(",".join(free[:2]) if len(free)>=2 else "")')
  [ -n "$DEVS" ] && echo "LEAK-TOLERATED: no kfd holder, using cards with leaked VRAM: $DEVS"
fi
[ -n "$DEVS" ] || { echo "GATE-GPU-FAIL"; exit 96; }
echo "T-$ENG start $(date -Is) DEVS=$DEVS" | tee "$LOG.log"

if [ "$ENG" = rq ]; then
  TR=/root/therock-venv/lib/python3.12/site-packages
  export LD_LIBRARY_PATH=$TR/_rocm_sdk_libraries/lib:$TR/_rocm_sdk_core/lib:$TR/_rocm_sdk_devel/lib:/opt/rocm-7.2.3/lib
  export HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_META_TOKEN_GRAPH=0 GGML_META_SPIN=2000000 GGML_CUDA_DISABLE_GRAPHS=1
  M=/root/bench-quants/ruquant/Qwen3.8-27B-W4A4-OST-GPTQ-wg64-rb4.gguf
  EXTRA_FLAGS="-ngl 99 -fa 1 -lm dio -sm tensor"
else
  source /root/w4a4-env.sh
  export GGML_CUDA_ALLREDUCE=internal GGML_CUDA_REPACK_KQUANT=0 GGML_GDN_COLTHREAD=1
  export HSA_KERNARG_POOL_SIZE=67108864 GPU_MAX_HW_QUEUES=8
  M=/root/bench-quants/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf
  EXTRA_FLAGS="-ngl 99 -fa on -lm dio -sm tensor"
fi
export HIP_VISIBLE_DEVICES=$DEVS
BIN=/root/ruquant-src/build/bin; [ "$ENG" = fork ] && BIN=/root/int4-noquant-wt/build/bin
[ -x "$BIN/llama-bench" ] || { echo "NO-BINARY $BIN/llama-bench"; exit 1; }
"$BIN/llama-bench" -m "$M" $EXTRA_FLAGS -p $PLIST -n 0 -r $REPS -o jsonl 2>&1 | tee -a "$LOG.log"
echo "T-$ENG DONE $(date -Is)" | tee -a "$LOG.log"
