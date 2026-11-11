#!/bin/bash
# Leg R: ruquant arm, THEIR RECIPE UNCHANGED + the two fixes (-devd ROCm0, -bs --samplers temperature)
# RUN-CARD sections 3/4/5/8. Usage: leg_r.sh <rep>
set -u -o pipefail
REP=$1
BASE=/root/ruquant-bench/redo
LOG=$BASE/legR-r$REP
mkdir -p "$BASE"
# G2 load: WAIT (outside the lock) until 1-min loadavg < 16, max 40x30s; then flock.
for i in $(seq 1 40); do
  awk '{exit !($1<16)}' /proc/loadavg && break
  [ "$i" = 40 ] && { echo "GATE-LOAD-FAIL $(cut -d' ' -f1 /proc/loadavg)"; exit 97; }
  sleep 30
done
exec 9>/root/gpu.lock
flock -w 1800 9 || { echo "LOCKFAIL R-r$REP"; exit 1; }
# G1 RAM
awk '/MemAvailable/{exit !($2>=12582912)}' /proc/meminfo || { echo "GATE-RAM-FAIL"; exit 99; }
# G3
DEVS=$(python3 -c '
import subprocess,re
o=subprocess.run(["rocm-smi","--showmeminfo","vram"],capture_output=True,text=True).stdout
free=[]
for l in o.splitlines():
    m=re.match(r"GPU\[(\d+)\].*Used Memory \(B\):\s*(\d+)",l)
    if m and int(m.group(2))<2*1024**3: free.append(m.group(1))
print(",".join(free[:2]) if len(free)>=2 else "")')
if [ -z "$DEVS" ] && rocm-smi --showpids 2>/dev/null | grep -q "No KFD PIDs"; then
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
# G4
ss -ltn | grep -qE ":8612 " && { echo "GATE-PORT-FAIL 8612"; exit 95; }
echo "legR r$REP start $(date -Is) DEVS=$DEVS" | tee "$LOG.log"

M=/root/bench-quants/ruquant/Qwen3.8-27B-W4A4-OST-GPTQ-wg64-rb4.gguf
D=/root/bench-quants/ruquant/Qwen3.8-27B-DFlash2-W4A4H-drafter-rb4.gguf
BIN=/root/ruquant-src/build/bin
TR=/root/therock-venv/lib/python3.12/site-packages
# W1 hashes
sha256sum "$M" "$D" | tee -a "$LOG.witness"
# W2 rotation metadata
python3 /root/int4-noquant-wt/scripts/rq_gguf_kv.py "$M" ruquant general.quantized_by | tee -a "$LOG.witness"
python3 /root/int4-noquant-wt/scripts/rq_gguf_kv.py "$D" ruquant general.organization | tee -a "$LOG.witness"

export LD_LIBRARY_PATH=$TR/_rocm_sdk_libraries/lib:$TR/_rocm_sdk_core/lib:$TR/_rocm_sdk_devel/lib:/opt/rocm-7.2.3/lib
export HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_META_TOKEN_GRAPH=0 GGML_META_SPIN=2000000 GGML_CUDA_DISABLE_GRAPHS=1
setsid nohup env HIP_VISIBLE_DEVICES=$DEVS $BIN/llama-server \
  -m "$M" -md "$D" -ngl 99 -fa 1 -lm dio -c 8192 -np 1 -sm tensor -devd ROCm0 \
  --spec-type draft-dflash --spec-draft-n-max 7 -bs --samplers temperature \
  --host 0.0.0.0 --port 8612 --no-webui -lv 4 > "$LOG.server.log" 2>&1 < /dev/null &
SRV=$!
echo "SRV_PID=$SRV" | tee -a "$LOG.witness"
OK=0
for i in $(seq 1 90); do
  kill -0 "$SRV" 2>/dev/null || { echo "SERVERDIED r$REP"; tail -30 "$LOG.server.log" | tee -a "$LOG.log"; exit 3; }
  curl -s -m 5 "http://127.0.0.1:8612/health" | grep -q '"ok"' && { OK=1; break; }
  sleep 10
done
[ "$OK" = 1 ] || { echo "HEALTHFAIL r$REP"; kill -TERM "$SRV"; exit 4; }
# W3 engagement
grep -E "loading model|loading draft model|draft-dflash|n_max=|block_size=" "$LOG.server.log" | head -8 | tee -a "$LOG.witness"
# W4 runtime witness + hard stop on apt runtime
tr '\0' '\n' < /proc/$SRV/environ | grep -E '^(LD_LIBRARY_PATH|GGML_|HSA_)' > "$LOG.env-witness"
grep -oE '/[^ ]*lib(amdhip64|rccl|rocblas)[^ ]*' /proc/$SRV/maps | sort -u | tee -a "$LOG.witness"
grep -q "/opt/rocm-7.2.3/lib/libamdhip64" "$LOG.witness" && { echo "W4-FAIL apt-runtime-mapped - STOP"; kill -TERM "$SRV"; exit 5; }
# W7 cmdline
tr '\0' ' ' < /proc/$SRV/cmdline | tee -a "$LOG.witness"
# W6 VRAM placement
rocm-smi --showmeminfo vram | grep -E "GPU\[|Used" | tee -a "$LOG.witness"
# Warmups
python3 /root/p2-prefill/probe_pp.py 8612 6 >/dev/null 2>&1
for i in 1 2 3; do curl -s -m 600 http://127.0.0.1:8612/completion -d \
 '{"prompt":"Warmup pass for the decode shape.","n_predict":256,"temperature":0,"cache_prompt":false,"ignore_eos":true}' >/dev/null; done
# Measured client (trailing 7 = correct n_max; NOT the artifact-producing 2)
timeout 5400 python3 /root/p2-prefill/bench.py 8612 rq-spec-redo-r$REP "$LOG.reps.jsonl" 7 2>&1 | tail -3 | tee -a "$LOG.log"
# Cross-check with THEIR acceptance harness (~4.27 expected)
BIN=$BIN GPUS=$DEVS ROCM_LIB=$TR/_rocm_sdk_libraries/lib:$TR/_rocm_sdk_core/lib \
  EXTRA="-sm tensor -devd ROCm0" \
  timeout 5400 python3 /root/ruquant-repo/rig/acc_bench.py "$M" "$D" 7 2>&1 | tail -15 | tee -a "$LOG.log"
# W5 acceptance (AUTHORITATIVE)
grep -E 'draft acceptance = ' "$LOG.server.log" | tail -3 | tee -a "$LOG.log"
grep -E 'statistics +draft-dflash' "$LOG.server.log" | tail -1 | tee -a "$LOG.log"
# Teardown (section 8)
kill -TERM "$SRV" 2>/dev/null
for i in $(seq 1 24); do kill -0 "$SRV" 2>/dev/null || break; sleep 5; done
kill -0 "$SRV" 2>/dev/null && { echo "TEARDOWN-STUCK pid=$SRV - STOP, human review"; exit 90; }
sleep 5; rocm-smi --showmeminfo vram | grep -E "Used" | tee -a "$LOG.log"
echo "legR r$REP DONE $(date -Is)" | tee -a "$LOG.log"
