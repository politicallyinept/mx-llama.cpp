#!/bin/bash
# Leg F5: F4q + FORCED 4-bit activations (GGML_CUDA_W4A4_MIN_TOKENS=0 explicit
# overrides the speed preset; profile stays speed for DOT8=3/KSPLIT/SUMW).
# CONFOUND (documented): MIN_TOKENS is process-global - the TARGET verify batches
# (width 2-5) also go u4act, and the target logits shift (acceptance may move).
# Lever probe, NOT a shippable config (target quality cost at narrow widths).
# RUN-CARD sections 3/6/8. Usage: leg_f.sh <rep>
set -u -o pipefail
REP=$1
BASE=/root/ruquant-bench/redo
LOG=$BASE/legF5-r$REP
mkdir -p "$BASE"
# G2 load: WAIT (outside the lock) until 1-min loadavg < 16, max 40x30s; then flock.
for i in $(seq 1 40); do
  awk '{exit !($1<16)}' /proc/loadavg && break
  [ "$i" = 40 ] && { echo "GATE-LOAD-FAIL $(cut -d' ' -f1 /proc/loadavg)"; exit 97; }
  sleep 30
done
exec 9>/root/gpu.lock
flock -w 1800 9 || { echo "LOCKFAIL F5-r$REP"; exit 1; }
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

ss -ltn | grep -qE ":8197 " && { echo "GATE-PORT-FAIL 8197"; exit 95; }
BIN=/root/int4-noquant-wt/build/bin
[ -x "$BIN/llama-server" ] || { echo "NO-BINARY build missing - run F0 first"; exit 1; }
echo "legF5 r$REP start $(date -Is) DEVS=$DEVS" | tee "$LOG.log"

M=/root/bench-quants/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf
D=/mnt/old/home/brad/models/Qwen3.8-27B-DFlash2-Q4_0.gguf
# W1 hashes
sha256sum "$M" "$D" | tee -a "$LOG.witness"
# W2 expect NO ruquant.* keys on both files (stock arm)
python3 /root/int4-noquant-wt/scripts/rq_gguf_kv.py "$M" ruquant 2>&1 | tee -a "$LOG.witness"
python3 /root/int4-noquant-wt/scripts/rq_gguf_kv.py "$D" ruquant 2>&1 | tee -a "$LOG.witness"

source /root/w4a4-env.sh
export GGML_CUDA_ALLREDUCE=internal GGML_CUDA_REPACK_KQUANT=0 GGML_GDN_COLTHREAD=1
export HIP_VISIBLE_DEVICES=$DEVS HSA_KERNARG_POOL_SIZE=67108864 GPU_MAX_HW_QUEUES=8
# FORCED W4A4: explicit env beats the speed preset (w4a4-profile.h resolve order)
export GGML_CUDA_W4A4_MIN_TOKENS=0
# phase attribution (draft fwd / verify per width)
export GGML_DFLASH_PROF=1
# NOTE: GGML_TP_AR_MTP_SAFE deliberately NOT set (MTP-specific); one-shot retry WITH it only if first decode aborts.
setsid nohup env HIP_VISIBLE_DEVICES=$DEVS $BIN/llama-server -m "$M" -md "$D" \
  --host 0.0.0.0 --port 8197 -ngl 99 -fa on -sm tensor --fit off \
  -c 65536 -ub 1024 -t 16 -np 1 -lm dio \
  --spec-type draft-dflash --spec-draft-n-max 4 > "$LOG.server.log" 2>&1 < /dev/null &
SRV=$!
echo "SRV_PID=$SRV" | tee -a "$LOG.witness"
OK=0
for i in $(seq 1 90); do
  kill -0 "$SRV" 2>/dev/null || { echo "SERVERDIED f-r$REP"; tail -30 "$LOG.server.log" | tee -a "$LOG.log"; exit 3; }
  curl -s -m 5 "http://127.0.0.1:8197/health" | grep -q '"ok"' && { OK=1; break; }
  sleep 10
done
[ "$OK" = 1 ] || { echo "HEALTHFAIL f-r$REP"; kill -TERM "$SRV"; exit 4; }
# W3 engagement (fork wording may differ; advisory)
grep -iE "loading draft model|draft-dflash|draft_dflash|n_max" "$LOG.server.log" | head -8 | tee -a "$LOG.witness"
# W4A4-forced witness: resolved floor must read MIN_TOKENS=0 [explicit]
grep -E "W4A4_PROFILE|MIN_TOKENS" "$LOG.server.log" | head -2 | tee -a "$LOG.witness"
grep -q "MIN_TOKENS=0" "$LOG.witness" || { echo "WITNESS-FAIL: floor not forced"; kill -TERM "$SRV"; exit 6; }
# W7 cmdline + env witness
tr '\0' ' ' < /proc/$SRV/cmdline | tee -a "$LOG.witness"
tr '\0' '\n' < /proc/$SRV/environ | grep -E '^(LD_LIBRARY_PATH|GGML_|HSA_|HIP_)' > "$LOG.env-witness"
# W6: replicated drafter -> both GPUs carry draft (~2.0GB delta each)
rocm-smi --showmeminfo vram | grep -E "GPU\[|Used" | tee -a "$LOG.witness"
# Warmups (same protocol)
python3 /root/p2-prefill/probe_pp.py 8197 6 >/dev/null 2>&1
for i in 1 2 3; do curl -s -m 600 http://127.0.0.1:8197/completion -d \
 '{"prompt":"Warmup pass for the decode shape.","n_predict":256,"temperature":0,"cache_prompt":false,"ignore_eos":true}' >/dev/null; done
# Measured client (trailing 4 = draft 4)
timeout 5400 python3 /root/p2-prefill/bench.py 8197 fork-q40-dflash4q-w4a4forced-r$REP "$LOG.reps.jsonl" 4 2>&1 | tail -3 | tee -a "$LOG.log"
# W5 acceptance (AUTHORITATIVE)
grep -E 'draft acceptance = ' "$LOG.server.log" | tail -3 | tee -a "$LOG.log"
grep -iE 'statistics +draft' "$LOG.server.log" | tail -1 | tee -a "$LOG.log"
# Teardown
kill -TERM "$SRV" 2>/dev/null
for i in $(seq 1 24); do kill -0 "$SRV" 2>/dev/null || break; sleep 5; done
kill -0 "$SRV" 2>/dev/null && { echo "TEARDOWN-STUCK pid=$SRV - STOP, human review"; exit 90; }
sleep 5; rocm-smi --showmeminfo vram | grep -E "Used" | tee -a "$LOG.log"
echo "legF r$REP DONE $(date -Is)" | tee -a "$LOG.log"
