#!/bin/bash
# F0: build INT4-noquant (RUN-CARD-ruquant-spec-redo.md section 6; flags = cvport-new-src CMakeCache-verified)
# Doubles as compile verification for the GGML_DFLASH_PROF instrumentation hunks.
set -u
LOG=/root/int4-noquant-wt/scripts/f0-build.log
AVAIL=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
echo "F0 start $(date -Is) MemAvailable=${AVAIL}GB" | tee "$LOG"
if [ "$AVAIL" -lt 12 ]; then echo "SKIPPED-RAM ${AVAIL}GB" | tee -a "$LOG"; exit 99; fi
cd /root/int4-noquant-wt || { echo "WT-MISSING" | tee -a "$LOG"; exit 1; }
TR=/root/therock-venv/lib/python3.12/site-packages
cmake -B build -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906 \
  -DCMAKE_HIP_COMPILER=$TR/_rocm_sdk_devel/bin/amdclang++ \
  -DCMAKE_PREFIX_PATH="$TR/_rocm_sdk_core;$TR/_rocm_sdk_libraries;$TR/_rocm_sdk_devel" \
  -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON \
  -DGGML_HIP_RCCL=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF \
  -DCMAKE_BUILD_TYPE=Release >> "$LOG" 2>&1 || { echo "CONFIGURE-FAILED rc=$?" | tee -a "$LOG"; exit 1; }
echo "configure ok $(date -Is)" | tee -a "$LOG"
cmake --build build -j8 >> "$LOG" 2>&1 || { echo "BUILD-FAILED rc=$? (tail:)" | tee -a "$LOG"; tail -15 "$LOG" | tee -a "$LOG"; exit 2; }
echo "== artifacts ==" >> "$LOG"
ls -la build/bin/llama-server build/bin/llama-bench >> "$LOG" 2>&1
build/bin/llama-server --version 2>/dev/null | head -1 >> "$LOG"
echo "F0 DONE $(date -Is)" | tee -a "$LOG"
