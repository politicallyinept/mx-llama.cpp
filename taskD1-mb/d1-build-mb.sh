#!/bin/bash
# taskD1: build the vec_dot-rewrite microbench. Extracts the generic mmvq kernel
# VERBATIM from mmvq.cu (no transcription, same range logic as k3-build-mb.sh)
# and compiles the standalone harness with the pinned TheRock compiler,
# -ffp-contract=off (host and device share one strict IEEE-754 binary32 sequence).
set -e
TREE=${1:-/root/wt-d1}
GC=$TREE/ggml/src/ggml-cuda
MB=$TREE/taskD1-mb
CXX=/root/therock-venv/lib/python3.12/site-packages/_rocm_sdk_devel/bin/amdclang++

# extraction range: mmvq_should_prefetch .. end of mul_mat_vec_q kernel (dynamic,
# survives small edits above/below the range)
S=$(grep -n "^#if __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK" $GC/mmvq.cu | head -1 | cut -d: -f1)
M=$(grep -n "^// Dedicated MoE multi-token kernel." $GC/mmvq.cu | head -1 | cut -d: -f1)
E=$((M-2))
[ "$(sed -n "${E}p" $GC/mmvq.cu)" = "}" ] || { echo "FATAL: mmvq.cu line $E is not the kernel close - re-locate the range"; exit 1; }
sed -n "${S},${E}p" $GC/mmvq.cu > $MB/mb_mmvq_extract.cuh
wc -l $MB/mb_mmvq_extract.cuh

# CPU seal first (independent of GPU, must pass before any GPU run)
g++ -O2 -std=gnu++17 -ffp-contract=off $MB/d1-cpu-seal.cpp -o $MB/d1-cpu-seal
$MB/d1-cpu-seal || { echo "CPU-SEAL-FAILED"; exit 1; }

$CXX -x hip --offload-arch=gfx906 -DGGML_USE_HIP -D__HIP_PLATFORM_AMD__=1 \
  -I"$TREE/ggml/src" -I"$TREE/ggml/include" -I"$GC" \
  -O3 -std=gnu++17 -ffp-contract=off \
  "$MB/rp_d1_vecdot_test.cu" -o $MB/rp_d1_vecdot_test \
  2> $MB/build.log
RC=$?
tail -5 $MB/build.log
ls -la $MB/rp_d1_vecdot_test
echo "MB-BUILD-RC=$RC"
