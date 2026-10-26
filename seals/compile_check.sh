#!/usr/bin/env bash
# Compile-clean check: syntax-check the modified TUs with the exact production
# flags (from build-merge/compile_commands.json), pointed at the worktree.
# CPU-side compilation only - no GPU is touched.
set -euo pipefail
WT=${WT:-/root/wt-sumw}
set +u; source /root/w4a4-env.sh; set -u
FLAGS='-DGGML_BACKEND_BUILD -DGGML_BACKEND_SHARED -DGGML_HIP_GRAPHS -DGGML_HIP_NO_VMM -DGGML_SCHED_MAX_COPIES=16 -DGGML_SCHED_MAX_SPLIT_INPUTS=64 -DGGML_SHARED -DGGML_USE_HIP -DGGML_USE_NCCL -DUSE_PROF_API=1 -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -Dggml_hip_EXPORTS -DGGML_CUDA_FA_BF16_BF16=1 -DGGML_CUDA_FA_F16_F16=1 -DGGML_CUDA_FA_Q4_0_Q4_0=1 -DGGML_CUDA_FA_Q8_0_Q8_0=1 -DGGML_CUDA_FA_QUANTS="q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16"'
INC="-I$WT/ggml/src/ggml-hip/.. -I$WT/ggml/src/../include"
QR=$WT/ggml/src/ggml-cuda/q8_repack
rc=0
for f in repack-common.cu buffer.cu mul-mat.cu mul-mat-id.cu; do
    echo "== $f"
    # shellcheck disable=SC2086
    if /opt/rocm/lib/llvm/bin/clang++ $FLAGS $INC -O2 -std=gnu++17 --offload-arch=gfx906 -fsyntax-only -x hip "$QR/$f" 2>/tmp/cc_$f.log; then
        echo "   OK"
    else
        echo "   FAIL"; tail -30 "/tmp/cc_$f.log"; rc=1
    fi
done
exit $rc
