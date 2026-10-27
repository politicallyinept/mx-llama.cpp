#pragma once

#include "common.cuh"

// Shared-expert / dense-f32 decode fusion (ported from CV hc-dec.cu + gcn-f16-matvec.cu).
// Env gate GGML_CUDA_SHE_FUSE (default OFF here, ON in CV). Trace knob GGML_CUDA_SHE_FUSE_TRACE=1 prints every
// fusion firing plus every remaining f32 mul_mat_vec_f launch (kernel-count-drop witness).

// shared-expert gate tail: MUL_MAT(1-row gate, x) -> SIGMOID -> MUL(shexp) -> ADD(moe) in one kernel (<= 8 tokens)
bool ggml_cuda_shexp_gate_add_supported(const ggml_tensor * mm, const ggml_tensor * sig, const ggml_tensor * mul,
                                        const ggml_tensor * add);
void ggml_cuda_shexp_gate_add(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * sig,
                              const ggml_tensor * mul, ggml_tensor * add);

// dense F32 x F32 matvec for 1..4 tokens and 65..8192 rows (MoE router): coalesced 16-byte loads, K split over waves
bool ggml_cuda_gcn_f32_matvec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// GGML_CUDA_SHE_FUSE_TRACE=1
bool ggml_cuda_she_fuse_trace(void);
