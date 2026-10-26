#pragma once

#include "common.cuh"

// GDN decode gate chain fusion (ported from CV hc-dec.cu): MUL_MAT(alpha) -> RESHAPE -> ADD(dt) -> SOFTPLUS -> MUL(A)
// and MUL_MAT(beta) -> RESHAPE -> SIGMOID in one kernel. Env gate GGML_CUDA_GDN_AB_DEC (default OFF here, ON in CV).
bool ggml_cuda_gdn_ab_dec_supported(int cc, const ggml_tensor * mm_a, const ggml_tensor * add, const ggml_tensor * sp,
                                    const ggml_tensor * mul, const ggml_tensor * mm_b, const ggml_tensor * sig);
void ggml_cuda_gdn_ab_dec(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_a, const ggml_tensor * add, const ggml_tensor * mm_b,
                          ggml_tensor * gate, ggml_tensor * sig);
