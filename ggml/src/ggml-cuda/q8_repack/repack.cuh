// Public API for the Q8_0 repacked-weight matmul path .
#pragma once

#include "../common.cuh"

#if defined(GGML_USE_MUSA)

inline bool ggml_backend_buft_is_cuda_repack(ggml_backend_buffer_type_t) {
    return false;
}

inline ggml_backend_buffer_type_t ggml_backend_cuda_repack_buffer_type(int) {
    return nullptr;
}

inline bool ggml_cuda_repack_tensor_supported(const ggml_tensor *) {
    return false;
}

inline bool ggml_cuda_repack_mul_mat_should_fire(const ggml_tensor *) {
    return false;
}

inline bool ggml_cuda_q8_yield_eligible(const ggml_tensor *) {
    return false;
}

inline void ggml_cuda_repack_set_tensor_async(
        int, cudaStream_t, ggml_tensor *, const void *, size_t, size_t) {
    GGML_ABORT("repacked weights are unavailable on MUSA");
}

inline void ggml_cuda_repack_async_release(int) {
}

inline bool ggml_cuda_repack_try_release_scratch(int) {
    return false;
}

inline bool ggml_cuda_repack_mmv_fusion_supported(const ggml_tensor *) {
    return false;
}

inline bool ggml_cuda_repack_mmv_id_fusion_supported(const ggml_tensor *) {
    return false;
}

inline void ggml_cuda_mul_mat_repacked(
        ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) {
    GGML_ABORT("repacked weights are unavailable on MUSA");
}

inline void ggml_cuda_mul_mat_id_repacked(
        ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) {
    GGML_ABORT("repacked weights are unavailable on MUSA");
}

inline bool ggml_cuda_repack_mmv_fusion_width_ok(int64_t, bool, ggml_type) {
    return false;
}

inline void ggml_cuda_mul_mat_vec_repacked_fused(
        ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *,
        const ggml_cuda_mm_fusion_args_host *) {
    GGML_ABORT("repacked weights are unavailable on MUSA");
}

inline void ggml_cuda_mul_mat_id_vec_repacked_fused(
        ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, const ggml_tensor *, ggml_tensor *,
        const ggml_cuda_mm_fusion_args_host *) {
    GGML_ABORT("repacked weights are unavailable on MUSA");
}

#else

bool ggml_backend_buft_is_cuda_repack(ggml_backend_buffer_type_t buft);

ggml_backend_buffer_type_t ggml_backend_cuda_repack_buffer_type(int device);

bool ggml_cuda_repack_tensor_supported(const ggml_tensor * t);

bool ggml_cuda_repack_mul_mat_should_fire(const ggml_tensor * src0);

// E4 gcn_kq yield (branch e4-gcnkq-yield): static half of the ggml_cuda_gcn_kq_gemm
// admission for dense Q8_0 — env mask bit 128 of GGML_CUDA_GCN_KQ, K%256==0,
// M>=GGML_CUDA_GCN_GEMM_MIN_M (default 128), ne2==ne3==1, not a view, and the Q8_0
// repack gate on. When true, the repack buffer ALSO stores a canonical block_q8_0
// copy after the repacked record (get_alloc_size grows), and the dense dispatch
// yields GEMMs that gcn_kq itself admits (runtime half: N>=32, alignment, ...) to
// ggml_cuda_gcn_kq_gemm on that copy — exactly the stock arm's kernel regime.
bool ggml_cuda_q8_yield_eligible(const ggml_tensor * t);

// Async-upload path: canonical chunks stage into per-device scratch and the
// device-side repack kernel runs when the tensor completes. The scratch is
// released on the first graph compute after a load.
void ggml_cuda_repack_set_tensor_async(int device, cudaStream_t stream,
    ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_cuda_repack_async_release(int device);
// Frees the device scratch early when no tensor upload is in progress, and returns whether any VRAM was released.
bool ggml_cuda_repack_try_release_scratch(int device);

// Dense fused MMV admits Q8_0 and MXFP4 only.
bool ggml_cuda_repack_mmv_fusion_supported(const ggml_tensor * src0);

// The MoE (mul_mat_id) fused MMV admits Q8_0, MXFP4 and, behind GGML_CUDA_REPACK_KQUANT_MOE_FUSION, the Q4_K, Q5_K, Q6_K and IQ4_NL expert weights.
bool ggml_cuda_repack_mmv_id_fusion_supported(const ggml_tensor * src0);

void ggml_cuda_mul_mat_repacked(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_cuda_mul_mat_id_repacked(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
    ggml_tensor * dst);

// True when a batch of n_tokens is narrow enough that the repacked path runs
// the multi-column mat-vec, which is the only shape the fused FFN entry points
// below can serve. Wider batches fall through to the tiled GEMM.
bool ggml_cuda_repack_mmv_fusion_width_ok(int64_t n_tokens, bool has_ids, ggml_type wt);

void ggml_cuda_mul_mat_vec_repacked_fused(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const ggml_cuda_mm_fusion_args_host * fusion);

void ggml_cuda_mul_mat_id_vec_repacked_fused(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
    ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion);

#endif
