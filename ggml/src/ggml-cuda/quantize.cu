#include "quantize.cuh"
#include "q8_repack/repack-common.cuh"   // block_q8_1_u4 (milestone 1)
#include <cstdint>

#if defined(BLACKWELL_MMA_AVAILABLE)
// this maps to 256-bit loads in PTX on supported devices,
// and otherwise falls back to 2 128-bit loads
struct __builtin_align__(32) float8 {
    float x; float y; float z; float w;
    float p; float q; float r; float s;
};

#if CUDART_VERSION >= 12080
static __device__ __forceinline__ float nvfp4_native_scale_error(
        const float vals[QK_NVFP4_SUB], const float inv_col_scale, const float inv_scale, const float scale) {
    const float scale_dequant = 2.0f * scale;
    float err = 0.0f;

#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB; k += 4) {
        const float v0 = vals[k + 0] * inv_col_scale;
        const float v1 = vals[k + 1] * inv_col_scale;
        const float v2 = vals[k + 2] * inv_col_scale;
        const float v3 = vals[k + 3] * inv_col_scale;

        const __nv_fp4x4_e2m1 q(make_float4(v0 * inv_scale, v1 * inv_scale, v2 * inv_scale, v3 * inv_scale));
        const __nv_fp4x4_storage_t q_storage = q.__x;
        const __nv_fp4x2_storage_t q_lo = static_cast<__nv_fp4x2_storage_t>(q_storage);
        const __nv_fp4x2_storage_t q_hi = static_cast<__nv_fp4x2_storage_t>(q_storage >> 8U);

        const __half2_raw hraw2_lo = __nv_cvt_fp4x2_to_halfraw2(q_lo, __NV_E2M1);
        const __half2_raw hraw2_hi = __nv_cvt_fp4x2_to_halfraw2(q_hi, __NV_E2M1);
        const __half2 h2_lo = static_cast<__half2>(hraw2_lo);
        const __half2 h2_hi = static_cast<__half2>(hraw2_hi);
        const float2 dq_lo = __half22float2(h2_lo);
        const float2 dq_hi = __half22float2(h2_hi);

        const float err0 = fabsf(v0) - fabsf(dq_lo.x) * scale_dequant;
        const float err1 = fabsf(v1) - fabsf(dq_lo.y) * scale_dequant;
        const float err2 = fabsf(v2) - fabsf(dq_hi.x) * scale_dequant;
        const float err3 = fabsf(v3) - fabsf(dq_hi.y) * scale_dequant;

        err = fmaf(err0, err0, err);
        err = fmaf(err1, err1, err);
        err = fmaf(err2, err2, err);
        err = fmaf(err3, err3, err);
    }

    return err;
}
#endif // CUDART_VERSION >= 12080
#endif // defined(BLACKWELL_MMA_AVAILABLE)

template <int block_size>
__launch_bounds__(block_size, 1)
static __global__ void quantize_q8_1(
        const float * x_ptr, void * vy_ptr,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT x  = x_ptr;
    void        * GGML_CUDA_RESTRICT vy = vy_ptr;
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    const int64_t & i00 = i0;
    const int64_t & i01 = i1;
    const int64_t & i02 = i2;
    const int64_t & i03 = i3;

    const int64_t i_cont = ((i3*ne2.z + i2) * ne1 + i1) * ne0 + i0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib  = i_cont / QK8_1; // block index
    const int64_t iqs = i_cont % QK8_1; // quant index

    ggml_cuda_pdl_sync();
    const float xi = i0 < ne00 ? x[i03*s03 + i02*s02 + i01*s01 + i00] : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, sum);
}

__device__ __forceinline__ uint8_t compute_e8m0_scale(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }

    // FP4 E2M1: max exponent (unbiased) is 2.
    constexpr int FP4_E2M1_EMAX = 2;

    const float e = log2f(amax);

    // "even" -> round-to-nearest integer, ties-to-even
    const int e_int = __float2int_rn(e);

    const int shared_exp = e_int - FP4_E2M1_EMAX;

    int biased = shared_exp + 127;

    biased = max(biased, 0);
    biased = min(biased, 254);

    return static_cast<uint8_t>(biased);
}

// scatter: grid over tokens, quantize once, write to all the token's compact rows
template <bool scatter, bool use_aligned_float8>
static __global__ void quantize_mmq_nvfp4(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy, float * __restrict__ scale,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int n_expert_used) {
#if defined(BLACKWELL_MMA_AVAILABLE)

    const int64_t blocks_per_col = (ne0 + QK_FP4_MMQ - 1) / QK_FP4_MMQ;

    int64_t base_idx;
    if constexpr (scatter) {
        base_idx = (int64_t) blockIdx.x * s02; // one physical row per token
    } else {
        const int64_t i2  = blockIdx.y % ne2;
        const int64_t i3  = blockIdx.y / ne2;
        const int64_t i01 = ids ? ids[blockIdx.x] : blockIdx.x;
        base_idx = i3 * s03 + i2 * s02 + i01 * s01;
    }
    const float * __restrict__  x_row = x + base_idx;

    float amax = 0.0f;
    if constexpr (use_aligned_float8) {
        for (int64_t i0 = 8 * threadIdx.x; i0 < ne00; i0 += 8 * blockDim.x) {
            const float * x_base = x_row + i0;
            const float8 v = reinterpret_cast<const float8 *>(x_base)[0];
            amax = fmaxf(amax, fabsf(v.x));
            amax = fmaxf(amax, fabsf(v.y));
            amax = fmaxf(amax, fabsf(v.z));
            amax = fmaxf(amax, fabsf(v.w));
            amax = fmaxf(amax, fabsf(v.p));
            amax = fmaxf(amax, fabsf(v.q));
            amax = fmaxf(amax, fabsf(v.r));
            amax = fmaxf(amax, fabsf(v.s));
        }
    } else {
        for (int64_t i0 = threadIdx.x; i0 < ne00; i0 += blockDim.x) {
            amax = fmaxf(amax, fabsf(x_row[i0]));
        }
    }

    amax = warp_reduce_max<WARP_SIZE>(amax);

    __shared__ float warp_amax[CUDA_QUANTIZE_BLOCK_SIZE_MMQ / WARP_SIZE];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        warp_amax[warp] = amax;
    }
    __syncthreads();

    if (warp == 0) {
        amax = threadIdx.x < int(CUDA_QUANTIZE_BLOCK_SIZE_MMQ / WARP_SIZE) ? warp_amax[lane] : 0.0f;
        amax = warp_reduce_max<WARP_SIZE>(amax);
        if (lane == 0) {
            warp_amax[0] = amax / (6.0f * 448.0f);
            if constexpr (scatter) {
#pragma unroll
                for (int slot = 0; slot < n_expert_used; ++slot) {
                    const int64_t i = ids[(int64_t) blockIdx.x * n_expert_used + slot];
                    scale[i] = warp_amax[0];
                }
            } else {
                scale[blockIdx.y * ne1 + blockIdx.x] = warp_amax[0];
            }
        }
    }
    __syncthreads();

    block_fp4_mmq * y = (block_fp4_mmq *) vy;
    const int64_t n_subblocks = (ne0 + QK_NVFP4_SUB - 1) / QK_NVFP4_SUB;

    for (int64_t isb = threadIdx.x; isb < n_subblocks; isb += blockDim.x) {
        const int64_t i0_base = isb * QK_NVFP4_SUB;
        const int64_t k_block = i0_base / QK_FP4_MMQ;
        const int sub = (i0_base % QK_FP4_MMQ) / QK_NVFP4_SUB;

        const float row_scale = warp_amax[0];
        const float inv_col_scale = row_scale > 0.0f ? 1.0f / row_scale : 0.0f;

        float vals[QK_NVFP4_SUB];
        if constexpr (use_aligned_float8) {
            const float * x_base = x_row + i0_base;
            const float8 v0 = i0_base +  7 < ne00 ? reinterpret_cast<const float8 *>(x_base)[0]     : float8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            const float8 v1 = i0_base + 15 < ne00 ? reinterpret_cast<const float8 *>(x_base + 8)[0] : float8{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            vals[0] = v0.x; vals[1] = v0.y; vals[2] = v0.z; vals[3] = v0.w;
            vals[4] = v0.p; vals[5] = v0.q; vals[6] = v0.r; vals[7] = v0.s;
            vals[8] = v1.x; vals[9] = v1.y; vals[10] = v1.z; vals[11] = v1.w;
            vals[12] = v1.p; vals[13] = v1.q; vals[14] = v1.r; vals[15] = v1.s;
        } else {
#pragma unroll
            for (int k = 0; k < QK_NVFP4_SUB; ++k) {
                const int64_t i00 = i0_base + k;
                vals[k] = i00 < ne00 ? x_row[i00] : 0.0f;
            }
        }

        uint32_t q0 = 0;
        uint32_t q1 = 0;

        float amax_sub = 0.0f;
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB; ++k) {
            amax_sub = fmaxf(amax_sub, fabsf(vals[k] * inv_col_scale));
        }

        static constexpr int test_offsets[5] = { 0, -1, 1, -2, 2 };
        const int first_fp8_code = (int) ggml_cuda_fp32_to_ue4m3(amax_sub / 6.0f);

        uint8_t fp8_code = (uint8_t) first_fp8_code;
        float subblock_scale = ggml_cuda_ue4m3_to_fp32(fp8_code);
        float inv_scale_err = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
#if CUDART_VERSION >= 12080
        float best_err = nvfp4_native_scale_error(vals, inv_col_scale, inv_scale_err, subblock_scale);
#else
        float best_err = 0.0f;
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB; ++k) {
            const float v = vals[k] * inv_col_scale;
            const uint8_t q = ggml_cuda_float_to_fp4_e2m1(v, inv_scale_err);
            const float err_diff = fabsf(v) - fabsf(kvalues_fp4[q & 0x7]) * subblock_scale;
            best_err = fmaf(err_diff, err_diff, best_err);
        }
#endif // CUDART_VERSION >= 12080

#pragma unroll
        for (int i = 1; i < 5; ++i) {
            const int test_code = first_fp8_code + test_offsets[i];
            if (test_code < 0 || test_code > 0x7e) {
                continue;
            }

            const float test_scale = ggml_cuda_ue4m3_to_fp32((uint8_t) test_code);
            const float test_inv_scale = test_scale > 0.0f ? 0.5f / test_scale : 0.0f;
#if CUDART_VERSION >= 12080
            const float cur_err = nvfp4_native_scale_error(vals, inv_col_scale, test_inv_scale, test_scale);
#else
            float cur_err = 0.0f;
#pragma unroll
            for (int k = 0; k < QK_NVFP4_SUB; ++k) {
                const float v = vals[k] * inv_col_scale;
                const uint8_t q = ggml_cuda_float_to_fp4_e2m1(v, test_inv_scale);
                const float err_diff = fabsf(v) - fabsf(kvalues_fp4[q & 0x7]) * test_scale;
                cur_err = fmaf(err_diff, err_diff, cur_err);
            }
#endif // CUDART_VERSION >= 12080

            if (cur_err < best_err) {
                best_err = cur_err;
                fp8_code = (uint8_t) test_code;
                subblock_scale = test_scale;
            }
        }
#if CUDART_VERSION >= 12080
        const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
        const float s = inv_col_scale * inv_scale;

        __nv_fp4x4_e2m1 q0_lo(make_float4(vals[0] * s, vals[8]  * s, vals[1] * s, vals[9]  * s));
        __nv_fp4x4_e2m1 q0_hi(make_float4(vals[2] * s, vals[10] * s, vals[3] * s, vals[11] * s));
        __nv_fp4x4_e2m1 q1_lo(make_float4(vals[4] * s, vals[12] * s, vals[5] * s, vals[13] * s));
        __nv_fp4x4_e2m1 q1_hi(make_float4(vals[6] * s, vals[14] * s, vals[7] * s, vals[15] * s));

        const char2 q0_lo_c = *reinterpret_cast<char2 *>(&q0_lo);
        const char2 q0_hi_c = *reinterpret_cast<char2 *>(&q0_hi);
        const char2 q1_lo_c = *reinterpret_cast<char2 *>(&q1_lo);
        const char2 q1_hi_c = *reinterpret_cast<char2 *>(&q1_hi);

        q0 = uint32_t(uint8_t(q0_lo_c.x)) | (uint32_t(uint8_t(q0_lo_c.y)) <<  8) |
            (uint32_t(uint8_t(q0_hi_c.x)) << 16) | (uint32_t(uint8_t(q0_hi_c.y)) << 24);
        q1 = uint32_t(uint8_t(q1_lo_c.x)) | (uint32_t(uint8_t(q1_lo_c.y)) <<  8) |
            (uint32_t(uint8_t(q1_hi_c.x)) << 16) | (uint32_t(uint8_t(q1_hi_c.y)) << 24);
#else
        const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB / 4; ++k) {
            q0 |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k + 0] * inv_col_scale, inv_scale)) << (8 * k);
            q0 |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k + 8] * inv_col_scale, inv_scale)) << (8 * k + 4);
            q1 |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k + 4] * inv_col_scale, inv_scale)) << (8 * k);
            q1 |= uint32_t(ggml_cuda_float_to_fp4_e2m1(vals[k + 12] * inv_col_scale, inv_scale)) << (8 * k + 4);
        }
#endif // CUDART_VERSION >= 12080

        if constexpr (scatter) {
#pragma unroll
            for (int slot = 0; slot < n_expert_used; ++slot) {
                const int64_t i = ids[(int64_t) blockIdx.x * n_expert_used + slot];
                block_fp4_mmq * yb = y + (k_block * ne1 + i);
                uint32_t * yqs = reinterpret_cast<uint32_t *>(yb->qs);
                yqs[2 * sub + 0] = q0;
                yqs[2 * sub + 1] = q1;
                reinterpret_cast<uint8_t *>(yb->d4)[sub] = fp8_code;
            }
        } else {
            block_fp4_mmq * yb = y + (blockIdx.y * ((int64_t) blocks_per_col * ne1) + k_block * ne1 + blockIdx.x);
            uint32_t * yqs = reinterpret_cast<uint32_t *>(yb->qs);
            yqs[2 * sub + 0] = q0;
            yqs[2 * sub + 1] = q1;
            reinterpret_cast<uint8_t *>(yb->d4)[sub] = fp8_code;
        }
    }
#else
    GGML_UNUSED_VARS(x, ids, vy, scale, ne00, s01, s02, s03, ne0, ne1, ne2, n_expert_used);
    NO_DEVICE_CODE; // This is for Blackwell NVFP4 activations only.
#endif // defined(BLACKWELL_MMA_AVAILABLE)

}

// quantize values in the format mxfp4 is stored which is interleaved nibbles
// i.e. a block a0-a31 is represented as a0a16,a1a17 ...a15a31
// scatter: grid over tokens, quantize once, write to all the token's compact rows
template <bool scatter>
static __global__ void quantize_mmq_mxfp4(const float * __restrict__ x,
                                          const int32_t * __restrict__ ids,
                                          void * __restrict__ vy,
                                          const int64_t ne00,
                                          const int64_t s01,
                                          const int64_t s02,
                                          const int64_t s03,
                                          const int64_t ne0,
                                          const int     ne1,
                                          const int     ne2,
                                          const int     n_expert_used) {
    constexpr int vals_per_scale = 32;
    constexpr int vals_per_warp  = 2 * vals_per_scale;  // Each warp processes 2 blocks of 32 = 64 values

    const int warp_id = threadIdx.y;
    const int lane_id_32 = threadIdx.x;

    const int nwarps = blockDim.y;

    const int64_t warp_start_offset = (blockIdx.y * nwarps + warp_id) * vals_per_warp;

    if (warp_start_offset >= ne0) {
        return;
    }

    const int64_t block_fp4_mmq_size = QK_FP4_MMQ;
    const int64_t k_block            = warp_start_offset / block_fp4_mmq_size;
    const int64_t quad_idx_in_block  = (warp_start_offset % block_fp4_mmq_size) / vals_per_warp;

    const int group_id = lane_id_32 / 4;
    const int lane_in_group = lane_id_32 % 4;
    const int base = group_id * 2;

    ggml_cuda_pdl_sync();
    int64_t base_pos;
    if constexpr (scatter) {
        base_pos = (int64_t) blockIdx.x * s02; // one physical row per token
    } else {
        const int64_t i2  = blockIdx.z % ne2;
        const int64_t i3  = blockIdx.z / ne2;
        const int64_t i01 = ids ? ids[blockIdx.x] : blockIdx.x;
        base_pos = i3 * s03 + i2 * s02 + i01 * s01;
    }

    uint8_t scales[2];
    char2   packed[2];

#pragma unroll
    for (int b = 0; b < 2; ++b) {
        const int64_t i0 = warp_start_offset + b * vals_per_scale + lane_id_32;
        const float xi = (i0 < ne00) ? x[base_pos + i0] : 0.0f;

        float amax = fabsf(xi);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, WARP_SIZE));
        }

        const uint8_t e = compute_e8m0_scale(amax);
        scales[b] = e;
        const float inv_s = (amax == 0.0f) ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(e));

#if CUDART_VERSION >= 12080
        const float scaled_val = xi * inv_s;

        const float val0 = __shfl_sync(0xFFFFFFFF, scaled_val, base, WARP_SIZE);
        const float val1 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 16, WARP_SIZE);
        const float val2 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 1, WARP_SIZE);
        const float val3 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 17, WARP_SIZE);

        __nv_fp4x4_e2m1 fp4_packed(make_float4(val0, val1, val2, val3));
        packed[b] = *(char2 *) &fp4_packed;
#else
        // Fallback: manual FP4 conversion using LUT
        const uint8_t q_val = ggml_cuda_float_to_fp4_e2m1(xi, inv_s);

        const uint8_t q_lo_0 = __shfl_sync(0xFFFFFFFF, q_val, base,      WARP_SIZE);
        const uint8_t q_lo_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 1,  WARP_SIZE);
        const uint8_t q_hi_0 = __shfl_sync(0xFFFFFFFF, q_val, base + 16, WARP_SIZE);
        const uint8_t q_hi_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 17, WARP_SIZE);

        char2 q;
        q.x = (q_hi_0 << 4) | q_lo_0;
        q.y = (q_hi_1 << 4) | q_lo_1;
        packed[b] = q;
#endif // CUDART_VERSION >= 12080
    }

    block_fp4_mmq * y = (block_fp4_mmq *) vy;
    if constexpr (scatter) {
#pragma unroll
        for (int slot = 0; slot < n_expert_used; ++slot) {
            const int64_t i = ids[(int64_t) blockIdx.x * n_expert_used + slot];
            block_fp4_mmq * yb = y + (k_block * ne1 + i);
            char2 * yqs2 = (char2 *) yb->qs;
            if (lane_in_group == 0) {
                yqs2[quad_idx_in_block * 16 + 0 * 8 + group_id] = packed[0];
                yqs2[quad_idx_in_block * 16 + 1 * 8 + group_id] = packed[1];
            }
            if (lane_id_32 == 0) {
                yb->d4[quad_idx_in_block] = (scales[1] << 8) | scales[0];
            }
        }
    } else {
        const int64_t ib0 = blockIdx.z * ((int64_t) ne1 * (ne0 / block_fp4_mmq_size));
        block_fp4_mmq * yb = y + (ib0 + k_block * ne1 + blockIdx.x);
        char2 * yqs2 = (char2 *) yb->qs;
        if (lane_in_group == 0) {
            yqs2[quad_idx_in_block * 16 + 0 * 8 + group_id] = packed[0];
            yqs2[quad_idx_in_block * 16 + 1 * 8 + group_id] = packed[1];
        }
        if (lane_id_32 == 0) {
            yb->d4[quad_idx_in_block] = (scales[1] << 8) | scales[0];
        }
    }
    GGML_UNUSED(n_expert_used);
}

// scatter: grid over tokens, quantize once, write to all the token's compact rows
template <mmq_q8_1_ds_layout ds_layout, bool scatter>
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2, const int n_expert_used) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i00 = i0;
    ggml_cuda_pdl_sync();

    int64_t base_idx;
    if constexpr (scatter) {
        base_idx = (int64_t) blockIdx.x * s02; // one physical row per token
    } else {
        const int64_t i2  = blockIdx.z % ne2;
        const int64_t i3  = blockIdx.z / ne2;
        const int64_t i01 = ids ? ids[blockIdx.x] : blockIdx.x;
        base_idx = i3*s03 + i2*s02 + i01*s01;
    }

    const float4 * x4 = (const float4 *) x;
    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t k_block = i0 / QK8_1_MMQ; // column block in the channel
    const int64_t iqs     = i0 % QK8_1_MMQ; // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(base_idx + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);
    const float d = 1.0f / d_inv;

    // write the block once (normal) or to each of the token's compact rows (scatter)
    const int nwrite = scatter ? n_expert_used : 1;
#pragma unroll
    for (int slot = 0; slot < nwrite; ++slot) {
        int64_t ib;
        if constexpr (scatter) {
            const int64_t i = ids[(int64_t) blockIdx.x * n_expert_used + slot];
            ib = k_block*ne1 + i;
        } else {
            const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
            ib = ib0 + k_block*ne1 + blockIdx.x;
        }

        // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
        char4 * yqs4 = (char4 *) y[ib].qs;
        yqs4[iqs/4] = q;

        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
            if (iqs % 16 == 0 && iqs < 96) {
                y[ib].d2s6[2 + iqs/16] = sum;
                if (iqs % 64 == 0) {
                    y[ib].d2s6[iqs/64] = d;
                }
            }
        } else if (iqs % 32 == 0) {
            if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
                y[ib].ds4[iqs/32] = make_half2(d, sum);
            } else {
                y[ib].d4[iqs/32]  = d;
            }
        }
    }
    GGML_UNUSED(n_expert_used);
}

// W4A4 milestone 2a: fp32 -> block_q8_1_u4 (per-32 asymmetric uint4, scheme B).
// Same rounding / clamps / scales as the host reference quantize_row_q8_1_u4
// (q8_repack/repack-common.cu:449): d4 = (mx-mn)/15, id = 1/d4 (0 if d4 == 0),
// roundf (round-half-away-from-zero), clamp to [0, 15], fp16 round-to-nearest-even
// for d4 and m. DELTA vs the host reference (intentional, per the milestone spec):
// the device kernel stores the SIGNED nibble (q-8)&0xF — the dp8a operand
// convention the milestone-2 dot loop needs (sdot8 + 8*SUMW bookkeeping) — while
// the host reference stores the raw unsigned code q. Parallelism only otherwise:
// one thread handles 4 values, min/max reduce across the 8 threads of each
// 32-value group via shuffles (min/max are exact under any fold order).
// Grid/block/stream structure mirrors quantize_mmq_q8_1 above. Out-of-range
// threads (i0 in [ne00, ne0)) still write the zero-input pattern (nibble 8,
// d4 = m = 0) into the row padding, exactly like the q8_1 MMQ quantizer writes
// zeros there; the padding blocks are never read. NOT wired into any dispatch yet.
static __global__ void quantize_mmq_q8_1_u4(
        const float * __restrict__ x, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    const int64_t i0 = ((int64_t) blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i2  = blockIdx.z % ne2;
    const int64_t i3  = blockIdx.z / ne2;
    const int64_t i01 = blockIdx.x;
    const int64_t base_idx = i3*s03 + i2*s02 + i01*s01;

    const float4 * x4 = (const float4 *) x;
    block_q8_1_u4 * y = (block_q8_1_u4 *) vy;

    const int64_t k_block = i0 / 32; // 32-value block index in the channel
    const int64_t iqs     = i0 % 32; // value index in the block (0, 4, 8, ... 28)

    const float4 xi = i0 < ne00 ? x4[(base_idx + i0)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // min/max across the 8 threads covering one 32-value block
    // (same fold order-invariant result as the host's sequential scan).
    // Mask style matches quantize_mmq_q8_1 above: vendors/hip.h maps
    // __shfl_xor_sync to the raw __shfl_xor (HIP_DISABLE_WARP_SYNC_BUILTINS),
    // and width=32 partitions the wave64 shuffle into 32-lane groups.
    float mn = fminf(fminf(xi.x, xi.y), fminf(xi.z, xi.w));
    float mx = fmaxf(fmaxf(xi.x, xi.y), fmaxf(xi.z, xi.w));
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        mn = fminf(mn, __shfl_xor_sync(0xFFFFFFFF, mn, offset, WARP_SIZE));
        mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, offset, WARP_SIZE));
    }

    const float d4  = (mx - mn) / 15.0f;
    const float id  = d4 != 0.0f ? 1.0f / d4 : 0.0f;

    // q = clamp(rint((x - mn)/d4), 0, 15); host uses roundf on (x-mn)*id.
    int q0 = (int) roundf((xi.x - mn) * id);
    int q1 = (int) roundf((xi.y - mn) * id);
    int q2 = (int) roundf((xi.z - mn) * id);
    int q3 = (int) roundf((xi.w - mn) * id);
    q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
    q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
    q2 = q2 < 0 ? 0 : (q2 > 15 ? 15 : q2);
    q3 = q3 < 0 ? 0 : (q3 > 15 ? 15 : q3);

    // Store the SIGNED nibble (q - 8) & 0xF, packed 2/byte low-first
    // (byte k = {2k, 2k+1} little-endian — the dp8a operand order).
    // 4 values/thread = 4 nibbles = 2 bytes: one uint16 store per thread at
    // qs + iqs/2 (offsets 0,2,...,14 — 8 threads fill the 16-byte qs exactly).
    const uint32_t packed = ((uint32_t) ((q0 - 8) & 0xF))
                        | ((uint32_t) ((q1 - 8) & 0xF) <<  4)
                        | ((uint32_t) ((q2 - 8) & 0xF) <<  8)
                        | ((uint32_t) ((q3 - 8) & 0xF) << 12);

    // d4 and m as fp16 in the two halves of the 4-byte field right after qs[16]
    // (offsetof d4 == 16, m == 18 — see block_q8_1_u4, repack-common.cuh:192).
    const uint32_t dm = ((uint32_t) __half_as_ushort((ggml_half) d4))
                      | ((uint32_t) __half_as_ushort((ggml_half) mn) << 16);

    const int64_t ib = blockIdx.z*((int64_t) gridDim.x*gridDim.y*blockDim.x*4/32) + k_block*ne1 + blockIdx.x;

    // one 16-bit store for the nibble quad, one 32-bit store for the {d4, m} pair
    *(uint16_t *) (y[ib].qs + iqs/2) = (uint16_t) packed;
    if (iqs == 0) {
        *(uint32_t *) ((char *) y + ib*sizeof(block_q8_1_u4) + 16) = dm;
    }
}

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(!ids);
    GGML_ASSERT(ne0 % QK8_1 == 0);

    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);

#if defined(GGML_USE_HIP)
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc == GGML_CUDA_CC_VEGA20) {
        constexpr int block_size = CUDA_QUANTIZE_BLOCK_SIZE_GFX906;
        const int64_t block_num_x = (ne0 + block_size - 1) / block_size;
        const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
        quantize_q8_1<block_size><<<num_blocks, dim3(block_size, 1, 1), 0, stream>>>(x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
        return;
    }
#endif // defined(GGML_USE_HIP)

    constexpr int block_size = CUDA_QUANTIZE_BLOCK_SIZE;
    const int64_t block_num_x = (ne0 + block_size - 1) / block_size;
    const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, dim3(block_size, 1, 1), 0, stream);
    ggml_cuda_kernel_launch(quantize_q8_1<block_size>, launch_params, x, vy, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
    GGML_UNUSED(type_src0);
}

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % QK8_1_MMQ == 0);

    // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the CUDA grid:
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, false>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, /*n_expert_used=*/0);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, false>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, /*n_expert_used=*/0);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6, false>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, /*n_expert_used=*/0);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// W4A4 milestone 2a launcher: fp32 -> block_q8_1_u4 blocks. Mirrors
// quantize_mmq_q8_1_cuda's grid/block/stream structure (ne1 as grid.x so the
// widest dim maps to the grid's fastest-varying axis). NOT called from any
// dispatch yet — milestone 2b switches the repacked mat-vec path onto it under
// the GGML_CUDA_W4A4_DOT8 gate.
void quantize_mmq_q8_1_u4_cuda(
        const float * x, void * vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % 32 == 0);

    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    quantize_mmq_q8_1_u4<<<num_blocks, block_size, 0, stream>>>(
        x, vy, ne00, s01, s02, s03, ne0, (int) ne1, (int) ne2);
}

// scatter=true reuses the quant kernel: grid over tokens, ids = inverse map (token slot -> compact row)
void quantize_scatter_mmq_q8_1_cuda(
        const float * x, const int32_t * ids_src1_inv, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t stride_token, const int64_t ne0,
        const int64_t n_tokens, const int64_t nrows_dst, const int n_expert_used, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % QK8_1_MMQ == 0);

    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(n_tokens, block_num_y, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids_src1_inv, vy, ne00, /*s01=*/0, /*s02=*/stride_token, /*s03=*/0, ne0, /*ne1=*/(int) nrows_dst, /*ne2=*/1, n_expert_used);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids_src1_inv, vy, ne00, /*s01=*/0, /*s02=*/stride_token, /*s03=*/0, ne0, /*ne1=*/(int) nrows_dst, /*ne2=*/1, n_expert_used);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids_src1_inv, vy, ne00, /*s01=*/0, /*s02=*/stride_token, /*s03=*/0, ne0, /*ne1=*/(int) nrows_dst, /*ne2=*/1, n_expert_used);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// scatter=true reuses the quant kernels: grid over tokens, ids = inverse map (token slot -> compact row)
void quantize_scatter_mmq_fp4_cuda(
        const float * x, const int32_t * ids_src1_inv, void * vy, float * scale, const ggml_type type_src0, const bool use_aligned_float8,
        const int64_t ne00, const int64_t stride_token, const int64_t ne0,
        const int64_t n_tokens, const int64_t nrows_dst, const int n_expert_used, cudaStream_t stream) {
    GGML_ASSERT(ne0 > 0);
    if (type_src0 == GGML_TYPE_NVFP4) {
        GGML_ASSERT(scale);
        GGML_ASSERT(ne00 % QK_NVFP4 == 0);
        const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
        const dim3 num_blocks(n_tokens, 1, 1);
        if (use_aligned_float8) {
            quantize_mmq_nvfp4<true, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids_src1_inv, vy, scale, ne00, /*s01=*/0, /*s02=*/stride_token, /*s03=*/0, ne0, /*ne1=*/nrows_dst, /*ne2=*/1, n_expert_used);
        } else {
            quantize_mmq_nvfp4<true, false><<<num_blocks, block_size, 0, stream>>>(
                x, ids_src1_inv, vy, scale, ne00, /*s01=*/0, /*s02=*/stride_token, /*s03=*/0, ne0, /*ne1=*/nrows_dst, /*ne2=*/1, n_expert_used);
        }
    } else {
        GGML_ASSERT(type_src0 == GGML_TYPE_MXFP4);
        constexpr int nwarps = 8;
        constexpr int vals_per_block = nwarps * 2 * QK_MXFP4;
        const int64_t block_num_y = (ne0 + vals_per_block - 1) / vals_per_block;
        const dim3 block_size(WARP_SIZE, nwarps, 1);
        const dim3 num_blocks(n_tokens, block_num_y, 1);
        quantize_mmq_mxfp4<true><<<num_blocks, block_size, 0, stream>>>(
            x, ids_src1_inv, vy, ne00, /*s01=*/0, /*s02=*/stride_token, /*s03=*/0, ne0, /*ne1=*/(int) nrows_dst, /*ne2=*/1, n_expert_used);
    }
}

void quantize_mmq_fp4_cuda(
        const float * x, const int32_t * ids, void * vy, float * scale, const ggml_type type_src0, const bool use_aligned_float8,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(type_src0 == GGML_TYPE_MXFP4 || type_src0 == GGML_TYPE_NVFP4);
    GGML_ASSERT(ne0 > 0);

    if (type_src0 == GGML_TYPE_NVFP4) {
        GGML_ASSERT(scale);
        GGML_ASSERT(ne00 % QK_NVFP4 == 0);
        const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
        const dim3 num_blocks(ne1, ne2 * ne3, 1);
        if (use_aligned_float8) {
            quantize_mmq_nvfp4<false, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids, vy, scale, ne00, s01, s02, s03, ne0, ne1, ne2, /*n_expert_used=*/0);
        } else {
            quantize_mmq_nvfp4<false, false><<<num_blocks, block_size, 0, stream>>>(
                x, ids, vy, scale, ne00, s01, s02, s03, ne0, ne1, ne2, /*n_expert_used=*/0);
        }
    } else {
        GGML_ASSERT(ne0 % (2 * QK_MXFP4) == 0);

        constexpr int nwarps = 8;
        constexpr int vals_per_warp  = 2 * QK_MXFP4;
        constexpr int vals_per_block = nwarps * vals_per_warp;

        const int64_t block_num_y = (ne0 + vals_per_block - 1) / vals_per_block;
        const dim3    num_blocks(ne1, block_num_y, ne2 * ne3);
        const dim3    block_size(WARP_SIZE, nwarps, 1);

        quantize_mmq_mxfp4<false><<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, /*n_expert_used=*/0);
    }
}
