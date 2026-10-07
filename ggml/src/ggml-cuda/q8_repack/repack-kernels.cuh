// Device kernels and helpers for the Q8_0 repacked-weight path (AMD GCN only). MMV
// and GEMM are templated on HAS_IDS so dense and MoE host paths share identical device
// code. All __global__ kernels and __device__ functions use
// #if defined(GGML_USE_HIP) && defined(__gfx906__) bodies; elsewhere NO_DEVICE_CODE stubs.
#pragma once

#include "../common.cuh"
#include "../mmq.cuh"
#include "../unary.cuh"
#include "repack-common.cuh"

struct repack_tile_meta {
    uint32_t expert;
    uint32_t a_base;
};

template <int BN>
static __global__ void repack_tile_off(
        const int32_t * __restrict__ expert_bounds, int32_t * __restrict__ tile_off,
        repack_tile_meta * __restrict__ tile_meta, const int n_expert) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    int acc = 0;
    tile_off[0] = 0;
    for (int e = 0; e < n_expert; e++) {
        const int cnt = expert_bounds[e + 1] - expert_bounds[e];
        const int nt  = (cnt + BN - 1) / BN;
        for (int t = 0; t < nt; t++) {
            tile_meta[acc + t].expert = (uint32_t) e;
            tile_meta[acc + t].a_base = (uint32_t) expert_bounds[e] + (uint32_t) t * BN;
        }
        acc += nt;
        tile_off[e + 1] = acc;
    }
#else
    GGML_UNUSED_VARS(expert_bounds, tile_off, tile_meta, n_expert);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ int mmvq_dp4a_u4(
        const uint4 w, const int x0, const int x1, const int x2, const int x3) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    int a = ggml_cuda_dp4a((int) w.x, x0, 0);
    int b = ggml_cuda_dp4a((int) w.z, x2, 0);
    a = ggml_cuda_dp4a((int) w.y, x1, a);
    b = ggml_cuda_dp4a((int) w.w, x3, b);
    return a + b;
#else
    GGML_UNUSED_VARS(w, x0, x1, x2, x3);
    return 0;
#endif
}

// W4A4 milestone 2b: dp8a sibling of mmvq_dp4a_u4. ONE uint4 = 16 B = 32 nibbles
// = one 32-value block; each 32-bit word carries 8 nibbles (byte k = {2k, 2k+1}
// little-endian) and dot8 consumes 8 signed-nibble MACs per call: w.x vs a.x,
// w.y vs a.y, w.z vs a.z, w.w vs a.w. Both operands must be TWO'S-COMPLEMENT
// nibbles (weights XOR-8'd from the excess-8 canonical codes, activations stored
// signed by the milestone-2a quantizer). No 0x0F masks anywhere in this path.
static __device__ __forceinline__ int mmvq_dp8a_u4(const uint4 w, const uint4 a) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    int r = ggml_cuda_dp8a((int) w.x, (int) a.x, 0);
    r = ggml_cuda_dp8a((int) w.y, (int) a.y, r);
    r = ggml_cuda_dp8a((int) w.z, (int) a.z, r);
    r = ggml_cuda_dp8a((int) w.w, (int) a.w, r);
    return r;
#else
    GGML_UNUSED_VARS(w, a);
    return 0;
#endif
}

// Sum of the 32 raw excess-8 nibble codes in a uint4 (8 nibbles per 32-bit word).
// SUMW (two's-complement weight sum) = nib_sum_u4(pk) - 256.
static __device__ __forceinline__ uint32_t nib_sum_u4(const uint4 v) {
    uint32_t s = 0;
    const uint32_t ws[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t x = ws[i];
        s += (x & 0xFu) + ((x >> 4) & 0xFu) + ((x >> 8) & 0xFu) + ((x >> 12) & 0xFu)
           + ((x >> 16) & 0xFu) + ((x >> 20) & 0xFu) + ((x >> 24) & 0xFu) + ((x >> 28) & 0xFu);
    }
    return s;
}

// Sum of the 8 raw excess-8 nibble codes in ONE 32-bit word (per-8 variant for
// the u4p path: SW_k = nib_sum_word(raw_k) - 64, signed before any float cast).
static __device__ __forceinline__ uint32_t nib_sum_word(const uint32_t x) {
    return (x & 0xFu) + ((x >> 4) & 0xFu) + ((x >> 8) & 0xFu) + ((x >> 12) & 0xFu)
         + ((x >> 16) & 0xFu) + ((x >> 20) & 0xFu) + ((x >> 24) & 0xFu) + ((x >> 28) & 0xFu);
}

// C1/B8 milestone M2: per-8 dp8a helper. ONE weight uint4 (raw excess-8 Q4_0
// bytes, 32 values = 4 sub-blocks of 8) against ONE activation uint4
// (block_q8_1_u4p::qs, signed two's-complement nibbles). Produces the FOUR
// per-sub-block quantities the form-1 epilogue needs, in SEPARATE unchained
// accumulators (the algebra requires them per sub-block — no per-32 chaining):
//   idot[k] = dp8a(w_k ^ 0x88888888, a_k, 0)   (two's-complement dot, 8 MACs)
//   sw[k]   = signed nibble sum of w_tc word k
//           = nib_sum_word(raw_k) - 64           (8 codes, signed = code - 8)
// CPU seal (docs/w4a4-per8-m1.md §3, form 1): exact. The fp16-stored m-fold
// variant is NOT (29/1504 fail) — keep m and the +8*SW correction.
struct mmvq_dp8a_u4p_result {
    int      idot[4];  // per-sub-block two's-complement dot (unchained)
    float    sw[4];    // per-sub-block signed weight sum (SW_k)
};

static __device__ __forceinline__ mmvq_dp8a_u4p_result mmvq_dp8a_u4p(
        const uint4 w_raw, const uint4 a) {
    mmvq_dp8a_u4p_result r;
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    const uint32_t ws[4] = { w_raw.x, w_raw.y, w_raw.z, w_raw.w };
    const uint32_t as[4] = { a.x, a.y, a.z, a.w };
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        r.idot[k] = ggml_cuda_dp8a((int) (ws[k] ^ 0x88888888u), (int) as[k], 0);
        r.sw[k]   = (float) ((int) nib_sum_word(ws[k]) - 64);
    }
#else
    GGML_UNUSED_VARS(w_raw, a);
#endif
    return r;
}

// Fused-FFN epilogue shared by the dense and MoE MMV paths: up result + x_bias,
// gate result + gate_bias, then the GLU op.
static __device__ __forceinline__ float rp_mmv_fusion_epilogue(
        const float up, const float gate, const float * x_bias, const float * gate_bias,
        const uint32_t bias_idx, const ggml_glu_op glu_op, const bool use_gate) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    float result = up;
    if (x_bias != nullptr) {
        result += x_bias[bias_idx];
    }
    if (use_gate) {
        float gate_value = gate;
        if (gate_bias != nullptr) {
            gate_value += gate_bias[bias_idx];
        }
        switch (glu_op) {
            case GGML_GLU_OP_SWIGLU:     result *= ggml_cuda_op_silu_single(gate_value); break;
            case GGML_GLU_OP_GEGLU:      result *= ggml_cuda_op_gelu_single(gate_value); break;
            case GGML_GLU_OP_SWIGLU_OAI: result = ggml_cuda_op_swiglu_oai_single(gate_value, result); break;
            default:                     result *= gate_value; break;
        }
    }
    return result;
#else
    GGML_UNUSED_VARS(gate, x_bias, gate_bias, bias_idx, glu_op, use_gate);
    return up;
#endif
}

template <int ROWS, int NWAVES, bool HAS_IDS, int LANES = 64, bool HAS_FUSION = false>
static __global__ void mul_mat_vec_q8_0_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const uint32_t nchannels_y, const size_t expert_stride,
        const uint32_t xs_id, const uint32_t dst_s1,
        const uint8_t * __restrict__ wbase_gate, const float * __restrict__ x_bias,
        const float * __restrict__ gate_bias, const ggml_glu_op glu_op) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    constexpr int NWARPS = (NWAVES * 64) / LANES;
    constexpr int WPR = NWARPS / ROWS;
    static_assert((NWAVES * 64) % (ROWS * LANES) == 0, "NWAVES*64 must be a multiple of ROWS*LANES");
    static_assert(WPR >= 1 && (WPR & (WPR - 1)) == 0, "WPR must be a power of two");

    if constexpr (!HAS_FUSION) {
        GGML_UNUSED_VARS(wbase_gate, x_bias, gate_bias, glu_op);
    }

    [[maybe_unused]] const bool use_gate = wbase_gate != nullptr;

    [[maybe_unused]] uint32_t expert = 0;
    if constexpr (HAS_IDS) {
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        expert = e;
        wbase += e * expert_stride;
        if constexpr (HAS_FUSION) {
            // The gate matrix is per expert too - without this every assignment
            // reads expert 0's gate weights.
            if (use_gate) {
                wbase_gate += e * expert_stride;
            }
        }
        xq    += (size_t)(a % nchannels_y) * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, nchannels_y, expert_stride, xs_id, dst_s1);
    }
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t qs_str   = repack_qs_stride(ne0);

    const int warp_id   = threadIdx.x / LANES;
    const int lane      = threadIdx.x % LANES;
    const int row_local = warp_id / WPR;
    const int k_slice   = warp_id % WPR;
    const int row       = blockIdx.x * ROWS + row_local;

    const uint32_t wrow     = min((uint32_t) row, ne1 - 1);
    const uint4    * w_row  = reinterpret_cast<const uint4 *>(wbase + (size_t) wrow * qs_str);
    const uint16_t * d_row  = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * qs_str) + (size_t) wrow * n_blocks;
    const uint16_t rmask  = (row < (int) ne1) ? (uint16_t) 0xffffu : (uint16_t) 0x0000u;

    [[maybe_unused]] const uint4 * w_row_gate = nullptr;
    [[maybe_unused]] const uint16_t * d_row_gate = nullptr;
    if constexpr (HAS_FUSION) {
        if (use_gate) {
            w_row_gate = reinterpret_cast<const uint4 *>(wbase_gate + (size_t) wrow * qs_str);
            d_row_gate = reinterpret_cast<const uint16_t *>(
                wbase_gate + (size_t) ne1 * qs_str) + (size_t) wrow * n_blocks;
        }
    }

    [[maybe_unused]] uint32_t bias_row = 0;
    if constexpr (HAS_FUSION) {
        if constexpr (HAS_IDS) {
            bias_row = expert * ne1 + (uint32_t) row;
        } else {
            bias_row = (uint32_t) row;
        }
    }

    float acc = 0.0f;
    [[maybe_unused]] float acc_gate = 0.0f;

    if constexpr (LANES < 64) {
        const uint32_t n_half = n_blocks * 2;
        const uint32_t hb_lo  = (n_half * k_slice) / WPR;
        const uint32_t hb_hi  = (n_half * (k_slice + 1)) / WPR;

        const uint32_t hb0 = hb_lo + lane;
        const uint32_t half = hb0 & 1;
        const uint4 * wp = w_row + hb0;
        const block_q8_1 * xp = xq + (hb0 >> 1);
        const uint16_t * dp = d_row + (hb0 >> 1);

        [[maybe_unused]] const uint4 * wpg = nullptr;
        [[maybe_unused]] const uint16_t * dpg = nullptr;
        if constexpr (HAS_FUSION) {
            if (use_gate) {
                wpg = w_row_gate + hb0;
                dpg = d_row_gate + (hb0 >> 1);
            }
        }

        uint32_t iters = 0;
        for (uint32_t hb = hb0; hb + LANES < hb_hi; hb += 2 * LANES) {
            const float dx0 = __low2float(xp->ds);
            const int * xq32_0 = reinterpret_cast<const int *>(xp->qs) + half * 4;
            const int x00 = xq32_0[0], x01 = xq32_0[1], x02 = xq32_0[2], x03 = xq32_0[3];
            const uint4 wv0 = *wp;
            const uint16_t db0 = *dp & rmask;

            const float dx1 = __low2float((xp + LANES / 2)->ds);
            const int * xq32_1 = reinterpret_cast<const int *>((xp + LANES / 2)->qs) + half * 4;
            const int x10 = xq32_1[0], x11 = xq32_1[1], x12 = xq32_1[2], x13 = xq32_1[3];
            const uint4 wv1 = *(wp + LANES);
            const uint16_t db1 = *(dp + LANES / 2) & rmask;

            const float dw0 = __half2float(*reinterpret_cast<const __half *>(&db0));
            int idot0 = 0;
            idot0 = ggml_cuda_dp4a((int) wv0.x, x00, idot0);
            idot0 = ggml_cuda_dp4a((int) wv0.y, x01, idot0);
            idot0 = ggml_cuda_dp4a((int) wv0.z, x02, idot0);
            idot0 = ggml_cuda_dp4a((int) wv0.w, x03, idot0);
            acc += dw0 * dx0 * (float) idot0;

            const float dw1 = __half2float(*reinterpret_cast<const __half *>(&db1));
            int idot1 = 0;
            idot1 = ggml_cuda_dp4a((int) wv1.x, x10, idot1);
            idot1 = ggml_cuda_dp4a((int) wv1.y, x11, idot1);
            idot1 = ggml_cuda_dp4a((int) wv1.z, x12, idot1);
            idot1 = ggml_cuda_dp4a((int) wv1.w, x13, idot1);
            acc += dw1 * dx1 * (float) idot1;

            if constexpr (HAS_FUSION) {
                if (use_gate) {
                    const uint4 wgv0 = *wpg;
                    const uint16_t dgb0 = *dpg & rmask;
                    const float dwg0 = __half2float(*reinterpret_cast<const __half *>(&dgb0));
                    acc_gate += dwg0 * dx0 * (float) mmvq_dp4a_u4(wgv0, x00, x01, x02, x03);

                    const uint4 wgv1 = *(wpg + LANES);
                    const uint16_t dgb1 = *(dpg + LANES / 2) & rmask;
                    const float dwg1 = __half2float(*reinterpret_cast<const __half *>(&dgb1));
                    acc_gate += dwg1 * dx1 * (float) mmvq_dp4a_u4(wgv1, x10, x11, x12, x13);
                }
            }

            wp += 2 * LANES;
            xp += LANES;
            dp += LANES;
            if constexpr (HAS_FUSION) {
                if (use_gate) {
                    wpg += 2 * LANES;
                    dpg += LANES;
                }
            }
            iters += 2;
        }
        if (hb0 + iters * LANES < hb_hi) {
            const float dx = __low2float(xp->ds);
            const int * xq32 = reinterpret_cast<const int *>(xp->qs) + half * 4;
            const int x0 = xq32[0], x1 = xq32[1], x2 = xq32[2], x3 = xq32[3];
            const uint4 wv = *wp;
            const uint16_t db = *dp & rmask;
            const float dw = __half2float(*reinterpret_cast<const __half *>(&db));
            int idot = 0;
            idot = ggml_cuda_dp4a((int) wv.x, x0, idot);
            idot = ggml_cuda_dp4a((int) wv.y, x1, idot);
            idot = ggml_cuda_dp4a((int) wv.z, x2, idot);
            idot = ggml_cuda_dp4a((int) wv.w, x3, idot);
            acc += dw * dx * (float) idot;

            if constexpr (HAS_FUSION) {
                if (use_gate) {
                    const uint4 wgv = *wpg;
                    const uint16_t dgb = *dpg & rmask;
                    const float dwg = __half2float(*reinterpret_cast<const __half *>(&dgb));
                    acc_gate += dwg * dx * (float) mmvq_dp4a_u4(wgv, x0, x1, x2, x3);
                }
            }
        }
    } else {
        const uint32_t n_half = n_blocks * 2;
        const uint32_t hb_lo  = (n_half * k_slice) / WPR;
        const uint32_t hb_hi  = (n_half * (k_slice + 1)) / WPR;

        uint32_t base = hb_lo;
        for (; base + 2u * (uint32_t) LANES <= hb_hi; base += 2u * (uint32_t) LANES) {
            const uint32_t hb    = base + (uint32_t) lane;
            const uint32_t sb0   = hb >> 1;
            const uint32_t half0 = hb & 1;
            const uint32_t hb1   = hb + LANES;
            const uint32_t sb1   = hb1 >> 1;
            const uint32_t half1 = hb1 & 1;

            const block_q8_1 * xb0 = xq + sb0;
            const float dx0 = __low2float(xb0->ds);
            const int * xq32_0 = reinterpret_cast<const int *>(xb0->qs) + half0 * 4;
            const int x00 = xq32_0[0], x01 = xq32_0[1], x02 = xq32_0[2], x03 = xq32_0[3];
            const uint4 wv0 = w_row[hb];
            const uint16_t db0 = d_row[sb0] & rmask;

            const block_q8_1 * xb1 = xq + sb1;
            const float dx1 = __low2float(xb1->ds);
            const int * xq32_1 = reinterpret_cast<const int *>(xb1->qs) + half1 * 4;
            const int x10 = xq32_1[0], x11 = xq32_1[1], x12 = xq32_1[2], x13 = xq32_1[3];
            const uint4 wv1 = w_row[hb1];
            const uint16_t db1 = d_row[sb1] & rmask;

            const float dw0 = __half2float(*reinterpret_cast<const __half *>(&db0));
            const int idot0 = mmvq_dp4a_u4(wv0, x00, x01, x02, x03);
            acc += dw0 * dx0 * (float) idot0;

            const float dw1 = __half2float(*reinterpret_cast<const __half *>(&db1));
            const int idot1 = mmvq_dp4a_u4(wv1, x10, x11, x12, x13);
            acc += dw1 * dx1 * (float) idot1;

            if constexpr (HAS_FUSION) {
                if (use_gate) {
                    const uint4 wgv0 = w_row_gate[hb];
                    const uint16_t dgb0 = d_row_gate[sb0] & rmask;
                    const float dwg0 = __half2float(*reinterpret_cast<const __half *>(&dgb0));
                    acc_gate += dwg0 * dx0 * (float) mmvq_dp4a_u4(wgv0, x00, x01, x02, x03);

                    const uint4 wgv1 = w_row_gate[hb1];
                    const uint16_t dgb1 = d_row_gate[sb1] & rmask;
                    const float dwg1 = __half2float(*reinterpret_cast<const __half *>(&dgb1));
                    acc_gate += dwg1 * dx1 * (float) mmvq_dp4a_u4(wgv1, x10, x11, x12, x13);
                }
            }
        }

#pragma unroll
        for (int part = 0; part < 2; ++part) {
            const uint32_t hb = base + (uint32_t) (part * LANES + lane);
            if (hb < hb_hi) {
                const uint32_t sb   = hb >> 1;
                const uint32_t half = hb & 1;
                const block_q8_1 * xb = xq + sb;
                const float dx = __low2float(xb->ds);
                const int * xq32 = reinterpret_cast<const int *>(xb->qs) + half * 4;
                const int x0 = xq32[0], x1 = xq32[1], x2 = xq32[2], x3 = xq32[3];
                const uint4 wv = w_row[hb];
                const uint16_t db = d_row[sb] & rmask;
                const float dw = __half2float(*reinterpret_cast<const __half *>(&db));
                const int idot = mmvq_dp4a_u4(wv, x0, x1, x2, x3);
                acc += dw * dx * (float) idot;

                if constexpr (HAS_FUSION) {
                    if (use_gate) {
                        const uint4 wgv = w_row_gate[hb];
                        const uint16_t dgb = d_row_gate[sb] & rmask;
                        const float dwg = __half2float(*reinterpret_cast<const __half *>(&dgb));
                        acc_gate += dwg * dx * (float) mmvq_dp4a_u4(wgv, x0, x1, x2, x3);
                    }
                }
            }
        }
    }

    if constexpr (WPR > 1) {
        __shared__ float s_part[ROWS][WPR];
        [[maybe_unused]] __shared__ float s_part_gate[ROWS][WPR];
        const float wsum = rp_warp_reduce_sum<LANES>(acc);
        if (lane == 0) {
            s_part[row_local][k_slice] = wsum;
        }
        if constexpr (HAS_FUSION) {
            const float wsum_gate = rp_warp_reduce_sum<LANES>(acc_gate);
            if (lane == 0) {
                s_part_gate[row_local][k_slice] = wsum_gate;
            }
        }
        __syncthreads();
        if (k_slice == 0 && lane == 0) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < WPR; w++) {
                sum += s_part[row_local][w];
            }
            if (row < (int) ne1) {
                if constexpr (HAS_FUSION) {
                    float gate_sum = 0.0f;
#pragma unroll
                    for (int w = 0; w < WPR; w++) {
                        gate_sum += s_part_gate[row_local][w];
                    }
                    y[row] = rp_mmv_fusion_epilogue(sum, gate_sum, x_bias, gate_bias, bias_row, glu_op, use_gate);
                } else {
                    y[row] = sum;
                }
            }
        }
    } else {
        const float a = rp_warp_reduce_sum<LANES>(acc);
        if constexpr (HAS_FUSION) {
            const float g = rp_warp_reduce_sum<LANES>(acc_gate);
            if (lane == 0 && row < (int) ne1) {
                y[row] = rp_mmv_fusion_epilogue(a, g, x_bias, gate_bias, bias_row, glu_op, use_gate);
            }
        } else {
            if (lane == 0 && row < (int) ne1) {
                y[row] = a;
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, nchannels_y, expert_stride, xs_id, dst_s1, wbase_gate, x_bias, gate_bias, glu_op);
    NO_DEVICE_CODE;
#endif
}

// Generic repacked mat-vec: any type with 32-value sub-blocks, fetched through
// rp_traits<WT>::load_w. Simple LANES-strided sub-block loop: both production
// instantiations of the Q8_0 mat-vec run WPR == 1 (dense <16,16,false,64>, MoE
// <64,16,true,16>), so this kernel requires WPR == 1 and skips the k-slice
// machinery. Q8_0 keeps its tuned bespoke kernel; every other type gets this
// one for free, including the fused up/gate variant.
template <ggml_type WT, int ROWS, int NWAVES, bool HAS_IDS, int LANES = 64,
          bool HAS_FUSION = false, bool U4ACT = false>
static __global__ void __launch_bounds__(NWAVES * 64) mul_mat_vec_rp(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1,
        const uint32_t nchannels_y, const size_t expert_stride,
        const uint32_t xs_id, const uint32_t dst_s1,
        const uint8_t * __restrict__ wbase_gate = nullptr,
        const float * __restrict__ x_bias = nullptr,
        const float * __restrict__ gate_bias = nullptr,
        const ggml_glu_op glu_op = GGML_GLU_OP_REGLU) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    constexpr int NWARPS = (NWAVES * 64) / LANES;
    constexpr int WPR    = NWARPS / ROWS;
    static_assert(WPR == 1, "generic repack mat-vec assumes one lane-group per row");
    // W4A4 milestone 4: same guards as mul_mat_vec_repacked_nc_impl -- the dp8a
    // u4-activation variant is defined only for Q4_0 weights and only without
    // the fused up/gate epilogue (the gate accumulator was never u4-ized).
    static_assert(!U4ACT || WT == GGML_TYPE_Q4_0, "U4ACT (dp8a) is only defined for Q4_0 weights");
    static_assert(!U4ACT || !HAS_FUSION, "U4ACT (dp8a) does not support the fused-FFN variant");

    if constexpr (!HAS_FUSION) {
        GGML_UNUSED_VARS(wbase_gate, x_bias, gate_bias, glu_op);
    }
    [[maybe_unused]] const bool use_gate = HAS_FUSION && wbase_gate != nullptr;

    [[maybe_unused]] uint32_t expert = 0;
    if constexpr (HAS_IDS) {
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        expert = e;
        wbase += e * expert_stride;
        if constexpr (HAS_FUSION) {
            if (use_gate) {
                wbase_gate += e * expert_stride;
            }
        }
        if constexpr (!U4ACT) {
            xq    += (size_t)(a % nchannels_y) * xs_id;
        }
        y     += (size_t) a * dst_s1;
    } else {
        GGML_UNUSED_VARS(ids_src1, nchannels_y, expert_stride, xs_id, dst_s1);
    }
    const uint32_t n_sub = ne0 >> 5;
    const typename rp_traits<WT>::geom wg(ne0, ne1);

    const int warp_id = threadIdx.x / LANES;
    const int lane    = threadIdx.x % LANES;
    const int row     = blockIdx.x * ROWS + warp_id;

    // Clamped rows read a valid row's data and discard the result at the write.
    const uint32_t wrow = min((uint32_t) row, ne1 - 1);

    // W4A4 milestone 4: u4 column base in u4 units (the stock 32-byte-unit
    // offset above is skipped under U4ACT; see the HAS_IDS block).
    [[maybe_unused]] const block_q8_1_u4 * xq4 =
        reinterpret_cast<const block_q8_1_u4 *>(xq);
    if constexpr (U4ACT && HAS_IDS) {
        xq4 += (size_t)(blockIdx.y % nchannels_y) * xs_id;
    }
    float acc = 0.0f;
    [[maybe_unused]] float acc_gate = 0.0f;
    for (uint32_t sb = (uint32_t) lane; sb < n_sub; sb += LANES) {
        if constexpr (U4ACT) {
            // W4A4 milestone 4: dp8a u4-activation decode in the generic repack
            // mat-vec (MoE single-token). Identical algebra to the nc_impl
            // U4ACT branch (numeric seal: /tmp/dp8a_dot_test.cc): raw Q4_0
            // uint4 XOR 0x88888888 -> two's-complement weight nibbles;
            // activations are block_q8_1_u4 (qs[16] signed nibbles + fp16
            // d4/m, 20 B stride keeps 4-byte alignment); dot =
            // d * (d4 * (idot + 8*SUMW) + m * SUMW) with SUMW =
            // (int)nib_sum_u4(raw) - 256 computed SIGNED (unsigned wrap on
            // sums < 256 was the milestone-3b corruption).
            // Dead rows: this kernel clamps wrow and guards the write, so the
            // clamped-row terms are finite and discarded -- same as the stock
            // body here, which does not slot-mask either.
            uint4 pk;
            typename rp_traits<WT>::slot_t db;
            rp_traits<WT>::load_w_raw(wbase, wg, wrow, sb, pk, db);
            const uint4 w_tc = make_uint4(pk.x ^ 0x88888888u, pk.y ^ 0x88888888u,
                                        pk.z ^ 0x88888888u, pk.w ^ 0x88888888u);
            const float d  = rp_traits<WT>::scale(db);
            const float sw = (float) ((int) nib_sum_u4(pk) - 256);
            const block_q8_1_u4 * xb4 = xq4 + sb;
            const uint32_t * q32 = reinterpret_cast<const uint32_t *>(xb4->qs);
            const uint4 a = make_uint4(q32[0], q32[1], q32[2], q32[3]);
            const int idot = mmvq_dp8a_u4(w_tc, a);
            const float d4 = __half2float(xb4->d4);
            const float m  = __half2float(xb4->m);
            acc += d * (d4 * ((float) idot + 8.0f * sw) + m * sw);
            continue;
        }
        uint4 lo, hi;
        typename rp_traits<WT>::slot_t ds;
        rp_traits<WT>::load_w(wbase, wg, wrow, sb, lo, hi, ds);
        const block_q8_1 * xb = xq + sb;
        const float dx = __low2float(xb->ds);
        const int * x32 = reinterpret_cast<const int *>(xb->qs);
        if constexpr (rp_traits<WT>::split_scales) {
            const int ilo = mmvq_dp4a_u4(lo, x32[0], x32[1], x32[2], x32[3]);
            const int ihi = mmvq_dp4a_u4(hi, x32[4], x32[5], x32[6], x32[7]);
            const float2 dd = rp_traits<WT>::scale2(ds);
            acc += dd.x * dx * (float) ilo + dd.y * dx * (float) ihi;
        } else if constexpr (rp_traits<WT>::affine) {
            // Affine (Q4_K): d*sc times the dot, minus dmin*m times the stored activation sum (block_q8_1.ds.y).
            int idot = mmvq_dp4a_u4(lo, x32[0], x32[1], x32[2], x32[3]);
            idot    += mmvq_dp4a_u4(hi, x32[4], x32[5], x32[6], x32[7]);
            const float2 dd = rp_traits<WT>::scale2(ds);
            acc += dd.x * dx * (float) idot - dd.y * __high2float(xb->ds);
        } else {
            int idot = mmvq_dp4a_u4(lo, x32[0], x32[1], x32[2], x32[3]);
            idot    += mmvq_dp4a_u4(hi, x32[4], x32[5], x32[6], x32[7]);
            acc += rp_traits<WT>::scale(ds) * dx * (float) idot;
        }
        if constexpr (HAS_FUSION) {
            if (use_gate) {
                uint4 glo, ghi;
                typename rp_traits<WT>::slot_t gds;
                rp_traits<WT>::load_w(wbase_gate, wg, wrow, sb, glo, ghi, gds);
                if constexpr (rp_traits<WT>::split_scales) {
                    const int glo_d = mmvq_dp4a_u4(glo, x32[0], x32[1], x32[2], x32[3]);
                    const int ghi_d = mmvq_dp4a_u4(ghi, x32[4], x32[5], x32[6], x32[7]);
                    const float2 gd = rp_traits<WT>::scale2(gds);
                    acc_gate += gd.x * dx * (float) glo_d + gd.y * dx * (float) ghi_d;
                } else if constexpr (rp_traits<WT>::affine) {
                    int gdot = mmvq_dp4a_u4(glo, x32[0], x32[1], x32[2], x32[3]);
                    gdot    += mmvq_dp4a_u4(ghi, x32[4], x32[5], x32[6], x32[7]);
                    const float2 gd = rp_traits<WT>::scale2(gds);
                    acc_gate += gd.x * dx * (float) gdot - gd.y * __high2float(xb->ds);
                } else {
                    int gdot = mmvq_dp4a_u4(glo, x32[0], x32[1], x32[2], x32[3]);
                    gdot    += mmvq_dp4a_u4(ghi, x32[4], x32[5], x32[6], x32[7]);
                    acc_gate += rp_traits<WT>::scale(gds) * dx * (float) gdot;
                }
            }
        }
    }
    const float a = rp_warp_reduce_sum<LANES>(acc);
    [[maybe_unused]] float g = 0.0f;
    if constexpr (HAS_FUSION) {
        g = rp_warp_reduce_sum<LANES>(acc_gate);
    }
    if (lane == 0 && row < (int) ne1) {
        float out = a;
        if constexpr (HAS_FUSION) {
            uint32_t bias_row = (uint32_t) row;
            if constexpr (HAS_IDS) {
                bias_row = expert * ne1 + (uint32_t) row;
            }
            out = rp_mmv_fusion_epilogue(a, g, x_bias, gate_bias,
                bias_row, glu_op, use_gate);
        }
        y[row] = out;
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, nchannels_y, expert_stride,
                     xs_id, dst_s1, wbase_gate, x_bias, gate_bias, glu_op);
    NO_DEVICE_CODE;
#endif
}

// G1 (W4A4 GEMM): U4ACT selects the u4-activation (dp8a) variant of the dense
// GEMM. Skeleton only: template param + asserts + LDS/register plumbing with
// TODO(G2) markers; no dispatch passes U4ACT=true until G3. Plan:
// docs/w4a4-gemm-u4-plan.md (workspace w4a4-g1).
// C1/B8 M4: U4P8 selects the per-8 u4p activation variant of the dense GEMM
// (same pattern as the decode M2 U4P8 branch; per-8 moves the epilogue
// granularity only — the dot8 count per 32-value block is unchanged).
// INTEGRATION (perf/w4a4-integrated): U4X16 (per-16 activations) and KSPLIT
// (K-sliced accumulation) are orthogonal template variants — the K-slice bounds
// (sb_lo/sb_hi) wrap whichever activation read path is selected, and the u4x16
// read is token-major (xcol*n_sub + sb) so a K-slice reads its own contiguous
// slab of the pool. Both can be active at once (gate 3 + GGML_RP_KSPLIT).
template <ggml_type WT, bool HAS_IDS, int CW, int TN_, int NRL, bool U4ACT = false, bool U4P8 = false, bool U4X16 = false,
          int KSPLIT = 1>
static __device__ void mmq_gemm_repacked_impl(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const repack_tile_meta * __restrict__ tile_meta,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
    // G1 asserts: the u4/dp8a GEMM variant is defined only for Q4_0 weights
    // (raw excess-8 codes XOR-8'd to two's complement) and only for the dense
    // path. MoE (HAS_IDS) keeps the q8_1-activation GEMM until the decode-side
    // id-path is revisited; fused variants are nc-only anyway.
    static_assert(!U4ACT || WT == GGML_TYPE_Q4_0, "U4ACT (dp8a) is only defined for Q4_0 weights");
    static_assert(!U4ACT || !HAS_IDS, "U4ACT (dp8a) GEMM is dense-only in G1-G3 scope");
    // C1/B8 M4: the per-8 u4p GEMM carries the same guards as U4ACT and is
    // mutually exclusive with it — the two activation layouts cannot mix.
    static_assert(!U4P8 || WT == GGML_TYPE_Q4_0, "U4P8 (dp8a per-8) is only defined for Q4_0 weights");
    static_assert(!U4P8 || !HAS_IDS, "U4P8 (dp8a per-8) GEMM is dense-only");
    static_assert(!(U4ACT && U4P8), "U4ACT (per-32) and U4P8 (per-8) are mutually exclusive");
    // GGML_RP_KSPLIT: K-sliced GEMM variant. Dense only (the partial-buffer
    // epilogue has no ids scatter); KSPLIT == 1 is the stock single-pass form.
    static_assert(KSPLIT == 1 || !HAS_IDS, "K-split GEMM is dense-only");
    // Phase-2 gate 3: the per-16 u4x16 GEMM carries the same guards as U4P8
    // and is mutually exclusive with both other activation layouts.
    static_assert(!U4X16 || WT == GGML_TYPE_Q4_0, "U4X16 (dp8a per-16) is only defined for Q4_0 weights");
    static_assert(!U4X16 || !HAS_IDS, "U4X16 (dp8a per-16) GEMM is dense-only");
    static_assert(!(U4ACT && U4X16), "U4ACT (per-32) and U4X16 (per-16) are mutually exclusive");
    static_assert(!(U4P8 && U4X16), "U4P8 (per-8) and U4X16 (per-16) are mutually exclusive");
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    const int t  = threadIdx.x + threadIdx.y * blockDim.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const uint32_t row0 = blockIdx.x * MMQ_RP_Q8_BM;
    constexpr uint32_t BN = CW * TN_;
    uint32_t tok0 = blockIdx.y * BN;
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const repack_tile_meta meta = tile_meta[blockIdx.y];
        const uint32_t e = meta.expert;
        a_base = meta.a_base;
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, tile_meta, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    // GGML_RP_KSPLIT: grid.z slices the K range into KSPLIT contiguous slabs
    // of kb_slice 32-value blocks (a multiple of MMQ_RP_Q8_BK so a full slice
    // keeps the BK-aligned pipeline shape). KSPLIT == 1 folds to the stock
    // [0, n_sub) range at compile time -- zero codegen change when off.
    const uint32_t kb_slice = KSPLIT > 1
        ? ((n_sub + KSPLIT - 1) / KSPLIT + (MMQ_RP_Q8_BK - 1)) / MMQ_RP_Q8_BK * MMQ_RP_Q8_BK
        : n_sub;
    const uint32_t sb_lo = KSPLIT > 1 ? blockIdx.z * kb_slice : 0u;
    const uint32_t sb_hi = KSPLIT > 1
        ? (sb_lo + kb_slice < n_sub ? sb_lo + kb_slice : n_sub) : n_sub;
    // All layout knowledge lives in rp_traits<WT>: geometry constants built
    // once, sub-block fetches through load_w. The LDS scale array stays
    // the trait's slot_t - traits::scale() decodes the raw slot.
    const typename rp_traits<WT>::geom wg(ne0, ne1);
    const block_q8_1_mmq_h * __restrict__ xmmq = reinterpret_cast<const block_q8_1_mmq_h *>(xq);

    // The mmq group index is sb/BK; a slice starts mid-pool at sb_lo/BK groups.
    const block_q8_1_mmq_h * x_group = KSPLIT > 1
        ? xmmq + (size_t) (sb_lo / MMQ_RP_Q8_BK) * n_tok
        : xmmq;

    // Raw-register types (MXFP4) prefetch one uint4 of packed nibbles per
    // sub-block - half the live registers of the expanded pair, which spilled
    // 16-22 VGPRs per lane - and expand once per element at the LDS store.
    // LDS stays expanded: expanding at consume instead multiplied the expand
    // by the row*column reuse and measured -40% prefill.
    constexpr bool RAW_REG = rp_traits<WT>::raw_lds;
    __shared__ uint4    sW_lo[MMQ_RP_Q8_BM][MMQ_RP_Q8_BK];
    // G2: sW_hi (stock lo/hi nibble plane) and sWs (u4 sw sums) are mutually
    // exclusive -- the dot8 path never touches the hi plane, the stock path
    // never touches sWs -- so they share storage via a union. This keeps the
    // static LDS total at the stock 32768 B in BOTH states: the G1 sWs[1][1]
    // addition pushed the affine 64-wide GEMM to 32772 B and dropped
    // launch_bounds(256, 2) occupancy to 1 (clang -Wpass-failed on the
    // Q4_0/Q5_K/Q6_K U4ACT=false variants). Under U4ACT the union's sWs
    // member is live and sW_hi is never read or written -- no aliasing.
    // Named union + reference aliases: an *anonymous* block-scope union is an
    // unnamed variable and cannot be implicitly captured by the stage lambdas.
    union rp_w_planes {
        uint4 sW_hi[MMQ_RP_Q8_BM][MMQ_RP_Q8_BK];
        float sWs[U4ACT ? MMQ_RP_Q8_BK : 1][U4ACT ? MMQ_RP_Q8_BM : 1];
        // C1/B8 M4: per-8 weight sums — FOUR SW_k per (row, slab), SW_k =
        // nib_sum_word(raw_k) - 64 (signed, exact in fp32, range [-64, 64]).
        // 4*4 B * BK * BM = 4096 B = exactly the sW_hi footprint, so the union
        // total stays at the stock 32768 B (64-wide) / 20736 B (w32) and the
        // launch_bounds occupancy is unchanged. U4P8 never touches sW_hi.
        float sWs8[U4P8 ? MMQ_RP_Q8_BK : 1][U4P8 ? MMQ_RP_Q8_BM : 1][4];
        // Phase-2 gate 3: per-16 weight sums — TWO SW_k per (row, slab),
        // SW_k = nib_sum_word(raw_2k) + nib_sum_word(raw_2k+1) - 128 (signed,
        // exact in fp32, range [-128, 112]). 2*4 B * BK * BM = 2048 B, half the
        // sW_hi footprint, so the union total stays at the stock size.
        float sWs16[U4X16 ? MMQ_RP_Q8_BK : 1][U4X16 ? MMQ_RP_Q8_BM : 1][2];
    };
    __shared__ rp_w_planes rp_wplane;
    auto & sW_hi  = rp_wplane.sW_hi;
    auto & sWs    = rp_wplane.sWs;
    auto & sWs8   = rp_wplane.sWs8;
    auto & sWs16  = rp_wplane.sWs16;
    using slot_t = typename rp_traits<WT>::slot_t;
    static_assert(16 % sizeof(slot_t) == 0, "slot must divide a uint4 fetch");
    constexpr int SPW = (int) (16 / sizeof(slot_t));   // slots per uint4 LDS fetch
    __shared__ slot_t   sWdh[MMQ_RP_Q8_BK][MMQ_RP_Q8_BM];
    __shared__ sXq_row_q8 sXq[BN];
    // M4: under U4P8 the per-8 d4/m ride the activation q[lk][1] plane as
    // packed fp16, so the per-32 sXd tile is unused — shrink it to [1] to
    // hand the LDS back (keeps the 64-wide block under the 32 KiB budget
    // with room to spare).
    __shared__ float    sXd[(U4P8 || U4X16) ? 1 : BN][MMQ_RP_Q8_BK + 1];
    // Activation sums, affine types only (the [1] alternative keeps the LDS footprint of every other type unchanged).
    __shared__ float    sXs[(rp_traits<WT>::affine && !U4P8 && !U4X16) ? BN : 1][MMQ_RP_Q8_BK + 1];
    // G1 (W4A4 GEMM): per-weight-block two's-complement nibble sum
    // (sw = (int)nib_sum_u4(raw) - 256), computed at prefetch so compute_stage
    // stays dot-bound (see plan §2). Lives in the sW_hi/sWs union above (G2)
    // so the static LDS total stays at the stock 32768 B in both gate states.
    __shared__ uint32_t sCol[BN];

    constexpr int NROW = MMQ_RP_Q8_BM / NRL;
    float acc[NROW][TN_] = {};

    constexpr int NTHREADS = CW * NRL;
    constexpr int W_ELM = MMQ_RP_Q8_BM * MMQ_RP_Q8_BK;
    constexpr int X_ELM = BN * MMQ_RP_Q8_BK;
    constexpr int W_EPT = (W_ELM + NTHREADS - 1) / NTHREADS;
    constexpr int X_EPT = (X_ELM + NTHREADS - 1) / NTHREADS;

    const int sxc0 = sX_swizzle<CW>(tx);

    uint4    pw_lo[W_EPT];
    uint4    pw_hi[W_EPT];
    slot_t   pw_d [W_EPT];
    // G1 (W4A4 GEMM): two's-complement weight sum per prefetched weight block
    // (sw), filled by the U4ACT prefetch branch in G2. Unused when U4ACT is off.
    [[maybe_unused]] float pw_s[W_EPT];
    // C1/B8 M4: per-8 weight sums per prefetched block (SW_k =
    // nib_sum_word(raw_k) - 64, signed before the float cast — the 3b
    // root-cause class). Filled by the U4P8 prefetch branches; unused otherwise.
    [[maybe_unused]] float pw_s8[W_EPT][4];
    // STEP2 v3: packed form of pw_s8 — four SW_k (each in [-64,64], fits int8)
    // in one uint32 so the prefetch->store handoff carries 1 VGPR instead of 4.
    [[maybe_unused]] uint32_t pw_s8p[W_EPT];
    // Phase-2 gate 3: the two per-16 SW_k (each in [-128,112], fits int8) in
    // one uint32 (bytes 0,1), same packing trick as pw_s8p.
    [[maybe_unused]] uint32_t pw_s16p[W_EPT];
    rp_x_sub px   [X_EPT];
    [[maybe_unused]] float px_s[X_EPT];
    if constexpr (HAS_IDS) {
        for (int i = t; i < (int) BN; i += NTHREADS) {
            const uint32_t a = a_base + i;
            sCol[i] = (a < a_end) ? (uint32_t) ids_src1[a] : 0u;
        }
        __syncthreads();
    }
    // Whole BM x BN x BK tile in-bounds -> use the unchecked loads below.
    const bool full_m = row0 + MMQ_RP_Q8_BM <= ne1;
    bool full_n;
    if constexpr (HAS_IDS) {
        full_n = a_base + BN <= a_end;
    } else {
        full_n = tok0 + BN <= n_tok;
    }
    const uint32_t slice_len = sb_hi - sb_lo;
    const uint32_t n_main = slice_len & ~(MMQ_RP_Q8_BK - 1u);
    // slice_len > 0: a tail slice can be empty when kb_slice rounds up; the
    // checked path then stages nothing and the block writes a zero plane.
    const bool full_k = n_main == slice_len && slice_len > 0;
    const bool full_tile = full_m && full_n && full_k &&
                           (NTHREADS <= X_ELM) && (NTHREADS <= W_ELM);


    auto prefetch_w_checked = [&](const uint32_t sb0) {
#pragma unroll
        for (int i = 0; i < W_EPT; i++) {
            const int e = t + i * NTHREADS;
            if constexpr (W_EPT * NTHREADS != W_ELM) {
                if (e >= W_ELM) {
                    break;
                }
            }
            const int      lr   = e / MMQ_RP_Q8_BK;
            const int      lk   = e % MMQ_RP_Q8_BK;
            const uint32_t sb   = sb0 + lk;
            const uint32_t wrow = row0 + lr;
            if (wrow < ne1 && sb < sb_hi) {
                if constexpr (U4ACT) {
                    // G1 skeleton (W4A4 GEMM): raw canonical Q4_0 block,
                    // XOR-8'd to two's-complement nibbles, with the signed
                    // nibble-sum (sw) computed here at prefetch so compute_stage
                    // stays dot-bound. SIGNED cast before the float cast —
                    // milestone 3b bug 3 (unsigned wrap on sums < 256).
                    uint4 raw;
                    rp_traits<WT>::load_w_raw(wbase, wg, wrow, sb, raw, pw_d[i]);
                    pw_lo[i] = make_uint4(raw.x ^ 0x88888888u, raw.y ^ 0x88888888u,
                                         raw.z ^ 0x88888888u, raw.w ^ 0x88888888u);
                    pw_s[i]  = (float) ((int) nib_sum_u4(raw) - 256);
                } else if constexpr (U4P8) {
                    // C1/B8 M4: same raw XOR-8 staging as U4ACT, but the
                    // per-8 sums: SW_k = nib_sum_word(raw_k) - 64 per word
                    // (8 codes, signed before the float cast — the 3b class).
                    uint4 raw;
                    rp_traits<WT>::load_w_raw(wbase, wg, wrow, sb, raw, pw_d[i]);
                    pw_lo[i] = make_uint4(raw.x ^ 0x88888888u, raw.y ^ 0x88888888u,
                                         raw.z ^ 0x88888888u, raw.w ^ 0x88888888u);
                    const uint32_t rw[4] = { raw.x, raw.y, raw.z, raw.w };
                    pw_s8p[i] = 0u;
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        pw_s8p[i] |= (uint32_t) (((int) nib_sum_word(rw[k]) - 64) & 0xFF) << (8 * k);
                    }
                } else if constexpr (U4X16) {
                    // Phase-2 gate 3: same raw XOR-8 staging; per-16 sums
                    // SW_k = nib_sum_word(raw_2k) + nib_sum_word(raw_2k+1) - 128
                    // (two words per sub-block; signed before the float cast).
                    uint4 raw;
                    rp_traits<WT>::load_w_raw(wbase, wg, wrow, sb, raw, pw_d[i]);
                    pw_lo[i] = make_uint4(raw.x ^ 0x88888888u, raw.y ^ 0x88888888u,
                                         raw.z ^ 0x88888888u, raw.w ^ 0x88888888u);
                    const uint32_t rw[4] = { raw.x, raw.y, raw.z, raw.w };
                    pw_s16p[i] = 0u;
#pragma unroll
                    for (int k = 0; k < 2; ++k) {
                        const int swk = (int) nib_sum_word(rw[2*k]) + (int) nib_sum_word(rw[2*k+1]) - 128;
                        pw_s16p[i] |= (uint32_t) (swk & 0xFF) << (8 * k);
                    }
                } else if constexpr (RAW_REG) {
                    rp_traits<WT>::load_w_raw(wbase, wg, wrow, sb, pw_lo[i], pw_d[i]);
                } else {
                    rp_traits<WT>::load_w(wbase, wg, wrow, sb, pw_lo[i], pw_hi[i], pw_d[i]);
                }
            } else {
                // Zero the payload too, not just the scale. Q8_0 tolerates garbage
                // here because d==0 exactly, but MXFP4's e==0 decodes to 2^-127,
                // not 0, so uninitialised registers would survive into the sum.
                pw_lo[i] = make_uint4(0, 0, 0, 0);
                pw_hi[i] = make_uint4(0, 0, 0, 0);
                pw_d [i] = {};
                if constexpr (U4ACT) {
                    pw_s[i] = 0.0f;
                }
                if constexpr (U4P8) {
                    pw_s8p[i] = 0u;
                }
                if constexpr (U4X16) {
                    pw_s16p[i] = 0u;
                }
            }
        }
    };

    auto prefetch_x_checked = [&](const uint32_t sb0, const block_q8_1_mmq_h * x_grp) {
        if constexpr (U4ACT) {
            // G2 (W4A4 GEMM): u4 activation fetch straight from the token-major
            // block_q8_1_u4 pool: xq_u4[xcol * nkb + sb], nkb = ne0/32. The
            // block_q8_1_mmq 4-block grouping (x_group advance) is bypassed
            // entirely -- the u4 quantizer emits one block per (token, 32
            // values), so the slab index rides the k index directly. HAS_IDS is
            // asserted false under U4ACT, so xcol is always the plain token col.
            GGML_UNUSED(x_grp);
            const block_q8_1_u4 * __restrict__ xq_u4 =
                reinterpret_cast<const block_q8_1_u4 *>(xq);
#pragma unroll
            for (int i = 0; i < X_EPT; i++) {
                const int e = t + i * NTHREADS;
                if constexpr (X_EPT * NTHREADS != X_ELM) {
                    if (e >= X_ELM) {
                        break;
                    }
                }
                const int      lr = e / MMQ_RP_Q8_BK;
                const int      lk = e % MMQ_RP_Q8_BK;
                const uint32_t sb = sb0 + lk;
                const uint32_t xcol = tok0 + lr;
                if (xcol < n_tok && sb < sb_hi) {
                    px[i] = rp_x_sub_u4_from_blocks(xq_u4, xcol * n_sub + sb, px_s[i]);
                } else {
                    // Out-of-range token/slab: zero d4 AND m so the sealed
                    // epilogue d*(d4*(idot+8*sw) + m*sw) collapses to 0
                    // regardless of the weight term (same trick as the stock
                    // affine dead-column zeroing).
                    px[i] = {};
                    px_s[i] = 0.0f;
                }
            }
            return;
        }
        if constexpr (U4P8) {
            // C1/B8 M4: u4p activation fetch from the token-major
            // block_q8_1_u4p pool: xq_u4p[xcol * n_sub + sb] (nkb = ne0/32 —
            // the u4p quantizer emits one 32 B block per (token, 32 values),
            // same token-major ib as the per-32 u4 quantizer). The 32 B stride
            // keeps every block 16 B aligned: qs AND the d4[4]/m[4] quad both
            // load as full uint4s (no 4xu32 workaround like the 20 B u4).
            // q1 carries the packed fp16 scales; the per-32 sXd/sXs tiles are
            // not used. HAS_IDS is asserted false, so xcol is the plain token col.
            GGML_UNUSED(x_grp);
            const block_q8_1_u4p * __restrict__ xq_u4p =
                reinterpret_cast<const block_q8_1_u4p *>(xq);
#pragma unroll
            for (int i = 0; i < X_EPT; i++) {
                const int e = t + i * NTHREADS;
                if constexpr (X_EPT * NTHREADS != X_ELM) {
                    if (e >= X_ELM) {
                        break;
                    }
                }
                const int      lr = e / MMQ_RP_Q8_BK;
                const int      lk = e % MMQ_RP_Q8_BK;
                const uint32_t sb = sb0 + lk;
                const uint32_t xcol = tok0 + lr;
                if (xcol < n_tok && sb < sb_hi) {
                    px[i] = rp_x_sub_u4p_from_blocks(xq_u4p, xcol * n_sub + sb);
                } else {
                    // Out-of-range token/slab: zero the nibbles AND the packed
                    // d4/m quad so the sealed per-8 epilogue collapses to 0
                    // regardless of the weight term (dead-column trick).
                    px[i] = {};
                }
            }
            return;
        }
        if constexpr (U4X16) {
            // Phase-2 gate 3: u4x16 activation fetch from the token-major
            // block_q8_1_u4x16 pool: xq_u4x16[xcol * n_sub + sb] (nkb = ne0/32,
            // same token-major ib as the per-32/per-8 quantizers). The 28 B
            // stride keeps every block 4 B aligned but NOT 16 B aligned, so qs
            // and the d4[2]/m[2] pair are assembled as u32 words inside the
            // fetch helper (the 20 B per-32 u4 precedent). HAS_IDS is asserted
            // false, so xcol is the plain token col.
            GGML_UNUSED(x_grp);
            const block_q8_1_u4x16 * __restrict__ xq_u4x16 =
                reinterpret_cast<const block_q8_1_u4x16 *>(xq);
#pragma unroll
            for (int i = 0; i < X_EPT; i++) {
                const int e = t + i * NTHREADS;
                if constexpr (X_EPT * NTHREADS != X_ELM) {
                    if (e >= X_ELM) {
                        break;
                    }
                }
                const int      lr = e / MMQ_RP_Q8_BK;
                const int      lk = e % MMQ_RP_Q8_BK;
                const uint32_t sb = sb0 + lk;
                const uint32_t xcol = tok0 + lr;
                if (xcol < n_tok && sb < sb_hi) {
                    // INTEGRATION: bound is sb_hi, not n_sub — identical when
                    // KSPLIT == 1 (sb_hi == n_sub), but under K-split the tail
                    // prefetch window extends past the slice end and those
                    // slabs MUST be zeroed or they double-count across z-planes.
                    px[i] = rp_x_sub_u4x16_from_blocks(xq_u4x16, xcol * n_sub + sb);
                } else {
                    // Dead token/slab: zero nibbles AND scales (dead-column trick).
                    px[i] = {};
                }
            }
            return;
        }
#pragma unroll
        for (int i = 0; i < X_EPT; i++) {
            const int e = t + i * NTHREADS;
            if constexpr (X_EPT * NTHREADS != X_ELM) {
                if (e >= X_ELM) {
                    break;
                }
            }
            const int      lr = e / MMQ_RP_Q8_BK;
            const int      lk = e % MMQ_RP_Q8_BK;
            const uint32_t sb = sb0 + lk;
            uint32_t xcol = tok0 + lr;
            bool     xval = xcol < n_tok;
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + lr;
                xval = a < a_end;
                xcol = xval ? sCol[lr] : 0;
            }
            if (xval && sb < sb_hi) {
                if constexpr (rp_traits<WT>::affine) {
                    px[i] = rp_x_sub_from_mmq_group_ds(x_grp, xcol, lk, px_s[i]);
                } else {
                    px[i] = rp_x_sub_from_mmq_group(x_grp, xcol, lk);
                }
            } else {
                px[i].d = 0.0f;
                if constexpr (rp_traits<WT>::affine) {
                    // d == 0 does not kill the affine sum term, zero it too
                    px_s[i] = 0.0f;
                }
            }
        }
    };

    auto prefetch_w_full = [&](const uint32_t sb0) {
#pragma unroll
        for (int i = 0; i < W_EPT; i++) {
            const int e = t + i * NTHREADS;
            if constexpr (W_EPT * NTHREADS != W_ELM) {
                if (e >= W_ELM) break;
            }
            const int lr = e / MMQ_RP_Q8_BK;
            const int lk = e % MMQ_RP_Q8_BK;

            const uint32_t wrow = row0 + lr;
            const uint32_t sb   = sb0 + lk;
            if constexpr (U4ACT) {
                // G2: same raw XOR-8 + signed sw as prefetch_w_checked (the G1
                // skeleton only covered the checked path; the full-tile path
                // needs it too or it would stage the expanded lo/hi pair).
                uint4 raw;
                rp_traits<WT>::load_w_raw(wbase, wg, wrow, sb, raw, pw_d[i]);
                pw_lo[i] = make_uint4(raw.x ^ 0x88888888u, raw.y ^ 0x88888888u,
                                     raw.z ^ 0x88888888u, raw.w ^ 0x88888888u);
                pw_s[i]  = (float) ((int) nib_sum_u4(raw) - 256);
            } else if constexpr (U4P8) {
                // C1/B8 M4: same raw XOR-8 + per-8 SW_k as the checked path.
                uint4 raw;
                rp_traits<WT>::load_w_raw(wbase, wg, wrow, sb, raw, pw_d[i]);
                pw_lo[i] = make_uint4(raw.x ^ 0x88888888u, raw.y ^ 0x88888888u,
                                     raw.z ^ 0x88888888u, raw.w ^ 0x88888888u);
                const uint32_t rw[4] = { raw.x, raw.y, raw.z, raw.w };
                pw_s8p[i] = 0u;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    pw_s8p[i] |= (uint32_t) (((int) nib_sum_word(rw[k]) - 64) & 0xFF) << (8 * k);
                }
            } else if constexpr (U4X16) {
                // Phase-2 gate 3: same raw XOR-8 + per-16 SW_k as the checked path.
                uint4 raw;
                rp_traits<WT>::load_w_raw(wbase, wg, wrow, sb, raw, pw_d[i]);
                pw_lo[i] = make_uint4(raw.x ^ 0x88888888u, raw.y ^ 0x88888888u,
                                     raw.z ^ 0x88888888u, raw.w ^ 0x88888888u);
                const uint32_t rw[4] = { raw.x, raw.y, raw.z, raw.w };
                pw_s16p[i] = 0u;
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const int swk = (int) nib_sum_word(rw[2*k]) + (int) nib_sum_word(rw[2*k+1]) - 128;
                    pw_s16p[i] |= (uint32_t) (swk & 0xFF) << (8 * k);
                }
            } else if constexpr (RAW_REG) {
                rp_traits<WT>::load_w_raw(wbase, wg, wrow, sb, pw_lo[i], pw_d[i]);
            } else {
                rp_traits<WT>::load_w(wbase, wg, wrow, sb, pw_lo[i], pw_hi[i], pw_d[i]);
            }
        }
    };

    auto prefetch_x_full = [&](const uint32_t sb0, const block_q8_1_mmq_h * x_grp) {
        if constexpr (U4ACT) {
            // G2 (W4A4 GEMM): same u4 fetch as prefetch_x_checked, unchecked:
            // full_tile guarantees every (xcol, sb) in the tile is in bounds.
            GGML_UNUSED(x_grp);
            const block_q8_1_u4 * __restrict__ xq_u4 =
                reinterpret_cast<const block_q8_1_u4 *>(xq);
#pragma unroll
            for (int i = 0; i < X_EPT; i++) {
                const int e = t + i * NTHREADS;
                if constexpr (X_EPT * NTHREADS != X_ELM) {
                    if (e >= X_ELM) break;
                }
                const int      lr = e / MMQ_RP_Q8_BK;
                const int      lk = e % MMQ_RP_Q8_BK;
                const uint32_t sb = sb0 + lk;
                px[i] = rp_x_sub_u4_from_blocks(xq_u4, (tok0 + lr) * n_sub + sb, px_s[i]);
            }
            return;
        }
        if constexpr (U4P8) {
            // C1/B8 M4: same u4p fetch as prefetch_x_checked, unchecked.
            GGML_UNUSED(x_grp);
            const block_q8_1_u4p * __restrict__ xq_u4p =
                reinterpret_cast<const block_q8_1_u4p *>(xq);
#pragma unroll
            for (int i = 0; i < X_EPT; i++) {
                const int e = t + i * NTHREADS;
                if constexpr (X_EPT * NTHREADS != X_ELM) {
                    if (e >= X_ELM) break;
                }
                const int      lr = e / MMQ_RP_Q8_BK;
                const int      lk = e % MMQ_RP_Q8_BK;
                const uint32_t sb = sb0 + lk;
                px[i] = rp_x_sub_u4p_from_blocks(xq_u4p, (tok0 + lr) * n_sub + sb);
            }
            return;
        }
        if constexpr (U4X16) {
            // Phase-2 gate 3: same u4x16 fetch as prefetch_x_checked, unchecked.
            GGML_UNUSED(x_grp);
            const block_q8_1_u4x16 * __restrict__ xq_u4x16 =
                reinterpret_cast<const block_q8_1_u4x16 *>(xq);
#pragma unroll
            for (int i = 0; i < X_EPT; i++) {
                const int e = t + i * NTHREADS;
                if constexpr (X_EPT * NTHREADS != X_ELM) {
                    if (e >= X_ELM) break;
                }
                const int      lr = e / MMQ_RP_Q8_BK;
                const int      lk = e % MMQ_RP_Q8_BK;
                const uint32_t sb = sb0 + lk;
                px[i] = rp_x_sub_u4x16_from_blocks(xq_u4x16, (tok0 + lr) * n_sub + sb);
            }
            return;
        }
#pragma unroll
        for (int i = 0; i < X_EPT; i++) {
            const int e = t + i * NTHREADS;
            if constexpr (X_EPT * NTHREADS != X_ELM) {
                if (e >= X_ELM) break;
            }
            const int lr = e / MMQ_RP_Q8_BK;
            const int lk = e % MMQ_RP_Q8_BK;

            uint32_t xcol = tok0 + lr;
            if constexpr (HAS_IDS) {
                xcol = sCol[lr];
            }

            if constexpr (rp_traits<WT>::affine) {
                px[i] = rp_x_sub_from_mmq_group_ds(x_grp, xcol, lk, px_s[i]);
            } else {
                px[i] = rp_x_sub_from_mmq_group(x_grp, xcol, lk);
            }
        }
    };

    auto store_stage = [&]() {
#pragma unroll
        for (int i = 0; i < W_EPT; i++) {
            const int e = t + i * NTHREADS;
            if constexpr (W_EPT * NTHREADS != W_ELM) {
                if (e >= W_ELM) {
                    break;
                }
            }
            const int lr = e / MMQ_RP_Q8_BK;
            const int lk = e % MMQ_RP_Q8_BK;
            if constexpr (U4ACT) {
                // G1 (W4A4 GEMM): weight side is the raw block XOR-8'd at
                // prefetch (two's-complement nibbles) — one uint4, no lo/hi
                // split, sW_hi stays unused. sw rides sWs for the epilogue.
                sW_lo[lr][lk] = pw_lo[i];
                sWdh [lk][lr] = pw_d[i];
                sWs  [lk][lr] = pw_s[i];
            } else if constexpr (U4P8) {
                // C1/B8 M4: weight side identical to U4ACT (raw XOR-8 uint4);
                // the per-8 SW_k ride sWs8 (union with sW_hi).
                sW_lo[lr][lk] = pw_lo[i];
                sWdh [lk][lr] = pw_d[i];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    sWs8[lk][lr][k] = (float) (int8_t) ((pw_s8p[i] >> (8 * k)) & 0xFF);
                }
            } else if constexpr (U4X16) {
                // Phase-2 gate 3: weight side identical to U4ACT/U4P8; the
                // per-16 SW_k ride sWs16 (union with sW_hi), int8 unpack.
                sW_lo[lr][lk] = pw_lo[i];
                sWdh [lk][lr] = pw_d[i];
#pragma unroll
                for (int k = 0; k < 2; ++k) {
                    sWs16[lk][lr][k] = (float) (int8_t) ((pw_s16p[i] >> (8 * k)) & 0xFF);
                }
            } else if constexpr (RAW_REG) {
                uint4 lo, hi;
                rp_traits<WT>::expand_raw(pw_lo[i], lo, hi);
                sW_lo[lr][lk] = lo;
                sW_hi[lr][lk] = hi;
            } else {
                sW_lo[lr][lk] = pw_lo[i];
                sW_hi[lr][lk] = pw_hi[i];
            }
            sWdh [lk][lr] = pw_d[i];
        }
#pragma unroll
        for (int i = 0; i < X_EPT; i++) {
            const int e = t + i * NTHREADS;
            if constexpr (X_EPT * NTHREADS != X_ELM) {
                if (e >= X_ELM) {
                    break;
                }
            }
            const int lr  = e / MMQ_RP_Q8_BK;
            const int lk  = e % MMQ_RP_Q8_BK;
            const int sXr = sX_swizzle<CW>(lr);

            // G1 (W4A4 GEMM): under U4ACT the activation payload is one uint4
            // (the raw u4 qs) in q[lk][0]; q[lk][1] is unused (zeroed by the
            // G2 fetch), sXd carries d4 and sXs carries m.
            if constexpr (U4P8 || U4X16) {
                // C1/B8 M4: q[lk][0] = the 32 signed nibbles; q[lk][1] = the
                // packed fp16 d4[4]/m[4] quad (rp_x_sub_u4p_from_blocks).
                // Phase-2 gate 3: same staging for U4X16 — q1 carries the
                // fp16 d4[2]/m[2] pair in .x/.y. The per-32 sXd/sXs tiles are
                // shrunk to [1] under both and are NOT written here (the
                // sub-block scales ride the q plane).
                sXq[sXr].q[lk][0] = px[i].q0;
                sXq[sXr].q[lk][1] = px[i].q1;
            } else {
                sXq[sXr].q[lk][0] = px[i].q0;
                sXq[sXr].q[lk][1] = px[i].q1;
                sXd[sXr][lk]    = px[i].d;
                if constexpr (rp_traits<WT>::affine) {
                    sXs[sXr][lk] = px_s[i];
                }
            }
        }
    };

    auto compute_stage = [&]() {
        if constexpr (U4ACT) {
            // G2 (W4A4 GEMM): the sealed form-B dot loop, GEMM twin of the
            // decode U4ACT branch (worked example: mul_mat_vec_repacked_nc_impl
            // at 88bbc3572; numeric seal /tmp/dp8a_dot_test.cc). Per 32-value
            // sub-block kk, per weight row r, per token col n:
            //     acc[r][n] += d * ( d4 * ((float) idot + 8.0f*sw) + mu * sw )
            //     idot = mmvq_dp8a_u4(w_tc, a_tc)
            // w_tc = sW_lo (raw Q4_0 uint4 XOR 0x88888888 staged at prefetch:
            // two's-complement nibbles), sw = sWs (SIGNED (int)nib_sum-256,
            // milestone 3b bug 3), a_tc = sXq q[kk][0] (the quantizer stores
            // the activation nibbles two's-complement already -- no XOR here),
            // d4 = sXd, mu = sXs (the affine slot now carries m, not the DS4
            // sum), d = Q4_0 f16 scale from the sWdh slot (zeroed for dead
            // rows at prefetch -> zeroes the whole term).
            // 4 dot8 per block per (r,n) vs the stock 8 dp4a: the win. No whi,
            // no scale2, no lo/hi split -- the whole branch replaces the stock
            // body so the q8_1 pipelining below stays byte-identical.
            for (int kk = 0; kk < MMQ_RP_Q8_BK; kk++) {
                const int row_base = ty * NROW;
                uint4 w = sW_lo[row_base][kk];
                const uint4 * dp16 = reinterpret_cast<const uint4 *>(&sWdh[kk][row_base]);
#pragma unroll
                for (int g = 0; g < NROW; g += SPW) {
                    const uint4 dh4 = dp16[g / SPW];
                    const slot_t * dh = reinterpret_cast<const slot_t *>(&dh4);
#pragma unroll
                    for (int rr = 0; rr < SPW; rr++) {
                        const int r = g + rr;
                        const uint4 w_c = w;
                        if (r + 1 < NROW) {
                            w = sW_lo[row_base + r + 1][kk];
                        }
                        const float d  = rp_traits<WT>::scale(dh[rr]);
                        const float sw = sWs[kk][row_base + r];
#pragma unroll
                        for (int n = 0; n < TN_; n++) {
                            // Activation operands read straight from LDS inside
                            // the n loop (no hoisted a[]/d4[]/mu[] arrays):
                            // keeps live VGPRs down -- the hoisted form spilled
                            // 178 at launch_bounds(256, 2) vs stock's 20.
                            const int xcol = sxc0 + n * CW;
                            const uint4 a  = sXq[xcol].q[kk][0];
                            const float d4 = sXd[xcol][kk];
                            const float mu = sXs[xcol][kk];
                            const int idot = mmvq_dp8a_u4(w_c, a);
                            acc[r][n] += d * (d4 * ((float) idot + 8.0f * sw) + mu * sw);
                        }
                    }
                }
            }
            return;
        }
        if constexpr (U4P8) {
            // C1/B8 M4: the sealed per-8 form-1 dot loop, GEMM twin of the
            // decode U4P8 branch (mul_mat_vec_repacked_nc_impl, M2) and of
            // the U4ACT loop above. Per 32-value sub-block kk, per weight row
            // r, per token col n, summed over the four per-8 sub-blocks k:
            //     acc[r][n] += d * ( d4_k * ((float) idot_k + 8.0f*SW_k) + m_k * SW_k )
            // with idot_k = dp8a(w_word_k, a_word_k) UNCHAINED per sub-block
            // (the algebra needs them separate — no per-32 chaining),
            // w_word = sW_lo (raw Q4_0 XOR-8'd two's-complement nibbles),
            // SW_k = sWs8 (SIGNED nib_sum_word-64, the 3b class),
            // a_word = sXq q[kk][0] words (quantizer stores signed nibbles),
            // d4_k/m_k = sXq q[kk][1] packed fp16 quad {d4_0,d4_1},{d4_2,
            // d4_3},{m_0,m_1},{m_2,m_3} (rp_x_sub_u4p_from_blocks),
            // d = Q4_0 f16 scale from sWdh (zeroed for dead rows at prefetch).
            // Dot count per block per (r,n) = 4 dot8 — SAME as the per-32
            // U4ACT GEMM (per-8 moves the epilogue granularity only).
            // Activation operands read straight from LDS inside the n loop
            // (the U4ACT spill lesson: hoisting spilled 178 VGPRs).
            // M4: forced unroll of the slab loop so the w32 variant emits the
            // full 128 static dot8s like the per-32 U4ACT w32 (the rolled
            // form emitted 32 static x 4 dynamic; same dynamic total, the
            // unroll restores the per-32 static shape the brief asks to
            // verify — spill-checked, see kernel-notes M4).
            for (int kk = 0; kk < MMQ_RP_Q8_BK; kk++) {
                const int row_base = ty * NROW;
                uint4 w = sW_lo[row_base][kk];
                const uint4 * dp16 = reinterpret_cast<const uint4 *>(&sWdh[kk][row_base]);
#pragma unroll
                for (int g = 0; g < NROW; g += SPW) {
                    const uint4 dh4 = dp16[g / SPW];
                    const slot_t * dh = reinterpret_cast<const slot_t *>(&dh4);
#pragma unroll
                    for (int rr = 0; rr < SPW; rr++) {
                        const int r = g + rr;
                        const uint4 w_c = w;
                        if (r + 1 < NROW) {
                            w = sW_lo[row_base + r + 1][kk];
                        }
                        const float d  = rp_traits<WT>::scale(dh[rr]);
                        // Four SW_k for this (slab, row): 16 B aligned in the
                        // union (float[4] rows), load as one float4.
                        const float4 sw4 = *reinterpret_cast<const float4 *>(
                            &sWs8[kk][row_base + r][0]);
#pragma unroll
                        for (int n = 0; n < TN_; n++) {
                            const int xcol = sxc0 + n * CW;
                            const uint4 a = sXq[xcol].q[kk][0];
                            const uint4 s = sXq[xcol].q[kk][1];
                            // k-pair fold (M4 anti-stuck): consume k=0,1 into
                            // acc, then k=2,3 — halves the live epilogue
                            // temporaries (4 idot + 8 scale floats -> 2 + 4)
                            // to claw back VGPRs at launch_bounds(256, 2).
                            const float2 p0 = __half22float2(*reinterpret_cast<const half2 *>(&s.x));
                            const float2 q0 = __half22float2(*reinterpret_cast<const half2 *>(&s.z));
                            const int i0 = ggml_cuda_dp8a((int) w_c.x, (int) a.x, 0);
                            const int i1 = ggml_cuda_dp8a((int) w_c.y, (int) a.y, 0);
                            acc[r][n] += d * (p0.x * ((float) i0 + 8.0f * sw4.x) + q0.x * sw4.x);
                            acc[r][n] += d * (p0.y * ((float) i1 + 8.0f * sw4.y) + q0.y * sw4.y);
                            const float2 p1 = __half22float2(*reinterpret_cast<const half2 *>(&s.y));
                            const float2 q1 = __half22float2(*reinterpret_cast<const half2 *>(&s.w));
                            const int i2 = ggml_cuda_dp8a((int) w_c.z, (int) a.z, 0);
                            const int i3 = ggml_cuda_dp8a((int) w_c.w, (int) a.w, 0);
                            acc[r][n] += d * (p1.x * ((float) i2 + 8.0f * sw4.z) + q1.x * sw4.z);
                            acc[r][n] += d * (p1.y * ((float) i3 + 8.0f * sw4.w) + q1.y * sw4.w);
                        }
                    }
                }
            }
            return;
        }
        if constexpr (U4X16) {
            // Phase-2 gate 3: the per-16 form-B dot loop, GEMM twin of the
            // U4P8 loop above with TWO sub-blocks of TWO dp8a words each. Per
            // slab kk, per weight row r, per token col n, summed over the two
            // per-16 sub-blocks k:
            //     acc[r][n] += d * ( d4_k * ((float) idot_k + 8.0f*SW_k) + m_k * SW_k )
            // idot_0 = dp8a(w.x,a.x) chained with dp8a(w.y,a.y) (words 0,1 =
            // values 0..7 + 16..23 — the canonical word-pair sub-block),
            // idot_1 = words 2,3 chained. Chaining is exact (int32 dot of 16
            // pairs, no per-32 cross-sub-block mixing — unlike the per-32
            // chain this stays inside one scale group).
            // SW_k = sWs16 (SIGNED nib_sum_word sums - 128, the 3b class),
            // a = sXq q[kk][0] (4 u32 words — 28 B stride is not 16 B aligned),
            // d4 pair = q[kk][1].x halves {d4_0,d4_1}, m pair = q[kk][1].y
            // (rp_x_sub_u4x16_from_blocks), d = Q4_0 f16 scale from sWdh
            // (zeroed for dead rows at prefetch).
            // Dot count per block per (r,n) = 4 dot8 — SAME as per-32/per-8.
            for (int kk = 0; kk < MMQ_RP_Q8_BK; kk++) {
                const int row_base = ty * NROW;
                uint4 w = sW_lo[row_base][kk];
                const uint4 * dp16 = reinterpret_cast<const uint4 *>(&sWdh[kk][row_base]);
#pragma unroll
                for (int g = 0; g < NROW; g += SPW) {
                    const uint4 dh4 = dp16[g / SPW];
                    const slot_t * dh = reinterpret_cast<const slot_t *>(&dh4);
#pragma unroll
                    for (int rr = 0; rr < SPW; rr++) {
                        const int r = g + rr;
                        const uint4 w_c = w;
                        if (r + 1 < NROW) {
                            w = sW_lo[row_base + r + 1][kk];
                        }
                        const float d  = rp_traits<WT>::scale(dh[rr]);
                        // Two SW_k for this (slab, row): 8 B aligned in the
                        // union (float[2] rows), load as one float2.
                        const float2 sw2 = *reinterpret_cast<const float2 *>(
                            &sWs16[kk][row_base + r][0]);
#pragma unroll
                        for (int n = 0; n < TN_; n++) {
                            const int xcol = sxc0 + n * CW;
                            const uint4 a = sXq[xcol].q[kk][0];
                            const uint4 s = sXq[xcol].q[kk][1];
                            const float2 p  = __half22float2(*reinterpret_cast<const half2 *>(&s.x));
                            const float2 mm = __half22float2(*reinterpret_cast<const half2 *>(&s.y));
                            const int i01 = ggml_cuda_dp8a((int) w_c.x, (int) a.x,
                                            ggml_cuda_dp8a((int) w_c.y, (int) a.y, 0));
                            const int i23 = ggml_cuda_dp8a((int) w_c.z, (int) a.z,
                                            ggml_cuda_dp8a((int) w_c.w, (int) a.w, 0));
                            acc[r][n] += d * (p.x * ((float) i01 + 8.0f * sw2.x) + mm.x * sw2.x);
                            acc[r][n] += d * (p.y * ((float) i23 + 8.0f * sw2.y) + mm.y * sw2.y);
                        }
                    }
                }
            }
            return;
        }
        for (int kk = 0; kk < MMQ_RP_Q8_BK; kk++) {
            float dx  [TN_];
            [[maybe_unused]] float sx[TN_];
            int   xq32[TN_][8];
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const int xcol = sxc0 + n * CW;
                const uint4 q0 = sXq[xcol].q[kk][0];
                const uint4 q1 = sXq[xcol].q[kk][1];
                dx[n] = sXd[xcol][kk];
                if constexpr (rp_traits<WT>::affine) {
                    sx[n] = sXs[xcol][kk];
                }
                xq32[n][0] = (int) q0.x;
                xq32[n][1] = (int) q0.y;
                xq32[n][2] = (int) q0.z;
                xq32[n][3] = (int) q0.w;
                xq32[n][4] = (int) q1.x;
                xq32[n][5] = (int) q1.y;
                xq32[n][6] = (int) q1.z;
                xq32[n][7] = (int) q1.w;
            }
            const int row_base = ty * NROW;
            uint4 wlo = sW_lo[row_base][kk];
            uint4 whi = sW_hi[row_base][kk];
            const uint4 * dp16 = reinterpret_cast<const uint4 *>(&sWdh[kk][row_base]);

#pragma unroll
            for (int g = 0; g < NROW; g += SPW) {
                const uint4 dh4 = dp16[g / SPW];
                const slot_t * dh = reinterpret_cast<const slot_t *>(&dh4);

#pragma unroll
                for (int rr = 0; rr < SPW; rr++) {
                    const int r = g + rr;

                    const uint4 wlo_c = wlo, whi_c = whi;
                    if (r + 1 < NROW) {
                        wlo = sW_lo[row_base + r + 1][kk];
                        whi = sW_hi[row_base + r + 1][kk];
                    }

                    [[maybe_unused]] float d = 0.0f, dlo = 0.0f, dhi = 0.0f;
                    if constexpr (rp_traits<WT>::split_scales || rp_traits<WT>::affine) {
                        const float2 dd = rp_traits<WT>::scale2(dh[rr]);
                        dlo = dd.x;
                        dhi = dd.y;
                    } else {
                        d = rp_traits<WT>::scale(dh[rr]);
                    }

#pragma unroll
                    for (int n = 0; n < TN_; n++) {
                        if constexpr (WT == GGML_TYPE_MXFP4 || rp_traits<WT>::split_scales) {
                            // Stock MXFP4 uses MMQ_DP4A_TXS_Q8_1, i.e. it applies the
                            // scale per 16-value HALF block, while Q8_0 uses TXS_Q8_0
                            // and scales per 32. Split the accumulator the same way -
                            // d*(lo) + d*(hi) instead of d*(lo+hi) - or the result is
                            // mathematically equal but not bit-identical to stock.
                            int lo = 0, hi = 0;
                            lo = ggml_cuda_dp4a((int) wlo_c.x, xq32[n][0], lo);
                            lo = ggml_cuda_dp4a((int) wlo_c.y, xq32[n][1], lo);
                            lo = ggml_cuda_dp4a((int) wlo_c.z, xq32[n][2], lo);
                            lo = ggml_cuda_dp4a((int) wlo_c.w, xq32[n][3], lo);
                            hi = ggml_cuda_dp4a((int) whi_c.x, xq32[n][4], hi);
                            hi = ggml_cuda_dp4a((int) whi_c.y, xq32[n][5], hi);
                            hi = ggml_cuda_dp4a((int) whi_c.z, xq32[n][6], hi);
                            hi = ggml_cuda_dp4a((int) whi_c.w, xq32[n][7], hi);
                            if constexpr (rp_traits<WT>::split_scales) {
                                // Per-16-value scales (Q6_K): each half gets its own.
                                acc[r][n] += dlo * dx[n] * (float) lo + dhi * dx[n] * (float) hi;
                            } else {
                                acc[r][n] += d * dx[n] * (float) lo + d * dx[n] * (float) hi;
                            }
                        } else {
                        int idot = 0;
                        idot = ggml_cuda_dp4a((int) wlo_c.x, xq32[n][0], idot);
                        idot = ggml_cuda_dp4a((int) whi_c.x, xq32[n][4], idot);
                        idot = ggml_cuda_dp4a((int) wlo_c.y, xq32[n][1], idot);
                        idot = ggml_cuda_dp4a((int) whi_c.y, xq32[n][5], idot);
                        idot = ggml_cuda_dp4a((int) wlo_c.z, xq32[n][2], idot);
                        idot = ggml_cuda_dp4a((int) whi_c.z, xq32[n][6], idot);
                        idot = ggml_cuda_dp4a((int) wlo_c.w, xq32[n][3], idot);
                        idot = ggml_cuda_dp4a((int) whi_c.w, xq32[n][7], idot);
                        if constexpr (rp_traits<WT>::affine) {
                            // Affine (Q4_K): d*sc times the dot, minus dmin*m times the sum.
                            acc[r][n] += dlo * dx[n] * (float) idot - dhi * sx[n];
                        } else {
                            acc[r][n] += d * dx[n] * (float) idot;
                        }
                        }
                    }
                }
            }
        }
    };

    // Pipelined K loop: store current slab -> sync -> prefetch next slab -> compute -> sync.
    if (full_tile) {
        prefetch_w_full(sb_lo);
        prefetch_x_full(sb_lo, x_group);

        for (uint32_t sb0 = sb_lo; sb0 < sb_hi; sb0 += MMQ_RP_Q8_BK) {
            store_stage();
            __syncthreads();

            if (sb0 + MMQ_RP_Q8_BK < sb_hi) {
                x_group += n_tok;
                prefetch_w_full(sb0 + MMQ_RP_Q8_BK);
                prefetch_x_full(sb0 + MMQ_RP_Q8_BK, x_group);
            }

            compute_stage();
            __syncthreads();
        }
    } else {
        prefetch_w_checked(sb_lo);
        prefetch_x_checked(sb_lo, x_group);

        for (uint32_t sb0 = sb_lo; sb0 < sb_hi; sb0 += MMQ_RP_Q8_BK) {
            store_stage();
            __syncthreads();

            if (sb0 + MMQ_RP_Q8_BK < sb_hi) {
                x_group += n_tok;
                prefetch_w_checked(sb0 + MMQ_RP_Q8_BK);
                prefetch_x_checked(sb0 + MMQ_RP_Q8_BK, x_group);
            }

            compute_stage();
            __syncthreads();
        }
    }

    // Epilogue: write accumulators to y (float4 when the row stride is aligned;
    // MoE scatters each column through ids_dst to its destination token).
    // KSPLIT > 1: y points at the fp32 partial scratch, [KSPLIT][n_tok][ne1]
    // row-major; this block writes its z plane, the fixed-order reduce kernel
    // sums the planes afterwards. Output pitch becomes ne1 (always 4-aligned
    // for the gated shapes, but vec_ok re-checks it).
    float * const  y_out  = KSPLIT > 1 ? y + (size_t) blockIdx.z * n_tok * ne1 : y;
    const uint32_t out_s1 = KSPLIT > 1 ? ne1 : dst_s1;
    const uint32_t rbase  = row0 + ty * NROW;
    const int      nvalid = min((int) ne1 - (int) rbase, NROW);
    const bool     vec_ok = (out_s1 & 3) == 0;
    if (nvalid > 0) {
#pragma unroll
        for (int n = 0; n < TN_; n++) {
            const uint32_t col = tx + n * CW;
            float * dst;
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + col;
                if (!(a < a_end && col < BN)) {
                    continue;
                }
                dst = y + (size_t) ids_dst[a] * dst_s1 + rbase;
            } else {
                const uint32_t tok = tok0 + col;
                if (tok >= n_tok) {
                    continue;
                }
                dst = y_out + (size_t) tok * out_s1 + rbase;
            }
            static_assert(NROW % 4 == 0, "float4 epilogue needs NROW % 4 == 0");
#pragma unroll
            for (int g = 0; g < NROW; g += 4) {
                if (vec_ok && g + 4 <= nvalid) {
                    const float4 v = { acc[g + 0][n], acc[g + 1][n], acc[g + 2][n], acc[g + 3][n] };
                    *reinterpret_cast<float4 *>(dst + g) = v;
                } else {
#pragma unroll
                    for (int r = g; r < min(g + 4, nvalid); r++) {
                        dst[r] = acc[r][n];
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, ids_src1, ids_dst, expert_bounds, tile_off, tile_meta, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif
}

// Launch wrapper: 64-wide waves, tuned for large batches (ne11 >= 128).
// G2: U4ACT threaded to the impl (the G1 commit message claimed the wrappers
// were already threaded; they were not -- done here). Default false keeps the
// stock instantiation byte-identical.
// M4: U4P8 threaded the same way (default false = byte-identical stock).
template <bool HAS_IDS, int TN_, int NRL, ggml_type WT = GGML_TYPE_Q8_0, bool U4ACT = false, bool U4P8 = false, bool U4X16 = false>
static __global__ void __launch_bounds__(64 * NRL, 2) mmq_gemm_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const repack_tile_meta * __restrict__ tile_meta,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    mmq_gemm_repacked_impl<WT, HAS_IDS, 64, TN_, NRL, U4ACT, U4P8, U4X16>(
        wbase, xq, y, ne0, ne1, n_tok, ids_src1, ids_dst,
        expert_bounds, tile_off, tile_meta, n_expert, expert_stride, dst_s1);
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, ids_src1, ids_dst, expert_bounds, tile_off, tile_meta, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif
}

// Launch wrapper: 32-wide waves, tuned for small batches.
// Narrow-batch mat-vec: NCOLS tokens per weight pass. A 2-8 token batch (a
// speculative verify step, a tiny ubatch) is too narrow for the tiled GEMM,
// which computes a full 32-wide tile whatever the batch is - measured 274 us
// against 140 us for the canonical mul_mat_vec_q at 4 tokens. Reading the
// weights once and keeping one accumulator per token amortizes the weight
// traffic the same way, on the repacked layout. Dense only: the single-token
// path and the MoE ids path keep their existing kernels.
template <int ROWS, int NWAVES, int NCOLS, int RPL = 1, bool HAS_FUSION = false,
          int LANES = 64, ggml_type WT = GGML_TYPE_Q8_0, bool U4ACT = false,
          bool U4P8 = false, bool U4X16 = false>
static __device__ __forceinline__ void mul_mat_vec_repacked_nc_impl(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t xs, const uint32_t ys,
        const uint8_t * __restrict__ wbase_gate, const float * __restrict__ x_bias,
        const float * __restrict__ gate_bias, const ggml_glu_op glu_op,
        const uint32_t bias_row_base) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    static_assert(!U4ACT || WT == GGML_TYPE_Q4_0, "U4ACT (dp8a) is only defined for Q4_0 weights");
    static_assert(!U4ACT || !HAS_FUSION, "U4ACT (dp8a) does not support the fused-FFN variant");
    // C1/B8 M2: the per-8 u4p variant carries the same guards as U4ACT (raw
    // Q4_0 nibbles + dp8a), and is mutually exclusive with the per-32 U4ACT
    // branch — the two activation layouts cannot mix in one kernel.
    static_assert(!U4P8 || WT == GGML_TYPE_Q4_0, "U4P8 (dp8a per-8) is only defined for Q4_0 weights");
    static_assert(!U4P8 || !HAS_FUSION, "U4P8 (dp8a per-8) does not support the fused-FFN variant");
    static_assert(!(U4ACT && U4P8), "U4ACT (per-32) and U4P8 (per-8) are mutually exclusive");
    // Phase-2 gate 3: per-16 twin, same guards, mutually exclusive with both.
    static_assert(!U4X16 || WT == GGML_TYPE_Q4_0, "U4X16 (dp8a per-16) is only defined for Q4_0 weights");
    static_assert(!U4X16 || !HAS_FUSION, "U4X16 (dp8a per-16) does not support the fused-FFN variant");
    static_assert(!(U4ACT && U4X16), "U4ACT (per-32) and U4X16 (per-16) are mutually exclusive");
    static_assert(!(U4P8 && U4X16), "U4P8 (per-8) and U4X16 (per-16) are mutually exclusive");
    // LANES is the number of lanes cooperating on ONE row. It is the geometry
    // knob that matters here: the strided loop runs n_blocks/LANES iterations
    // and is then followed by a log2(LANES)-step DPP reduction, so a wide lane
    // group on a short row spends most of its time reducing. At ne0=5120,
    // n_blocks is 160, which is 2.5 iterations at 64 lanes against a 6-step
    // reduction, and 10 iterations at 16 lanes against a 4-step one.
    constexpr int NWARPS = (NWAVES * 64) / LANES;
    static_assert(LANES >= 1 && LANES <= 64 && (LANES & (LANES - 1)) == 0,
                  "LANES must be a power of two, at most one wave");
    static_assert(ROWS == NWARPS * RPL, "row geometry");
    const int warp_id = threadIdx.x / LANES;
    const int lane    = threadIdx.x % LANES;
    const int row     = blockIdx.x * ROWS + warp_id * RPL;

    const uint32_t n_blocks = ne0 >> 5;
    const typename rp_traits<WT>::geom wg(ne0, ne1);

    // RPL rows per lane: the activation block is loaded once into registers and
    // reused against every row the lane owns, so the activation side stops
    // being re-read per row. This is what the canonical mat-vec does with its
    // register blocking, and at NCOLS=4 the activations are otherwise 4x the
    // weight bytes per step.
    uint32_t wrows [RPL];
    bool rvalid[RPL];
#pragma unroll
    for (int r = 0; r < RPL; ++r) {
        const int rr = row + r;
        wrows [r] = min((uint32_t) rr, ne1 - 1);
        // Masking the raw scale slot zeroes dead-row terms for f16; for e8m0 a
        // zero slot decodes to a tiny nonzero, which is finite and discarded at
        // the guarded write below either way.
        rvalid[r] = rr < (int) ne1;
    }

    // Fused FFN: the gate matrix is read in the same pass as up, so the
    // activation blocks, the row bookkeeping and the launch are all shared.
    // Widths 2-8 otherwise run up, gate and the GLU as three launches where
    // the canonical path runs one.
    constexpr int GR = HAS_FUSION ? RPL   : 1;
    constexpr int GC = HAS_FUSION ? NCOLS : 1;

    float acc[RPL][NCOLS] = {};
    [[maybe_unused]] float accg[GR][GC] = {};

    // Whole 32-quant blocks per lane step: one weight-scale read and one
    // activation block per column per 32 quants. The half-block stride paid
    // both twice, and at sharded row counts the loop is a handful of
    // iterations, so per-step overhead is most of the kernel.
    for (uint32_t sb = (uint32_t) lane; sb < n_blocks; sb += (uint32_t) LANES) {
        if constexpr (U4ACT) {
            // W4A4 milestone 2b: dp8a decode, replacing the whole stock body.
            // Weight block = the RAW packed uint4 (canonical Q4_0 bytes:
            // excess-8 codes, interleaved pairs). XOR 0x88888888 per word
            // turns the excess-8 codes into two's-complement nibbles
            // (cw ^ 8 == cw - 8 as 4-bit signed). Activations are
            // block_q8_1_u4 (20 B: qs[16] + fp16 d4/m); read qs as four
            // u32s — the 20 B stride keeps every block 4-byte aligned, so
            // no misaligned uint4 load. xs keeps its "blocks per token
            // column" meaning; only the element size changes with the type.
            //
            // Epilogue (numeric seal: /tmp/dp8a_dot_test.cc, 1000 random
            // blocks, exact match):
            //   dot = d * ( d4 * (idot + 8*SUMW) + m * SUMW )
            // with idot = sdot8(w_tc, a_tc) over the block, SUMW =
            // sum(cw - 8) = nib_sum_u4(raw) - 256. The shorter
            // d*(d4*idot + m*SUMW) form is NOT exact (fails the seal) —
            // the activation value is q*d4 + m with q = a_tc + 8, and the
            // +8*d4*SUMW term is real.
            //
            // Dead rows: zero the scale slot (rp_slot_keep) exactly like the
            // stock path — d == 0 zeroes the whole term.
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
                uint4 pk;
                typename rp_traits<WT>::slot_t db;
                rp_traits<WT>::load_w_raw(wbase, wg, wrows[r], sb, pk, db);
                const uint4 w_tc = make_uint4(pk.x ^ 0x88888888u, pk.y ^ 0x88888888u,
                                            pk.z ^ 0x88888888u, pk.w ^ 0x88888888u);
                const float d  = rp_traits<WT>::scale(rp_slot_keep(db, rvalid[r]));
                // SUMW must be computed SIGNED: the 32 raw excess-8 nibble codes
                // average ~7.5 each (sum ~240), so nib_sum_u4(pk) < 256 for
                // roughly half of all blocks. The unsigned subtraction wrapped to
                // ~2^32 there, injecting ~1e9-magnitude garbage into the dot
                // (gate-on decode emitted token 0). Cast to int before subtracting.
                const float sw = (float) ((int) nib_sum_u4(pk) - 256);
#pragma unroll
                for (int c = 0; c < NCOLS; ++c) {
                    const block_q8_1_u4 * xb4 =
                        reinterpret_cast<const block_q8_1_u4 *>(xq) + (size_t) c * xs + sb;
                    const uint32_t * q32 = reinterpret_cast<const uint32_t *>(xb4->qs);
                    const uint4 a = make_uint4(q32[0], q32[1], q32[2], q32[3]);
                    const int idot = mmvq_dp8a_u4(w_tc, a);
                    const float d4 = __half2float(xb4->d4);
                    const float m  = __half2float(xb4->m);
                    acc[r][c] += d * (d4 * ((float) idot + 8.0f * sw) + m * sw);
                }
            }
            continue;
        }
        if constexpr (U4P8) {
            // C1/B8 milestone M2: per-8 dp8a decode (scheme B8). Same raw-Q4_0
            // weight read as the U4ACT branch (load_w_raw: one uint4 = 32
            // values = 4 per-8 sub-blocks, word k = values 8k..8k+7, excess-8
            // codes). Activations are block_q8_1_u4p (32 B: qs[16] + fp16
            // d4[4]/m[4]); the 32 B stride keeps every block 16 B aligned, so
            // qs loads as a full uint4 (unlike the 20 B per-32 u4, which had
            // to go through 4x u32). xs keeps its "blocks per token column"
            // meaning; only the element size changes with the type.
            //
            // Epilogue FORM 1 (CPU seal: docs/w4a4-per8-m1.md §3, 1504 blocks,
            // 0 fails, worst rel 8.4e-08) — per sub-block k, summed:
            //   acc += d * ( d4_k * (idot_k + 8*SW_k) + m_k * SW_k )
            // with idot_k = dp8a(w_tc_k, a_k, 0) UNCHAINED per sub-block and
            // SW_k = nib_sum_word(raw_k) - 64 (signed before the float cast —
            // the 3b root-cause class). The fp16-stored m-fold variant is NOT
            // exact (29/1504 fail) — do not substitute it.
            //
            // Dead rows: zero the scale slot (rp_slot_keep) exactly like the
            // stock path — d == 0 zeroes the whole term.
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
                uint4 pk;
                typename rp_traits<WT>::slot_t db;
                rp_traits<WT>::load_w_raw(wbase, wg, wrows[r], sb, pk, db);
                const float d = rp_traits<WT>::scale(rp_slot_keep(db, rvalid[r]));
#pragma unroll
                for (int c = 0; c < NCOLS; ++c) {
                    const block_q8_1_u4p * xb4 =
                        reinterpret_cast<const block_q8_1_u4p *>(xq) + (size_t) c * xs + sb;
                    const uint4 a = *reinterpret_cast<const uint4 *>(xb4->qs);
                    const mmvq_dp8a_u4p_result res = mmvq_dp8a_u4p(pk, a);
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const float d4 = __half2float(xb4->d4[k]);
                        const float m  = __half2float(xb4->m[k]);
                        acc[r][c] += d * (d4 * ((float) res.idot[k] + 8.0f * res.sw[k])
                                         + m * res.sw[k]);
                    }
                }
            }
            continue;
        }
        if constexpr (U4X16) {
            // Phase-2 gate 3: per-16 dp8a (scheme B16), nc twin of the GEMM
            // U4X16 loop. Two sub-blocks of two dp8a words each (word pair
            // 0,1 = sub-block 0, pair 2,3 = sub-block 1), form-B epilogue per
            // sub-block: acc += d * ( d4_k * (idot_k + 8*SW_k) + m_k * SW_k )
            // with idot_k the chained dp8a over the pair and SW_k =
            // nib_sum_word(raw_2k) + nib_sum_word(raw_2k+1) - 128 (signed —
            // the 3b class). Activations are block_q8_1_u4x16 (28 B: qs[16] +
            // fp16 d4[2]/m[2]); the 28 B stride keeps every block 4 B aligned,
            // so qs is read as four u32s (the 20 B per-32 u4 precedent).
            // Dead rows: zero the scale slot (rp_slot_keep) as shipped.
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
                uint4 pk;
                typename rp_traits<WT>::slot_t db;
                rp_traits<WT>::load_w_raw(wbase, wg, wrows[r], sb, pk, db);
                const float d = rp_traits<WT>::scale(rp_slot_keep(db, rvalid[r]));
                const uint4 w_tc = make_uint4(pk.x ^ 0x88888888u, pk.y ^ 0x88888888u,
                                            pk.z ^ 0x88888888u, pk.w ^ 0x88888888u);
                const float sw0 = (float) ((int) nib_sum_word(pk.x) + (int) nib_sum_word(pk.y) - 128);
                const float sw1 = (float) ((int) nib_sum_word(pk.z) + (int) nib_sum_word(pk.w) - 128);
#pragma unroll
                for (int c = 0; c < NCOLS; ++c) {
                    const block_q8_1_u4x16 * xb4 =
                        reinterpret_cast<const block_q8_1_u4x16 *>(xq) + (size_t) c * xs + sb;
                    const uint32_t * q32 = reinterpret_cast<const uint32_t *>(xb4->qs);
                    const int i01 = ggml_cuda_dp8a((int) w_tc.x, (int) q32[0],
                                    ggml_cuda_dp8a((int) w_tc.y, (int) q32[1], 0));
                    const int i23 = ggml_cuda_dp8a((int) w_tc.z, (int) q32[2],
                                    ggml_cuda_dp8a((int) w_tc.w, (int) q32[3], 0));
                    const float d40 = __half2float(xb4->d4[0]);
                    const float m0  = __half2float(xb4->m[0]);
                    const float d41 = __half2float(xb4->d4[1]);
                    const float m1  = __half2float(xb4->m[1]);
                    acc[r][c] += d * (d40 * ((float) i01 + 8.0f * sw0) + m0 * sw0);
                    acc[r][c] += d * (d41 * ((float) i23 + 8.0f * sw1) + m1 * sw1);
                }
            }
            continue;
        }
        // Weights for the lane's rows stay in registers; the activation block is
        // re-read per row from L1, which is cheaper than the register pressure
        // of holding NCOLS x 8 ints alongside the accumulators.
        uint4 wv0[RPL], wv1[RPL];
        [[maybe_unused]] float  dw [RPL];
        [[maybe_unused]] float2 dw2[RPL];
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
            typename rp_traits<WT>::slot_t db;
            rp_traits<WT>::load_w(wbase, wg, wrows[r], sb, wv0[r], wv1[r], db);
            if constexpr (rp_traits<WT>::split_scales || rp_traits<WT>::affine) {
                dw2[r] = rp_traits<WT>::scale2(rp_slot_keep(db, rvalid[r]));
            } else {
                dw[r] = rp_traits<WT>::scale(rp_slot_keep(db, rvalid[r]));
            }
        }
        [[maybe_unused]] uint4 gv0[GR], gv1[GR];
        [[maybe_unused]] float  dg [GR];
        [[maybe_unused]] float2 dg2[GR];
        if constexpr (HAS_FUSION) {
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
                typename rp_traits<WT>::slot_t gb;
                rp_traits<WT>::load_w(wbase_gate, wg, wrows[r], sb, gv0[r], gv1[r], gb);
                if constexpr (rp_traits<WT>::split_scales || rp_traits<WT>::affine) {
                    dg2[r] = rp_traits<WT>::scale2(rp_slot_keep(gb, rvalid[r]));
                } else {
                    dg[r] = rp_traits<WT>::scale(rp_slot_keep(gb, rvalid[r]));
                }
            }
        }
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            const block_q8_1 * xb = xq + (size_t) c * xs + sb;
            const float dx = __low2float(xb->ds);
            [[maybe_unused]] const float sx = __high2float(xb->ds);
            const int * x = reinterpret_cast<const int *>(xb->qs);
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
                int ilo = 0, ihi = 0;
                ilo = ggml_cuda_dp4a((int) wv0[r].x, x[0], ilo);
                ilo = ggml_cuda_dp4a((int) wv0[r].y, x[1], ilo);
                ilo = ggml_cuda_dp4a((int) wv0[r].z, x[2], ilo);
                ilo = ggml_cuda_dp4a((int) wv0[r].w, x[3], ilo);
                ihi = ggml_cuda_dp4a((int) wv1[r].x, x[4], ihi);
                ihi = ggml_cuda_dp4a((int) wv1[r].y, x[5], ihi);
                ihi = ggml_cuda_dp4a((int) wv1[r].z, x[6], ihi);
                ihi = ggml_cuda_dp4a((int) wv1[r].w, x[7], ihi);
                if constexpr (rp_traits<WT>::split_scales) {
                    acc[r][c] += dw2[r].x * dx * (float) ilo + dw2[r].y * dx * (float) ihi;
                } else if constexpr (rp_traits<WT>::affine) {
                    acc[r][c] += dw2[r].x * dx * (float) (ilo + ihi) - dw2[r].y * sx;
                } else {
                    // one add of the two halves keeps the Q8_0 result bit-identical
                    acc[r][c] += dw[r] * dx * (float) (ilo + ihi);
                }
                if constexpr (HAS_FUSION) {
                    int glo_d = 0, ghi_d = 0;
                    glo_d = ggml_cuda_dp4a((int) gv0[r].x, x[0], glo_d);
                    glo_d = ggml_cuda_dp4a((int) gv0[r].y, x[1], glo_d);
                    glo_d = ggml_cuda_dp4a((int) gv0[r].z, x[2], glo_d);
                    glo_d = ggml_cuda_dp4a((int) gv0[r].w, x[3], glo_d);
                    ghi_d = ggml_cuda_dp4a((int) gv1[r].x, x[4], ghi_d);
                    ghi_d = ggml_cuda_dp4a((int) gv1[r].y, x[5], ghi_d);
                    ghi_d = ggml_cuda_dp4a((int) gv1[r].z, x[6], ghi_d);
                    ghi_d = ggml_cuda_dp4a((int) gv1[r].w, x[7], ghi_d);
                    if constexpr (rp_traits<WT>::split_scales) {
                        accg[r][c] += dg2[r].x * dx * (float) glo_d + dg2[r].y * dx * (float) ghi_d;
                    } else if constexpr (rp_traits<WT>::affine) {
                        accg[r][c] += dg2[r].x * dx * (float) (glo_d + ghi_d) - dg2[r].y * sx;
                    } else {
                        accg[r][c] += dg[r] * dx * (float) (glo_d + ghi_d);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < RPL; ++r) {
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            const float a = rp_warp_reduce_sum<LANES>(acc[r][c]);
            [[maybe_unused]] float g = 0.0f;
            if constexpr (HAS_FUSION) {
                g = rp_warp_reduce_sum<LANES>(accg[r][c]);
            }
            if (lane == 0 && (row + r) < (int) ne1) {
                float out = a;
                if constexpr (HAS_FUSION) {
                    out = rp_mmv_fusion_epilogue(a, g, x_bias, gate_bias,
                        bias_row_base + (uint32_t)(row + r), glu_op, true);
                }
                y[(size_t) c * ys + row + r] = out;
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, xs, ys,
                     wbase_gate, x_bias, gate_bias, glu_op, bias_row_base);
    NO_DEVICE_CODE;
#endif
}

template <int ROWS, int NWAVES, int NCOLS, int RPL = 1, int LANES = 64, ggml_type WT = GGML_TYPE_Q8_0,
          bool U4ACT = false, bool U4P8 = false, bool U4X16 = false>
static __global__ void mul_mat_vec_repacked_nc(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t xs, const uint32_t ys) {
    mul_mat_vec_repacked_nc_impl<ROWS, NWAVES, NCOLS, RPL, false, LANES, WT, U4ACT, U4P8, U4X16>(
        wbase, xq, y, ne0, ne1, xs, ys, nullptr, nullptr, nullptr,
        GGML_GLU_OP_REGLU, 0);
}

// MoE narrow batch: one assignment per block, one launch for all of them.
// Expert token counts are per expert, so at verify widths nearly every active
// expert holds a single assignment and the 32-wide tile computes 32 columns to
// serve it. Each block resolves its own expert from expert_bounds below, and
// ids_src1/ids_dst give the source and destination rows.
// Which expert owns assignment a. expert_bounds is the monotonic per-expert
// start offset, so this is an upper_bound minus one - cheap enough per block
// that building a metadata array up front is not worth a kernel launch.
static __device__ __forceinline__ uint32_t expert_bounds_search(
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const uint32_t a) {
    uint32_t lo = 0, hi = n_expert;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if ((uint32_t) expert_bounds[mid + 1] <= a) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

template <int ROWS, int NWAVES, int RPL = 1, ggml_type WT = GGML_TYPE_Q8_0,
          bool U4ACT = false>
static __global__ void mul_mat_vec_repacked_id1(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs, const uint32_t ys) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    const uint32_t a = blockIdx.y;
    const uint32_t e = expert_bounds_search(expert_bounds, n_expert, a);
    // W4A4 milestone 4: under U4ACT the activation pool holds 20-byte
    // block_q8_1_u4 blocks, so the per-assignment column offset must be
    // applied in u4 units (cast first, THEN add) -- the stock block_q8_1*
    // pointer arithmetic would stride 32 B per block past the data. The
    // impl's U4ACT branch reinterprets the already-offset base back to
    // block_q8_1_u4* and adds only c*xs (NCOLS=1 here, c=0).
    const block_q8_1 * xq_a = U4ACT
        ? reinterpret_cast<const block_q8_1 *>(
              reinterpret_cast<const block_q8_1_u4 *>(xq) + (size_t) ids_src1[a] * xs)
        : xq + (size_t) ids_src1[a] * xs;
    mul_mat_vec_repacked_nc_impl<ROWS, NWAVES, 1, RPL, false, 64, WT, U4ACT>(
        wbase + (size_t) e * expert_stride,
        xq_a,
        y  + (size_t) ids_dst[a]  * ys,
        ne0, ne1, xs, ys, nullptr, nullptr, nullptr, GGML_GLU_OP_REGLU, 0);
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds,
                     n_expert, expert_stride, xs, ys);
    NO_DEVICE_CODE;
#endif
}

// Fused MoE narrow batch: one assignment per block, up and gate in one pass.
// The ADD_ID bias is [ne1, n_expert], so the row index carries the expert.
template <int ROWS, int NWAVES, int RPL = 1, ggml_type WT = GGML_TYPE_Q8_0>
static __global__ void mul_mat_vec_repacked_id1_fused(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const size_t expert_stride, const uint32_t xs, const uint32_t ys,
        const uint8_t * __restrict__ wbase_gate, const float * __restrict__ x_bias,
        const float * __restrict__ gate_bias, const ggml_glu_op glu_op) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    const uint32_t a = blockIdx.y;
    const uint32_t e = expert_bounds_search(expert_bounds, n_expert, a);
    mul_mat_vec_repacked_nc_impl<ROWS, NWAVES, 1, RPL, true, 64, WT>(
        wbase + (size_t) e * expert_stride,
        xq + (size_t) ids_src1[a] * xs,
        y  + (size_t) ids_dst[a]  * ys,
        ne0, ne1, xs, ys,
        wbase_gate + (size_t) e * expert_stride,
        x_bias, gate_bias, glu_op, e * ne1);
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds,
                     n_expert, expert_stride, xs, ys,
                     wbase_gate, x_bias, gate_bias, glu_op);
    NO_DEVICE_CODE;
#endif
}

template <bool HAS_IDS, int TN_, int NRL, ggml_type WT = GGML_TYPE_Q8_0, bool U4ACT = false, bool U4P8 = false, bool U4X16 = false,
          int KSPLIT = 1>
static __global__ void __launch_bounds__(32 * NRL, HAS_IDS && (WT == GGML_TYPE_Q4_K || WT == GGML_TYPE_Q5_K) ? 2 : 3) mmq_gemm_repacked_w32(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const repack_tile_meta * __restrict__ tile_meta,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    mmq_gemm_repacked_impl<WT, HAS_IDS, 32, TN_, NRL, U4ACT, U4P8, U4X16, KSPLIT>(
        wbase, xq, y, ne0, ne1, n_tok, ids_src1, ids_dst,
        expert_bounds, tile_off, tile_meta, n_expert, expert_stride, dst_s1);
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, ids_src1, ids_dst, expert_bounds, tile_off, tile_meta, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif
}

// GGML_RP_KSPLIT reduce: sum the KSPLIT fp32 partial planes (laid out
// [z][tok][row], row pitch ne1) into dst in FIXED z order -- deterministic,
// no atomics. One thread per (tok, row) output element.
template <int KSPLIT>
static __global__ void mmq_gemm_repacked_ksplit_reduce(
        const float * __restrict__ part, float * __restrict__ y,
        const uint32_t ne1, const uint32_t n_tok, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    const size_t plane = (size_t) n_tok * ne1;
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= plane) {
        return;
    }
    float s = 0.0f;
#pragma unroll
    for (int z = 0; z < KSPLIT; ++z) {
        s += part[(size_t) z * plane + i];
    }
    const uint32_t tok = (uint32_t) (i / ne1);
    const uint32_t row = (uint32_t) (i % ne1);
    y[(size_t) tok * dst_s1 + row] = s;
#else
    GGML_UNUSED_VARS(part, y, ne1, n_tok, dst_s1);
    NO_DEVICE_CODE;
#endif
}

// W4A4 G2: force-instantiation compile-check for the U4ACT=true GEMM variants
// (kept #if 0 as documentation; compiled standalone with the exact build flags
// at G2 time -- see docs/w4a4-kernel-notes.md G2 section for the verified
// codegen numbers). Uncomment to re-verify codegen without dispatch wiring.
// M4: the U4P8=true variants are force-instantiated the same way (the dispatch
// now emits them under gate=2; the explicit instantiations below pin the exact
// template args for the save-temps dot8 count check — 256 (64-wide) / 128
// (w32) v_dot8_i32_i4, same as the per-32 U4ACT GEMM: per-8 moves the
// epilogue granularity only).
#if 0
template __global__ void mmq_gemm_repacked<false, MMQ_RP_Q8_TN, MMQ_RP_Q8_NROW_LANES, GGML_TYPE_Q4_0, true>(
    const uint8_t *, const block_q8_1 *, float *, const uint32_t, const uint32_t,
    const uint32_t, const int32_t *, const int32_t *, const int32_t *, const int32_t *,
    const repack_tile_meta *, const uint32_t, const size_t, const uint32_t);
template __global__ void mmq_gemm_repacked_w32<false, 1, MMQ_RP_Q8_NROW_LANES * 2, GGML_TYPE_Q4_0, true>(
    const uint8_t *, const block_q8_1 *, float *, const uint32_t, const uint32_t,
    const uint32_t, const int32_t *, const int32_t *, const int32_t *, const int32_t *,
    const repack_tile_meta *, const uint32_t, const size_t, const uint32_t);
template __global__ void mmq_gemm_repacked<false, MMQ_RP_Q8_TN, MMQ_RP_Q8_NROW_LANES, GGML_TYPE_Q4_0, false, true>(
    const uint8_t *, const block_q8_1 *, float *, const uint32_t, const uint32_t,
    const uint32_t, const int32_t *, const int32_t *, const int32_t *, const int32_t *,
    const repack_tile_meta *, const uint32_t, const size_t, const uint32_t);
template __global__ void mmq_gemm_repacked_w32<false, 1, MMQ_RP_Q8_NROW_LANES * 2, GGML_TYPE_Q4_0, false, true>(
    const uint8_t *, const block_q8_1 *, float *, const uint32_t, const uint32_t,
    const uint32_t, const int32_t *, const int32_t *, const int32_t *, const int32_t *,
    const repack_tile_meta *, const uint32_t, const size_t, const uint32_t);
#endif
