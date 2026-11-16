// d1-mmvq.cuh - taskD1: bit-identical restructured mmvq twins for Q6_K and Q4_0.
//
// The generic mul_mat_vec_q calls vec_dot_q<type>_q8_1 once per (kbx, col j, row i).
// Every call re-loads and re-decodes the weight nibbles (invariant across the column
// loop j) and re-loads the q8_1 activation words + f16 scales (invariant across the
// row loop i). This twin hoists both: weight decode once per (kbx, row), activation
// once per (kbx, col). All per-(j,i,kbx) integer products and the fp32 accumulation
// sequence (sumf=0; sumf += d8[0]*(dot0*sc0); sumf += d8[1]*(dot1*sc1); d*sumf;
// tmp[j][i] += ...) are preserved EXACTLY, so the output is bit-identical by
// construction: same per-lane kbx chain, same kqs split, same reduce tree.
//
// GCN launch geometry cited per review rule 2 (calc_nwarps / calc_rows_per_block,
// MMVQ_PARAMETERS_GCN table, mmvq.cu @ int4-fix4):
//   calc_nwarps(GCN): ncols 1..4 -> 2 warps; ncols 5..8 -> 1 warp (Q8_0@5 -> 2).
//   calc_rows_per_block(GCN): ncols 1 -> 1 (2 with GGML_CUDA_MMVQ_GCN_RPB1=2,
//     the production lever); ncols 2..8 -> 2.
//   Q6_K: qi=QI6_K=32, vdr=1  -> kbx = tid/32, kqs = tid%32, kbx += 2*nwarps.
//   Q4_0: qi=QI4_0=4,  vdr=2  -> kbx = tid/2,  kqs = 2*(tid%2), kbx += 32*nwarps.
//   warp_size = 64 (gfx906 wave64).
//
// D1_DECODE=1 selects the vsubss4-free Q6_K decode:
//   raw = (vil|vih) in [0,63] per byte (fits signed i8);
//   dot(raw-32, u) == dot(raw, u) - 32*sum(u) as EXACT integers (no overflow:
//   |sum| <= 4*63*128), sum(u) = dp4a(0x01010101, u, 0). The int result is the
//   same value the generic __vsubss4 path produces (saturation never fires for
//   operands in [0,63]), so the fp sequence downstream is untouched.
//   Also folds ((vh>>4)<<4)&0x30303030 -> vh&0x30303030 (bit-exact identity:
//   0x30303030 has no bits in any low nibble).
#pragma once

#include "common.cuh"
#include "vecdotq.cuh"

// ---- Q6_K twin ---------------------------------------------------------------
// Geometry mirrors mul_mat_vec_q<GGML_TYPE_Q6_K, T, false, false, false, GRPB1>
// on the GCN table exactly (see header). nwarps/rpb are passed as template args
// computed by the launcher from calc_nwarps/calc_rows_per_block.
template <int ncols_dst, int nwarps_c, int rpb_c, int d1_decode, int unroll_c>
__launch_bounds__(nwarps_c*64, 1)
static __global__ void mul_mat_vec_q_q6k_d1(
        const void * __restrict__ vx_ptr, const void * __restrict__ vy_ptr, float * dst_ptr,
        const uint32_t ncols_x, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst) {

    const int tid = 64*threadIdx.y + threadIdx.x;
    const int row0 = rpb_c*blockIdx.x;
    const int blocks_per_row_x = ncols_x / QK_K;
    constexpr int blocks_per_iter = 2*nwarps_c;   // vdr*nwarps*64/qi = 1*nwarps*64/32

    const block_q8_1 * y = (const block_q8_1 *) vy_ptr;
    float * dst = dst_ptr + row0;
    const int kbx_offset = row0*stride_row_x;

    const int kqs = tid % 32;
    const int bq8_offset    = 4*(kqs / 16) + (kqs % 16) / 8;
    const int scale_offset  = 8*(kqs / 16) + (kqs % 16) / 4;
    const int vh_idx_shift  = 2*((kqs % 16) / 8);
    const int vh_idx        = 8*(kqs / 16) + kqs % 8;

    float tmp[ncols_dst][rpb_c] = {{0.0f}};

    // UNROLL sweep: the generic kbx loop has its #pragma unroll commented out;
    // unrolling here only lets the compiler software-pipeline loads across kbx
    // iterations - the per-lane kbx chain and accumulation order are unchanged,
    // so the result stays bit-identical.
#pragma unroll unroll_c
    for (int kbx = tid / 32; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (QK_K/QK8_1);   // = 8*kbx

        // ---- weight decode, hoisted across the column loop (row-invariant of j) ----
        struct wdec { int vi0, vi1; float d; int sc0, sc1; };
        wdec w[rpb_c];
#pragma unroll
        for (int i = 0; i < rpb_c; ++i) {
            const block_q6_K * bq6_K = (const block_q6_K *) vx_ptr + kbx_offset + i*stride_row_x + kbx;
            const int vl = get_int_b2(bq6_K->ql, kqs);
            const int vh = get_int_b2(bq6_K->qh, vh_idx) >> vh_idx_shift;
            const int8_t * scales = bq6_K->scales + scale_offset;
            w[i].sc0 = scales[0];
            w[i].sc1 = scales[4];
            w[i].d   = __half2float(bq6_K->d);
            if (d1_decode) {
                const int raw0 = (vl & 0x0F0F0F0F) | ((vh << 4) & 0x30303030);
                const int raw1 = ((vl >> 4) & 0x0F0F0F0F) | (vh & 0x30303030);
                w[i].vi0 = raw0;
                w[i].vi1 = raw1;
            } else {
                w[i].vi0 = __vsubss4((vl & 0x0F0F0F0F) | ((vh << 4) & 0x30303030), 0x20202020);
                w[i].vi1 = __vsubss4(((vl >> 4) & 0x0F0F0F0F) | (vh & 0x30303030), 0x20202020);
            }
        }

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const block_q8_1 * bq8_1 = y + j*stride_col_y + kby;
            // ---- activation, hoisted across the row loop (col-invariant of i) ----
            const int u0 = get_int_b4(bq8_1[bq8_offset + 0].qs, kqs % QI8_1);
            const int u1 = get_int_b4(bq8_1[bq8_offset + 2].qs, kqs % QI8_1);
            const float d80 = __low2float(bq8_1[bq8_offset + 0].ds);
            const float d81 = __low2float(bq8_1[bq8_offset + 2].ds);
            const int sumu0 = d1_decode ? ggml_cuda_dp4a(0x01010101, u0, 0) : 0;
            const int sumu1 = d1_decode ? ggml_cuda_dp4a(0x01010101, u1, 0) : 0;
#pragma unroll
            for (int i = 0; i < rpb_c; ++i) {
                float sumf = 0.0f;
                if (d1_decode) {
                    const int dot0 = ggml_cuda_dp4a(w[i].vi0, u0, 0) - (sumu0 << 5);
                    const int dot1 = ggml_cuda_dp4a(w[i].vi1, u1, 0) - (sumu1 << 5);
                    sumf += d80 * (dot0 * w[i].sc0);
                    sumf += d81 * (dot1 * w[i].sc1);
                } else {
                    sumf += d80 * (ggml_cuda_dp4a(w[i].vi0, u0, 0) * w[i].sc0);
                    sumf += d81 * (ggml_cuda_dp4a(w[i].vi1, u1, 0) * w[i].sc1);
                }
                tmp[j][i] += w[i].d * sumf;
            }
        }
    }

    // ---- reduce + epilogue: verbatim structure from mul_mat_vec_q ----
    __shared__ float tmp_shared[nwarps_c-1 > 0 ? nwarps_c-1 : 1][ncols_dst][rpb_c][64];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rpb_c; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rpb_c; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps_c-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum<64>(tmp[j][i]);
            if (threadIdx.x == i && (rpb_c == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                dst[j*stride_col_dst + i] = tmp[j][i];
            }
        }
    }
}

// ---- Q4_0 twin ---------------------------------------------------------------
// Geometry mirrors mul_mat_vec_q<GGML_TYPE_Q4_0, T, false, false, false, GRPB1>
// on the GCN table: qi=QI4_0=4, vdr=2 -> kbx = tid/2, kqs = 2*(tid%2),
// blocks_per_iter = 2*nwarps*64/4 = 32*nwarps. kby = kbx (qk/QK8_1 = 1).
// Per-call arithmetic (verbatim vec_dot_q4_0_q8_1_impl<2>):
//   sumi=0; sdot4(v0&F,u0); sdot4((v0>>4)&F,u1); sdot4(v1&F,u2); sdot4((v1>>4)&F,u3);
//   d4*(sumi*ds8f.x - 4*ds8f.y)   [8*vdr/QI4_0 = 4]
template <int ncols_dst, int nwarps_c, int rpb_c, int unroll_c>
__launch_bounds__(nwarps_c*64, 1)
static __global__ void mul_mat_vec_q_q40_d1(
        const void * __restrict__ vx_ptr, const void * __restrict__ vy_ptr, float * dst_ptr,
        const uint32_t ncols_x, const uint32_t stride_row_x, const uint32_t stride_col_y,
        const uint32_t stride_col_dst) {

    const int tid = 64*threadIdx.y + threadIdx.x;
    const int row0 = rpb_c*blockIdx.x;
    const int blocks_per_row_x = ncols_x / QK4_0;
    constexpr int blocks_per_iter = 32*nwarps_c;

    const block_q8_1 * y = (const block_q8_1 *) vy_ptr;
    float * dst = dst_ptr + row0;
    const int kbx_offset = row0*stride_row_x;

    const int kqs = 2*(tid % 2);

    float tmp[ncols_dst][rpb_c] = {{0.0f}};

#pragma unroll unroll_c
    for (int kbx = tid / 2; kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx;

        struct wdec { int a0, a1, a2, a3; float d4; };
        wdec w[rpb_c];
#pragma unroll
        for (int i = 0; i < rpb_c; ++i) {
            const block_q4_0 * bq4_0 = (const block_q4_0 *) vx_ptr + kbx_offset + i*stride_row_x + kbx;
            const int v0 = get_int_b2(bq4_0->qs, kqs + 0);
            const int v1 = get_int_b2(bq4_0->qs, kqs + 1);
            w[i].a0 = v0 & 0x0F0F0F0F;
            w[i].a1 = (v0 >> 4) & 0x0F0F0F0F;
            w[i].a2 = v1 & 0x0F0F0F0F;
            w[i].a3 = (v1 >> 4) & 0x0F0F0F0F;
            w[i].d4 = __half2float(bq4_0->d);
        }

#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const block_q8_1 * bq8_1 = y + j*stride_col_y + kby;
            const int u0 = get_int_b4(bq8_1->qs, kqs + 0);
            const int u1 = get_int_b4(bq8_1->qs, kqs + 0 + QI4_0);
            const int u2 = get_int_b4(bq8_1->qs, kqs + 1);
            const int u3 = get_int_b4(bq8_1->qs, kqs + 1 + QI4_0);
            const float2 ds8f = __half22float2(bq8_1->ds);
#pragma unroll
            for (int i = 0; i < rpb_c; ++i) {
                int sumi = 0;
                sumi = ggml_cuda_dp4a(w[i].a0, u0, sumi);
                sumi = ggml_cuda_dp4a(w[i].a1, u1, sumi);
                sumi = ggml_cuda_dp4a(w[i].a2, u2, sumi);
                sumi = ggml_cuda_dp4a(w[i].a3, u3, sumi);
                tmp[j][i] += w[i].d4 * (sumi * ds8f.x - 4 * ds8f.y);
            }
        }
    }

    __shared__ float tmp_shared[nwarps_c-1 > 0 ? nwarps_c-1 : 1][ncols_dst][rpb_c][64];

    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rpb_c; ++i) {
                tmp_shared[threadIdx.y-1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rpb_c; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps_c-1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum<64>(tmp[j][i]);
            if (threadIdx.x == i && (rpb_c == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                dst[j*stride_col_dst + i] = tmp[j][i];
            }
        }
    }
}
