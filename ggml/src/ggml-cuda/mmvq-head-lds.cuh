// taskK2: LDS-staged Q6_K narrow mat-vec for the DFlash2 drafter vocab head.
//
// The pinned DFlash2 drafter (DFlash2-Q4_0-ownhead-q4.gguf) runs its 0.97 GiB
// output.weight (Q6_K, K=5120, N=248320) through the generic mul_mat_vec_q at
// draft widths: on the GCN table that is nwarps=1, rows_per_cuda_block=2, so a
// 64-thread block reads the whole q8_1 activation (T * K * 2 B, 28.8 KB at T=5)
// from L2 to produce just two rows - a 12:1 activation:weight L2 read ratio
// that pins the kernel at ~287 GB/s of weight streaming (~3.5 ms/step) against
// a ~1.02 TB/s HBM peak. This kernel stages each K-step's activation blocks in
// LDS once per workgroup and shares them across W waves x 2 rows, so the L2
// activation traffic falls by 2W while the weight stream is unchanged.
//
// BIT-IDENTITY: the per-row arithmetic replicates mul_mat_vec_q<Q6_K, T> on the
// GCN table (nwarps=1, rpb=2) exactly:
//   * same lane split: lanes 0-31 take even superblocks (kbx = s), lanes 32-63
//     take odd (kbx = s+1), kqs = lane & 31, kbx ascending by 2;
//   * same vec_dot_q6_K_q8_1 call on the same bytes (weights stream from global
//     exactly as the generic does; only the activation pointer moves to LDS);
//   * same accumulation: acc[r][j] += vec_dot per (kbx, j), fp32, same order;
//   * same reduction: warp_reduce_sum<64> (shfl_xor 32,16,8,4,2,1).
// Rows are independent, so the rows-per-block / waves-per-block geometry is
// free; the row -> result mapping is a bijection of the generic's.
// Verified bit-exact in mb_k2_head_test at T=2..5, N=248320/124160, W=4/8.
//
// Dispatch gate: GGML_DFLASH_HEAD_KERNEL=1 (default OFF) at the
// ggml_cuda_mul_mat_vec_q funnel in mmvq.cu; GGML_DFLASH_HEAD_W selects the
// waves per workgroup (4 or 8, default 8). Gate-off never reaches this code.
//
// Scope: Q6_K, no ids, no fusion, widths T in 2..5 (the generic uses a
// different, 2-warp geometry at T=1 - deliberately not covered), K=5120,
// N=248320 (the drafter head shape; the TP2 target head is N=124160 per card
// and stays on the generic path).

#pragma once

#include "common.cuh"
#include "vecdotq.cuh"

template <int T, int W>
static __global__ void __launch_bounds__(64 * W) mmvq_head_q6k_lds(
        const void * __restrict__ vx_ptr, const void * __restrict__ vy_ptr,
        float * __restrict__ dst,
        const uint32_t nrows_x, const uint32_t stride_row_x,
        const uint32_t stride_col_y, const uint32_t stride_col_dst) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    static_assert(sizeof(block_q8_1) % 4 == 0, "stage copy assumes 4-byte multiples");

    const block_q6_K * __restrict__ vx = (const block_q6_K *) vx_ptr;
    const block_q8_1 * __restrict__ vy = (const block_q8_1 *) vy_ptr;

    // one K-step = 2 superblocks (512 values); 8 q8_1 blocks per superblock per column
    __shared__ block_q8_1 stage[T][2][8];

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp * 64 + lane;
    const int row  = ((int) blockIdx.x * W + warp) * 2;

    const uint32_t nkb = stride_row_x;          // superblocks per row
    const int      kb0 = lane >> 5;             // 0: lanes 0-31 even sb, 1: lanes 32-63 odd sb
    const int      kqs = lane & 31;

    uint32_t rr[2];
    bool     rv[2];
    float    acc[2][T] = {};
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        rv[r] = row + r < (int) nrows_x;
        rr[r] = rv[r] ? (uint32_t) (row + r) : 0;
    }

    for (uint32_t s = 0; s < nkb; s += 2) {
        __syncthreads();                        // previous step's LDS reads are done
        for (int i = tid; i < 16 * T; i += 64 * W) {
            const int b  = i & 7, kb = (i >> 3) & 1, j = i >> 4;
            const uint32_t kbx = s + kb;
            if (kbx < nkb) {
                stage[j][kb][b] = vy[(size_t) j * stride_col_y + (size_t) kbx * 8 + b];
            }
        }
        __syncthreads();

        const int kbx = (int) (s + kb0);
        if (kbx < (int) nkb) {
#pragma unroll
            for (int j = 0; j < T; ++j) {
                const block_q8_1 * bq8 = &stage[j][kb0][0];
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    if (r == 1 && !rv[1]) {
                        break;
                    }
                    acc[r][j] += vec_dot_q6_K_q8_1(vx + (size_t) rr[r] * stride_row_x, bq8, kbx, kqs);
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        if (!rv[r]) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < T; ++j) {
            const float a = warp_reduce_sum<64>(acc[r][j]);
            if (lane == 0) {
                dst[(size_t) j * stride_col_dst + row + r] = a;
            }
        }
    }
#else
    GGML_UNUSED_VARS(vx_ptr, vy_ptr, dst, nrows_x, stride_row_x, stride_col_y, stride_col_dst);
    NO_DEVICE_CODE;
#endif
}

static void mmvq_head_q6k_lds_launch(
        const void * vx, const void * vy, float * dst,
        const int64_t K, const int64_t N, const int64_t T, const int W,
        const int64_t stride_col_y, const int64_t stride_col_dst, cudaStream_t stream) {
    const int64_t rows_per_block = 2 * W;
    dim3 grid((unsigned) ((N + rows_per_block - 1) / rows_per_block));
    dim3 block(64, (unsigned) W);
    const uint32_t srow = (uint32_t) (K / 256);
    switch (T) {
        case 2:
            if (W == 4) mmvq_head_q6k_lds<2, 4><<<grid, block, 0, stream>>>(vx, vy, dst, (uint32_t) N, srow, (uint32_t) stride_col_y, (uint32_t) stride_col_dst);
            else        mmvq_head_q6k_lds<2, 8><<<grid, block, 0, stream>>>(vx, vy, dst, (uint32_t) N, srow, (uint32_t) stride_col_y, (uint32_t) stride_col_dst);
            break;
        case 3:
            if (W == 4) mmvq_head_q6k_lds<3, 4><<<grid, block, 0, stream>>>(vx, vy, dst, (uint32_t) N, srow, (uint32_t) stride_col_y, (uint32_t) stride_col_dst);
            else        mmvq_head_q6k_lds<3, 8><<<grid, block, 0, stream>>>(vx, vy, dst, (uint32_t) N, srow, (uint32_t) stride_col_y, (uint32_t) stride_col_dst);
            break;
        case 4:
            if (W == 4) mmvq_head_q6k_lds<4, 4><<<grid, block, 0, stream>>>(vx, vy, dst, (uint32_t) N, srow, (uint32_t) stride_col_y, (uint32_t) stride_col_dst);
            else        mmvq_head_q6k_lds<4, 8><<<grid, block, 0, stream>>>(vx, vy, dst, (uint32_t) N, srow, (uint32_t) stride_col_y, (uint32_t) stride_col_dst);
            break;
        case 5:
            if (W == 4) mmvq_head_q6k_lds<5, 4><<<grid, block, 0, stream>>>(vx, vy, dst, (uint32_t) N, srow, (uint32_t) stride_col_y, (uint32_t) stride_col_dst);
            else        mmvq_head_q6k_lds<5, 8><<<grid, block, 0, stream>>>(vx, vy, dst, (uint32_t) N, srow, (uint32_t) stride_col_y, (uint32_t) stride_col_dst);
            break;
        default:
            GGML_ABORT("mmvq_head_q6k_lds: unsupported width %lld", (long long) T);
    }
}
