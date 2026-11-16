// taskK3: LDS-staged canonical Q6_K narrow-batch mat-vec (the Q6_K lm_head twin of
// taskQ's rp_q4_nc_lds).
//
// Reference shape: mul_mat_vec_q<GGML_TYPE_Q6_K, T, false> on the GCN parameter
// table (widths 5..8: nwarps=1, rows_per_block=2, physical warp 64). This kernel
// keeps the SAME per-lane kbx mapping (kbx = lane/32 + iter*2, kqs = lane%32),
// the SAME vec_dot_q6_K_q8_1 calls, the SAME per-element accumulation order and
// the SAME warp_reduce_sum<64> tree => bit-identical to the generic mmvq path
// for every output element (the claim the microbench seals).
//
// What changes:
//  * each step's activation blocks are staged in LDS once per workgroup and
//    shared by all W warps (the generic path re-reads every token's q8_1 blocks
//    from L2 once per row: at verify width that is several GB per pass), and
//  * with STAGE_W=true the Q6_K row chunks are staged in LDS with coalesced
//    4-byte loads: the generic path reads ql/qh through get_int_b2 2-byte
//    strided accesses and the 210 B / 256-value block stride is sector-hostile
//    (a 64-lane wavefront fetches ~2x the useful bytes).
//
// Env gate at the dispatch site: GGML_CUDA_RP_Q6K_LDS (default 0, OFF) in mmvq.cu.
// Requires ne0 % 512 == 0 (even number of 256-value blocks) so every staged
// chunk is 4-byte aligned; the dispatch gate enforces it.
#ifndef GGML_CUDA_Q6K_LDS_CUH
#define GGML_CUDA_Q6K_LDS_CUH

#include "common.cuh"
#include "vecdotq.cuh"

template <int T, int W, int STEP, bool STAGE_W, int RPL = 2>
static __global__ void __launch_bounds__(64 * W) rp_q6k_nc_lds(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t srx, const uint32_t xs, const uint32_t ys) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    static_assert(STEP % 2 == 0, "Q6_K LDS twin stages 2 kbx per generic iteration");
    constexpr int BS = 210;  // sizeof(block_q6_K)
    static_assert((STEP * BS) % 4 == 0, "staged chunk must be 4-byte words");

    __shared__ block_q8_1 sq[T][STEP * 8];
    __shared__ uint       sw[W][RPL][STEP * BS / 4];

    const int lane = threadIdx.x;   // 0..63
    const int warp = threadIdx.y;   // 0..W-1
    const int row  = ((int) blockIdx.x * W + warp) * RPL;
    const uint32_t n_blocks = ne0 / 256;
    const int t32 = lane >> 5;
    const int kqs = lane & 31;

    float acc[T][RPL] = {};

    for (uint32_t s0 = 0; s0 < n_blocks; s0 += STEP) {
        const uint32_t nb_step = min((uint32_t) STEP, n_blocks - s0);
        const uint32_t nab     = nb_step * 8;   // activation blocks this step
        __syncthreads();                        // the previous step's LDS reads are done
        // Stage this step's activation blocks: T tokens x nab blocks of 36 B,
        // coalesced word copies within each token row.
        {
            const int  tid    = warp * 64 + lane;
            const uint32_t nw = nab * 9;       // 36 B = 9 words per block_q8_1
            for (int t = 0; t < T; ++t) {
                const uint * g   = reinterpret_cast<const uint *>(xq + (size_t) t * xs + (size_t) s0 * 8);
                uint       * lds = reinterpret_cast<uint *>(&sq[t][0]);
                for (uint32_t wi = tid; wi < nw; wi += 64 * W) lds[wi] = g[wi];
            }
        }
        if constexpr (STAGE_W) {
            // Stage this warp's RPL row chunks: nb_step*210 B per row. Row byte
            // stride srx*210 and even s0 keep every source word 4-byte aligned.
            // nb_step is always even here (n_blocks even via the ne0%512 gate and
            // STEP even), so nb_step*210 is an exact word count -- BS/4 would
            // drop the trailing 2 bytes (the d half) of every block.
            const uint32_t nw = (nb_step * BS) / 4;
            for (int r = 0; r < RPL; ++r) {
                const uint32_t rr  = min((uint32_t) row + r, ne1 - 1);
                const uint * g     = reinterpret_cast<const uint *>(wbase + ((size_t) rr * srx + s0) * BS);
                uint *       lds   = &sw[warp][r][0];
                for (uint32_t wi = lane; wi < nw; wi += 64) lds[wi] = g[wi];
            }
        }
        __syncthreads();                        // staged data visible
        for (uint32_t it = 0; it < nb_step; it += 2) {
            const int kbx_loc = (int) it + t32; // generic kbx = s0 + kbx_loc
#pragma unroll
            for (int j = 0; j < T; ++j) {
                const block_q8_1 * yb = &sq[j][kbx_loc * 8];
#pragma unroll
                for (int r = 0; r < RPL; ++r) {
                    if constexpr (STAGE_W) {
                        acc[j][r] += vec_dot_q6_K_q8_1(&sw[warp][r][0], yb, kbx_loc, kqs);
                    } else {
                        const uint32_t rr = min((uint32_t) row + r, ne1 - 1);
                        acc[j][r] += vec_dot_q6_K_q8_1(
                            wbase + (size_t) rr * srx * BS, yb, (int) s0 + kbx_loc, kqs);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < T; ++j) {
#pragma unroll
        for (int r = 0; r < RPL; ++r) {
            const float a = warp_reduce_sum<64>(acc[j][r]);
            if (lane == r && row + r < (int) ne1) {
                y[(size_t) j * ys + row + r] = a;
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, srx, xs, ys);
    NO_DEVICE_CODE;
#endif
}

// ---------------------------------------------------------------------------
// Host-side gate: GGML_CUDA_RP_Q6K_LDS=<2|4|8> selects the wave count W of the
// workgroup (default 0 = OFF, generic mmvq path byte-identical). The twin serves
// plain 2D Q6_K matmuls at verify widths 5..8 only (the widths where the GCN
// table gives mul_mat_vec_q<Q6_K,T> nwarps=1/rpb=2 geometry, which this kernel
// mirrors bit-exactly). STEP=4 x 256-value blocks per staging step, full staging
// (activations + coalesced Q6_K row chunks).
// ---------------------------------------------------------------------------
static int rp_q6k_lds_w() {
    static const int v = [] {
        const char * e = getenv("GGML_CUDA_RP_Q6K_LDS");
        const int r = e ? atoi(e) : 0;
        if (r == 2 || r == 4 || r == 8) {
            GGML_LOG_WARN("GGML_CUDA_RP_Q6K_LDS=%d ACTIVE (Q6_K narrow matvec LDS twin, widths 5..8)\n", r);
            return r;
        }
        return 0;
    }();
    return v;
}

static bool rp_q6k_lds_try_launch(
        ggml_type type, const void * vx, const void * vy, float * dst,
        int ncols_x, int nrows_x, int ncols_dst,
        int stride_row_x, int stride_col_y, int stride_col_dst,
        int nchannels_dst, int nsamples_dst, const int32_t * ids,
        const ggml_cuda_mm_fusion_args_device & fusion, cudaStream_t stream) {
    const int W = rp_q6k_lds_w();
    if (!W) {
        return false;
    }
    if (type != GGML_TYPE_Q6_K || ids != nullptr || ncols_dst < 5 || ncols_dst > 8) {
        return false;
    }
    if (nchannels_dst != 1 || nsamples_dst != 1 || (ncols_x % 512) != 0) {
        return false;
    }
    if ((stride_row_x % 2) != 0) {
        // odd row stride in blocks => staged source words not 4-byte aligned
        return false;
    }
    if (fusion.gate || fusion.x_bias || fusion.gate_bias ||
        fusion.x_scale || fusion.gate_scale || fusion.shared_up) {
        return false;
    }
    const dim3 grid((nrows_x + 2 * W - 1) / (2 * W), 1, 1);
    const dim3 block(64, W, 1);
    const uint8_t * w = (const uint8_t *) vx;
    const block_q8_1 * xq = (const block_q8_1 *) vy;
#define RP_Q6K_GO(T_, W_) \
    if (ncols_dst == T_ && W == W_) { \
        rp_q6k_nc_lds<T_, W_, 4, true, 2><<<grid, block, 0, stream>>>( \
            w, xq, dst, ncols_x, nrows_x, stride_row_x, stride_col_y, stride_col_dst); \
        return true; \
    }
#define RP_Q6K_T(T_) RP_Q6K_GO(T_, 2) RP_Q6K_GO(T_, 4) RP_Q6K_GO(T_, 8)
    RP_Q6K_T(5) RP_Q6K_T(6) RP_Q6K_T(7) RP_Q6K_T(8)
#undef RP_Q6K_T
#undef RP_Q6K_GO
    return false;
}

#endif // GGML_CUDA_Q6K_LDS_CUH