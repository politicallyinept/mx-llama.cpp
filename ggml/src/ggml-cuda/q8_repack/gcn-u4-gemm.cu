// Tier-3.7b: gfx906 tiled GEMM for repacked-u4 (Q4_0) weights — see
// gcn-u4-gemm.cuh for scope and the re-seal note.
//
// Structure follows gcn_q8_gemm / gcn_kq4_gemm (gcn-q8-gemm.cu): a 256-thread
// block computes 128 weight rows x 64 tokens, each thread an 8 x 4 register
// tile (rows rg + 16*i, cols cg + 16*j). Per 32-value block the thread keeps
// one integer dp4a sum and applies the block scales once in the affine Q4_0
// form acc += d_w * (d_x * idot - 8 * sx) (identical algebra to the generic
// loop's dw2.x*dx*idot - dw2.y*sx; the fma association differs — re-seal).
// Rows of a thread are interleaved (row group + 16*i) and LDS rows are padded
// to 9 dwords so the 16 row groups of a wave hit 16 different banks. The next
// K step (4 blocks) is fetched into registers while the current one computes
// from LDS. Wave-quantization-aware: launch_bounds(256, 2) puts exactly 2
// blocks per gfx906 CU (64-lane waves), LDS 31744 B/block <= 32 KiB.
//
// Weight side: repacked Q4_0 = de-aliased nibble rows (16 B per 32-value
// block, byte i = value i | value(16+i) << 4, excess-8 codes) + an f16 scale
// plane (rp_traits<Q4_0> layout). Nibbles are expanded to int8 at LDS store
// (the generic GEMM's expanded-staging lesson: expand-at-consume measured -40%).
// Activation side: the existing block_q8_1_mmq DS4 pool (quantize_mmq_q8_1 with
// DS4 for Q4_0): per (token, 4-block group) { half2 ds4[4] = {d, sum_x}; qs }.
// One K step (4 blocks) is exactly one mmq group: contiguous 144 B per token.
#include "gcn-u4-gemm.cuh"

#include "repack-common.cuh"
#include "repack-kernels.cuh"   // repack_tile_meta
#include "ggml-cuda.h"         // GGML_LOG_WARN for the gate-engagement marker

#include <cstdlib>

bool ggml_cuda_repack_q4_0_tiled_on() {
    static const bool s_on = [] {
        const char * e = getenv("GGML_CUDA_REPACK_Q4_0_TILED");
        const bool on = e != nullptr && atoi(e) != 0;
        if (on) {
            GGML_LOG_WARN("CUDA repack: GGML_CUDA_REPACK_Q4_0_TILED gate ACTIVE: repacked Q4_0 GEMMs run the gfx906 tiled gcn_u4_gemm (128x64 tiles, LDS-staged). Tile order changes FP rounding vs the generic MMQ loop: seal hashes shift, re-seal required\n");
        }
        return on;
    }();
    return s_on;
}

#if defined(GGML_USE_HIP) && defined(__gfx906__)

static constexpr int GU4_BM  = 128; // weight rows per block
static constexpr int GU4_BN  = 64;  // tokens per block
static constexpr int GU4_NT  = 256;
static constexpr int GU4_KB  = 4;   // 32-value blocks per K step (= one mmq group)
static constexpr int GU4_PAD = 9;   // dwords per LDS row (8 + 1)

// MOE: MUL_MAT_ID grouped by expert, tiles from repack_tile_off<GU4_BN>.
// blockIdx.x = row tile, blockIdx.y = tile index (loose grid, tile_off guard).
// Dense: blockIdx.x = token tile (fastest, so blocks sharing a weight tile run
// together and all but the first read it from L2), blockIdx.y = row tile.
template <bool MOE>
__launch_bounds__(GU4_NT, 2)
static __global__ void gcn_u4_gemm(const uint8_t * __restrict__ wbase,
        const block_q8_1_mmq_ds_h * __restrict__ xmmq, float * __restrict__ dst,
        const int M, const int N, const int K, const int n_groups, const uint32_t dst_s1,
        const size_t expert_stride,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const repack_tile_meta * __restrict__ tile_meta, const uint32_t n_expert) {
    int a_base = 0, ncols = 0;
    if constexpr (MOE) {
        if ((int) blockIdx.y >= tile_off[n_expert]) {
            return;
        }
        const repack_tile_meta meta = tile_meta[blockIdx.y];
        wbase += (size_t) meta.expert * expert_stride;
        a_base = (int) meta.a_base;
        ncols  = expert_bounds[meta.expert + 1] - a_base;
    }
    const int row0 = MOE ? (int) blockIdx.x * GU4_BM : (int) blockIdx.y * GU4_BM;
    const int col0 = MOE ? 0 : (int) blockIdx.x * GU4_BN;

    const int n_sub  = K >> 5;                    // 32-value blocks along K
    const int nstep  = (n_sub + GU4_KB - 1) / GU4_KB;

    __shared__ __align__(16) int    aq[GU4_KB][GU4_BM][GU4_PAD]; // expanded int8 weights
    __shared__             float    ad[GU4_KB][GU4_BM];          // weight f16 scale d
    __shared__ __align__(16) int    bq[GU4_KB][GU4_BN][GU4_PAD]; // activation qs
    __shared__             float2   bd[GU4_KB][GU4_BN];          // activation (d, sum_x)

    const int tid = threadIdx.x;

    // ---- loader roles -------------------------------------------------------
    // A: row tid/2, blocks 2*(tid%2) and +1 of the step: two contiguous 16 B
    //    nibble blocks (one uint4 pair) + the two f16 scales (two u16 loads;
    //    a u32 pair load would need an even n_sub).
    const int  la_row = tid >> 1;
    const int  la_h   = tid & 1;
    const typename rp_traits<GGML_TYPE_Q4_0>::geom wg(K, M);
    const uint4 * const qsp = reinterpret_cast<const uint4 *>(wbase);
    const int   wrow  = min(row0 + la_row, M - 1);
    const uint4 * const wa  = qsp + (size_t) wrow * wg.rs_u4 + 2 * la_h;
    const uint16_t * const wd = reinterpret_cast<const uint16_t *>(wbase + wg.dplane_off)
                              + (size_t) wrow * wg.n_sub + 2 * la_h;
    // B: column tid/4, block tid%4 of the step: one 32 B qs slice (2 uint4,
    //    16 B aligned inside the group) + the 4 B ds4 slot.
    const int  lb_col = tid >> 2;
    const int  lb_kb  = tid & 3;
    int   xrow;
    bool  xvalid;
    if constexpr (MOE) {
        const int a = a_base + lb_col;
        xvalid = a < a_base + ncols;
        xrow   = ids_src1[xvalid ? a : (a_base + ncols - 1)];
    } else {
        xvalid = col0 + lb_col < N;
        xrow   = min(col0 + lb_col, N - 1);
    }
    const block_q8_1_mmq_ds_h * const yb = xmmq + (size_t) xrow * n_groups;

    // ---- compute roles ------------------------------------------------------
    const int rg = tid % 16;   // rows rg + 16*i
    const int cg = tid / 16;   // cols cg + 16*j

    float acc[8][4] = {};

    int ra[9];   // 8 qs dwords + packed scales (s0 | s1<<16)
    int rb[9];   // ds4 slot + 8 qs dwords
    auto fetch = [&](const int step) {
        const int sb0 = step * GU4_KB + 2 * la_h;
        if (sb0 + 1 < n_sub) {
            const uint4 p0 = rp_ldcs_u4(wa + (size_t) step * GU4_KB);
            const uint4 p1 = rp_ldcs_u4(wa + (size_t) step * GU4_KB + 1);
            ra[0] = p0.x; ra[1] = p0.y; ra[2] = p0.z; ra[3] = p0.w;
            ra[4] = p1.x; ra[5] = p1.y; ra[6] = p1.z; ra[7] = p1.w;
            ra[8] = (int) (((uint32_t) wd[1] << 16) | (uint32_t) wd[0]);
        } else if (sb0 < n_sub) {
            const uint4 p0 = rp_ldcs_u4(wa + (size_t) step * GU4_KB);
            ra[0] = p0.x; ra[1] = p0.y; ra[2] = p0.z; ra[3] = p0.w;
            ra[4] = 0; ra[5] = 0; ra[6] = 0; ra[7] = 0;
            ra[8] = (int) (uint32_t) wd[0];
        } else {
#pragma unroll
            for (int i = 0; i < 9; ++i) {
                ra[i] = 0;
            }
        }
        const block_q8_1_mmq_ds_h * gb = yb + step;
        if (xvalid && step * GU4_KB + lb_kb < n_sub) {
            rb[0] = (int) *reinterpret_cast<const uint32_t *>(&gb->ds4[lb_kb]);
            const uint4 * pb = reinterpret_cast<const uint4 *>(gb->qs + lb_kb * QK8_1);
            const uint4 q0 = rp_ldcs_u4(pb);
            const uint4 q1 = rp_ldcs_u4(pb + 1);
            rb[1] = q0.x; rb[2] = q0.y; rb[3] = q0.z; rb[4] = q0.w;
            rb[5] = q1.x; rb[6] = q1.y; rb[7] = q1.z; rb[8] = q1.w;
        } else {
#pragma unroll
            for (int i = 0; i < 9; ++i) {
                rb[i] = 0;
            }
        }
    };
    auto store = [&]() {
        // nibble expand: byte i = value i (low) | value(16+i) (high); dwords
        // 0..3 = values 0..15, dwords 4..7 = values 16..31 — the same dword
        // order as the activation qs, so dp4a pairs align.
        const int kb0 = 2 * la_h;
        aq[kb0    ][la_row][0] = ra[0] & 0x0F0F0F0F;
        aq[kb0    ][la_row][1] = ra[1] & 0x0F0F0F0F;
        aq[kb0    ][la_row][2] = ra[2] & 0x0F0F0F0F;
        aq[kb0    ][la_row][3] = ra[3] & 0x0F0F0F0F;
        aq[kb0    ][la_row][4] = (ra[0] >> 4) & 0x0F0F0F0F;
        aq[kb0    ][la_row][5] = (ra[1] >> 4) & 0x0F0F0F0F;
        aq[kb0    ][la_row][6] = (ra[2] >> 4) & 0x0F0F0F0F;
        aq[kb0    ][la_row][7] = (ra[3] >> 4) & 0x0F0F0F0F;
        aq[kb0 + 1][la_row][0] = ra[4] & 0x0F0F0F0F;
        aq[kb0 + 1][la_row][1] = ra[5] & 0x0F0F0F0F;
        aq[kb0 + 1][la_row][2] = ra[6] & 0x0F0F0F0F;
        aq[kb0 + 1][la_row][3] = ra[7] & 0x0F0F0F0F;
        aq[kb0 + 1][la_row][4] = (ra[4] >> 4) & 0x0F0F0F0F;
        aq[kb0 + 1][la_row][5] = (ra[5] >> 4) & 0x0F0F0F0F;
        aq[kb0 + 1][la_row][6] = (ra[6] >> 4) & 0x0F0F0F0F;
        aq[kb0 + 1][la_row][7] = (ra[7] >> 4) & 0x0F0F0F0F;
        ad[kb0    ][la_row] = __half2float(__ushort_as_half((unsigned short) (ra[8] & 0xFFFF)));
        ad[kb0 + 1][la_row] = __half2float(__ushort_as_half((unsigned short) ((uint32_t) ra[8] >> 16)));
        bq[lb_kb][lb_col][0] = rb[1];
        bq[lb_kb][lb_col][1] = rb[2];
        bq[lb_kb][lb_col][2] = rb[3];
        bq[lb_kb][lb_col][3] = rb[4];
        bq[lb_kb][lb_col][4] = rb[5];
        bq[lb_kb][lb_col][5] = rb[6];
        bq[lb_kb][lb_col][6] = rb[7];
        bq[lb_kb][lb_col][7] = rb[8];
        bd[lb_kb][lb_col] = __half22float2(*reinterpret_cast<const half2 *>(&rb[0]));
    };

    fetch(0);
    for (int step = 0; step < nstep; ++step) {
        store();
        __syncthreads();
        if (step + 1 < nstep) {
            fetch(step + 1);
        }
#pragma unroll
        for (int kb = 0; kb < GU4_KB; ++kb) {
            // rows in two halves of 4: 16 live integer sums instead of 32
            // (fewer VGPRs, more waves), B read twice (gcn_q8_gemm GQ_HALVES).
            float db[4], m8s[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = bd[kb][cg + 16*j];
                db[j]  = f.x;
                m8s[j] = -8.0f * f.y;   // -8 * sum_x (DS4 slot, fp16-rounded as shipped)
            }
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int s[4][4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        s[i][j] = 0;
                    }
                }
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
                    int a[4], b[4];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        a[i] = aq[kb][rg + 16*(4*h + i)][kk];
                    }
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        b[j] = bq[kb][cg + 16*j][kk];
                    }
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            s[i][j] = ggml_cuda_dp4a(a[i], b[j], s[i][j]);
                        }
                    }
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float da = ad[kb][rg + 16*(4*h + i)];
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        // acc += d_w * (d_x * idot - 8 * sum_x)
                        acc[4*h + i][j] = fmaf(da, fmaf(db[j], (float) s[i][j], m8s[j]),
                                               acc[4*h + i][j]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float * colout;
        if constexpr (MOE) {
            const int a = a_base + cg + 16*j;
            if (a >= a_base + ncols) {
                continue;
            }
            colout = dst + (size_t) ids_dst[a] * dst_s1;
        } else {
            const int col = col0 + cg + 16*j;
            if (col >= N) {
                continue;
            }
            colout = dst + (size_t) col * dst_s1;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = row0 + rg + 16*i;
            if (row < M) {
                colout[row] = acc[i][j];
            }
        }
    }
}

void ggml_cuda_gcn_u4_gemm_dense(const uint8_t * wbase, const block_q8_1_mmq_ds_h * xmmq,
        float * dst, int M, int N, int K, int n_groups, uint32_t dst_s1,
        cudaStream_t stream) {
    const dim3 grid((N + GU4_BN - 1) / GU4_BN, (M + GU4_BM - 1) / GU4_BM, 1);
    gcn_u4_gemm<false><<<grid, GU4_NT, 0, stream>>>(wbase, xmmq, dst, M, N, K, n_groups,
            dst_s1, 0, nullptr, nullptr, nullptr, nullptr, nullptr, 0);
}

void ggml_cuda_gcn_u4_gemm_moe(const uint8_t * wbase, const block_q8_1_mmq_ds_h * xmmq,
        float * dst, int M, int K, int n_groups, uint32_t dst_s1,
        const int32_t * ids_src1, const int32_t * ids_dst, const int32_t * expert_bounds,
        const int32_t * tile_off, const repack_tile_meta * tile_meta, uint32_t n_expert,
        size_t expert_stride, int max_tiles, cudaStream_t stream) {
    const dim3 grid((M + GU4_BM - 1) / GU4_BM, max_tiles, 1);
    gcn_u4_gemm<true><<<grid, GU4_NT, 0, stream>>>(wbase, xmmq, dst, M, 0, K, n_groups,
            dst_s1, expert_stride, ids_src1, ids_dst, expert_bounds, tile_off, tile_meta,
            n_expert);
}

#else  // !HIP-gfx906

void ggml_cuda_gcn_u4_gemm_dense(const uint8_t *, const block_q8_1_mmq_ds_h *, float *,
        int, int, int, int, uint32_t, cudaStream_t) {
    GGML_ABORT("gcn_u4_gemm is gfx906-only");
}

void ggml_cuda_gcn_u4_gemm_moe(const uint8_t *, const block_q8_1_mmq_ds_h *, float *,
        int, int, int, uint32_t, const int32_t *, const int32_t *, const int32_t *,
        const int32_t *, const repack_tile_meta *, uint32_t, size_t, int, cudaStream_t) {
    GGML_ABORT("gcn_u4_gemm is gfx906-only");
}

#endif
