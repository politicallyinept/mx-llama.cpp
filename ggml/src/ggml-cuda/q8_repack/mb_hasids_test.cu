// mb_hasids_test.cu - E4 re-scoped step 1c (task E6): HAS_IDS (mul_mat_id) Q8_0 expert
// GEMM bit-diff: repacked mmq_gemm_repacked*<true,...> vs stock mmq-id (mul_mat_q with
// ids_dst/expert_bounds + MoE tile map), on IDENTICAL input bytes at routed-token widths.
//
// Feeds the same logical matrix (src1 fp32 -> q8_1 D4 bytes quantized ONCE on the host,
// bit-mirror of quantize_mmq_q8_1<DS_LAYOUT_D4>, quantize.cu:908) and the same Q8_0
// expert weights (canonical blocks for stock, repack_q8_0_host byte permutation for the
// repacked arm) through both kernels at every shape.
//
// Stock side mirrors ggml_cuda_mul_mat_q's HAS_IDS branch (mmq.cu:382-510) for gfx906
// Q8_0: J sweep with ncols_opt=ne12 and the ids J>64 skip (mmq.cu:246), the GCN short-K
// MoE J=32 override (mmq.cuh mul_mat_q_switch_J), and the non-stream-K launch with the
// MoE tile map (mmq.cuh launch_mul_mat_q, default ON). ids_src1/ids_dst/expert_bounds
// are computed on the host with the exact semantics of mm_ids_helper (mmid.cu):
//   ids_dst[compact] = original assignment (it*n_expert_used + iex)
//   ids_src1[compact] = src1 column (forward map, write_inverse=false)
//   expert_bounds = exclusive prefix of per-expert counts
// Activation buffers mirror the runtime: the stock arm gets y in COMPACT order
// (row a = q8_1 of src1 column ids_src1[a], as quantize_mmq_q8_1_cuda with ids_src1
// produces); the repacked arm gets xq in ORIGINAL order (quantize with ids=nullptr,
// mul-mat-id.cu:441) which the kernel indexes through ids_src1 (sCol[]).
//
// Repacked side mirrors q8_repack/mul-mat-id.cu GEMM arm: repack_tile_off<BN> +
// mmq_gemm_repacked_w32<true,1,NRL*2,Q8_0> (BN=32) and mmq_gemm_repacked<true,TN,NRL,
// Q8_0> (BN=64*TN=128); BOTH variants are launched at every shape (the runtime picks
// w32 whenever n_assign < 2*BN_ID*ne02, which is every width in this grid).
//
// Per-output fp32 bit-diff + launch determinism (each arm launched twice) + independent
// ascending-K fmaf CPU reference (gated by size).
//
// Build (TheRock 7.15, source /root/w4a4-env.sh):
//   hipcc -O3 -std=c++17 --offload-arch=gfx906 -DGGML_USE_HIP \
//     -Iggml/include -Iggml/src -Iggml/src/ggml-cuda \
//     ggml/src/ggml-cuda/q8_repack/mb_hasids_test.cu -o mb_hasids
// Run on ONE idle card under flock (GPU2 by convention):
//   flock -w 600 /root/gpu.lock env HIP_VISIBLE_DEVICES=2 ./mb_hasids
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <climits>
#include <cmath>
#include <vector>
#include <random>
#include <cstdarg>
#include <hip/hip_runtime.h>

#include "ggml.h"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mmq.cuh"
#include "q8_repack/repack-kernels.cuh"

// Link stub: the standalone binary does not link libggml; GGML_ABORT needs this.
void ggml_abort(const char* file, int line, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "ABORT %s:%d: ", file, line);
    vfprintf(stderr, fmt, ap); va_end(ap);
    fprintf(stderr, "\n"); exit(3);
}

#define CK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// ---------------------------------------------------------------------------
// Host bit-mirror of quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, false> (quantize.cu:465).
// Same as mb_e4_gemm_test.cu (validated bit-identical to the kernel in E1).
// Layout: y[kb*ncols + j] = { float d4[4]; int8 qs[128] }.
// ---------------------------------------------------------------------------
static void quantize_mmq_q8_1_host(const float * x, int64_t K, int64_t N,
                                   block_q8_1_mmq * y) {
    const int64_t n_groups = K / QK8_1_MMQ;
    for (int64_t kb = 0; kb < n_groups; ++kb) {
        for (int64_t j = 0; j < N; ++j) {
            block_q8_1_mmq & blk = y[kb*N + j];
            for (int k = 0; k < 4; ++k) {
                const float * p = x + (size_t) j*K + kb*QK8_1_MMQ + k*32;
                float amax = 0.0f;
                for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(p[i]));
                const float d_inv = 127.0f / amax;
                int8_t * qs = &blk.qs[k*32];
                for (int i = 0; i < 32; ++i) qs[i] = (int8_t) roundf(p[i]*d_inv);
                blk.d4[k] = 1.0f / d_inv;
            }
        }
    }
}

// Byte-permutation repack of one expert's Q8_0 blocks (repack_q8_0_host semantics,
// q8_repack/repack-common.cu:350). dst must be pre-zeroed to repack_gcn_nbytes bytes.
static void repack_q8_0_host_local(const block_q8_0 * blocks, uint8_t * dst,
                                   const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 32;
    const int64_t qs_str   = repack_qs_stride(ne0);
    const size_t  qs_len   = (size_t) ne1 * qs_str;
    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q8_0 * b = &blocks[row * n_blocks + blk];
            memcpy(dst + (size_t) row * qs_str + (size_t) blk * 32, b->qs, 32);
            memcpy(dst + qs_len + (size_t)(row * n_blocks + blk) * 2, &b->d, 2);
        }
    }
}

// Host mirror of mm_ids_helper (mmid.cu) with write_inverse=false, nchannels_y=e_used,
// sis1=e_used: stable counting sort by expert, token order within expert.
static void mm_ids_helper_host(const std::vector<int32_t> & ids, int n_tokens, int e_used, int n_experts,
                               std::vector<int32_t> & ids_src1, std::vector<int32_t> & ids_dst,
                               std::vector<int32_t> & bounds) {
    const int n_assign = n_tokens * e_used;
    bounds.assign(n_experts + 1, 0);
    for (int it = 0; it < n_tokens; ++it)
        for (int iex = 0; iex < e_used; ++iex)
            bounds[ids[it*e_used + iex] + 1]++;
    for (int e = 0; e < n_experts; ++e) bounds[e+1] += bounds[e];
    ids_src1.assign(n_assign, 0);
    ids_dst.assign(n_assign, 0);
    std::vector<int> cur(bounds.begin(), bounds.end() - 1);
    for (int it = 0; it < n_tokens; ++it) {
        for (int iex = 0; iex < e_used; ++iex) {
            const int e   = ids[it*e_used + iex];
            const int pos = cur[e]++;
            ids_dst[pos]  = it*e_used + iex;              // compact -> original assignment
            ids_src1[pos] = it*e_used + iex;              // forward map (sis1 = e_used)
        }
    }
}

// ---------------------------------------------------------------------------
// Stock J selection: mirror of ggml_cuda_mul_mat_q's sweep for HAS_IDS gfx906 Q8_0
// (mmq.cu:212-256) + the GCN short-K MoE override (mmq.cuh:1585-1593).
// ---------------------------------------------------------------------------
static int pick_J_stock_ids(int M, int n_tokens, int n_assign, int n_expert, int K,
                            int nsm, size_t smpbo) {
    GGML_UNUSED(nsm);
    const bool fallback = ggml_cuda_mmq_needs_fallback(M);
    int J_best = 0, ntiles_best = INT_MAX;
    for (int J = 8; J <= 128 && ntiles_best > 1; J += 8) {
        const ggml_cuda_mmq_config cfg =
            ggml_cuda_mmq_get_config(GGML_TYPE_Q8_0, J, fallback, GGML_CUDA_CC_VEGA20, GGML_PREC_Q8);
        if (cfg.type == GGML_TYPE_COUNT) continue;
        if (mmq_get_nbytes_shared(cfg, GGML_CUDA_CC_VEGA20) > smpbo) continue;
        const int ntiles_x = (n_tokens + cfg.J - 1) / cfg.J;   // ncols_opt = ne12
        if (cfg.J > 64) continue;   // gfx906 MoE: ids never takes wide J (mmq.cu:246-248)
        if (ntiles_x < ntiles_best) { J_best = J; ntiles_best = ntiles_x; }
    }
    // short-K MoE override: GCN && expert_bounds && K<2048 && n_assign <= 16*n_experts
    if (K < 2048 && n_assign <= 16*n_expert &&
            ggml_cuda_mmq_get_config(GGML_TYPE_Q8_0, 32, fallback, GGML_CUDA_CC_VEGA20, GGML_PREC_Q8).type != GGML_TYPE_COUNT) {
        J_best = 32;
    }
    return J_best;
}

// ---------------------------------------------------------------------------
// Stock arm: mul_mat_q with ids_dst/expert_bounds + MoE tile map (non-stream-K,
// mirror of launch_mul_mat_q's !stream_k branch, mmq.cuh:1505-1528).
// ---------------------------------------------------------------------------
template <int J, bool FB>
static void launch_stock_ids(int M, int K, int n_tokens, int e_used, int n_expert, int n_assign,
                             const char * x, const int * y, const int32_t * ids_dst, const int32_t * bounds,
                             float * dst, size_t smpbo, cudaStream_t st) {
    const ggml_cuda_mmq_config cfg =
        ggml_cuda_mmq_get_config(GGML_TYPE_Q8_0, J, FB, GGML_CUDA_CC_VEGA20, GGML_PREC_Q8);
    const int nbytes  = mmq_get_nbytes_shared(cfg, GGML_CUDA_CC_VEGA20);
    GGML_ASSERT((size_t) nbytes <= smpbo);
    const int nwarps  = cfg.nthreads / 64;
    const int nty     = (M + cfg.I - 1) / cfg.I;
    const int ntx     = (n_tokens + cfg.J - 1) / cfg.J;               // ncols_max = ne12
    const int n_tiles_ub = (n_assign + cfg.J - 1) / cfg.J + n_expert; // tile-map grid.y

    int2 * tile_map = nullptr;
    CK(hipMalloc(&tile_map, n_tiles_ub * sizeof(int2)));
    mmq_build_tile_map<<<1, MMQ_TILE_MAP_THREADS, 0, st>>>(bounds, n_expert, cfg.J, tile_map, n_tiles_ub);

    const int64_t ne10_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
    const int workspace_s12 = (int) ((int64_t) e_used * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int)));
    const int workspace_s13 = n_tokens * workspace_s12;

    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<GGML_TYPE_Q8_0, J, FB, GGML_PREC_Q8>), nbytes);
    mul_mat_q<GGML_TYPE_Q8_0, J, FB, GGML_PREC_Q8><<<dim3(nty, n_tiles_ub, 1), dim3(64, nwarps, 1), nbytes, st>>>(
        x, y, ids_dst, bounds, dst, nullptr, nullptr,
        init_fastdiv_values(K / 32),
        M, n_assign, K / 32, n_assign, M,
        init_fastdiv_values(1), init_fastdiv_values(n_expert), (int) ((int64_t) M * (K / 32)), workspace_s12, (int) ((int64_t) M * e_used),
        init_fastdiv_values(1), init_fastdiv_values(1), 0, workspace_s13, 0,
        init_fastdiv_values(ntx), tile_map);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    hipFree(tile_map);
}

template <bool FB>
static void launch_stock_ids_J(int J, int M, int K, int n_tokens, int e_used, int n_expert, int n_assign,
                               const char * x, const int * y, const int32_t * ids_dst, const int32_t * bounds,
                               float * dst, size_t smpbo, cudaStream_t st) {
    switch (J) {
        case 8:   launch_stock_ids<8,   FB>(M, K, n_tokens, e_used, n_expert, n_assign, x, y, ids_dst, bounds, dst, smpbo, st); break;
        case 16:  launch_stock_ids<16,  FB>(M, K, n_tokens, e_used, n_expert, n_assign, x, y, ids_dst, bounds, dst, smpbo, st); break;
        case 24:  launch_stock_ids<24,  FB>(M, K, n_tokens, e_used, n_expert, n_assign, x, y, ids_dst, bounds, dst, smpbo, st); break;
        case 32:  launch_stock_ids<32,  FB>(M, K, n_tokens, e_used, n_expert, n_assign, x, y, ids_dst, bounds, dst, smpbo, st); break;
        case 40:  launch_stock_ids<40,  FB>(M, K, n_tokens, e_used, n_expert, n_assign, x, y, ids_dst, bounds, dst, smpbo, st); break;
        case 48:  launch_stock_ids<48,  FB>(M, K, n_tokens, e_used, n_expert, n_assign, x, y, ids_dst, bounds, dst, smpbo, st); break;
        case 56:  launch_stock_ids<56,  FB>(M, K, n_tokens, e_used, n_expert, n_assign, x, y, ids_dst, bounds, dst, smpbo, st); break;
        case 64:  launch_stock_ids<64,  FB>(M, K, n_tokens, e_used, n_expert, n_assign, x, y, ids_dst, bounds, dst, smpbo, st); break;
        default: fprintf(stderr, "unhandled J=%d\n", J); exit(2);
    }
}

// ---------------------------------------------------------------------------
// Repacked arm: mirror of q8_repack/mul-mat-id.cu GEMM dispatch (Q8_0 case).
// w32 variant: repack_tile_off<32> + mmq_gemm_repacked_w32<true,1,NRL*2,Q8_0>.
// 64-wide:     repack_tile_off<64*TN> + mmq_gemm_repacked<true,TN,NRL,Q8_0>.
// ---------------------------------------------------------------------------
static void launch_repack_ids(bool w32, int M, int K, int n_expert, int n_assign, size_t expert_stride,
                              const uint8_t * w, const block_q8_1 * xq,
                              const int32_t * ids_src1, const int32_t * ids_dst, const int32_t * bounds,
                              float * dst, cudaStream_t st) {
    const int64_t max_tiles = n_assign / (w32 ? 32 : 64*MMQ_RP_Q8_TN) + n_expert;
    int32_t * tile_off = nullptr;
    repack_tile_meta * tile_meta = nullptr;
    CK(hipMalloc(&tile_off, (n_expert + 1) * sizeof(int32_t)));
    CK(hipMalloc(&tile_meta, max_tiles * sizeof(repack_tile_meta)));
    const dim3 grid((M + MMQ_RP_Q8_BM - 1) / MMQ_RP_Q8_BM, (unsigned) max_tiles, 1);
    if (w32) {
        repack_tile_off<32 * 1><<<1, 1, 0, st>>>(bounds, tile_off, tile_meta, n_expert);
        mmq_gemm_repacked_w32<true, 1, MMQ_RP_Q8_NROW_LANES * 2, GGML_TYPE_Q8_0>
            <<<grid, dim3(32, MMQ_RP_Q8_NROW_LANES * 2), 0, st>>>(
            w, xq, dst, (uint32_t) K, (uint32_t) M, (uint32_t) n_assign,
            ids_src1, ids_dst, bounds, tile_off, tile_meta,
            (uint32_t) n_expert, expert_stride, (uint32_t) M);
    } else {
        repack_tile_off<64 * MMQ_RP_Q8_TN><<<1, 1, 0, st>>>(bounds, tile_off, tile_meta, n_expert);
        mmq_gemm_repacked<true, MMQ_RP_Q8_TN, MMQ_RP_Q8_NROW_LANES, GGML_TYPE_Q8_0>
            <<<grid, dim3(64, MMQ_RP_Q8_NROW_LANES), 0, st>>>(
            w, xq, dst, (uint32_t) K, (uint32_t) M, (uint32_t) n_assign,
            ids_src1, ids_dst, bounds, tile_off, tile_meta,
            (uint32_t) n_expert, expert_stride, (uint32_t) M);
    }
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    hipFree(tile_off); hipFree(tile_meta);
}

// ---------------------------------------------------------------------------
static int compare_bits(const float * a, const float * b, size_t n,
                        float * max_abs, float * max_rel) {
    int ndiff = 0;
    *max_abs = 0.0f; *max_rel = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        uint32_t ua, ub; memcpy(&ua, &a[i], 4); memcpy(&ub, &b[i], 4);
        if (ua != ub) {
            ++ndiff;
            const float d = fabsf(a[i] - b[i]);
            if (d > *max_abs) *max_abs = d;
            const float m = fmaxf(fabsf(a[i]), fabsf(b[i]));
            const float r = m > 0.0f ? d / m : 1.0f;
            if (r > *max_rel) *max_rel = r;
        }
    }
    return ndiff;
}

// CPU reference: per compact column a (expert e from bounds), per row i:
//   acc = 0; for b ascending: acc = fmaf((float)d_w * d_x, (float)idot, acc)
// written to dst[ids_dst[a]*M + i] (both kernels scatter identically).
static void cpu_ref_hasids(int M, int K, int n_assign, int n_expert,
                           const block_q8_0 * w, const block_q8_1_mmq * y,
                           const int32_t * bounds, const int32_t * ids_dst,
                           std::vector<float> & out) {
    out.assign((size_t) M * n_assign, 0.0f);
    const int n_sub = K / 32;
    for (int a = 0; a < n_assign; ++a) {
        int e = 0;
        while (e + 1 < n_expert && bounds[e + 1] <= a) ++e;
        const block_q8_0 * we = w + (size_t) e * M * n_sub;
        float * col = out.data() + (size_t) ids_dst[a] * M;
        for (int i = 0; i < M; ++i) {
            float acc = 0.0f;
            for (int b = 0; b < n_sub; ++b) {
                const block_q8_0 & wb = we[(size_t) i * n_sub + b];
                const block_q8_1_mmq & xa = y[(size_t) (b / 4) * n_assign + a];
                const int kq = b % 4;
                const float d_w = (float) wb.d;
                const float d_x = xa.d4[kq];
                const int8_t * wq = wb.qs;
                const int8_t * xq = &xa.qs[kq * 32];
                int idot = 0;
                for (int v = 0; v < 32; ++v) idot += (int) wq[v] * (int) xq[v];
                acc = fmaf(d_w * d_x, (float) idot, acc);
            }
            col[i] = acc;
        }
    }
}

static void run_shape(const char * tag, int M, int K, int n_tokens, int e_used, int n_expert,
                      int nsm, size_t smpbo, std::mt19937 & rng) {
    const int n_assign = n_tokens * e_used;
    const int64_t ne10_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
    const int64_t n_groups    = ne10_padded / QK8_1_MMQ;
    const int64_t n_real_grp  = K / QK8_1_MMQ;

    // ---- routing: random experts (empty experts and multi-token experts both occur) ----
    std::uniform_int_distribution<int> ed(0, n_expert - 1);
    std::vector<int32_t> ids(n_assign);
    for (auto & v : ids) v = ed(rng);
    std::vector<int32_t> ids_src1, ids_dst, bounds;
    mm_ids_helper_host(ids, n_tokens, e_used, n_expert, ids_src1, ids_dst, bounds);

    // ---- weights: random Q8_0 blocks per expert (identical logical bytes to both arms) ----
    const int64_t nblk_e = (int64_t) M * (K / 32);
    std::vector<block_q8_0> wq((size_t) n_expert * nblk_e);
    std::uniform_real_distribution<float> dd(0.0005f, 0.02f);
    std::uniform_int_distribution<int> qq(-127, 127);
    for (auto & b : wq) {
        b.d = (ggml_half) dd(rng);
        for (int i = 0; i < 32; ++i) b.qs[i] = (int8_t) qq(rng);
    }
    const size_t expert_stride = repack_gcn_nbytes(GGML_TYPE_Q8_0, K, M);
    std::vector<uint8_t> w_rp((size_t) n_expert * expert_stride, 0);
    for (int e = 0; e < n_expert; ++e) {
        repack_q8_0_host_local(wq.data() + (size_t) e * nblk_e, w_rp.data() + e * expert_stride, K, M);
    }
    // canonical buffer + zero tail (stock reads past the last block on K-tail rows)
    const size_t wq_bytes = wq.size() * sizeof(block_q8_0) + 512;
    std::vector<uint8_t> wq_pad(wq_bytes, 0);
    memcpy(wq_pad.data(), wq.data(), wq.size() * sizeof(block_q8_0));

    // ---- activations: fp32 in ORIGINAL assignment order, quantized ONCE ----
    std::vector<float> xf((size_t) n_assign * K);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & v : xf) v = nd(rng);
    std::vector<block_q8_1_mmq> xq_orig((size_t) n_groups * n_assign);
    memset(xq_orig.data(), 0, xq_orig.size() * sizeof(block_q8_1_mmq));
    {
        std::vector<block_q8_1_mmq> yreal((size_t) n_real_grp * n_assign);
        quantize_mmq_q8_1_host(xf.data(), K, n_assign, yreal.data());
        for (int64_t g = 0; g < n_real_grp; ++g)
            for (int j = 0; j < n_assign; ++j)
                xq_orig[(size_t) g * n_assign + j] = yreal[(size_t) g * n_assign + j];
    }
    // stock arm buffer: COMPACT order (row a = src1 column ids_src1[a])
    std::vector<block_q8_1_mmq> y_cmpact(xq_orig.size());
    for (int64_t g = 0; g < n_groups; ++g)
        for (int a = 0; a < n_assign; ++a)
            y_cmpact[(size_t) g * n_assign + a] = xq_orig[(size_t) g * n_assign + ids_src1[a]];

    // ---- device buffers ----
    uint8_t * wq_d = nullptr; uint8_t * wrp_d = nullptr;
    block_q8_1_mmq * yq_d = nullptr; block_q8_1_mmq * xq_d = nullptr;
    int32_t * ids_src1_d = nullptr; int32_t * ids_dst_d = nullptr; int32_t * bounds_d = nullptr;
    float *dst_s1=nullptr,*dst_s2=nullptr,*dst_w1=nullptr,*dst_w2=nullptr,*dst_61=nullptr,*dst_62=nullptr;
    const size_t dstn = (size_t) M * n_assign;
    CK(hipMalloc(&wq_d, wq_bytes));
    CK(hipMalloc(&wrp_d, w_rp.size()));
    CK(hipMalloc(&yq_d, y_cmpact.size() * sizeof(block_q8_1_mmq)));
    CK(hipMalloc(&xq_d, xq_orig.size() * sizeof(block_q8_1_mmq)));
    // ids_dst padded by J-1 for the kernel's unconditional shared load (mmq.cu:419)
    CK(hipMalloc(&ids_dst_d, (n_assign + 128) * sizeof(int32_t)));
    CK(hipMalloc(&ids_src1_d, n_assign * sizeof(int32_t)));
    CK(hipMalloc(&bounds_d, (n_expert + 1) * sizeof(int32_t)));
    CK(hipMalloc(&dst_s1, dstn*4)); CK(hipMalloc(&dst_s2, dstn*4));
    CK(hipMalloc(&dst_w1, dstn*4)); CK(hipMalloc(&dst_w2, dstn*4));
    CK(hipMalloc(&dst_61, dstn*4)); CK(hipMalloc(&dst_62, dstn*4));
    CK(hipMemcpy(wq_d, wq_pad.data(), wq_bytes, hipMemcpyHostToDevice));
    CK(hipMemcpy(wrp_d, w_rp.data(), w_rp.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(yq_d, y_cmpact.data(), y_cmpact.size()*sizeof(block_q8_1_mmq), hipMemcpyHostToDevice));
    CK(hipMemcpy(xq_d, xq_orig.data(), xq_orig.size()*sizeof(block_q8_1_mmq), hipMemcpyHostToDevice));
    CK(hipMemcpy(ids_dst_d, ids_dst.data(), n_assign*sizeof(int32_t), hipMemcpyHostToDevice));
    CK(hipMemset((char*)ids_dst_d + n_assign*sizeof(int32_t), 0, 128*sizeof(int32_t)));
    CK(hipMemcpy(ids_src1_d, ids_src1.data(), n_assign*sizeof(int32_t), hipMemcpyHostToDevice));
    CK(hipMemcpy(bounds_d, bounds.data(), (n_expert+1)*sizeof(int32_t), hipMemcpyHostToDevice));
    CK(hipMemset(dst_s1, 0xFF, dstn*4)); CK(hipMemset(dst_s2, 0xFF, dstn*4));
    CK(hipMemset(dst_w1, 0xFF, dstn*4)); CK(hipMemset(dst_w2, 0xFF, dstn*4));
    CK(hipMemset(dst_61, 0xFF, dstn*4)); CK(hipMemset(dst_62, 0xFF, dstn*4));

    const int J = pick_J_stock_ids(M, n_tokens, n_assign, n_expert, K, nsm, smpbo);
    const bool fb = ggml_cuda_mmq_needs_fallback(M);
    if (J == 0) { fprintf(stderr, "no J for M=%d N=%d\n", M, n_tokens); exit(2); }

    #define LS(JA) do { if (J == JA) { if (fb) { \
        launch_stock_ids_J<true >(JA,M,K,n_tokens,e_used,n_expert,n_assign,(const char*)wq_d,(const int*)yq_d,ids_dst_d,bounds_d,dst_s1,smpbo,0); \
        launch_stock_ids_J<true >(JA,M,K,n_tokens,e_used,n_expert,n_assign,(const char*)wq_d,(const int*)yq_d,ids_dst_d,bounds_d,dst_s2,smpbo,0); } \
        else { \
        launch_stock_ids_J<false>(JA,M,K,n_tokens,e_used,n_expert,n_assign,(const char*)wq_d,(const int*)yq_d,ids_dst_d,bounds_d,dst_s1,smpbo,0); \
        launch_stock_ids_J<false>(JA,M,K,n_tokens,e_used,n_expert,n_assign,(const char*)wq_d,(const int*)yq_d,ids_dst_d,bounds_d,dst_s2,smpbo,0); } } } while (0)
    LS(8); LS(16); LS(24); LS(32); LS(40); LS(48); LS(56); LS(64);
    #undef LS

    launch_repack_ids(true,  M, K, n_expert, n_assign, expert_stride, wrp_d, (const block_q8_1*) xq_d,
                      ids_src1_d, ids_dst_d, bounds_d, dst_w1, 0);
    launch_repack_ids(true,  M, K, n_expert, n_assign, expert_stride, wrp_d, (const block_q8_1*) xq_d,
                      ids_src1_d, ids_dst_d, bounds_d, dst_w2, 0);
    launch_repack_ids(false, M, K, n_expert, n_assign, expert_stride, wrp_d, (const block_q8_1*) xq_d,
                      ids_src1_d, ids_dst_d, bounds_d, dst_61, 0);
    launch_repack_ids(false, M, K, n_expert, n_assign, expert_stride, wrp_d, (const block_q8_1*) xq_d,
                      ids_src1_d, ids_dst_d, bounds_d, dst_62, 0);

    std::vector<float> hs1(dstn), hs2(dstn), hw1(dstn), hw2(dstn), h61(dstn), h62(dstn);
    CK(hipMemcpy(hs1.data(), dst_s1, dstn*4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hs2.data(), dst_s2, dstn*4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hw1.data(), dst_w1, dstn*4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hw2.data(), dst_w2, dstn*4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(h61.data(), dst_61, dstn*4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(h62.data(), dst_62, dstn*4, hipMemcpyDeviceToHost));

    float ma, mr, ma2, mr2, ma3, mr3, ma4, mr4, ma5, mr5;
    const int d_stock = compare_bits(hs1.data(), hs2.data(), dstn, &ma, &mr);
    const int d_w32   = compare_bits(hw1.data(), hw2.data(), dstn, &ma2, &mr2);
    const int d_64    = compare_bits(h61.data(), h62.data(), dstn, &ma3, &mr3);
    const int x_w32   = compare_bits(hs1.data(), hw1.data(), dstn, &ma4, &mr4);
    const int x_64    = compare_bits(hs1.data(), h61.data(), dstn, &ma5, &mr5);
    const int x_w32_64 = compare_bits(hw1.data(), h61.data(), dstn, &ma, &mr);

    int ref_ok = -1; float ref_maxrel = 0.0f;
    if ((int64_t) M * n_assign * K <= (1LL << 26)) {
        std::vector<float> ref;
        cpu_ref_hasids(M, K, n_assign, n_expert, wq.data(), y_cmpact.data(), bounds.data(), ids_dst.data(), ref);
        float worst = 0.0f; int bad = 0;
        for (size_t t = 0; t < dstn; ++t) {
            const float m = fmaxf(fabsf(ref[t]), 1e-6f);
            const float r1 = fabsf(hs1[t] - ref[t]) / m;
            const float r2 = fabsf(hw1[t] - ref[t]) / m;
            const float r3 = fabsf(h61[t] - ref[t]) / m;
            if (!(r1 < 1e-4f) || !(r2 < 1e-4f) || !(r3 < 1e-4f)) ++bad;
            worst = fmaxf(worst, fmaxf(r1, fmaxf(r2, r3)));
        }
        ref_ok = bad == 0 ? 1 : 0;
        ref_maxrel = worst;
    }

    printf("%s,%d,%d,%d,%d,%d,%d,%d,%zu,%d,%d,%d,%d,%d,%d,%.3g,%.3g,%.3g,%.3g,%d,%.3g\n",
           tag, n_tokens, e_used, n_expert, M, K, J, n_assign, dstn,
           d_stock, d_w32, d_64, x_w32, x_64, x_w32_64, ma4, mr4, ma5, mr5, ref_ok, ref_maxrel);
    fflush(stdout);

    hipFree(wq_d); hipFree(wrp_d); hipFree(yq_d); hipFree(xq_d);
    hipFree(ids_src1_d); hipFree(ids_dst_d); hipFree(bounds_d);
    hipFree(dst_s1); hipFree(dst_s2); hipFree(dst_w1); hipFree(dst_w2); hipFree(dst_61); hipFree(dst_62);
}

int main(int argc, char ** argv) {
    CK(hipSetDevice(0));
    int nsm = 0;
    CK(hipDeviceGetAttribute(&nsm, hipDeviceAttributeMultiprocessorCount, 0));
    int sm_pb = 0;
    CK(hipDeviceGetAttribute(&sm_pb, hipDeviceAttributeMaxSharedMemoryPerBlock, 0));
    const size_t smpbo = (size_t) sm_pb;
    fprintf(stderr, "# device nsm=%d smpbo=%zu\n", nsm, smpbo);

    printf("tag,n_tokens,e_used,n_expert,M,K,J_stock,n_assign,n_out,"
           "stock_det,repw32_det,rep64_det,diff_stock_w32,diff_stock_64,diff_w32_64,"
           "max_abs_w32,max_rel_w32,max_abs_64,max_rel_64,ref_ok,ref_maxrel\n");
    fflush(stdout);

    std::mt19937 rng(0xE6u);

    // ---- main grid: routed-token widths incl. the 9..31 spec-verify class ----
    static const int Ns[] = {1, 2, 4, 8, 9, 16, 31, 32};
    static const int Es[] = {1, 2, 4};            // experts used per token
    static const int Xs[] = {1, 2, 8};            // expert count (1 = dense form)
    static const int Ks[] = {1024, 1152, 2048, 2176, 4096, 5120, 6144};  // 1152/2176: K%256=128
    static const int Ms[] = {1024, 2048};
    for (int X : Xs) for (int E : Es) for (int N : Ns) for (int K : Ks) for (int M : Ms) {
        run_shape("main", M, K, N, E, X, nsm, smpbo, rng);
    }

    // ---- real MoE shapes: Qwen3-30B-A3B gate/up (K=2048,M=768) and down (K=768,M=2048),
    //      64 experts, 4 used; short-K J=32 override engages on the down proj ----
    static const int RNs[] = {9, 16, 31};
    for (int N : RNs) {
        run_shape("real_up",   768, 2048, N, 4, 64, nsm, smpbo, rng);
        run_shape("real_down", 2048, 768, N, 4, 64, nsm, smpbo, rng);
    }
    return 0;
}
