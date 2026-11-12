// mb_e4_gemm.cu - E4 step 0 ground-truth microbench: stock MMQ vs repacked Q8_0 GEMM.
//
// Feeds IDENTICAL q8_1 activation bytes (block_q8_1_mmq D4 layout, produced by a
// host bit-mirror of quantize_mmq_q8_1<DS_LAYOUT_D4> - the exact quantizer BOTH
// runtime paths use for the GEMM: stock mul_mat_q via quantize_mmq_q8_1_cuda,
// repacked mmq_gemm_repacked* via the same call in q8_repack/mul-mat.cu) and
// IDENTICAL Q8_0 weights (raw blocks for stock, repack_q8_0_host byte-permutation
// for the repacked kernel) through both kernels at every shape in the E4 grid.
//
// Per-output fp32 bit-diff + launch determinism (each kernel run twice).
// This isolates the GEMM fp32 arithmetic: same input values -> same output bits?
//
// Stock side mirrors ggml_cuda_mul_mat_q's host launch logic for gfx906 dense
// Q8_0 (J sweep with the CU-occupancy gate, non-stream-K branch; the gfx906 GCN
// config table has stream_k=false for every Q8_0 row, so no fixup reduce exists).
//
// Build (TheRock 7.15, source /root/w4a4-env.sh):
//   hipcc -O3 -std=c++17 --offload-arch=gfx906 -DGGML_USE_HIP \
//     -Iggml/include -Iggml/src -Iggml/src/ggml-cuda \
//     ggml/src/ggml-cuda/q8_repack/mb_e4_gemm.cu -o mb_e4_gemm
// Run on ONE idle card under flock (GPU2 by convention):
//   flock -w 600 /root/gpu.lock env HIP_VISIBLE_DEVICES=2 ./mb_e4_gemm
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
// Same ops: amax over 32 values, d_inv = 127/amax, q = roundf(x*d_inv), d = 1/d_inv
// stored as float. Layout: y[kb*ncols_y + j] = { float d4[4]; int8 qs[128] }.
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

// Byte-permutation repack of Q8_0 blocks (verbatim copy of repack_q8_0_host,
// q8_repack/repack-common.cu:350 - pure permutation, no arithmetic).
static void repack_q8_0_host_local(const block_q8_0 * blocks, uint8_t * dst,
                                   const int64_t ne0, const int64_t ne1) {
    const int64_t n_blocks = ne0 / 32;
    const int64_t qs_str   = repack_qs_stride(ne0);
    const size_t  qs_len   = (size_t) ne1 * qs_str;
    memset(dst, 0, qs_len + (size_t) ne1 * n_blocks * 2);
    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q8_0 * b = &blocks[row * n_blocks + blk];
            memcpy(dst + (size_t) row * qs_str + (size_t) blk * 32, b->qs, 32);
            memcpy(dst + qs_len + (size_t)(row * n_blocks + blk) * 2, &b->d, 2);
        }
    }
}

// ---------------------------------------------------------------------------
// Stock side: mirror of ggml_cuda_mul_mat_q's J sweep (mmq.cu:215-256) for
// dense single-channel/sample gfx906 Q8_0, plus the non-stream-K launch
// (mmq.cuh:1501-1526). All gfx906 Q8_0 configs have stream_k=false.
// ---------------------------------------------------------------------------
static int pick_J_stock(int M, int N, int nsm, size_t smpbo) {
    const bool fallback = ggml_cuda_mmq_needs_fallback(M);
    int J_best = 0, ntiles_best = INT_MAX;
    for (int J = 8; J <= 128 && ntiles_best > 1; J += 8) {
        const ggml_cuda_mmq_config cfg =
            ggml_cuda_mmq_get_config(GGML_TYPE_Q8_0, J, fallback, GGML_CUDA_CC_VEGA20, GGML_PREC_Q8);
        if (cfg.type == GGML_TYPE_COUNT) continue;
        if (mmq_get_nbytes_shared(cfg, GGML_CUDA_CC_VEGA20) > smpbo) continue;
        const int ntiles_x = (N + cfg.J - 1) / cfg.J;
        if (cfg.J > 64) { // gfx906 CU-occupancy gate (mmq.cu)
            const int64_t nty = (M + cfg.I - 1) / cfg.I;
            if (nty * ntiles_x < nsm) continue;
        }
        if (ntiles_x < ntiles_best) { J_best = J; ntiles_best = ntiles_x; }
    }
    return J_best;
}

template <int J, bool FB>
static void launch_stock(int M, int K, int N, const char * x, const int * y,
                         float * dst, cudaStream_t st) {
    const ggml_cuda_mmq_config cfg =
        ggml_cuda_mmq_get_config(GGML_TYPE_Q8_0, J, FB, GGML_CUDA_CC_VEGA20, GGML_PREC_Q8);
    const int nbytes  = mmq_get_nbytes_shared(cfg, GGML_CUDA_CC_VEGA20);
    const int nwarps  = cfg.nthreads / 64;
    const int nty     = (M + cfg.I - 1) / cfg.I;
    const int ntx     = (N + cfg.J - 1) / cfg.J;
    CUDA_SET_SHARED_MEMORY_LIMIT((mul_mat_q<GGML_TYPE_Q8_0, J, FB, GGML_PREC_Q8>), nbytes);
    mul_mat_q<GGML_TYPE_Q8_0, J, FB, GGML_PREC_Q8><<<dim3(nty, ntx, 1), dim3(64, nwarps, 1), nbytes, st>>>(
        x, y, nullptr, nullptr, dst, nullptr, nullptr,
        init_fastdiv_values(K / 32),
        M, N, K / 32, N, M,
        init_fastdiv_values(1), init_fastdiv_values(1), 0, 0, 0,
        init_fastdiv_values(1), init_fastdiv_values(1), 0, 0, 0,
        init_fastdiv_values(ntx), nullptr);
}

template <bool FB>
static void launch_stock_J(int J, int M, int K, int N, const char * x, const int * y,
                           float * dst, cudaStream_t st) {
    switch (J) {
        case 8:   launch_stock<8,   FB>(M, K, N, x, y, dst, st); break;
        case 16:  launch_stock<16,  FB>(M, K, N, x, y, dst, st); break;
        case 24:  launch_stock<24,  FB>(M, K, N, x, y, dst, st); break;
        case 32:  launch_stock<32,  FB>(M, K, N, x, y, dst, st); break;
        case 40:  launch_stock<40,  FB>(M, K, N, x, y, dst, st); break;
        case 48:  launch_stock<48,  FB>(M, K, N, x, y, dst, st); break;
        case 56:  launch_stock<56,  FB>(M, K, N, x, y, dst, st); break;
        case 64:  launch_stock<64,  FB>(M, K, N, x, y, dst, st); break;
        case 72:  launch_stock<72,  FB>(M, K, N, x, y, dst, st); break;
        case 80:  launch_stock<80,  FB>(M, K, N, x, y, dst, st); break;
        case 88:  launch_stock<88,  FB>(M, K, N, x, y, dst, st); break;
        case 96:  launch_stock<96,  FB>(M, K, N, x, y, dst, st); break;
        case 104: launch_stock<104, FB>(M, K, N, x, y, dst, st); break;
        case 112: launch_stock<112, FB>(M, K, N, x, y, dst, st); break;
        case 120: launch_stock<120, FB>(M, K, N, x, y, dst, st); break;
        case 128: launch_stock<128, FB>(M, K, N, x, y, dst, st); break;
        default: fprintf(stderr, "unhandled J=%d\n", J); exit(2);
    }
}

// ---------------------------------------------------------------------------
// Repacked side: mirrors q8_repack/mul-mat.cu dispatch for GGML_TYPE_Q8_0:
//   ne11 >= 128 -> mmq_gemm_repacked<false, TN=2, NRL=4, Q8_0>, block(64, NRL)
//   ne11 <  128 -> mmq_gemm_repacked_w32<false, 1, NRL*2, Q8_0>, block(32, NRL*2)
// xq is the SAME block_q8_1_mmq buffer as the stock y (reinterpreted, as at runtime).
// ---------------------------------------------------------------------------
static void launch_repacked(int M, int K, int N, const uint8_t * w, const block_q8_1 * xq,
                            float * dst, cudaStream_t st) {
    if (N >= 128) {
        const dim3 grid((M + MMQ_RP_Q8_BM - 1) / MMQ_RP_Q8_BM,
                        (N + 64*MMQ_RP_Q8_TN - 1) / (64*MMQ_RP_Q8_TN), 1);
        mmq_gemm_repacked<false, MMQ_RP_Q8_TN, MMQ_RP_Q8_NROW_LANES, GGML_TYPE_Q8_0>
            <<<grid, dim3(64, MMQ_RP_Q8_NROW_LANES), 0, st>>>(
            w, xq, dst, (uint32_t) K, (uint32_t) M, (uint32_t) N,
            nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) M);
    } else {
        const dim3 grid((M + MMQ_RP_Q8_BM - 1) / MMQ_RP_Q8_BM, (N + 31) / 32, 1);
        mmq_gemm_repacked_w32<false, 1, MMQ_RP_Q8_NROW_LANES*2, GGML_TYPE_Q8_0>
            <<<grid, dim3(32, MMQ_RP_Q8_NROW_LANES*2), 0, st>>>(
            w, xq, dst, (uint32_t) K, (uint32_t) M, (uint32_t) N,
            nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) M);
    }
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

// CPU reference: exact fp32 term sequence both kernels are supposed to emit:
//   per output (row i, token j): acc = 0; for b in 0..K/32-1 ascending:
//     idot = sum of 32 int products (exact int32);
//     acc += ((float)d_w * d_x) * (float)idot
static void cpu_ref_q8_0_mmq(int M, int K, int N,
                             const block_q8_0 * w, const block_q8_1_mmq * y,
                             std::vector<float> & out) {
    out.assign((size_t) M * N, 0.0f);
    const int n_sub = K / 32;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            float acc = 0.0f;
            for (int b = 0; b < n_sub; ++b) {
                const block_q8_0 & wb = w[(size_t) i * n_sub + b];
                const block_q8_1_mmq & xa = y[(size_t)(b / 4) * N + j];
                const int kq = b % 4;
                const float d_w = (float) wb.d;
                const float d_x = xa.d4[kq];
                const int8_t * wq = wb.qs;
                const int8_t * xq = &xa.qs[kq * 32];
                int idot = 0;
                for (int v = 0; v < 32; ++v) idot += (int) wq[v] * (int) xq[v];
                // Match the kernels' emitted ISA: v_mul_f32(d_w,d_x) then
                // v_fmac_f32(acc, scaleprod, cvt(idot)) - contracted form.
                acc = fmaf(d_w * d_x, (float) idot, acc);
            }
            out[(size_t) j * M + i] = acc;
        }
    }
}

static void run_shape(int M, int K, int N, int nsm, size_t smpbo, bool do_ref,
                      const char * grid_tag) {
    // ---- weights: random Q8_0 blocks (identical bytes to both paths) ----
    const int64_t nblk = (int64_t) M * (K / 32);
    std::vector<block_q8_0> wq(nblk);
    std::mt19937 rng(0xE4u + M * 131 + K * 7 + N);
    std::uniform_real_distribution<float> dd(0.0005f, 0.02f);
    std::uniform_int_distribution<int> qq(-127, 127);
    for (auto & b : wq) {
        b.d = (ggml_half) dd(rng);
        for (int i = 0; i < 32; ++i) b.qs[i] = (int8_t) qq(rng);
    }
    std::vector<uint8_t> w_rp((size_t) M * repack_qs_stride(K) + (size_t) M * (K / 32) * 2);
    repack_q8_0_host_local(wq.data(), w_rp.data(), K, M);

    // stock x buffer + zero tail: the K-tail overrun of the last kb0 iteration
    // reads up to 8 blocks past a row (neutralized by zero y padding; ggml
    // tensors live inside padded pools, so mirror that with a zero tail).
    const size_t wq_bytes2 = wq.size() * sizeof(block_q8_0) + 512;
    std::vector<uint8_t> wq_pad(wq_bytes2, 0);
    memcpy(wq_pad.data(), wq.data(), wq.size() * sizeof(block_q8_0));

    // ---- activations: random fp32, quantized ONCE to mmq D4 (identical bytes) ----
    const int64_t ne10_padded = GGML_PAD(K, MATRIX_ROW_PADDING);
    const int64_t n_groups    = ne10_padded / QK8_1_MMQ;   // incl. zero pad groups
    const int64_t n_real_grp  = K / QK8_1_MMQ;
    std::vector<float> xf((size_t) N * K);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto & v : xf) v = nd(rng);
    std::vector<block_q8_1_mmq> yq((size_t) n_groups * N);
    memset(yq.data(), 0, yq.size() * sizeof(block_q8_1_mmq));
    {
        std::vector<block_q8_1_mmq> yreal((size_t) n_real_grp * N);
        quantize_mmq_q8_1_host(xf.data(), K, N, yreal.data());
        for (int64_t g = 0; g < n_real_grp; ++g)
            for (int64_t j = 0; j < N; ++j)
                yq[(size_t) g * N + j] = yreal[(size_t) g * N + j];
    }

    uint8_t * wq_d = nullptr; uint8_t * wrp_d = nullptr;
    block_q8_1_mmq * yq_d = nullptr;
    float *dst_s1=nullptr,*dst_s2=nullptr,*dst_r1=nullptr,*dst_r2=nullptr;
    const size_t dstn = (size_t) M * N;
    CK(hipMalloc(&wq_d, wq_bytes2));
    CK(hipMalloc(&wrp_d, w_rp.size()));
    CK(hipMalloc(&yq_d, yq.size() * sizeof(block_q8_1_mmq)));
    CK(hipMalloc(&dst_s1, dstn*4)); CK(hipMalloc(&dst_s2, dstn*4));
    CK(hipMalloc(&dst_r1, dstn*4)); CK(hipMalloc(&dst_r2, dstn*4));
    CK(hipMemcpy(wq_d, wq_pad.data(), wq_bytes2, hipMemcpyHostToDevice));
    CK(hipMemcpy(wrp_d, w_rp.data(), w_rp.size(), hipMemcpyHostToDevice));
    CK(hipMemcpy(yq_d, yq.data(), yq.size()*sizeof(block_q8_1_mmq), hipMemcpyHostToDevice));
    CK(hipMemset(dst_s1, 0xFF, dstn*4)); CK(hipMemset(dst_s2, 0xFF, dstn*4));
    CK(hipMemset(dst_r1, 0xFF, dstn*4)); CK(hipMemset(dst_r2, 0xFF, dstn*4));

    const int J = pick_J_stock(M, N, nsm, smpbo);
    const bool fb = ggml_cuda_mmq_needs_fallback(M);
    if (J == 0) { fprintf(stderr, "no J for M=%d N=%d\n", M, N); exit(2); }

    #define LS(JA) do { if (J == JA) { if (fb) { launch_stock_J<true >(JA,M,K,N,(const char*)wq_d,(const int*)yq_d,dst_s1,0); launch_stock_J<true >(JA,M,K,N,(const char*)wq_d,(const int*)yq_d,dst_s2,0); } \
                                  else  { launch_stock_J<false>(JA,M,K,N,(const char*)wq_d,(const int*)yq_d,dst_s1,0); launch_stock_J<false>(JA,M,K,N,(const char*)wq_d,(const int*)yq_d,dst_s2,0); } } } while (0)
    LS(8); LS(16); LS(24); LS(32); LS(40); LS(48); LS(56); LS(64);
    LS(72); LS(80); LS(88); LS(96); LS(104); LS(112); LS(120); LS(128);
    #undef LS
    CK(hipDeviceSynchronize());
    launch_repacked(M, K, N, wrp_d, (const block_q8_1*) yq_d, dst_r1, 0);
    CK(hipDeviceSynchronize());
    launch_repacked(M, K, N, wrp_d, (const block_q8_1*) yq_d, dst_r2, 0);
    CK(hipDeviceSynchronize());

    std::vector<float> hs1(dstn), hs2(dstn), hr1(dstn), hr2(dstn);
    CK(hipMemcpy(hs1.data(), dst_s1, dstn*4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hs2.data(), dst_s2, dstn*4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hr1.data(), dst_r1, dstn*4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(hr2.data(), dst_r2, dstn*4, hipMemcpyDeviceToHost));

    float ma, mr, ma2, mr2, ma3, mr3;
    const int ds = compare_bits(hs1.data(), hs2.data(), dstn, &ma, &mr);
    const int dr = compare_bits(hr1.data(), hr2.data(), dstn, &ma2, &mr2);
    const int dx = compare_bits(hs1.data(), hr1.data(), dstn, &ma3, &mr3);

    int ref_ok = -1; float ref_maxrel = 0.0f; float ref_abs_over_rms = 0.0f;
    if (do_ref) {
        std::vector<float> ref;
        cpu_ref_q8_0_mmq(M, K, N, wq.data(), yq.data(), ref);
        double ss = 0.0;
        for (size_t t = 0; t < dstn; ++t) ss += (double) ref[t] * ref[t];
        const float rms = (float) sqrt(ss / (double) dstn);
        float worst = 0.0f; int bad = 0; float worst_abs = 0.0f;
        for (size_t t = 0; t < dstn; ++t) {
            const float m = fmaxf(fabsf(ref[t]), 1e-6f);
            const float e1 = fabsf(hs1[t] - ref[t]);
            const float e2 = fabsf(hr1[t] - ref[t]);
            const float r1 = e1 / m;
            const float r2 = e2 / m;
            if (!(r1 < 1e-4f) || !(r2 < 1e-4f)) ++bad;
            worst = fmaxf(worst, fmaxf(r1, r2));
            worst_abs = fmaxf(worst_abs, fmaxf(e1, e2));
        }
        ref_ok = bad == 0 ? 1 : 0;
        ref_maxrel = worst;
        ref_abs_over_rms = rms > 0.0f ? worst_abs / rms : worst_abs;
    }

    printf("%s,%d,%d,%d,%d,%s,%zu,%d,%d,%d,%.9g,%.9g,%d,%.3g,%.3g\n",
           grid_tag, N, M, K, J, (N >= 128 ? "gemm64" : "gemm_w32"), dstn,
           ds, dr, dx, ma3, mr3, ref_ok, ref_maxrel, ref_abs_over_rms);
    fflush(stdout);

    hipFree(wq_d); hipFree(wrp_d); hipFree(yq_d);
    hipFree(dst_s1); hipFree(dst_s2); hipFree(dst_r1); hipFree(dst_r2);
}

int main(int argc, char ** argv) {
    CK(hipSetDevice(0));
    int nsm = 0;
    CK(hipDeviceGetAttribute(&nsm, hipDeviceAttributeMultiprocessorCount, 0));
    int sm_pb = 0;
    // HIP path: ggml uses prop.sharedMemPerBlock for smpbo (ggml-cuda.cu:373)
    CK(hipDeviceGetAttribute(&sm_pb, hipDeviceAttributeMaxSharedMemoryPerBlock, 0));
    const size_t smpbo = (size_t) sm_pb;
    fprintf(stderr, "# device nsm=%d smpbo=%zu\n", nsm, smpbo);

    printf("grid,N,M,K,J_stock,rp_kernel,n_out,stock_bitdiff,rep_bitdiff,diff_stock_vs_rep,max_abs,max_rel,ref_ok,ref_maxrel,ref_abs_over_rms\n");
    fflush(stdout);

    // ---- main grid (validation-plan leg 1) ----
    static const int Ns[] = {9, 16, 31, 32, 33, 64, 127, 128, 256};
    static const int Ks[] = {1024, 2048, 4096, 5120, 6144};
    static const int Ms[] = {2048, 4096, 5120};
    for (int M : Ms) for (int K : Ks) for (int N : Ns) {
        const bool ref = ((int64_t) M * N * K) <= (1LL << 28);
        run_shape(M, K, N, nsm, smpbo, ref, "main");
    }

    // ---- divergence-class grid: the classes the map names but the main grid cannot reach:
    //      M < 128 (stock fallback geometry) and K % 256 != 0 (row padding) ----
    static const int DN[]  = {9, 32, 128};
    static const int DK[]  = {1024, 1152, 2176, 4096};   // 1152/2176: K%256=128
    static const int DM[]  = {100, 192, 2048};           // 100/192: fallback
    for (int M : DM) for (int K : DK) for (int N : DN) {
        run_shape(M, K, N, nsm, smpbo, true, "divclass");
    }
    return 0;
}
