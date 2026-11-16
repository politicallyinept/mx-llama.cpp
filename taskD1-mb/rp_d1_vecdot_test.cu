// rp_d1_vecdot_test.cu - taskD1 microbench: restructured bit-identical mmvq twins
// (mul_mat_vec_q_q6k_d1 / mul_mat_vec_q_q40_d1) vs the generic mul_mat_vec_q on
// the GCN table, at the shapes that serve the mmvq family:
//   Q6_K: lm_head 5120x248320 (full) / 5120x124160 (TP2 card) + attn_q 5120x6144
//   Q4_0: attn_q 5120x6144 (369-tensor class) + 5120x9216 + 5120x1024
//
// Bars (review template, taskK3 precedent):
//   BIT:   twin == generic VERBATIM, bit-exact, every width T=1..8 (same lanes,
//          kbx chain, kqs split, products, fp order, reduce tree). Both decode
//          variants for Q6_K (d1_decode 0/1).
//   ORACLE: generic == CPU mirror of the same accumulation (widths 2, 5, 8).
//   DETERM: every arm launched twice, outputs must match.
//   PERF:  median us/call + effective weight GB/s vs the 226-380 GB/s baseline.
//   GO:    >=25% faster than generic verbatim on the Q6_K head shape AND the
//          Q4_0 attn_q shape at widths 1-8, bit-exact.
//
// Arithmetic model: host and device both compile -ffp-contract=off (taskE2/taskQ
// precedent). The generic kernel is extracted VERBATIM from mmvq.cu at build time
// (d1-build-mb.sh) - no transcription. Standalone (own main + ggml_abort stub),
// pinned TheRock amdclang++ -x hip; run on ONE clean card under flock (GPU1).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <functional>
#include <hip/hip_runtime.h>
#include "ggml.h"
#include "common.cuh"
#include "mmvq.cuh"
#include "vecdotq.cuh"
#include "unary.cuh"
#include "mb_mmvq_extract.cuh"
#include "d1-mmvq.cuh"

// Link stubs: standalone binary does not link libggml.
#include <cstdarg>
void ggml_abort(const char* file, int line, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "ABORT %s:%d: ", file, line);
    vfprintf(stderr, fmt, ap); va_end(ap);
    fprintf(stderr, "\n"); exit(3);
}
extern "C" bool ggml_is_quantized(enum ggml_type type) { (void) type; return true; }
extern "C" void ggml_log_warn(const char * fmt, ...) { (void) fmt; }

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

// ---------------- CPU mirrors (K3 template) ----------------
static inline int dp4a_cpu(int a, int b, int c) {
    int s = c;
    for (int i = 0; i < 4; ++i) {
        s += (int)(int8_t)((a >> (8*i)) & 0xFF) * (int)(int8_t)((b >> (8*i)) & 0xFF);
    }
    return s;
}
static inline int get_int_b2_cpu(const void * x, int i32) {
    const uint16_t * x16 = (const uint16_t *) x;
    int v = (int) x16[2*i32 + 0];
    v |= ((int) x16[2*i32 + 1]) << 16;
    return v;
}
static inline int get_int_b4_cpu(const void * x, int i32) {
    int v; memcpy(&v, (const int8_t *) x + 4*i32, 4); return v;
}
static inline int vsubss4_cpu(int a, int b) {
    int r = 0;
    for (int i = 0; i < 4; ++i) {
        int d = (int)(uint8_t)(a >> (8*i)) - (int)(uint8_t)(b >> (8*i));
        if (d < -128) d = -128; if (d > 127) d = 127;
        r |= (d & 0xFF) << (8*i);
    }
    return r;
}
static inline float half_lo_f32(uint16_t h2lo) {
    // low half of a half2 stored little-endian: the first f16
    __half h; memcpy(&h, &h2lo, 2); return __half2float(h);
}
// CPU mirror of vec_dot_q6_K_q8_1 (verbatim arithmetic model)
static inline float vec_dot_q6_K_q8_1_cpu(const block_q6_K * bq6_K, const block_q8_1 * bq8_1, int iqs) {
    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));
    const int vl = get_int_b2_cpu(bq6_K->ql, iqs);
    const int vh = get_int_b2_cpu(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;
    const int8_t * scales = bq6_K->scales + scale_offset;
    int u[QR6_K]; float d8[QR6_K];
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_b4_cpu(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        uint16_t dslo; memcpy(&dslo, &bq8_1[bq8_offset + 2*i].ds, 2);
        d8[i] = half_lo_f32(dslo);
    }
    float sumf = 0.0f;
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];
        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
        const int vi = vsubss4_cpu((vil | vih), 0x20202020);
        sumf += d8[i] * (float) (dp4a_cpu(vi, u[i], 0) * sc);
    }
    return __half2float(bq6_K->d) * sumf;
}
// CPU mirror of vec_dot_q4_0_q8_1 (verbatim arithmetic model, VDR=2)
static inline float vec_dot_q4_0_q8_1_cpu(const block_q4_0 * bq4_0, const block_q8_1 * bq8_1, int iqs) {
    int v[2], u[4];
    for (int i = 0; i < 2; ++i) {
        v[i]     = get_int_b2_cpu(bq4_0->qs, iqs + i);
        u[2*i+0] = get_int_b4_cpu(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_b4_cpu(bq8_1->qs, iqs + i + QI4_0);
    }
    int sumi = 0;
    for (int i = 0; i < 2; ++i) {
        sumi = dp4a_cpu((v[i] >> 0) & 0x0F0F0F0F, u[2*i+0], sumi);
        sumi = dp4a_cpu((v[i] >> 4) & 0x0F0F0F0F, u[2*i+1], sumi);
    }
    uint16_t dslo, dshi; memcpy(&dslo, &bq8_1->ds, 2); memcpy(&dshi, (const uint8_t*)&bq8_1->ds + 2, 2);
    const float dx = half_lo_f32(dslo);
    const float dy = half_lo_f32(dshi);
    const float d4 = __half2float(bq4_0->d);
    return d4 * ((float)sumi * dx - 4.0f * dy);
}
// generic-row oracle: models mul_mat_vec_q lanes + reduce for one (row, col).
// nwarps/rpb from the GCN tables; warp0 lane l accumulates warp1 lane l first.
static float generic_row_oracle(int type, uint32_t row, uint32_t col, uint32_t n_blocks,
                                const void * w, uint32_t srx,
                                const block_q8_1 * xq, uint32_t xs,
                                int T, int rpb) {
    const int nwarps = (type == 0)
        ? calc_nwarps(GGML_TYPE_Q6_K, T, MMVQ_PARAMETERS_GCN)
        : calc_nwarps(GGML_TYPE_Q4_0, T, MMVQ_PARAMETERS_GCN);
    const int nthreads = 64*nwarps;
    std::vector<std::vector<float>> tmp(nthreads, std::vector<float>(rpb, 0.0f));
    for (int tid = 0; tid < nthreads; ++tid) {
        if (type == 0) { // Q6_K: qi/vdr = 32
            const int kqs = tid % 32;
            for (int kbx = tid / 32; kbx < (int) n_blocks; kbx += 2*nwarps) {
                const block_q6_K * wr = (const block_q6_K *) w + (size_t) row * srx + kbx;
                const block_q8_1 * xb = xq + (size_t) col * xs + (size_t) kbx * 8;
                for (int i = 0; i < rpb; ++i) {
                    tmp[tid][i] += vec_dot_q6_K_q8_1_cpu(wr + (size_t) i * srx, xb, kqs);
                }
            }
        } else { // Q4_0: qi/vdr = 2, kqs = 2*(tid%2)
            const int kqs = 2*(tid % 2);
            for (int kbx = tid / 2; kbx < (int) n_blocks; kbx += 32*nwarps) {
                const block_q4_0 * wr = (const block_q4_0 *) w + (size_t) row * srx + kbx;
                const block_q8_1 * xb = xq + (size_t) col * xs + (size_t) kbx;
                for (int i = 0; i < rpb; ++i) {
                    tmp[tid][i] += vec_dot_q4_0_q8_1_cpu(wr + (size_t) i * srx, xb, kqs);
                }
            }
        }
    }
    // cross-warp: warp0 lane l += warp1 lane l (nwarps=2), then xor tree
    std::vector<float> v(64);
    for (int l = 0; l < 64; ++l) {
        float s = tmp[l][0];
        for (int w = 1; w < nwarps; ++w) s += tmp[w*64 + l][0];
        v[l] = s;
    }
    for (int off : {32, 16, 8, 4, 2, 1}) {
        std::vector<float> nv(64);
        for (int i = 0; i < 64; ++i) nv[i] = v[i] + v[i ^ off];
        v = nv;
    }
    return v[0];
}

// ---------------- launchers ----------------
template <int T, int GRPB1 = 0>
static void launch_generic_q6k(const uint8_t * w, const block_q8_1 * xq, float * y,
                               uint32_t K, uint32_t N, uint32_t srx, uint32_t xs, hipStream_t s) {
    constexpr int nwarps = calc_nwarps(GGML_TYPE_Q6_K, T, MMVQ_PARAMETERS_GCN);
    int rpb = calc_rows_per_block(T, MMVQ_PARAMETERS_GCN, false, nwarps);
    if (T == 1 && GRPB1 > 0) rpb = GRPB1;
    dim3 grid((N + rpb - 1) / rpb, 1, 1), block(64, nwarps);
    ggml_cuda_mm_fusion_args_device fusion{};
    mul_mat_vec_q<GGML_TYPE_Q6_K, T, false, false, false, GRPB1><<<grid, block, 0, s>>>(
        w, xq, nullptr, fusion, y, K, make_uint3(1, 0, 0), srx, xs, N,
        init_fastdiv_values(1), 0, 0, 0, init_fastdiv_values(1), 0, 0, 0, 0);
}
template <int T, int GRPB1 = 0>
static void launch_generic_q40(const uint8_t * w, const block_q8_1 * xq, float * y,
                               uint32_t K, uint32_t N, uint32_t srx, uint32_t xs, hipStream_t s) {
    constexpr int nwarps = calc_nwarps(GGML_TYPE_Q4_0, T, MMVQ_PARAMETERS_GCN);
    int rpb = calc_rows_per_block(T, MMVQ_PARAMETERS_GCN, false, nwarps);
    if (T == 1 && GRPB1 > 0) rpb = GRPB1;
    dim3 grid((N + rpb - 1) / rpb, 1, 1), block(64, nwarps);
    ggml_cuda_mm_fusion_args_device fusion{};
    mul_mat_vec_q<GGML_TYPE_Q4_0, T, false, false, false, GRPB1><<<grid, block, 0, s>>>(
        w, xq, nullptr, fusion, y, K, make_uint3(1, 0, 0), srx, xs, N,
        init_fastdiv_values(1), 0, 0, 0, init_fastdiv_values(1), 0, 0, 0, 0);
}
template <int T, int DD, int U, int GRPB1 = 0>
static void launch_d1_q6k(const uint8_t * w, const block_q8_1 * xq, float * y,
                          uint32_t K, uint32_t N, uint32_t srx, uint32_t xs, hipStream_t s) {
    constexpr int nwarps = calc_nwarps(GGML_TYPE_Q6_K, T, MMVQ_PARAMETERS_GCN);
    constexpr int rpb0 = calc_rows_per_block(T, MMVQ_PARAMETERS_GCN, false, nwarps);
    constexpr int rpb = (T == 1 && GRPB1 > 0) ? GRPB1 : rpb0;
    dim3 grid((N + rpb - 1) / rpb, 1, 1), block(64, nwarps);
    mul_mat_vec_q_q6k_d1<T, nwarps, rpb, DD, U><<<grid, block, 0, s>>>(w, xq, y, K, srx, xs, N);
}
template <int T, int U, int GRPB1 = 0>
static void launch_d1_q40(const uint8_t * w, const block_q8_1 * xq, float * y,
                          uint32_t K, uint32_t N, uint32_t srx, uint32_t xs, hipStream_t s) {
    constexpr int nwarps = calc_nwarps(GGML_TYPE_Q4_0, T, MMVQ_PARAMETERS_GCN);
    constexpr int rpb0 = calc_rows_per_block(T, MMVQ_PARAMETERS_GCN, false, nwarps);
    constexpr int rpb = (T == 1 && GRPB1 > 0) ? GRPB1 : rpb0;
    dim3 grid((N + rpb - 1) / rpb, 1, 1), block(64, nwarps);
    mul_mat_vec_q_q40_d1<T, nwarps, rpb, U><<<grid, block, 0, s>>>(w, xq, y, K, srx, xs, N);
}

// ---------------- roofline diagnostic ----------------
// Same grid/block geometry and same weight-load pattern as the generic kernel,
// but no activations and no dots: folds the loaded weight words into a sink.
// Measures the max weight-read bandwidth this access pattern can achieve; if
// generic ~= roofline-read, the kernel is memory-latency bound and decode ALU
// work is irrelevant (refutation evidence for the ledger).
template <int NTHREADS, int BPI, int RPB>
__global__ __launch_bounds__(NTHREADS, 1)
void roofline_read_q6k(const void * __restrict__ vx, float * dst,
                       uint32_t ncols_x, uint32_t stride_row_x) {
    const int tid = 64*threadIdx.y + threadIdx.x;
    const int row0 = RPB*blockIdx.x;
    const int blocks_per_row_x = ncols_x / QK_K;
    const int kqs = tid % 32;
    const int vh_idx = 8*(kqs/16) + kqs%8;
    const int vh_sh  = 2*((kqs%16)/8);
    const int sc_off = 8*(kqs/16) + (kqs%16)/4;
    int sink = 0;
    for (int kbx = tid/32; kbx < blocks_per_row_x; kbx += BPI) {
#pragma unroll
        for (int i = 0; i < RPB; ++i) {
            const block_q6_K * b = (const block_q6_K *) vx + row0*stride_row_x + i*stride_row_x + kbx;
            sink ^= get_int_b2(b->ql, kqs) ^ (get_int_b2(b->qh, vh_idx) >> vh_sh)
                  ^ get_int_b4(b->scales, sc_off/4) ^ (int) b->d;
        }
    }
    if (tid == 0) dst[blockIdx.x] = __int_as_float(sink);
}
template <int NTHREADS, int BPI, int RPB>
__global__ __launch_bounds__(NTHREADS, 1)
void roofline_read_q40(const void * __restrict__ vx, float * dst,
                       uint32_t ncols_x, uint32_t stride_row_x) {
    const int tid = 64*threadIdx.y + threadIdx.x;
    const int row0 = RPB*blockIdx.x;
    const int blocks_per_row_x = ncols_x / QK4_0;
    const int kqs = 2*(tid % 2);
    int sink = 0;
    for (int kbx = tid/2; kbx < blocks_per_row_x; kbx += BPI) {
#pragma unroll
        for (int i = 0; i < RPB; ++i) {
            const block_q4_0 * b = (const block_q4_0 *) vx + row0*stride_row_x + i*stride_row_x + kbx;
            sink ^= get_int_b2(b->qs, kqs) ^ get_int_b2(b->qs, kqs+1) ^ (int) b->d;
        }
    }
    if (tid == 0) dst[blockIdx.x] = __int_as_float(sink);
}

// ---------------- timing ----------------
static double bench(std::function<void()> fn, int iters) {
    std::vector<float> ms;
    hipEvent_t e0, e1; HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
    for (int i = 0; i < 3; ++i) fn();
    HIP_CHECK(hipStreamSynchronize(0));
    for (int i = 0; i < iters; ++i) {
        HIP_CHECK(hipEventRecord(e0, 0)); fn(); HIP_CHECK(hipEventRecord(e1, 0));
        HIP_CHECK(hipEventSynchronize(e1));
        float msf; HIP_CHECK(hipEventElapsedTime(&msf, e0, e1));
        ms.push_back(msf);
    }
    HIP_CHECK(hipEventDestroy(e0)); HIP_CHECK(hipEventDestroy(e1));
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2] * 1000.0; // us, median
}

// ---------------- data fill ----------------
static void fill_random(int type, uint8_t * w, block_q8_1 * xq, uint32_t K, uint32_t N,
                        uint32_t xs, uint32_t T, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte_d(0, 255);
    std::uniform_real_distribution<float> dscale(0.001f, 0.5f);
    std::uniform_real_distribution<float> act_d(0.001f, 2.0f);
    if (type == 0) { // Q6_K, 210 B blocks
        const uint32_t nb = K / 256;
        block_q6_K * wq = (block_q6_K *) w;
        for (size_t i = 0; i < (size_t) N * nb; ++i) {
            for (int j = 0; j < 128; ++j) wq[i].ql[j] = byte_d(rng);
            for (int j = 0; j < 64;  ++j) wq[i].qh[j] = byte_d(rng);
            for (int j = 0; j < 16;  ++j) wq[i].scales[j] = (int8_t) byte_d(rng);
            wq[i].d = __float2half(dscale(rng));
        }
    } else { // Q4_0, 18 B blocks
        const uint32_t nb = K / 32;
        block_q4_0 * wq = (block_q4_0 *) w;
        for (size_t i = 0; i < (size_t) N * nb; ++i) {
            for (int j = 0; j < 16; ++j) wq[i].qs[j] = (uint8_t) byte_d(rng);
            wq[i].d = __float2half(dscale(rng));
        }
    }
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t b = 0; b < xs; ++b) {
            block_q8_1 * xb = xq + (size_t) t * xs + b;
            xb->ds = __floats2half2_rn(act_d(rng), (float) ((int)(rng() % 64) - 32));
            for (int j = 0; j < 32; ++j) xb->qs[j] = (int8_t) byte_d(rng);
        }
    }
}

int main(int argc, char ** argv) {
    const bool quick = argc > 1 && !strcmp(argv[1], "--quick");
    const char * only = argc > 2 ? argv[2] : nullptr;
    int dev; HIP_CHECK(hipSetDevice(0)); HIP_CHECK(hipGetDevice(&dev));
    printf("# taskD1 vec_dot rewrite microbench, device %d\n", dev);

    struct Shape { int type; uint32_t K, N; const char * tag; };
    std::vector<Shape> shapes = {
        { 0, 5120, 6144,    "Q6_K attn_q 5120x6144" },
        { 0, 5120, 248320,  "Q6_K head 5120x248320" },
        { 1, 5120, 6144,    "Q4_0 attn_q 5120x6144" },
        { 1, 5120, 9216,    "Q4_0 ffn 5120x9216" },
    };

    for (auto & sh : shapes) {
        if (only && !strstr(sh.tag, only)) continue;
        const uint32_t K = sh.K, N = sh.N;
        const uint32_t srx = (sh.type == 0) ? K/256 : K/32;
        const uint32_t xs  = K/32;
        const size_t wbytes = (size_t) N * srx * (sh.type == 0 ? 210 : 18);
        for (uint32_t T : {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
            if (quick && T != 1 && T != 4 && T != 8) continue;
            uint8_t * w; block_q8_1 * xq; float * y1; float * y2;
            HIP_CHECK(hipMalloc(&w, wbytes));
            HIP_CHECK(hipMalloc(&xq, (size_t) T * xs * sizeof(block_q8_1)));
            HIP_CHECK(hipMalloc(&y1, (size_t) N * T * 4));
            HIP_CHECK(hipMalloc(&y2, (size_t) N * T * 4));
            std::vector<uint8_t> hw(wbytes); std::vector<block_q8_1> hxq((size_t) T * xs);
            fill_random(sh.type, hw.data(), hxq.data(), K, N, xs, T, 1234 + T);
            HIP_CHECK(hipMemcpy(w, hw.data(), wbytes, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(xq, hxq.data(), (size_t) T * xs * sizeof(block_q8_1), hipMemcpyHostToDevice));

            // generic reference
            std::vector<float> hy((size_t) N * T);
            auto gen = [&](float * y) {
                switch (T) {
                    case 1: if (sh.type==0) launch_generic_q6k<1>(w,xq,y,K,N,srx,xs,0); else launch_generic_q40<1>(w,xq,y,K,N,srx,xs,0); break;
                    case 2: if (sh.type==0) launch_generic_q6k<2>(w,xq,y,K,N,srx,xs,0); else launch_generic_q40<2>(w,xq,y,K,N,srx,xs,0); break;
                    case 3: if (sh.type==0) launch_generic_q6k<3>(w,xq,y,K,N,srx,xs,0); else launch_generic_q40<3>(w,xq,y,K,N,srx,xs,0); break;
                    case 4: if (sh.type==0) launch_generic_q6k<4>(w,xq,y,K,N,srx,xs,0); else launch_generic_q40<4>(w,xq,y,K,N,srx,xs,0); break;
                    case 5: if (sh.type==0) launch_generic_q6k<5>(w,xq,y,K,N,srx,xs,0); else launch_generic_q40<5>(w,xq,y,K,N,srx,xs,0); break;
                    case 6: if (sh.type==0) launch_generic_q6k<6>(w,xq,y,K,N,srx,xs,0); else launch_generic_q40<6>(w,xq,y,K,N,srx,xs,0); break;
                    case 7: if (sh.type==0) launch_generic_q6k<7>(w,xq,y,K,N,srx,xs,0); else launch_generic_q40<7>(w,xq,y,K,N,srx,xs,0); break;
                    case 8: if (sh.type==0) launch_generic_q6k<8>(w,xq,y,K,N,srx,xs,0); else launch_generic_q40<8>(w,xq,y,K,N,srx,xs,0); break;
                }
            };
            HIP_CHECK(hipMemset(y1, 0, (size_t) N * T * 4));
            gen(y1); HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(hy.data(), y1, (size_t) N * T * 4, hipMemcpyDeviceToHost));
            std::vector<float> hy2(hy);
            HIP_CHECK(hipMemset(y2, 0, (size_t) N * T * 4));
            gen(y2); HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipMemcpy(hy2.data(), y2, (size_t) N * T * 4, hipMemcpyDeviceToHost));
            const bool gen_det = hy == hy2;
            double us_gen = bench([&]{ gen(y1); }, quick ? 15 : 30);
            const double gbs_gen = wbytes / (us_gen * 1e-6) / 1e9;
            printf("T=%u %-26s generic %9.1f us  %7.1f GB/s (%4.1f%% of 1024) det=%d\n",
                   T, sh.tag, us_gen, gbs_gen, 100.0 * gbs_gen / 1024.0, (int) gen_det);

            // CPU oracle (widths 2, 5, 8; 512-row subset)
            if ((T == 2 || T == 5 || T == 8) && N >= 512) {
                const int nwarps = (sh.type == 0)
                    ? calc_nwarps((ggml_type)(sh.type==0?GGML_TYPE_Q6_K:GGML_TYPE_Q4_0), T, MMVQ_PARAMETERS_GCN) : 1;
                const int rpb = calc_rows_per_block(T, MMVQ_PARAMETERS_GCN, false, nwarps);
                long bad = 0;
                for (uint32_t r = 0; r < 512; ++r) {
                    for (uint32_t c = 0; c < T; ++c) {
                        const float o = generic_row_oracle(sh.type, r, c, srx, hw.data(), srx, hxq.data(), xs, T, rpb);
                        if (memcmp(&o, &hy[(size_t) c * N + r], 4) != 0) ++bad;
                    }
                }
                printf("   oracle(512 rows): mismatches=%ld %s\n", bad, bad ? "FAIL" : "PASS");
            }

            // roofline-read: same geometry, weights only, no dots (diagnostic)
            {
                auto ro = [&]() {
                    if (sh.type == 0) {
                        switch (T) {
#define RO6(TT) case TT: { \
        constexpr int nw = calc_nwarps(GGML_TYPE_Q6_K, TT, MMVQ_PARAMETERS_GCN); \
        constexpr int rp = calc_rows_per_block(TT, MMVQ_PARAMETERS_GCN, false, nw); \
        dim3 g((N + rp - 1) / rp, 1, 1), b(64, nw); \
        roofline_read_q6k<64*nw, 2*nw, (TT==1?1:2)><<<g, b>>>(w, y1, K, srx); } break;
                            RO6(1) RO6(2) RO6(3) RO6(4) RO6(5) RO6(6) RO6(7) RO6(8)
#undef RO6
                        }
                    } else {
                        switch (T) {
#define RO4(TT) case TT: { \
        constexpr int nw = calc_nwarps(GGML_TYPE_Q4_0, TT, MMVQ_PARAMETERS_GCN); \
        constexpr int rp = calc_rows_per_block(TT, MMVQ_PARAMETERS_GCN, false, nw); \
        dim3 g((N + rp - 1) / rp, 1, 1), b(64, nw); \
        roofline_read_q40<64*nw, 32*nw, (TT==1?1:2)><<<g, b>>>(w, y1, K, srx); } break;
                            RO4(1) RO4(2) RO4(3) RO4(4) RO4(5) RO4(6) RO4(7) RO4(8)
#undef RO4
                        }
                    }
                };
                double us_ro = bench(ro, quick ? 15 : 30);
                printf("   roofline-read  %9.1f us  %7.1f GB/s  (generic is %4.1f%% of read-roof)\n",
                       us_ro, wbytes / (us_ro * 1e-6) / 1e9, 100.0 * us_ro / us_gen);
            }

            // d1 arms: bit-exact vs generic + perf. variant table:
            //  Q6_K: 0=hoist u1, 1=hoist u2, 2=hoist u4, 3=raw u1, 4=raw u2, 5=raw u4
            //  Q4_0: 0=u1, 1=u2, 2=u4
            auto arm_d1 = [&](int variant, float * y) {
#define LQ6(TT, V) do { switch (V) { \
    case 0: launch_d1_q6k<TT,0,1>(w,xq,y,K,N,srx,xs,0); break; \
    case 1: launch_d1_q6k<TT,0,2>(w,xq,y,K,N,srx,xs,0); break; \
    case 2: launch_d1_q6k<TT,0,4>(w,xq,y,K,N,srx,xs,0); break; \
    case 3: launch_d1_q6k<TT,1,1>(w,xq,y,K,N,srx,xs,0); break; \
    case 4: launch_d1_q6k<TT,1,2>(w,xq,y,K,N,srx,xs,0); break; \
    case 5: launch_d1_q6k<TT,1,4>(w,xq,y,K,N,srx,xs,0); break; } } while (0)
#define LQ4(TT, V) do { switch (V) { \
    case 0: launch_d1_q40<TT,1>(w,xq,y,K,N,srx,xs,0); break; \
    case 1: launch_d1_q40<TT,2>(w,xq,y,K,N,srx,xs,0); break; \
    case 2: launch_d1_q40<TT,4>(w,xq,y,K,N,srx,xs,0); break; } } while (0)
                switch (T) {
                    case 1: if (sh.type==0) LQ6(1, variant); else LQ4(1, variant); break;
                    case 2: if (sh.type==0) LQ6(2, variant); else LQ4(2, variant); break;
                    case 3: if (sh.type==0) LQ6(3, variant); else LQ4(3, variant); break;
                    case 4: if (sh.type==0) LQ6(4, variant); else LQ4(4, variant); break;
                    case 5: if (sh.type==0) LQ6(5, variant); else LQ4(5, variant); break;
                    case 6: if (sh.type==0) LQ6(6, variant); else LQ4(6, variant); break;
                    case 7: if (sh.type==0) LQ6(7, variant); else LQ4(7, variant); break;
                    case 8: if (sh.type==0) LQ6(8, variant); else LQ4(8, variant); break;
                }
#undef LQ6
#undef LQ4
            };
            const int nvar = (sh.type == 0) ? 6 : 3;
            static const char * vname_q6k[6] = { "d1-hoist", "d1-hoist-u2", "d1-hoist-u4",
                                                 "d1-raw", "d1-raw-u2", "d1-raw-u4" };
            static const char * vname_q40[3] = { "d1-hoist", "d1-hoist-u2", "d1-hoist-u4" };
            for (int variant = 0; variant < nvar; ++variant) {
                const char * vname = (sh.type == 0) ? vname_q6k[variant] : vname_q40[variant];
                HIP_CHECK(hipMemset(y2, 0, (size_t) N * T * 4));
                arm_d1(variant, y2); HIP_CHECK(hipDeviceSynchronize());
                std::vector<float> hya((size_t) N * T);
                HIP_CHECK(hipMemcpy(hya.data(), y2, (size_t) N * T * 4, hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemset(y2, 0, (size_t) N * T * 4));
                arm_d1(variant, y2); HIP_CHECK(hipDeviceSynchronize());
                std::vector<float> hyb((size_t) N * T);
                HIP_CHECK(hipMemcpy(hyb.data(), y2, (size_t) N * T * 4, hipMemcpyDeviceToHost));
                const bool bit = hya == hy;
                const bool det = hya == hyb;
                double us = bench([&]{ arm_d1(variant, y2); }, quick ? 15 : 30);
                const double gbs = wbytes / (us * 1e-6) / 1e9;
                printf("   %-14s %9.1f us  %7.1f GB/s  vs generic %+6.1f%%  bit=%d det=%d\n",
                       vname, us, gbs, 100.0 * (us_gen / us - 1.0), (int) bit, (int) det);
            }
            HIP_CHECK(hipFree(w)); HIP_CHECK(hipFree(xq));
            HIP_CHECK(hipFree(y1)); HIP_CHECK(hipFree(y2));
        }
    }
    printf("MB-DONE\n");
    return 0;
}
