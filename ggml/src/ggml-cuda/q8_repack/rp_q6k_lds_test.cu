// rp_q6k_lds_test.cu - taskK3 microbench: LDS-staged canonical Q6_K narrow mat-vec
// (rp_q6k_nc_lds) vs the generic mmvq shape mul_mat_vec_q<Q6_K, T, false> on the
// GCN table (the kernel that serves the Q6_K lm_head at verify widths today).
//
// Bars:
//   BIT: rp_q6k_nc_lds<T,W,STEP,STAGE_W> == mul_mat_vec_q<Q6_K,T,false> BIT-EXACT
//        at every swept width T=5..8 (same lanes, kbx order, products, sums,
//        reduction), both STAGE_W arms.
//   ORACLE: generic == CPU mirror of the same accumulation (widths 5 and 8,
//        512-row subset).
//   DETERM: every arm launched twice, outputs must match.
//   PERF: median us/call + effective weight GB/s at the Q6_K shapes that exist in
//        Qwen3.8-27B-Q4_0.gguf: lm_head 5120x248320 (full) / 5120x124160 (per
//        TP2 card), attn_q 5120x6144. T=1/2/4 generic-only reference (the LDS
//        gate covers widths 5..8 only).
//
// Arithmetic model: host and device both compile -ffp-contract=off so the oracle
// and the kernels share one strict IEEE-754 binary32 sequence (taskE2/taskQ
// precedent). The generic kernel is extracted VERBATIM from mmvq.cu at build time
// (see k3-build-mb.sh) - no transcription. Standalone (own main + ggml_abort
// stub), amdclang++ -x hip per toolchain rule; run on ONE clean card under flock.
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
#include "q6k-lds.cuh"

// Link stub: the standalone binary does not link libggml; GGML_ABORT needs this.
#include <cstdarg>
void ggml_abort(const char* file, int line, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "ABORT %s:%d: ", file, line);
    vfprintf(stderr, fmt, ap); va_end(ap);
    fprintf(stderr, "\n"); exit(3);
}
// Link stubs: the extracted host helpers reference these libggml-base symbols but
// are never called by the harness.
extern "C" bool ggml_is_quantized(enum ggml_type type) { (void) type; return true; }
extern "C" void ggml_log_warn(const char * fmt, ...) { (void) fmt; }

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

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
// CPU mirror of vec_dot_q6_K_q8_1 (vecdotq.cuh) for one (block, iqs).
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
        d8[i] = __low2float(bq8_1[bq8_offset + 2*i].ds);
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
// CPU mirror of the generic mmvq row: per-lane chains kbx = lane/32, +2, ... then
// the warp_reduce_sum<64> xor tree (offsets 32,16,8,4,2,1).
static float q6k_generic_oracle(uint32_t row, uint32_t col, uint32_t n_blocks,
                                const block_q6_K * w, uint32_t srx,
                                const block_q8_1 * xq, uint32_t xs) {
    float v[64];
    for (int lane = 0; lane < 64; ++lane) {
        float acc = 0.0f;
        const int kqs = lane & 31;
        for (int kbx = lane >> 5; kbx < (int) n_blocks; kbx += 2) {
            acc += vec_dot_q6_K_q8_1_cpu(w + (size_t) row * srx + kbx,
                                         xq + (size_t) col * xs + (size_t) kbx * 8, kqs);
        }
        v[lane] = acc;
    }
    for (int off : {32, 16, 8, 4, 2, 1}) {
        float nv[64];
        for (int i = 0; i < 64; ++i) nv[i] = v[i] + v[i ^ off];
        memcpy(v, nv, sizeof(v));
    }
    return v[0];
}

// ---------------- launchers ----------------
// Generic: replicate calc_launch_params for Q6_K on the GCN table (plain matmul:
// one channel, one sample, no ids, no fusion).
template <int T, int GRPB1 = 0>
static void launch_generic(const uint8_t * w, const block_q8_1 * xq, float * y,
                           uint32_t K, uint32_t N, uint32_t srx, uint32_t xs,
                           uint32_t ys, hipStream_t s) {
    constexpr int nwarps = calc_nwarps(GGML_TYPE_Q6_K, T, MMVQ_PARAMETERS_GCN);
    int rpb = calc_rows_per_block(T, MMVQ_PARAMETERS_GCN, false, nwarps);
    if (T == 1 && GRPB1 > 0) rpb = GRPB1;
    dim3 grid((N + rpb - 1) / rpb, 1, 1), block(64, nwarps);
    ggml_cuda_mm_fusion_args_device fusion{};
    mul_mat_vec_q<GGML_TYPE_Q6_K, T, false, false, false, GRPB1><<<grid, block, 0, s>>>(
        w, xq, nullptr, fusion, y, K, make_uint3(1, 0, 0), srx, xs, ys,
        init_fastdiv_values(1), 0, 0, 0, init_fastdiv_values(1), 0, 0, 0, 0);
}

template <int T, int W, int STEP, bool STAGE_W>
static void launch_lds(const uint8_t * w, const block_q8_1 * xq, float * y,
                       uint32_t K, uint32_t N, uint32_t srx, uint32_t xs,
                       uint32_t ys, hipStream_t s) {
    dim3 grid(((N + 1) / 2 + W - 1) / W, 1, 1), block(64, W);
    rp_q6k_nc_lds<T, W, STEP, STAGE_W, 2><<<grid, block, 0, s>>>(w, xq, y, K, N, srx, xs, ys);
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

static uint64_t fnv(const float * p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n * 4; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

// ---------------- data fill ----------------
static void fill_random(uint8_t * w, block_q8_1 * xq, uint32_t K, uint32_t N,
                        uint32_t xs, uint32_t T, uint32_t seed) {
    std::mt19937 rng(seed);
    const uint32_t nb = K / 256;
    std::uniform_int_distribution<int> byte_d(0, 255);
    std::uniform_real_distribution<float> dscale(0.001f, 0.5f);
    block_q6_K * wq = (block_q6_K *) w;
    for (size_t i = 0; i < (size_t) N * nb; ++i) {
        for (int j = 0; j < 128; ++j) wq[i].ql[j] = byte_d(rng);
        for (int j = 0; j < 64;  ++j) wq[i].qh[j] = byte_d(rng);
        for (int j = 0; j < 16;  ++j) wq[i].scales[j] = (int8_t) byte_d(rng);
        wq[i].d = __float2half(dscale(rng));   // keep d finite + positive (no NaN class)
    }
    std::uniform_real_distribution<float> act_d(0.001f, 2.0f);
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t b = 0; b < xs; ++b) {
            block_q8_1 * xb = xq + (size_t) t * xs + b;
            xb->ds = __floats2half2_rn(act_d(rng), (float) ((int)(rng() % 64) - 32));
            for (int j = 0; j < 32; ++j) xb->qs[j] = (int8_t) byte_d(rng);
        }
    }
}

template <int T, int W, int STEP, bool SW>
void ldsfn(const uint8_t* w, const block_q8_1* x, float* y, uint32_t K, uint32_t N, uint32_t srx, uint32_t xs, uint32_t ys, hipStream_t s) {
    launch_lds<T, W, STEP, SW>(w, x, y, K, N, srx, xs, ys, s);
}

int main(int argc, char ** argv) {
    const bool quick = argc > 1 && !strcmp(argv[1], "--quick");
    int dev; HIP_CHECK(hipSetDevice(0)); HIP_CHECK(hipGetDevice(&dev));
    printf("# taskK3 Q6_K LDS microbench, device %d\n", dev);

    struct Shape { uint32_t K, N; const char * tag; };
    std::vector<Shape> shapes = {
        { 5120, 6144,    "attn_q 5120x6144" },
        { 5120, 124160,  "head/card 5120x124160 (TP2)" },
        { 5120, 248320,  "head 5120x248320 (full)" },
    };
    const uint32_t xs = 5120 / 32;  // q8_1 blocks per token (K padded to 5120)

    for (auto & sh : shapes) {
        const uint32_t K = sh.K, N = sh.N, srx = K / 256;
        const size_t wbytes = (size_t) N * srx * 210;
        for (uint32_t T : {1u, 2u, 4u, 5u, 6u, 7u, 8u}) {
            if (quick && T != 5 && T != 8) continue;
            uint8_t * w; block_q8_1 * xq; float * y_gen; float * y_arm; float * y_arm2;
            HIP_CHECK(hipMalloc(&w, wbytes));
            HIP_CHECK(hipMalloc(&xq, (size_t) T * xs * sizeof(block_q8_1)));
            HIP_CHECK(hipMalloc(&y_gen, (size_t) N * T * 4));
            HIP_CHECK(hipMalloc(&y_arm, (size_t) N * T * 4));
            HIP_CHECK(hipMalloc(&y_arm2, (size_t) N * T * 4));
            std::vector<uint8_t> hw(wbytes); std::vector<block_q8_1> hxq((size_t) T * xs);
            fill_random(hw.data(), hxq.data(), K, N, xs, T, 1234 + T);
            HIP_CHECK(hipMemcpy(w, hw.data(), wbytes, hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(xq, hxq.data(), (size_t) T * xs * sizeof(block_q8_1), hipMemcpyHostToDevice));

            auto gen = [&](float * y) {
                switch (T) {
                    case 1: launch_generic<1, 2>(w, xq, y, K, N, srx, xs, N, 0); break;
                    case 2: launch_generic<2>(w, xq, y, K, N, srx, xs, N, 0); break;
                    case 4: launch_generic<4>(w, xq, y, K, N, srx, xs, N, 0); break;
                    case 5: launch_generic<5>(w, xq, y, K, N, srx, xs, N, 0); break;
                    case 6: launch_generic<6>(w, xq, y, K, N, srx, xs, N, 0); break;
                    case 7: launch_generic<7>(w, xq, y, K, N, srx, xs, N, 0); break;
                    case 8: launch_generic<8>(w, xq, y, K, N, srx, xs, N, 0); break;
                }
            };
            HIP_CHECK(hipMemset(y_gen, 0, (size_t) N * T * 4));
            gen(y_gen); HIP_CHECK(hipDeviceSynchronize());
            double us_gen = bench([&]{ gen(y_gen); }, quick ? 15 : 30);
            std::vector<float> hy((size_t) N * T);
            HIP_CHECK(hipMemcpy(hy.data(), y_gen, (size_t) N * T * 4, hipMemcpyDeviceToHost));
            // determinism of the reference
            HIP_CHECK(hipMemset(y_arm, 0, (size_t) N * T * 4));
            gen(y_arm); HIP_CHECK(hipDeviceSynchronize());
            std::vector<float> hy2((size_t) N * T);
            HIP_CHECK(hipMemcpy(hy2.data(), y_arm, (size_t) N * T * 4, hipMemcpyDeviceToHost));
            const bool gen_det = hy == hy2;

            const double gbs_gen = wbytes / (us_gen * 1e-6) / 1e9;
            printf("T=%u %-28s generic %9.1f us  %7.1f GB/s (%4.1f%% of 1024) det=%d\n",
                   T, sh.tag, us_gen, gbs_gen, 100.0 * gbs_gen / 1024.0, (int) gen_det);

            // CPU oracle on a 512-row subset (widths 5 and 8)
            if ((T == 5 || T == 8) && N >= 512) {
                long bad = 0;
                for (uint32_t r = 0; r < 512; ++r) {
                    for (uint32_t c = 0; c < T; ++c) {
                        const float o = q6k_generic_oracle(r, c, srx,
                            (const block_q6_K *) hw.data(), srx, hxq.data(), xs);
                        if (memcmp(&o, &hy[(size_t) c * N + r], 4) != 0) ++bad;
                    }
                }
                printf("   oracle(512 rows): mismatches=%ld %s\n", bad, bad ? "FAIL" : "PASS");
            }

            if (T >= 5) {
                const char * cfgs[] = {
                    "lds W2 S2 act", "lds W4 S2 act", "lds W8 S2 act", "lds W4 S4 act",
                    "lds W2 S2 full", "lds W4 S2 full", "lds W8 S2 full", "lds W4 S4 full",
                };
                for (auto & name : cfgs) {
                    auto arm = [&](float * y) {
                        #define GOQ(TT) \
                            if (T == TT) { \
                                if (!strcmp(name,"lds W2 S2 act"))  { auto f = ldsfn<TT,2,2,false>; f(w,xq,y,K,N,srx,xs,N,0); } \
                                if (!strcmp(name,"lds W4 S2 act"))  { auto f = ldsfn<TT,4,2,false>; f(w,xq,y,K,N,srx,xs,N,0); } \
                                if (!strcmp(name,"lds W8 S2 act"))  { auto f = ldsfn<TT,8,2,false>; f(w,xq,y,K,N,srx,xs,N,0); } \
                                if (!strcmp(name,"lds W4 S4 act"))  { auto f = ldsfn<TT,4,4,false>; f(w,xq,y,K,N,srx,xs,N,0); } \
                                if (!strcmp(name,"lds W2 S2 full")) { auto f = ldsfn<TT,2,2,true>;  f(w,xq,y,K,N,srx,xs,N,0); } \
                                if (!strcmp(name,"lds W4 S2 full")) { auto f = ldsfn<TT,4,2,true>;  f(w,xq,y,K,N,srx,xs,N,0); } \
                                if (!strcmp(name,"lds W8 S2 full")) { auto f = ldsfn<TT,8,2,true>;  f(w,xq,y,K,N,srx,xs,N,0); } \
                                if (!strcmp(name,"lds W4 S4 full")) { auto f = ldsfn<TT,4,4,true>;  f(w,xq,y,K,N,srx,xs,N,0); } \
                            }
                        GOQ(5) GOQ(6) GOQ(7) GOQ(8)
                        #undef GOQ
                        #undef GO
                    };
                    HIP_CHECK(hipMemset(y_arm, 0, (size_t) N * T * 4));
                    arm(y_arm); HIP_CHECK(hipDeviceSynchronize());
                    HIP_CHECK(hipMemset(y_arm2, 0, (size_t) N * T * 4));
                    arm(y_arm2); HIP_CHECK(hipDeviceSynchronize());
                    std::vector<float> hya((size_t) N * T), hyb((size_t) N * T);
                    HIP_CHECK(hipMemcpy(hya.data(), y_arm, (size_t) N * T * 4, hipMemcpyDeviceToHost));
                    HIP_CHECK(hipMemcpy(hyb.data(), y_arm2, (size_t) N * T * 4, hipMemcpyDeviceToHost));
                    const bool bit = hya == hy;   // vs generic
                    const bool det = hya == hyb;
                    double us = bench([&]{ arm(y_arm); }, quick ? 15 : 30);
                    const double gbs = wbytes / (us * 1e-6) / 1e9;
                    printf("   %-16s %9.1f us  %7.1f GB/s  vs generic %+6.1f%%  bit=%d det=%d\n",
                           name, us, gbs, 100.0 * (us_gen / us - 1.0), (int) bit, (int) det);
                }
            }
            HIP_CHECK(hipFree(w)); HIP_CHECK(hipFree(xq));
            HIP_CHECK(hipFree(y_gen)); HIP_CHECK(hipFree(y_arm)); HIP_CHECK(hipFree(y_arm2));
        }
    }
    printf("MB-DONE\n");
    return 0;
}
