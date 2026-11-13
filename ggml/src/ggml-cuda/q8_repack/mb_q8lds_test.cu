// mb_q8lds_test.cu - taskJ2 microbench: LDS-staged Q8_0 narrow mat-vec (port of
// ruquant 82c73b5 rq_q8_nc) vs the current repacked nc shapes.
//
// Bars:
//   BIT: rp_q8_nc_lds<T,W> == mul_mat_vec_repacked_nc<2,1,T,2,64,Q8_0> bit-exact
//        (the port claim: same lanes, block order, products, sums, reduction),
//        and both == the CPU oracle of the generic accumulation form.
//   DIFF: rp_q8_nc_lds vs the RESTYLE default mul_mat_vec_q8_0_repacked8_nc is
//        EXPECTED to differ (different accumulation granularity) - reported, not
//        a failure.
//   DETERM: every kernel launch run twice, outputs must match.
//   PERF: median us/call + effective weight GB/s per arm at the shapes the
//        commit claims (ssm_out 5120x6144 at 8 tokens; lm_head tall at 4-8).
//   M=1,2: no LDS arm exists (gate dispatches widths 4-8 only); the current
//        shapes are timed for reference.
//
// Arithmetic model: host and device both compile -ffp-contract=off so the oracle
// and the kernels share one strict IEEE-754 binary32 sequence (taskE2 precedent).
// Standalone (own main + ggml_abort stub), hipcc per toolchain rule; run on ONE
// idle card under flock.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <hip/hip_runtime.h>
#include "ggml.h"
#include "ggml-cuda/common.cuh"
#include "repack-kernels.cuh"

// Link stub: the standalone binary does not link libggml; GGML_ABORT needs this.
#include <cstdarg>
void ggml_abort(const char* file, int line, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "ABORT %s:%d: ", file, line);
    vfprintf(stderr, fmt, ap); va_end(ap);
    fprintf(stderr, "\n"); exit(3);
}

#define HIP_CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

static inline int dp4a_cpu(int a, int b, int c) {
    int s = c;
    for (int i = 0; i < 4; ++i) {
        s += (int)(int8_t)((a >> (8*i)) & 0xFF) * (int)(int8_t)((b >> (8*i)) & 0xFF);
    }
    return s;
}

// CPU mirror of the GENERIC repacked nc form for one (row, col): per-lane chain
// over strided blocks sb = lane, lane+64, ... with acc += dw*dx*(float)(ilo+ihi),
// then the rp_warp_reduce_sum<64> tree: xor offsets ASCENDING 1,2,4,8,16,32.
static float generic_oracle(uint32_t row, uint32_t col, uint32_t nkb,
                            const uint8_t * w, uint32_t qs_str, const uint8_t * dplane,
                            const block_q8_1 * xq, uint32_t xs) {
    std::vector<float> v(64, 0.0f);
    const int * w_row = reinterpret_cast<const int *>(w + (size_t) row * qs_str);
    const uint16_t * dh_row = reinterpret_cast<const uint16_t *>(dplane) + (size_t) row * nkb;
    for (int lane = 0; lane < 64; ++lane) {
        float acc = 0.0f;
        for (uint32_t sb = lane; sb < nkb; sb += 64) {
            const block_q8_1 * xb = xq + (size_t) col * xs + sb;
            const int * xv = reinterpret_cast<const int *>(xb->qs);
            const float dw = __half2float(*reinterpret_cast<const __half *>(&dh_row[sb]));
            const float dx = __low2float(xb->ds);
            int ilo = 0, ihi = 0;
            for (int i = 0; i < 4; ++i) ilo = dp4a_cpu(w_row[8*sb + i], xv[i], ilo);
            for (int i = 4; i < 8; ++i) ihi = dp4a_cpu(w_row[8*sb + i], xv[i], ihi);
            acc += dw * dx * (float) (ilo + ihi);
        }
        v[lane] = acc;
    }
    for (int off : {1, 2, 4, 8, 16, 32}) {
        std::vector<float> nv(64);
        for (int i = 0; i < 64; ++i) nv[i] = v[i] + v[i ^ off];
        v = nv;
    }
    return v[0];
}

// shipped RESTYLE nc warp table: 2 warps at widths 2-5, 1 warp at 6-8
static int restyle_nw(int T) { return T <= 5 ? 2 : 1; }

struct Arms {
    uint8_t * dw; block_q8_1 * dx;
    float * d_gen; float * d_lds8; float * d_lds4; float * d_r8;
    float * d_gen2; float * d_lds82;
    std::vector<float> y_gen, y_lds8, y_lds4, y_r8, y_gen2, y_lds82;
};

template <int T>
static void launch_all(Arms & a, uint32_t K, uint32_t N, hipStream_t s, bool full) {
    const uint32_t nkb = K / 32;
    // generic <2,1,T,2,64>
    mul_mat_vec_repacked_nc<2, 1, T, 2, 64, GGML_TYPE_Q8_0>
        <<<dim3((N + 1) / 2), 64, 0, s>>>(a.dw, a.dx, a.d_gen, K, N, nkb, N);
    if (full) {
        mul_mat_vec_repacked_nc<2, 1, T, 2, 64, GGML_TYPE_Q8_0>
            <<<dim3((N + 1) / 2), 64, 0, s>>>(a.dw, a.dx, a.d_gen2, K, N, nkb, N);
    }
    // LDS W=8 / W=4
    rp_q8_nc_lds<T, 8><<<dim3((N + 15) / 16), 512, 0, s>>>(a.dw, a.dx, a.d_lds8, K, N, nkb, N);
    if (full) {
        rp_q8_nc_lds<T, 8><<<dim3((N + 15) / 16), 512, 0, s>>>(a.dw, a.dx, a.d_lds82, K, N, nkb, N);
        rp_q8_nc_lds<T, 4><<<dim3((N + 7) / 8), 256, 0, s>>>(a.dw, a.dx, a.d_lds4, K, N, nkb, N);
    }
    // RESTYLE default
    if (T <= 5) {
        mul_mat_vec_q8_0_repacked8_nc<2, T><<<dim3(N), dim3(64, 2), 0, s>>>(a.dw, a.dx, a.d_r8, K, N, nkb, N);
    } else {
        mul_mat_vec_q8_0_repacked8_nc<1, T><<<dim3(N), dim3(64, 1), 0, s>>>(a.dw, a.dx, a.d_r8, K, N, nkb, N);
    }
}

template <int T>
static double time_arm(Arms & a, uint32_t K, uint32_t N, int which) {
    // which: 0 generic, 1 lds8, 2 lds4, 3 restyle
    const uint32_t nkb = K / 32;
    auto one = [&]() {
        switch (which) {
            case 0: mul_mat_vec_repacked_nc<2, 1, T, 2, 64, GGML_TYPE_Q8_0>
                        <<<dim3((N + 1) / 2), 64>>>(a.dw, a.dx, a.d_gen, K, N, nkb, N); break;
            case 1: rp_q8_nc_lds<T, 8><<<dim3((N + 15) / 16), 512>>>(a.dw, a.dx, a.d_lds8, K, N, nkb, N); break;
            case 2: rp_q8_nc_lds<T, 4><<<dim3((N + 7) / 8), 256>>>(a.dw, a.dx, a.d_lds4, K, N, nkb, N); break;
            default:
                if (T <= 5) mul_mat_vec_q8_0_repacked8_nc<2, T><<<dim3(N), dim3(64, 2)>>>(a.dw, a.dx, a.d_r8, K, N, nkb, N);
                else        mul_mat_vec_q8_0_repacked8_nc<1, T><<<dim3(N), dim3(64, 1)>>>(a.dw, a.dx, a.d_r8, K, N, nkb, N);
        }
    };
    for (int i = 0; i < 10; ++i) one();
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<double> reps;
    for (int r = 0; r < 7; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 30; ++i) one();
        HIP_CHECK(hipDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        reps.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / 30.0);
    }
    std::sort(reps.begin(), reps.end());
    return reps[reps.size() / 2];
}

static int g_fails = 0;

template <int T>
static void run_cell(uint32_t K, uint32_t N, std::mt19937 & rng, bool oracle) {
    const uint32_t nkb = K / 32;
    const uint32_t qs_str = repack_qs_stride(K);
    const size_t wbytes = (size_t) N * qs_str + (size_t) N * nkb * 2;
    std::vector<uint8_t> w(wbytes, 0);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> sd(0.001f, 0.9f);
    for (uint32_t r = 0; r < N; ++r) {
        int8_t * qs = reinterpret_cast<int8_t *>(w.data() + (size_t) r * qs_str);
        uint16_t * dh = reinterpret_cast<uint16_t *>(w.data() + (size_t) N * qs_str) + (size_t) r * nkb;
        for (uint32_t b = 0; b < nkb; ++b) {
            for (int i = 0; i < 32; ++i) qs[b * 32 + i] = (int8_t) qd(rng);
            const __half h = __float2half_rn(sd(rng));
            dh[b] = *reinterpret_cast<const uint16_t *>(&h);
        }
    }
    std::vector<block_q8_1> x((size_t) T * nkb);
    for (size_t i = 0; i < x.size(); ++i) {
        for (int j = 0; j < 32; ++j) x[i].qs[j] = (int8_t) qd(rng);
        const __half h = __float2half_rn(sd(rng));
        const __half2 h2 = __halves2half2(h, h);
        memcpy(&x[i].ds, &h2, sizeof(h2));
    }
    Arms a;
    const size_t ybytes = (size_t) T * N * sizeof(float);
    HIP_CHECK(hipMalloc(&a.dw, wbytes));
    HIP_CHECK(hipMalloc(&a.dx, x.size() * sizeof(block_q8_1)));
    HIP_CHECK(hipMalloc(&a.d_gen,  ybytes));
    HIP_CHECK(hipMalloc(&a.d_lds8, ybytes));
    HIP_CHECK(hipMalloc(&a.d_lds4, ybytes));
    HIP_CHECK(hipMalloc(&a.d_r8,   ybytes));
    HIP_CHECK(hipMalloc(&a.d_gen2, ybytes));
    HIP_CHECK(hipMalloc(&a.d_lds82, ybytes));
    HIP_CHECK(hipMemcpy(a.dw, w.data(), wbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(a.dx, x.data(), x.size() * sizeof(block_q8_1), hipMemcpyHostToDevice));
    a.y_gen.resize(T * N); a.y_lds8.resize(T * N); a.y_lds4.resize(T * N); a.y_r8.resize(T * N);
    a.y_gen2.resize(T * N); a.y_lds82.resize(T * N);

    launch_all<T>(a, K, N, 0, true);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(a.y_gen.data(),   a.d_gen,   ybytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(a.y_lds8.data(),  a.d_lds8,  ybytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(a.y_lds4.data(),  a.d_lds4,  ybytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(a.y_r8.data(),    a.d_r8,    ybytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(a.y_gen2.data(),  a.d_gen2,  ybytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(a.y_lds82.data(), a.d_lds82, ybytes, hipMemcpyDeviceToHost));

    // bit-match bars
    const bool bit_lds8 = memcmp(a.y_lds8.data(), a.y_gen.data(), ybytes) == 0;
    const bool bit_lds4 = memcmp(a.y_lds4.data(), a.y_gen.data(), ybytes) == 0;
    const bool det_gen  = memcmp(a.y_gen.data(),  a.y_gen2.data(),  ybytes) == 0;
    const bool det_lds  = memcmp(a.y_lds8.data(), a.y_lds82.data(), ybytes) == 0;
    const bool diff_r8  = memcmp(a.y_lds8.data(), a.y_r8.data(), ybytes) != 0;
    long oracle_bad = 0;
    if (oracle) {
        for (uint32_t c = 0; c < T; ++c) {
            for (uint32_t r = 0; r < N; ++r) {
                const float ref = generic_oracle(r, c, nkb, w.data(), qs_str,
                                                 w.data() + (size_t) N * qs_str, x.data(), nkb);
                const size_t i = (size_t) c * N + r;
                if (memcmp(&a.y_gen[i], &ref, 4) != 0) ++oracle_bad;
            }
        }
    }
    // perf
    const double wgt = (double) N * K + (double) N * nkb * 2;
    const double t_gen  = time_arm<T>(a, K, N, 0);
    const double t_lds8 = time_arm<T>(a, K, N, 1);
    const double t_lds4 = time_arm<T>(a, K, N, 2);
    const double t_r8   = time_arm<T>(a, K, N, 3);

    printf("T=%d K=%5u N=%6u | gen %8.1fus %6.1fGB/s | lds8 %8.1fus %6.1fGB/s (%+6.1f%%) | lds4 %8.1fus (%+6.1f%%) | r8 %8.1fus | "
           "bit8=%s bit4=%s det=%s vsR8diff=%s oracle_bad=%ld\n",
           T, K, N,
           t_gen, wgt / t_gen / 1e3, t_lds8, wgt / t_lds8 / 1e3, 100.0 * (t_gen / t_lds8 - 1.0),
           t_lds4, 100.0 * (t_gen / t_lds4 - 1.0), t_r8,
           bit_lds8 ? "OK" : "FAIL", bit_lds4 ? "OK" : "FAIL", (det_gen && det_lds) ? "OK" : "FAIL",
           diff_r8 ? "yes(exp)" : "NO(?!)", oracle_bad);
    if (!bit_lds8 || !bit_lds4 || !det_gen || !det_lds || (oracle && oracle_bad != 0)) ++g_fails;

    HIP_CHECK(hipFree(a.dw)); HIP_CHECK(hipFree(a.dx));
    HIP_CHECK(hipFree(a.d_gen)); HIP_CHECK(hipFree(a.d_lds8));
    HIP_CHECK(hipFree(a.d_lds4)); HIP_CHECK(hipFree(a.d_r8));
    HIP_CHECK(hipFree(a.d_gen2)); HIP_CHECK(hipFree(a.d_lds82));
}

// M=1 reference: current single-token shapes only (no LDS arm exists).
static void run_m1(uint32_t K, uint32_t N, std::mt19937 & rng) {
    const uint32_t nkb = K / 32;
    const uint32_t qs_str = repack_qs_stride(K);
    const size_t wbytes = (size_t) N * qs_str + (size_t) N * nkb * 2;
    std::vector<uint8_t> w(wbytes, 0);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> sd(0.001f, 0.9f);
    for (uint32_t r = 0; r < N; ++r) {
        int8_t * qs = reinterpret_cast<int8_t *>(w.data() + (size_t) r * qs_str);
        uint16_t * dh = reinterpret_cast<uint16_t *>(w.data() + (size_t) N * qs_str) + (size_t) r * nkb;
        for (uint32_t b = 0; b < nkb; ++b) {
            for (int i = 0; i < 32; ++i) qs[b * 32 + i] = (int8_t) qd(rng);
            const __half h = __float2half_rn(sd(rng));
            dh[b] = *reinterpret_cast<const uint16_t *>(&h);
        }
    }
    std::vector<block_q8_1> x(nkb);
    for (uint32_t b = 0; b < nkb; ++b) {
        for (int j = 0; j < 32; ++j) x[b].qs[j] = (int8_t) qd(rng);
        const __half h = __float2half_rn(sd(rng));
        const __half2 h2 = __halves2half2(h, h);
        memcpy(&x[b].ds, &h2, sizeof(h2));
    }
    uint8_t * dw; block_q8_1 * dx; float * dy1; float * dy2;
    HIP_CHECK(hipMalloc(&dw, wbytes));
    HIP_CHECK(hipMalloc(&dx, nkb * sizeof(block_q8_1)));
    HIP_CHECK(hipMalloc(&dy1, (size_t) N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dy2, (size_t) N * sizeof(float)));
    HIP_CHECK(hipMemcpy(dw, w.data(), wbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, x.data(), nkb * sizeof(block_q8_1), hipMemcpyHostToDevice));
    const double wgt = (double) N * K + (double) N * nkb * 2;
    const double t1 = [&]() {
        auto f = [&]() { mul_mat_vec_q8_0_repacked8<2><<<dim3(N), dim3(64, 2)>>>(dw, dx, dy1, K, N, nullptr, nullptr, nullptr, GGML_GLU_OP_REGLU); };
        for (int i = 0; i < 10; ++i) f(); HIP_CHECK(hipDeviceSynchronize());
        std::vector<double> reps;
        for (int r = 0; r < 7; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 30; ++i) f();
            HIP_CHECK(hipDeviceSynchronize());
            reps.push_back(std::chrono::duration<double, std::micro>(
                std::chrono::high_resolution_clock::now() - t0).count() / 30.0);
        }
        std::sort(reps.begin(), reps.end()); return reps[3];
    }();
    const double t2 = [&]() {
        auto f = [&]() { mul_mat_vec_repacked_nc<2, 1, 1, 2, 64, GGML_TYPE_Q8_0><<<dim3((N + 1) / 2), 64>>>(dw, dx, dy2, K, N, nkb, N); };
        for (int i = 0; i < 10; ++i) f(); HIP_CHECK(hipDeviceSynchronize());
        std::vector<double> reps;
        for (int r = 0; r < 7; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 30; ++i) f();
            HIP_CHECK(hipDeviceSynchronize());
            reps.push_back(std::chrono::duration<double, std::micro>(
                std::chrono::high_resolution_clock::now() - t0).count() / 30.0);
        }
        std::sort(reps.begin(), reps.end()); return reps[3];
    }();
    printf("T=1 K=%5u N=%6u | r8(MMV) %8.1fus %6.1fGB/s | gen<2,1,1,2,64> %8.1fus | LDS: n/a (gate covers 4-8)\n",
           K, N, t1, wgt / t1 / 1e3, t2);
    HIP_CHECK(hipFree(dw)); HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dy1)); HIP_CHECK(hipFree(dy2));
}

int main(int argc, char ** argv) {
    const int dev = argc > 1 ? atoi(argv[1]) : 0;
    HIP_CHECK(hipSetDevice(dev));
    std::mt19937 rng(42);
    printf("mb_q8lds_test: LDS-staged Q8_0 nc (taskJ2 port) vs current repacked shapes\n");
    printf("shapes: ssm_out 5120x6144 (the claimed one), 5120x5120, 2048x6144, lm_head 5120x151936\n");
    // oracle on the small shapes only (CPU cost)
    run_cell<4>(5120, 6144, rng, true);
    run_cell<5>(5120, 6144, rng, false);
    run_cell<8>(5120, 6144, rng, true);
    run_cell<2>(5120, 6144, rng, false);   // exploratory: LDS below the gate window
    run_cell<4>(2048, 6144, rng, false);
    run_cell<8>(2048, 6144, rng, false);
    run_cell<4>(5120, 5120, rng, false);
    run_cell<8>(5120, 5120, rng, false);
    run_cell<4>(5120, 151936, rng, false); // lm_head tall
    run_cell<8>(5120, 151936, rng, false);
    run_m1(5120, 6144, rng);
    run_m1(5120, 151936, rng);
    printf(g_fails == 0 ? "ALL BIT-MATCH BARS PASS\n" : "BIT-MATCH FAILURES: %d\n", g_fails);
    return g_fails == 0 ? 0 : 1;
}
