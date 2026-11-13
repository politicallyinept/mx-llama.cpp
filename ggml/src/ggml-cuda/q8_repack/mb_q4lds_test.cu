// mb_q4lds_test.cu - taskQ microbench: LDS-staged Q4_0 narrow mat-vec (port of
// the taskJ2 rp_q8_nc_lds pattern to the Q4_0 affine + dp8a decode arms) vs the
// current repacked nc shapes.
//
// Bars:
//   BIT: rp_q4_nc_lds<T,W,false,SUMW> == mul_mat_vec_repacked_nc<2,1,T,2,64,Q4_0>
//        and rp_q4_nc_lds<T,W,true,SUMW> == mul_mat_vec_repacked_nc<2,1,T,2,64,
//        Q4_0,true,false,false,false,SUMW> bit-exact (the port claim: same lanes,
//        block order, products, sums, reduction), and the affine pair == the CPU
//        oracle of the generic accumulation form.
//   DETERM: every kernel launch run twice, outputs must match.
//   PERF: median us/call + effective weight GB/s per arm at the Q4_0 shapes that
//        exist in Qwen3.8-27B-Q4_0.gguf (ffn 5120x17408, attn_qkv 5120x10240,
//        ffn_down 17408x5120, attn_gate/ssm 5120x6144). lm_head is Q6_K on this
//        file - NOT served by this kernel (coverage note in the ledger).
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

// Sum of the 32 raw excess-8 nibble codes in 16 bytes (host twin of nib_sum_u4).
static uint32_t nib_sum_bytes(const uint8_t * p) {
    uint32_t s = 0;
    for (int i = 0; i < 16; ++i) s += (p[i] & 0xFu) + (p[i] >> 4);
    return s;
}

// CPU mirror of the GENERIC repacked nc form (Q4_0 affine decode) for one
// (row, col): per-lane chain over strided blocks sb = lane, lane+64, ... with
//   pk = raw packed uint4; lo = pk & 0x0F..., hi = (pk >> 4) & 0x0F...
//   acc += d * dx * (float)(ilo + ihi) - (8*d) * sx
// then the rp_warp_reduce_sum<64> tree: xor offsets ASCENDING 1,2,4,8,16,32.
static float generic_affine_oracle(uint32_t row, uint32_t col, uint32_t nkb,
                                   const uint8_t * w, uint32_t qs_str, const uint8_t * dplane,
                                   const block_q8_1 * xq, uint32_t xs) {
    std::vector<float> v(64, 0.0f);
    const uint8_t * w_row = w + (size_t) row * qs_str;
    const uint16_t * dh_row = reinterpret_cast<const uint16_t *>(dplane) + (size_t) row * nkb;
    for (int lane = 0; lane < 64; ++lane) {
        float acc = 0.0f;
        for (uint32_t sb = lane; sb < nkb; sb += 64) {
            const int * pk = reinterpret_cast<const int *>(w_row + (size_t) sb * 16);
            const block_q8_1 * xb = xq + (size_t) col * xs + sb;
            const int * xv = reinterpret_cast<const int *>(xb->qs);
            const float d  = __half2float(*reinterpret_cast<const __half *>(&dh_row[sb]));
            const float d8 = 8.0f * d;
            const float dx = __low2float(xb->ds);
            const float sx = __high2float(xb->ds);
            int ilo = 0, ihi = 0;
            for (int i = 0; i < 4; ++i) {
                const int lo = pk[i] & 0x0F0F0F0F;
                const int hi = (pk[i] >> 4) & 0x0F0F0F0F;
                ilo = dp4a_cpu(lo, xv[i], ilo);
                ihi = dp4a_cpu(hi, xv[4 + i], ihi);
            }
            acc += d * dx * (float) (ilo + ihi) - d8 * sx;
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

struct Arms {
    uint8_t * dw; block_q8_1 * dx;
    float * d_gen; float * d_lds8; float * d_lds4;
    float * d_gen2; float * d_lds82;
    std::vector<float> y_gen, y_lds8, y_lds4, y_gen2, y_lds82;
};

// WT decode arms: 0 = stock q8_1 affine, 1 = U4ACT dp8a (SUMW plane on).
template <int T, int ARM>
static void launch_all(Arms & a, uint32_t K, uint32_t N, hipStream_t s, bool full) {
    const uint32_t nkb = K / 32;
    if constexpr (ARM == 0) {
        mul_mat_vec_repacked_nc<2, 1, T, 2, 64, GGML_TYPE_Q4_0>
            <<<dim3((N + 1) / 2), 64, 0, s>>>(a.dw, a.dx, a.d_gen, K, N, nkb, N);
        if (full) {
            mul_mat_vec_repacked_nc<2, 1, T, 2, 64, GGML_TYPE_Q4_0>
                <<<dim3((N + 1) / 2), 64, 0, s>>>(a.dw, a.dx, a.d_gen2, K, N, nkb, N);
        }
        rp_q4_nc_lds<T, 8><<<dim3((N + 15) / 16), 512, 0, s>>>(a.dw, a.dx, a.d_lds8, K, N, nkb, N);
        if (full) {
            rp_q4_nc_lds<T, 8><<<dim3((N + 15) / 16), 512, 0, s>>>(a.dw, a.dx, a.d_lds82, K, N, nkb, N);
            rp_q4_nc_lds<T, 4><<<dim3((N + 7) / 8), 256, 0, s>>>(a.dw, a.dx, a.d_lds4, K, N, nkb, N);
        }
    } else {
        // production U4ACT instantiation: SUMW plane on (speed profile SUMW=1)
        mul_mat_vec_repacked_nc<2, 1, T, 2, 64, GGML_TYPE_Q4_0, true, false, false, false, true>
            <<<dim3((N + 1) / 2), 64, 0, s>>>(a.dw, a.dx, a.d_gen, K, N, nkb, N);
        if (full) {
            mul_mat_vec_repacked_nc<2, 1, T, 2, 64, GGML_TYPE_Q4_0, true, false, false, false, true>
                <<<dim3((N + 1) / 2), 64, 0, s>>>(a.dw, a.dx, a.d_gen2, K, N, nkb, N);
        }
        rp_q4_nc_lds<T, 8, true, true><<<dim3((N + 15) / 16), 512, 0, s>>>(a.dw, a.dx, a.d_lds8, K, N, nkb, N);
        if (full) {
            rp_q4_nc_lds<T, 8, true, true><<<dim3((N + 15) / 16), 512, 0, s>>>(a.dw, a.dx, a.d_lds82, K, N, nkb, N);
            rp_q4_nc_lds<T, 4, true, true><<<dim3((N + 7) / 8), 256, 0, s>>>(a.dw, a.dx, a.d_lds4, K, N, nkb, N);
        }
    }
}

template <int T, int ARM>
static double time_arm(Arms & a, uint32_t K, uint32_t N, int which) {
    // which: 0 generic, 1 lds8, 2 lds4
    const uint32_t nkb = K / 32;
    auto one = [&]() {
        if constexpr (ARM == 0) {
            switch (which) {
                case 0: mul_mat_vec_repacked_nc<2, 1, T, 2, 64, GGML_TYPE_Q4_0>
                            <<<dim3((N + 1) / 2), 64>>>(a.dw, a.dx, a.d_gen, K, N, nkb, N); break;
                case 1: rp_q4_nc_lds<T, 8><<<dim3((N + 15) / 16), 512>>>(a.dw, a.dx, a.d_lds8, K, N, nkb, N); break;
                default: rp_q4_nc_lds<T, 4><<<dim3((N + 7) / 8), 256>>>(a.dw, a.dx, a.d_lds4, K, N, nkb, N);
            }
        } else {
            switch (which) {
                case 0: mul_mat_vec_repacked_nc<2, 1, T, 2, 64, GGML_TYPE_Q4_0, true, false, false, false, true>
                            <<<dim3((N + 1) / 2), 64>>>(a.dw, a.dx, a.d_gen, K, N, nkb, N); break;
                case 1: rp_q4_nc_lds<T, 8, true, true><<<dim3((N + 15) / 16), 512>>>(a.dw, a.dx, a.d_lds8, K, N, nkb, N); break;
                default: rp_q4_nc_lds<T, 4, true, true><<<dim3((N + 7) / 8), 256>>>(a.dw, a.dx, a.d_lds4, K, N, nkb, N);
            }
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

template <int T, int ARM>
static void run_cell(uint32_t K, uint32_t N, std::mt19937 & rng, bool oracle) {
    const uint32_t nkb = K / 32;
    // Q4_0 repack layout: qs plane rows of rs = repack_qs_row_stride(Q4_0, K)
    // BYTES (packed nibbles, K/2 + de-alias bump), then the f16 d plane, then
    // the int16 SUMW plane (geom::swplane_off).
    const uint32_t qs_str = repack_qs_row_stride(GGML_TYPE_Q4_0, K);
    const size_t dplane_off = (size_t) N * qs_str;
    const size_t swplane_off = dplane_off + (size_t) N * nkb * 2;
    const size_t wbytes = swplane_off + (size_t) N * nkb * 2;
    std::vector<uint8_t> w(wbytes, 0);
    std::uniform_int_distribution<int> qb(0, 255);
    std::uniform_real_distribution<float> sd(0.001f, 0.9f);
    for (uint32_t r = 0; r < N; ++r) {
        uint8_t * qs = w.data() + (size_t) r * qs_str;
        uint16_t * dh = reinterpret_cast<uint16_t *>(w.data() + dplane_off) + (size_t) r * nkb;
        int16_t * swp = reinterpret_cast<int16_t *>(w.data() + swplane_off) + (size_t) r * nkb;
        for (uint32_t b = 0; b < nkb; ++b) {
            for (int i = 0; i < 16; ++i) qs[b * 16 + i] = (uint8_t) qb(rng);
            const __half h = __float2half_rn(sd(rng));
            dh[b] = *reinterpret_cast<const uint16_t *>(&h);
            swp[b] = (int16_t) ((int) nib_sum_bytes(qs + b * 16) - 256);
        }
    }
    // Activation pool: affine = block_q8_1 with DISTINCT dx/sx halves (a swap
    // bug must fail the bit bar); u4 = block_q8_1_u4 (20 B) with random d4/m.
    std::vector<block_q8_1> x((size_t) T * nkb);
    if (ARM == 0) {
        for (size_t i = 0; i < x.size(); ++i) {
            for (int j = 0; j < 32; ++j) x[i].qs[j] = (int8_t) qb(rng);
            const __half hx = __float2half_rn(sd(rng));
            const __half hs = __float2half_rn(sd(rng));
            const __half2 h2 = __halves2half2(hx, hs);
            memcpy(&x[i].ds, &h2, sizeof(h2));
        }
    }
    const size_t xbytes = (size_t) T * nkb * (ARM == 0 ? sizeof(block_q8_1) : sizeof(block_q8_1_u4));
    Arms a;
    const size_t ybytes = (size_t) T * N * sizeof(float);
    HIP_CHECK(hipMalloc(&a.dw, wbytes));
    HIP_CHECK(hipMalloc(&a.dx, xbytes));
    HIP_CHECK(hipMalloc(&a.d_gen,  ybytes));
    HIP_CHECK(hipMalloc(&a.d_lds8, ybytes));
    HIP_CHECK(hipMalloc(&a.d_lds4, ybytes));
    HIP_CHECK(hipMalloc(&a.d_gen2, ybytes));
    HIP_CHECK(hipMalloc(&a.d_lds82, ybytes));
    HIP_CHECK(hipMemcpy(a.dw, w.data(), wbytes, hipMemcpyHostToDevice));
    if (ARM == 0) {
        HIP_CHECK(hipMemcpy(a.dx, x.data(), xbytes, hipMemcpyHostToDevice));
    } else {
        std::vector<block_q8_1_u4> x4((size_t) T * nkb);
        for (size_t i = 0; i < x4.size(); ++i) {
            for (int j = 0; j < 16; ++j) x4[i].qs[j] = (uint8_t) qb(rng);
            x4[i].d4 = __float2half_rn(sd(rng));
            x4[i].m  = __float2half_rn(sd(rng));
        }
        HIP_CHECK(hipMemcpy(a.dx, x4.data(), xbytes, hipMemcpyHostToDevice));
    }
    a.y_gen.resize(T * N); a.y_lds8.resize(T * N); a.y_lds4.resize(T * N);
    a.y_gen2.resize(T * N); a.y_lds82.resize(T * N);

    launch_all<T, ARM>(a, K, N, 0, true);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(a.y_gen.data(),   a.d_gen,   ybytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(a.y_lds8.data(),  a.d_lds8,  ybytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(a.y_lds4.data(),  a.d_lds4,  ybytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(a.y_gen2.data(),  a.d_gen2,  ybytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(a.y_lds82.data(), a.d_lds82, ybytes, hipMemcpyDeviceToHost));

    const bool bit_lds8 = memcmp(a.y_lds8.data(), a.y_gen.data(), ybytes) == 0;
    const bool bit_lds4 = memcmp(a.y_lds4.data(), a.y_gen.data(), ybytes) == 0;
    const bool det_gen  = memcmp(a.y_gen.data(),  a.y_gen2.data(),  ybytes) == 0;
    const bool det_lds  = memcmp(a.y_lds8.data(), a.y_lds82.data(), ybytes) == 0;
    long oracle_bad = 0;
    if (oracle && ARM == 0) {
        for (uint32_t c = 0; c < T; ++c) {
            for (uint32_t r = 0; r < N; ++r) {
                const float ref = generic_affine_oracle(r, c, nkb, w.data(), qs_str,
                                                        w.data() + dplane_off, x.data(), nkb);
                const size_t i = (size_t) c * N + r;
                if (memcmp(&a.y_gen[i], &ref, 4) != 0) ++oracle_bad;
            }
        }
    }
    const double wgt = (double) N * qs_str + (double) N * nkb * 2;
    const double t_gen  = time_arm<T, ARM>(a, K, N, 0);
    const double t_lds8 = time_arm<T, ARM>(a, K, N, 1);
    const double t_lds4 = time_arm<T, ARM>(a, K, N, 2);

    printf("T=%d %s K=%5u N=%6u | gen %8.1fus %6.1fGB/s | lds8 %8.1fus %6.1fGB/s (%+6.1f%%) | lds4 %8.1fus (%+6.1f%%) | "
           "bit8=%s bit4=%s det=%s oracle_bad=%ld\n",
           T, ARM == 0 ? "aff" : "u4 ", K, N,
           t_gen, wgt / t_gen / 1e3, t_lds8, wgt / t_lds8 / 1e3, 100.0 * (t_gen / t_lds8 - 1.0),
           t_lds4, 100.0 * (t_gen / t_lds4 - 1.0),
           bit_lds8 ? "OK" : "FAIL", bit_lds4 ? "OK" : "FAIL", (det_gen && det_lds) ? "OK" : "FAIL", oracle_bad);
    if (!bit_lds8 || !bit_lds4 || !det_gen || !det_lds || (oracle && ARM == 0 && oracle_bad != 0)) ++g_fails;

    HIP_CHECK(hipFree(a.dw)); HIP_CHECK(hipFree(a.dx));
    HIP_CHECK(hipFree(a.d_gen)); HIP_CHECK(hipFree(a.d_lds8));
    HIP_CHECK(hipFree(a.d_lds4)); HIP_CHECK(hipFree(a.d_gen2)); HIP_CHECK(hipFree(a.d_lds82));
}

// M=1/2 reference: current shapes only (no LDS arm below width 4).
template <int T>
static void run_ref(uint32_t K, uint32_t N, std::mt19937 & rng) {
    const uint32_t nkb = K / 32;
    const uint32_t qs_str = repack_qs_row_stride(GGML_TYPE_Q4_0, K);
    const size_t dplane_off = (size_t) N * qs_str;
    const size_t wbytes = dplane_off + (size_t) N * nkb * 2;
    std::vector<uint8_t> w(wbytes, 0);
    std::uniform_int_distribution<int> qb(0, 255);
    std::uniform_real_distribution<float> sd(0.001f, 0.9f);
    for (uint32_t r = 0; r < N; ++r) {
        uint8_t * qs = w.data() + (size_t) r * qs_str;
        uint16_t * dh = reinterpret_cast<uint16_t *>(w.data() + dplane_off) + (size_t) r * nkb;
        for (uint32_t b = 0; b < nkb; ++b) {
            for (int i = 0; i < 16; ++i) qs[b * 16 + i] = (uint8_t) qb(rng);
            const __half h = __float2half_rn(sd(rng));
            dh[b] = *reinterpret_cast<const uint16_t *>(&h);
        }
    }
    std::vector<block_q8_1> x((size_t) T * nkb);
    for (size_t i = 0; i < x.size(); ++i) {
        for (int j = 0; j < 32; ++j) x[i].qs[j] = (int8_t) qb(rng);
        const __half h = __float2half_rn(sd(rng));
        const __half2 h2 = __halves2half2(h, h);
        memcpy(&x[i].ds, &h2, sizeof(h2));
    }
    uint8_t * dw; block_q8_1 * dx; float * dy1; float * dy2;
    HIP_CHECK(hipMalloc(&dw, wbytes));
    HIP_CHECK(hipMalloc(&dx, x.size() * sizeof(block_q8_1)));
    HIP_CHECK(hipMalloc(&dy1, (size_t) T * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dy2, (size_t) T * N * sizeof(float)));
    HIP_CHECK(hipMemcpy(dw, w.data(), wbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, x.data(), x.size() * sizeof(block_q8_1), hipMemcpyHostToDevice));
    const double wgt = (double) N * qs_str + (double) N * nkb * 2;
    const double t1 = [&]() {  // generic affine
        auto f = [&]() { mul_mat_vec_repacked_nc<2, 1, T, 2, 64, GGML_TYPE_Q4_0><<<dim3((N + 1) / 2), 64>>>(dw, dx, dy1, K, N, nkb, N); };
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
    const double t2 = [&]() {  // generic U4ACT (SUMW plane NOT allocated: use SUMW=false twin)
        auto f = [&]() { mul_mat_vec_repacked_nc<2, 1, T, 2, 64, GGML_TYPE_Q4_0, true><<<dim3((N + 1) / 2), 64>>>(dw, dx, dy2, K, N, nkb, N); };
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
    printf("T=%d ref K=%5u N=%6u | gen-affine %8.1fus %6.1fGB/s | gen-u4 %8.1fus | LDS: n/a (gate covers 4-8)\n",
           T, K, N, t1, wgt / t1 / 1e3, t2);
    HIP_CHECK(hipFree(dw)); HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(dy1)); HIP_CHECK(hipFree(dy2));
}

int main(int argc, char ** argv) {
    const int dev = argc > 1 ? atoi(argv[1]) : 0;
    HIP_CHECK(hipSetDevice(dev));
    std::mt19937 rng(42);
    printf("mb_q4lds_test: LDS-staged Q4_0 nc (taskQ port) vs current repacked shapes\n");
    printf("shapes from Qwen3.8-27B-Q4_0.gguf: ffn 5120x17408, attn_qkv 5120x10240,\n"
           "ffn_down 17408x5120, attn_gate/ssm 5120x6144 (TP2 halves N per slice)\n");
    // oracle on the small shape only (CPU cost)
    run_cell<4, 0>(5120, 6144, rng, true);
    run_cell<5, 0>(5120, 6144, rng, false);
    run_cell<8, 0>(5120, 6144, rng, true);
    run_cell<4, 0>(5120, 17408, rng, false);
    run_cell<8, 0>(5120, 17408, rng, false);
    run_cell<4, 0>(5120, 10240, rng, false);
    run_cell<8, 0>(5120, 10240, rng, false);
    run_cell<4, 0>(17408, 5120, rng, false);
    run_cell<8, 0>(17408, 5120, rng, false);
    // U4ACT (floor-0) arm, SUMW plane on
    run_cell<4, 1>(5120, 6144, rng, false);
    run_cell<5, 1>(5120, 6144, rng, false);
    run_cell<8, 1>(5120, 6144, rng, false);
    run_cell<4, 1>(5120, 17408, rng, false);
    run_cell<5, 1>(5120, 17408, rng, false);
    run_cell<8, 1>(5120, 17408, rng, false);
    run_cell<5, 1>(5120, 10240, rng, false);
    run_cell<8, 1>(5120, 10240, rng, false);
    run_cell<5, 1>(17408, 5120, rng, false);
    run_cell<8, 1>(17408, 5120, rng, false);
    run_ref<1>(5120, 17408, rng);
    run_ref<2>(5120, 17408, rng);
    printf(g_fails == 0 ? "ALL BIT-MATCH BARS PASS\n" : "BIT-MATCH FAILURES: %d\n", g_fails);
    return g_fails == 0 ? 0 : 1;
}
