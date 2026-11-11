// mb_i8narrow.cc - Task C GPU microbench: mul_mat_i8narrow_q40 timings + device-vs-host bit-match.
// Standalone (own main); run on ONE idle card under flock. Shapes: K=5120/N=17408 and
// K=17408/N=5120, widths T=1/2/4/8/16 (NWAVES=4, KSPLIT=1, INTERLEAVED=false).
// Bars: T8 <= 1.5x T1; outputs vs CPU reference (exact integer identity, fp32 epilogue
// order replicated; tolerance 1e-3 rel as in seal_i8split).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <hip/hip_runtime.h>
#include "ggml.h"
#include "ggml-cuda/common.cuh"
#include "w4a4-i8narrow.cu"

typedef uint16_t ggml_half_pod;
// Link stub: the standalone binary does not link libggml; GGML_ABORT needs this.
#include <cstdarg>
void ggml_abort(const char* file, int line, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "ABORT %s:%d: ", file, line);
    vfprintf(stderr, fmt, ap); va_end(ap);
    fprintf(stderr, "\n"); exit(3);
}
static inline float half2float_h(uint16_t h) {
    __half hr = *reinterpret_cast<const __half*>(&h);
    return __half2float(hr);
}

struct Shapes { uint32_t K, N; };

// CPU reference: exact same fp32 op sequence as the kernel epilogue, sb-order accumulation.
static void cpu_ref(uint32_t K, uint32_t N, const std::vector<uint8_t>& qs, uint32_t rs,
                    const std::vector<uint16_t>& dplane,
                    const std::vector<uint8_t>& hi, const std::vector<uint8_t>& lo,
                    const std::vector<uint16_t>& ad, uint32_t nkb, uint32_t T,
                    std::vector<float>& y) {
    y.assign((size_t)T * N, 0.0f);
    for (uint32_t c = 0; c < T; ++c) {
        for (uint32_t row = 0; row < N; ++row) {
            float acc = 0.0f;
            const uint8_t* qrow = &qs[(size_t)row * rs];
            for (uint32_t sb = 0; sb < nkb; ++sb) {
                const uint8_t* p = qrow + (size_t)sb * 16;
                int t = 0, sw = 0;
                const uint8_t* hb = &hi[(size_t)(c * nkb + sb) * 16];
                const uint8_t* lb = &lo[(size_t)(c * nkb + sb) * 16];
                for (int j = 0; j < 16; ++j) {
                    const int v_l = (p[j] & 0xF) - 8;      // value j
                    const int v_h = (p[j] >> 4) - 8;       // value j+16
                    const int h_l = (int8_t)((hb[j] & 0xF) << 4) >> 4;   // hi nibble j (signed int4)
                    const int h_h = (int8_t)((hb[j] & 0xF0)) >> 4;       // hi nibble j+16
                    const int l_l = ((lb[j] & 0xF) ^ 8) - 8;             // lo nibble j -> a&0xF
                    const int l_h = (((lb[j] >> 4) & 0xF) ^ 8) - 8;      // lo nibble j+16
                    const int a_l = 16 * h_l + l_l;        // int8 code, value j
                    const int a_h = 16 * h_h + l_h;        // int8 code, value j+16
                    t += v_l * a_l + v_h * a_h;
                    sw += v_l + v_h;
                }
                const float d_w = half2float_h(dplane[(size_t)row * nkb + sb]);
                const float d_a = half2float_h(ad[(size_t)c * nkb + sb]);
                const float sw8 = 8.0f * (float)sw;
                acc += d_w * (d_a * ((float)t + sw8));     // kernel op order
            }
            y[(size_t)c * N + row] = acc;
        }
    }
}

int main(int argc, char** argv) {
    Shapes shapes[] = { {5120, 17408}, {17408, 5120} };
    int rc = 0;
    for (auto sh : shapes) {
        const uint32_t K = sh.K, N = sh.N, nkb = K / 32;
        const uint32_t rs = ((K / 2) + 15) / 16 * 16;   // repack_qs_stride alignment
        const size_t qs_bytes = (size_t)N * rs;
        const size_t d_bytes = (size_t)N * nkb * 2;
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> nib(0, 15);
        std::uniform_real_distribution<float> uf(0.0f, 1.0f);
        // Weights: canonical Q4_0 qs bytes (excess-8, byte j = {v_j low, v_j+16 high}) + f16 d plane.
        std::vector<uint8_t> qs(qs_bytes, 0);
        std::vector<uint16_t> dplane(N * nkb);
        for (uint32_t row = 0; row < N; ++row)
            for (uint32_t sb = 0; sb < nkb; ++sb) {
                uint8_t* p = &qs[(size_t)row * rs + (size_t)sb * 16];
                for (int j = 0; j < 16; ++j) {
                    const int v_l = nib(rng) - 8, v_h = nib(rng) - 8;
                    p[j] = (uint8_t)((v_l + 8) | ((v_h + 8) << 4));
                }
                dplane[(size_t)row * nkb + sb] = 0;  // patch below via float->f16
                float dw = 0.01f + 0.03f * uf(rng);
                __half h = __float2half(dw);
                dplane[(size_t)row * nkb + sb] = *reinterpret_cast<uint16_t*>(&h);
            }
        // Activations: T=16 random float rows -> packer -> deinterleave to planes.
        const uint32_t TMAX = 16, NWAVES = 4;
        std::vector<float> x((size_t)TMAX * K);
        for (auto& v : x) v = (uf(rng) * 2.0f - 1.0f) * 3.0f;
        std::vector<uint8_t> packed((size_t)TMAX * nkb * 36);
        for (uint32_t c = 0; c < TMAX; ++c)
            quantize_row_i8split(&x[(size_t)c * K],
                reinterpret_cast<block_i8split*>(&packed[(size_t)c * nkb * 36]), K);
        std::vector<uint8_t> hi((size_t)TMAX * nkb * 16), lo((size_t)TMAX * nkb * 16);
        std::vector<uint16_t> ad((size_t)TMAX * nkb);
        for (uint32_t c = 0; c < TMAX; ++c)
            for (uint32_t sb = 0; sb < nkb; ++sb) {
                const block_i8split* b = reinterpret_cast<const block_i8split*>(&packed[((size_t)c * nkb + sb) * 36]);
                memcpy(&hi[((size_t)c * nkb + sb) * 16], b->hi, 16);
                memcpy(&lo[((size_t)c * nkb + sb) * 16], b->lo, 16);
                ad[(size_t)c * nkb + sb] = b->d;
            }
        // Device buffers
        uint8_t *d_qs, *d_hi, *d_lo; uint16_t *d_dpl, *d_ad; float* d_y;
        hipMalloc(&d_qs, qs_bytes); hipMalloc(&d_dpl, d_bytes);
        hipMalloc(&d_hi, hi.size()); hipMalloc(&d_lo, lo.size()); hipMalloc(&d_ad, ad.size());
        hipMalloc(&d_y, (size_t)TMAX * N * sizeof(float));
        hipMemcpy(d_qs, qs.data(), qs_bytes, hipMemcpyHostToDevice);
        hipMemcpy(d_dpl, dplane.data(), d_bytes, hipMemcpyHostToDevice);
        hipMemcpy(d_hi, hi.data(), hi.size(), hipMemcpyHostToDevice);
        hipMemcpy(d_lo, lo.data(), lo.size(), hipMemcpyHostToDevice);
        hipMemcpy(d_ad, ad.data(), ad.size(), hipMemcpyHostToDevice);
        // Verify + time each width
        printf("== shape K=%u N=%u (nkb=%u rs=%u) ==\n", K, N, nkb, rs);
        float t1_med = 0;
        for (int Ti : {1, 2, 4, 8, 16}) {
            const uint32_t T = (uint32_t)Ti;
            const dim3 grid((N + NWAVES * 64 - 1) / (NWAVES * 64), 1, 1);
            const dim3 blk(NWAVES * 64);
            // correctness (1 call)
            mul_mat_i8narrow_launch(Ti, d_qs, d_hi, d_lo, (const ggml_half*) d_ad, d_y, K, N, nkb, N, 0);
            hipDeviceSynchronize();
            std::vector<float> y((size_t)T * N);
            hipMemcpy(y.data(), d_y, y.size() * sizeof(float), hipMemcpyDeviceToHost);
            std::vector<float> ref;
            cpu_ref(K, N, qs, rs, dplane, hi, lo, ad, nkb, T, ref);
            double maxrel = 0; size_t bad = 0;
            for (size_t i = 0; i < y.size(); ++i) {
                const double denom = std::fabs(ref[i]) > 1e-9 ? std::fabs(ref[i]) : 1e-9;
                const double rel = std::fabs((double)y[i] - ref[i]) / denom;
                if (rel > maxrel) maxrel = rel;
                if (rel > 1e-3) ++bad;
            }
            if (bad) { printf("  T=%2u VERIFY FAIL: %zu/%zu bad (maxrel %.3g)\n", T, bad, y.size(), maxrel); rc = 1; }
            // timing
            hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
            std::vector<float> ms;
            for (int r = 0; r < 35; ++r) {
                hipEventRecord(e0, 0);
                mul_mat_i8narrow_launch(Ti, d_qs, d_hi, d_lo, (const ggml_half*) d_ad, d_y, K, N, nkb, N, 0);
                hipEventRecord(e1, 0); hipEventSynchronize(e1);
                float msr; hipEventElapsedTime(&msr, e0, e1);
                if (r >= 5) ms.push_back(msr);
            }
            std::sort(ms.begin(), ms.end());
            const float med = ms[ms.size() / 2];
            if (T == 1) t1_med = med;
            printf("  T=%2u median %.3f ms  (%.1f us/1k-tokens)  verify-maxrel %.3g %s\n",
                   T, med, med * 1000.0, maxrel, bad ? "FAIL" : "ok");
            if (T == 8) {
                const double ratio = (double)med / (double)t1_med;
                printf("  BAR T8/T1 = %.2fx (%s; pass <= 1.50x)\n", ratio, ratio <= 1.5 ? "PASS" : "FAIL");
                if (ratio > 1.5) rc = 1;
            }
            hipEventDestroy(e0); hipEventDestroy(e1);
        }
        hipFree(d_qs); hipFree(d_dpl); hipFree(d_hi); hipFree(d_lo); hipFree(d_ad); hipFree(d_y);
    }
    printf("MB-RC=%d\n", rc);
    return rc;
}
