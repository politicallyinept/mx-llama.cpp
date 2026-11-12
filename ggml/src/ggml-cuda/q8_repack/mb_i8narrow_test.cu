// mb_i8narrow.cc - Task C GPU microbench: mul_mat_i8narrow_q40 timings + device-vs-host bit-match.
// Standalone (own main); run on ONE idle card under flock. Shapes: K=5120/N=17408 and
// K=17408/N=5120, widths T=1/2/4/8/16 (NWAVES=4, INTERLEAVED=false).
// KSPLIT sweep (task C2): shape 1 stays KSPLIT=1 (regression arm); shape 2 runs
// KSPLIT=1/2/4/8. Bars: T8 <= 1.5x T1 per KSPLIT; outputs vs CPU reference
// (exact integer identity, fp32 epilogue order replicated; tolerance 1e-2 rel as
// in seal_i8split). KSPLIT>1 refs come from cpu_ref_grouped, a CPU mirror of the
// kernel's fixed-order wave/z grouping (KSPLIT changes the fp32 summation ORDER,
// so the KSPLIT=1 cpu_ref is NOT the right oracle for KSPLIT>1 cells).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
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

struct Shapes { uint32_t K, N; int ksplit_max; };

// CPU reference: exact same fp32 op sequence as the kernel epilogue, sb-order accumulation.
// KSPLIT=1 oracle ONLY (the KSPLIT=1 kernel accumulates sb sequentially).
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

// CPU reference for KSPLIT>1: mirrors the kernel's fixed-order fp32 grouping
// EXACTLY (the KSPLIT-matched oracle - KSPLIT changes the fp32 summation order,
// so KSPLIT=1's sequential sum is the wrong oracle for these cells).
// Per (c,row): compute term[sb] with the same fp32 ops as the kernel inner body,
// then group: z segments [z*seg, min((z+1)*seg, nkb)) with seg = ceil(nkb/ks);
// inside a z segment NWAVES wave sub-segments (wseg = ceil(len/NWAVES), clamped,
// empty waves contribute 0.0f) summed in fixed order w=0..NWAVES-1; z partials
// then summed in fixed order z=0..ks-1 starting from 0.0f (reducer order).
// All ks_list variants share one term[] pass (grouping is cheap).
static void cpu_ref_grouped(uint32_t K, uint32_t N, const std::vector<uint8_t>& qs, uint32_t rs,
                            const std::vector<uint16_t>& dplane,
                            const std::vector<uint8_t>& hi, const std::vector<uint8_t>& lo,
                            const std::vector<uint16_t>& ad, uint32_t nkb, uint32_t T,
                            const std::vector<int>& ks_list, uint32_t nwaves,
                            std::vector<std::vector<float>>& outs) {
    outs.assign(ks_list.size(), std::vector<float>((size_t)T * N, 0.0f));
    std::vector<float> term(nkb);
    for (uint32_t c = 0; c < T; ++c) {
        for (uint32_t row = 0; row < N; ++row) {
            const uint8_t* qrow = &qs[(size_t)row * rs];
            for (uint32_t sb = 0; sb < nkb; ++sb) {
                const uint8_t* p = qrow + (size_t)sb * 16;
                int t = 0, sw = 0;
                const uint8_t* hb = &hi[(size_t)(c * nkb + sb) * 16];
                const uint8_t* lb = &lo[(size_t)(c * nkb + sb) * 16];
                for (int j = 0; j < 16; ++j) {
                    const int v_l = (p[j] & 0xF) - 8;
                    const int v_h = (p[j] >> 4) - 8;
                    const int h_l = (int8_t)((hb[j] & 0xF) << 4) >> 4;
                    const int h_h = (int8_t)((hb[j] & 0xF0)) >> 4;
                    const int l_l = ((lb[j] & 0xF) ^ 8) - 8;
                    const int l_h = (((lb[j] >> 4) & 0xF) ^ 8) - 8;
                    const int a_l = 16 * h_l + l_l;
                    const int a_h = 16 * h_h + l_h;
                    t += v_l * a_l + v_h * a_h;
                    sw += v_l + v_h;
                }
                const float d_w = half2float_h(dplane[(size_t)row * nkb + sb]);
                const float d_a = half2float_h(ad[(size_t)c * nkb + sb]);
                term[sb] = d_w * (d_a * ((float)t + 8.0f * (float)sw));
            }
            for (size_t k = 0; k < ks_list.size(); ++k) {
                const uint32_t ks = (uint32_t) ks_list[k];
                const uint32_t seg = (nkb + ks - 1) / ks;
                float acc = 0.0f;
                for (uint32_t z = 0; z < ks; ++z) {
                    const uint32_t k_lo = z * seg, k_hi = std::min(k_lo + seg, nkb);
                    float zsum = 0.0f;
                    if (k_hi > k_lo) {
                        const uint32_t wseg = (k_hi - k_lo + nwaves - 1) / nwaves;
                        for (uint32_t w = 0; w < nwaves; ++w) {
                            const uint32_t wlo = std::min(k_lo + w * wseg, k_hi);
                            const uint32_t whi = std::min(wlo + wseg, k_hi);
                            float wsum = 0.0f;
                            for (uint32_t sb = wlo; sb < whi; ++sb) wsum += term[sb];
                            zsum += wsum;   // fixed wave order
                        }
                    }
                    acc += zsum;            // fixed z order (reducer semantics)
                }
                outs[k][(size_t)c * N + row] = acc;
            }
        }
    }
}

static double max_rel_err(const std::vector<float>& y, const std::vector<float>& ref) {
    double maxrel = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        const double denom = std::fabs(ref[i]) > 1e-9 ? std::fabs(ref[i]) : 1e-9;
        const double rel = std::fabs((double)y[i] - ref[i]) / denom;
        if (rel > maxrel) maxrel = rel;
    }
    return maxrel;
}

int main(int argc, char** argv) {
    (void) argc; (void) argv;
    // ksplit_max: shape 1 = regression arm (KSPLIT=1 only, must match prior numbers);
    // shape 2 = the KSPLIT sweep target.
    Shapes shapes[] = { {5120, 17408, 1}, {17408, 5120, 8} };
    int rc = 0;
    int shape_idx = -1;
    hipStream_t stream; hipStreamCreate(&stream);
    for (auto sh : shapes) {
        ++shape_idx;
        const uint32_t K = sh.K, N = sh.N, nkb = K / 32;
        uint32_t rs = K / 2;                             // repack_qs_stride: nibble bytes
        if (rs % 128 == 0) rs += 16;                     // de-alias bump (+16 when 128-aligned)
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
                memcpy(&ad[(size_t)c * nkb + sb], &b->d, sizeof(uint16_t)); /* bit copy: half -> uint16_t truncation would zero every sub-0.5 scale */
            }
        // Device buffer: ONE contiguous [qs | d] plane (the kernel reads the
        // weight d scale at wbase + dplane_off = ne1*rs; separate allocations
        // would IMA). Sumw plane is not read by this kernel (inline sum).
        std::vector<uint8_t> wbuf(qs_bytes + d_bytes, 0);
        memcpy(wbuf.data(), qs.data(), qs_bytes);
        memcpy(wbuf.data() + qs_bytes, dplane.data(), d_bytes);
        uint8_t *d_wb, *d_hi, *d_lo; uint16_t *d_ad; float* d_y; float* d_ws = nullptr;
        hipMalloc(&d_wb, wbuf.size());
        hipMalloc(&d_hi, hi.size()); hipMalloc(&d_lo, lo.size()); hipMalloc(&d_ad, ad.size() * sizeof(uint16_t));
        hipMalloc(&d_y, (size_t)TMAX * N * sizeof(float));
        if (sh.ksplit_max > 1)
            hipMalloc(&d_ws, (size_t)sh.ksplit_max * N * TMAX * sizeof(float)); // [ksplit][N][T] floats, caller-owned
        hipMemcpy(d_wb, wbuf.data(), wbuf.size(), hipMemcpyHostToDevice);
        hipMemcpy(d_hi, hi.data(), hi.size(), hipMemcpyHostToDevice);
        hipMemcpy(d_lo, lo.data(), lo.size(), hipMemcpyHostToDevice);
        hipMemcpy(d_ad, ad.data(), ad.size() * sizeof(uint16_t), hipMemcpyHostToDevice);
        {   // graft (wt-t16fix): upload-in-bytes readback proof (uint16 vec, not bytes)
            std::vector<uint16_t> chk(ad.size());
            hipMemcpy(chk.data(), d_ad, ad.size() * sizeof(uint16_t), hipMemcpyDeviceToHost);
            size_t mm = 0;
            for (size_t i = 0; i < ad.size(); ++i) if (chk[i] != ad[i]) ++mm;
            printf("  [diag] d_ad readback mismatches=%zu/%zu\n", mm, ad.size());
            if (mm) { printf("  [diag] D_AD UPLOAD BROKEN\n"); rc = 1; }
        }
        printf("== shape K=%u N=%u (nkb=%u rs=%u) ksplit_max=%d ==\n", K, N, nkb, rs, sh.ksplit_max);
        // KSPLIT list for this shape: {1} or {1,2,4,8}
        std::vector<int> ks_list;
        for (int ks = 1; ks <= sh.ksplit_max; ks = (ks == 1 ? 2 : ks * 2)) ks_list.push_back(ks);
        const size_t y_sent_size = (size_t)TMAX * N;
        std::vector<float> t1_med(ks_list.size(), 0.0f);   // T=1 median per KSPLIT (bar denominators)
        double best_ratio = 1e9; int best_ks = 1;
        for (int Ti : {1, 2, 4, 8, 16}) {
            const uint32_t T = (uint32_t) Ti;
            // KSPLIT=1 oracle (unchanged code path from the pre-KSPLIT harness)
            std::vector<float> ref1;
            cpu_ref(K, N, qs, rs, dplane, hi, lo, ad, nkb, T, ref1);
            // KSPLIT>1 oracles: one grouped pass produces refs for every ks>1
            std::vector<int> ks_gt1;
            for (size_t ki = 0; ki < ks_list.size(); ++ki) if (ks_list[ki] > 1) ks_gt1.push_back(ks_list[ki]);
            std::vector<std::vector<float>> refg;
            if (!ks_gt1.empty())
                cpu_ref_grouped(K, N, qs, rs, dplane, hi, lo, ad, nkb, T, ks_gt1, NWAVES, refg);
            std::vector<float> y_ks1;  // device KSPLIT=1 output, for the delta-class column
            for (size_t ki = 0; ki < ks_list.size(); ++ki) {
                const int ks = ks_list[ki];
                // correctness (1 call)
                {   // graft (wt-t16fix): sentinel pre-fill BEFORE the launch (distinguishes unwritten from zero-written)
                    std::vector<float> sent(y_sent_size, 12345.0f);
                    hipMemcpy(d_y, sent.data(), sent.size() * sizeof(float), hipMemcpyHostToDevice);
                }
                mul_mat_i8narrow_launch(Ti, d_wb, d_hi, d_lo, (const ggml_half*) d_ad, d_y, K, N, nkb, N, stream, ks, d_ws);
                hipError_t le = hipGetLastError();
                hipDeviceSynchronize();
                if (le != hipSuccess) { printf("  [diag] T=%d ks=%d LAUNCH ERR: %s\n", Ti, ks, hipGetErrorString(le)); rc = 1; }
                std::vector<float> y((size_t)T * N);
                hipMemcpy(y.data(), d_y, y.size() * sizeof(float), hipMemcpyDeviceToHost);
                const std::vector<float>& ref = (ks == 1) ? ref1 : refg[ki - 1];
                const double maxrel = max_rel_err(y, ref);
                // bad = rel > 1e-2 AND absdiff > 1e-3. The abs escape is the
                // documented cancellation class: a near-zero output (|ref| ~ 1e-4
                // while summed terms are ~ 1e-2) makes the RELATIVE bar
                // meaningless; fp32 contraction-class abs error is <= ~1e-4 here
                // (measured max |device-ks - device-ks1| = 1.14e-4). A real bug
                // (e.g. the known T=16 plane corruption, absdiff ~ 1e6) still fails.
                size_t bad = 0, cancel_exempt = 0;
                size_t worst = 0; double worst_rel = 0;
                for (size_t i = 0; i < y.size(); ++i) {
                    const double denom = std::fabs(ref[i]) > 1e-9 ? std::fabs(ref[i]) : 1e-9;
                    const double absd = std::fabs((double)y[i] - ref[i]);
                    const double rel = absd / denom;
                    if (std::isnan(y[i])) { ++bad; continue; }  // graft: NaN counts as bad
                    if (rel > 1e-2) { if (absd > 1e-3) ++bad; else ++cancel_exempt; }
                    if (rel > worst_rel) { worst_rel = rel; worst = i; }
                }
                // delta class: KSPLIT>1 vs the KSPLIT=1 DEVICE output (pure fp32
                // reordering; documents the order-differ-from-KSPLIT=1 delta)
                double vs1 = -1.0, vs1_abs = -1.0;
                if (ks == 1) y_ks1 = y;
                else if (y_ks1.size() == y.size()) {
                    vs1 = max_rel_err(y, y_ks1);
                    for (size_t i = 0; i < y.size(); ++i) {
                        const double a = std::fabs((double)y[i] - y_ks1[i]);
                        if (a > vs1_abs) vs1_abs = a;
                    }
                }
                if (bad) {
                    printf("  T=%2d KSPLIT=%d VERIFY FAIL: %zu/%zu bad (maxrel %.3g) worst i=%zu (c=%zu row=%zu) ref=%.9g y=%.9g absdiff=%.3g\n",
                           Ti, ks, bad, y.size(), maxrel, worst, worst % (size_t)T, worst / T,
                           (double)ref[worst], (double)y[worst], std::fabs((double)y[worst] - ref[worst]));
                    rc = 1;
                }
                // determinism: second launch must be BIT-identical (fixed-order
                // reduce, no atomics) - the whole point of the KSPLIT design.
                mul_mat_i8narrow_launch(Ti, d_wb, d_hi, d_lo, (const ggml_half*) d_ad, d_y, K, N, nkb, N, stream, ks, d_ws);
                hipDeviceSynchronize();
                std::vector<float> y2((size_t)T * N);
                hipMemcpy(y2.data(), d_y, y2.size() * sizeof(float), hipMemcpyDeviceToHost);
                const bool bitmatch = memcmp(y2.data(), y.data(), y.size() * sizeof(float)) == 0;
                if (!bitmatch) { printf("  T=%2d KSPLIT=%d DETERMINISM FAIL: second launch not bit-identical\n", Ti, ks); rc = 1; }
                // timing
                std::vector<float> ms;
                for (int r = 0; r < 35; ++r) {
                    hipDeviceSynchronize();
                    auto t0 = std::chrono::steady_clock::now();
                    mul_mat_i8narrow_launch(Ti, d_wb, d_hi, d_lo, (const ggml_half*) d_ad, d_y, K, N, nkb, N, stream, ks, d_ws);
                    hipDeviceSynchronize();
                    auto t1 = std::chrono::steady_clock::now();
                    if (r >= 5) ms.push_back(std::chrono::duration<float, std::milli>(t1 - t0).count());
                }
                std::sort(ms.begin(), ms.end());
                const float med = ms[ms.size() / 2];
                if (T == 1) t1_med[ki] = med;
                printf("  T=%2d KSPLIT=%d median %.3f ms  verify-maxrel %.3g %s (cancel-exempt %zu) det-bitmatch %s  vs-ks1-delta %.3g (abs %.3g)\n",
                       Ti, ks, med, maxrel, bad ? "FAIL" : "ok", cancel_exempt, bitmatch ? "ok" : "FAIL", vs1, vs1_abs);
                if (T == 8) {
                    const double ratio = (double)med / (double)t1_med[ki];
                    printf("  BAR KSPLIT=%d T8/T1 = %.2fx (%s; pass <= 1.50x)\n",
                           ks, ratio, ratio <= 1.5 ? "PASS" : "FAIL");
                    if (ks > 1 && ratio < best_ratio) { best_ratio = ratio; best_ks = ks; }
                }
            }
        }
        if (sh.ksplit_max > 1) {
            printf("  KSPLIT-BAR: best KSPLIT=%d T8/T1 = %.2fx (%s)\n",
                   best_ks, best_ratio, best_ratio <= 1.5 ? "PASS" : "FAIL");
            if (best_ratio > 1.5) rc = 1;
        }
        hipFree(d_wb); hipFree(d_hi); hipFree(d_lo); hipFree(d_ad); hipFree(d_y); hipFree(d_ws);
    }
    hipStreamDestroy(stream);
    printf("MB-RC=%d\n", rc);
    return rc;
}
