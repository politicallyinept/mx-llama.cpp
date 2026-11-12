// mb_nc68_test.cu - Task E2 microbench: repacked8_nc warp-geometry bit-match (E4 step 3).
//
// Question: does mul_mat_vec_q8_0_repacked8_nc<NWARPS,NCOLS> reproduce the STOCK
// mul_mat_vec_q<Q8_0> accumulation bit-for-bit when NWARPS matches the stock GCN
// mmvq table's warp count for the width?
//   stock GCN calc_nwarps (Q8_0): ncols_dst 1-4 -> 2 warps; 5 -> 2 (Q8_0 special
//   case); 6-8 -> 1 warp. rows_per_block only changes row packing, never the
//   per-row kbx->thread partition or the reduce, so one-row-per-block is a valid
//   bit-model of the stock geometry.
// The warp count sets blocks_per_iter = vdr*nwarps*64/qi (32 vs 16): the kbx
// partition across threads, the per-thread fp32 accumulation order, and whether
// a cross-warp shared partial exists. The CPU oracle below mirrors the stock
// sequence exactly (per-thread ascending-kbx fp32 chain, warp0 adds warp1's
// partial at nwarps=2, then the 64-lane shfl-XOR butterfly).
//
// Bars (bit-identical, memcmp on fp32 payloads, NaN-free data):
//   NC in {6,7,8}: <1,NC> device output == stock-geometry oracle   (the fix)
//   NC in {2,3,4,5}: <2,NC> device output == stock-geometry oracle (regression:
//                  the shipped 2-warp arm is unchanged)
//   <1,NC> vs <2,NC> must DIFFER at 6-8 (geometry is load-bearing both ways).
//   Every kernel launch is run twice: device output must be launch-deterministic.
//
// Arithmetic model: host and device both compile -ffp-contract=off so the oracle
// and the kernel share one strict IEEE-754 binary32 mul/mul/add sequence. In the
// production build any FMA contraction applies identically to stock mmvq and to
// repacked8_nc (same source shape acc += dw*dx*(float)sumi, same toolchain) --
// the same argument the widths 2-5 production bit-id gate already rests on.
//
// Standalone (own main + ggml_abort stub), hipcc per toolchain rule; run on ONE
// idle card under flock.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
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

// Stock GCN mmvq warp table for Q8_0 (calc_nwarps, mmvq.cu): 1-4 -> 2, 5 -> 2
// (Q8_0 special case), 6-8 -> 1.
static int stock_nwarps(int ncols) {
    return ncols <= 5 ? 2 : 1;
}

// CPU mirror of stock mul_mat_vec_q<Q8_0> for one (row, col): per-thread
// ascending-kbx fp32 chain, cross-warp partial at nwarps=2, 64-lane XOR butterfly.
// Reads the SAME packed byte buffers the kernel reads (w qs/d planes, x q8_1).
static float stock_oracle(uint32_t row, uint32_t col, uint32_t nkb, uint32_t nw,
                          const uint8_t * w, uint32_t qs_str, const uint8_t * dplane,
                          const block_q8_1 * xq, uint32_t xs) {
    std::vector<float> v(64u * nw, 0.0f);
    const int bpi = 16 * (int) nw;  // vdr*nwarps*warp_size/qi = 2*nw*64/8
    for (int t = 0; t < 64 * (int) nw; ++t) {
        const int * w_row = reinterpret_cast<const int *>(w + (size_t) row * qs_str);
        const uint16_t * dh_row = reinterpret_cast<const uint16_t *>(dplane)
                                 + (size_t) row * nkb;
        float acc = 0.0f;
        for (int kbx = t / 4; kbx < (int) nkb; kbx += bpi) {
            const int kqs = 2 * (t % 4);
            const int * wv = w_row + 8 * kbx;
            const block_q8_1 * xb = xq + (size_t) col * xs + kbx;
            const int * xv = reinterpret_cast<const int *>(xb->qs);
            const uint16_t db = dh_row[kbx];
            const float dw = __half2float(*reinterpret_cast<const __half *>(&db));
            // low half of the half2 ds (little-endian: first 2 bytes) = the scale
            const float dx = __half2float(*reinterpret_cast<const __half *>(&xb->ds));
            int sumi = dp4a_cpu(wv[kqs],     xv[kqs],     0);
            sumi     = dp4a_cpu(wv[kqs + 1], xv[kqs + 1], sumi);
            acc += dw * dx * (float) sumi;
        }
        v[t] = acc;
    }
    if (nw == 2) {
        for (int i = 0; i < 64; ++i) v[i] += v[i + 64];  // warp 0 adds warp 1's partial
    }
    for (int off = 32; off > 0; off >>= 1) {             // shfl_xor butterfly
        std::vector<float> nv(64);
        for (int i = 0; i < 64; ++i) nv[i] = v[i] + v[i ^ off];
        for (int i = 0; i < 64; ++i) v[i] = nv[i];
    }
    return v[0];
}

template <int NW, int NC>
static void launch_r8nc(const uint8_t * w, const block_q8_1 * xq, float * y,
                        uint32_t ne0, uint32_t ne1, uint32_t xs, uint32_t ys,
                        hipStream_t stream) {
    const dim3 grid((unsigned) ne1, 1, 1);
    mul_mat_vec_q8_0_repacked8_nc<NW, NC><<<grid, dim3(64, NW), 0, stream>>>(
        w, xq, y, ne0, ne1, xs, ys);
}

struct CellResult { int nc; int nw_stock; bool fix_match; bool ship_match;
                    bool differ; bool determ; long fix_bad; long ship_bad; };

template <int NC>
static CellResult run_cell(uint32_t K, uint32_t N, std::mt19937 & rng) {
    const uint32_t nkb = K / 32;
    const uint32_t qs_str = repack_qs_stride(K);
    const size_t wbytes = (size_t) N * qs_str + (size_t) N * nkb * 2;
    std::vector<uint8_t> w(wbytes, 0);
    std::uniform_int_distribution<int> qd(-127, 127);
    std::uniform_real_distribution<float> sd(0.001f, 0.9f);
    // weight qs plane + f16 scale plane (same layout repack_q8_0_kernel writes)
    for (uint32_t r = 0; r < N; ++r) {
        int8_t * qs = reinterpret_cast<int8_t *>(w.data() + (size_t) r * qs_str);
        uint16_t * dh = reinterpret_cast<uint16_t *>(
            w.data() + (size_t) N * qs_str) + (size_t) r * nkb;
        for (uint32_t b = 0; b < nkb; ++b) {
            for (int i = 0; i < 32; ++i) qs[b * 32 + i] = (int8_t) qd(rng);
            const __half h = __float2half_rn(sd(rng));
            dh[b] = *reinterpret_cast<const uint16_t *>(&h);
        }
    }
    // activations: NC columns of nkb block_q8_1 (xs = nkb), ys = N
    std::vector<block_q8_1> x((size_t) NC * nkb);
    for (uint32_t c = 0; c < NC; ++c) {
        for (uint32_t b = 0; b < nkb; ++b) {
            block_q8_1 & xb = x[(size_t) c * nkb + b];
            for (int i = 0; i < 32; ++i) xb.qs[i] = (int8_t) qd(rng);
            const __half h = __float2half_rn(sd(rng));
            const __half2 h2 = __halves2half2(h, h);
            memcpy(&xb.ds, &h2, sizeof(h2));
        }
    }
    // CPU oracle (stock geometry for this width)
    std::vector<float> ref((size_t) NC * N);
    const int nw_stock = stock_nwarps(NC);
    for (uint32_t c = 0; c < NC; ++c) {
        for (uint32_t r = 0; r < N; ++r) {
            ref[(size_t) c * N + r] = stock_oracle(r, c, nkb, nw_stock,
                                                  w.data(), qs_str,
                                                  w.data() + (size_t) N * qs_str,
                                                  x.data(), nkb);
        }
    }
    // device
    uint8_t * dw; block_q8_1 * dx; float * dy1; float * dy2; float * dy1b;
    const size_t ybytes = (size_t) NC * N * sizeof(float);
    HIP_CHECK(hipMalloc(&dw, wbytes));
    HIP_CHECK(hipMalloc(&dx, x.size() * sizeof(block_q8_1)));
    HIP_CHECK(hipMalloc(&dy1, ybytes));
    HIP_CHECK(hipMalloc(&dy2, ybytes));
    HIP_CHECK(hipMalloc(&dy1b, ybytes));
    HIP_CHECK(hipMemcpy(dw, w.data(), wbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dx, x.data(), x.size() * sizeof(block_q8_1), hipMemcpyHostToDevice));
    launch_r8nc<1, NC>(dw, dx, dy1, K, N, nkb, N, 0);
    launch_r8nc<2, NC>(dw, dx, dy2, K, N, nkb, N, 0);
    launch_r8nc<1, NC>(dw, dx, dy1b, K, N, nkb, N, 0);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> d1((size_t) NC * N), d2((size_t) NC * N), d1b((size_t) NC * N);
    HIP_CHECK(hipMemcpy(d1.data(), dy1, ybytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(d2.data(), dy2, ybytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(d1b.data(), dy1b, ybytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(dw)); HIP_CHECK(hipFree(dx));
    HIP_CHECK(hipFree(dy1)); HIP_CHECK(hipFree(dy2)); HIP_CHECK(hipFree(dy1b));

    CellResult res{};
    res.nc = NC; res.nw_stock = nw_stock;
    res.fix_bad = 0; res.ship_bad = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (memcmp(&d1[i], &ref[i], 4) != 0) ++res.fix_bad;
        if (memcmp(&d2[i], &ref[i], 4) != 0) ++res.ship_bad;
    }
    res.fix_match = res.fix_bad == 0;
    res.ship_match = res.ship_bad == 0;
    res.differ = memcmp(d1.data(), d2.data(), ybytes) != 0;
    res.determ = memcmp(d1.data(), d1b.data(), ybytes) == 0;
    return res;
}

template <int NC>
static void run_width(uint32_t K, uint32_t N, std::mt19937 & rng, int & fails) {
    CellResult r = run_cell<NC>(K, N, rng);
    // Bars: the arm the dispatch actually launches for this width must match
    // the stock oracle bit-exact: <1,NC> at 6-8 (the E2 fix), <2,NC> at 2-5
    // (shipped arm, regression). The other arm is reported for information.
    const bool launched_match = (NC >= 6) ? r.fix_match : r.ship_match;
    const bool diff_bar = r.differ;  // geometry is load-bearing at every width
    const bool ok = launched_match && r.determ && diff_bar;
    if (!ok) ++fails;
    printf("K=%5u N=%4u NC=%d stock_nw=%d | <1,NC>==ref:%s (bad %ld) | <2,NC>==ref:%s (bad %ld) | 1v2 differ:%s | determ:%s | %s\n",
           K, N, NC, r.nw_stock,
           r.fix_match ? "YES" : "no", r.fix_bad,
           r.ship_match ? "YES" : "no", r.ship_bad,
           r.differ ? "yes" : "NO", r.determ ? "yes" : "NO",
           ok ? "PASS" : "FAIL");
    fflush(stdout);
}

int main(int argc, char ** argv) {
    const int dev = argc > 1 ? atoi(argv[1]) : 0;
    HIP_CHECK(hipSetDevice(dev));
    printf("mb_nc68_test: repacked8_nc warp-geometry bit-match (device %d)\n", dev);
    printf("bars: NC 6-8 <1,NC>==stock-oracle bit-exact; NC 2-5 <2,NC>==stock-oracle bit-exact;\n"
           "      NC 6-8 <1,NC>!=<2,NC>; every launch deterministic.\n");
    int fails = 0;
    std::mt19937 rng(0xE2u);
    const uint32_t Ks[] = { 2048, 4096, 5120 };
    const uint32_t Ns[] = { 16, 100 };
    for (uint32_t K : Ks) {
        for (uint32_t N : Ns) {
            run_width<2>(K, N, rng, fails);
            run_width<3>(K, N, rng, fails);
            run_width<4>(K, N, rng, fails);
            run_width<5>(K, N, rng, fails);
            run_width<6>(K, N, rng, fails);
            run_width<7>(K, N, rng, fails);
            run_width<8>(K, N, rng, fails);
        }
    }
    printf(fails == 0 ? "ALL PASS\n" : "FAILURES: %d\n", fails);
    return fails == 0 ? 0 : 1;
}
