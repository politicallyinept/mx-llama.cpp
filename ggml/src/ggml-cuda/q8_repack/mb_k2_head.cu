// mb_k2_head.cu - taskK2 microbench: the generic mul_mat_vec_q<Q6_K, T> vs the
// LDS-staged drafter-head kernel (mmvq-head-lds.cuh) at the head shape, driven
// through a single-op ggml graph on ONE card (taskQ/taskJ2 discipline).
//
// The two arms are the SAME binary: the gate (GGML_DFLASH_HEAD_KERNEL=1, plus
// optional GGML_DFLASH_HEAD_W=4|8) is read once inside libggml at first use, so
// a cell is two subprocess runs (gate off / on) whose dst FNV-1a hashes must be
// byte-identical. Timing: min/median over iters of the synchronous graph
// compute (single node, ~15-20 us fixed overhead against a multi-ms kernel).
//
// usage: mb_k2_head graph <T> <N> <iters> [K]
//        mb_k2_head bw            (1 GiB read-bandwidth reference)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <random>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#define HIPCHK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1);} } while (0)
#include <hip/hip_runtime.h>

static uint64_t fnv1a(const void * p, size_t n) {
    const uint8_t * b = (const uint8_t *) p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

__global__ void bw_read_kernel(const float4 * __restrict__ p, size_t n4, float * __restrict__ sink) {
    float s = 0.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (size_t) gridDim.x * blockDim.x) {
        const float4 v = p[i];
        s += v.x + v.y + v.z + v.w;
    }
    if (s == 1e30f) { sink[0] = s; }
}

static int run_bw() {
    const size_t bytes = (size_t) 1 << 30;
    float * a = nullptr, * b = nullptr;
    HIPCHK(hipMalloc(&a, bytes));
    HIPCHK(hipMalloc(&b, 4));
    HIPCHK(hipMemset(a, 0x3f, bytes));
    float * sink = nullptr;
    HIPCHK(hipMalloc(&sink, 4));
    std::vector<float> ts;
    for (int it = 0; it < 7; ++it) {
        auto t0 = std::chrono::steady_clock::now();
        bw_read_kernel<<<1024, 256>>>((const float4 *) a, bytes / sizeof(float4), sink);
        HIPCHK(hipDeviceSynchronize());
        auto t1 = std::chrono::steady_clock::now();
        ts.push_back(std::chrono::duration<float, std::micro>(t1 - t0).count());
    }
    std::sort(ts.begin(), ts.end());
    printf("BW bytes=%zu min_us=%.1f GB/s=%.1f\n", bytes, ts[0], bytes / ts[0] / 1e3);
    HIPCHK(hipFree(a)); HIPCHK(hipFree(b)); HIPCHK(hipFree(sink));
    return 0;
}

int main(int argc, char ** argv) {
    if (argc >= 2 && !strcmp(argv[1], "bw")) {
        return run_bw();
    }
    if (argc < 5 || strcmp(argv[1], "graph")) {
        fprintf(stderr, "usage: %s graph <T> <N> <iters> [K] | bw\n", argv[0]);
        return 2;
    }
    const int64_t T  = atoll(argv[2]);
    const int64_t N  = atoll(argv[3]);
    const int     it = atoi(argv[4]);
    const int64_t K  = (argc > 5) ? atoll(argv[5]) : 5120;

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { fprintf(stderr, "no GPU dev\n"); return 1; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    if (!be) { fprintf(stderr, "backend init failed\n"); return 1; }

    ggml_init_params ip{.mem_size = (size_t) (1 << 20), .no_alloc = true};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, K, N);
    ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
    ggml_set_name(src0, "output.weight");
    if (ggml_backend_alloc_ctx_tensors(ctx, be) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "device alloc failed\n"); return 1; }

    // deterministic fill: gaussian rows quantized to Q6_K, activations gaussian
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 0.35f);
    {
        const int64_t chunk = 512;
        std::vector<float> f((size_t) chunk * K);
        std::vector<uint8_t> q(ggml_row_size(GGML_TYPE_Q6_K, K) * chunk);
        for (int64_t r0 = 0; r0 < N; r0 += chunk) {
            const int64_t rows = std::min(chunk, N - r0);
            for (size_t i = 0; i < (size_t) rows * K; ++i) f[i] = nd(rng);
            ggml_quantize_chunk(GGML_TYPE_Q6_K, f.data(), q.data(), 0, rows, K, nullptr);
            ggml_backend_tensor_set(src0, q.data(), 0, ggml_row_size(GGML_TYPE_Q6_K, K) * rows);
        }
        std::vector<float> x((size_t) K * T);
        for (auto & v : x) v = nd(rng);
        ggml_backend_tensor_set(src1, x.data(), 0, sizeof(float) * x.size());
    }

    ggml_cgraph * gb = ggml_new_graph(ctx);
    ggml_tensor * dst = ggml_mul_mat(ctx, src0, src1);
    ggml_build_forward_expand(gb, dst);

    ggml_gallocr_t gal = ggml_gallocr_new(ggml_backend_get_default_allocator(be));
    if (!ggml_gallocr_alloc_graph(gal, gb)) { fprintf(stderr, "alloc graph failed\n"); return 1; }

    std::vector<float> out((size_t) N * T);
    const size_t out_bytes = sizeof(float) * out.size();
    std::vector<float> ts;
    uint64_t h = 0;
    for (int i = 0; i < it; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(be, gb) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
        auto t1 = std::chrono::steady_clock::now();
        ts.push_back(std::chrono::duration<float, std::micro>(t1 - t0).count());
        ggml_backend_tensor_get(dst, out.data(), 0, out_bytes);
        const uint64_t hh = fnv1a(out.data(), out_bytes);
        if (i == 0) h = hh; else if (hh != h) { printf("NONDETERMINISTIC hash %016llx vs %016llx\n", (unsigned long long) h, (unsigned long long) hh); h = 0; }
    }
    std::sort(ts.begin(), ts.end());
    const double wbytes = (double) N * (K / 256.0) * 212.0;   // Q6_K weight bytes
    printf("GRAPH T=%lld N=%lld K=%lld hash=%016llx min_us=%.1f med_us=%.1f mean_us=%.1f wGB/s=%.1f\n",
           (long long) T, (long long) N, (long long) K, (unsigned long long) h,
           ts[0], ts[ts.size() / 2],
           std::chrono::duration<double, std::micro>(std::accumulate(ts.begin(), ts.end(), 0.0)).count() / it,
           wbytes / ts[ts.size() / 2] / 1e3);
    return 0;
}
