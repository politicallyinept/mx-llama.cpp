// E4 step-1b re-scope (task E5): runtime-baseline witness + repack-vs-gcn_kq bit-diff harness.
//
// Runs ONE dense MUL_MAT (Q8_0 src0 K x M, f32 src1 K x N) through the REAL ggml-cuda
// dispatch and dumps the fp32 dst bytes. The serving arm is selected by env (set by the
// runner, never here):
//   stock arm             (no repack env)                    buft=cuda   -> gcn_kq if admitted, else MMQ
//   MMQ arm               GGML_CUDA_GCN_KQ=0                 buft=cuda   -> MMQ on every shape
//   repack arm yield ON   REPACK_Q8_0=1 STOCK_ACC=0          buft=repack -> yield to gcn_kq where admitted
//   repack arm yield OFF  + REPACK_Q8_0_YIELD=0              buft=repack -> repacked GEMM on every admitted shape
// Pairwise bit-diffs of the dumps answer: repack-vs-gcn_kq (yield OFF vs stock on yield
// shapes), repack-vs-MMQ (runtime re-check of E1), yield-arm-vs-stock (canonical-copy
// byte-exactness), and gcn_kq-vs-MMQ (the stock regime split itself).
//
// Measurement-only harness: no kernel code touched. Named *_test.cu so the ggml-hip
// CMake glob (q8_repack/*.cu minus *_test.cu) never compiles it into the library.
//
// usage: e5_gcnkq_bitdiff_test <K> <M> <N> <cuda|repack> <out.bin>
// GPU2 only (HIP_VISIBLE_DEVICES=2), under flock -w 600 /root/gpu.lock.
//
// dst is read back with a raw hipMemcpy: the repack buffer deliberately sets
// iface.get_tensor=nullptr (weights are write-once in repacked layout), but dst is a
// plain f32 tensor the GEMM wrote, so a direct device read is the correct path.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <unistd.h>
#include <vector>

// Resolved from the SAME libggml-hip.so image the backend registry dlopened (the
// repack buft is defined in ggml/src/ggml-cuda/q8_repack/buffer.cu). Deliberately
// NOT linked directly: libggml-hip.so has no SONAME, and a second mapping of it
// would split the repack gate state from the one the registry registered. The
// registry loads it from the executable directory, so we resolve that image with
// RTLD_NOLOAD against the same path (no new mapping is created).
typedef ggml_backend_buffer_type_t (*repack_buft_fn)(int);
static repack_buft_fn resolve_repack_buft(void) {
    char exe[4096];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (n <= 0) { fprintf(stderr, "readlink /proc/self/exe failed\n"); return nullptr; }
    exe[n] = '\0';
    char * slash = strrchr(exe, '/');
    if (!slash) { fprintf(stderr, "bad exe path\n"); return nullptr; }
    strcpy(slash + 1, "libggml-hip.so");
    void * h = dlopen(exe, RTLD_NOW | RTLD_NOLOAD);
    if (!h) { fprintf(stderr, "libggml-hip.so (%s) not loaded by registry: %s\n", exe, dlerror()); return nullptr; }
    return (repack_buft_fn) dlsym(h, "_Z36ggml_backend_cuda_repack_buffer_typei");
}

static uint64_t g_s;
static inline uint64_t rnd(void) {
    g_s = g_s * 6364136223846793005ULL + 1442695040888963407ULL;
    return g_s;
}

int main(int argc, char ** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s K M N cuda|repack out.bin\n", argv[0]);
        return 1;
    }
    const int64_t K = atoll(argv[1]), M = atoll(argv[2]), N = atoll(argv[3]);
    const char * buft_kind = argv[4];
    const char * out_path  = argv[5];
    if (K % 32 != 0 || M < 1 || N < 1) { fprintf(stderr, "bad shape\n"); return 1; }

    ggml_init_params params = { /*mem_size*/ 64u*1024*1024, /*mem_buffer*/ nullptr, /*no_alloc*/ true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 2; }
    ggml_backend_load_all();  // fork convention: apps load the dynamic backends explicitly

    ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, M);
    ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_set_name(src0, "e5_w");
    ggml_set_name(src1, "e5_x");
    ggml_tensor * dst = ggml_mul_mat(ctx, src0, src1);
    ggml_set_name(dst, "e5_dst");
    ggml_init_params gparams = { /*mem_size*/ 1u*1024*1024, /*mem_buffer*/ nullptr, /*no_alloc*/ false };
    ggml_context * ctx_g = ggml_init(gparams);
    ggml_cgraph * gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) { fprintf(stderr, "no GPU device (HIP_VISIBLE_DEVICES?)\n"); return 2; }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) { fprintf(stderr, "backend init failed\n"); return 2; }

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    if (!strcmp(buft_kind, "repack")) {
        repack_buft_fn repack_buft = resolve_repack_buft();
        if (!repack_buft) { fprintf(stderr, "repack buft fn null\n"); return 2; }
        buft = repack_buft(0);  // HIP_VISIBLE_DEVICES pins one GPU as device 0
        if (!buft) { fprintf(stderr, "repack buft null\n"); return 2; }
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buf) { fprintf(stderr, "tensor alloc failed on buft %s\n", buft_kind); return 3; }

    // deterministic pseudo-random Q8_0 weights: half scale + 32 int8 per 32-value block
    const size_t nb0 = ggml_nbytes(src0);
    std::vector<uint8_t> wb(nb0);
    g_s = 0x9E3779B97F4A7C15ULL ^ ((uint64_t) K << 40) ^ ((uint64_t) M << 16) ^ 0xE5ULL;
    for (size_t i = 0; i + 34 <= nb0; i += 34) {
        const float  sc = 0.001f + (float)(rnd() >> 40) / (float)(1ULL << 24) * 0.01f;
        const __half h  = __float2half_rn(sc);
        memcpy(&wb[i], &h, 2);
        for (int j = 0; j < 32; ++j) {
            wb[i + 2 + j] = (uint8_t)(uint8_t)(int8_t)(uint8_t)(rnd() >> 56);
        }
    }
    std::vector<float> xb((size_t) K * N);
    for (auto & v : xb) {
        v = ((float)(rnd() >> 33) / (float)(1ULL << 31)) - 1.0f;
    }

    ggml_backend_tensor_set(src0, wb.data(), 0, nb0);
    ggml_backend_tensor_set(src1, xb.data(), 0, xb.size() * sizeof(float));

    const enum ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) { fprintf(stderr, "graph compute failed: %d\n", (int) st); return 4; }

    const size_t n_out = (size_t) M * N;
    std::vector<float> db(n_out);
    const hipError_t ce = hipMemcpy(db.data(), dst->data, n_out * sizeof(float), hipMemcpyDeviceToHost);
    if (ce != hipSuccess) { fprintf(stderr, "dst readback failed: %s\n", hipGetErrorString(ce)); return 4; }

    FILE * f = fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out_path); return 4; }
    fwrite(db.data(), sizeof(float), n_out, f);
    fclose(f);
    printf("OK K=%lld M=%lld N=%lld buft=%s out=%s\n",
           (long long) K, (long long) M, (long long) N, buft_kind, out_path);
    return 0;
}
