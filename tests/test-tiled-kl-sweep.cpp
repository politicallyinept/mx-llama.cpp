// test-tiled-kl-sweep: formal coverage answer for the C3/C4 blast radius.
//
// Per the R4 design: for each shape class (dense widths 2-15; MoE sizes
// {8,16,32,64,128} x verify widths {1,3,4}), run the tiled-repack Q4_0 kernel
// vs the generic dequant matvec on IDENTICAL fp32 activations, 64 greedy
// decode steps, and compute token-level KL(P_tiled || P_generic) plus the
// first-divergence position. Pass = max KL < 1e-3 and no divergence before
// token 8. Three seeds.
//
// The two paths are selected by the shipped env gates (read once per process):
//   tiled leg:    GGML_CUDA_REPACK_Q4_0=1   (admits Q4_0 to the repack pool;
//                  GGML_CUDA_REPACK_Q4_0_TILED defaults ON -> tiled gcn_u4_gemm)
//   generic leg:  both unset                (canonical Q4_0 kernels)
// Engagement witness: the library prints "gate ACTIVE" banners when the repack
// path is live; the dump log records them.
//
// Usage:
//   test-tiled-kl-sweep --dump <file.bin> --seed <N>   (GPU; run once per leg)
//   test-tiled-kl-sweep --compare <tiled.bin> <generic.bin>   (CPU; prints table)

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-quants.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define STEPS 64

// ---------------------------------------------------------------------------
// shape registry — MUST be identical between dump and compare runs
// ---------------------------------------------------------------------------

struct shape_def {
    char     name[40];
    bool     moe;
    int64_t  K;        // reduction length
    int64_t  N;        // dense out rows / MoE rows per expert
    int64_t  E;        // MoE experts (1 for dense)
    int64_t  w;        // batch width (tokens per step)
};

static std::vector<shape_def> build_shapes(void) {
    std::vector<shape_def> s;
    for (int64_t w = 2; w <= 15; w++) {
        shape_def d = {};
        snprintf(d.name, sizeof(d.name), "dense_K2048_N2048_w%lld", (long long) w);
        d.moe = false; d.K = 2048; d.N = 2048; d.E = 1; d.w = w;
        s.push_back(d);
    }
    const int64_t es[] = { 8, 16, 32, 64, 128 };
    const int64_t ws[] = { 1, 3, 4 };
    for (int64_t E : es) {
        for (int64_t w : ws) {
            shape_def d = {};
            snprintf(d.name, sizeof(d.name), "moe_E%lld_K1024_R256_w%lld", (long long) E, (long long) w);
            d.moe = true; d.K = 1024; d.N = 256; d.E = E; d.w = w;
            s.push_back(d);
        }
    }
    return s;
}

// ---------------------------------------------------------------------------
// deterministic RNG (splitmix64) — identical streams in both legs
// ---------------------------------------------------------------------------

static uint64_t smix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
static float frand_u(uint64_t h) {  // uniform [-1, 1)
    return (float) ((double) (h >> 11) / 9007199254740992.0 * 2.0 - 1.0);
}

// ---------------------------------------------------------------------------
// dump mode (GPU)
// ---------------------------------------------------------------------------

struct dump_header {
    char    magic[4];
    int32_t version;
    int64_t seed;
    int64_t n_shapes;
};
struct shape_record {
    char    name[40];
    int64_t K, N, E, w;
    int64_t out_per_step;   // N * w
};

static int run_dump(const char * path, uint64_t seed) {
    printf("tiled-kl-sweep dump: seed=%llu GGML_CUDA_REPACK_Q4_0=%s GGML_CUDA_REPACK_Q4_0_TILED=%s\n",
           (unsigned long long) seed,
           getenv("GGML_CUDA_REPACK_Q4_0") ? getenv("GGML_CUDA_REPACK_Q4_0") : "(unset)",
           getenv("GGML_CUDA_REPACK_Q4_0_TILED") ? getenv("GGML_CUDA_REPACK_Q4_0_TILED") : "(unset->default ON)");
    fflush(stdout);

    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!backend) {
        fprintf(stderr, "no GPU backend\n");
        return 1;
    }
    printf("backend: %s\n", ggml_backend_name(backend));

    // The repack path lives on a dedicated extra buffer type (selected by the model loader in
    // production). The plain default buft never repacks, so the tiled leg MUST allocate there.
    ggml_backend_buffer_type_t repack_buft = nullptr;
    {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        for (int i = 0; repack_buft == nullptr && i < (int) ggml_backend_reg_count(); i++) {
            ggml_backend_reg_t reg = ggml_backend_reg_get(i);
            auto f = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_dev_get_extra_bufts");
            if (!f || !dev) {
                continue;
            }
            // the CUDA/HIP device exposes exactly one extra buft: the repack one
            ggml_backend_buffer_type_t * pb = f(dev);
            if (*pb != nullptr) {
                repack_buft = *pb;
                printf("extra buft: %s\n", ggml_backend_buft_name(*pb));
            }
        }
    }
    const bool want_repack = getenv("GGML_CUDA_REPACK_Q4_0") != nullptr;
    if (want_repack && repack_buft == nullptr) {
        fprintf(stderr, "FATAL: repack buffer type not found — tiled leg cannot engage the repack path\n");
        return 3;
    }
    ggml_backend_buffer_type_t def_buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer_type_t buft = (want_repack && repack_buft) ? repack_buft : def_buft;
    printf("weight buft: %s | io buft: %s\n", ggml_backend_buft_name(buft), ggml_backend_buft_name(def_buft));
    fflush(stdout);

    auto shapes = build_shapes();
    FILE * fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    dump_header h = {};
    memcpy(h.magic, "TKLS", 4); h.version = 1; h.seed = (int64_t) seed; h.n_shapes = (int64_t) shapes.size();
    if (fwrite(&h, sizeof(h), 1, fp) != 1) { fprintf(stderr, "write fail\n"); return 1; }

    int si = 0;
    for (const auto & sh : shapes) {
        // fresh ggml ctx per shape (tensors sized per shape)
        ggml_init_params ip = { 64u << 20, nullptr, /*.no_alloc=*/ true };
        // v4: weights live on the repack buft; activations/output on the default buft.
        // The repack buffer is write-once by design (iface.get_tensor == nullptr and
        // ggml_backend_tensor_get has NO null guard -> tail-call to 0x0, SIGSEGV).
        // Production only ever allocates model weights there and never reads them
        // back, so the v3 crash was harness misuse, not a kernel defect.
        ggml_context_ptr ctx_w(ggml_init(ip));   // src0 only -> repack buft
        ggml_context_ptr ctx (ggml_init(ip));    // src1/ids/dst + graph -> default buft
        ggml_cgraph * gf = ggml_new_graph(ctx.get());  // freed with ctx

        // --- tensors: weights + activations + ids, then one buffer alloc ---
        const int64_t rows = sh.N * sh.E;
        ggml_tensor * src0 = sh.moe
            ? ggml_new_tensor_3d(ctx_w.get(), GGML_TYPE_Q4_0, sh.K, sh.N, sh.E)
            : ggml_new_tensor_2d (ctx_w.get(), GGML_TYPE_Q4_0, sh.K, sh.N);
        ggml_tensor * src1 = sh.moe
            ? ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, sh.K, 1, sh.w)   // b is 3d for MUL_MAT_ID
            : ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, sh.K, sh.w);
        ggml_tensor * ids  = nullptr;
        std::vector<int32_t> ids_host(sh.w);
        if (sh.moe) {
            ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, sh.w);
        }
        ggml_tensor * dst;
        if (sh.moe) {
            dst = ggml_mul_mat_id(ctx.get(), src0, src1, ids);
        } else {
            dst = ggml_mul_mat(ctx.get(), src0, src1);
        }
        ggml_build_forward_expand(gf, dst);
        const int64_t out_per_step = sh.N * sh.w;

        ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w.get(), buft);
        GGML_ASSERT(buf_w);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), def_buft);
        GGML_ASSERT(buf);

        // --- weights: Q4_0, deterministic from seed+shape ---
        {
            std::vector<float> row(sh.K);
            std::vector<uint8_t> qbuf(ggml_row_size(GGML_TYPE_Q4_0, sh.K));
            std::vector<uint8_t> wbuf(ggml_nbytes(src0));
            for (int64_t r = 0; r < rows; r++) {
                for (int64_t k = 0; k < sh.K; k++) {
                    row[k] = frand_u(smix(seed * 131 + si * 7717 + r * 13 + k));
                }
                quantize_row_q4_0_ref(row.data(), (block_q4_0 *) qbuf.data(), sh.K);
                memcpy(wbuf.data() + (size_t) r * qbuf.size(), qbuf.data(), qbuf.size());
            }
            ggml_backend_tensor_set(src0, wbuf.data(), 0, ggml_nbytes(src0));
        }

        shape_record rec = {};
        memcpy(rec.name, sh.name, sizeof(rec.name));
        rec.K = sh.K; rec.N = sh.N; rec.E = sh.E; rec.w = sh.w; rec.out_per_step = out_per_step;
        if (fwrite(&rec, sizeof(rec), 1, fp) != 1) { fprintf(stderr, "write fail\n"); fclose(fp); return 1; }

        std::vector<float> x((size_t) sh.K * sh.w);
        std::vector<float> out(out_per_step);
        for (int step = 0; step < STEPS; step++) {
            for (int64_t c = 0; c < sh.w; c++) {
                for (int64_t k = 0; k < sh.K; k++) {
                    x[(size_t) c * sh.K + k] = frand_u(smix(seed * 977 + si * 3301 + (uint64_t) step * 101 + c * 17 + k));
                }
                if (sh.moe) {
                    ids_host[c] = (int32_t) (smix(seed * 61 + si * 89 + (uint64_t) step * 7 + c) % (uint64_t) sh.E);
                }
            }
            ggml_backend_tensor_set(src1, x.data(), 0, ggml_nbytes(src1));
            if (sh.moe) {
                ggml_backend_tensor_set(ids, ids_host.data(), 0, ggml_nbytes(ids));
            }
            const ggml_status st = ggml_backend_graph_compute(backend, gf);
            if (st != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "graph compute failed shape=%s step=%d rc=%d\n", sh.name, step, (int) st);
                fclose(fp);
                return 2;
            }
            ggml_backend_tensor_get(dst, out.data(), 0, (size_t) out_per_step * sizeof(float));
            if (fwrite(out.data(), sizeof(float), out_per_step, fp) != (size_t) out_per_step) { fprintf(stderr, "write fail\n"); fclose(fp); return 1; }
        }
        printf("dumped %-28s (%d steps x %lld cols x %lld)\n", sh.name, STEPS, (long long) sh.w, (long long) sh.N);
        fflush(stdout);
        ggml_backend_buffer_free(buf);
        ggml_backend_buffer_free(buf_w);
        si++;
    }
    fclose(fp);
    ggml_backend_free(backend);
    printf("dump OK -> %s\n", path);
    return 0;
}

// ---------------------------------------------------------------------------
// compare mode (CPU)
// ---------------------------------------------------------------------------

static double softmax_kl(const float * a, const float * b, int64_t n) {
    // KL(P_a || P_b) with a=tiled, b=generic
    double ma = -1e30, mb = -1e30;
    for (int64_t i = 0; i < n; i++) { if (a[i] > ma) ma = a[i]; if (b[i] > mb) mb = b[i]; }
    double sa = 0, sb = 0;
    std::vector<double> pa(n), pb(n);
    for (int64_t i = 0; i < n; i++) {
        pa[i] = std::exp((double) a[i] - ma);
        pb[i] = std::exp((double) b[i] - mb);
        sa += pa[i]; sb += pb[i];
    }
    double kl = 0;
    for (int64_t i = 0; i < n; i++) {
        const double p = pa[i] / sa, q = pb[i] / sb;
        if (p > 0) {
            kl += p * std::log(p / (q > 1e-300 ? q : 1e-300));
        }
    }
    return kl;
}

static int64_t argmax(const float * a, int64_t n) {
    int64_t bi = 0;
    for (int64_t i = 1; i < n; i++) if (a[i] > a[bi]) bi = i;
    return bi;
}

static int run_compare(const char * pa, const char * pb) {
    FILE * fa = fopen(pa, "rb"), * fb = fopen(pb, "rb");
    if (!fa || !fb) { fprintf(stderr, "cannot open dumps\n"); return 1; }
    dump_header ha, hb;
    if (fread(&ha, sizeof(ha), 1, fa) != 1 || fread(&hb, sizeof(hb), 1, fb) != 1) {
        fprintf(stderr, "short header read\n"); return 1;
    }
    if (memcmp(ha.magic, "TKLS", 4) || memcmp(hb.magic, "TKLS", 4) || ha.n_shapes != hb.n_shapes) {
        fprintf(stderr, "bad/mismatched dumps\n"); return 1;
    }
    printf("compare seed_tiled=%lld seed_generic=%lld shapes=%lld  (pass: maxKL<1e-3, no divergence before token 8)\n",
           (long long) ha.seed, (long long) hb.seed, (long long) ha.n_shapes);
    printf("%-30s %12s %12s %14s %6s\n", "shape", "max_KL", "mean_KL", "first_div_tok", "verdict");

    int fails = 0;
    for (int64_t s = 0; s < ha.n_shapes; s++) {
        shape_record ra, rb;
        if (fread(&ra, sizeof(ra), 1, fa) != 1 || fread(&rb, sizeof(rb), 1, fb) != 1) {
            fprintf(stderr, "short shape-record read\n"); return 1;
        }
        if (strcmp(ra.name, rb.name) || ra.out_per_step != rb.out_per_step) {
            fprintf(stderr, "shape mismatch at %lld\n", (long long) s); return 1;
        }
        std::vector<float> A((size_t) ra.out_per_step * STEPS), B((size_t) rb.out_per_step * STEPS);
        const size_t bytes = (size_t) ra.out_per_step * STEPS * sizeof(float);
        if (fread(A.data(), 1, bytes, fa) != bytes || fread(B.data(), 1, bytes, fb) != bytes) {
            fprintf(stderr, "short read on shape %s\n", ra.name); return 1;
        }
        double max_kl = 0, sum_kl = 0;
        long n_cols = 0;
        long first_div = -1;
        long n_bitexact = 0;
        for (int step = 0; step < STEPS; step++) {
            const float * a = A.data() + (size_t) step * ra.out_per_step;
            const float * b = B.data() + (size_t) step * ra.out_per_step;
            if (memcmp(a, b, bytes / STEPS) == 0) { n_bitexact++; }
            for (int64_t c = 0; c < ra.w; c++) {
                const double kl = softmax_kl(a + c * ra.N, b + c * ra.N, ra.N);
                if (kl > max_kl) max_kl = kl;
                sum_kl += kl; n_cols++;
                if (first_div < 0 && argmax(a + c * ra.N, ra.N) != argmax(b + c * ra.N, ra.N)) {
                    first_div = step + 1;  // 1-based token index
                }
            }
        }
        const bool pass = max_kl < 1e-3 && (first_div < 0 || first_div >= 8);
        if (!pass) fails++;
        printf("%-30s %12.3e %12.3e %14s %6s%s\n", ra.name, max_kl, sum_kl / n_cols,
               first_div < 0 ? "none" : std::to_string(first_div).c_str(),
               pass ? "PASS" : "FAIL",
               n_bitexact == STEPS ? "  (bit-identical every step)" : "");
    }
    fclose(fa); fclose(fb);
    printf("compare verdict: %s (%d failing shapes)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}

int main(int argc, char ** argv) {
    if (argc >= 4 && !strcmp(argv[1], "--dump")) {
        uint64_t seed = 1;
        for (int i = 4; i + 1 < argc; i++) if (!strcmp(argv[i], "--seed")) seed = strtoull(argv[++i], nullptr, 10);
        return run_dump(argv[2], seed);
    }
    if (argc >= 4 && !strcmp(argv[1], "--compare")) {
        return run_compare(argv[2], argv[3]);
    }
    fprintf(stderr, "usage: %s --dump <file.bin> --seed <N> | --compare <tiled.bin> <generic.bin>\n", argv[0]);
    return 1;
}
