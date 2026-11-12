// mb_e4_bytegen_test.cu - E4 step 1a: byte-equivalence of the GPU runtime repack
// kernel (repack_q8_0_kernel, q8_repack/buffer.cu:137) vs the host permutation
// (repack_q8_0_host, q8_repack/repack-common.cu:350) across the E4 shape grid.
//
// Both paths consume IDENTICAL canonical Q8_0 bytes (deterministic PRNG) and
// write the two-plane record (qs rows with the de-alias stride bump + f16
// scale plane). Comparisons per shape:
//   A. host record vs GPU kernel into a ZERO-filled dst  -> full-record memcmp
//   B. host record vs GPU kernel into a 0xA5-filled dst  -> every differing
//      byte classified qs-live / qs-gap / d-plane (the gap is the 16 B bump
//      repack_qs_stride adds when ne0%128==0; the host memsets it, the kernel
//      never writes it)
//   C. GPU kernel twice into zero-filled dst -> launch determinism
// Launch parameters mirror the runtime async path (buffer.cu:487-495):
//   qs_str = repack_qs_row_stride(Q8_0, ne0), qs_len = ne1*qs_str,
//   src_str = ggml_nbytes/ne2 (contiguous tensor: ne1*n_blocks*34),
//   dst_str = repack_gcn_nbytes(Q8_0, ne0, ne1), n_out = ne2*ne1*n_blocks,
//   block = 256, grid = min(ceil(n_out/block), 65535).
//
// Build (TheRock, source /root/w4a4-env.sh):
//   hipcc -O3 -std=c++17 --offload-arch=gfx906 -DGGML_USE_HIP \
//     -Iggml/include -Iggml/src -Iggml/src/ggml-cuda \
//     ggml/src/ggml-cuda/q8_repack/mb_e4_bytegen_test.cu -o mb_e4_bytegen
// Run on ONE idle card under flock (GPU2 by convention):
//   flock -w 600 /root/gpu.lock env HIP_VISIBLE_DEVICES=2 ./mb_e4_bytegen
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cstdarg>
#include <hip/hip_runtime.h>

#include "ggml.h"
#include "ggml-cuda/common.cuh"
#include "q8_repack/repack-common.cuh"

// Link stub: the standalone binary does not link libggml; GGML_ASSERT needs this.
void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "ABORT %s:%d: ", file, line);
    vfprintf(stderr, fmt, ap); va_end(ap);
    fprintf(stderr, "\n"); exit(3);
}

#define CK(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes for the 34B-stride source");

// ---------------------------------------------------------------------------
// Verbatim copy of repack_q8_0_host (q8_repack/repack-common.cu:350).
// ---------------------------------------------------------------------------
static void repack_q8_0_host_local(const block_q8_0 * blocks, uint8_t * dst,
                                   const int64_t ne0, const int64_t ne1) {
    GGML_ASSERT(ne0 % 32 == 0);
    const int64_t n_blocks = ne0 / 32;
    const int64_t qs_str   = repack_qs_stride(ne0);
    const size_t  qs_len   = (size_t) ne1 * qs_str;

    memset(dst, 0, qs_len + (size_t) ne1 * n_blocks * 2);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q8_0 * b = &blocks[row * n_blocks + blk];
            memcpy(dst + (size_t) row * qs_str + (size_t) blk * 32, b->qs, 32);
            memcpy(dst + qs_len + (size_t)(row * n_blocks + blk) * 2, &b->d, 2);
        }
    }
}

// ---------------------------------------------------------------------------
// Verbatim copy of repack_q8_0_kernel (q8_repack/buffer.cu:137).
// ---------------------------------------------------------------------------
static __global__ void repack_q8_0_kernel_copy(
        const uint8_t * __restrict__ src, uint8_t * __restrict__ dst,
        const int64_t ne1, const int64_t n_blocks, const int64_t qs_str,
        const int64_t qs_len, const int64_t src_stride, const int64_t dst_stride,
        const int64_t total) {
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += (int64_t) gridDim.x * blockDim.x) {
        const int64_t blk = i % n_blocks;
        const int64_t row = (i / n_blocks) % ne1;
        const int64_t e   = i / (n_blocks * ne1);

        uint8_t * d_qs = dst + e * dst_stride + row * qs_str + blk * 32;
        uint8_t * d_d  = dst + e * dst_stride + qs_len + (row * n_blocks + blk) * 2;

        const uint8_t * sb = src + e * src_stride + (row * n_blocks + blk) * 34;
#pragma unroll
        for (int k = 0; k < 32; k++) {
            d_qs[k] = sb[2 + k];
        }
        d_d[0] = sb[0];
        d_d[1] = sb[1];
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static uint64_t xs64 = 0x9E3779B97F4A7C15ULL;
static inline uint64_t rnd64(void) {
    xs64 ^= xs64 >> 12; xs64 ^= xs64 << 25; xs64 ^= xs64 >> 27;
    return xs64 * 0x2545F4914F6CDD1DULL;
}

static uint64_t fnv(const uint8_t * p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ULL; }
    return h;
}

// Map a byte offset inside one expert's record to a record field.
static const char * classify(int64_t r, int64_t n_blocks, int64_t qs_str, size_t qs_len) {
    if ((size_t) r < qs_len) {
        const int64_t col = r % qs_str;
        return col < n_blocks * 32 ? "qs-live" : "qs-gap";
    }
    return "d-plane";
}

struct Shape { int64_t ne0, ne1, ne2; const char * tag; };

int main(void) {
    const Shape shapes[] = {
        // E1 grid dense: K x M (ne0 x ne1), all K%128==0 -> 16 B de-alias gap
        { 1024, 2048, 1, "e1-dense" }, { 1024, 4096, 1, "e1-dense" }, { 1024, 5120, 1, "e1-dense" },
        { 2048, 2048, 1, "e1-dense" }, { 2048, 4096, 1, "e1-dense" }, { 2048, 5120, 1, "e1-dense" },
        { 4096, 2048, 1, "e1-dense" }, { 4096, 4096, 1, "e1-dense" }, { 4096, 5120, 1, "e1-dense" },
        { 5120, 2048, 1, "e1-dense" }, { 5120, 4096, 1, "e1-dense" }, { 5120, 5120, 1, "e1-dense" },
        { 6144, 2048, 1, "e1-dense" }, { 6144, 4096, 1, "e1-dense" }, { 6144, 5120, 1, "e1-dense" },
        // MoE expert tensors (35B-A3B class: 64 experts); exercises ne2>1 and
        // the 65535-block grid cap (4096x2048x64 -> n_out = 16777216 > 65535*256)
        { 2048, 2048, 64, "moe64" }, { 4096, 2048, 64, "moe64" }, { 2048, 4096, 64, "moe64" },
        { 4096, 2048,  8, "moe8"  },
        // K%32==0 but K%128!=0: NO de-alias gap (qs_str == n_blocks*32)
        { 1056, 2048, 1, "nogap" }, { 4128, 4096, 1, "nogap" }, { 5152, 2048, 1, "nogap" },
        // small / multi-expert edges
        {   32,   32, 1, "tiny" }, {   64,   96, 2, "tiny" }, {  128,  128, 1, "tiny" },
        { 6144, 5120, 2, "big3d" },
    };
    const int n_shapes = (int) (sizeof(shapes) / sizeof(shapes[0]));

    printf("ne0,ne1,ne2,tag,n_out,rec_bytes,A_host_vs_gpu_zeroed,A_first_diff,"
           "B_gap_only,B_gap_bytes,B_first_diff,C_determinism,fnv_host,fnv_gpu\n");
    fflush(stdout);

    int pass = 0, fail = 0;
    for (int s = 0; s < n_shapes; s++) {
        const int64_t ne0 = shapes[s].ne0, ne1 = shapes[s].ne1, ne2 = shapes[s].ne2;
        const char * tag = shapes[s].tag;
        const int64_t n_blocks = ne0 / 32;
        // runtime async-path parameters (buffer.cu:487-495)
        const int64_t qs_str   = repack_qs_row_stride(GGML_TYPE_Q8_0, ne0);
        const size_t  qs_len   = (size_t) ne1 * qs_str;
        const size_t  src_str  = (size_t) ne1 * n_blocks * 34;      // ggml_nbytes/ne2, contiguous
        const size_t  dst_str  = repack_gcn_nbytes(GGML_TYPE_Q8_0, ne0, ne1);
        const int64_t n_out    = ne2 * ne1 * n_blocks;
        const int     block    = 256;
        const int     grid     = (int) std::min<int64_t>((n_out + block - 1) / block, 65535);
        const size_t  src_bytes = src_str * ne2;
        const size_t  rec_bytes = dst_str * ne2;

        // deterministic canonical Q8_0 source: random f16 scale bits + random qs bytes
        std::vector<uint8_t> h_src(src_bytes);
        for (size_t i = 0; i + 8 <= src_bytes; i += 8) {
            const uint64_t v = rnd64();
            memcpy(&h_src[i], &v, 8);
        }
        for (size_t i = src_bytes & ~7ULL; i < src_bytes; i++) h_src[i] = (uint8_t) rnd64();

        // host path: repack per expert into a 0xA5-poisoned buffer (host memsets
        // its own region, so poison only detects under-writes)
        std::vector<uint8_t> h_host(rec_bytes, 0xA5);
        for (int64_t e = 0; e < ne2; e++) {
            repack_q8_0_host_local((const block_q8_0 *) (h_src.data() + e * src_str),
                                   h_host.data() + e * dst_str, ne0, ne1);
        }

        uint8_t *d_src, *d_dst0, *d_dstA, *d_dst2;
        CK(cudaMalloc(&d_src, src_bytes));
        CK(cudaMalloc(&d_dst0, rec_bytes));
        CK(cudaMalloc(&d_dstA, rec_bytes));
        CK(cudaMalloc(&d_dst2, rec_bytes));
        CK(cudaMemcpy(d_src, h_src.data(), src_bytes, cudaMemcpyHostToDevice));
        CK(cudaMemset(d_dst0, 0x00, rec_bytes));
        CK(cudaMemset(d_dstA, 0xA5, rec_bytes));
        CK(cudaMemset(d_dst2, 0x00, rec_bytes));

        repack_q8_0_kernel_copy<<<grid, block>>>(d_src, d_dst0, ne1, n_blocks, qs_str,
                                                 qs_len, src_str, dst_str, n_out);
        repack_q8_0_kernel_copy<<<grid, block>>>(d_src, d_dstA, ne1, n_blocks, qs_str,
                                                 qs_len, src_str, dst_str, n_out);
        repack_q8_0_kernel_copy<<<grid, block>>>(d_src, d_dst2, ne1, n_blocks, qs_str,
                                                 qs_len, src_str, dst_str, n_out);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());

        std::vector<uint8_t> h_g0(rec_bytes), h_gA(rec_bytes), h_g2(rec_bytes);
        CK(cudaMemcpy(h_g0.data(), d_dst0, rec_bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_gA.data(), d_dstA, rec_bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_g2.data(), d_dst2, rec_bytes, cudaMemcpyDeviceToHost));
        cudaFree(d_src); cudaFree(d_dst0); cudaFree(d_dstA); cudaFree(d_dst2);

        // A: host vs GPU-into-zeroed (the equivalence question)
        size_t a_diff = 0; long long a_first = -1;
        for (size_t i = 0; i < rec_bytes; i++) {
            if (h_host[i] != h_g0[i]) { a_diff++; if (a_first < 0) a_first = (long long) i; }
        }
        // B: host vs GPU-into-0xA5 (characterizes unwritten gap bytes)
        size_t b_diff = 0, b_gap = 0, b_other = 0; long long b_first = -1, b_first_other = -1;
        for (size_t i = 0; i < rec_bytes; i++) {
            if (h_host[i] != h_gA[i]) {
                b_diff++;
                if (b_first < 0) b_first = (long long) i;
                const char * c = classify((int64_t)(i % dst_str), n_blocks, qs_str, qs_len);
                if (strcmp(c, "qs-gap") == 0) b_gap++;
                else { b_other++; if (b_first_other < 0) b_first_other = (long long) i; }
            }
        }
        // C: GPU determinism (two launches, same zeroed start state)
        const bool c_eq = memcmp(h_g0.data(), h_g2.data(), rec_bytes) == 0;

        const bool ok = (a_diff == 0) && (b_other == 0) && c_eq;
        if (ok) pass++; else fail++;

        char b_first_buf[32];
        if (b_first_other >= 0) snprintf(b_first_buf, sizeof(b_first_buf), "live@%lld", b_first_other);
        else snprintf(b_first_buf, sizeof(b_first_buf), "none");

        printf("%lld,%lld,%lld,%s,%lld,%lld,%s,%lld,%s,%lld,%s,%s,%016llx,%016llx\n",
               (long long) ne0, (long long) ne1, (long long) ne2, tag,
               (long long) n_out, (long long) rec_bytes,
               a_diff == 0 ? "MATCH" : "MISMATCH", a_first,
               b_other == 0 ? "gap-only" : "LIVE-DIFF", (long long) b_gap,
               b_first_buf,
               c_eq ? "yes" : "NO",
               (unsigned long long) fnv(h_host.data(), rec_bytes),
               (unsigned long long) fnv(h_g0.data(), rec_bytes));
        fflush(stdout);

        h_src.clear(); h_host.clear(); h_g0.clear(); h_gA.clear(); h_g2.clear();
    }
    printf("SUMMARY: %d/%d shapes byte-identical (host vs GPU-zeroed, gap-only deltas otherwise)\n",
           pass, n_shapes);
    return fail == 0 ? 0 : 1;
}
