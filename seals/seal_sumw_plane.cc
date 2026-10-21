// CPU seal for the repack-time precomputed SUMW plane (GGML_CUDA_W4A4_SUMW).
//
// Proves, on CPU only:
//   (a) the plane written by the production host converter (repack_q40_host,
//       extracted verbatim into seal_gen.h by build_seal.sh) equals the inline
//       device computation (float)((int)nib_sum_u4(pk) - 256) - extracted
//       verbatim from repack-kernels.cuh - for adversarial packs (all-zero,
//       all-0xF, alternating, single-nibble-set), a random fuzz, and real
//       Q4_0 weight tensors read from a GGUF file;
//   (b) the SIGNED 32-bit semantics are preserved (milestone-3b class: the
//       unsigned wrap on nibble sums < 256 must NOT appear);
//   (c) gate-off is byte-identical to the pre-change path: the record size
//       matches the old formula and repack_q40_host (gate off) writes exactly
//       what repack_iq4nl_host writes;
//   (d) the device reader's plane offset (rp_traits<Q4_0>::geom::swplane_off,
//       extracted) equals the host writer's offset (rp_sumw_plane_off).
//
// The gate is a process-static (fork pattern), so the seal runs each half in a
// child process with the env set accordingly and aggregates the results.
//
// Build: build_seal.sh (extracts production code; extraction misses are hard
// failures - that is the falsification signal on a reverted tree).
// Run:   ./seal_sumw_plane [path-to-Q4_0.gguf]
#include "sumw-plane.h"   // rp_sumw_block, rp_sumw_from_block, rp_sumw_plane_off
#include "seal_gen.h"     // production code extracted from the worktree

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <unistd.h>

static int g_fail = 0;
#define CHECK(name, cond)                                                       \
    do {                                                                        \
        const bool ok_ = (cond);                                                \
        printf("INVARIANT %-52s %s\n", name, ok_ ? "PASS" : "FAIL");           \
        if (!ok_) g_fail++;                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// reference: the EXACT inline epilogue expression, over the 4 uint32 words of
// one repacked qs block (device: `sw = (float) ((int) nib_sum_u4(pk) - 256)`).
static inline float inline_sw(const uint8_t * qs16) {
    uint4 pk;
    std::memcpy(&pk.x, qs16 + 0,  4);
    std::memcpy(&pk.y, qs16 + 4,  4);
    std::memcpy(&pk.z, qs16 + 8,  4);
    std::memcpy(&pk.w, qs16 + 12, 4);
    return (float) ((int) nib_sum_u4(pk) - 256);
}

// fill a synthetic block matrix with a per-block byte pattern
static void fill_blocks(std::vector<block_q4_0> & b, int pattern, uint64_t & rng) {
    auto rnd = [&rng]() { rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
                          return (uint8_t) ((rng >> 33) & 0xFF); };
    for (size_t i = 0; i < b.size(); ++i) {
        b[i].d = 0x3C00;   // f16 1.0, irrelevant to SUMW
        for (int j = 0; j < 16; ++j) {
            switch (pattern) {
                case 0: b[i].qs[j] = 0x00; break;                 // all-zero
                case 1: b[i].qs[j] = 0xFF; break;                 // all-0xF
                case 2: b[i].qs[j] = (j & 1) ? 0xF0 : 0x0F; break; // alternating
                case 3: {                                          // single nibble set
                    const int pos = (int) (i % 32);
                    const int byte = pos >> 1, nib = pos & 1;
                    b[i].qs[j] = (j == byte) ? (uint8_t) (nib ? 0x10 : 0x01) : 0x00;
                    break;
                }
                default: b[i].qs[j] = rnd(); break;               // fuzz
            }
        }
    }
}

// repack with the production converter (gate state = process env) and check the
// plane against the inline reference for every block. Returns 1 on full match.
static int check_plane(const std::vector<block_q4_0> & blocks, int64_t ne0, int64_t ne1,
                       bool require_negative) {
    const size_t need = repack_gcn_nbytes(GGML_TYPE_Q4_0, ne0, ne1);
    std::vector<uint8_t> dst(need, 0xAB);
    repack_q40_host(blocks.data(), dst.data(), ne0, ne1);

    const rp_traits<GGML_TYPE_Q4_0>::geom g((uint32_t) ne0, (uint32_t) ne1);
    const int64_t n_sub = ne0 / 32;
    bool saw_negative = false;
    for (int64_t row = 0; row < ne1; ++row) {
        for (int64_t sb = 0; sb < n_sub; ++sb) {
            const int64_t blk = row * n_sub + sb;
            // what the device epilogue would compute inline from the qs plane:
            const uint8_t * qs = dst.data() + (size_t) row * g.rs + (size_t) sb * 16;
            const float want = inline_sw(qs);
            // what the plane reader (load_sw) would return:
            int16_t raw;
            std::memcpy(&raw, dst.data() + g.swplane_off + ((size_t) row * n_sub + sb) * 2, 2);
            const float got = (float) (int) raw;
            if (got != want) return 0;
            if ((int) raw < 0) saw_negative = true;
            // int16 range invariant ([-256, +224] ⊂ int16)
            if ((int) raw < -256 || (int) raw > 224) return 0;
            (void) blk;
        }
    }
    if (require_negative && !saw_negative) return 0;
    return 1;
}

// ---------------------------------------------------------------------------
// minimal GGUF reader: iterate Q4_0 2-D tensors, read up to `cap_blocks` whole
// rows' worth of blocks per tensor, up to `cap_tensors` tensors.
struct gguf_rd { FILE * f; uint64_t data_start; };
static uint32_t rd_u32(gguf_rd & r) { uint32_t v = 0; if (fread(&v, 4, 1, r.f) != 1) v = 0; return v; }
static uint64_t rd_u64(gguf_rd & r) { uint64_t v = 0; if (fread(&v, 8, 1, r.f) != 1) v = 0; return v; }
static std::string rd_str(gguf_rd & r) {
    const uint64_t n = rd_u64(r);
    std::string s(n, '\0');
    if (n && fread(&s[0], 1, n, r.f) != n) s.clear();
    return s;
}

static int seal_gguf(const char * path, int * tensors_seen, int * blocks_seen) {
    FILE * f = fopen(path, "rb");
    if (!f) { printf("  (GGUF not readable: %s)\n", path); return 0; }
    gguf_rd r{ f, 0 };
    char magic[4] = {};
    if (fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "GGUF", 4) != 0) { fclose(f); return 0; }
    const uint32_t version = rd_u32(r);
    if (version < 2) { fclose(f); return 0; }
    const uint64_t n_tensors = rd_u64(r);
    const uint64_t n_kv      = rd_u64(r);
    uint64_t alignment = 32;
    static const int ggml_type_q4_0 = 2;   // GGML_TYPE_Q4_0
    for (uint64_t i = 0; i < n_kv; ++i) {
        const std::string key = rd_str(r);      // key
        const uint32_t t = rd_u32(r);           // type
        switch (t) {
            case 0: fseek(f, 1, SEEK_CUR); break;    // uint8
            case 1: fseek(f, 1, SEEK_CUR); break;    // int8
            case 2: fseek(f, 2, SEEK_CUR); break;    // uint16
            case 3: fseek(f, 2, SEEK_CUR); break;    // int16
            case 4: {                                // uint32
                const uint32_t v = rd_u32(r);
                if (key == "general.alignment" && v != 0) alignment = v;
                break;
            }
            case 5: fseek(f, 4, SEEK_CUR); break;    // int32
            case 6: fseek(f, 4, SEEK_CUR); break;    // float32
            case 7: fseek(f, 1, SEEK_CUR); break;    // bool
            case 8: rd_str(r); break;               // string
            case 9: { // array
                const uint32_t et = rd_u32(r);
                const uint64_t n  = rd_u64(r);
                if (et == 8) { for (uint64_t j = 0; j < n; ++j) rd_str(r); break; }  // string array
                size_t esz = 0;
                switch (et) { case 0: case 1: case 7: esz = 1; break;
                              case 2: case 3: esz = 2; break;
                              case 4: case 5: case 6: esz = 4; break;
                              default: esz = 0; }
                if (esz) fseek(f, (long) (n * esz), SEEK_CUR);
                break;
            }
            case 10: fseek(f, 8, SEEK_CUR); break;   // uint64
            case 11: fseek(f, 8, SEEK_CUR); break;   // int64
            case 12: fseek(f, 8, SEEK_CUR); break;   // float64
            default: fclose(f); return 0;
        }
    }
    // tensor infos
    struct tinfo { std::string name; int64_t ne0, ne1; uint64_t off; };
    std::vector<tinfo> q40;
    for (uint64_t i = 0; i < n_tensors; ++i) {
        tinfo t;
        t.name = rd_str(r);
        const uint32_t nd = rd_u32(r);
        int64_t dims[4] = { 1, 1, 1, 1 };
        for (uint32_t d = 0; d < nd && d < 4; ++d) dims[d] = (int64_t) rd_u64(r);
        const uint32_t dt = rd_u32(r);
        t.off = rd_u64(r);
        t.ne0 = dims[0]; t.ne1 = dims[1];
        if (dt == (uint32_t) ggml_type_q4_0 && nd == 2 && t.ne0 % 32 == 0) q40.push_back(t);
    }
    r.data_start = (r.data_start == 0)
        ? ((ftello(f) + (long) alignment - 1) / (long) alignment * (long) alignment) : r.data_start;
    fseek(f, (long) r.data_start, SEEK_SET);

    const int    cap_tensors = 8;
    const int64_t cap_rows   = 64;   // whole rows per tensor: 64 rows * ne0/32 blocks
    int seen = 0, blocks = 0, ok = 1;
    for (const tinfo & t : q40) {
        if (seen >= cap_tensors) break;
        const int64_t rows = t.ne1 < cap_rows ? t.ne1 : cap_rows;
        const int64_t n_sub = t.ne0 / 32;
        std::vector<block_q4_0> blocks_buf((size_t) rows * n_sub);
        fseek(f, (long) (r.data_start + t.off), SEEK_SET);
        const size_t want = blocks_buf.size() * sizeof(block_q4_0);
        if (fread(blocks_buf.data(), 1, want, f) != want) continue;
        if (!check_plane(blocks_buf, t.ne0, rows, /*require_negative=*/false)) { ok = 0; break; }
        seen++; blocks += (int) blocks_buf.size();
    }
    fclose(f);
    *tensors_seen = seen; *blocks_seen = blocks;
    return ok && seen > 0 && blocks > 0;
}

// ---------------------------------------------------------------------------
static int run_gate_off(void) {
    // G1: gate-off record size == the PRE-CHANGE formula, ne1*(qs_stride(ne0/2)
    // + n_sub*2), for de-aliased and non-de-aliased shapes.
    bool g1 = true;
    const int64_t shapes[][2] = { {512, 8}, {480, 8}, {5120, 4}, {2048, 3} };
    for (auto & s : shapes) {
        const int64_t ne0 = s[0], ne1 = s[1], n_sub = ne0 / 32;
        int64_t qs = ne0 / 2; if (qs % 128 == 0) qs += 16;   // repack_qs_stride pre-change
        const size_t legacy = (size_t) ne1 * ((size_t) qs + (size_t) n_sub * 2);
        if (repack_gcn_nbytes(GGML_TYPE_Q4_0, ne0, ne1) != legacy) g1 = false;
    }
    CHECK("gate-off: Q4_0 record size == pre-change 18B/block formula", g1);

    // G2: gate-off repack_q40_host is byte-identical to repack_iq4nl_host.
    uint64_t rng = 12345;
    const int64_t ne0 = 512, ne1 = 6, n_sub = ne0 / 32;
    std::vector<block_q4_0> blocks((size_t) ne1 * n_sub);
    fill_blocks(blocks, 4, rng);
    const size_t sz = repack_gcn_nbytes(GGML_TYPE_Q4_0, ne0, ne1);
    std::vector<uint8_t> a(sz, 0xAB), b(sz, 0xCD);
    repack_q40_host(blocks.data(), a.data(), ne0, ne1);
    repack_iq4nl_host((const block_iq4_nl *) blocks.data(), b.data(), ne0, ne1);
    CHECK("gate-off: repack_q40_host bytes == repack_iq4nl_host bytes", a == b);

    // G3: geom offset is defined and consistent even gate-off (harmless: unread).
    const rp_traits<GGML_TYPE_Q4_0>::geom g((uint32_t) ne0, (uint32_t) ne1);
    CHECK("geom.swplane_off == dplane_off + ne1*n_sub*2 (gate off)",
          g.swplane_off == g.dplane_off + (size_t) ne1 * n_sub * 2);
    return g_fail;
}

static int run_gate_on(const char * gguf) {
    uint64_t rng = 987654321;
    const int64_t shapes[][2] = { {512, 6}, {480, 5} };   // de-aliased + plain rows
    struct Inv { const char * name; int pattern; bool need_neg; };
    const Inv invs[] = {
        { "plane == inline nib_sum_u4-256: all-zero pack",        0, true  },
        { "plane == inline nib_sum_u4-256: all-0xF pack",         1, false },
        { "plane == inline nib_sum_u4-256: alternating pack",     2, false },
        { "plane == inline nib_sum_u4-256: single-nibble-set",    3, true  },
        { "plane == inline nib_sum_u4-256: random fuzz (~9M blk: 20 iters x 10k rows x 2 shapes)", 4, true },
    };
    for (const Inv & inv : invs) {
        bool ok = true;
        const int64_t iters = (inv.pattern == 4) ? 20 : 1;   // fuzz: 20 x 10k blocks
        for (auto & s : shapes) {
            const int64_t ne0 = s[0], ne1 = s[1], n_sub = ne0 / 32;
            for (int64_t it = 0; it < iters; ++it) {
                std::vector<block_q4_0> blocks((size_t) (inv.pattern == 4 ? 10000 : ne1) * n_sub);
                fill_blocks(blocks, inv.pattern, rng);
                const int64_t rows = (inv.pattern == 4) ? 10000 / 1 : ne1;
                // rebuild with exact row count
                blocks.resize((size_t) rows * n_sub);
                fill_blocks(blocks, inv.pattern, rng);
                if (!check_plane(blocks, ne0, rows, inv.need_neg)) ok = false;
            }
        }
        CHECK(inv.name, ok);
    }

    // milestone-3b sentinel: all-zero pack must give EXACTLY -256 (the unsigned
    // wrap would give 4294967040.0f).
    {
        const int64_t ne0 = 512, ne1 = 1, n_sub = ne0 / 32;
        std::vector<block_q4_0> blocks((size_t) ne1 * n_sub);
        fill_blocks(blocks, 0, rng);
        const size_t sz = repack_gcn_nbytes(GGML_TYPE_Q4_0, ne0, ne1);
        std::vector<uint8_t> dst(sz);
        repack_q40_host(blocks.data(), dst.data(), ne0, ne1);
        const rp_traits<GGML_TYPE_Q4_0>::geom g((uint32_t) ne0, (uint32_t) ne1);
        int16_t v; std::memcpy(&v, dst.data() + g.swplane_off, 2);
        CHECK("signed semantics: all-zero plane == -256 (no unsigned wrap)", (int) v == -256);
    }

    // layout: gate-on record == gate-off legacy + ne1*n_sub*2
    {
        bool ok = true;
        for (auto & s : shapes) {
            const int64_t ne0 = s[0], ne1 = s[1], n_sub = ne0 / 32;
            int64_t qs = ne0 / 2; if (qs % 128 == 0) qs += 16;
            const size_t legacy = (size_t) ne1 * ((size_t) qs + (size_t) n_sub * 2);
            const size_t now    = repack_gcn_nbytes(GGML_TYPE_Q4_0, ne0, ne1);
            if (now != legacy + (size_t) ne1 * n_sub * 2) ok = false;
            // host writer offset == device reader offset
            const rp_traits<GGML_TYPE_Q4_0>::geom g((uint32_t) ne0, (uint32_t) ne1);
            if (g.swplane_off != rp_sumw_plane_off(ne0, ne1, qs)) ok = false;
        }
        CHECK("layout: +2B/block, host writer offset == device geom reader offset", ok);
    }

    // real weights: every sampled block of real Q4_0 tensors matches inline
    {
        int tensors = 0, blocks = 0;
        const int ok = seal_gguf(gguf, &tensors, &blocks);
        printf("  (GGUF %s: %d tensors, %d blocks sampled)\n", gguf, tensors, blocks);
        CHECK("real GGUF Q4_0 tensor: plane == inline for every sampled block", ok);
    }
    return g_fail;
}

int main(int argc, char ** argv) {
    const char * gguf = (argc > 1) ? argv[1]
        : "/mnt/old/home/brad/models/Qwen3.8-27B-Q4_0.gguf";
    if (const char * c = getenv("SUMW_SEAL_CHILD")) {
        return (c[0] == '0') ? run_gate_off() : run_gate_on(gguf);
    }
    // gate is a process-static: run each half in a child with the env set.
    int fails = 0;
    char exe[4096];
    ssize_t n = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    std::string self = (n > 0) ? std::string(exe, (size_t) n) : std::string(argv[0]);
    std::string cmd = "\"" + self + "\"";
    if (argc > 1) cmd += " \"" + std::string(argv[1]) + "\"";
    printf("== gate-off children (GGML_CUDA_W4A4_SUMW unset) ==\n");
    fflush(stdout);
    {
        FILE * p = popen(("env -u GGML_CUDA_W4A4_SUMW SUMW_SEAL_CHILD=0 " + cmd + " 2>&1").c_str(), "r");
        char line[512];
        while (fgets(line, sizeof(line), p)) { fputs(line, stdout); }
        const int rc = pclose(p);
        if (rc != 0) { printf("gate-off child FAILED (rc=%d)\n", rc); fails++; }
    }
    printf("== gate-on children (GGML_CUDA_W4A4_SUMW=1) ==\n");
    fflush(stdout);
    {
        FILE * p = popen(("env GGML_CUDA_W4A4_SUMW=1 SUMW_SEAL_CHILD=1 " + cmd + " 2>&1").c_str(), "r");
        char line[512];
        while (fgets(line, sizeof(line), p)) { fputs(line, stdout); }
        const int rc = pclose(p);
        if (rc != 0) { printf("gate-on child FAILED (rc=%d)\n", rc); fails++; }
    }
    printf("\n%s\n", fails == 0 ? "SEAL: ALL INVARIANTS PASS" : "SEAL: FAILURES PRESENT");
    return fails == 0 ? 0 : 1;
}
