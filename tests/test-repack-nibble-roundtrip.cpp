// test-repack-nibble-roundtrip: nibble-layout round-trip property test for the
// gfx906 repacked-weight planes (ggml/src/ggml-cuda/q8_repack).
//
// WHY THIS EXISTS (it replaces the wrong-assumption reference):
// The standalone tiled-GEMM unit tests' CPU references mirrored the kernel's own
// nibble-pairing assumption, so the tier3 pairing bug (activations byte_perm'd
// against a NATURAL-order weight layout) was invisible to them — self-consistent
// bit-exactness. See the "PAIRING BUG FIX (tier3-tiledfix 2026-10-30)" note in
// ggml/src/ggml-cuda/q8_repack/gcn-u4-gemm.cu: the bug was only caught by an
// in-model dual-run vs the generic repack GEMM. This test anchors to the
// CANONICAL block order instead, with three properties per (type, shape, blocks):
//
//   P1  byte round-trip: pack(canonical) -> unpack(planes -> canonical) -> pack
//       again must be byte-identical (the plane layout is a bijection over the
//       payload bytes at the documented offsets, incl. the de-alias row bump).
//   P2  value anchor: decoding the planes with the READER's documented
//       value-index mapping (byte j low nibble = value j, high nibble = value
//       j+16, per-plane scale positions) must reproduce dequantize_row_<type>()
//       of the canonical blocks. A pairing/index bug breaks P2 even when P1
//       holds — this is the property the old self-consistent references lacked.
//   P3  shipped-converter cross-check: when the HIP ggml is linked (repack_host
//       symbol present), the mirrored pack must be byte-identical to the
//       shipped host converter in repack-common.cu.
//
// Pure CPU: no GPU, no model files, no network.
//
// The pack mirrors below are literal transcriptions of the shipped converters
// (repack_q40_kernel / repack_q6k_kernel / ... in buffer.cu and the *_host
// functions in repack-common.cu); P3 exists precisely to keep the mirror honest.

#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-common.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <dlfcn.h>

// Shipped host converter (defined in ggml-cuda/q8_repack/repack-common.cu,
// declared in repack-common.cuh). Resolved at runtime via dlopen/dlsym because
// the test links only ggml-base (link-time weak refs are stripped by
// --as-needed). P3 is skipped gracefully when the HIP backend library or the
// ROCm runtime is unavailable.
typedef void (*repack_host_fn)(ggml_type type, const void * blocks, uint8_t * dst, int64_t ne0, int64_t ne1);

static repack_host_fn resolve_repack_host(void) {
    const char * names[] = { "libggml-hip.so.0", "libggml-cuda.so.0" };
    for (const char * n : names) {
        void * h = dlopen(n, RTLD_NOW | RTLD_NOLOAD);
        if (!h) {
            h = dlopen(n, RTLD_NOW);
        }
        if (!h) {
            continue;
        }
        // Itanium-ABI mangled name of repack_host(ggml_type, const void*, uint8_t*, int64_t, int64_t)
        void * sym = dlsym(h, "_Z11repack_host9ggml_typePKvPhll");
        if (sym) {
            return (repack_host_fn) sym;
        }
    }
    return nullptr;
}

static repack_host_fn g_repack_host = nullptr;

// ---------------------------------------------------------------------------
// layout geometry — mirrors repack_qs_stride()/repack_gcn_nbytes() in
// ggml/src/ggml-cuda/q8_repack/repack-common.cuh
// ---------------------------------------------------------------------------

static int64_t rp_stride(int64_t x) {
    return (x % 128 == 0) ? x + 16 : x;   // de-alias bump
}

static bool rp_is_nibble_type(ggml_type t) {
    switch (t) {
        case GGML_TYPE_MXFP4: case GGML_TYPE_IQ4_NL: case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q4_K:  case GGML_TYPE_Q5_K:  case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q4_0:  case GGML_TYPE_Q4_1:  return true;
        default: return false;
    }
}

struct rp_geom {
    int64_t rs, n_sub, nds;
    size_t  h_off, s_off, d_off, total;
};

static rp_geom mk_geom(ggml_type t, int64_t ne0, int64_t ne1) {
    rp_geom g;
    g.rs    = rp_stride(rp_is_nibble_type(t) ? ne0/2 : ne0);
    g.n_sub = ne0/32;
    g.nds   = (g.n_sub + 7)/8;
    const size_t qs = (size_t) ne1 * g.rs;
    g.h_off = g.s_off = g.d_off = qs;
    g.total = 0;
    switch (t) {
        case GGML_TYPE_Q8_0:
            g.total = qs + (size_t) ne1 * g.n_sub * 2; break;
        case GGML_TYPE_MXFP4:
            g.total = qs + (size_t) ne1 * g.n_sub; break;
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q4_0:
            g.total = qs + (size_t) ne1 * g.n_sub * 2; break;
        case GGML_TYPE_Q4_1:
            g.total = qs + (size_t) ne1 * g.n_sub * 4; break;
        case GGML_TYPE_Q5_1:
            g.d_off = qs + (size_t) ne1 * g.n_sub * 4;
            g.total = g.d_off + (size_t) ne1 * g.n_sub * 4; break;
        case GGML_TYPE_Q4_K:
            g.d_off = qs + (size_t) ne1 * g.nds * 12;
            g.total = g.d_off + (size_t) ne1 * g.nds * 4; break;
        case GGML_TYPE_Q5_K:
            g.s_off = qs + (size_t) ne1 * g.n_sub * 4;
            g.d_off = g.s_off + (size_t) ne1 * g.nds * 12;
            g.total = g.d_off + (size_t) ne1 * g.nds * 4; break;
        case GGML_TYPE_Q6_K:
            g.s_off = qs + (size_t) ne1 * g.n_sub * 8;
            g.d_off = g.s_off + (size_t) ne1 * g.n_sub * 2;
            g.total = g.d_off + (size_t) ne1 * g.nds * 2; break;
        default:
            fprintf(stderr, "mk_geom: unsupported type %d\n", (int) t);
            abort();
    }
    return g;
}

// canonical row byte offsets (from ggml-common.h block structs)
struct canon_off { int64_t rb; int64_t d, m, dmin, scales, qh, qs; };
static canon_off canon_layout(ggml_type t) {
    canon_off c = {};
    switch (t) {
        case GGML_TYPE_Q8_0:   c = { 34,  0, -1, -1, -1, -1,  2 }; break; // d f16, qs 32 int8
        case GGML_TYPE_MXFP4:  c = { 17,  0, -1, -1, -1, -1,  1 }; break; // e byte, qs 16
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q4_0:   c = { 18,  0, -1, -1, -1, -1,  2 }; break; // d f16, qs 16
        case GGML_TYPE_Q4_1:   c = { 20,  0,  2, -1, -1, -1,  4 }; break; // {d,m} f16x2, qs 16
        case GGML_TYPE_Q5_1:   c = { 24,  0,  2, -1, -1,  4,  8 }; break; // {d,m}, qh 4, qs 16
        case GGML_TYPE_Q4_K:   c = {144,  0, -1,  2,  4, -1, 16 }; break; // d,dmin,scales12,qs128
        case GGML_TYPE_Q5_K:   c = {176,  0, -1,  2,  4, 16, 48 }; break; // + qh 32
        case GGML_TYPE_Q6_K:   c = {210,208, -1, -1,192,128,  0 }; break; // fork layout: ql128,qh64,scales16,d (d at END)
        default: fprintf(stderr, "canon_layout: unsupported type %d\n", (int) t); abort();
    }
    return c;
}

// ---------------------------------------------------------------------------
// pack: canonical -> repacked planes (mirror of the shipped converters)
// ---------------------------------------------------------------------------

static void pack_type(ggml_type t, const uint8_t * canon, uint8_t * dst,
                      int64_t ne0, int64_t ne1) {
    const rp_geom g = mk_geom(t, ne0, ne1);
    const canon_off c = canon_layout(t);
    const int64_t rb = c.rb;
    memset(dst, 0, g.total);

    for (int64_t row = 0; row < ne1; row++) {
        // rows are contiguous: canonical row stride = rb * blocks_per_row
        const uint8_t * rsrc = canon + row * rb * ((t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K) ? g.nds : g.n_sub);
        uint8_t * rdst = dst + row * g.rs;

        switch (t) {
            case GGML_TYPE_Q8_0:
                for (int64_t blk = 0; blk < g.n_sub; blk++) {
                    const uint8_t * sb = rsrc + blk * 34;
                    memcpy(rdst + blk * 32, sb + 2, 32);
                    memcpy(dst + g.d_off + (row * g.n_sub + blk) * 2, sb, 2);
                }
                break;
            case GGML_TYPE_MXFP4:
                for (int64_t blk = 0; blk < g.n_sub; blk++) {
                    const uint8_t * sb = rsrc + blk * 17;
                    memcpy(rdst + blk * 16, sb + 1, 16);
                    dst[g.d_off + row * g.n_sub + blk] = sb[0];
                }
                break;
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_IQ4_NL:
                for (int64_t blk = 0; blk < g.n_sub; blk++) {
                    const uint8_t * sb = rsrc + blk * 18;
                    memcpy(rdst + blk * 16, sb + 2, 16);
                    memcpy(dst + g.d_off + (row * g.n_sub + blk) * 2, sb, 2);
                }
                break;
            case GGML_TYPE_Q4_1:
                for (int64_t blk = 0; blk < g.n_sub; blk++) {
                    const uint8_t * sb = rsrc + blk * 20;
                    memcpy(rdst + blk * 16, sb + 4, 16);
                    memcpy(dst + g.d_off + (row * g.n_sub + blk) * 4, sb, 4);
                }
                break;
            case GGML_TYPE_Q5_1:
                for (int64_t blk = 0; blk < g.n_sub; blk++) {
                    const uint8_t * sb = rsrc + blk * 24;
                    const int64_t wi = row * g.n_sub + blk;
                    memcpy(rdst + blk * 16, sb + 8, 16);
                    memcpy(dst + g.h_off + wi * 4, sb + 4, 4);
                    memcpy(dst + g.d_off + wi * 4, sb, 4);
                }
                break;
            case GGML_TYPE_Q4_K: {
                for (int64_t sb = 0; sb < g.n_sub; sb++) {
                    const int k = (int) (sb & 7), sbk = (int) (sb >> 3);
                    const uint8_t * b = rsrc + sbk * 144;
                    const uint8_t * q = b + 16 + 32 * (k >> 1);
                    const int shift = (k & 1) * 4;
                    uint8_t * lows = rdst + sb * 16;
                    for (int j = 0; j < 16; j++) {
                        lows[j] = (uint8_t) (((q[j] >> shift) & 0xF) | (((q[j + 16] >> shift) & 0xF) << 4));
                    }
                    if (k == 0) {
                        const int64_t sbg = row * g.nds + sbk;
                        memcpy(dst + g.s_off + sbg * 12, b + 4, 12);
                        memcpy(dst + g.d_off + sbg * 4, b, 4);
                    }
                }
                break;
            }
            case GGML_TYPE_Q5_K: {
                for (int64_t sb = 0; sb < g.n_sub; sb++) {
                    const int k = (int) (sb & 7), sbk = (int) (sb >> 3);
                    const uint8_t * b = rsrc + sbk * 176;
                    const uint8_t * q = b + 48 + 32 * (k >> 1);
                    const int shift = (k & 1) * 4;
                    const int hbit  = 2 * (k >> 1) + (k & 1);
                    const int64_t wi = row * g.n_sub + sb;
                    uint8_t * lows = rdst + sb * 16;
                    uint32_t h = 0;
                    for (int j = 0; j < 16; j++) {
                        lows[j] = (uint8_t) (((q[j] >> shift) & 0xF) | (((q[j + 16] >> shift) & 0xF) << 4));
                    }
                    for (int j = 0; j < 32; j++) {
                        h |= (uint32_t) ((b[16 + j] >> hbit) & 1) << j;
                    }
                    memcpy(dst + g.h_off + wi * 4, &h, 4);
                    if (k == 0) {
                        const int64_t sbg = row * g.nds + sbk;
                        memcpy(dst + g.s_off + sbg * 12, b + 4, 12);
                        memcpy(dst + g.d_off + sbg * 4, b, 4);
                    }
                }
                break;
            }
            case GGML_TYPE_Q6_K: {
                for (int64_t sb = 0; sb < g.n_sub; sb++) {
                    const int k = (int) (sb & 7), sbk = (int) (sb >> 3);
                    const uint8_t * b = rsrc + sbk * 210;
                    const uint8_t * ql = b + 0;    // fork layout: ql first
                    const uint8_t * qh = b + 128;
                    uint8_t raw[32];
                    for (int j = 0; j < 32; j++) {
                        const int v = k * 32 + j;
                        const int h = v >> 7;
                        const int l = v & 127;
                        const uint8_t * qlh = ql + 64 * h;
                        const uint8_t * qhh = qh + 32 * h;
                        uint8_t r6;
                        if      (l <  32) { r6 = (uint8_t) ((qlh[l]      & 0xF) | ((((qhh[l]      >> 0) & 3) << 4))); }
                        else if (l <  64) { r6 = (uint8_t) ((qlh[l]      & 0xF) | ((((qhh[l - 32] >> 2) & 3) << 4))); }
                        else if (l <  96) { r6 = (uint8_t) ((qlh[l - 64] >>  4) | ((((qhh[l - 64] >> 4) & 3) << 4))); }
                        else              { r6 = (uint8_t) ((qlh[l - 64] >>  4) | ((((qhh[l - 96] >> 6) & 3) << 4))); }
                        raw[j] = r6;
                    }
                    const int64_t wi = row * g.n_sub + sb;
                    uint8_t * lows  = rdst + sb * 16;
                    uint8_t * highs = dst + g.h_off + wi * 8;
                    for (int j = 0; j < 16; j++) {
                        lows[j] = (uint8_t) ((raw[j] & 0xF) | ((raw[j + 16] & 0xF) << 4));
                    }
                    for (int m = 0; m < 8; m++) {
                        const int bs = (m & 3) * 4 + (m >> 2) * 16;
                        highs[m] = (uint8_t) (((raw[bs + 0] >> 4) << 0) | ((raw[bs + 1] >> 4) << 2)
                                            | ((raw[bs + 2] >> 4) << 4) | ((raw[bs + 3] >> 4) << 6));
                    }
                    memcpy(dst + g.s_off + wi * 2, b + 192 + 2 * k, 2);
                    if (k == 0) {
                        memcpy(dst + g.d_off + (row * g.nds + sbk) * 2, b + 208, 2);
                    }
                }
                break;
            }
            default:
                fprintf(stderr, "pack_type: unsupported type %d\n", (int) t); abort();
        }
    }
}

// ---------------------------------------------------------------------------
// unpack: repacked planes -> canonical blocks (inverse of the mapping above)
// ---------------------------------------------------------------------------

static void unpack_type(ggml_type t, const uint8_t * planes, uint8_t * canon,
                        int64_t ne0, int64_t ne1) {
    const rp_geom g = mk_geom(t, ne0, ne1);
    const int64_t rb = canon_layout(t).rb;
    const int64_t blocks_per_row = (t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K) ? g.nds : g.n_sub;
    memset(canon, 0, rb * blocks_per_row * ne1);

    for (int64_t row = 0; row < ne1; row++) {
        const uint8_t * rsrc = planes + row * g.rs;
        uint8_t * rdst = canon + row * rb * blocks_per_row;

        switch (t) {
            case GGML_TYPE_Q8_0:
                for (int64_t blk = 0; blk < g.n_sub; blk++) {
                    uint8_t * sb = rdst + blk * 34;
                    memcpy(sb + 2, rsrc + blk * 32, 32);
                    memcpy(sb, planes + g.d_off + (row * g.n_sub + blk) * 2, 2);
                }
                break;
            case GGML_TYPE_MXFP4:
                for (int64_t blk = 0; blk < g.n_sub; blk++) {
                    uint8_t * sb = rdst + blk * 17;
                    memcpy(sb + 1, rsrc + blk * 16, 16);
                    sb[0] = planes[g.d_off + row * g.n_sub + blk];
                }
                break;
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_IQ4_NL:
                for (int64_t blk = 0; blk < g.n_sub; blk++) {
                    uint8_t * sb = rdst + blk * 18;
                    memcpy(sb + 2, rsrc + blk * 16, 16);
                    memcpy(sb, planes + g.d_off + (row * g.n_sub + blk) * 2, 2);
                }
                break;
            case GGML_TYPE_Q4_1:
                for (int64_t blk = 0; blk < g.n_sub; blk++) {
                    uint8_t * sb = rdst + blk * 20;
                    memcpy(sb + 4, rsrc + blk * 16, 16);
                    memcpy(sb, planes + g.d_off + (row * g.n_sub + blk) * 4, 4);
                }
                break;
            case GGML_TYPE_Q5_1:
                for (int64_t blk = 0; blk < g.n_sub; blk++) {
                    const int64_t wi = row * g.n_sub + blk;
                    uint8_t * sb = rdst + blk * 24;
                    memcpy(sb + 8, rsrc + blk * 16, 16);
                    memcpy(sb + 4, planes + g.h_off + wi * 4, 4);
                    memcpy(sb, planes + g.d_off + wi * 4, 4);
                }
                break;
            case GGML_TYPE_Q4_K: {
                for (int64_t sb = 0; sb < g.n_sub; sb++) {
                    const int k = (int) (sb & 7), sbk = (int) (sb >> 3);
                    uint8_t * b = rdst + sbk * 144;
                    uint8_t * q = b + 16 + 32 * (k >> 1);
                    const int shift = (k & 1) * 4;
                    const uint8_t * lows = rsrc + sb * 16;
                    for (int j = 0; j < 16; j++) {
                        q[j]      = (uint8_t) ((q[j]      & ~(0xF << shift)) | ((lows[j]      & 0xF) << shift));
                        q[j + 16] = (uint8_t) ((q[j + 16] & ~(0xF << shift)) | ((lows[j] >> 4) << shift));
                    }
                    if (k == 0) {
                        const int64_t sbg = row * g.nds + sbk;
                        memcpy(b + 4, planes + g.s_off + sbg * 12, 12);
                        memcpy(b, planes + g.d_off + sbg * 4, 4);
                    }
                }
                break;
            }
            case GGML_TYPE_Q5_K: {
                for (int64_t sb = 0; sb < g.n_sub; sb++) {
                    const int k = (int) (sb & 7), sbk = (int) (sb >> 3);
                    uint8_t * b = rdst + sbk * 176;
                    uint8_t * q = b + 48 + 32 * (k >> 1);
                    const int shift = (k & 1) * 4;
                    const int hbit  = 2 * (k >> 1) + (k & 1);
                    const int64_t wi = row * g.n_sub + sb;
                    const uint8_t * lows = rsrc + sb * 16;
                    for (int j = 0; j < 16; j++) {
                        q[j]      = (uint8_t) ((q[j]      & ~(0xF << shift)) | ((lows[j]      & 0xF) << shift));
                        q[j + 16] = (uint8_t) ((q[j + 16] & ~(0xF << shift)) | ((lows[j] >> 4) << shift));
                    }
                    uint32_t h;
                    memcpy(&h, planes + g.h_off + wi * 4, 4);
                    for (int j = 0; j < 32; j++) {
                        b[16 + j] = (uint8_t) (b[16 + j] | (((h >> j) & 1) << hbit));
                    }
                    if (k == 0) {
                        const int64_t sbg = row * g.nds + sbk;
                        memcpy(b + 4, planes + g.s_off + sbg * 12, 12);
                        memcpy(b, planes + g.d_off + sbg * 4, 4);
                    }
                }
                break;
            }
            case GGML_TYPE_Q6_K: {
                for (int64_t sb = 0; sb < g.n_sub; sb++) {
                    const int k = (int) (sb & 7), sbk = (int) (sb >> 3);
                    uint8_t * b = rdst + sbk * 210;   // fork layout: ql@0 qh@128 scales@192 d@208
                    const int64_t wi = row * g.n_sub + sb;
                    const uint8_t * lows  = rsrc + sb * 16;
                    const uint8_t * highs = planes + g.h_off + wi * 8;
                    for (int j = 0; j < 32; j++) {
                        const uint8_t lo  = (j < 16) ? (uint8_t) (lows[j] & 0xF) : (uint8_t) (lows[j - 16] >> 4);
                        const uint8_t hi2 = (uint8_t) ((highs[j / 4] >> (2 * (j % 4))) & 3);
                        const uint8_t r6  = (uint8_t) (lo | (hi2 << 4));
                        const int v = k * 32 + j;
                        const int h = v >> 7;
                        const int l = v & 127;
                        uint8_t * ql = b + 0 + 64 * h;
                        uint8_t * qh = b + 128 + 32 * h;
                        if (l < 32)        { ql[l]      = (uint8_t) ((ql[l]      & 0xF0) | (r6 & 0xF));          qh[l]      = (uint8_t) ((qh[l]      & 0xFC) | ((r6 >> 4) << 0)); }
                        else if (l < 64)   { ql[l]      = (uint8_t) ((ql[l]      & 0xF0) | (r6 & 0xF));          qh[l - 32] = (uint8_t) ((qh[l - 32] & 0xF3) | ((r6 >> 4) << 2)); }
                        else if (l < 96)   { ql[l - 64] = (uint8_t) ((ql[l - 64] & 0x0F) | ((r6 & 0xF) << 4));   qh[l - 64] = (uint8_t) ((qh[l - 64] & 0xCF) | ((r6 >> 4) << 4)); }
                        else               { ql[l - 64] = (uint8_t) ((ql[l - 64] & 0x0F) | ((r6 & 0xF) << 4));   qh[l - 96] = (uint8_t) ((qh[l - 96] & 0x3F) | ((r6 >> 4) << 6)); }
                    }
                    memcpy(b + 192 + 2 * k, planes + g.s_off + wi * 2, 2);
                    if (k == 0) {
                        memcpy(b + 208, planes + g.d_off + (row * g.nds + sbk) * 2, 2);
                    }
                }
                break;
            }
            default:
                fprintf(stderr, "unpack_type: unsupported type %d\n", (int) t); abort();
        }
    }
}

// ---------------------------------------------------------------------------
// decode: repacked planes -> f32 values, using the READER's documented
// value-index mapping (rp_traits<WT>::load_w + the affine epilogues).
// This is the anti-pairing-bug anchor (P2).
// ---------------------------------------------------------------------------

// IQ4_NL codebook — frozen by the format spec; copied from
// ggml/src/ggml-common.h kvalues_iq4nl (the GGML_TABLE section is not visible
// through the public include path).
static const int8_t ktest_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// Q4_K/Q5_K 6-bit scale/min unpacking — literal copy of get_scale_min_k4()
// from ggml/src/ggml-quants.c (the scales record packs 6-bit fields across
// byte boundaries; a naive nibble split is WRONG).
static void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

static float fp16(const uint8_t * p) {
    uint16_t u; memcpy(&u, p, 2);
    return ggml_fp16_to_fp32(u);
}

static void decode_planes(ggml_type t, const uint8_t * planes, float * out,
                          int64_t ne0, int64_t ne1) {
    const rp_geom g = mk_geom(t, ne0, ne1);
    for (int64_t row = 0; row < ne1; row++) {
        const uint8_t * rsrc = planes + row * g.rs;
        float * rdst = out + row * ne0;
        switch (t) {
            case GGML_TYPE_Q8_0:
                for (int64_t sb = 0; sb < g.n_sub; sb++) {
                    const float d = fp16(planes + g.d_off + (row * g.n_sub + sb) * 2);
                    for (int j = 0; j < 32; j++) {
                        rdst[sb * 32 + j] = d * (float) ((int8_t) rsrc[sb * 32 + j]);
                    }
                }
                break;
            case GGML_TYPE_Q4_0:
                // canonical Q4_0 order: byte j low nibble = value j, high = value j+16; x = d*(q-8)
                for (int64_t sb = 0; sb < g.n_sub; sb++) {
                    const float d = fp16(planes + g.d_off + (row * g.n_sub + sb) * 2);
                    for (int j = 0; j < 16; j++) {
                        const uint8_t b = rsrc[sb * 16 + j];
                        rdst[sb * 32 + j]      = d * (float) ((b & 0xF) - 8);
                        rdst[sb * 32 + 16 + j] = d * (float) ((b >> 4) - 8);
                    }
                }
                break;
            case GGML_TYPE_IQ4_NL:
                for (int64_t sb = 0; sb < g.n_sub; sb++) {
                    const float d = fp16(planes + g.d_off + (row * g.n_sub + sb) * 2);
                    for (int j = 0; j < 16; j++) {
                        const uint8_t b = rsrc[sb * 16 + j];
                        rdst[sb * 32 + j]      = d * (float) ktest_kvalues_iq4nl[b & 0xF];
                        rdst[sb * 32 + 16 + j] = d * (float) ktest_kvalues_iq4nl[b >> 4];
                    }
                }
                break;
            case GGML_TYPE_Q4_1:
                for (int64_t sb = 0; sb < g.n_sub; sb++) {
                    const float d = fp16(planes + g.d_off + (row * g.n_sub + sb) * 4);
                    const float m = fp16(planes + g.d_off + (row * g.n_sub + sb) * 4 + 2);
                    for (int j = 0; j < 16; j++) {
                        const uint8_t b = rsrc[sb * 16 + j];
                        rdst[sb * 32 + j]      = d * (float) (b & 0xF) + m;
                        rdst[sb * 32 + 16 + j] = d * (float) (b >> 4) + m;
                    }
                }
                break;
            case GGML_TYPE_Q5_1:
                for (int64_t sb = 0; sb < g.n_sub; sb++) {
                    const int64_t wi = row * g.n_sub + sb;
                    const float d = fp16(planes + g.d_off + wi * 4);
                    const float m = fp16(planes + g.d_off + wi * 4 + 2);
                    uint32_t h; memcpy(&h, planes + g.h_off + wi * 4, 4);
                    for (int j = 0; j < 16; j++) {
                        const uint8_t b = rsrc[sb * 16 + j];
                        const int v0 = (b & 0xF) | (int) (((h >> j)      & 1) << 4);
                        const int v1 = (b >> 4)  | (int) (((h >> (j+16)) & 1) << 4);
                        rdst[sb * 32 + j]      = d * (float) v0 + m;
                        rdst[sb * 32 + 16 + j] = d * (float) v1 + m;
                    }
                }
                break;
            case GGML_TYPE_Q4_K:
                for (int64_t sb = 0; sb < g.n_sub; sb++) {
                    const int k = (int) (sb & 7), sbk = (int) (sb >> 3);
                    const int64_t sbg = row * g.nds + sbk;
                    const float d   = fp16(planes + g.d_off + sbg * 4);
                    const float dmn = fp16(planes + g.d_off + sbg * 4 + 2);
                    const uint8_t * rec = planes + g.s_off + sbg * 12;
                    uint8_t sc6, mn6; get_scale_min_k4(k, rec, &sc6, &mn6);
                    const float sc = d   * (float) sc6;
                    const float mn = dmn * (float) mn6;
                    for (int j = 0; j < 16; j++) {
                        const uint8_t b = rsrc[sb * 16 + j];
                        rdst[sb * 32 + j]      = sc * (float) (b & 0xF) - mn;
                        rdst[sb * 32 + 16 + j] = sc * (float) (b >> 4) - mn;
                    }
                }
                break;
            case GGML_TYPE_Q5_K:
                for (int64_t sb = 0; sb < g.n_sub; sb++) {
                    const int k = (int) (sb & 7), sbk = (int) (sb >> 3);
                    const int64_t wi = row * g.n_sub + sb;
                    const int64_t sbg = row * g.nds + sbk;
                    const float d   = fp16(planes + g.d_off + sbg * 4);
                    const float dmn = fp16(planes + g.d_off + sbg * 4 + 2);
                    const uint8_t * rec = planes + g.s_off + sbg * 12;
                    uint8_t sc6, mn6; get_scale_min_k4(k, rec, &sc6, &mn6);
                    const float sc = d   * (float) sc6;
                    const float mn = dmn * (float) mn6;
                    const int hbit = 2 * (k >> 1) + (k & 1);
                    uint32_t h; memcpy(&h, planes + g.h_off + wi * 4, 4);
                    for (int j = 0; j < 16; j++) {
                        const uint8_t b = rsrc[sb * 16 + j];
                        const int v0 = (b & 0xF) | (int) (((h >> j)      & 1) << 4);
                        const int v1 = (b >> 4)  | (int) (((h >> (j+16)) & 1) << 4);
                        rdst[sb * 32 + j]      = sc * (float) v0 - mn;
                        rdst[sb * 32 + 16 + j] = sc * (float) v1 - mn;
                    }
                }
                break;
            case GGML_TYPE_Q6_K:
                for (int64_t sb = 0; sb < g.n_sub; sb++) {
                    const int k = (int) (sb & 7), sbk = (int) (sb >> 3);
                    const int64_t wi = row * g.n_sub + sb;
                    const float d = fp16(planes + g.d_off + (row * g.nds + sbk) * 2);
                    (void) k;
                    for (int j = 0; j < 32; j++) {
                        const uint8_t lo  = (j < 16) ? (uint8_t) (rsrc[sb * 16 + j] & 0xF)
                                                     : (uint8_t) (rsrc[sb * 16 + (j - 16)] >> 4);
                        const uint8_t hi2 = (uint8_t) ((planes[g.h_off + wi * 8 + j / 4] >> (2 * (j % 4))) & 3);
                        const int q = (int) (lo | (hi2 << 4)) - 32;
                        const int8_t sc = (int8_t) planes[g.s_off + wi * 2 + (j >= 16)];
                        rdst[sb * 32 + j] = d * (float) sc * (float) q;
                    }
                    (void) sbk;
                }
                break;
            default:
                fprintf(stderr, "decode_planes: unsupported type %d\n", (int) t); abort();
        }
    }
}

// ---------------------------------------------------------------------------
// block generation
// ---------------------------------------------------------------------------

static uint32_t rng_state = 0xC0FFEE11;
static float frand(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return (float) (rng_state % 2000001u) / 1000000.0f - 1.0f;
}

static void quantize_blocks(ggml_type t, const float * f32, uint8_t * out, int64_t ne) {
    switch (t) {
        case GGML_TYPE_Q8_0:   quantize_row_q8_0_ref(f32, (block_q8_0 *) out, ne); break;
        case GGML_TYPE_Q4_0:   quantize_row_q4_0_ref(f32, (block_q4_0 *) out, ne); break;
        case GGML_TYPE_Q4_1:   quantize_row_q4_1_ref(f32, (block_q4_1 *) out, ne); break;
        case GGML_TYPE_Q5_1:   quantize_row_q5_1_ref(f32, (block_q5_1 *) out, ne); break;
        case GGML_TYPE_IQ4_NL: quantize_row_iq4_nl_ref(f32, (block_iq4_nl *) out, ne); break;
        case GGML_TYPE_Q4_K:   quantize_row_q4_K_ref(f32, (block_q4_K *) out, ne); break;
        case GGML_TYPE_Q5_K:   quantize_row_q5_K_ref(f32, (block_q5_K *) out, ne); break;
        case GGML_TYPE_Q6_K:   quantize_row_q6_K_ref(f32, (block_q6_K *) out, ne); break;
        case GGML_TYPE_MXFP4:  quantize_row_mxfp4_ref(f32, (block_mxfp4 *) out, ne); break;
        default: fprintf(stderr, "quantize_blocks: unsupported type %d\n", (int) t); abort();
    }
}

static void dequant_canon(ggml_type t, const uint8_t * canon, float * out, int64_t ne) {
    switch (t) {
        case GGML_TYPE_Q8_0:   dequantize_row_q8_0((const block_q8_0 *) canon, out, ne); break;
        case GGML_TYPE_Q4_0:   dequantize_row_q4_0((const block_q4_0 *) canon, out, ne); break;
        case GGML_TYPE_Q4_1:   dequantize_row_q4_1((const block_q4_1 *) canon, out, ne); break;
        case GGML_TYPE_Q5_1:   dequantize_row_q5_1((const block_q5_1 *) canon, out, ne); break;
        case GGML_TYPE_IQ4_NL: dequantize_row_iq4_nl((const block_iq4_nl *) canon, out, ne); break;
        case GGML_TYPE_Q4_K:   dequantize_row_q4_K((const block_q4_K *) canon, out, ne); break;
        case GGML_TYPE_Q5_K:   dequantize_row_q5_K((const block_q5_K *) canon, out, ne); break;
        case GGML_TYPE_Q6_K:   dequantize_row_q6_K((const block_q6_K *) canon, out, ne); break;
        case GGML_TYPE_MXFP4:  dequantize_row_mxfp4((const block_mxfp4 *) canon, out, ne); break;
        default: fprintf(stderr, "dequant_canon: unsupported type %d\n", (int) t); abort();
    }
}

// structured blocks: pattern-filled payload bytes with valid scale fields.
// Any nibble/scale byte pattern is a valid block for these types, so we can
// probe all-nibble-extreme layouts that random quantization never produces.
static void structured_blocks(ggml_type t, uint8_t * out, int64_t ne0, int64_t ne1, int variant) {
    const canon_off c = canon_layout(t);
    const int64_t blocks_per_row = (t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K) ? (ne0 / 256) : (ne0 / 32);
    memset(out, 0, c.rb * blocks_per_row * ne1);
    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < blocks_per_row; blk++) {
            uint8_t * b = out + (row * blocks_per_row + blk) * c.rb;
            for (int64_t i = 0; i < c.rb; i++) {
                switch (variant) {
                    case 0: b[i] = 0x00; break;
                    case 1: b[i] = 0xFF; break;
                    case 2: b[i] = (i & 1) ? 0x0F : 0xF0; break;
                    case 3: b[i] = (uint8_t) ((i * 37 + row * 7 + blk) & 0xFF); break;
                    case 4: b[i] = 0x55; break;
                    case 5: b[i] = 0xAA; break;
                }
            }
            // valid scale fields
            const uint16_t d_bits = 0x3FC0;   // 1.5f
            const uint16_t m_bits = 0xBF00;   // -0.5f
            memcpy(b + c.d, &d_bits, 2);
            if (c.m >= 0)    { memcpy(b + c.m, &m_bits, 2); }
            if (c.dmin >= 0) { const uint16_t dm = 0x3900; memcpy(b + c.dmin, &dm, 2); } // 0.5f
            if (t == GGML_TYPE_MXFP4) { b[0] = (uint8_t) (120 + (blk % 15)); }           // e8m0, avoid 255=NaN
        }
    }
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

static int run_case(ggml_type t, int64_t ne0, int64_t ne1, int blocks_mode, int variant) {
    const rp_geom g = mk_geom(t, ne0, ne1);
    const canon_off c = canon_layout(t);
    const int64_t blocks_per_row = (t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K) ? g.nds : g.n_sub;
    const size_t canon_bytes = (size_t) c.rb * blocks_per_row * ne1;

    std::vector<uint8_t> canon(canon_bytes), canon2(canon_bytes);
    std::vector<uint8_t> dst1(g.total + 4096, 0xAA), dst2(g.total + 4096, 0xAA), dst3(g.total + 4096, 0xAA);
    std::vector<float> dec((size_t) ne0 * ne1), ref((size_t) ne0 * ne1);

    if (blocks_mode == 0) {
        std::vector<float> f32((size_t) ne0 * ne1);
        for (auto & x : f32) x = frand();
        quantize_blocks(t, f32.data(), canon.data(), ne0 * ne1);
    } else {
        structured_blocks(t, canon.data(), ne0, ne1, variant);
    }

    // P1: pack -> unpack -> pack byte-identity
    pack_type(t, canon.data(), dst1.data(), ne0, ne1);
    unpack_type(t, dst1.data(), canon2.data(), ne0, ne1);
    pack_type(t, canon2.data(), dst2.data(), ne0, ne1);
    if (memcmp(dst1.data(), dst2.data(), g.total) != 0) {
        size_t bad = 0, first = 0;
        for (size_t i = 0; i < g.total; i++) if (dst1[i] != dst2[i]) { if (!bad) first = i; bad++; }
        printf("FAIL P1 %s ne0=%lld ne1=%lld mode=%d var=%d: repack differs (%zu bytes, first @%zu: %02x vs %02x)\n",
               ggml_type_name(t), (long long) ne0, (long long) ne1, blocks_mode, variant, bad, first, dst1[first], dst2[first]);
        return 1;
    }
    if (memcmp(canon.data(), canon2.data(), canon_bytes) != 0) {
        printf("FAIL P1 %s ne0=%lld ne1=%lld mode=%d var=%d: unpacked canonical differs from original\n",
               ggml_type_name(t), (long long) ne0, (long long) ne1, blocks_mode, variant);
        return 1;
    }

    // P2: decode-from-planes == dequantize_row_<type>(canonical)
    if (t != GGML_TYPE_MXFP4) {  // MXFP4: byte round-trip only (e8m0 decode lives behind a CUDA-only macro)
        decode_planes(t, dst1.data(), dec.data(), ne0, ne1);
        dequant_canon(t, canon.data(), ref.data(), ne0 * ne1);
        double worst = 0.0; size_t wi = 0;
        for (size_t i = 0; i < dec.size(); i++) {
            const double tol = 1e-3 + 1e-2 * fabs((double) ref[i]);
            const double err = fabs((double) dec[i] - (double) ref[i]);
            if (err > tol && err / tol > worst) { worst = err / tol; wi = i; }
        }
        if (worst > 1.0) {
            printf("FAIL P2 %s ne0=%lld ne1=%lld mode=%d var=%d: plane decode deviates from canonical dequant "
                   "(worst @%zu: got %.6f want %.6f)\n",
                   ggml_type_name(t), (long long) ne0, (long long) ne1, blocks_mode, variant,
                   wi, (double) dec[wi], (double) ref[wi]);
            return 1;
        }
    }

    // P3: mirror == shipped converter (when linked)
    if (g_repack_host) {
        memset(dst3.data(), 0xAA, g.total);
        g_repack_host(t, canon.data(), dst3.data(), ne0, ne1);
        if (memcmp(dst1.data(), dst3.data(), g.total) != 0) {
            size_t bad = 0, first = 0;
            for (size_t i = 0; i < g.total; i++) if (dst1[i] != dst3[i]) { if (!bad) first = i; bad++; }
            printf("FAIL P3 %s ne0=%lld ne1=%lld mode=%d var=%d: mirror pack differs from repack_host (%zu bytes, first @%zu)\n",
                   ggml_type_name(t), (long long) ne0, (long long) ne1, blocks_mode, variant, bad, first);
            {
                printf("  P3 diffs (plane layout: rs=%lld qs=%zu h=%zu s=%zu d=%zu total=%zu):\n",
                       (long long) g.rs, (size_t) ne1 * g.rs, g.h_off, g.s_off, g.d_off, g.total);
                size_t shown = 0;
                for (size_t i = 0; i < g.total && shown < 16; i++) {
                    if (dst1[i] != dst3[i]) { printf("   @%zu mirror=%02x shipped=%02x\n", i, dst1[i], dst3[i]); shown++; }
                }
            }
            return 1;
        }
    }

    printf("ok: %-8s ne0=%5lld ne1=%lld %-10s P1=pass P2=%s P3=%s\n",
           ggml_type_name(t), (long long) ne0, (long long) ne1,
           blocks_mode == 0 ? "random" : "structured",
           t == GGML_TYPE_MXFP4 ? "n/a" : "pass",
           (g_repack_host) ? "pass" : "skipped");
    return 0;
}

int main(void) {
    setenv("GGML_CUDA_W4A4_SUMW", "0", 1);  // keep the shipped Q4_0 converter at the 18 B/block layout

    const ggml_type types[] = {
        GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1,
        GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
    };
    // 5120/4096 exercise the de-alias row bump (ne0/2 % 128 == 0); 320 does not (non-K types only).
    const int64_t shapes[][2] = { {256, 2}, {5120, 3}, {4096, 2}, {320, 1}, {1024, 5} };

    g_repack_host = resolve_repack_host();
    if (!g_repack_host) {
        printf("WARNING: P3 SKIPPED FOR EVERY CASE -- the shipped converter (repack_host) could not\n"
               "WARNING: be resolved from libggml-hip.so.0 / libggml-cuda.so.0 (HIP backend library or\n"
               "WARNING: ROCm runtime not on the loader path). The mirror was validated against itself\n"
               "WARNING: only (P1/P2); P3 is the only check against the shipped converter. Build/run\n"
               "WARNING: with the HIP backend library loadable to activate P3.\n");
    }

    int rc = 0, cases = 0;
    for (ggml_type t : types) {
        for (auto & s : shapes) {
            const int64_t ne0 = s[0], ne1 = s[1];
            const bool is_k = (t == GGML_TYPE_Q4_K || t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K);
            if (is_k && (ne0 % 256) != 0) continue;   // K super-blocks need ne0 % 256 == 0
            rc += run_case(t, ne0, ne1, 0, 0); cases++;
            for (int v = 0; v < 6; v++) { rc += run_case(t, ne0, ne1, 1, v); cases++; }
        }
    }

    printf("test-repack-nibble-roundtrip: %s (%d cases, P3 %s)\n",
           rc == 0 ? "PASS" : "FAIL", cases, (g_repack_host) ? "active" : "skipped (repack_host not resolvable)");
    return rc;
}
