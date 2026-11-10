/*
# Seal: exact-8-bit (i8split) nibble-dot path for narrow-batch Q4_0 GEMV/GEMM

Proves, on CPU only (no GPU, no build):

  (a) INTEGER IDENTITY (the quality claim): for every 32-value Q4_0 block and
      every int8 activation vector produced by the stock q8_1 quantizer
      (quantize_row_q8_1_ref semantics, ggml-quants.c:302), the i8split path

          S = 16*D_hi + D_lo + 8*SUMW          (exact int32)

      equals the stock int8 path's integer partial sum

          S_stock = (ilo + ihi) - 8*SA         (dot4 on excess-8 nibble bytes)

      and both equal the direct integer dot  sum_i (c_i - 8) * a_i.
      Hence the fp32 epilogue consumes the SAME integer partial sums as the
      stock q8_1 path: no quality cost.

  (b) SIGN HANDLING: a_hi = a >> 4 (arithmetic) is a two's-complement nibble
      for the full range [-128, 127]; a_lo = a & 0xF is stored biased as
      (a_lo ^ 8) so the dot8 operand is signed; the +8*SUMW term is the exact
      correction for that bias. -128 and +127 are exercised explicitly.

  (c) PAIRING: nibble planes use the canonical Q4_0 byte order
      byte j = {value j (low nibble), value j+16 (high nibble)}
      (quantize_row_q4_0_ref, ggml-quants.c:137-146). A wrong pairing that
      packs adjacent values {2k, 2k+1} per byte is DETECTED (deterministic
      witness + randomized check).

  (d) PER-REGIME COEFFICIENT ASSERTS: each adversarial regime declares the
      expected coefficients (D_hi, D_lo, S) as multiples of SUMW and the
      harness asserts the realized coefficients; degenerate regimes declare
      zero coefficients and the harness asserts the degeneracy (no vacuous
      passes: non-degenerate regimes require SUMW != 0 and a nonzero
      activation-lo spread witness).

  (e) fp32 EPILOGUE: the documented fold
          acc += d_w * (d_a * S)        (per 32-block, fp32)
      matches a double-precision dequant reference within a stated tolerance
      on random fuzz, and the fp16 scale-storage variant
      (d_w, d_a stored f16 as in block_q8_1 / the d plane) within a wider
      tolerance.

Build:  g++ -O2 -std=c++17 -o seal_i8split seal_i8split.cc
Run:    ./seal_i8split
Exit:   0 iff every invariant passes.
*/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

static int g_fail = 0;
static int g_pass = 0;
#define CHECK(name, cond)                                                       \
    do {                                                                        \
        const bool ok_ = (cond);                                                \
        printf("INVARIANT %-64s %s\n", name, ok_ ? "PASS" : "FAIL");           \
        if (!ok_) g_fail++; else g_pass++;                                      \
    } while (0)

// ---------------------------------------------------------------------------
// f16 (IEEE binary16, round-to-nearest-even) — matches GGML_FP32_TO_FP16.
static uint16_t f32_to_f16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man  = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u));
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t) sign;
        man |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t half = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1u);
        const uint32_t half_even = half & 1u;
        if (rem > (uint32_t)(1 << (shift - 1)) || (rem == (uint32_t)(1 << (shift - 1)) && half_even)) half++;
        return (uint16_t)(sign | half);
    }
    uint32_t half = (uint32_t)exp << 10 | (man >> 13);
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half++;
    return (uint16_t)(sign | half);
}
static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t man  = h & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        // subnormal: value = man * 2^-24
        const float v = std::ldexp((float) man, -24);
        std::memcpy(&out, &v, 4); out |= sign;
    } else if (exp == 0x1F) out = sign | 0x7F800000u | (man << 13);
    else { out = sign | ((exp + 112u) << 23) | (man << 13); }
    float f; std::memcpy(&f, &out, 4); return f;
}

// ---------------------------------------------------------------------------
// Hardware-op emulations (verbatim semantics of the tree's portable fallbacks).
// ggml_cuda_dp8a fallback, common.cuh:763-772: 8 signed-nibble MACs,
// nibble i = bits 4i..4i+3 sign-extended (little-endian nibble order).
static inline int dp8a(int32_t a, int32_t b, int32_t c) {
    const uint32_t ua = (uint32_t) a, ub = (uint32_t) b;
    for (int i = 0; i < 8; ++i) {
        const int va = (int)(ua << (28 - 4*i)) >> 28;
        const int vb = (int)(ub << (28 - 4*i)) >> 28;
        c += va * vb;
    }
    return c;
}
// ggml_cuda_dp4a scalar fallback, common.cuh:736-738 (signed bytes, LE).
static inline int dp4a(int32_t a, int32_t b, int32_t c) {
    const int8_t * pa = (const int8_t *) &a;
    const int8_t * pb = (const int8_t *) &b;
    return c + pa[0]*pb[0] + pa[1]*pb[1] + pa[2]*pb[2] + pa[3]*pb[3];
}
// mmvq_dp8a_u4, repack-kernels.cuh:63-74: 4 chained dot8 over a uint4.
struct U4 { uint32_t x, y, z, w; };
static inline int mmvq_dp8a_u4(const U4 & w, const U4 & a) {
    int r = dp8a((int32_t)w.x, (int32_t)a.x, 0);
    r = dp8a((int32_t)w.y, (int32_t)a.y, r);
    r = dp8a((int32_t)w.z, (int32_t)a.z, r);
    r = dp8a((int32_t)w.w, (int32_t)a.w, r);
    return r;
}
// nib_sum_u4, repack-kernels.cuh:78-88 (unsigned nibble sum of 32 codes).
static inline uint32_t nib_sum_u4(const U4 & v) {
    const uint32_t ws[4] = { v.x, v.y, v.z, v.w };
    uint32_t s = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t x = ws[i];
        s += (x & 0xFu) + ((x >> 4) & 0xFu) + ((x >> 8) & 0xFu) + ((x >> 12) & 0xFu)
           + ((x >> 16) & 0xFu) + ((x >> 20) & 0xFu) + ((x >> 24) & 0xFu) + ((x >> 28) & 0xFu);
    }
    return s;
}
static inline int32_t sumw_signed(const U4 & raw) { return (int32_t) nib_sum_u4(raw) - 256; }

// ---------------------------------------------------------------------------
// Q4_0 block (canonical): f16 scale + 16 bytes, byte j = {val j low, val j+16 high},
// excess-8 codes. quantize_row_q4_0_ref, ggml-quants.c:113-147.
struct BlockQ40 { uint16_t d; U4 qs; };

static void quantize_row_q4_0_ref(const float * x, BlockQ40 * y, int nb) {
    for (int i = 0; i < nb; ++i) {
        const float * xb = x + i * 32;
        float amax = 0.0f, maxv = 0.0f;
        for (int j = 0; j < 32; ++j) {
            const float v = xb[j];
            if (amax < std::fabs(v)) { amax = std::fabs(v); maxv = v; }
        }
        const float d  = maxv / -8.0f;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = f32_to_f16(d);
        uint8_t * q = (uint8_t *) &y[i].qs;
        for (int j = 0; j < 16; ++j) {
            const float x0 = xb[j]      * id;
            const float x1 = xb[j + 16] * id;
            const uint8_t xi0 = (uint8_t) std::min(15, (int) (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = (uint8_t) std::min(15, (int) (int8_t)(x1 + 8.5f));
            q[j] = xi0 | (uint8_t)(xi1 << 4);   // byte j = {val j low, val j+16 high}
        }
    }
}

// Stock q8_1 activation quantizer (quantize_row_q8_1_ref, ggml-quants.c:302):
// d = amax/127 stored f16; a_j = roundf(x_j / d) as int8, PLAIN value order.
struct BlockQ81 { uint16_t d; int8_t qs[32]; };
static void quantize_row_q8_1_ref(const float * x, BlockQ81 * y, int nb) {
    for (int i = 0; i < nb; ++i) {
        const float * xb = x + i * 32;
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(xb[j]));
        const float d = amax / 127.0f;
        y[i].d = f32_to_f16(d);
        const float id = d ? 1.0f / d : 0.0f;   // NOTE: id from the f32 d, as in the ref
        for (int j = 0; j < 16; ++j) {
            y[i].qs[j]      = (int8_t) roundf(xb[j]      * id);
            y[i].qs[16 + j] = (int8_t) roundf(xb[j + 16] * id);
        }
    }
}

// ---------------------------------------------------------------------------
// i8split activation planes (design: DESIGN-i8-narrow-kernel.md).
// hi plane: 32 signed nibbles a_hi = a >> 4 (arithmetic), canonical pairing:
//           byte j = {a_hi(j) low nibble, a_hi(j+16) high nibble}.
// lo plane: 32 nibbles a_lo = a & 0xF stored BIAS-8: stored = (a_lo ^ 8),
//           i.e. the signed nibble value is a_lo - 8 in [-8, 7]; same pairing.
static inline int8_t nib_s(uint8_t b, int low) {   // signed nibble of byte b
    return (int8_t)((b >> (low ? 0 : 4)) << 4) >> 4;
}
static inline int nib_u(uint8_t b, int low) {      // unsigned nibble of byte b
    return (int)((b >> (low ? 0 : 4)) & 0xFu);
}
static U4 pack_hi(const int8_t * a) {  // a[32] int8 activations, value order
    U4 r; uint8_t * p = (uint8_t *) &r;
    for (int j = 0; j < 16; ++j) {
        const uint8_t lo = (uint8_t)((a[j]      >> 4) & 0xF);
        const uint8_t hi = (uint8_t)((a[j + 16] >> 4) & 0xF);
        p[j] = lo | (uint8_t)(hi << 4);
    }
    return r;
}
static U4 pack_lo(const int8_t * a) {
    U4 r; uint8_t * p = (uint8_t *) &r;
    for (int j = 0; j < 16; ++j) {
        const uint8_t lo = (uint8_t)((a[j]      & 0xF) ^ 8);
        const uint8_t hi = (uint8_t)((a[j + 16] & 0xF) ^ 8);
        p[j] = lo | (uint8_t)(hi << 4);
    }
    return r;
}
// WRONG pairing (the bug class): adjacent values {2k, 2k+1} per byte.
static U4 pack_hi_wrong(const int8_t * a) {
    U4 r; uint8_t * p = (uint8_t *) &r;
    for (int j = 0; j < 16; ++j) {
        const uint8_t lo = (uint8_t)((a[2*j]     >> 4) & 0xF);
        const uint8_t hi = (uint8_t)((a[2*j + 1] >> 4) & 0xF);
        p[j] = lo | (uint8_t)(hi << 4);
    }
    return r;
}
static U4 pack_lo_wrong(const int8_t * a) {
    U4 r; uint8_t * p = (uint8_t *) &r;
    for (int j = 0; j < 16; ++j) {
        const uint8_t lo = (uint8_t)((a[2*j]     & 0xF) ^ 8);
        const uint8_t hi = (uint8_t)((a[2*j + 1] & 0xF) ^ 8);
        p[j] = lo | (uint8_t)(hi << 4);
    }
    return r;
}

struct I8Result { int32_t D_hi, D_lo, SUMW, S; };
// The i8split integer path: two dot8 quads + the exact bias correction.
static I8Result i8split_dot(const U4 & w_raw, const int8_t * a) {
    const U4 w_tc = { w_raw.x ^ 0x88888888u, w_raw.y ^ 0x88888888u,
                      w_raw.z ^ 0x88888888u, w_raw.w ^ 0x88888888u };
    I8Result r;
    r.D_hi  = mmvq_dp8a_u4(w_tc, pack_hi(a));
    r.D_lo  = mmvq_dp8a_u4(w_tc, pack_lo(a));
    r.SUMW  = sumw_signed(w_raw);
    r.S     = 16 * r.D_hi + r.D_lo + 8 * r.SUMW;
    return r;
}
// The stock q8_1 integer path (nc_impl stock branch, repack-kernels.cuh:2400-2448):
// masked nibble BYTES (excess-8 codes) vs int8 activations in plain value order.
struct StockResult { int32_t I, SA, S; };
static StockResult stock_dot(const U4 & w_raw, const int8_t * a) {
    const int32_t w[8] = {
        (int32_t)(w_raw.x & 0x0F0F0F0Fu), (int32_t)(w_raw.y & 0x0F0F0F0Fu),
        (int32_t)(w_raw.z & 0x0F0F0F0Fu), (int32_t)(w_raw.w & 0x0F0F0F0Fu),
        (int32_t)((w_raw.x >> 4) & 0x0F0F0F0Fu), (int32_t)((w_raw.y >> 4) & 0x0F0F0F0Fu),
        (int32_t)((w_raw.z >> 4) & 0x0F0F0F0Fu), (int32_t)((w_raw.w >> 4) & 0x0F0F0F0Fu) };
    const int32_t x[8] = { [&]{ int32_t v; std::memcpy(&v, a + 0,  4); return v; }(),
                           [&]{ int32_t v; std::memcpy(&v, a + 4,  4); return v; }(),
                           [&]{ int32_t v; std::memcpy(&v, a + 8,  4); return v; }(),
                           [&]{ int32_t v; std::memcpy(&v, a + 12, 4); return v; }(),
                           [&]{ int32_t v; std::memcpy(&v, a + 16, 4); return v; }(),
                           [&]{ int32_t v; std::memcpy(&v, a + 20, 4); return v; }(),
                           [&]{ int32_t v; std::memcpy(&v, a + 24, 4); return v; }(),
                           [&]{ int32_t v; std::memcpy(&v, a + 28, 4); return v; }() };
    int ilo = 0, ihi = 0, sa = 0;
    for (int i = 0; i < 4; ++i) ilo = dp4a(w[i],     x[i],     ilo);
    for (int i = 0; i < 4; ++i) ihi = dp4a(w[4 + i], x[4 + i], ihi);
    for (int i = 0; i < 32; ++i) sa += a[i];
    StockResult r; r.I = ilo + ihi; r.SA = sa; r.S = r.I - 8 * sa;
    return r;
}
// Direct integer reference: sum (c_i - 8) * a_i, c read nibble-by-nibble.
static int32_t direct_dot(const U4 & w_raw, const int8_t * a) {
    const uint8_t * q = (const uint8_t *) &w_raw;
    int32_t s = 0;
    for (int j = 0; j < 16; ++j) {
        s += (int32_t)(nib_u(q[j], 1) - 8) * a[j];        // value j
        s += (int32_t)(nib_u(q[j], 0) - 8) * a[j + 16];   // value j+16
    }
    return s;
}

// ---------------------------------------------------------------------------
// Weight-block generators (adversarial nibble packs).
static U4 w_pattern(int pat, int seed) {
    U4 r; uint8_t * p = (uint8_t *) &r;
    for (int j = 0; j < 16; ++j) {
        switch (pat) {
            case 0: p[j] = 0x00; break;                       // all codes 0  (w = -8)
            case 1: p[j] = 0xFF; break;                       // all codes 15 (w = +7)
            case 2: p[j] = (j & 1) ? 0xF0 : 0x0F; break;      // alternating
            case 3: {                                          // single nibble lane set
                const int pos = seed % 32, byte = pos >> 1, nib = pos & 1;
                p[j] = (j == byte) ? (uint8_t)(nib ? 0x10 : 0x01) : 0x00; break; }
            default: { uint32_t s = (uint32_t)(seed * 2654435761u + j * 40503u + pat);
                      s ^= s >> 13; s *= 1274126177u; s ^= s >> 16;
                      p[j] = (uint8_t) s; } break;
        }
    }
    return r;
}

// Regime descriptor: declares the EXPECTED coefficients so the harness can
// assert the realized ones (vacuity guard: several regimes are degenerate).
struct Regime {
    const char * name;
    int const_a;          // INT32_MIN = varying pattern
    int c_hi, c_lo, c_s;  // expected D_hi = c_hi*SUMW, D_lo = c_lo*SUMW, S = c_s*SUMW
    int degenerate;       // 1 if some coefficient is 0 (term vanishes by design)
};

static void fill_a(int8_t * a, const Regime & R, int seed) {
    if (R.const_a != INT32_MIN) { for (int i = 0; i < 32; ++i) a[i] = (int8_t) R.const_a; return; }
    switch (seed % 4) {
        case 0: for (int i = 0; i < 32; ++i) a[i] = (int8_t)(i % 16); break;              // all-lows
        case 1: for (int i = 0; i < 32; ++i) a[i] = (int8_t)(112 + i % 16); break;        // all-highs
        case 2: for (int i = 0; i < 32; ++i) a[i] = (int8_t)((i & 1) ? -(i % 64) : (i % 64)); break;
        default: for (int i = 0; i < 32; ++i) a[i] = (int8_t)(((i * 37 + seed * 11) % 255) - 128); break;
    }
}

static void run_regimes() {
    printf("\n== per-regime coefficient asserts ==\n");
    const Regime regimes[] = {
        // name          const_a   c_hi c_lo  c_s  degenerate
        { "const-0",          0,     0,   -8,   0, 1 },  // D_hi vanishes; D_lo = -8*SUMW
        { "const-127",      127,     7,    7, 127, 0 },
        { "const-128",     -128,    -8,   -8,-128, 0 },  // the two's-complement extreme
        { "const-8",          8,     0,    0,   8, 1 },  // hi=0, lo=8->stored 0: both dots vanish
        { "const-minus8",    -8,    -1,    0,  -8, 1 },  // lo = 8 -> stored 0
        { "const-1",          1,     0,   -7,   1, 1 },  // hi=0 (sign trap: -1>>4 = -1, 1>>4 = 0)
        { "const-minus1",    -1,    -1,    7,  -1, 0 },
        { "varying-lows", INT32_MIN, 0,    0,   0, 1 },  // coefficients unused (varying)
        { "varying-mixed",INT32_MIN, 0,    0,   0, 1 },
    };
    for (const Regime & R : regimes) {
        int bad = 0, vacuous = 0, checked = 0;
        for (int pat = 0; pat < 6; ++pat) {
            for (int seed = 0; seed < 8; ++seed) {
                const U4 w = w_pattern(pat, seed);
                const int32_t SUMW = sumw_signed(w);
                int8_t a[32]; fill_a(a, R, seed);
                const I8Result r = i8split_dot(w, a);
                const int32_t direct = direct_dot(w, a);
                const StockResult st = stock_dot(w, a);
                if (r.S != direct || st.S != direct) bad++;
                checked++;
                if (SUMW == 0) { vacuous++; continue; }   // coefficient asserts vacuous here
                if (R.const_a != INT32_MIN) {
                    if (r.D_hi != R.c_hi * SUMW) bad++;
                    if (r.D_lo != R.c_lo * SUMW) bad++;
                    if (r.S     != R.c_s  * SUMW) bad++;
                }
            }
        }
        char nm[96]; snprintf(nm, sizeof nm, "regime %s (coeffs + exactness)", R.name);
        // vacuity guard: at least one non-degenerate weight pack must have SUMW != 0
        // for the coefficient asserts to be meaningful.
        int saw_nonzero_sumw = 0;
        for (int pat = 0; pat < 6 && !saw_nonzero_sumw; ++pat)
            for (int seed = 0; seed < 8; ++seed)
                if (sumw_signed(w_pattern(pat, seed)) != 0) { saw_nonzero_sumw = 1; break; }
        CHECK(nm, bad == 0 && checked > 0 && (R.const_a == INT32_MIN || saw_nonzero_sumw));
    }
}

// Per-nibble-lane sign sweep: only value L nonzero, a_hi = h for all h in [-8,7].
// Declared: D_hi = w_tc[L]*h, D_lo = -8*SUMW (bias on every lane), S = w_tc[L]*16h.
static void run_lane_sweep() {
    int bad = 0, checked = 0;
    for (int L = 0; L < 32; ++L) {
        const U4 w = w_pattern(4, L + 1);
        const int32_t SUMW = sumw_signed(w);
        const uint8_t * q = (const uint8_t *) &w;
        const int codeL = (L < 16) ? nib_u(q[L], 1) : nib_u(q[L - 16], 0);
        const int32_t wL = codeL - 8;
        for (int h = -8; h <= 7; ++h) {
            int8_t a[32] = {}; a[L] = (int8_t)(16 * h);
            const I8Result r = i8split_dot(w, a);
            if (r.D_hi != wL * h) bad++;
            if (r.D_lo != -8 * SUMW) bad++;
            if (r.S != wL * 16 * h) bad++;
            if (r.S != direct_dot(w, a)) bad++;
            checked++;
        }
    }
    CHECK("per-nibble-lane sign sweep (32 lanes x h=-8..7)", bad == 0 && checked == 32 * 16);
}

// Pairing check: wrong pairing {2k,2k+1} must be DETECTED (differ from direct).
static void run_pairing() {
    int detected = 0, total = 0;
    for (int seed = 0; seed < 64; ++seed) {
        const U4 w = w_pattern(4, seed + 101);
        int8_t a[32]; for (int i = 0; i < 32; ++i) a[i] = (int8_t)(((i * 53 + seed * 29) % 255) - 128);
        const I8Result r = i8split_dot(w, a);
        if (r.S != direct_dot(w, a)) { printf("  correct pairing mismatch seed %d\n", seed); g_fail++; }
        // wrong-pairing activations against the same weights:
        const U4 w_tc = { w.x ^ 0x88888888u, w.y ^ 0x88888888u, w.z ^ 0x88888888u, w.w ^ 0x88888888u };
        const int32_t S_wrong = 16 * mmvq_dp8a_u4(w_tc, pack_hi_wrong(a))
                              + mmvq_dp8a_u4(w_tc, pack_lo_wrong(a)) + 8 * sumw_signed(w);
        total++;
        if (S_wrong != r.S) detected++;
    }
    printf("  wrong-pairing detection: %d/%d randomized blocks distinguished\n", detected, total);
    CHECK("pairing check: canonical {i,i+16} exact, wrong {2k,2k+1} detected",
          detected == total && total == 64);
}

// ---------------------------------------------------------------------------
// Adversarial scale-boundary regime: activation quantization at the int8 edge.
// Stock q8_1 (d = amax/127, a = roundf(x*id)) can NEVER produce -128 from real
// data (|x*id| <= 127 by construction) - the harness must exercise it anyway:
// the kernel consumes int8 and the two's-complement extreme must split exactly.
static void run_scale_boundary() {
    // (a) real-quantizer edge: x with a single dominant value -> a = +/-127.
    int edge_bad = 0, edge_checked = 0, saw_127 = 0;
    for (int seed = 0; seed < 256; ++seed) {
        float x[32];
        uint32_t s = seed * 2246822519u + 1;
        auto rnd = [&]() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return ((int32_t) s % 1000) / 500.0f; };
        for (int i = 0; i < 32; ++i) x[i] = rnd();
        x[seed % 32] = (seed & 1) ? 97.3f : -97.3f;   // dominant value -> |a| = 127
        BlockQ81 b; quantize_row_q8_1_ref(x, &b, 1);
        int mn = 127, mx = -128;
        for (int i = 0; i < 32; ++i) { mn = std::min(mn, (int) b.qs[i]); mx = std::max(mx, (int) b.qs[i]); }
        if (mn == -128) edge_bad++;                 // must be unreachable
        if (mx == 127 || mn == -127) saw_127++;
        edge_checked++;
    }
    CHECK("scale boundary: stock q8_1 never emits -128, reaches +/-127",
          edge_bad == 0 && edge_checked == 256 && saw_127 > 0);

    // (b) synthetic -128/+127 activations against adversarial weight packs:
    //     exactness must hold at the two's-complement extremes.
    int b2 = 0, n2 = 0;
    for (int pat = 0; pat < 6; ++pat) for (int seed = 0; seed < 8; ++seed) {
        const U4 w = w_pattern(pat, seed);
        int8_t a[32];
        for (int i = 0; i < 32; ++i) a[i] = (i & 1) ? -128 : 127;   // extreme mix
        I8Result r = i8split_dot(w, a);
        StockResult st = stock_dot(w, a);
        if (r.S != direct_dot(w, a) || st.S != direct_dot(w, a)) b2++;
        n2++;
    }
    CHECK("scale boundary: -128/+127 mix exact vs direct and stock", b2 == 0 && n2 == 48);
}

// ---------------------------------------------------------------------------
// Randomized fuzz through the REAL quantizers (Q4_0 weights, q8_1 activations):
// integer identity i8split == stock == direct on every block, then the fp32
// epilogue against a double-precision dequant reference over a full row.
static uint64_t g_rng = 0x9E3779B97F4A7C15ULL;
static inline uint64_t rnd64() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return g_rng; }
static inline float rndf() { return ((int32_t) (rnd64() >> 33) % 200000) / 1000.0f - 100.0f; }

static void run_fuzz_int(int nblocks) {
    int bad = 0;
    for (int t = 0; t < nblocks; ++t) {
        float xw[32], xa[32];
        for (int i = 0; i < 32; ++i) { xw[i] = rndf(); xa[i] = rndf(); }
        if (t % 7 == 0) xa[t % 32] = 0.0f;          // sprinkle exact zeros
        BlockQ40 w; quantize_row_q4_0_ref(xw, &w, 1);
        BlockQ81 b; quantize_row_q8_1_ref(xa, &b, 1);
        const I8Result r = i8split_dot(w.qs, b.qs);
        const StockResult st = stock_dot(w.qs, b.qs);
        const int32_t direct = direct_dot(w.qs, b.qs);
        if (r.S != direct || st.S != direct) { bad++; if (bad < 4) printf("  fuzz int mismatch at %d: S=%d stock=%d direct=%d\n", t, r.S, st.S, direct); }
    }
    char nm[80]; snprintf(nm, sizeof nm, "fuzz: i8split == stock == direct (%d blocks)", nblocks);
    CHECK(nm, bad == 0);
}

// fp32 epilogue: acc += d_w * (d_a * S) per block, f16 scales as stored.
// Reference: double dequant of the SAME quantized values (not the raw x -
// the claim is about the integer path, not the quantizer).
static void run_fuzz_fp32(int nrows, int nblocks_per_row) {
    double worst = 0.0;
    int bad = 0;
    for (int row = 0; row < nrows; ++row) {
        float acc = 0.0f;
        double ref = 0.0;
        for (int blk = 0; blk < nblocks_per_row; ++blk) {
            float xw[32], xa[32];
            for (int i = 0; i < 32; ++i) { xw[i] = rndf(); xa[i] = rndf(); }
            BlockQ40 w; quantize_row_q4_0_ref(xw, &w, 1);
            BlockQ81 b; quantize_row_q8_1_ref(xa, &b, 1);
            const I8Result r = i8split_dot(w.qs, b.qs);
            const float d_w = f16_to_f32(w.d), d_a = f16_to_f32(b.d);
            acc += d_w * (d_a * (float) r.S);          // documented fold
            const uint8_t * q = (const uint8_t *) &w.qs;
            for (int i = 0; i < 16; ++i) {
                ref += (double) d_w * (nib_u(q[i], 1) - 8) * ((double) d_a * b.qs[i]);
                ref += (double) d_w * (nib_u(q[i], 0) - 8) * ((double) d_a * b.qs[i + 16]);
            }
        }
        const double scale = std::max(1.0, std::fabs(ref));
        const double rel = std::fabs((double) acc - ref) / scale;
        worst = std::max(worst, rel);
        if (rel > 1e-3) bad++;
    }
    printf("  fp32 epilogue worst relative error: %.3e (bar 1e-3: fp32 fold-over-blocks class, same as stock)\n", worst);
    char nm[80]; snprintf(nm, sizeof nm, "fp32 epilogue within 1e-3 rel (%d rows x %d blocks)", nrows, nblocks_per_row);
    CHECK(nm, bad == 0);
}

// int32 headroom: per-block |S| <= 32*8*127 = 32512; the 16*D_hi term <= 32768.
static void run_overflow_bound() {
    int32_t worst = 0;
    for (int t = 0; t < 20000; ++t) {
        const U4 w = w_pattern(4 + (int) (rnd64() % 2), (int) rnd64());
        int8_t a[32];
        for (int i = 0; i < 32; ++i) a[i] = (int8_t) ((int) (rnd64() % 256) - 128);
        const I8Result r = i8split_dot(w, a);
        worst = std::max(worst, std::abs(r.S));
    }
    printf("  worst |S| observed: %d (bound 32512)\n", worst);
    CHECK("int32 headroom: |S| <= 32512 per 32-block", worst <= 32512);
}

int main() {
    printf("seal_i8split: exact-8-bit nibble-dot path for narrow-batch Q4_0\n");
    run_regimes();
    run_lane_sweep();
    run_pairing();
    run_scale_boundary();
    run_fuzz_int(200000);
    run_fuzz_fp32(2000, 160);      // 160 blocks = K=5120 row
    run_overflow_bound();
    printf("\n%d invariants, %d FAIL\n", g_pass + g_fail, g_fail);
    return g_fail ? 1 : 0;
}
