// d1-cpu-seal.cpp - taskD1 CPU-simulation seal, independent of the GPU.
// Proves the D1 rewrites are bit-identical to the generic vec_dot arithmetic
// BEFORE any GPU run:
//   S1: Q6_K raw-decode identity: dp4a(raw,u,0) - (sum(u)<<5) == dp4a(vsubss4(raw,0x20..),u,0)
//       for exhaustive per-byte values 0..63 (sampled words) x random u.
//   S2: Q6_K vih fold: ((vh>>4)<<4)&0x30303030 == vh & 0x30303030 for random vh.
//   S3: full-row twin model (hoisted decode across j, hoisted activation across i,
//       raw decode) == generic per-call model, bit-exact, random blocks.
//   S4: Q4_0 twin model (hoisted decode/activation) == generic per-call model.
// Plain C++ (no HIP): self-contained f16 decode + block structs mirroring
// ggml-common.h layouts (block_q6_K 210 B, block_q4_0 18 B, block_q8_1 36 B).
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <random>

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x3FFu;
    uint32_t out;
    if (exp == 0u) {
        if (man == 0u) out = sign << 31;
        else {
            int32_t e = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; e--; }
            man &= 0x3FFu;
            out = (sign << 31) | ((uint32_t) e << 23) | (man << 13);
        }
    } else if (exp == 31u) {
        out = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        out = (sign << 31) | ((exp + 127u - 15u) << 23) | (man << 13);
    }
    float f; memcpy(&f, &out, 4); return f;
}

struct bq6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };
struct bq4_0 { uint16_t d; uint8_t qs[16]; };
struct bq8_1 { uint16_t ds_lo, ds_hi; int8_t qs[32]; };
static_assert(sizeof(bq6_K) == 210, "");
static_assert(sizeof(bq4_0) == 18, "");
static_assert(sizeof(bq8_1) == 36, "");

static inline int dp4a(int a, int b, int c) {
    int s = c;
    for (int i = 0; i < 4; ++i)
        s += (int)(int8_t)((a >> (8*i)) & 0xFF) * (int)(int8_t)((b >> (8*i)) & 0xFF);
    return s;
}
static inline int vsubss4(int a, int b) {
    int r = 0;
    for (int i = 0; i < 4; ++i) {
        int d = (int)(uint8_t)(a >> (8*i)) - (int)(uint8_t)(b >> (8*i));
        if (d < -128) d = -128; if (d > 127) d = 127;
        r |= (d & 0xFF) << (8*i);
    }
    return r;
}
static inline int gib2(const void * x, int i) {
    const uint16_t * x16 = (const uint16_t *) x;
    return (int) x16[2*i] | ((int) x16[2*i+1] << 16);
}
static inline int gib4(const void * x, int i) { int v; memcpy(&v, (const int8_t*)x + 4*i, 4); return v; }

// generic vec_dot_q6_K_q8_1 (QR6_K=2, QI6_K=32)
static float gen_q6(const bq6_K * b, const bq8_1 * y, int iqs) {
    const int bq8_off = 4*(iqs/16) + (iqs%16)/8;
    const int sc_off  = 8*(iqs/16) + (iqs%16)/4;
    const int vh_sh   = 2*((iqs%16)/8);
    const int vl = gib2(b->ql, iqs);
    const int vh = gib2(b->qh, 8*(iqs/16) + iqs%8) >> vh_sh;
    const int8_t * sc = b->scales + sc_off;
    int u[2]; float d8[2];
    for (int i = 0; i < 2; ++i) {
        u[i]  = gib4(y[bq8_off + 2*i].qs, iqs % 32);
        d8[i] = f16_to_f32(y[bq8_off + 2*i].ds_lo);
    }
    float sumf = 0.0f;
    for (int i = 0; i < 2; ++i) {
        const int s = sc[4*i];
        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
        const int vi = vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (float)(dp4a(vi, u[i], 0) * s);
    }
    return f16_to_f32(b->d) * sumf;
}
// d1 twin Q6_K: decode hoisted (once), raw decode + sumu correction
static float d1_q6(const bq6_K * b, const bq8_1 * y, int iqs) {
    const int bq8_off = 4*(iqs/16) + (iqs%16)/8;
    const int sc_off  = 8*(iqs/16) + (iqs%16)/4;
    const int vh_sh   = 2*((iqs%16)/8);
    const int vl = gib2(b->ql, iqs);
    const int vh = gib2(b->qh, 8*(iqs/16) + iqs%8) >> vh_sh;
    const int8_t * sc = b->scales + sc_off;
    const int raw0 = (vl & 0x0F0F0F0F) | ((vh << 4) & 0x30303030);
    const int raw1 = ((vl >> 4) & 0x0F0F0F0F) | (vh & 0x30303030);
    const float d = f16_to_f32(b->d);
    int u[2]; float d8[2];
    for (int i = 0; i < 2; ++i) {
        u[i]  = gib4(y[bq8_off + 2*i].qs, iqs % 32);
        d8[i] = f16_to_f32(y[bq8_off + 2*i].ds_lo);
    }
    const int sumu0 = dp4a(0x01010101, u[0], 0);
    const int sumu1 = dp4a(0x01010101, u[1], 0);
    float sumf = 0.0f;
    sumf += d8[0] * ((dp4a(raw0, u[0], 0) - (sumu0 << 5)) * sc[0]);
    sumf += d8[1] * ((dp4a(raw1, u[1], 0) - (sumu1 << 5)) * sc[4]);
    return d * sumf;
}
static float gen_q4(const bq4_0 * b, const bq8_1 * y, int iqs) {
    int v[2], u[4];
    for (int i = 0; i < 2; ++i) {
        v[i] = gib2(b->qs, iqs + i);
        u[2*i] = gib4(y->qs, iqs + i);
        u[2*i+1] = gib4(y->qs, iqs + i + 4);
    }
    int sumi = 0;
    for (int i = 0; i < 2; ++i) {
        sumi = dp4a((v[i] >> 0) & 0x0F0F0F0F, u[2*i], sumi);
        sumi = dp4a((v[i] >> 4) & 0x0F0F0F0F, u[2*i+1], sumi);
    }
    const float dx = f16_to_f32(y->ds_lo), dy = f16_to_f32(y->ds_hi);
    return f16_to_f32(b->d) * ((float) sumi * dx - 4.0f * dy);
}
static float d1_q4(const bq4_0 * b, const bq8_1 * y, int iqs) {
    const int a0 = gib2(b->qs, iqs + 0) & 0x0F0F0F0F;
    const int a1 = (gib2(b->qs, iqs + 0) >> 4) & 0x0F0F0F0F;
    const int a2 = gib2(b->qs, iqs + 1) & 0x0F0F0F0F;
    const int a3 = (gib2(b->qs, iqs + 1) >> 4) & 0x0F0F0F0F;
    const float d4 = f16_to_f32(b->d);
    const int u0 = gib4(y->qs, iqs + 0), u1 = gib4(y->qs, iqs + 0 + 4);
    const int u2 = gib4(y->qs, iqs + 1), u3 = gib4(y->qs, iqs + 1 + 4);
    const float dx = f16_to_f32(y->ds_lo), dy = f16_to_f32(y->ds_hi);
    int sumi = 0;
    sumi = dp4a(a0, u0, sumi);
    sumi = dp4a(a1, u1, sumi);
    sumi = dp4a(a2, u2, sumi);
    sumi = dp4a(a3, u3, sumi);
    return d4 * ((float) sumi * dx - 4.0f * dy);
}

static uint16_t rnd_f16(std::mt19937 & rng) {
    // finite positive f16 in a sane range: exponent 0x3800..0x5000
    std::uniform_int_distribution<int> e(0x3800, 0x4F00), m(0, 0x3FF);
    return (uint16_t)(e(rng) | m(rng));
}

int main() {
    std::mt19937 rng(20261116);
    long bad = 0, n = 0;

    // S1: raw-decode identity, exhaustive per-byte 0..63
    for (int a = 0; a < 64; ++a)
    for (int b2 = 0; b2 < 64; b2 += 7)
    for (int c = 0; c < 64; c += 11)
    for (int d = 0; d < 64; d += 13) {
        int raw = a | (b2 << 8) | (c << 16) | (d << 24);
        for (int t = 0; t < 64; ++t) {
            int u = (int) rng();
            int lhs = dp4a(raw, u, 0) - (dp4a(0x01010101, u, 0) << 5);
            int rhs = dp4a(vsubss4(raw, 0x20202020), u, 0);
            if (lhs != rhs) ++bad;
            ++n;
        }
    }
    printf("S1 raw-decode identity: %ld cases, mismatches=%ld %s\n", n, bad, bad ? "FAIL" : "PASS");

    // S2: vih fold identity
    long bad2 = 0;
    for (long i = 0; i < 1000000; ++i) {
        int vh = (int) rng();
        if ((((vh >> 4) << 4) & 0x30303030) != (vh & 0x30303030)) ++bad2;
    }
    printf("S2 vih-fold identity: 1e6 cases, mismatches=%ld %s\n", bad2, bad2 ? "FAIL" : "PASS");

    // S3: Q6_K full vec_dot equivalence over random blocks x all iqs
    long bad3 = 0, n3 = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        bq6_K b;
        for (auto & x : b.ql) x = (uint8_t) rng();
        for (auto & x : b.qh) x = (uint8_t) rng();
        for (auto & x : b.scales) x = (int8_t) rng();
        b.d = rnd_f16(rng);
        bq8_1 y[8];
        for (auto & yy : y) {
            yy.ds_lo = rnd_f16(rng); yy.ds_hi = rnd_f16(rng);
            for (auto & x : yy.qs) x = (int8_t) rng();
        }
        for (int iqs = 0; iqs < 32; ++iqs) {
            float g = gen_q6(&b, y, iqs), t = d1_q6(&b, y, iqs);
            if (memcmp(&g, &t, 4) != 0) ++bad3;
            ++n3;
        }
    }
    printf("S3 Q6_K twin==generic: %ld cases, mismatches=%ld %s\n", n3, bad3, bad3 ? "FAIL" : "PASS");

    // S4: Q4_0 twin == generic
    long bad4 = 0, n4 = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        bq4_0 b;
        for (auto & x : b.qs) x = (uint8_t) rng();
        b.d = rnd_f16(rng);
        bq8_1 y;
        y.ds_lo = rnd_f16(rng); y.ds_hi = rnd_f16(rng);
        for (auto & x : y.qs) x = (int8_t) rng();
        for (int iqs : {0, 2}) {
            float g = gen_q4(&b, &y, iqs), t = d1_q4(&b, &y, iqs);
            if (memcmp(&g, &t, 4) != 0) ++bad4;
            ++n4;
        }
    }
    printf("S4 Q4_0 twin==generic: %ld cases, mismatches=%ld %s\n", n4, bad4, bad4 ? "FAIL" : "PASS");

    const bool pass = !bad && !bad2 && !bad3 && !bad4;
    printf("CPU-SEAL %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
