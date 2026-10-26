// seal_nvfp4_identity.cc — CPU-only seal for the NVFP4 E2M1 x UE4M3 integer identity.
// gfx906 repack port, Phase 0. No GPU, no I/O beyond stdout.
//
// IDENTITY UNDER TEST (per 16-element sub-block j of a block_nvfp4: d[4] UE4M3 + qs[32] packed E2M1):
//   k_i   = kvalues_fp4[code_i]        int8, |k| <= 12, = 2 x E2M1 value  (exact integer)
//   D_j   = sum_i a_i * k_i            int8 x int8 -> int32 (dp8a/V_DOT8 class), EXACT:
//                                       |D| <= 16*127*12 = 24384  (int4 act: <= 16*8*12 = 1536)
//   s_j   = ue4m3(d_j)                 exact fp32; s == 0 OR s = M * 2^E with M <= 15 (4 sig bits).
//                                       NOTE: NOT powers of two in general (UE4M3 has a 3-bit
//                                       mantissa) — the "power-of-two scale" hint is FALSE;
//                                       exactness survives via the M<=15 bound instead.
//   term_j= (float)D_j * s_j * 0.5f    the /2 folds the k-doubling (D = 2*sum(a*e2m1)). EXACT in
//                                       fp32: |D| < 2^15 (15 sig bits) x M (4 sig bits) = 19 <= 24,
//                                       and *0.5 is a pure exponent shift. fp16 fold REJECTED (report).
//   out   = sum_j term_j               compared in double against an INDEPENDENT double reference
//                                       (per-element a_i * e2m1_ref * ue4m3_ref). EXACT equality.
//
// Declared-spec assertions (SEAL INPUT-VALIDATION RULE): realized coefficients are checked
// against the DECLARED spec per regime, not a liveness heuristic; vacuous-trial fraction reported.
//
// Build: g++ -O2 -std=c++17 -o seal_nvfp4_identity seal_nvfp4_identity.cc
// Run:   ./seal_nvfp4_identity [none|table-off1|skip-div2|sign-drop]
//   mutations deliberately break the formula to prove the seal is not vacuous.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <array>
#include <random>
#include <string>

enum Mut { MUT_NONE = 0, MUT_TABLE_OFF1 = 1, MUT_SKIP_DIV2 = 2, MUT_SIGN_DROP = 3 };
static int g_mut = MUT_NONE;
static long g_failures = 0;
static long g_trials = 0;

#define FAIL(...) do { g_failures++; if (g_failures <= 8) { printf("SEAL FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

// ---- DECLARED SPEC: decode table transcribed from upstream master ggml-common.h
//      "e2m1 values (doubled), shared by MXFP4 and NVFP4" (fetched 2026-10-21) ----
static const int8_t kvalues_fp4[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};

// ---- re-derived E2M1 decode: bit3=sign, bit2:1=exp (bias 1), bit0=mantissa ----
// subnormal (exp=0): val = m*0.5            -> doubled: m
// normal:            val = (1+m/2)*2^(e-1)  -> doubled: (2+m) << (e-1)
// (Q4_0 excess-8 XOR 0x88 does NOT apply: E2M1 codes are sign-magnitude FP, no integer bias.)
static int e2m1_doubled_derived(int code) {
    int s = (code >> 3) & 1, e = (code >> 1) & 3, m = code & 1;
    int mag = (e == 0) ? m : ((2 + m) << (e - 1));
    return s ? -mag : mag;
}

// ---- pure UE4M3 decode (matches ggml_ue4m3_to_fp32 semantics: 0x00 and 0x7F(NaN) -> 0, /2 fold) ----
static float ue4m3_pure(int x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int e = (x >> 3) & 0xF, m = x & 7;
    float raw = (e == 0) ? (float)m * (1.0f / 512.0f)          // m * 2^-9
                         : ldexpf((float)(8 + m), e - 10);     // (1+m/8) * 2^(e-7)
    return raw * 0.5f;                                          // the /2 fold
}

// ---- identity-path wrappers (mutations apply HERE only; global spec checks use the pure forms) ----
static int kdecode(int code) {
    int v = kvalues_fp4[code & 15];
    if (g_mut == MUT_TABLE_OFF1 && (code & 15) == 3) v = 4;   // off-by-one in decode table
    if (g_mut == MUT_SIGN_DROP && v < 0) v = -v;              // drop the sign bit
    return v;
}
static float sdecode(int d) { return ue4m3_pure(d); }

// ---- independent double reference (separate derivation, not the table) ----
static double e2m1_ref(int code) {
    int s = (code >> 3) & 1, e = (code >> 1) & 3, m = code & 1;
    double v = (e == 0) ? 0.5 * m : ldexp(1.0 + 0.5 * m, e - 1);
    return s ? -v : v;
}
static double ue4m3_ref(int x) {
    if (x == 0 || x == 0x7F) return 0.0;
    int e = (x >> 3) & 0xF, m = x & 7;
    double v = (e == 0) ? ldexp((double)m, -9) : ldexp(1.0 + (double)m / 8.0, e - 7);
    return v * 0.5;
}

// ---- regimes (DECLARED spec regimes tracked per trial for the vacuous-fraction report) ----
enum { R_ZERO_SCALE, R_UNDEF_SCALE, R_MAX_EXP_SCALE, R_SUB_SCALE, R_NONPOW2_SCALE,
       R_NEG_ZERO_W, R_ALL_SUBNIB, R_SIGN_EXTREMES, R_BIG_D, R_MAX_D, NREG };
static const char* R_NAME[NREG] = {
    "zero scale (d=0x00)", "undefined scale probe (0x7F/0xFF)", "max-exponent scale (exp=15)",
    "subnormal scale (exp=0)", "non-power-of-2 scale (mant!=0)", "negative-zero weight (code 0x8)",
    "all-subnormal nibbles", "sign extremes (+6 and -6 in block)", "|D| >= 20000", "|D| == 24384 (max)"};
static std::array<long, NREG> g_hits{};

static double g_max_f32acc_rel = 0.0;   // informational: fp32 accumulation vs double (NOT part of the identity claim)

static void trial(const uint8_t d[4], const uint8_t qs[32], const int8_t a[64]) {
    g_trials++;
    std::array<bool, NREG> hit{};
    double ref = 0.0, acc = 0.0;
    float accf = 0.0f;
    bool all_subnib = true;
    bool has_pos6 = false, has_neg6 = false;

    for (int j = 0; j < 4; j++) {
        int32_t D = 0;
        for (int i = 0; i < 16; i++) {
            int c = (i & 1) ? (qs[8 * j + i / 2] >> 4) : (qs[8 * j + i / 2] & 0xF);  // low nibble first
            int k = kdecode(c);
            D += (int32_t)a[16 * j + i] * k;
            ref += (double)a[16 * j + i] * e2m1_ref(c) * ue4m3_ref(d[j]);
            // declared-spec assertion: decoded coefficients are integers in [-12, 12]
            if (k < -12 || k > 12) FAIL("trial %ld: decoded k=%d out of [-12,12]", g_trials, k);
            if (c == 0x8) hit[R_NEG_ZERO_W] = true;
            if (((c >> 1) & 3) != 0) all_subnib = false;
            if (c == 0x7) has_pos6 = true;
            if (c == 0xF) has_neg6 = true;
        }
        if (D < -24384 || D > 24384) FAIL("trial %ld: |D|=%d exceeds declared bound 24384", g_trials, D);
        if (D >= 20000 || D <= -20000) hit[R_BIG_D] = true;
        if (D == 24384 || D == -24384) hit[R_MAX_D] = true;

        // declared-spec assertion on the realized scale. DECLARED DOMAIN: valid UE4M3 bytes are
        // 0x00..0x7E (bit7 unused, 0x7F=NaN excluded; encoder clamps to 0x7E). 0x7F/0xFF are
        // undefined-encoding PROBES: the tree maps 0x7F->0 but 0xFF->240 (bit7 shifted out) —
        // an inconsistency the port must handle explicitly. Spec: s==0 OR s*2^10 integer <=229376
        // AND <=4 significant bits.
        double sd = ue4m3_ref(d[j]);
        if (d[j] >= 0x7F) {
            hit[R_UNDEF_SCALE] = true;
            float expect = (float)((d[j] == 0x7F) ? 0.0 : ue4m3_ref(d[j]));  // tree semantics via the independent reference: NaN check catches only 0x7F; bit7 otherwise shifted out
            if (sdecode(d[j]) != expect)
                FAIL("trial %ld: undefined scale 0x%02X decoded %.9g, tree semantics say %.9g",
                     g_trials, d[j], sdecode(d[j]), expect);
        }
        if (sd == 0.0) {
            if (d[j] == 0) hit[R_ZERO_SCALE] = true;
        } else {
            double q = sd * 1024.0;
            if (q != floor(q) || (d[j] <= 0x7E && q > 229376.0))
                FAIL("trial %ld: scale d=0x%02X violates spec (s*2^10=%g)", g_trials, d[j], q);
            int ex; double fr = frexp(sd, &ex);
            if (fr * 16.0 != floor(fr * 16.0))
                FAIL("trial %ld: scale d=0x%02X has >4 significant bits", g_trials, d[j]);
        }
        if ((((d[j] >> 3) & 0xF) == 15) && d[j] <= 0x7E) hit[R_MAX_EXP_SCALE] = true;
        if (d[j] >= 0x01 && d[j] <= 0x07) hit[R_SUB_SCALE] = true;
        if ((d[j] & 0x7) != 0 && d[j] != 0x7F) hit[R_NONPOW2_SCALE] = true;
        if (has_pos6 && has_neg6) hit[R_SIGN_EXTREMES] = true;

        float s = sdecode(d[j]);
        float term = (float)D * s;
        if (g_mut != MUT_SKIP_DIV2) term *= 0.5f;            // the /2 fold (mutation skips it)
        double td = (double)D * sd * 0.5;
        // load-bearing exactness assertion: the fp32 product equals the exact double product bit-for-bit
        if ((double)term != td)
            FAIL("trial %ld sub-block %d: fp32 term not exact: D=%d s=%.17g -> got %.17g want %.17g",
                 g_trials, j, D, sd, (double)term, td);
        if ((d[j] == 0 || d[j] == 0x7F) && term != 0.0f)
            FAIL("trial %ld sub-block %d: zero/NaN scale produced nonzero term %g", g_trials, j, term);
        acc += (double)term;
        accf += term;
    }
    if (all_subnib) hit[R_ALL_SUBNIB] = true;

    // THE seal: exact equality against the independent double reference
    if (acc != ref)
        FAIL("trial %ld: identity sum %.17g != double reference %.17g (delta %.6g)", g_trials, acc, ref, acc - ref);

    // informational: fp32 accumulation error (ordinary accumulation class, not the identity claim)
    if (fabs(ref) >= 1.0) {
        double rel = fabs((double)accf - ref) / fabs(ref);
        if (rel > g_max_f32acc_rel) g_max_f32acc_rel = rel;
    }
    for (int r = 0; r < NREG; r++) if (hit[r]) g_hits[r]++;
}

static void pack(std::mt19937& rng, uint8_t qs[32], const int codes[64]) {
    memset(qs, 0, 32);
    for (int i = 0; i < 64; i++) {
        uint8_t c = (uint8_t)(codes[i] & 0xF);
        if (i & 1) qs[i / 2] |= (uint8_t)(c << 4); else qs[i / 2] = c;
    }
}

int main(int argc, char** argv) {
    if (argc > 1) {
        std::string m = argv[1];
        if (m == "none") g_mut = MUT_NONE;
        else if (m == "table-off1") g_mut = MUT_TABLE_OFF1;
        else if (m == "skip-div2") g_mut = MUT_SKIP_DIV2;
        else if (m == "sign-drop") g_mut = MUT_SIGN_DROP;
        else { printf("usage: %s [none|table-off1|skip-div2|sign-drop]\n", argv[0]); return 2; }
    }
    printf("=== NVFP4 E2M1 x UE4M3 integer-identity seal (mutation=%s) ===\n",
           g_mut == MUT_NONE ? "none" : argv[1]);

    // ---- global declared-spec checks (pure forms; mutations must fail at the dot level) ----
    for (int c = 0; c < 16; c++)
        if ((int)kvalues_fp4[c] != e2m1_doubled_derived(c))
            FAIL("decode table entry %d = %d != re-derived %d", c, kvalues_fp4[c], e2m1_doubled_derived(c));
    for (int x = 0; x < 256; x++) {
        float got = ue4m3_pure(x);
        float want = (float)ue4m3_ref(x);
        if (memcmp(&got, &want, 4) != 0)
            FAIL("ue4m3(0x%02X) = %.9g != reference %.9g", x, got, want);
    }
    printf("[global] E2M1 table == re-derived formula for all 16 codes: %s\n",
           g_failures ? "FAIL" : "PASS");
    printf("[global] UE4M3 decode == independent reference for all 256 bytes: %s\n",
           g_failures ? "FAIL" : "PASS");
    // counterexample to the "scales are powers of two" hint:
    printf("[global] NOTE: hint 'scales are powers of two' is FALSE: d=0x0B -> %.10g = 11*2^-10 (non-pow2)\n",
           ue4m3_pure(0x0B));
    printf("[global] NOTE: undefined encodings d>=0x7F: bit7 is shifted out, so 0x80..0xFE ALIAS their low-7-bit "
           "value (0x81->%.9g) and 0xFF->%.9g, while 0x7F->%.9g via the NaN check — same 7-bit pattern as 0xFF, "
           "different result. Encoder clamps to 0x7E so it never emits d>=0x7F; the port must reject or normalize.\n",
           ue4m3_pure(0x81), ue4m3_pure(0xFF), ue4m3_pure(0x7F));

    // ---- ~1000 random blocks ----
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dc(0, 15), ds(0, 0x7E), da(-127, 127);  // scales drawn from the DECLARED domain 0x00..0x7E
    for (int t = 0; t < 1000; t++) {
        uint8_t d[4], qs[32]; int8_t a[64]; int codes[64];
        for (int i = 0; i < 4; i++) d[i] = (uint8_t)ds(rng);
        for (int i = 0; i < 64; i++) codes[i] = dc(rng);
        for (int i = 0; i < 64; i++) a[i] = (int8_t)da(rng);
        pack(rng, qs, codes);
        trial(d, qs, a);
    }
    printf("[random] 1000 random blocks done (failures so far: %ld)\n", g_failures);

    // ---- structurally adversarial blocks ----
    uint8_t qs[32]; int8_t a[64]; int codes[64];
    auto fill_a = [&](int v) { for (int i = 0; i < 64; i++) a[i] = (int8_t)(v == 99 ? ((i & 1) ? 127 : -127) : v); };
    auto fill_c = [&](int v) { for (int i = 0; i < 64; i++) codes[i] = v; };

    struct Adv { const char* name; uint8_t d[4]; };
    Adv adv[] = {
        {"zero scale",        {0x00, 0x00, 0x00, 0x00}},
        {"NaN scale 0x7F",    {0x7F, 0x7F, 0x7F, 0x7F}},
        {"undef scale 0xFF",  {0xFF, 0xFF, 0xFF, 0xFF}},
        {"max-exp scale 0x7E",{0x7E, 0x7E, 0x7E, 0x7E}},
        {"subnormal scales",  {0x01, 0x02, 0x03, 0x07}},
        {"mixed exp span",    {0x01, 0x08, 0x40, 0x7E}},
        {"non-pow2 scales",   {0x0B, 0x15, 0x2A, 0x55}},
    };
    for (auto& A : adv) {
        for (int pat = 0; pat < 3; pat++) {
            if (pat == 0) { fill_c(0x7); fill_a(127); }
            else if (pat == 1) { fill_c(0xF); fill_a(-127); }
            else { for (int i = 0; i < 64; i++) codes[i] = (i & 1) ? 0xF : 0x7; fill_a(99); }
            pack(rng, qs, codes);
            trial(A.d, qs, a);
        }
    }
    // negative-zero weights (code 0x8) with random activations
    fill_c(0x8); fill_a(0);
    { std::uniform_int_distribution<int> da2(-127, 127); for (int i = 0; i < 64; i++) a[i] = (int8_t)da2(rng); }
    pack(rng, qs, codes); { uint8_t d[4] = {0x38, 0x38, 0x38, 0x38 }; trial(d, qs, a); }
    // all-subnormal nibbles (codes 0,1,8,9) with min subnormal scale
    { uint8_t d[4] = {0x01, 0x01, 0x01, 0x01};
      for (int i = 0; i < 64; i++) codes[i] = (i % 4); pack(rng, qs, codes);
      for (int i = 0; i < 64; i++) a[i] = (int8_t)da(rng); trial(d, qs, a); }
    // max-D x max-scale worst case: k=12, a=127 -> D=24384, s=224
    { uint8_t d[4] = {0x7E, 0x7E, 0x7E, 0x7E}; fill_c(0x7); fill_a(127); pack(rng, qs, codes); trial(d, qs, a); }
    printf("[adversarial] named adversarial blocks done (failures so far: %ld)\n", g_failures);

    // ---- systematic sweeps: every scale byte at max D; every code at a fixed scale ----
    fill_c(0x7); fill_a(127); pack(rng, qs, codes);
    for (int x = 0; x < 256; x++) { uint8_t d[4] = {(uint8_t)x, (uint8_t)x, (uint8_t)x, (uint8_t)x}; trial(d, qs, a); }
    { uint8_t d[4] = {0x38, 0x38, 0x38, 0x38}; fill_a(99);
      for (int c = 0; c < 16; c++) { fill_c(c); pack(rng, qs, codes); trial(d, qs, a); } }
    printf("[sweeps] 256 scale-byte + 16 code sweeps done\n");

    // ---- report ----
    printf("\ntrials=%ld  failures=%ld\n", g_trials, g_failures);
    printf("\nregime coverage (vacuous-trial fraction = fraction of trials that never exercised the regime):\n");
    double worst = 0.0;
    for (int r = 0; r < NREG; r++) {
        double vac = 1.0 - (double)g_hits[r] / (double)g_trials;
        if (vac > worst) worst = vac;
        printf("  %-38s hits=%5ld  vacuous=%.4f\n", R_NAME[r], g_hits[r], vac);
    }
    printf("WORST vacuous-trial fraction across regimes: %.4f\n", worst);
    printf("int32 margin: |D|max=24384 (int8 act) / 1536 (int4 act) vs 2^31-1 -> %.0fx / %.0fx\n",
           2147483647.0 / 24384.0, 2147483647.0 / 1536.0);
    printf("fp32 product bits: 15 (D) + 4 (scale M) = 19 <= 24 -> exact. fp16 fold REJECTED: D needs 15 bits > 11;\n"
           "  even int4-act D (11 bits) x 4-bit scale = 15 > 11, and 1536*224=344064 > fp16 max 65504.\n");
    printf("informational fp32-accumulation max rel delta vs double: %.3g (ordinary accumulation class, not the identity)\n",
           g_max_f32acc_rel);
    if (g_failures == 0) printf("\nSEAL PASS: identity exact vs double reference on all %ld trials.\n", g_trials);
    else printf("\nSEAL FAIL: %ld failures.\n", g_failures);
    return g_failures ? 1 : 0;
}
