// k3-stagecheck.cpp - CPU simulation of rp_q6k_nc_lds staging + accumulation.
// Mirrors the kernel's LDS copy loops and kbx_loc indexing with plain host arrays
// and compares bit-exactly against the generic-path CPU oracle. Catches staging
// layout bugs (word-count, block placement, indexing) without a GPU.
// Build: g++ -O2 -std=c++17 -ffp-contract=off -o k3-stagecheck k3-stagecheck.cpp
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

typedef uint8_t uint8; typedef int8_t int8; typedef int16_t int16; typedef uint16_t uint16;
typedef uint32_t uint32; typedef int32_t int32;
struct ggml_half { uint16 v; };
struct ggml_half2 { ggml_half lo, hi; };
#define QK_K 256
#define QK8_1 32
#define QR6_K 2
#define QI6_K (QK_K / (4*QR6_K))
#define QI8_1 (QK8_1 / 4)  // = 8 (QK8_1/(4*QR8_1), QR8_1=1)
struct block_q6_K { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; ggml_half d; };
struct block_q8_1 { ggml_half2 ds; int8 qs[32]; };
static_assert(sizeof(block_q6_K) == 210, "q6_K size");
static_assert(sizeof(block_q8_1) == 36, "q8_1 size");
static float half2f(ggml_half h) { uint32 bits = ((uint32)h.v) << 16; float f; memcpy(&f, &bits, 4);
  // proper half->float
  uint32 sign = (h.v >> 15) & 1; int e = (h.v >> 10) & 0x1F; uint32 m = h.v & 0x3FF;
  if (e == 0) { if (!m) return sign ? -0.0f : 0.0f; float r = (float)m / 1024.0f * std::ldexp(1.0f, -14); return sign ? -r : r; }
  if (e == 31) { float r = m ? NAN : INFINITY; return sign ? -r : r; }
  uint32 out = (sign << 31) | ((e + 112) << 23) | (m << 13); float f2; memcpy(&f2, &out, 4); return f2; }
static ggml_half f2half(float f) { // round-to-nearest-even, finite inputs only
  uint32 b; memcpy(&b, &f, 4); uint32 sign = (b >> 16) & 0x8000; int32 e = ((b >> 23) & 0xFF) - 127 + 15;
  uint32 m = b & 0x7FFFFF; if (e >= 31) { ggml_half h{(uint16)(sign | 0x7C00)}; return h; }
  if (e <= 0) { ggml_half h{0}; return h; }
  uint32 lsb = (m >> 13) & 1; m += 0xFFF + ((m >> 13) & 1); m >>= 13; if (m > 0x3FF) { m = 0; e++; if (e >= 31) { ggml_half h{(uint16)(sign | 0x7C00)}; return h; } }
  ggml_half h{(uint16)(sign | (e << 10) | m)}; return h; }
static float lo2f(ggml_half2 ds) { return half2f(ds.lo); } // __low2float of ds == low half
static ggml_half2 fl2h(float a, float b) { ggml_half2 h{f2half(a), f2half(b)}; return h; }

static inline int dp4a_cpu(int a, int b, int c) { int s = c;
  for (int i = 0; i < 4; ++i) s += (int)(int8_t)((a >> (8*i)) & 0xFF) * (int)(int8_t)((b >> (8*i)) & 0xFF);
  return s; }
static inline int get_int_b2_cpu(const void * x, int i32) { const uint16 * x16 = (const uint16 *) x;
  return (int) x16[2*i32] | ((int) x16[2*i32+1] << 16); }
static inline int get_int_b4_cpu(const void * x, int i32) { int v; memcpy(&v, (const int8_t*)x + 4*i32, 4); return v; }
static inline int vsubss4_cpu(int a, int b) { int r = 0;
  for (int i = 0; i < 4; ++i) { int d = (int)(uint8)(a >> (8*i)) - (int)(uint8)(b >> (8*i));
    if (d < -128) d = -128; if (d > 127) d = 127; r |= (d & 0xFF) << (8*i); } return r; }
static inline float vec_dot_q6_K_q8_1_cpu(const block_q6_K * bq6_K, const block_q8_1 * bq8_1, int iqs) {
  const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
  const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
  const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));
  const int vl = get_int_b2_cpu(bq6_K->ql, iqs);
  const int vh = get_int_b2_cpu(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;
  const int8_t * scales = bq6_K->scales + scale_offset;
  int u[QR6_K]; float d8[QR6_K];
  for (int i = 0; i < QR6_K; ++i) { u[i] = get_int_b4_cpu(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1); d8[i] = lo2f(bq8_1[bq8_offset + 2*i].ds); }
  float sumf = 0.0f;
  for (int i = 0; i < QR6_K; ++i) { const int sc = scales[4*i];
    const int vil = (vl >> (4*i)) & 0x0F0F0F0F; const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
    const int vi = vsubss4_cpu((vil | vih), 0x20202020);
    sumf += d8[i] * (float)(dp4a_cpu(vi, u[i], 0) * sc); }
  return half2f(bq6_K->d) * sumf; }

static float q6k_generic_oracle(uint32 row, uint32 col, uint32 n_blocks,
                                const block_q6_K * w, uint32 srx, const block_q8_1 * xq, uint32 xs) {
  float v[64];
  for (int lane = 0; lane < 64; ++lane) { float acc = 0.0f; const int kqs = lane & 31;
    for (int kbx = lane >> 5; kbx < (int) n_blocks; kbx += 2)
      acc += vec_dot_q6_K_q8_1_cpu(w + (size_t) row * srx + kbx, xq + (size_t) col * xs + (size_t) kbx * 8, kqs);
    v[lane] = acc; }
  for (int off : {32, 16, 8, 4, 2, 1}) { float nv[64];
    for (int i = 0; i < 64; ++i) nv[i] = v[i] + v[i ^ off]; memcpy(v, nv, sizeof(v)); }
  return v[0]; }

// ---- CPU simulation of rp_q6k_nc_lds<T,W,STEP=4,STAGE_W=true,RPL=2> for one row pair ----
// (per-warp rows are independent; act staging is a word-copy bijection)
// Proper per-lane sim:
static float q6k_lds_sim2(uint32 row, uint32 col, uint32 n_blocks,
                          const block_q6_K * w, uint32 srx, const block_q8_1 * xq, uint32 xs, int r_out) {
  const int STEP = 4, RPL = 2, BS = 210;
  std::vector<uint32> sw(RPL * (STEP * BS / 4));
  std::vector<uint32> sq((size_t) STEP * 8 * 9);
  const uint * wsrc = (const uint *) w;
  const uint * xsrc = (const uint *) xq;
  float v[64];
  for (int lane = 0; lane < 64; ++lane) {
    float acc = 0.0f; const int kqs = lane & 31, t32 = lane >> 5;
    for (uint32 s0 = 0; s0 < n_blocks; s0 += STEP) {
      const uint32 nb_step = std::min((uint32) STEP, n_blocks - s0);
      const uint32 nab = nb_step * 8;
      // act staging for this token
      { const uint * g = xsrc + ((size_t) col * xs + (size_t) s0 * 8) * 9;
        for (uint32 wi = 0; wi < nab * 9; ++wi) sq[wi] = g[wi]; }
      // weight staging (byte-addressed like the kernel: 210 B blocks, word-exact only at even block offsets)
      const uint32 nw = (nb_step * BS) / 4;
      for (int r = 0; r < RPL; ++r) {
        const uint32 rr = row + r;
        const uint * g = (const uint *) ((const uint8 *) wsrc + ((size_t) rr * srx + s0) * BS);
        uint * lds = &sw[(size_t) r * (STEP * BS / 4)];
        for (uint32 wi = 0; wi < nw; ++wi) lds[wi] = g[wi];
      }
      for (uint32 it = 0; it < nb_step; it += 2) {
        const int kbx_loc = (int) it + t32;
        const block_q8_1 * yb = (const block_q8_1 *) &sq[kbx_loc * 72];  // 8 q8_1 blocks (9 words each) per Q6_K block
        const block_q6_K * staged = (const block_q6_K *) &sw[(size_t) r_out * (STEP * BS / 4)];
        acc += vec_dot_q6_K_q8_1_cpu(staged + kbx_loc, yb, kqs);
      }
    }
    v[lane] = acc;
  }
  for (int off : {32, 16, 8, 4, 2, 1}) { float nv[64];
    for (int i = 0; i < 64; ++i) nv[i] = v[i] + v[i ^ off]; memcpy(v, nv, sizeof(v)); }
  return v[0];
}

int main() {
  const uint32 K = 5120, N = 64, srx = K / 256, xs = K / 32, T = 8;
  std::vector<block_q6_K> w((size_t) N * srx);
  std::vector<block_q8_1> xq((size_t) T * xs);
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> byte_d(0, 255);
  std::uniform_real_distribution<float> dscale(0.001f, 0.5f), act_d(0.001f, 2.0f);
  for (auto & b : w) { for (int j = 0; j < 128; ++j) b.ql[j] = byte_d(rng);
    for (int j = 0; j < 64; ++j) b.qh[j] = byte_d(rng);
    for (int j = 0; j < 16; ++j) b.scales[j] = (int8_t) byte_d(rng);
    b.d = f2half(dscale(rng)); }
  for (auto & b : xq) { b.ds = fl2h(act_d(rng), 0); for (int j = 0; j < 32; ++j) b.qs[j] = (int8_t) byte_d(rng); }
  long bad = 0, n = 0;
  for (uint32 r = 0; r < N; ++r) for (uint32 c = 0; c < T; ++c) {
    const float o = q6k_generic_oracle(r, c, srx, w.data(), srx, xq.data(), xs);
    const float s = q6k_lds_sim2(r, c, srx, w.data(), srx, xq.data(), xs, 0);
    ++n; if (memcmp(&o, &s, 4) != 0) ++bad; }
  // second row of each staged pair (r_out=1): sim stages row and row+1, reads row+1
  for (uint32 r = 0; r + 1 < N; r += 2) for (uint32 c = 0; c < T; ++c) {
    const float o = q6k_generic_oracle(r + 1, c, srx, w.data(), srx, xq.data(), xs);
    const float s = q6k_lds_sim2(r, c, srx, w.data(), srx, xq.data(), xs, 1);
    ++n; if (memcmp(&o, &s, 4) != 0) ++bad; }
  printf("stagecheck: %ld comparisons, mismatches=%ld %s\n", n, bad, bad ? "FAIL" : "PASS");
  return bad != 0;
}
