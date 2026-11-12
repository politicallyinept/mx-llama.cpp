// i8narrow: ruquant-style narrow-batch Q4_0 kernel with EXACT 8-bit activations.
// Design: docs/DESIGN-i8-narrow-kernel.md  CPU seal: seals/seal_i8split.cc
//
// STATUS (fb18241213 era): AUTHORED, NOT COMPILED - pause point. No llama.cpp
// build on this box while RAM is below the build gate; the math is CPU-sealed.
//
// The exact-8-bit identity this kernel implements (per 32-value block, int32):
//   a_hi = a >> 4 (arithmetic), a_lo = a & 0xF stored as (a_lo ^ 8)
//   D_hi = 4x chained v_dot8_i32_i4(w_tc, hi_plane)   w_tc = raw ^ 0x88888888
//   D_lo = 4x chained v_dot8_i32_i4(w_tc, lo_plane)
//   S    = 16*D_hi + D_lo + 8*SUMW  ==  sum_i (c_i - 8) * a_i   (EXACT)
// and sum_i (c_i - 8)*a_i is bit-equal to the stock q8_1 path's integer partial
// sum (ilo+ihi) - 8*SA, so there is NO quality cost (seal invariants 1-5).

#pragma once

#include "../common.cuh"          // ggml_cuda_dp8a / ggml_cuda_dp4a
#include "repack-common.cuh"      // rp_traits<GGML_TYPE_Q4_0>, planes, MMV max tokens

// ---------------------------------------------------------------------------
// SIZEOF_PROBE: pairs a sizeof static_assert with an alignof static_assert and
// exports both as constexprs, so a host seal harness can cross-check its mirror
// struct against the device TU byte-for-byte (the layout IS the interface).
// Usage: SIZEOF_PROBE(block_i8split, 36, 2);
#define SIZEOF_PROBE(type, want_size, want_align)                                    \
    static_assert(sizeof(type) == (want_size), "SIZEOF_PROBE size: " #type);         \
    static_assert(alignof(type) == (want_align), "SIZEOF_PROBE align: " #type);      \
    static constexpr uint32_t sizeof_probe_##type  = sizeof(type);                   \
    static constexpr uint32_t alignof_probe_##type = alignof(type)

// ---------------------------------------------------------------------------
// Activation staging block (host packer view). The DEVICE reads three SPLIT
// planes (hi 16 B, lo 16 B, d f16 per (token, 32-block)); this struct is the
// per-(token, block) reference layout the host quantizer and the seal share.
//
// hi: 32 signed nibbles a>>4, canonical Q4_0 pairing byte j = {val j low,
//     val j+16 high} (quantize_row_q4_0_ref, ggml-quants.c:137-146 - verified
//     at fb18241213; the WARNING comment at repack-common.cuh:206-211 is STALE
//     for the host ref - read the code, repack-common.cu:789-794).
// lo: 32 nibbles (a & 0xF) ^ 8  (dot8 sees the signed value a_lo - 8).
// d : the stock q8_1 scale, amax/127 stored f16 (quantize_row_q8_1_ref,
//     ggml-quants.c:302) - bit-identical int8 codes to the stock path.
struct block_i8split {
    uint8_t   hi[16];
    uint8_t   lo[16];
    ggml_half d;
    uint8_t   pad[2];       // explicit tail: 36 B, mirrors block_q8_1's size
};
static_assert(sizeof(block_i8split) == 36, "block_i8split must be 36 bytes");
static_assert(alignof(block_i8split) == 2, "block_i8split must be 2-byte aligned");
SIZEOF_PROBE(block_i8split, 36, 2);

// Split-plane geometry (device layout; token-major, width-independent - one
// layout serves every batch width 1..16). nkb = ne0/32 blocks per token row.
struct i8split_geom {
    uint32_t nkb;             // 32-value blocks per token row
    size_t   hi_bytes;        // hi plane: 16 B per block
    size_t   lo_bytes;        // lo plane: 16 B per block (offset = hi_bytes*count)
    size_t   d_bytes;         // d  plane: 2 B per block
    i8split_geom() : nkb(0), hi_bytes(0), lo_bytes(0), d_bytes(0) {}
    i8split_geom(uint32_t ne0, uint32_t count)
        : nkb(ne0 >> 5), hi_bytes((size_t) count * nkb * 16),
          lo_bytes(hi_bytes), d_bytes((size_t) count * nkb * 2) {}
};
static_assert(sizeof(i8split_geom) >= 24, "i8split_geom layout sanity");

// Wave-interleaved weight run offset (v2 knob, DESIGN §5.1): wave group of 64
// rows x one 32-value block = one contiguous 1 KB run, 16 B per lane.
static __host__ __device__ inline size_t i8narrow_w_interleaved_off(uint32_t row, uint32_t sb, uint32_t n_blocks) {
    return ((size_t) (row >> 6) * n_blocks + sb) * 1024 + (size_t) (row & 63) * 16;
}

// Host reference packer (fp32 row -> split planes), defined in w4a4-i8narrow.cu.
// Produces the SAME int8 codes as quantize_row_q8_1_ref, then the lossless
// nibble split. k % 32 == 0.
void quantize_row_i8split(const float * GGML_RESTRICT x, block_i8split * GGML_RESTRICT y, int64_t k);

// Device quantizer launcher (mirrors quantize_mmq_q8_1_u4_cuda's signature and
// the narrow-path alloc math of mul-mat.cu:357-387, three planes instead of one).
void quantize_mmq_i8split_cuda(const float * __restrict__ x, void * __restrict__ vhi,
        void * __restrict__ vlo, void * __restrict__ vd,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int ne1, int ne2, cudaStream_t stream);

// Launchers (defined in w4a4-i8narrow.cu). T = tokens (batch width), NWAVES =
// waves per block, KSPLIT = grid.z K segments (1 = no workgroup split),
// INTERLEAVED = wave-interleaved weight runs (§5.1 v2 knob).
void mul_mat_i8narrow_launch(int T, const uint8_t * wbase, const void * hi, const void * lo,
        const ggml_half * dplane, float * y, uint32_t ne0, uint32_t ne1,
        uint32_t xs, uint32_t ys, cudaStream_t stream);

// KSPLIT-aware overload. ksplit=1 == the overload above (single pass, no ws).
// ksplit in {2,4,8}: pass 1 grid=(ceil(ne1/64), 1, ksplit) writes per-workgroup
// fp32 partials to ksplit_ws[ksplit][ne1][T] (CALLER-ALLOCATED, must outlive the
// two launches), then a fixed-order second-pass reducer writes y. Deterministic:
// no fp32 atomics anywhere.
void mul_mat_i8narrow_launch(int T, const uint8_t * wbase, const void * hi, const void * lo,
        const ggml_half * dplane, float * y, uint32_t ne0, uint32_t ne1,
        uint32_t xs, uint32_t ys, cudaStream_t stream, int ksplit, float * ksplit_ws);
