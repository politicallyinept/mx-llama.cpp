// SUMW precomputed plane (W4A4): repack-time weight-nibble-sum plane for the
// Q4_0 u4 (dp8a) weight path.
//
// WHY: the u4 epilogue computes, per (row, sub-block) and per kernel call:
//     sw = (float) ((int) nib_sum_u4(pk) - 256)
// (repack-kernels.cuh, documented at the nib_sum_u4 header). SUMW is a
// function of the WEIGHT pack only, so it is loop-invariant over tokens and
// hoists to repack time. mul-mat-id.cu blames exactly this recompute for the
// MoE single-token loss: "the SUMW recompute per block never amortizes at
// [single token]".
//
// STORAGE DESIGN: one extra plane in the Q4_0 repack record, AFTER the f16 d
// plane: int16 SUMW per 32-value block, row-major [ne1 x n_sub]. int16 because
// SUMW = sum(32 excess-8 nibble codes) - 256 spans [-256, +224] (9-bit signed);
// int8 would clip the adversarial extremes (all-0xF gives +224, all-zero -256).
// The SIGNED 32-bit semantics of the inline form are preserved exactly: the
// plane stores (int16_t) ((int) nib_sum - 256) and the reader sign-extends the
// int16 back to int before the float cast, so the epilogue value is bit-equal
// to the inline computation for every pack (seal invariant).
// Memory cost: +2 B per 32-value block = +11.1% of the repacked Q4_0 record
// (18 B -> 20 B per block; +0.5 bpw on top of Q4_0's 4.5 bpw). Allocated ONLY
// when the gate is on; gate-off keeps the 18 B/block record byte-for-byte.
//
// GATE: GGML_CUDA_W4A4_SUMW=1 (default OFF = bit-identical to today: the plane
// is not allocated, not written, and no kernel reads it). Process-wide static,
// same fork pattern as GGML_CUDA_REPACK_Q4_0 / GGML_CUDA_W4A4_DOT8.
//
// SCOPE: Q4_0 U4ACT only. The Q4_1 U4Q41 branches keep the inline computation
// (its 20 B record has no aligned 2 B slot and Q4_1 admission is already the
// narrow DOT8=4 gate; documented at the U4Q41 branches in repack-kernels.cuh).
//
// This header is deliberately free of ggml/HIP dependencies so the CPU seal
// (seal_sumw_plane.cc) compiles the EXACT converter and layout helpers below.
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>

// Host-side gate, defined in repack-common.cu (function-local static + WARN
// marker, fork pattern). True when GGML_CUDA_W4A4_SUMW is set to a nonzero int.
bool repack_sumw_plane_on();

// Canonical block_q4_0 / block_iq4_nl record: f16 d, then 16 nibble bytes
// (byte j = {value j low, value j+16 high}). Byte-identical to block_q4_0.
struct rp_sumw_block {
    uint16_t d;
    uint8_t  qs[16];
};
static_assert(sizeof(rp_sumw_block) == 18, "rp_sumw_block must be 18 bytes");

// The ONE definition of the repack-time SUMW, shared by the host converter and
// the CPU seal. SIGNED semantics exactly as the device epilogue requires: sum
// the 32 raw excess-8 nibble codes as unsigned, subtract 256 in SIGNED 32-bit
// (the milestone-3b corruption class was the unsigned wrap on sums < 256), and
// store as int16 (the value provably fits: [0, 480] - 256 = [-256, +224]).
inline int16_t rp_sumw_from_block(const rp_sumw_block * b) {
    uint32_t s = 0;
    for (int i = 0; i < 16; ++i) {
        s += (uint32_t) (b->qs[i] & 0xFu) + (uint32_t) (b->qs[i] >> 4);
    }
    return (int16_t) ((int) s - 256);
}

// Fill the SUMW plane for one (expert) matrix from its canonical Q4_0 blocks.
// sw_plane points at the plane start (after the d plane); n_blocks = ne0/32.
// Called by repack_q40_host() (sync upload path) ONLY when the gate is on.
inline void rp_sumw_fill_plane(const rp_sumw_block * blocks, uint8_t * sw_plane,
                               const int64_t ne1, const int64_t n_blocks) {
    const int64_t total = ne1 * n_blocks;
    for (int64_t i = 0; i < total; ++i) {
        const int16_t sw = rp_sumw_from_block(&blocks[i]);
        std::memcpy(sw_plane + (size_t) i * 2, &sw, 2);
    }
}

// Offset of the SUMW plane inside one Q4_0 repack record (gate on):
//   [ qs plane: ne1 x qs_str ] [ d plane: ne1 x n_sub x 2 ] [ sw plane: ... ]
// The d plane offset is the qs-plane size; the sw plane follows the d plane.
static inline size_t rp_sumw_plane_off(const int64_t ne0, const int64_t ne1, const int64_t qs_str) {
    return (size_t) ne1 * (size_t) qs_str + (size_t) ne1 * (size_t) (ne0 / 32) * 2;
}
