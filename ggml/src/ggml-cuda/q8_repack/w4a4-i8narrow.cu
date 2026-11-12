// i8narrow kernel + activation quantizer. STATUS: AUTHORED, NOT COMPILED
// (build pause point - see docs/DESIGN-i8-narrow-kernel.md §10 and taskC-LEDGER.md).
// Design + seal references: docs/DESIGN-i8-narrow-kernel.md, seals/seal_i8split.cc.

#include "w4a4-i8narrow.cuh"
#include "../quantize.cuh"   // CUDA_QUANTIZE_BLOCK_SIZE_MMQ (quantizer grid math)

#include <cassert>

// ---------------------------------------------------------------------------
// Host reference packer: fp32 -> block_i8split planes. Same int8 codes as
// quantize_row_q8_1_ref (ggml-quants.c:302: d = amax/127 f16, a = roundf(x*id)),
// then the lossless split a_hi = a>>4 (arithmetic), lo stored (a&0xF)^8.
void quantize_row_i8split(const float * GGML_RESTRICT x, block_i8split * GGML_RESTRICT y, int64_t k) {
    assert(k % 32 == 0);
    const int nb = (int) (k / 32);
    for (int ib = 0; ib < nb; ++ib) {
        const float * xb = x + ib * 32;
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(xb[j]));
        const float d  = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        y[ib].d = (ggml_half) d;
        // canonical Q4_0 pairing: byte j = {value j low nibble, value j+16 high}
        for (int j = 0; j < 16; ++j) {
            const int8_t a0 = (int8_t) roundf(xb[j]      * id);
            const int8_t a1 = (int8_t) roundf(xb[j + 16] * id);
            y[ib].hi[j] = (uint8_t)(((a0 >> 4) & 0xF) | (((a1 >> 4) & 0xF) << 4));
            y[ib].lo[j] = (uint8_t)(((a0 & 0xF) ^ 8)  | ((((a1 & 0xF) ^ 8)) << 4));
        }
    }
}


// SIGNED weight sum of one raw Q4_0 qs uint4: (int) nib_sum_u4 - 256, cast to
// int BEFORE subtracting (the milestone-3b unsigned-wrap class, see the warning
// at repack-kernels.cuh:2247-2251). Local copy of the semantics of
// nib_sum_u4 (repack-kernels.cuh:78-88) so this TU stays self-contained.
static __device__ __forceinline__ int i8n_sumw_signed(const uint4 v) {
    const uint32_t ws[4] = { v.x, v.y, v.z, v.w };
    uint32_t s = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t x = ws[i];
        s += (x & 0xFu) + ((x >> 4) & 0xFu) + ((x >> 8) & 0xFu) + ((x >> 12) & 0xFu)
           + ((x >> 16) & 0xFu) + ((x >> 20) & 0xFu) + ((x >> 24) & 0xFu) + ((x >> 28) & 0xFu);
    }
    return (int) s - 256;
}

// ---------------------------------------------------------------------------
// The narrow kernel. One lane per output row (ruquant w4a4_skinny shape).
//   T        tokens (batch width 1..16), one layout for every width
//   NWAVES   waves per block (4 default; KSPLIT>1: waves split K, LDS reduce)
//   KSPLIT   grid.z workgroup K segments (1 = none)
//   INTERLEAVED  wave-interleaved 1 KB weight runs per step (v2 knob)
// Weight planes: the existing repacked Q4_0 qs/d/SUMW planes (rp_traits<Q4_0>).
// Register bar: <= 64 VGPR at T <= 8, ZERO spills (census cliff, w4a8-census.md:41).
template <int T, int NWAVES, int KSPLIT, bool INTERLEAVED>
static __global__ void __launch_bounds__(NWAVES * 64) mul_mat_i8narrow_q40(
        const uint8_t * __restrict__ wbase,   // repacked qs plane (raw canonical)
        const uint8_t * __restrict__ hi,      // activation hi plane, 16 B / (tok, blk)
        const uint8_t * __restrict__ lo,      // activation lo plane, 16 B / (tok, blk)
        const ggml_half * __restrict__ ad,    // activation scale plane, f16 / (tok, blk)
        float * __restrict__ y,
        const uint32_t ne0, const uint32_t ne1,
        const uint32_t xs, const uint32_t ys,   // xs = blocks per token column (nkb)
        float * __restrict__ ksplit_ws) {       // KSPLIT partial workspace (or null)
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    const typename rp_traits<GGML_TYPE_Q4_0>::geom wg(ne0, ne1);
    const uint32_t n_blocks = ne0 >> 5;

    const int wave  = threadIdx.x >> 6;
    const int lane  = threadIdx.x & 63;
    // Row ownership (DESIGN §6):
    //  KSPLIT == 1: each wave owns its own 64-row group; one lane per row.
    //  KSPLIT >  1: ALL waves cooperate on the SAME 64 rows, splitting the
    //               block's K segment across waves (reduced via LDS below).
    const uint32_t row0 = (KSPLIT == 1)
        ? (uint32_t) blockIdx.x * (NWAVES * 64) + wave * 64
        : (uint32_t) blockIdx.x * 64;
    const uint32_t row  = row0 + lane;
    const bool     rvalid = row < ne1;

    // K segmentation: grid.z picks a contiguous segment (workgroup level); with
    // KSPLIT>1 the block's segment is further split per wave (wave level).
    // Rotation by an odd constant (7) spreads HBM channel pressure (DESIGN §7).
    const uint32_t seg = (n_blocks + KSPLIT - 1) / KSPLIT;
    uint32_t k_lo = blockIdx.z * seg;
    uint32_t k_hi = min(k_lo + seg, n_blocks);
    if constexpr (KSPLIT > 1) {
        const uint32_t wseg = (k_hi - k_lo + NWAVES - 1) / NWAVES;
        const uint32_t wlo  = k_lo + wave * wseg;
        k_lo = min(wlo, k_hi);
        k_hi = min(wlo + wseg, k_hi);
    }

    float acc[T] = {};
    if (rvalid) {
        for (uint32_t sb = k_lo; sb < k_hi; ++sb) {
            // 1x 16 B weight load per lane per step (contiguous 1 KB per wave
            // when INTERLEAVED), XOR to two's complement, SIGNED SUMW.
            uint4 pk;
            uint16_t dslot;
            if constexpr (INTERLEAVED) {
                const uint4 * qsp = reinterpret_cast<const uint4 *>(wbase);
                pk  = qsp[i8narrow_w_interleaved_off(row, sb, n_blocks) >> 4];
                dslot = reinterpret_cast<const uint16_t *>(wbase + wg.dplane_off)[(size_t) row * wg.n_sub + sb];
            } else {
                rp_traits<GGML_TYPE_Q4_0>::load_w_raw(wbase, wg, row, sb, pk, dslot);
            }
            const uint4 w_tc = make_uint4(pk.x ^ 0x88888888u, pk.y ^ 0x88888888u,
                                          pk.z ^ 0x88888888u, pk.w ^ 0x88888888u);
            const float d_w = rp_traits<GGML_TYPE_Q4_0>::scale(rp_slot_keep(dslot, rvalid));
            const float sw  = (float) i8n_sumw_signed(pk);   // SIGNED (3b class)
            const float sw8 = 8.0f * sw;
            // Per token: 2 LDS/global uint4 reads + 4+4 chained dot8 + fold.
            #pragma unroll
            for (int c = 0; c < T; ++c) {
                const uint32_t * h32 = reinterpret_cast<const uint32_t *>(hi) + ((size_t) c * xs + sb) * 4;
                const uint32_t * l32 = reinterpret_cast<const uint32_t *>(lo) + ((size_t) c * xs + sb) * 4;
                const uint4 a_hi = make_uint4(h32[0], h32[1], h32[2], h32[3]);
                const uint4 a_lo = make_uint4(l32[0], l32[1], l32[2], l32[3]);
                int D_hi = ggml_cuda_dp8a((int) w_tc.x, (int) a_hi.x, 0);
                D_hi = ggml_cuda_dp8a((int) w_tc.y, (int) a_hi.y, D_hi);
                D_hi = ggml_cuda_dp8a((int) w_tc.z, (int) a_hi.z, D_hi);
                D_hi = ggml_cuda_dp8a((int) w_tc.w, (int) a_hi.w, D_hi);
                int D_lo = ggml_cuda_dp8a((int) w_tc.x, (int) a_lo.x, 0);
                D_lo = ggml_cuda_dp8a((int) w_tc.y, (int) a_lo.y, D_lo);
                D_lo = ggml_cuda_dp8a((int) w_tc.z, (int) a_lo.z, D_lo);
                D_lo = ggml_cuda_dp8a((int) w_tc.w, (int) a_lo.w, D_lo);
                const int32_t t = (D_hi << 4) + D_lo;                 // int32, exact
                const float d_a = __half2float(ad[(size_t) c * xs + sb]);
                acc[c] += d_w * (d_a * ((float) t + sw8));            // documented fold
            }
            // NOTE: no barrier inside this loop - v1 reads activations straight
            // from global/L1 (token-major, wave-broadcast). The LDS staging +
            // ONE __syncthreads per token block (ruquant lesson: without it the
            // compiler hoists scalar loads and spills) belongs to the staged
            // variant below / KSPLIT>1, where the barrier is placed between the
            // staging store and its consumption - never inside the token loop.
        }
    }

    // Wave-level K split (KSPLIT>1): waves share the SAME rows and reduce their
    // per-(row, token) fp32 partials through ONE shared buffer all waves add
    // into, FIXED tree order (deterministic). Integer accumulation across waves
    // is impossible with per-block f16 scales; integer exactness is used within
    // a block only (DESIGN §7).
    if constexpr (KSPLIT > 1) {
        __shared__ float red[NWAVES][64][T];
        if (rvalid) {
            #pragma unroll
            for (int c = 0; c < T; ++c) red[wave][lane][c] = acc[c];
        }
        __syncthreads();
        if (wave == 0 && rvalid) {
            #pragma unroll
            for (int c = 0; c < T; ++c) {
                float s = acc[c];
                #pragma unroll
                for (int w = 1; w < NWAVES; ++w) s += red[w][lane][c];   // fixed order
                acc[c] = s;
            }
        }
    }

    if (KSPLIT == 1) {
        if (rvalid) {
            #pragma unroll
            for (int c = 0; c < T; ++c) y[(size_t) c * ys + row] = acc[c];
        }
    } else if (wave == 0) {
        // Workgroup partials (wave 0 holds the wave-reduced value) to the
        // workspace; a fixed-order second pass reduces (no fp32 atomics -
        // nondeterministic order). Layout: [z][row*T + c].
        if (rvalid) {
            #pragma unroll
            for (int c = 0; c < T; ++c)
                ksplit_ws[((size_t) blockIdx.z * ne1 + row) * T + c] = acc[c];
        }
    }
#else
    __builtin_trap(); /* i8narrow requires HIP gfx906 */
#endif
}

// ---------------------------------------------------------------------------
// Second-pass reducer (KSPLIT > 1). Pass 1 wrote per-workgroup fp32 partials to
// ksplit_ws[z][row*T + c]; this kernel sums them over z = 0..KSPLIT-1 in FIXED
// ORDER (deterministic; no fp32 atomics) and writes y[c*ys + row]. One thread
// per output element: reads are coalesced in the ws layout (consecutive i =
// row*T + c), the strided write is fine at these sizes (N*T <= 278528 floats).
// The fp32 sum order here (0 + p0 + p1 + ...) matches the CPU mirror in the
// microbench; empty z segments never occur (dispatch guarantees seg coverage).
static __global__ void mul_mat_i8narrow_reduce_q40(
        const float * __restrict__ ksplit_ws, float * __restrict__ y,
        const uint32_t ne1, const uint32_t T, const uint32_t ys,
        const uint32_t ksplit) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne1 * T) return;
    const uint32_t row = i / T;
    const uint32_t c   = i - row * T;
    float s = 0.0f;
    for (uint32_t z = 0; z < ksplit; ++z)
        s += ksplit_ws[(size_t) z * ne1 * T + i];   // fixed z order
    y[(size_t) c * ys + row] = s;
}

// Non-template dispatch (standard llama.cpp pattern): T at runtime, switch over
// the kernel instantiations. ksplit=1 keeps the original single-pass path
// (grid.z=1, no workspace); ksplit in {2,4,8} launches pass 1 with grid.z=ksplit
// (all NWAVES waves cooperate on one 64-row group, splitting K per wave) writing
// partials to the CALLER-OWNED workspace [ksplit][ne1][T] floats, then the
// fixed-order reducer above. KSPLIT is a template parameter because the kernel
// derives its K segmentation from it at compile time - grid.z must equal it.
void mul_mat_i8narrow_launch(int T, const uint8_t * wbase, const void * hi, const void * lo,
        const ggml_half * dplane, float * y, uint32_t ne0, uint32_t ne1,
        uint32_t xs, uint32_t ys, cudaStream_t stream) {
    mul_mat_i8narrow_launch(T, wbase, hi, lo, dplane, y, ne0, ne1, xs, ys, stream, 1, nullptr);
}

void mul_mat_i8narrow_launch(int T, const uint8_t * wbase, const void * hi, const void * lo,
        const ggml_half * dplane, float * y, uint32_t ne0, uint32_t ne1,
        uint32_t xs, uint32_t ys, cudaStream_t stream, int ksplit, float * ksplit_ws) {
    constexpr int NWAVES = 4;
    if (ksplit <= 1) {
        const dim3 grid((ne1 + NWAVES * 64 - 1) / (NWAVES * 64), 1, 1);
        const dim3 blk(NWAVES * 64);
        switch (T) {
            case 1:  mul_mat_i8narrow_q40<1, NWAVES, 1, false><<<grid, blk>>>(
                         wbase, (const uint8_t *) hi, (const uint8_t *) lo, dplane, y, ne0, ne1, xs, ys, nullptr); break;
            case 2:  mul_mat_i8narrow_q40<2, NWAVES, 1, false><<<grid, blk>>>(
                         wbase, (const uint8_t *) hi, (const uint8_t *) lo, dplane, y, ne0, ne1, xs, ys, nullptr); break;
            case 4:  mul_mat_i8narrow_q40<4, NWAVES, 1, false><<<grid, blk>>>(
                         wbase, (const uint8_t *) hi, (const uint8_t *) lo, dplane, y, ne0, ne1, xs, ys, nullptr); break;
            case 8:  mul_mat_i8narrow_q40<8, NWAVES, 1, false><<<grid, blk>>>(
                         wbase, (const uint8_t *) hi, (const uint8_t *) lo, dplane, y, ne0, ne1, xs, ys, nullptr); break;
            case 16: mul_mat_i8narrow_q40<16, NWAVES, 1, false><<<grid, blk>>>(
                         wbase, (const uint8_t *) hi, (const uint8_t *) lo, dplane, y, ne0, ne1, xs, ys, nullptr); break;
            default: GGML_ABORT("i8narrow: unsupported T");
        }
        return;
    }

    GGML_ASSERT(ksplit_ws != nullptr && "i8narrow KSPLIT>1 needs a caller-allocated [ksplit][ne1][T] float workspace");
    const dim3 grid((ne1 + 64 - 1) / 64, 1, ksplit);
    const dim3 blk(NWAVES * 64);
#define I8N_KSPLIT_CASE(TV, KV) \
        mul_mat_i8narrow_q40<TV, NWAVES, KV, false><<<grid, blk, 0, stream>>>( \
            wbase, (const uint8_t *) hi, (const uint8_t *) lo, dplane, y, ne0, ne1, xs, ys, ksplit_ws)
#define I8N_KSPLIT_SWITCH(TV)                                                    \
        switch (ksplit) {                                                        \
            case 2: I8N_KSPLIT_CASE(TV, 2); break;                               \
            case 4: I8N_KSPLIT_CASE(TV, 4); break;                               \
            case 8: I8N_KSPLIT_CASE(TV, 8); break;                               \
            default: GGML_ABORT("i8narrow: unsupported KSPLIT (want 1/2/4/8)");  \
        }
    switch (T) {
        case 1:  I8N_KSPLIT_SWITCH(1);  break;
        case 2:  I8N_KSPLIT_SWITCH(2);  break;
        case 4:  I8N_KSPLIT_SWITCH(4);  break;
        case 8:  I8N_KSPLIT_SWITCH(8);  break;
        case 16: I8N_KSPLIT_SWITCH(16); break;
        default: GGML_ABORT("i8narrow: unsupported T");
    }
#undef I8N_KSPLIT_SWITCH
#undef I8N_KSPLIT_CASE

    // Pass 2: fixed-order reduce of the [ksplit][ne1][T] partials into y.
    const uint32_t total = ne1 * (uint32_t) T;
    mul_mat_i8narrow_reduce_q40<<<(total + 255) / 256, 256, 0, stream>>>(
            ksplit_ws, y, ne1, (uint32_t) T, ys, (uint32_t) ksplit);
}

// ---------------------------------------------------------------------------
// Device quantizer: fp32 -> three SPLIT planes (hi | lo | d), token-major.
// Authored for task B2 (was a GGML_ABORT stub). Mirrors quantize_mmq_q8_1_u4
// (quantize.cu:579-672): same thread mapping (one thread per 4 values, 8
// threads per 32-value block, canonical Q4_0 pairing byte j = {val j low,
// val j+16 high}), same shuffle-reduce style - but with the q8_1 SYMMETRIC
// semantics of quantize_row_q8_1_ref (ggml-quants.c:302): amax over the
// 32-value block, d = amax/127 (f32, stored f16), id = 1/d, a = roundf(x*id)
// clamped to [-127,127]. That produces bit-identical int8 codes to the stock
// q8_1 activation path (seal invariant), then the lossless nibble split
// a_hi = (a>>4)&0xF, lo stored (a&0xF)^8 (dot8 sees the signed a&0xF - 8).
//
// LAYOUT (i8split_geom, w4a4-i8narrow.cuh): three separate planes,
//   ib = (z*ne1 + x)*nkb + k_block   (token-major; nkb = ne0/32)
//   hi[ib] 16 B, lo[ib] 16 B, d[ib] 2 B (f16 bits).
// Out-of-range threads (i0 in [ne00, ne0)) write the zero-input pattern
// (a = 0, d = 0) into the row padding, exactly like the q8_1 MMQ quantizer;
// the padding blocks are never read by the kernel (it stops at ne00/32).
static __global__ void quantize_mmq_i8split(
        const float * __restrict__ x,
        void * __restrict__ vhi, void * __restrict__ vlo, void * __restrict__ vd,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    const int64_t i0 = ((int64_t) blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i2  = blockIdx.z % ne2;
    const int64_t i3  = blockIdx.z / ne2;
    const int64_t i01 = blockIdx.x;
    const int64_t base_idx = i3*s03 + i2*s02 + i01*s01;

    const int64_t k_block = i0 / 32; // 32-value block index in the channel
    const int64_t iqs     = i0 % 32; // value index in the block (0, 4, ..., 28)

    // De-interleaved value mapping mirroring the canonical Q4_0 WEIGHT layout
    // (same as quantize_mmq_q8_1_u4: thread t of the 8 covering one block owns
    // output bytes 2t,2t+1: low nibbles = values 2t,2t+1; high = 2t+16,2t+17).
    const int64_t blk = i0 - iqs;
    const int     t   = (int) (iqs / 4);

    const float2 va = (blk + 2*t + 1  < ne00)
        ? *(const float2 *) (x + base_idx + blk + 2*t)      : make_float2(0.0f, 0.0f);
    const float2 vb = (blk + 2*t + 17 < ne00)
        ? *(const float2 *) (x + base_idx + blk + 2*t + 16) : make_float2(0.0f, 0.0f);

    // amax across the 8 threads covering one 32-value block (exact under any
    // fold order; same shuffle idiom as quantize_mmq_q8_1_u4).
    float amax = fmaxf(fmaxf(fabsf(va.x), fabsf(va.y)), fmaxf(fabsf(vb.x), fabsf(vb.y)));
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    // quantize_row_q8_1_ref semantics: f32 d/id, f16 STORE of d, roundf,
    // clamp [-127,127] (the clamp is a no-op in range - |x*id| <= 127 - and
    // guards NaN/inf edges exactly like the stock device quantizer).
    const float d  = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;

    int a0 = (int) roundf(va.x * id);   // value 2t
    int a1 = (int) roundf(va.y * id);   // value 2t+1
    int a2 = (int) roundf(vb.x * id);   // value 2t+16
    int a3 = (int) roundf(vb.y * id);   // value 2t+17
    a0 = a0 < -127 ? -127 : (a0 > 127 ? 127 : a0);
    a1 = a1 < -127 ? -127 : (a1 > 127 ? 127 : a1);
    a2 = a2 < -127 ? -127 : (a2 > 127 ? 127 : a2);
    a3 = a3 < -127 ? -127 : (a3 > 127 ? 127 : a3);

    // Nibble split, canonical pairing: byte 2t   = {val 2t hi/lo, val 2t+16},
    // byte 2t+1 = {val 2t+1, val 2t+17}. hi nibble = (a>>4)&0xF (arithmetic
    // shift then mask - same expression as the host packer quantize_row_i8split);
    // lo nibble stored (a&0xF)^8 so dot8 sees the signed value (a&0xF)-8.
    const uint32_t hi_packed =  ((uint32_t) ((a0 >> 4) & 0xF))
                             | ((uint32_t) ((a2 >> 4) & 0xF) <<  4)
                             | ((uint32_t) ((a1 >> 4) & 0xF) <<  8)
                             | ((uint32_t) ((a3 >> 4) & 0xF) << 12);
    const uint32_t lo_packed =  ((uint32_t) ((a0 & 0xF) ^ 8))
                             | ((uint32_t) ((a2 & 0xF) ^ 8) <<  4)
                             | ((uint32_t) ((a1 & 0xF) ^ 8) <<  8)
                             | ((uint32_t) ((a3 & 0xF) ^ 8) << 12);

    const int64_t nkb = ne0 / 32; // 32-value blocks per channel column
    const int64_t ib  = ((int64_t) blockIdx.z * ne1 + blockIdx.x) * nkb + k_block;

    // one 16-bit store per plane for the nibble pair, one 16-bit d store by
    // the block's first thread (planes are separate allocations - i8split_geom).
    *(uint16_t *) ((char *) vhi + ib*16 + 2*t) = (uint16_t) hi_packed;
    *(uint16_t *) ((char *) vlo + ib*16 + 2*t) = (uint16_t) lo_packed;
    if (iqs == 0) {
        *(uint16_t *) ((char *) vd + ib*2) = __half_as_ushort((ggml_half) d);
    }
}

void quantize_mmq_i8split_cuda(const float * __restrict__ x, void * __restrict__ vhi,
        void * __restrict__ vlo, void * __restrict__ vd,
        int64_t ne00, int64_t s01, int64_t s02, int64_t s03,
        int64_t ne0, int ne1, int ne2, int ne3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % 32 == 0);
    // Grid/block structure identical to quantize_mmq_q8_1_u4_cuda: one block.y
    // per 512 padded values, block.x = 128 threads (4 values each), z = slices.
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks((unsigned) ne1, (unsigned) block_num_y, (unsigned) (ne2*ne3));
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    quantize_mmq_i8split<<<num_blocks, block_size, 0, stream>>>(
        x, vhi, vlo, vd, ne00, s01, s02, s03, ne0, (int) ne1, (int) ne2);
}

// The dispatch function's switch implicitly instantiates all five widths.
