// i8narrow kernel + activation quantizer. STATUS: AUTHORED, NOT COMPILED
// (build pause point - see docs/DESIGN-i8-narrow-kernel.md §10 and taskC-LEDGER.md).
// Design + seal references: docs/DESIGN-i8-narrow-kernel.md, seals/seal_i8split.cc.

#include "w4a4-i8narrow.cuh"

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

#if defined(GGML_USE_HIP) && defined(__gfx906__)

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
        // TODO(post-pause): epilogue reduce kernel, fixed order over blockIdx.z.
    }
}

template <int T, int NWAVES, int KSPLIT, bool INTERLEAVED>
void mul_mat_i8narrow_launch(const uint8_t * wbase, const void * hi, const void * lo,
        const ggml_half * dplane, float * y, uint32_t ne0, uint32_t ne1,
        uint32_t xs, uint32_t ys, float * ksplit_ws, cudaStream_t stream) {
    // KSPLIT==1: NWAVES*64 rows per block; KSPLIT>1: 64 rows per block, the
    // other waves split K (kernel geometry comment).
    const dim3 grid(KSPLIT == 1 ? (ne1 + NWAVES * 64 - 1) / (NWAVES * 64)
                                : (ne1 + 63) / 64, 1, KSPLIT);
    mul_mat_i8narrow_q40<T, NWAVES, KSPLIT, INTERLEAVED>
        <<<grid, NWAVES * 64, 0, stream>>>(wbase, (const uint8_t *) hi, (const uint8_t *) lo,
                                           dplane, y, ne0, ne1, xs, ys, ksplit_ws);
}

// TODO(post-pause): quantize_mmq_i8split_cuda device kernel - mirror
// quantize_mmq_q8_1_u4 (quantize.cu:579-672): same min/max shuffle reduce, same
// roundf/clamp/f16-scale semantics as quantize_row_q8_1_ref, then emit the two
// nibble planes + scale plane (token-major ib = (z*ne1 + x)*nkb + k_block,
// quantize.cu:657-665). NOT authored yet: the microbench can seed activations
// through the host packer first.
void quantize_mmq_i8split_cuda(const float * __restrict__, void * __restrict__,
        void * __restrict__, void * __restrict__, int64_t, int64_t, int64_t, int64_t,
        int64_t, int, int, cudaStream_t) {
    GGML_ABORT("i8narrow device quantizer not authored yet (pre-pause skeleton)");
}

#else // !GGML_USE_HIP || !__gfx906__

template <int T, int NWAVES, int KSPLIT, bool INTERLEAVED>
void mul_mat_i8narrow_launch(const uint8_t *, const void *, const void *,
        const ggml_half *, float *, uint32_t, uint32_t, uint32_t, uint32_t,
        float *, cudaStream_t) {
    NO_DEVICE_CODE;
}
void quantize_mmq_i8split_cuda(const float * __restrict__, void * __restrict__,
        void * __restrict__, void * __restrict__, int64_t, int64_t, int64_t, int64_t,
        int64_t, int, int, cudaStream_t) {
    NO_DEVICE_CODE;
}

#endif

// Explicit instantiation set for the microbench (widths 1/2/4/8/16, KSPLIT 1):
template void mul_mat_i8narrow_launch<1, 4, 1, false>(const uint8_t *, const void *,
    const void *, const ggml_half *, float *, uint32_t, uint32_t, uint32_t, uint32_t,
    float *, cudaStream_t);
template void mul_mat_i8narrow_launch<2, 4, 1, false>(const uint8_t *, const void *,
    const void *, const ggml_half *, float *, uint32_t, uint32_t, uint32_t, uint32_t,
    float *, cudaStream_t);
template void mul_mat_i8narrow_launch<4, 4, 1, false>(const uint8_t *, const void *,
    const void *, const ggml_half *, float *, uint32_t, uint32_t, uint32_t, uint32_t,
    float *, cudaStream_t);
template void mul_mat_i8narrow_launch<8, 4, 1, false>(const uint8_t *, const void *,
    const void *, const ggml_half *, float *, uint32_t, uint32_t, uint32_t, uint32_t,
    float *, cudaStream_t);
template void mul_mat_i8narrow_launch<16, 4, 1, false>(const uint8_t *, const void *,
    const void *, const ggml_half *, float *, uint32_t, uint32_t, uint32_t, uint32_t,
    float *, cudaStream_t);
