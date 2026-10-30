// Tier-3.7b: gfx906 tiled GEMM for repacked-u4 (Q4_0) weights.
// Port of the gcn_q8_gemm / gcn_kq4_gemm tiling pattern (gcn-q8-gemm.cu) to the
// repacked-u4 buffer layout: 256-thread blocks, 128 weight rows x 64 tokens,
// 8x4 register tile per thread, LDS staging with 9-dword row padding, register
// prefetch of the next K step. Gate: GGML_CUDA_REPACK_Q4_0_TILED (default OFF).
//
// NUMERICS: the tile order and the per-block epilogue association differ from
// the generic mmq_gemm_repacked loop (fma contraction of d*(dx*s - 8*sx)), so
// FP rounding shifts. Enabling this gate REQUIRES A RE-SEAL of the Q4_0 path.
#pragma once

#include "../common.cuh"

#include <cstdint>

struct repack_tile_meta;
struct block_q8_1_mmq_ds_h;

// Gate: GGML_CUDA_REPACK_Q4_0_TILED (default OFF). When on, the repacked Q4_0
// dense and MoE-id tiled GEMMs run gcn_u4_gemm instead of the generic MMQ loop.
bool ggml_cuda_repack_q4_0_tiled_on();

// Dense: M weight rows x N tokens, K = ne00 values. xmmq is the block_q8_1_mmq
// (DS4) activation pool, n_groups = GGML_PAD(K, MATRIX_ROW_PADDING)/128.
void ggml_cuda_gcn_u4_gemm_dense(const uint8_t * wbase, const block_q8_1_mmq_ds_h * xmmq,
        float * dst, int M, int N, int K, int n_groups, uint32_t dst_s1,
        cudaStream_t stream);

// MoE (MUL_MAT_ID): tiles from repack_tile_off<64>; grid.y walks the tiles.
void ggml_cuda_gcn_u4_gemm_moe(const uint8_t * wbase, const block_q8_1_mmq_ds_h * xmmq,
        float * dst, int M, int K, int n_groups, uint32_t dst_s1,
        const int32_t * ids_src1, const int32_t * ids_dst, const int32_t * expert_bounds,
        const int32_t * tile_off, const repack_tile_meta * tile_meta, uint32_t n_expert,
        size_t expert_stride, int max_tiles, cudaStream_t stream);
