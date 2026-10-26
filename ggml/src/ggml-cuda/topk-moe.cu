#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "topk-moe.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

// Kernel config struct - passed by value to CUDA kernel
struct topk_moe_config {
    bool use_sigmoid;
    bool use_sqrt_softplus;
    bool with_norm;
    bool delayed_softmax;
};

// Warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path.
template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    // Match the standalone softmax reduction order: reduce contiguous groups of
    // WARP_SIZE values first, then reduce the group results. Keeping the fused and
    // unfused paths bitwise-equivalent prevents graph-layout-dependent routing.
    float group_reductions[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        group_reductions[i] = warp_reduce_max(active ? vals[i] : -INFINITY);
    }

    float max_val = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (lane == i) {
            max_val = group_reductions[i];
        }
    }

    max_val = warp_reduce_max(max_val);

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i]         = val;
        } else {
            vals[i] = 0.f;
        }
        group_reductions[i] = warp_reduce_sum(vals[i]);
    }

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (lane == i) {
            sum = group_reductions[i];
        }
    }

    sum = warp_reduce_sum(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + expf(-vals[i])) : -INFINITY;
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sqrt_softplus_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? sqrtf(vals[i] > 20.0f ? vals[i] : logf(1.0f + expf(vals[i]))) : -INFINITY;
    }
}

/*
    This kernel does the following:
    1. optionally softmax over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits

    It is intended as fusion of softmax->top-k->get_rows pipeline for MoE models
*/
template <int n_experts, bool has_bias>
__launch_bounds__(TOPK_MOE_ROWS_PER_BLOCK * WARP_SIZE, 1)
__global__ void topk_moe_cuda(const float *         logits,
                              float *               weights,
                              int32_t *             ids,
                              float *               bias,
                              const int             n_rows,
                              const int             n_expert_used,
                              const float           clamp_val,
                              const float           scale_val,
                              const topk_moe_config config) {
#if defined(GGML_USE_MUSA)
    // MUSA: every warp of a partially filled block must reach the barrier below.
    const int row = MIN(blockIdx.x * blockDim.y + threadIdx.y, n_rows - 1);
#else
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
#endif // defined(GGML_USE_MUSA)
    if (row >= n_rows) {
        return;
    }

    logits += n_experts * row;
    weights += n_expert_used * row;
    ids += n_experts * row;

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];

    // Initialize all slots to -INFINITY
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    // Weights and IDs can alias logits, so wait until every row in the block reads its logits.
    __syncthreads();

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
           sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else if (config.use_sqrt_softplus) {
           sqrt_softplus_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else {
           softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs.
    // NaN comparisons always return false, which would cause the same expert to be
    // selected repeatedly. -FLT_MAX compares normally and is still excluded by the
    // -INFINITY sentinel used after each selection round.
    // More relevant for the cuBLAS path. See https://github.com/ggml-org/llama.cpp/issues/19659
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // selection_wt is only needed when bias is present (selection uses wt + bias)
    // when no bias, we use wt directly for both selection and weight values
    [[maybe_unused]] float selection_wt[has_bias ? experts_per_thread : 1];

    if constexpr (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            const int expert = i + threadIdx.x;
            selection_wt[i / WARP_SIZE] =
                (n_experts % WARP_SIZE == 0 || expert < n_experts) ? wt[i / WARP_SIZE] + bias[expert] : -INFINITY;
        }
    }

    //at this point, each thread holds either a portion of the softmax distribution
    //or the raw logits. We do the argmax reduce over n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ggml_cuda_pdl_lc();
    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = threadIdx.x;

        if constexpr (has_bias) {
            float max_val_s = selection_wt[0];

#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && selection_wt[i] > max_val_s) {
                    max_val    = wt[i];
                    max_val_s  = selection_wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const float val_s  = __shfl_xor_sync(0xFFFFFFFF, max_val_s, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                    max_val    = val;
                    max_val_s  = val_s;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                selection_wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        } else {
#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                    max_val    = wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val > max_val || (val == max_val && expert < max_expert)) {
                    max_val    = val;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        }

        if ((k & (WARP_SIZE - 1)) == threadIdx.x) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            ids[k] = max_expert;
        }
    }

    if (config.with_norm) {
        // The standalone path reduces the selected weights in contiguous
        // WARP_SIZE groups, clamps that sum, then applies GGML_OP_DIV.
        float group_sums[experts_per_thread];

#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            group_sums[i] = warp_reduce_sum(output_weights[i]);
        }

        float wt_sum = 0.f;

#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            if (threadIdx.x == i) {
                wt_sum = group_sums[i];
            }
        }

        wt_sum = warp_reduce_sum(wt_sum);
        wt_sum = fmaxf(wt_sum, clamp_val);

        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] /= wt_sum;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template<bool has_bias>
static void launch_topk_moe_cuda(ggml_backend_cuda_context & ctx,
                                 const float *               logits,
                                 float *                     weights,
                                 int32_t *                   ids,
                                 float *                     bias,
                                 const int                   n_rows,
                                 const int                   n_expert,
                                 const int                   n_expert_used,
                                 const float                 clamp_val,
                                 const float                 scale_val,
                                 const topk_moe_config       config) {
    GGML_ASSERT(!(config.with_norm && config.delayed_softmax) &&
                "delayed softmax is not supported with weight normalization");
    const int    cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int    rows_per_block = cc == GGML_CUDA_CC_VEGA20 && n_rows == 1 ? 1 : TOPK_MOE_ROWS_PER_BLOCK;
    dim3         grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(WARP_SIZE, rows_per_block, 1);
    cudaStream_t stream = ctx.stream();
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    switch (n_expert) {
        case 1:
            ggml_cuda_kernel_launch(topk_moe_cuda<1, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 2:
            ggml_cuda_kernel_launch(topk_moe_cuda<2, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 4:
            ggml_cuda_kernel_launch(topk_moe_cuda<4, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 8:
            ggml_cuda_kernel_launch(topk_moe_cuda<8, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 16:
            ggml_cuda_kernel_launch(topk_moe_cuda<16, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 32:
            ggml_cuda_kernel_launch(topk_moe_cuda<32, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 64:
            ggml_cuda_kernel_launch(topk_moe_cuda<64, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 128:
            ggml_cuda_kernel_launch(topk_moe_cuda<128, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 256:
            ggml_cuda_kernel_launch(topk_moe_cuda<256, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 288: // StepFun 3.7
            ggml_cuda_kernel_launch(topk_moe_cuda<288, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 384: // DeepSeek-V4.1-Flash
            ggml_cuda_kernel_launch(topk_moe_cuda<384, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 512:
            ggml_cuda_kernel_launch(topk_moe_cuda<512, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 576:
            ggml_cuda_kernel_launch(topk_moe_cuda<576, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        default:
            GGML_ASSERT(false && "fatal error");
            break;
    }
}

// CV split-k top-k router port: the warp kernel above is latency bound on gfx906 (n_expert_used rounds of argmax,
// each with dependent shuffle pairs through LDS, on half a wave). Here each lane sorts its NE/64 probabilities in
// registers, then n_expert_used rounds take the wave maximum of the list heads with DPP moves (no LDS) and pop the
// winner; ties go to the lowest expert index as in the argmax rounds. The softmax sum and the weight sum are
// accumulated exactly like the reference warp kernel (32-lane partials over experts L + 32 i in order i, then the
// same butterfly), so ids and weights are bit-identical to the reference reduction order.
#ifdef GGML_USE_HIP
// width-parameterized warp reductions (the fork's common.cuh helpers have no width template parameter)
template <int width>
static __device__ __forceinline__ float topk_warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

template <int width>
static __device__ __forceinline__ float topk_warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
    }
    return x;
}

template <int ctrl, int row_mask = 0xF>
static __device__ __forceinline__ uint32_t topk_dpp(const uint32_t v) {
    return (uint32_t) __builtin_amdgcn_update_dpp((int) v, (int) v, ctrl, row_mask, 0xF, false);
}

// wave-wide unsigned max / min (result in every lane)
template <bool MAX>
static __device__ __forceinline__ uint32_t topk_wave_reduce(uint32_t v) {
    const auto op = [](uint32_t a, uint32_t b) { return MAX ? max(a, b) : min(a, b); };
    v = op(v, topk_dpp<0xB1>(v));        // quad_perm [1,0,3,2]
    v = op(v, topk_dpp<0x4E>(v));        // quad_perm [2,3,0,1]
    v = op(v, topk_dpp<0x141>(v));       // row_half_mirror
    v = op(v, topk_dpp<0x140>(v));       // row_mirror: every lane has its row's result
    v = op(v, topk_dpp<0x142, 0xA>(v));  // row_bcast15
    v = op(v, topk_dpp<0x143, 0xC>(v));  // row_bcast31: lane 63 has the wave's result
    return (uint32_t) __builtin_amdgcn_readlane((int) v, 63);
}

template <int NE>
__launch_bounds__(64)
static __global__ void topk_moe_sorted_gcn(const float * __restrict__ logits, float * __restrict__ weights, int32_t * __restrict__ ids,
                                           const int n_expert_used, const float clamp_val, const float scale_val, const bool with_norm) {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__GFX9__) // selected on GCN only: other targets compile an empty kernel
    constexpr int NL = 32;       // lanes of the reference warp kernel
    constexpr int J  = NE/64;    // values per lane
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;
    __shared__ float sp[NE];
    __shared__ float sel_p[64];
    __shared__ int   sel_e[64];
    __shared__ float red;

    float v[J];
#pragma unroll
    for (int j = 0; j < J; ++j) {
        v[j] = logits[(int64_t) row*NE + lane + 64*j];
    }
    float m = v[0];
#pragma unroll
    for (int j = 1; j < J; ++j) {
        m = fmaxf(m, v[j]);
    }
    m = topk_warp_reduce_max<64>(m);
#pragma unroll
    for (int j = 0; j < J; ++j) {
        v[j] = expf(v[j] - m);
        sp[lane + 64*j] = v[j];
    }
    __syncthreads();
    // reference: lane L < 32 sums experts L + 32 i for i = 0 .. NE/32 - 1 in order, then a 32-lane butterfly
    if (lane < NL) {
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < NE/NL; ++i) {
            s += sp[lane + NL*i];
        }
        s = topk_warp_reduce_sum<NL>(s);
        if (lane == 0) {
            red = s;
        }
    }
    __syncthreads();
    const float inv_sum = 1.0f / red;
    uint32_t key[J];
    int      ex[J];
#pragma unroll
    for (int j = 0; j < J; ++j) {
        float p = v[j] * inv_sum;
        if (__isnanf(p)) {
            p = -FLT_MAX;
        }
        const uint32_t u = __float_as_uint(p);
        key[j] = u & 0x80000000u ? ~u : u | 0x80000000u; // larger float -> larger key, never 0
        ex[j]  = lane + 64*j;
    }
    // odd-even transposition sort, descending key (equal keys keep the lower expert first: e grows with j)
#pragma unroll
    for (int pass = 0; pass < J; ++pass) {
#pragma unroll
        for (int j = pass & 1; j + 1 < J; j += 2) {
            if (key[j + 1] > key[j]) {
                const uint32_t tk = key[j]; key[j] = key[j + 1]; key[j + 1] = tk;
                const int      te = ex[j];  ex[j]  = ex[j + 1];  ex[j + 1]  = te;
            }
        }
    }
    for (int r = 0; r < n_expert_used; ++r) {
        const uint32_t mk = topk_wave_reduce<true>(key[0]);
        const uint64_t tied = __ballot(key[0] == mk);
        int win;
        if (__popcll(tied) == 1) {
            win = __ffsll((unsigned long long) tied) - 1;
        } else {
            const uint32_t me = topk_wave_reduce<false>(key[0] == mk ? (uint32_t) ex[0] : 0xFFFFFFFFu);
            win = __ffsll((unsigned long long) __ballot(key[0] == mk && (uint32_t) ex[0] == me)) - 1;
        }
        if (lane == win) {
            const uint32_t k = key[0];
            sel_p[r] = __uint_as_float(k & 0x80000000u ? k & 0x7FFFFFFFu : ~k);
            sel_e[r] = ex[0];
#pragma unroll
            for (int j = 0; j + 1 < J; ++j) {
                key[j] = key[j + 1];
                ex[j]  = ex[j + 1];
            }
            key[J - 1] = 0;
            ex[J - 1]  = 0x7FFFFFFF;
        }
    }
    __syncthreads();
    if (lane < n_expert_used) {
        ids[(int64_t) row*NE + lane] = sel_e[lane];
    }
    if (lane < NL) {
        float inv = 1.0f;
        if (with_norm) {
            float s = 0.0f;
            for (int q = 0; q < n_expert_used; ++q) {
                if (sel_e[q] % NL == lane) {
                    s += sel_p[q];
                }
            }
            s = topk_warp_reduce_sum<NL>(s);
            s = fmaxf(s, clamp_val);
            inv = 1.0f / s;
        }
        for (int q = lane; q < n_expert_used; q += NL) {
            weights[(int64_t) row*n_expert_used + q] = (with_norm ? sel_p[q]*inv : sel_p[q]) * scale_val;
        }
    }
#endif // gfx9 device code
}

// one-line engagement witness for the tail-kernel ports (GGML_CUDA_TAIL_TRACE=1)
static void topk_moe_rank_trace(const int n_experts, const int n_expert_used, const int n_rows) {
    static const bool trace = [] { const char * e = getenv("GGML_CUDA_TAIL_TRACE"); return e && atoi(e) != 0; }();
    static bool fired = false;
    if (trace && !fired) {
        fired = true;
        fprintf(stderr, "[topk_moe_sorted_gcn] n_experts=%d n_expert_used=%d n_rows=%d\n", n_experts, n_expert_used, n_rows);
    }
}
#endif // GGML_USE_HIP

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;
    float *       bias_d    = bias ? (float *) bias->data : nullptr;

    float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const int n_expert_used = weights->ne[1];

    const bool with_norm = clamp != nullptr;

    float clamp_val = -INFINITY;
    if (clamp) {
        clamp_val = ggml_get_op_params_f32(clamp, 0);
    }

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = with_norm;
    config.delayed_softmax   = args.delayed_softmax;

#ifdef GGML_USE_HIP
    static const bool rank_env = [] { const char * e = getenv("GGML_CUDA_TOPK_MOE_RANK"); return !e || atoi(e) != 0; }();
    if (rank_env && !bias && !config.use_sigmoid && !config.use_sqrt_softplus && !config.delayed_softmax &&
            n_expert_used <= 64 && GGML_CUDA_CC_IS_GCN(ggml_cuda_info().devices[ctx.device].cc) &&
            (n_experts == 128 || n_experts == 256 || n_experts == 512)) {
        cudaStream_t stream = ctx.stream();
        topk_moe_rank_trace(n_experts, n_expert_used, n_rows);
        switch (n_experts) {
            case 128: topk_moe_sorted_gcn<128><<<n_rows, 64, 0, stream>>>(logits_d, weights_d, ids_d, n_expert_used, clamp_val, scale_val, with_norm); break;
            case 256: topk_moe_sorted_gcn<256><<<n_rows, 64, 0, stream>>>(logits_d, weights_d, ids_d, n_expert_used, clamp_val, scale_val, with_norm); break;
            default:  topk_moe_sorted_gcn<512><<<n_rows, 64, 0, stream>>>(logits_d, weights_d, ids_d, n_expert_used, clamp_val, scale_val, with_norm); break;
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#endif // GGML_USE_HIP

    if (bias) {
        launch_topk_moe_cuda<true>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    } else {
        launch_topk_moe_cuda<false>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    }
}

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids) {
    // must match an instantiation of launch_topk_moe_cuda: a power of 2 up to 512,
    // or one of the non-power-of-2 expert counts of supported models
    const int n_expert = ids->nb[1] / ids->nb[0];
    if (((n_expert & (n_expert - 1)) != 0 || n_expert > 512) && n_expert != 288 && n_expert != 384 && n_expert != 576) {
        return false;
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }

    if (gating_op->op == GGML_OP_SOFT_MAX) {
        const ggml_tensor * softmax  = gating_op;
        float               scale    = 1.0f;
        float               max_bias = 0.0f;

        memcpy(&scale, (const float *) softmax->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) softmax->op_params + 1, sizeof(float));

        if (!ggml_is_contiguous(softmax->src[0])) {
            return false;
        }

        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }

        // don't fuse when masks or sinks are present
        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
    } else if (gating_op->op == GGML_OP_UNARY) {
        ggml_unary_op op = ggml_get_unary_op(gating_op);

        if (op != GGML_UNARY_OP_SIGMOID && op != GGML_UNARY_OP_SOFTPLUS) {
            return false;
        }
    }

    return true;
}
