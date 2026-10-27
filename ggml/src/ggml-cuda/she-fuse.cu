#include "she-fuse.cuh"

// Ported from CV (castagna-veloce): hc-dec.cu shexp_gate_add + gcn-f16-matvec.cu gcn_f32_mv.
// Env gate inverted vs CV: GGML_CUDA_SHE_FUSE defaults to 0 (OFF) here; CV defaults its two kernels ON
// (GGML_CUDA_SHEXP_GATE_FUSE / GGML_CUDA_GCN_F32_MV). KS tuning knob GGML_CUDA_SHE_FUSE_KS (CV: GGML_CUDA_GCN_F32_MV_KS).
// GGML_CUDA_SHE_FUSE_TRACE=1 prints every fusion firing and every remaining f32 mul_mat_vec_f launch so the
// per-step kernel-count drop can be witnessed without rocprofv3.

#include <cstdio>
#include <cstdlib>
#include <climits>
#include <mutex>
#include <string>
#include <unordered_map>

static int she_fuse_env(const char * name, int dflt) {
    static std::unordered_map<std::string, int> cache;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    auto it = cache.find(name);
    if (it != cache.end()) {
        return it->second;
    }
    const char * e = getenv(name);
    const int v = e ? atoi(e) : dflt;
    cache.emplace(name, v);
    return v;
}

bool ggml_cuda_she_fuse_trace(void) {
    return she_fuse_env("GGML_CUDA_SHE_FUSE_TRACE", 0) != 0;
}

// ---------------------------------------------------------------------------
// Shared-expert gate tail (CV hc-dec.cu):
// MUL_MAT(w_g [K], x) -> SIGMOID -> MUL(shexp, .) -> ADD(moe, .) in one kernel:
//   out[e,t] = moe[e,t] + shexp[e,t] * sigmoid(dot(w_g, x[:,t]))
// One block per token (the dot product needs the whole row first). Four launches -> one at decode (<= 8 tokens).
// ---------------------------------------------------------------------------
static __global__ void __launch_bounds__(256) shexp_gate_add(
        const float * __restrict__ wg, const float * __restrict__ x, const float * __restrict__ shexp,
        const float * __restrict__ moe, float * __restrict__ out, const int K, const int E,
        const int64_t s_x, const int64_t s_sh, const int64_t s_moe, const int64_t s_out) {
    const int64_t t = blockIdx.x;
    const int tid = threadIdx.x;
    const float4 * xp = (const float4 *) (x + t*s_x);
    const float4 * wp = (const float4 *) wg;
    float acc = 0.0f;
    for (int k = tid; k < K/4; k += 256) {
        const float4 a = xp[k], b = wp[k];
        acc += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
    }
    constexpr int ws = ggml_cuda_get_physical_warp_size();
    __shared__ float red[256/ws];
    acc = warp_reduce_sum<ws>(acc);
    if (tid % ws == 0) {
        red[tid / ws] = acc;
    }
    __syncthreads();
    float dot = 0.0f;
#pragma unroll
    for (int i = 0; i < 256/ws; ++i) {
        dot += red[i];
    }
    const float g = 1.0f / (1.0f + expf(-dot));
    for (int e = tid; e < E; e += 256) {
        out[t*s_out + e] = moe[t*s_moe + e] + shexp[t*s_sh + e]*g;
    }
}

static bool shexp_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data, * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

bool ggml_cuda_shexp_gate_add_supported(const ggml_tensor * mm, const ggml_tensor * sig, const ggml_tensor * mul,
                                        const ggml_tensor * add) {
    static const int enabled = she_fuse_env("GGML_CUDA_SHE_FUSE", 0);
    if (!enabled) {
        return false;
    }
    const ggml_tensor * wg = mm->src[0], * x = mm->src[1];
    if (sig->src[0] != mm || !(mul->src[0] == sig || mul->src[1] == sig) || !(add->src[0] == mul || add->src[1] == mul)) {
        return false;
    }
    const ggml_tensor * sh  = mul->src[0] == sig ? mul->src[1] : mul->src[0];
    const ggml_tensor * moe = add->src[0] == mul ? add->src[1] : add->src[0];
    const int64_t K = wg->ne[0], E = sh->ne[0], N = x->ne[1];
    // same-index elementwise tail (out may replace moe/shexp exactly); the dot inputs may not be overwritten
    // by another token's block
    const auto same_or_disjoint = [](const ggml_tensor * o, const ggml_tensor * in) {
        return !shexp_overlap(o, in) || (o->data == in->data && o->nb[1] == in->nb[1]);
    };
    // decode / verify only: at prefill the launches do not matter and the unfused path keeps prefill bit-identical
    return N <= 8 && wg->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 && sh->type == GGML_TYPE_F32 && moe->type == GGML_TYPE_F32 &&
        add->type == GGML_TYPE_F32 && wg->ne[1] == 1 && ggml_is_contiguous(wg) && K % 4 == 0 &&
        x->ne[0] == K && x->nb[0] == sizeof(float) && x->nb[1] % 16 == 0 && ((uintptr_t) x->data) % 16 == 0 &&
        ((uintptr_t) wg->data) % 16 == 0 && x->ne[2] == 1 && x->ne[3] == 1 &&
        mm->ne[0] == 1 && mm->ne[1] == N && ggml_are_same_shape(sh, moe) && ggml_are_same_shape(add, sh) &&
        sh->ne[1] == N && sh->ne[2] == 1 && sh->ne[3] == 1 && sh->nb[0] == sizeof(float) && moe->nb[0] == sizeof(float) &&
        add->nb[0] == sizeof(float) && E > 0 &&
        !shexp_overlap(add, wg) && same_or_disjoint(add, x) && same_or_disjoint(add, sh) && same_or_disjoint(add, moe);
}

void ggml_cuda_shexp_gate_add(ggml_backend_cuda_context & ctx, const ggml_tensor * mm, const ggml_tensor * sig,
                              const ggml_tensor * mul, ggml_tensor * add) {
    const ggml_tensor * wg = mm->src[0], * x = mm->src[1];
    const ggml_tensor * sh  = mul->src[0] == sig ? mul->src[1] : mul->src[0];
    const ggml_tensor * moe = add->src[0] == mul ? add->src[1] : add->src[0];
    if (ggml_cuda_she_fuse_trace()) {
        fprintf(stderr, "[she-fuse] shexp_gate_add K=%d E=%d N=%d\n", (int) wg->ne[0], (int) sh->ne[0], (int) x->ne[1]);
    }
    shexp_gate_add<<<(unsigned) x->ne[1], 256, 0, ctx.stream()>>>(
        (const float *) wg->data, (const float *) x->data, (const float *) sh->data, (const float *) moe->data,
        (float *) add->data, (int) wg->ne[0], (int) sh->ne[0], x->nb[1]/sizeof(float), sh->nb[1]/sizeof(float),
        moe->nb[1]/sizeof(float), add->nb[1]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Dense F32 x F32 matvec (CV gcn-f16-matvec.cu gcn_f32_mv): F32 weights with 65..8192 rows (the MoE router).
// KS waves per row, each streaming a rotated slice of the row in 16-byte loads; the activations come from global
// memory (L1/L2) as float4. Fewer rows than 65 keep mul_mat_vec_f.
// ---------------------------------------------------------------------------
template <int NT, int KS>
__launch_bounds__(256)
static __global__ void gcn_f32_mv(const float * __restrict__ W, const float * __restrict__ y, float * __restrict__ dst,
        const int K, const int M, const int64_t w_stride, const int64_t y_stride, const int64_t d_stride) {
    __shared__ float red[KS > 1 ? 4*NT : 1];
    const int lane = threadIdx.x % 64;
    const int wave = threadIdx.x / 64;
    const int ks   = wave % KS;
    const int row  = blockIdx.x*(4/KS) + wave / KS;
    const bool live = row < M;
    const int nc   = K / 4; // 16-byte chunks
    const int ns   = (nc + 63) / 64;
    const int s_lo = ks*ns/KS;
    const int sn   = (ks + 1)*ns/KS - s_lo;
    const int rot  = sn > 0 ? row % sn : 0;
    const float4 * wr = (const float4 *) (W + (int64_t) min(row, M - 1)*w_stride);
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        acc[t] = 0.0f;
    }
    if (live) {
#pragma unroll 4
        for (int s0 = 0; s0 < sn; ++s0) {
            const int c = (s_lo + (s0 + rot < sn ? s0 + rot : s0 + rot - sn))*64 + lane;
            if (c >= nc) {
                continue;
            }
            const float4 q = wr[c];
#pragma unroll
            for (int t = 0; t < NT; ++t) {
                const float4 a = *(const float4 *) (y + t*y_stride + 4*c);
                float s = acc[t];
                s = fmaf(q.x, a.x, s);
                s = fmaf(q.y, a.y, s);
                s = fmaf(q.z, a.z, s);
                s = fmaf(q.w, a.w, s);
                acc[t] = s;
            }
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        float v = acc[t];
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            v += __shfl_xor(v, off, 64);
        }
        if constexpr (KS == 1) {
            if (lane == t && live) {
                dst[t*d_stride + row] = v;
            }
        } else if (lane == t) {
            red[wave*NT + t] = v;
        }
    }
    if constexpr (KS > 1) {
        __syncthreads();
        if (ks == 0 && lane < NT && live) {
            float v = 0.0f;
#pragma unroll
            for (int k = 0; k < KS; ++k) {
                v += red[(wave + k)*NT + lane];
            }
            dst[lane*d_stride + row] = v;
        }
    }
}

bool ggml_cuda_gcn_f32_matvec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static const int env    = she_fuse_env("GGML_CUDA_SHE_FUSE", 0);
    static const int env_ks = she_fuse_env("GGML_CUDA_SHE_FUSE_KS", 0);
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int64_t K = src0->ne[0], M = src0->ne[1], N = src1->ne[1];
    if (!env || !GGML_CUDA_CC_IS_GCN(cc) || src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 ||
            dst->type != GGML_TYPE_F32 || N < 1 || N > 4 || M <= 64 || M > 8192 || K % 4 != 0 || K < 1024 || K > INT_MAX ||
            src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            src0->nb[0] != sizeof(float) || src0->nb[1] % 16 != 0 || ((uintptr_t) src0->data) % 16 != 0 ||
            src1->nb[0] != sizeof(float) || (N > 1 && src1->nb[1] % 16 != 0) || ((uintptr_t) src1->data) % 16 != 0 ||
            dst->nb[0] != sizeof(float)) {
        return false;
    }
    const int ks = env_ks ? env_ks : M < 1024 ? 4 : M < 4096 ? 2 : 1;
    if (ggml_cuda_she_fuse_trace()) {
        fprintf(stderr, "[she-fuse] gcn_f32_mv M=%d K=%d N=%d ks=%d\n", (int) M, (int) K, (int) N, ks);
    }
    const int64_t ws = src0->nb[1] / sizeof(float);
    const int64_t ys = N > 1 ? src1->nb[1] / sizeof(float) : 0;
    const int64_t ds = N > 1 ? dst->nb[1] / sizeof(float) : 0;
    cudaStream_t stream = ctx.stream();
#define F32MV(NT_, KS_) gcn_f32_mv<NT_, KS_><<<(unsigned) ((M + 4/(KS_) - 1)/(4/(KS_))), 256, 0, stream>>>( \
        (const float *) src0->data, (const float *) src1->data, (float *) dst->data, (int) K, (int) M, ws, ys, ds)
#define F32MV_KS(NT_) switch (ks) { case 4: F32MV(NT_, 4); break; case 2: F32MV(NT_, 2); break; default: F32MV(NT_, 1); break; }
    switch (N) {
        case 1:  F32MV_KS(1); break;
        case 2:  F32MV_KS(2); break;
        case 3:  F32MV_KS(3); break;
        default: F32MV_KS(4); break;
    }
#undef F32MV_KS
#undef F32MV
    CUDA_CHECK(cudaGetLastError());
    return true;
}
