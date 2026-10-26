#include "gdn-fuse.cuh"

// Ported from CV (castagna-veloce) hc-dec.cu:234-330 @3483d46. Env gate inverted vs CV:
// GGML_CUDA_GDN_AB_DEC defaults to 0 (OFF) here; CV defaults it ON. Set =1 to enable.

static int gdn_fuse_env(const char * name, int dflt) {
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

// GDN gate inputs at decode: MUL_MAT(alpha) -> RESHAPE -> ADD(dt) -> SOFTPLUS -> MUL(A) and MUL_MAT(beta) -> RESHAPE ->
// SIGMOID share the activation: one wave per output row (alpha rows first, then beta rows), f32 weights
template <int NT>
__launch_bounds__(256)
static __global__ void gdn_ab_dec(
        const float * __restrict__ x, const float * __restrict__ wa, const float * __restrict__ wb, const float * __restrict__ dt,
        const float * __restrict__ A, float * __restrict__ gate, float * __restrict__ beta, const int K, const int H,
        const int64_t x_st, const int64_t wa_s, const int64_t wb_s, const int64_t g_st, const int64_t b_st) {
    const int r    = blockIdx.x*4 + threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
    if (r >= 2*H) {
        return;
    }
    const bool is_b = r >= H;
    const int  h    = is_b ? r - H : r;
    const float4 * w4 = (const float4 *) (is_b ? wb + h*wb_s : wa + h*wa_s);
    float acc[NT] = {0.0f};
    for (int k = lane; k < K/4; k += 64) {
        const float4 w = w4[k];
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4 v = ((const float4 *) (x + t*x_st))[k];
            acc[t] += w.x*v.x + w.y*v.y + w.z*v.z + w.w*v.w;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
#pragma unroll
        for (int off = 32; off > 0; off >>= 1) {
            acc[t] += __shfl_xor(acc[t], off, 64);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            if (is_b) {
                beta[h + t*b_st] = 1.0f/(1.0f + expf(-acc[t]));
            } else {
                const float v  = acc[t] + dt[h];
                const float sp = v > 20.0f ? v : logf(1.0f + expf(v)); // as ggml softplus
                gate[h + t*g_st] = sp*A[h];
            }
        }
    }
}

// nodes: mm_a, (reshape), add, softplus, mul, (reshape), mm_b, (reshape), sigmoid
bool ggml_cuda_gdn_ab_dec_supported(int cc, const ggml_tensor * mm_a, const ggml_tensor * add, const ggml_tensor * sp,
                                    const ggml_tensor * mul, const ggml_tensor * mm_b, const ggml_tensor * sig) {
    static const int enabled = gdn_fuse_env("GGML_CUDA_GDN_AB_DEC", 0);
    if (!enabled || !GGML_CUDA_CC_IS_GCN(cc)) {
        return false;
    }
    const ggml_tensor * x  = mm_a->src[1];
    const ggml_tensor * wa = mm_a->src[0];
    const ggml_tensor * wb = mm_b->src[0];
    const ggml_tensor * dt = add->src[1];
    const ggml_tensor * A  = mul->src[0] == sp ? mul->src[1] : mul->src[0];
    const int64_t nt = x->ne[1];
    return mm_b->src[1] == x && x->type == GGML_TYPE_F32 && wa->type == GGML_TYPE_F32 && wb->type == GGML_TYPE_F32 &&
        nt >= 1 && nt <= 4 && x->ne[2] == 1 && x->ne[3] == 1 && x->nb[0] == sizeof(float) && x->ne[0] % 4 == 0 &&
        (x->nb[1] % 16) == 0 && ((uintptr_t) x->data) % 16 == 0 &&
        ggml_is_contiguous(wa) && ggml_is_contiguous(wb) && ggml_are_same_shape(wa, wb) && wa->ne[0] == x->ne[0] &&
        wa->ne[2] == 1 && ((uintptr_t) wa->data) % 16 == 0 && ((uintptr_t) wb->data) % 16 == 0 &&
        add->src[0]->view_src == mm_a && ggml_is_contiguous(add) && ggml_is_contiguous(mm_a) &&
        dt->type == GGML_TYPE_F32 && ggml_nelements(dt) == wa->ne[1] && ggml_is_contiguous(dt) &&
        sp->src[0] == add && ggml_get_unary_op(sp) == GGML_UNARY_OP_SOFTPLUS &&
        (mul->src[0] == sp || mul->src[1] == sp) && A->type == GGML_TYPE_F32 && ggml_nelements(A) == wa->ne[1] &&
        ggml_is_contiguous(A) && ggml_is_contiguous(mul) && ggml_nelements(mul) == wa->ne[1]*nt &&
        ggml_get_unary_op(sig) == GGML_UNARY_OP_SIGMOID && sig->src[0]->view_src == mm_b && ggml_is_contiguous(mm_b) &&
        ggml_is_contiguous(sig) && ggml_nelements(sig) == wa->ne[1]*nt && mul->type == GGML_TYPE_F32 && sig->type == GGML_TYPE_F32;
}

void ggml_cuda_gdn_ab_dec(ggml_backend_cuda_context & ctx, const ggml_tensor * mm_a, const ggml_tensor * add, const ggml_tensor * mm_b,
                          ggml_tensor * gate, ggml_tensor * sig) {
    const ggml_tensor * x  = mm_a->src[1];
    const ggml_tensor * wa = mm_a->src[0];
    const ggml_tensor * wb = mm_b->src[0];
    const ggml_tensor * sp = gate->src[0]->op == GGML_OP_UNARY ? gate->src[0] : gate->src[1];
    const ggml_tensor * A  = gate->src[0] == sp ? gate->src[1] : gate->src[0];
    const int K = (int) x->ne[0], H = (int) wa->ne[1], nt = (int) x->ne[1];
    const dim3 grid((2*H + 3)/4);
    cudaStream_t stream = ctx.stream();
#define GDN_AB(NT) gdn_ab_dec<NT><<<grid, 256, 0, stream>>>((const float *) x->data, (const float *) wa->data, (const float *) wb->data, \
        (const float *) add->src[1]->data, (const float *) A->data, (float *) gate->data, (float *) sig->data, K, H, \
        x->nb[1]/sizeof(float), wa->nb[1]/sizeof(float), wb->nb[1]/sizeof(float), H, H)
    switch (nt) {
        case 1:  GDN_AB(1); break;
        case 2:  GDN_AB(2); break;
        case 3:  GDN_AB(3); break;
        default: GDN_AB(4); break;
    }
#undef GDN_AB
    CUDA_CHECK(cudaGetLastError());
}
