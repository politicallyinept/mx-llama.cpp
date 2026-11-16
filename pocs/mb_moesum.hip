// taskL2 microbench: cost of the MoE weighted-sum chain (1 mul + (k-1) adds over
// [n_embd x k x W] fp32) vs a single fused weighted-sum kernel.
// Shapes: n_embd=2048, k=8, W in {1,5}. Reports per-forward cost x41 layers.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { auto e=(x); if(e!=hipSuccess){printf("ERR %s @%d\n",hipGetErrorString(e),__LINE__);exit(1);} } while(0)

__global__ void mul_k(const float* __restrict__ in, const float* __restrict__ w, float* __restrict__ out, int n, int slice) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) { int s = i / slice; out[i] = __fmul_rn(in[i], w[s]); }
}
__global__ void add_k(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ o, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) o[i] = __fadd_rn(a[i], b[i]);
}
// fused: out[t*ne + e] = fold_s( out_prev=0; + __fmul_rn(w[s*t], slice[s][t*ne+e]) )
__global__ void fused_sum(const float* __restrict__ slices, const float* __restrict__ w,
                          float* __restrict__ out, int ne, int W, int k) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int total = ne*W;
    if (i >= total) return;
    int t = i / ne, e = i % ne;
    float acc = 0.0f;
    for (int s = 0; s < k; ++s)
        acc = __fadd_rn(acc, __fmul_rn(w[s*W+t], slices[((size_t)s*W + t)*ne + e]));
    out[i] = acc;
}

template <typename F>
float bench(F f, int iters) {
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    f(); hipDeviceSynchronize();
    hipEventRecord(a);
    for (int i = 0; i < iters; ++i) f();
    hipEventRecord(b); hipEventSynchronize(b);
    float ms; hipEventElapsedTime(&ms, a, b);
    return ms/iters*1000.0f; // us
}

int main() {
    const int NE = 2048, K = 8;
    for (int W : {1, 5}) {
        int slice = NE*W, total = K*slice;
        float *in, *w, *tmp, *out;
        CK(hipMalloc(&in, total*4)); CK(hipMalloc(&w, K*W*4));
        CK(hipMalloc(&tmp, total*4)); CK(hipMalloc(&out, slice*4));
        CK(hipMemset(in, 0, total*4)); CK(hipMemset(w, 0, K*W*4));
        int thr = 256;
        auto chain = [&]{
            mul_k<<<(total+thr-1)/thr, thr>>>(in, w, tmp, total, slice);
            float* acc = tmp; // slot0
            for (int s = 1; s < K; ++s) {
                add_k<<<(slice+thr-1)/thr, thr>>>(acc, tmp + (size_t)s*slice, tmp /*fold into slot0 region*/, slice);
                acc = tmp;
            }
        };
        // more faithful: adds write into tmp[0..slice) reading tmp[0..slice) + tmp[s*slice..]
        auto fused = [&]{ fused_sum<<<(slice+thr-1)/thr, thr>>>(in, w, out, NE, W, K); };
        int iters = 2000;
        float tc = bench(chain, iters), tf = bench(fused, iters);
        printf("W=%d  chain(1 mul + %d adds) = %7.2f us   fused = %7.2f us   saving = %7.2f us  (x41 layers = %6.1f us/forward)\n",
               W, K-1, tc, tf, tc-tf, (tc-tf)*41/1000.0f);
        hipFree(in); hipFree(w); hipFree(tmp); hipFree(out);
    }
    return 0;
}
