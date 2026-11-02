#include "gdn-conv-scatter.cuh"

// One grid.y row per (snap, seq): copy the k1*C window of conv_input starting at column
// (n_new - snap) straight into ring row ids[snap][seq] of conv_states_all.
//
// The staging chain this replaces was:
//   window = view_3d(conv_input, [k1, C, n_seqs], offs = (n_new - snap) * 4)   // strided view
//   rows   = reshape_2d(cont(window), row_count, n_seqs)                       // staging copy
//   set_rows(conv_states_all, rows, ids[snap])                                  // scatter launch
// cont(window) lays the window out as row-major over (t fastest, then c), so scattered row
// element i corresponds to window (c = i / k1, t = i % k1), i.e. conv_input column (n_new - snap + t)
// of channel c. Pure float copy: no arithmetic, no reordering => byte-identical contents.
static __global__ void k_gdn_conv_scatter(const float * __restrict__ src,
                                          float * __restrict__ dst,
                                          const int32_t * __restrict__ i0,
                                          const int32_t * __restrict__ i1,
                                          const int32_t * __restrict__ i2,
                                          const int32_t * __restrict__ i3,
                                          const int32_t * __restrict__ i4,
                                          const int32_t * __restrict__ i5,
                                          const int32_t * __restrict__ i6,
                                          const int32_t * __restrict__ i7,
                                          const int n_written,
                                          const int n_seqs,
                                          const int T,
                                          const int k1,
                                          const int C,
                                          const int n_new,
                                          const int row_count) {
    ggml_cuda_pdl_sync();

    const int r = blockIdx.y;
    GGML_UNUSED(n_written);

    const int snap = r / n_seqs;
    const int j    = r % n_seqs;

    const int32_t * ids = snap == 0 ? i0 : snap == 1 ? i1 : snap == 2 ? i2 : snap == 3 ? i3
                        : snap == 4 ? i4 : snap == 5 ? i5 : snap == 6 ? i6 : i7;

    const int64_t row  = ids[j];
    const float * srow = src + (int64_t) j * C * T + (n_new - snap);
    float *       drow = dst + row * row_count;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < row_count; i += gridDim.x * blockDim.x) {
        const int c = i / k1;
        drow[i] = srow[(int64_t) c * T + (i - c * k1)];
    }
}

void ggml_cuda_gdn_conv_scatter(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_scatter_args & a) {
    GGML_ASSERT(a.src != nullptr && a.dst != nullptr);
    GGML_ASSERT(a.n_written >= 1 && a.n_written <= GDN_CONV_SCATTER_MAX_SNAPS);
    GGML_ASSERT(a.n_seqs >= 1 && a.k1 >= 1 && a.C > 0);
    GGML_ASSERT(a.row_count == a.k1 * a.C);
    GGML_ASSERT(a.T == a.k1 + a.n_new);
    GGML_ASSERT(((uintptr_t) a.src) % 16 == 0 && ((uintptr_t) a.dst) % 16 == 0);
    for (int s = 0; s < a.n_written; ++s) {
        GGML_ASSERT(a.ids[s] != nullptr);
    }

    const int block_dims_x = 256;
    dim3 block_dims(block_dims_x);
    dim3 grid_dims((a.row_count + block_dims_x - 1) / block_dims_x, a.n_written * a.n_seqs, 1);

    cudaStream_t stream = ctx.stream();

    k_gdn_conv_scatter<<<grid_dims, block_dims, 0, stream>>>(
        a.src, a.dst,
        a.n_written > 0 ? a.ids[0] : nullptr,
        a.n_written > 1 ? a.ids[1] : nullptr,
        a.n_written > 2 ? a.ids[2] : nullptr,
        a.n_written > 3 ? a.ids[3] : nullptr,
        a.n_written > 4 ? a.ids[4] : nullptr,
        a.n_written > 5 ? a.ids[5] : nullptr,
        a.n_written > 6 ? a.ids[6] : nullptr,
        a.n_written > 7 ? a.ids[7] : nullptr,
        a.n_written, a.n_seqs, a.T, a.k1, a.C, a.n_new, a.row_count);
}
