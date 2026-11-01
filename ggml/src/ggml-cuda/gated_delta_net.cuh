#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1 or ids != nullptr)
    // ring mode (see ggml_cuda_try_gdn_cache_scatter): snapshot slot s of sequence j lands in cache
    // row ids[j + n_seqs*s]; slot_stride is unused (0) when ids is set. nullptr = plane-indexed path.
    const int32_t * ids = nullptr;
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// gate check for the recurrent-state GET_ROWS direct-read (graph matcher in ggml-cuda.cu):
// the decode GDN reads its state row from the cache via the GET_ROWS ids. Gate GGML_CUDA_GDN_STATE_ROWS_OK, default OFF.
bool ggml_cuda_gdn_state_rows_ok(const ggml_tensor * get_rows, const ggml_tensor * gdn);
