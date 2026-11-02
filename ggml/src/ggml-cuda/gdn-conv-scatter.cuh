#pragma once

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include "common.cuh"

// ring-mode conv-state direct scatter: the concat (conv_input) that feeds the causal conv is also
// the source of the per-snapshot conv-state rows scattered into the ring cache via ggml_cont +
// ggml_set_rows. This kernel writes those rows straight from the concat buffer (same floats, same
// rows as the staging chain it replaces => byte-identical by construction).
// See ggml_cuda_try_gdn_conv_scatter in ggml-cuda.cu for the matcher and admission gates.

#define GDN_CONV_SCATTER_MAX_SNAPS 8

struct ggml_cuda_gdn_conv_scatter_args {
    // src: concat output, contiguous [k1 + n_new, C, n_seqs] f32 (dim0 = time)
    // dst: conv_states_all cache, [row_count, n_cells] f32, row stride = row_count
    const float *     src;
    float *           dst;
    const int32_t *   ids[GDN_CONV_SCATTER_MAX_SNAPS]; // per-snap row ids, each ne0 == n_seqs
    int               n_written; // number of snapshot chains matched (1..MAX)
    int               n_seqs;
    int               T;         // concat ne0 == k1 + n_new
    int               k1;        // conv_kernel_size - 1 (window width)
    int               C;         // conv_channels (concat ne1)
    int               n_new;     // new-token columns appended by the concat
    int               row_count; // k1 * C floats per scattered row
};

void ggml_cuda_gdn_conv_scatter(ggml_backend_cuda_context & ctx, const ggml_cuda_gdn_conv_scatter_args & args);
