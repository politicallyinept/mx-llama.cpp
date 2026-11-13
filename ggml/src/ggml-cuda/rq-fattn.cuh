#pragma once
#include "common.cuh"

// taskJ1 port of ruquant's q4_0 KV attention at decode / verify widths with int4 x int4 Q.K (rq-fattn.cu):
// v_dot8 on gfx906. Env GGML_CUDA_FA_DOT8: unset/0 off (fork default), 1 int4 Q, 2 int8 Q as two int4
// planes. GGML_CUDA_FA_DOT8_CHUNKS optionally overrides the KV chunk count. gfx906 only.
bool ggml_cuda_rq_fattn_q4_applies(const ggml_tensor * dst, int device);
void ggml_cuda_rq_fattn_q4(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
