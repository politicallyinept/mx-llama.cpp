// Dense (non-MoE) entry point for repacked Q8_0 weights: quantize src1, then dispatch
// the mat-vec (single token) or tiled GEMM (multi-token) path per 2D slice.
#include "repack.cuh"
#include "repack-common.cuh"
#include "repack-kernels.cuh"
#include "../quantize.cuh"
#include "../mmq.cuh"
#include "../w4a4_skip.cuh"   // Lever 2: GGML_CUDA_W4A4_SKIP prefix exclusion

static void ggml_cuda_mul_mat_repacked_slice(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const uint8_t * w, const block_q8_1 * xq,
        float * dst_d, int64_t ne00, int64_t ne01, int64_t ne11,
        cudaStream_t stream, bool u4act = false, uint32_t dst_s1 = 0,
        bool u4p8 = false, bool u4x16 = false, bool u4q41 = false);
template <ggml_type WT, bool U4ACT = false, bool U4P8 = false, bool U4Q41 = false>
static void ggml_cuda_mul_mat_repacked_nc_t(
        const uint8_t * w, const block_q8_1 * xq, float * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne11,
        const uint32_t xs, const uint32_t ys, cudaStream_t stream);

void ggml_cuda_mul_mat_repacked(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    GGML_ASSERT(ne10 == ne00);

    cudaStream_t stream = ctx.stream();

    // Views are re-packed into a persistent cache buffer (pool allocs are invalid
    // inside CUDA graph capture).
    const uint8_t * w;
    if (src0->view_src != nullptr && ggml_cuda_repack_tensor_supported(src0->view_src)) {
        w = repack_view_get_cached(src0, src0->view_src, stream);
    } else {
        w = (const uint8_t *) src0->data;
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;

    // W4A4 milestone 3: dense narrow-batch decode through the dp8a u4-activation
    // kernels. Function-local static, fork pattern (mirrors GGML_RP_GENERIC_MMV
    // in the slice dispatch below). Unset gate = stock q8_1-activation path:
    // every u4 branch is inside `if (u4act)`, so gate-off behavior is unchanged.
    // C1/B8 M4 GATE REDEFINITION (read with atoi):
    //   unset/0 = stock q8_1 activations
    //   1       = per-32 asymmetric u4 decode + GEMM (shipped; byte-identical)
    //   2       = decode per-32 (SAME as gate 1) + GEMM per-8 u4p
    //   3       = Phase-2 per-16 (REDEFINITION of the old batch-gated value 3,
    //           which was validated in BENCH.md §43 and is retired here):
    //           decode per-32 u4 (SAME as gates 1/2) + GEMM per-16 u4x16.
    //           Gate-2 semantics for dispatch: only the tiled GEMM path
    //           changes; the narrow/decode paths keep the per-32 u4act kernels.
    //           Rationale: channel-persistent activation outliers crush the
    //           per-32 blocks (phase-1 diagnostic); per-16 sub-block scales
    //           halve the scale-plane bandwidth of the per-8 u4p (gate 2)
    //           while recovering most of its KLD gap closure.
    //   4       = SUPERSET of gate 1: per-32 u4 on Q4_0 weights (identical to
    //           gate 1, same u4act predicates below) PLUS the Q4_1 u4 (dp8a)
    //           path on Q4_1 weights. A Q4_1-only file is a degenerate case:
    //           real Q4_1-heavy files are ~97.6% Q4_0 with a small ffn_down
    //           Q4_1 guard band (~2.4%), so gate 4 must carry the Q4_0 u4 path
    //           too or the overwhelming majority of MACs would run stock.
    //           Modes 2/3 semantics stay EXCLUSIVE: u4p (per-8 GEMM) and
    //           u4x16 (per-16 GEMM) never engage at mode 4.
    // The per-8 u4p DECODE branch (M3) stays compiled but NO gate selects it:
    // BENCH.md §36 rejected per-8 decode (-9.2% tg). `u4p` below is GEMM-only;
    // the narrow-batch decode sites use `u4act` under both gates 1 and 2.
    // Gate=0/1 stay byte-identical to the shipped states.
    static const int w4a4_dot8 = [] {
        const char * e = getenv("GGML_CUDA_W4A4_DOT8");
        const int mode = e != nullptr ? atoi(e) : 0;
        if (mode != 0) {
            // Audit: 'gate ACTIVE' alone cannot distinguish floor-on from
            // floor-off runs, so the marker prints the RESOLVED floor. Read
            // the same way the floor lambda below reads it (unset/0 = off).
            const char * e_floor = getenv("GGML_CUDA_W4A4_MIN_TOKENS");
            const long long floor_n = e_floor != nullptr ? atoll(e_floor) : 0;
            GGML_LOG_WARN("CUDA w4a4: GGML_CUDA_W4A4_DOT8=%d gate ACTIVE (%s; GGML_CUDA_W4A4_MIN_TOKENS=%lld)\n", mode,
                          mode == 1 ? "per-32 u4 activations" :
                          mode == 2 ? "per-32 u4 decode + per-8 u4p GEMM" :
                          mode == 3 ? "per-16 u4x16 GEMM (decode stays per-32 u4)" :
                          mode == 4 ? "SUPERSET of gate 1: per-32 u4 on Q4_0 + Q4_1 u4 (dp8a) on Q4_1 (repack admission + u4 decode)" :
                                      "unrecognised mode: no W4A4 dispatch",
                          floor_n);
        }
        return mode;
    }();
    // W4A4 WIDTH FLOOR (GGML_CUDA_W4A4_MIN_TOKENS): the gate1-vs-stock prefill
    // crossover (BENCH.md §61, /root/bench-results/gate-crossover/) shows the
    // u4act path LOSES to stock q8_1 below ne11 ~128 (pp4 -47.5%, pp8 -65.7%,
    // pp16-64 -15..-20%) and wins +27..+32% at ne11 >= 128. The loss is
    // thread-count-independent (t=32 vs t=128 ratios identical,
    // gate-crossover-threads/). MTP speculative verify runs at ne11 ~= n_max+1
    // = 5 rows, squarely in the losing zone, so gate1+MTP regresses decode
    // ~10% vs gate1 no-MTP (BENCH.md §59) even though accept-rate is intact.
    // This floor makes the gate AUTO-DISABLE below the floor: the u4act
    // predicate gains `ne11 >= min_tokens`, so narrow-width calls (decode,
    // MTP verify, small chunks) fall back to the stock q8_1 activation path
    // per-GEMM-call, while wide prefill keeps the full gate win. Unset/0 =
    // floor OFF: behavior is byte-identical to the shipped gates (the added
    // clause is constant-true). Same-site precedent: ksplit checks
    // 16 <= ne11 <= 64 right here. The MoE dispatch (mul-mat-id.cu) reads
    // the same knob with the same semantics. Repack admission is weight-side and
    // unaffected: flopped calls run the REPACKED stock path (q8_1 activations),
    // exactly like a GGML_CUDA_W4A4_SKIP tensor.
    static const int64_t w4a4_min_tokens = [] {
        const char * e = getenv("GGML_CUDA_W4A4_MIN_TOKENS");
        const int64_t n = e != nullptr ? atoll(e) : 0;
        if (n > 0) {
            GGML_LOG_WARN("CUDA w4a4: GGML_CUDA_W4A4_MIN_TOKENS=%lld floor ACTIVE (u4act falls back to stock q8_1 below this batch width)\n",
                          (long long) n);
        }
        return n;
    }();
    // DEBUG (perf/w4a4-attrib): GGML_W4A4_ONLY=attn|ffn restricts the u4act
    // path to one tensor class, for KLD attribution (attn-linears vs FFN).
    // 'attn' matches *attn* (attn_qkv/attn_o/attn_q/k/v/attn_gate), 'ffn'
    // matches *ffn*; every other Q4_0 linear (ssm_*) stays on the stock path
    // in both variants. Unset = no filter (shipped behavior).
    static const int w4a4_only = [] {
        const char * e = getenv("GGML_W4A4_ONLY");
        if (e == nullptr) return 0;
        if (strstr(e, "attn") != nullptr) return 1;
        if (strstr(e, "ffn")  != nullptr) return 2;
        return 0;
    }();
    // DEBUG (perf/w4a4-selective): tensor-class match for the filter.
    auto w4a4_match = [&](const ggml_tensor * t) {
        return !w4a4_only || strstr(t->name, w4a4_only == 1 ? "attn" : "ffn") != nullptr;
    };
    // DEBUG (perf/w4a4-selective): under gates 1/2/4 the filter restricts WHICH
    // tensors use the u4act path at all (phase-1 attribution vs stock). Under
    // gate 3 the filter's meaning MOVES: it selects which tensors get per-16
    // u4x16 GEMM activations; every Q4_0 tensor keeps the per-32 u4act base
    // (non-matching tensors fall back to the u4act per-32 GEMM path), so the
    // selective run is measured against gate1-full, not against stock.
    // Lever 2 (GGML_CUDA_W4A4_SKIP): prefix-excluded tensors drop out of u4act
    // entirely and run the stock q8_1 activation path (W4A8 semantics,
    // bitwise-stock for those GEMMs). Applies under EVERY gate, including
    // gate 3: an excluded tensor loses the per-16 u4x16 GEMM too (it is not a
    // u4act tensor at all). ONLY applies first, then SKIP subtracts.
    // Gate unset/empty: ggml_cuda_w4a4_skip() is a constant false (empty
    // static vector), so gate-off dispatch is unchanged.
    const bool w4a4_skip = ggml_cuda_w4a4_skip(src0->name);
    // Gate 4 is a SUPERSET of gate 1 for Q4_0: the per-32 u4act path fires at
    // modes 1/2/3/4 (the GGML_W4A4_ONLY filter keeps its gate-1 meaning at
    // mode 4 — only mode 3 moves the filter to the u4x16 selection). The Q4_1
    // half of gate 4 is the u4q41 predicate below; the two are disjoint by the
    // src0->type test (Q4_0 vs Q4_1), so no tensor ever takes both paths.
    const bool u4act = ((w4a4_dot8 == 1 || w4a4_dot8 == 2 || w4a4_dot8 == 3 || w4a4_dot8 == 4)
        && src0->type == GGML_TYPE_Q4_0)
        && (w4a4_dot8 == 3 || w4a4_match(src0))
        && !w4a4_skip
        // Width floor: below the floor this call runs the stock q8_1 activation
        // path (see w4a4_min_tokens above). min_tokens<=0 (default) => always
        // true => shipped gate behavior unchanged.
        && (w4a4_min_tokens <= 0 || ne11 >= w4a4_min_tokens);
    // Mode semantics are EXCLUSIVE here: per-8 u4p is gate-2-only, per-16
    // u4x16 is gate-3-only — neither may fire at mode 4 (gate 4 = gate-1
    // per-32 Q4_0 semantics + Q4_1 u4).
    const bool u4p   = w4a4_dot8 == 2 && src0->type == GGML_TYPE_Q4_0 && !w4a4_skip;
    // Phase-2 gate 3: per-16 GEMM activations (gate-2 semantics: GEMM path
    // only; the narrow/decode paths above keep the per-32 u4act kernels).
    // DEBUG (perf/w4a4-selective): the GGML_W4A4_ONLY filter now gates this
    // selection too. The dispatch is per-tensor (this whole function runs per
    // mul_mat node, each with its own pool alloc), so a selective run mixes
    // u4x16 and u4act pools across tensors with no cross-contamination; the
    // plane-offset fix is per-call.
    const bool u4x16 = w4a4_dot8 == 3 && src0->type == GGML_TYPE_Q4_0 && w4a4_match(src0) && !w4a4_skip;
    // Q4_1 u4 (q41-u4-probe): DOT8=4 selects the Q4_1 dp8a path on Q4_1 weights
    // (gate 4 is a SUPERSET: the Q4_0 half is the u4act per-32 path above, so a
    // mostly-Q4_0 file with a Q4_1 guard band gets u4 on BOTH classes).
    // The SAME env value drives the repack admission (ggml_cuda_repack_tensor_supported
    // admits Q4_1 only at DOT8=4), so admission and decode can never disagree: a
    // Q4_1 tensor either runs the repack fast path (gate on) or is not repacked at
    // all and runs the canonical stock path (gate off) — never repacked-but-stock-decoded.
    // A skipped tensor (GGML_CUDA_W4A4_SKIP) under gate 4 runs the REPACKED STOCK
    // path (q8_1 activations, affine epilogue) — same semantics as SKIP under gates 1-3.
    const bool u4q41 = w4a4_dot8 == 4 && src0->type == GGML_TYPE_Q4_1 && !w4a4_skip;

    if (ne11 >= 1 && ne11 <= MMQ_RP_Q8_MMV_MAX_TOKENS) {
        // One token, or a narrow batch: plain Q8_1 rows, one weight pass. The
        // tiled path below computes a full 32-wide tile whatever the batch is.
        //
        // C1/B8 M4: the u4p DECODE branch below is PARKED — no gate selects it
        // (BENCH.md §36: per-8 decode REJECTED, -9.2% tg). It stays compiled
        // for reference; `u4p_decode` is a constant false so the branch is
        // unreachable at runtime while the template instantiation survives.
        // Gate=2 decode runs the per-32 u4act path (same as gate=1).
        const bool u4p_decode = false;
        ggml_cuda_pool_alloc<block_q8_1> src1_q8_1_own;
        ggml_cuda_pool_alloc<block_q8_1_u4> src1_u4_own(ctx.pool());
        ggml_cuda_pool_alloc<block_q8_1_u4p> src1_u4p_own(ctx.pool());
        block_q8_1 * src1_q8_1_d = nullptr;
        {
            const int64_t s11 = src1->nb[1] / sizeof(float);
            const int64_t s12 = src1->nb[2] / sizeof(float);
            const int64_t s13 = src1->nb[3] / sizeof(float);
            if (u4p_decode) {
                // C1/B8 M3 ALLOC MATH: block_q8_1_u4p is 32 B — the SAME size
                // as block_q8_1 — and the BLOCK COUNT is identical too
                // (ne10_padded/32 per token row = x_stride), so xs keeps its
                // "blocks per token column" meaning and the kernel's
                // `c*xs + sb` indexing is unchanged. Bypasses the q8_1
                // activation cache for the same reason the u4 path does:
                // different layout, must not share cache entries.
                src1_u4p_own.alloc((size_t) (ne13 * ne12 * ne11 * x_stride));
                quantize_mmq_q8_1_u4p_cuda((const float *) src1->data, src1_u4p_own.get(),
                    ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
            } else if (u4act || u4q41) {
                // ALLOC MATH: block_q8_1_u4 is 20 B vs block_q8_1's 36 B, but
                // the BLOCK COUNT is identical (ne10_padded/32 per token row =
                // x_stride), so xs keeps its "blocks per token column" meaning
                // and the kernel's `c*xs + sb` indexing is unchanged -- only the
                // element size differs with the type. Bypasses the q8_1
                // activation cache (variant 0 is the q8_1 layout; u4 blocks
                // are a different layout and must not share those entries).
                src1_u4_own.alloc((size_t) (ne13 * ne12 * ne11 * x_stride));
                quantize_mmq_q8_1_u4_cuda((const float *) src1->data, src1_u4_own.get(),
                    ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream, src0->name);
            } else {
                src1_q8_1_d = repack_quantize_src1_q8_1(ctx, src0, src1, ne10, ne10_padded,
                    s11, s12, s13, ne11, ne12, ne13,
                    (size_t) (ne13 * ne12 * ne11 * x_stride), src1_q8_1_own, stream);
            }
        }
        const uint32_t dst_s1 = dst->nb[1] / sizeof(float);
        for (int64_t i3 = 0; i3 < ne13; i3++) {
        for (int64_t i2 = 0; i2 < ne12; i2++) {
            // PLANE OFFSETS MUST BE COMPUTED IN THE POOL'S OWN ELEMENT SIZE.
            // The block COUNT per plane is identical under every layout, but
            // the element SIZE is not: block_q8_1 is 36 B, block_q8_1_u4 is
            // 20 B, block_q8_1_u4p is 32 B. Adding the plane offset in
            // block_q8_1 units to a u4/u4p pool mis-places every plane after
            // the first (ne12 > 1): garbage reads (NaN logits) or reads past
            // the pool (illegal access). ne12 == 1 has offset 0 and was
            // always clean, which hid this until multi-sequence shapes ran.
            const block_q8_1 * xq = (u4p_decode
                    ? reinterpret_cast<const block_q8_1 *>(
                        (const char *) src1_u4p_own.get()
                        + (size_t) (i3 * ne12 + i2) * ne11 * x_stride * sizeof(block_q8_1_u4p))
                    : (u4act || u4q41)
                    ? reinterpret_cast<const block_q8_1 *>(
                        (const char *) src1_u4_own.get()
                        + (size_t) (i3 * ne12 + i2) * ne11 * x_stride * sizeof(block_q8_1_u4))
                    : src1_q8_1_d + (i3 * ne12 + i2) * ne11 * x_stride);
            float * dst_d = (float *)((char *) dst->data + i3 * dst->nb[3] + i2 * dst->nb[2]);
            if (ne11 == 1) {
                if (u4p_decode) {
                    // C1/B8 M3 single-token decode: u4p nc kernel at NCOLS=1,
                    // same 2-row / RPL=2 / 64-lane shape as the per-32 twin.
                    const dim3 grid((ne01 + 1) / 2, 1, 1);
                    mul_mat_vec_repacked_nc<2, 1, 1, 2, 64, GGML_TYPE_Q4_0, false, true>
                        <<<grid, 64, 0, stream>>>(w, xq, dst_d,
                            (uint32_t) ne00, (uint32_t) ne01, (uint32_t) x_stride, dst_s1);
                } else if (u4act) {
                    // Single-token decode: the nc kernel at NCOLS=1, same
                    // 2-row / RPL=2 / 64-lane shape as widths 4-8. Stock
                    // single-token kernels stay for every other weight type.
                    const dim3 grid((ne01 + 1) / 2, 1, 1);
                    mul_mat_vec_repacked_nc<2, 1, 1, 2, 64, GGML_TYPE_Q4_0, true>
                        <<<grid, 64, 0, stream>>>(w, xq, dst_d,
                            (uint32_t) ne00, (uint32_t) ne01, (uint32_t) x_stride, dst_s1);
                } else if (u4q41) {
                    // Q4_1 u4 single-token decode: same nc shape, U4Q41 variant.
                    const dim3 grid((ne01 + 1) / 2, 1, 1);
                    mul_mat_vec_repacked_nc<2, 1, 1, 2, 64, GGML_TYPE_Q4_1, false, false, false, true>
                        <<<grid, 64, 0, stream>>>(w, xq, dst_d,
                            (uint32_t) ne00, (uint32_t) ne01, (uint32_t) x_stride, dst_s1);
                } else {
                    ggml_cuda_mul_mat_repacked_slice(ctx, src0, w, xq, dst_d,
                        ne00, ne01, ne11, stream);
                }
            } else {
                switch (src0->type) {
                    case GGML_TYPE_Q8_0:
                        ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_Q8_0>(w, xq, dst_d,
                            ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        break;
                    case GGML_TYPE_MXFP4:
                        ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_MXFP4>(w, xq, dst_d,
                            ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        break;
                    case GGML_TYPE_IQ4_NL:
                        ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_IQ4_NL>(w, xq, dst_d,
                            ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        break;
                    case GGML_TYPE_Q4_0:
                        if (u4p_decode) {
                            ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_Q4_0, false, true>(w, xq, dst_d,
                                ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        } else if (u4act) {
                            ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_Q4_0, true>(w, xq, dst_d,
                                ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        } else {
                            ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_Q4_0>(w, xq, dst_d,
                                ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        }
                        break;
                    case GGML_TYPE_Q4_1:
                        if (u4q41) {
                            ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_Q4_1, false, false, true>(w, xq, dst_d,
                                ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        } else {
                            ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_Q4_1>(w, xq, dst_d,
                                ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        }
                        break;
                    case GGML_TYPE_Q5_1:
                        ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_Q5_1>(w, xq, dst_d,
                            ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        break;
                    case GGML_TYPE_Q5_K:
                        ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_Q5_K>(w, xq, dst_d,
                            ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        break;
                    case GGML_TYPE_Q4_K:
                        ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_Q4_K>(w, xq, dst_d,
                            ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        break;
                    case GGML_TYPE_Q6_K:
                        ggml_cuda_mul_mat_repacked_nc_t<GGML_TYPE_Q6_K>(w, xq, dst_d,
                            ne00, ne01, ne11, (uint32_t) x_stride, dst_s1, stream);
                        break;
                    default: GGML_ABORT("unsupported repack type");
                }
            }
        }
        }
        return;
    }

    // Multi-token: grouped MMQ Q8_1 layout for the tiled GEMM.
    // G3 (W4A4 GEMM): under the u4act gate the activation pool is the
    // token-major block_q8_1_u4 layout instead. ALLOC MATH: the stock pool
    // holds 4*QK8_1 values per group at 144 B/group = 36 B per 32-value
    // block-equivalent; block_q8_1_u4 holds one 32-value block at 20 B.
    // Block COUNT per token is identical, only the element size differs --
    // same mirror as the dense m3 narrow path above. NOTE the stride
    // difference vs the narrow path: the GEMM kernel derives the per-token
    // stride internally as n_sub = ne00/32 (it never receives xs), so the
    // u4 quantizer is invoked with ne0 = ne00, NOT ne10_padded. Q4_0
    // weights guarantee ne00 % 32 == 0. The stock k-major layout hides
    // this because its padding sits at the tail of k and the kernel stops
    // at n_sub slabs; the token-major u4 layout bakes the stride in.
    const uint64_t n_groups   = (uint64_t) ne10_padded / (4 * QK8_1);
    const uint64_t nkb_u4     = (uint64_t) ne00 / 32;

    int64_t chunk_ne11 = ctx.repack_workspace_cols_cap > 0 ?
        std::min(ne11, ctx.repack_workspace_cols_cap) : ne11;
    ggml_cuda_pool_alloc<block_q8_1_mmq_h> src1_q8_1(ctx.pool());
    ggml_cuda_pool_alloc<block_q8_1_u4>    src1_u4(ctx.pool());
    // C1/B8 M4: under gate=2 the GEMM activations are per-8 u4p. ALLOC MATH:
    // block_q8_1_u4p is 32 B/block vs the u4 20 B/block, but the BLOCK COUNT
    // per token is identical (nkb_u4 = ne00/32) — only the element size
    // changes with the type, same mirror as every other layout swap here.
    ggml_cuda_pool_alloc<block_q8_1_u4p>   src1_u4p(ctx.pool());
    // Phase-2 gate 3: per-16 u4x16 GEMM activations. ALLOC MATH: 28 B/block
    // vs the u4p 32 B/block, BLOCK COUNT per token identical (nkb_u4).
    ggml_cuda_pool_alloc<block_q8_1_u4x16> src1_u4x16(ctx.pool());
    auto try_workspace = [&]() {
        if (u4x16) {
            return (void *) src1_u4x16.try_alloc(ne13 * ne12 * chunk_ne11 * nkb_u4);
        }
        if (u4p) {
            return (void *) src1_u4p.try_alloc(ne13 * ne12 * chunk_ne11 * nkb_u4);
        }
        return (u4act || u4q41)
            ? (void *) src1_u4.try_alloc(ne13 * ne12 * chunk_ne11 * nkb_u4)
            : (void *) src1_q8_1.try_alloc(ne13 * ne12 * chunk_ne11 * n_groups);
    };

    if (!try_workspace()) {
        // Cached activation conversions are optional. Their consumers precede
        // this operation on the same stream, so they can be released before a
        // required repack conversion is split.
        if (!ctx.q8_1_cache.empty()) {
            ctx.q8_1_cache.clear();
            if (!ctx.q8_1_cache_pressure_logged) {
                GGML_LOG_WARN("CUDA q8_1 cache[%d]: repack workspace did not fit; releasing cached activations\n",
                              ctx.device);
                ctx.q8_1_cache_pressure_logged = true;
            }
        }

        while (!try_workspace()) {
            if (chunk_ne11 == 1) {
                GGML_ABORT("CUDA repack: failed to allocate the minimum quantization workspace");
            }
            chunk_ne11 = (chunk_ne11 + 1) / 2;
        }
    }

    if (chunk_ne11 != ne11) {
        if (ctx.repack_workspace_cols_cap == 0 || chunk_ne11 < ctx.repack_workspace_cols_cap) {
            ctx.repack_workspace_cols_cap = chunk_ne11;
        }
        if (!ctx.repack_workspace_cols_cap_logged) {
            GGML_LOG_WARN("CUDA repack: device %d workspace pressure, splitting %lld columns into chunks of %lld\n",
                          ctx.device, (long long) ne11, (long long) chunk_ne11);
            ctx.repack_workspace_cols_cap_logged = true;
        }
    }

    const int64_t s11 = src1->nb[1] / sizeof(float);
    const int64_t s12 = src1->nb[2] / sizeof(float);
    const int64_t s13 = src1->nb[3] / sizeof(float);
    const uint32_t dst_s1 = dst->nb[1] / sizeof(float);
    for (int64_t col = 0; col < ne11; col += chunk_ne11) {
        const int64_t iter_ne11 = std::min(chunk_ne11, ne11 - col);
        if (u4x16) {
            // Phase-2 gate 3: token-major u4x16 quantizer (same launcher args
            // as the per-32/per-8 calls below; one 28 B block per (token, 32
            // values), token-major ib = (z*ne1 + x)*nkb + k_block; ne0 = ne00,
            // NOT ne10_padded — see the G3 alloc-math note above).
            quantize_mmq_q8_1_u4x16_cuda((const float *) src1->data + col*s11, src1_u4x16.get(),
                ne10, s11, s12, s13, ne00, iter_ne11, ne12, ne13, stream);
        } else if (u4p) {
            // C1/B8 M4: token-major u4p quantizer (same launcher args as the
            // per-32 u4 GEMM call below — the u4p quantizer emits one 32 B
            // block per (token, 32 values) with the same token-major ib =
            // (z*ne1 + x)*nkb + k_block, so the kernel's [tok*nkb + sb] read
            // matches directly; ne0 = ne00, NOT ne10_padded — see the G3
            // alloc-math note above).
            quantize_mmq_q8_1_u4p_cuda((const float *) src1->data + col*s11, src1_u4p.get(),
                ne10, s11, s12, s13, ne00, iter_ne11, ne12, ne13, stream);
        } else if (u4act || u4q41) {
            // G3: token-major u4 quantizer (same launcher args as the narrow
            // path; the chunk offset rides s11 exactly like the stock call).
            // Q4_1 u4 (u4q41) uses the SAME activation layout — only the weight
            // decode differs.
            quantize_mmq_q8_1_u4_cuda((const float *) src1->data + col*s11, src1_u4.get(),
                ne10, s11, s12, s13, ne00, iter_ne11, ne12, ne13, stream, src0->name);
        } else {
            quantize_mmq_q8_1_cuda((const float *) src1->data + col*s11, nullptr, src1_q8_1.get(),
                src0->type, ne10, s11, s12, s13, ne10_padded, iter_ne11, ne12, ne13, stream);
        }

        for (int64_t i3 = 0; i3 < ne13; i3++) {
        for (int64_t i2 = 0; i2 < ne12; i2++) {
            // Slice offset in BLOCK units: identical COUNT under either
            // layout (nkb per token row vs n_groups*4 per token row), but the
            // element SIZE differs (block_q8_1 34 B vs block_q8_1_u4 20 B vs
            // block_q8_1_u4p 32 B), so the plane offset for u4/u4p must be
            // taken in BYTES of the pool's own element type. Adding it in
            // block_q8_1 units mis-places every plane after the first
            // (ne12 > 1): gate1 reads past the pool (illegal access), gate2
            // reads in-pool garbage (silent corruption, PPL ~300). ne12 == 1
            // has offset 0 and was always clean.
            const block_q8_1 * xq = (u4x16
                    ? reinterpret_cast<const block_q8_1 *>(
                        (const char *) src1_u4x16.get()
                        + (size_t) (i3 * ne12 + i2) * iter_ne11 * nkb_u4 * sizeof(block_q8_1_u4x16))
                    : u4p
                    ? reinterpret_cast<const block_q8_1 *>(
                        (const char *) src1_u4p.get()
                        + (size_t) (i3 * ne12 + i2) * iter_ne11 * nkb_u4 * sizeof(block_q8_1_u4p))
                    : (u4act || u4q41)
                    ? reinterpret_cast<const block_q8_1 *>(
                        (const char *) src1_u4.get()
                        + (size_t) (i3 * ne12 + i2) * iter_ne11 * nkb_u4 * sizeof(block_q8_1_u4))
                    : reinterpret_cast<const block_q8_1 *>(
                        src1_q8_1.get() + (i3 * ne12 + i2) * iter_ne11 * n_groups));
            float * dst_d = (float *)((char *) dst->data + i3 * dst->nb[3] + i2 * dst->nb[2])
                           + col*dst_s1;
            ggml_cuda_mul_mat_repacked_slice(ctx, src0, w, xq, dst_d,
                ne00, ne01, iter_ne11, stream, u4act && !u4p && !u4x16, dst_s1, u4p, u4x16, u4q41);
        }
        }
    }
}

// Narrow batch (2..8 tokens): one multi-column mat-vec pass per slice.
template <ggml_type WT, bool U4ACT, bool U4P8, bool U4Q41>
static void ggml_cuda_mul_mat_repacked_nc_t(
        const uint8_t * w, const block_q8_1 * xq, float * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne11,
        const uint32_t xs, const uint32_t ys, cudaStream_t stream) {
    // Geometry per width, measured on Qwen3.8-27B (pp512 at that ubatch, 4x
    // MI50 -sm tensor). The row count per lane is small once the tensor is
    // split, so scheduling granularity beats per-group reuse and 64-thread
    // workgroups spread over the CUs win. Two rows per lane reuse the
    // activation block across rows and pay from 3 tokens up; at 2 there is
    // too little to amortize, and 4 rows spills.
    // Lane width per row, chosen from the ROW LENGTH and the batch width.
    //
    // LANES is the number of lanes cooperating on one row. The strided loop
    // runs ne0/32/LANES iterations and is followed by a log2(LANES)-step DPP
    // reduction, so the useful knob is how much work a lane gets before it has
    // to reduce. Narrowing the lane group raises that, and raises occupancy,
    // but it also cuts the block count that hides memory latency.
    //
    // Which effect wins is decided by the row length, and two models disagree
    // about the batch width alone, so that is the wrong axis. Measured pp512
    // at the ubatch, 2x MI50 -sm layer, narrow-lane against the 64-lane shape:
    //
    //                       ne0    K-blocks   width 2   width 3
    //   Qwen3.6-27B        5120       160      +13.8%    +2.1%
    //   Qwen3.6-35B-A3B    2048        64       -9.5%    -0.7%
    //
    // A 2048-wide row is 64 K-blocks, so splitting it over 16 lanes leaves 4
    // iterations against a 4-step reduction while cutting the grid by 16x -
    // the row is simply too short to pay for the lost parallelism. Gate on the
    // block count so the choice follows the shape and not the model.
    // Narrowing the lane group raises the work a lane does before it reduces,
    // and multiplies the rows a block covers, which shrinks the grid. Both
    // have a floor and BOTH must hold.
    //
    // A lane accumulates n_blocks/LANES times, then pays a log2(LANES)-step
    // reduction, so require at least four accumulations at the widest group we
    // would narrow to. Measured on two models: 160 and 128 blocks win, 64
    // loses at every lane width, so the break is between them.
    //
    // The grid condition is what the row length alone cannot see - a 48-row
    // tensor is a 2-block grid however long its rows are.
    constexpr int RP_NC_MIN_ACCUM   = 4;
    constexpr int RP_NC_LANES_WIDE  = 32;   // widest group we ever narrow to
    const int     nsm       = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int64_t n_blocks  = ne00 / 32;
    const int64_t min_grid  = 2LL * nsm;
    const bool    work_ok   = n_blocks >= RP_NC_MIN_ACCUM * RP_NC_LANES_WIDE;
    const bool    lanes16_ok = work_ok && (ne01 + 31) / 32 >= min_grid;
    const bool    lanes32_ok = work_ok && (ne01 +  7) /  8 >= min_grid;
    switch (ne11) {
        case 2: {
            if (lanes16_ok) {
                // 16 lanes to a row, 32 rows per block, one row per lane.
                const dim3 grid((ne01 + 31) / 32, 1, 1);
                mul_mat_vec_repacked_nc<32, 8, 2, 1, 16, WT, U4ACT, U4P8, false, U4Q41><<<grid, 512, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
            } else {
                const dim3 grid((ne01 + 1) / 2, 1, 1);
                mul_mat_vec_repacked_nc<2, 2, 2, 1, 64, WT, U4ACT, U4P8, false, U4Q41><<<grid, 128, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
            }
        } break;
        case 3: {
            if (lanes32_ok) {
                // 32 lanes to a row, 8 rows per block, one row per lane.
                const dim3 grid((ne01 + 7) / 8, 1, 1);
                mul_mat_vec_repacked_nc<8, 4, 3, 1, 32, WT, U4ACT, U4P8, false, U4Q41><<<grid, 256, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
            } else {
                const dim3 grid((ne01 + 1) / 2, 1, 1);
                mul_mat_vec_repacked_nc<2, 1, 3, 2, 64, WT, U4ACT, U4P8, false, U4Q41><<<grid, 64, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
            }
        } break;
        case 4: {
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 4, 2, 64, WT, U4ACT, U4P8, false, U4Q41><<<grid, 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
        } break;
        case 5: {
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 5, 2, 64, WT, U4ACT, U4P8, false, U4Q41><<<grid, 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
        } break;
        case 6: {
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 6, 2, 64, WT, U4ACT, U4P8, false, U4Q41><<<grid, 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
        } break;
        case 7: {
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 7, 2, 64, WT, U4ACT, U4P8, false, U4Q41><<<grid, 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
        } break;
        case 8: {
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 8, 2, 64, WT, U4ACT, U4P8, false, U4Q41><<<grid, 64, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, xs, ys);
        } break;
        default: GGML_ABORT("nc mat-vec: unsupported batch width");
    }
}

// Dispatch one 2D slice: mat-vec vs tiled GEMM.
// G3: `u4act` selects the block_q8_1_u4 activation layout (token-major,
// 20 B/block). Only the Q4_0 GEMM arms consume it; the ne11==1 arm under
// u4act routes to the single-token u4 nc kernel (same shape as the narrow
// path) because the stock single-token kernels expect the q8_1 layout.
// M4: `u4p8` selects the block_q8_1_u4p per-8 GEMM variant (32 B/block).
// The caller passes u4act=false together with u4p8=true under gate=2 (the
// two are mutually exclusive in the kernel); the ne11==1 arm never sees
// u4p8 because gate=2 decode is per-32 (u4act) — see the gate comment.
static void ggml_cuda_mul_mat_repacked_slice(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const uint8_t * w, const block_q8_1 * xq,
        float * dst_d, const int64_t ne00, const int64_t ne01, const int64_t ne11,
        cudaStream_t stream, bool u4act, uint32_t dst_s1, bool u4p8, bool u4x16, bool u4q41) {
    if (ne11 == 1) {
        if (u4x16) {
            // Phase-2 gate 3 edge case: workspace pressure shrank a gate-3
            // GEMM chunk to a single token. The pool holds u4x16 blocks, so
            // route through the single-token u4x16 nc kernel (M3 shape,
            // NCOLS=1) with xs = ne00/32 (quantizer invoked with ne0 = ne00).
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 1, 2, 64, GGML_TYPE_Q4_0, false, false, true>
                <<<grid, 64, 0, stream>>>(w, xq, dst_d,
                    (uint32_t) ne00, (uint32_t) ne01,
                    (uint32_t) (ne00 / 32),
                    dst_s1);
            return;
        }
        if (u4p8) {
            // C1/B8 M4 edge case: workspace pressure shrank a gate-2 GEMM
            // chunk to a single token. The pool holds u4p blocks, so route
            // through the single-token u4p nc kernel (M3 shape, NCOLS=1)
            // with xs = ne00/32 (quantizer invoked with ne0 = ne00).
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 1, 2, 64, GGML_TYPE_Q4_0, false, true>
                <<<grid, 64, 0, stream>>>(w, xq, dst_d,
                    (uint32_t) ne00, (uint32_t) ne01,
                    (uint32_t) (ne00 / 32),
                    dst_s1);
            return;
        }
        if (u4act) {
            // xs = blocks per token row. The u4 quantizer for this path was
            // invoked with ne0 = ne00 (see GEMM alloc-math note above), so
            // the pool stride is ne00/32 -- NOT the padded x_stride the
            // narrow path uses. Q4_0 guarantees ne00 % 32 == 0.
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 1, 2, 64, GGML_TYPE_Q4_0, true>
                <<<grid, 64, 0, stream>>>(w, xq, dst_d,
                    (uint32_t) ne00, (uint32_t) ne01,
                    (uint32_t) (ne00 / 32),
                    dst_s1);
            return;
        }
        if (u4q41) {
            // Q4_1 u4 single-token (workspace-pressure chunk of 1): same shape,
            // U4Q41 variant, xs = ne00/32 (Q4_1 guarantees ne00 % 32 == 0).
            const dim3 grid((ne01 + 1) / 2, 1, 1);
            mul_mat_vec_repacked_nc<2, 1, 1, 2, 64, GGML_TYPE_Q4_1, false, false, false, true>
                <<<grid, 64, 0, stream>>>(w, xq, dst_d,
                    (uint32_t) ne00, (uint32_t) ne01,
                    (uint32_t) (ne00 / 32),
                    dst_s1);
            return;
        }
        switch (src0->type) {
            case GGML_TYPE_Q8_0: {
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                // Measured A/B hook: route Q8_0 through the shared generic
                // mat-vec. If it holds parity the bespoke kernel retires.
                static const bool generic_mmv = getenv("GGML_RP_GENERIC_MMV") != nullptr;
                if (generic_mmv) {
                    mul_mat_vec_rp<GGML_TYPE_Q8_0, 16, 16, false><<<grid, 1024, 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                        nullptr, 1, 0, 0, 0);
                    break;
                }
                mul_mat_vec_q8_0_repacked<16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, nullptr, nullptr, 0, 1, 0, 0, 0,
                    nullptr, nullptr, nullptr, GGML_GLU_OP_REGLU);
            } break;
            case GGML_TYPE_MXFP4: {
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                mul_mat_vec_rp<GGML_TYPE_MXFP4, 16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, 1, 0, 0, 0);
            } break;
            case GGML_TYPE_IQ4_NL: {
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                mul_mat_vec_rp<GGML_TYPE_IQ4_NL, 16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, 0, 0, 0, 1, 0, 0, 0);
            } break;
            case GGML_TYPE_Q4_0: {
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                mul_mat_vec_rp<GGML_TYPE_Q4_0, 16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, 0, 0, 0, 1, 0, 0, 0);
            } break;
            case GGML_TYPE_Q4_1: {
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                mul_mat_vec_rp<GGML_TYPE_Q4_1, 16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, 0, 0, 0, 1, 0, 0, 0);
            } break;
            case GGML_TYPE_Q5_1: {
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                mul_mat_vec_rp<GGML_TYPE_Q5_1, 16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, 0, 0, 0, 1, 0, 0, 0);
            } break;
            case GGML_TYPE_Q5_K: {
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                mul_mat_vec_rp<GGML_TYPE_Q5_K, 16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, 0, 0, 0, 1, 0, 0, 0);
            } break;
            case GGML_TYPE_Q4_K: {
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                mul_mat_vec_rp<GGML_TYPE_Q4_K, 16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, 0, 0, 0, 1, 0, 0, 0);
            } break;
            case GGML_TYPE_Q6_K: {
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                mul_mat_vec_rp<GGML_TYPE_Q6_K, 16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, 0, 0, 0, 1, 0, 0, 0);
            } break;
            default: GGML_ABORT("unsupported repack type");
        }
        return;
    }

    // Multi-token: 64-wide GEMM for large batches, 32-wide for small ones.
    const int nrl  = MMQ_RP_Q8_NROW_LANES;
    const int mmq_bm = MMQ_RP_Q8_BM;
    if (ne11 >= 128) {
        const int bn = 64 * MMQ_RP_Q8_TN;
        const dim3 grid((ne01 + mmq_bm - 1) / mmq_bm,
                        (ne11 + bn - 1) / bn, 1);
        switch (src0->type) {
            case GGML_TYPE_Q8_0:
                mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_Q8_0><<<grid, dim3(64, nrl), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_MXFP4:
                mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_MXFP4><<<grid, dim3(64, nrl), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_IQ4_NL:
                mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_IQ4_NL><<<grid, dim3(64, nrl), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q4_0:
                // G3: under the u4act gate, the dot8 u4-activation GEMM.
                // M4: under gate=2 (u4p8), the per-8 u4p dot8 GEMM instead.
                if (u4x16) {
                    mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_Q4_0, false, false, true><<<grid, dim3(64, nrl), 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                        nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                } else if (u4p8) {
                    mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_Q4_0, false, true><<<grid, dim3(64, nrl), 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                        nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                } else if (u4act) {
                    mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_Q4_0, true><<<grid, dim3(64, nrl), 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                        nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                } else {
                    mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_Q4_0><<<grid, dim3(64, nrl), 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                        nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                }
                break;
            case GGML_TYPE_Q4_1:
                // Q4_1 u4 (q41-u4-probe): under DOT8=4 the dp8a u4 GEMM; the
                // gate-off/SKIP case runs the repacked STOCK affine GEMM (the
                // q8_1 DS4 activation sum arrives automatically — mmq_get_q8_1_ds_layout
                // already maps Q4_1 to DS4).
                if (u4q41) {
                    mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_Q4_1, false, false, false, true><<<grid, dim3(64, nrl), 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                        nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                } else {
                    mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_Q4_1><<<grid, dim3(64, nrl), 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                        nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                }
                break;
            case GGML_TYPE_Q5_1:
                mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_Q5_1><<<grid, dim3(64, nrl), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q5_K:
                mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_Q5_K><<<grid, dim3(64, nrl), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q4_K:
                mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_Q4_K><<<grid, dim3(64, nrl), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q6_K:
                mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_Q6_K><<<grid, dim3(64, nrl), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            default: GGML_ABORT("unsupported repack type");
        }
    } else {
        const int bn = 32;
        // GGML_RP_KSPLIT: 1 = default slice count (4), 2/4/8 = explicit. K-splits
        // the w32 GEMM along K when the batch is in the skinny winning regime
        // (16 <= ne11 <= 64, standalone bench 2.1-3.0x, /root/w4-bench/bench4):
        // grid.z slices write fp32 partials to a pool scratch, a fixed-order
        // reduce kernel sums them into dst. Reorders fp32 accumulation vs the
        // single pass (maxrel ~1e-5, see /root/ksplit-results.txt). Unset =
        // stock single-pass, byte-identical. Q4_0 only, not the u4p8 variant.
        static const int rp_ksplit_env = [] {
            const char * e = getenv("GGML_RP_KSPLIT");
            const int n = e != nullptr ? atoi(e) : 0;
            if (n != 0) {
                GGML_LOG_WARN("CUDA repack: GGML_RP_KSPLIT=%d gate ACTIVE (K-sliced w32 GEMM at 16<=ne11<=64)\n", n);
            }
            return n;
        }();
        const int ksplit_slices = rp_ksplit_env >= 2 ? rp_ksplit_env
                                : (rp_ksplit_env == 1 ? 4 : 0);
        // K-split's measured positives are on the STOCK q8_1 activation path only: under gate1
        // (u4act) it is NEGATIVE on both lineages, and the u4x16 combination is unmeasured.
        // Restrict it to the stock path so the gate cannot silently engage a known-negative
        // combination (BENCH 52a + the 2026-10-09 sign resolution).
        bool ksplit = ksplit_slices > 0 && src0->type == GGML_TYPE_Q4_0 && !u4p8
                   && !u4act && !u4x16
                   && ne11 >= 16 && ne11 <= 64
                   && ne00 / 32 >= 4 * (int64_t) ksplit_slices;
        ggml_cuda_pool_alloc<float> ksplit_part(ctx.pool());
        if (ksplit) {
            float * p = ksplit_part.try_alloc((size_t) ksplit_slices * ne11 * ne01);
            if (p == nullptr) {
                ksplit = false;   // scratch did not fit: stock single-pass
            }
        }
        const dim3 grid((ne01 + mmq_bm - 1) / mmq_bm,
                        (ne11 + bn - 1) / bn, 1);
        switch (src0->type) {
            case GGML_TYPE_Q8_0:
                mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_Q8_0><<<grid, dim3(32, nrl*2), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_MXFP4:
                mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_MXFP4><<<grid, dim3(32, nrl*2), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_IQ4_NL:
                mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_IQ4_NL><<<grid, dim3(32, nrl*2), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q4_0:
                // GGML_RP_KSPLIT: K-sliced w32 GEMM + fixed-order reduce.
                // INTEGRATION (perf/w4a4-integrated): K-split now wraps the
                // per-32 u4act AND the per-16 u4x16 read paths as well as the
                // stock q8_1 path — the u4x16 activation read is token-major
                // (xcol*n_sub + sb), so a K-slice reads its own contiguous slab
                // of the pool and the sb_lo/sb_hi bounds do the rest. The per-8
                // u4p variant stays excluded (unchanged from side A).
                if (ksplit) {
                    float * const ypart = ksplit_part.get();
                    const dim3 gk(grid.x, grid.y, (unsigned) ksplit_slices);
#define RP_KSPLIT_LAUNCH(KS) \
                    mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_Q4_0, false, false, false, false, KS> \
                        <<<gk, dim3(32, nrl*2), 0, stream>>>( \
                            w, xq, ypart, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11, \
                            nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01); \
                    { const size_t plane = (size_t) ne11 * ne01; \
                      mmq_gemm_repacked_ksplit_reduce<KS> \
                          <<<(unsigned) ((plane + 255) / 256), 256, 0, stream>>>( \
                              ypart, dst_d, (uint32_t) ne01, (uint32_t) ne11, dst_s1); }
                    // ksplit implies the stock q8_1 activation path (guarded above), so only
                    // the stock w32 variant is ever launched here.
                    switch (ksplit_slices) {
                        case 2:  RP_KSPLIT_LAUNCH(2) break;
                        case 8:  RP_KSPLIT_LAUNCH(8) break;
                        default: RP_KSPLIT_LAUNCH(4) break;
                    }
#undef RP_KSPLIT_LAUNCH
                } else if (u4x16) {
                    // G3: under the u4x16 gate, the dot8 per-16 GEMM (w32).
                    mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_Q4_0, false, false, true><<<grid, dim3(32, nrl*2), 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                        nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                } else if (u4p8) {
                    mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_Q4_0, false, true><<<grid, dim3(32, nrl*2), 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                        nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                } else if (u4act) {
                    mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_Q4_0, true><<<grid, dim3(32, nrl*2), 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                        nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                } else {
                    mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_Q4_0><<<grid, dim3(32, nrl*2), 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                        nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                }
                break;
            case GGML_TYPE_Q4_1:
                // Q4_1 u4 (q41-u4-probe): see the 64-wide arm above. ksplit is
                // never set here (its predicate is Q4_0-only).
                if (u4q41) {
                    mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_Q4_1, false, false, false, true><<<grid, dim3(32, nrl*2), 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                        nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                } else {
                    mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_Q4_1><<<grid, dim3(32, nrl*2), 0, stream>>>(
                        w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                        nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                }
                break;
            case GGML_TYPE_Q5_1:
                mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_Q5_1><<<grid, dim3(32, nrl*2), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q5_K:
                mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_Q5_K><<<grid, dim3(32, nrl*2), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q4_K:
                mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_Q4_K><<<grid, dim3(32, nrl*2), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            case GGML_TYPE_Q6_K:
                mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_Q6_K><<<grid, dim3(32, nrl*2), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
                    nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
                break;
            default: GGML_ABORT("unsupported repack type");
        }
    }
    GGML_UNUSED(ctx);
}

// Fused MMV entry for repacked weights (dense): single-token only, fuses the up and
// gate dot products (shared quantized input) plus optional biases and the GLU op.
// Dispatch: ggml_cuda_try_fuse `{op,op,GLU}` and `{op,ADD}` repack branches.
void ggml_cuda_mul_mat_vec_repacked_fused(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));
    GGML_ASSERT(dst->ne[1] == 1);   // fused MMV is single-token only
    GGML_ASSERT(fusion != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    GGML_ASSERT(ne10 == ne00);

    cudaStream_t stream = ctx.stream();

    const uint8_t * w;
    if (src0->view_src != nullptr && ggml_cuda_repack_tensor_supported(src0->view_src)) {
        w = repack_view_get_cached(src0, src0->view_src, stream);
    } else {
        w = (const uint8_t *) src0->data;
    }

    const uint8_t * w_gate = nullptr;
    if (fusion->gate != nullptr) {
        const ggml_tensor * gate = fusion->gate;
        if (gate->view_src != nullptr && ggml_cuda_repack_tensor_supported(gate->view_src)) {
            w_gate = repack_view_get_cached(gate, gate->view_src, stream);
        } else {
            w_gate = (const uint8_t *) gate->data;
        }
    }

    const float * x_bias     = fusion->x_bias     ? (const float *) fusion->x_bias->data     : nullptr;
    const float * gate_bias  = fusion->gate_bias  ? (const float *) fusion->gate_bias->data  : nullptr;
    const ggml_glu_op glu_op = fusion->glu_op;

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;

    ggml_cuda_pool_alloc<block_q8_1> src1_q8_1_own;
    block_q8_1 * src1_q8_1_d;
    {
        const int64_t s11 = src1->nb[1] / sizeof(float);
        const int64_t s12 = src1->nb[2] / sizeof(float);
        const int64_t s13 = src1->nb[3] / sizeof(float);
        src1_q8_1_d = repack_quantize_src1_q8_1(ctx, src0, src1, ne10, ne10_padded,
            s11, s12, s13, 1, ne12, ne13,
            (size_t) (ne13 * ne12 * x_stride), src1_q8_1_own, stream);
    }
    for (int64_t i3 = 0; i3 < ne13; i3++) {
    for (int64_t i2 = 0; i2 < ne12; i2++) {
        const block_q8_1 * xq = src1_q8_1_d
                          + (i3 * ne12 + i2) * x_stride;
        float * dst_d = (float *)((char *) dst->data + i3 * dst->nb[3] + i2 * dst->nb[2]);
        // Bias is shaped like the single-token dst ([ne0, 1, ne2]): per-channel stride ne01.
        const float * x_bias_s    = x_bias    ? x_bias    + (i3 * ne12 + i2) * ne01 : nullptr;
        const float * gate_bias_s = gate_bias ? gate_bias + (i3 * ne12 + i2) * ne01 : nullptr;

        const dim3 grid((ne01 + 15) / 16, 1, 1);
        if (src0->type == GGML_TYPE_MXFP4) {
            mul_mat_vec_rp<GGML_TYPE_MXFP4, 16, 16, false, 64, true><<<grid, 1024, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                nullptr, 1, 0, 0, 0,
                w_gate, x_bias_s, gate_bias_s, glu_op);
        } else if (src0->type == GGML_TYPE_IQ4_NL) {
            mul_mat_vec_rp<GGML_TYPE_IQ4_NL, 16, 16, false, 64, true><<<grid, 1024, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                nullptr, 1, 0, 0, 0,
                w_gate, x_bias_s, gate_bias_s, glu_op);
        } else if (src0->type == GGML_TYPE_Q5_K) {
            mul_mat_vec_rp<GGML_TYPE_Q5_K, 16, 16, false, 64, true><<<grid, 1024, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                nullptr, 1, 0, 0, 0,
                w_gate, x_bias_s, gate_bias_s, glu_op);
        } else if (src0->type == GGML_TYPE_Q4_K) {
            mul_mat_vec_rp<GGML_TYPE_Q4_K, 16, 16, false, 64, true><<<grid, 1024, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                nullptr, 1, 0, 0, 0,
                w_gate, x_bias_s, gate_bias_s, glu_op);
        } else if (src0->type == GGML_TYPE_Q6_K) {
            mul_mat_vec_rp<GGML_TYPE_Q6_K, 16, 16, false, 64, true><<<grid, 1024, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                nullptr, 1, 0, 0, 0,
                w_gate, x_bias_s, gate_bias_s, glu_op);
        } else {
            mul_mat_vec_q8_0_repacked<16, 16, false, 64, true><<<grid, 1024, 0, stream>>>(
                w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                nullptr, nullptr, nullptr, 0, 1, 0, 0, 0,
                w_gate, x_bias_s, gate_bias_s, glu_op);
        }
    }
    }
}
