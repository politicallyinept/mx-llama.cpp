// Host-side support for the Q8_0 repacked-weight path: tensor support check, host
// repack routine, and the persistent per-view cache shared by dense and MoE paths.
#include "repack.cuh"
#include "repack-common.cuh"
#include "ggml-cuda.h"   // GGML_LOG_WARN for the gate-engagement markers
#include "ggml-impl.h"   // ggml_scratch_gen_bump
#include "w4a4-profile.h"   // GGML_CUDA_W4A4_PROFILE: resolved int4-knob presets (after ggml-cuda.h so the real GGML_LOG_WARN macro wins over the header fallback)

// Resolve the profile at library load (see w4a4-profile.h): the summary/unknown
// WARN must be readable from the log even when every knob resolves to 0.
static const w4a4_profile_detail::w4a4_profile_init s_w4a4_profile_init;

#include <cstdlib>

#include <atomic>
#include <cstring>
#include <map>
#include <mutex>

// SUMW plane gate (see sumw-plane.h): GGML_CUDA_W4A4_SUMW=1 turns on the
// repack-time precomputed SUMW plane for the Q4_0 u4 path. Default OFF:
// repack_gcn_nbytes keeps the 18 B/block record, repack_q40_host writes exactly
// what repack_iq4nl_host wrote, and every kernel keeps the inline nib_sum_u4
// computation - bit-identical to pre-change behavior.
bool repack_sumw_plane_on() {
    static const bool s_on = [] {
        // RESOLVED via the profile helper: explicit env > profile preset
        // (speed presets SUMW=1) > 0 (off). No profile + no explicit env ->
        // off, exactly the pre-profile read.
        const w4a4_profile_knobs & prof = w4a4_profile();
        const bool on = prof.sumw != 0;
        if (on) {
            GGML_LOG_WARN("CUDA repack: GGML_CUDA_W4A4_SUMW gate ACTIVE [%s]: Q4_0 u4 reads the precomputed SUMW plane (+2 B/block)\n",
                          prof.sumw_src);
        }
        return on;
    }();
    return s_on;
}

// Resolve the Q8_0 repack admission: opt-in, with a shape-gated level. Env levels:
//   unset / =0 : OFF. The default path is untouched.
//   =1         : FORCE admission ON for every model - the historical meaning of
//                =1, kept backward-compatible with every existing script and A/B
//                measurement config that exports it (escape hatch for A/B on a
//                model whose own shape would decline admission; the dense-Q8_0
//                buffer-doubling OOM is then the operator's own risk).
//   >=2        : admit QUALIFYING models only (opt-in shape gate). A model
//                qualifies when its own per-model shape hint says so: 3D expert
//                tensors, dense (non-expert) Q8_0 at or above the
//                GGML_CUDA_REPACK_Q8_0_MIN_DENSE floor, and tensor split -
//                computed per model by the loader (llama-model.cpp).
//                The hint is read from the TENSOR (ggml_tensor_model_has_expert_tensors):
//                the loader stamps it on every weight tensor of the owning model
//                before that model's buffers are allocated, and the meta backend
//                copies it onto every per-lane tensor, so in a multi-model process
//                (MoE main + small spec-decode draft) each model is admitted by its
//                OWN shape - a small draft whose dense Q8_0 is below the floor is
//                never admitted by the main model's hint (the buffer-doubling OOM
//                on small drafts). A process-global latch would corrupt exactly
//                this case: the main loads first and the draft inherits its value.
// The level semantics were SWAPPED on 2026-11-09: =1 had briefly become the
// shape-gated level, which silently downgraded every existing =1 script on a
// dense model or under layer-split to the stock path. =1 keeps its historical
// force meaning; the shape-gated level got the new value (=2) that no existing
// config had used, so the new mode is strictly opt-in.
// The repacked prefill GEMM is still not bit-identical to stock on the shapes the
// gcn_kq yield declines (9..31-token prefill etc): the divergence is fp32
// K-accumulation geometry (repacked: serial full-K per thread; stock: K split
// across tile lanes + shared reduce - see the STOCK_ACC gate below for the
// divergence map). Wiring the existing RESTYLE GEMM template branch would NOT
// fix that: it only reorders the 8 dp4a ops of a 32-value block inside an exact
// int32 accumulator, which is order-invariant.
static int q8_0_repack_env_mode(void) {
    static const int env = [] {
        const char * e = getenv("GGML_CUDA_REPACK_Q8_0");
        if (e == nullptr) {
            return 0;
        }
        const int v = atoi(e);
        if (v <= 0) {
            return -1;              // explicit OFF
        }
        return v >= 2 ? 2 : 1;      // 1 = forced for all (historical), 2 = shape-gated qualifying models only
    }();
    return env;
}

static bool q8_0_repack_admitted(const ggml_tensor * t) {
    switch (q8_0_repack_env_mode()) {
        case 1:  return true;                                     // forced for every model (=1, historical)
        case 2:  return ggml_tensor_model_has_expert_tensors(t);  // per-model shape hint (=2)
        default: return false;                                    // unset or =0: OFF
    }
}

bool ggml_cuda_repack_tensor_supported(const ggml_tensor * t) {
    // Views are never in repacked layout (the scale-plane offset needs the FULL ne1).
    if (t->view_src != nullptr) return false;
    if ((ggml_n_dims(t) != 2 && ggml_n_dims(t) != 3) || !ggml_is_contiguous(t)) {
        return false;
    }
    // Q4_0 admission is opt-in: GGML_CUDA_REPACK_Q4_0=1. Unset/0 keeps the
    // stock canonical-kernel path for Q4_0 exactly as before.
    static const bool s_q4_0 = [] {
        const char * e = getenv("GGML_CUDA_REPACK_Q4_0");
        const bool on = e != nullptr && atoi(e) != 0;
        if (on) {
            GGML_LOG_WARN("CUDA repack: GGML_CUDA_REPACK_Q4_0 gate ACTIVE: Q4_0 admitted to the gfx906 repack path\n");
        }
        return on;
    }();
    // Q4_1 admission is opt-in on the W4A4 gate: GGML_CUDA_W4A4_DOT8=4 admits
    // Q4_1 to the repack pool AND selects the u4 (dp8a) decode. A Q4_1 tensor
    // with the gate unset is NOT admitted at all (this returns false), so it
    // never lands in repacked layout and silently runs a wrong fast path — the
    // buffer falls back to the canonical stock path for it, byte-identical to
    // pre-branch behavior.
    // MIRROR RULE (peer review): admission must always mirror the decode
    // predicates in q8_repack/mul-mat.cu. Gate 4 is a SUPERSET (per-32 Q4_0 u4
    // + Q4_1 u4): Q4_0 admission is unchanged (GGML_CUDA_REPACK_Q4_0, modes
    // 1-4 all use the same repacked Q4_0 layout), Q4_1 admission stays
    // DOT8==4-only. A tensor that is admitted is decoded by exactly the path
    // its type selects; a tensor that is not admitted never reaches a u4 path.
    static const bool s_q4_1 = [] {
        // DOT8 is an int4-path knob: read the RESOLVED value (explicit env >
        // profile preset). No profile + no explicit env -> 0, exactly the
        // pre-profile read. No profile presets DOT8=4, so admission behavior
        // is unchanged; the read is routed for single-source consistency.
        const bool on = w4a4_profile().dot8 == 4;
        if (on) {
            GGML_LOG_WARN("CUDA repack: GGML_CUDA_W4A4_DOT8=4 gate ACTIVE: Q4_1 admitted to the gfx906 repack path (u4 dp8a decode)\n");
        }
        return on;
    }();
    // A/B INSTRUMENT (gcn-repro 2026-10-24): GGML_CUDA_REPACK_KQUANT=0 keeps
    // Q4_K/Q5_K/Q6_K tensors in canonical layout (never repacked), so the GCN
    // K-quant GEMM port can be A/B'd against the repack GEMM path. Mirrors the
    // Q4_0 opt-in: admission-off runs the stock path exactly as before.
    static const bool s_kquant_off = [] {
        const char * e = getenv("GGML_CUDA_REPACK_KQUANT");
        const bool off = e != nullptr && atoi(e) == 0;
        if (off) {
            GGML_LOG_WARN("CUDA repack: GGML_CUDA_REPACK_KQUANT=0 gate ACTIVE: Q4_K/Q5_K/Q6_K stay canonical (stock path)\n");
        }
        return off;
    }();
    // Q8_0 admission: OPT-IN (GGML_CUDA_REPACK_Q8_0=1). The repacked Q8_0 path
    // (mat-vec restyled to stock granularity + the gcn_kq yield) is bit-identical
    // to the stock path on the shapes it covers and measured +1.3..+1.5% decode on
    // MoE models, but a net regression on a dense all-Q8_0 model (-1.6% decode,
    // -8.4% prefill, +24 GB VRAM for the dual-layout canon copy). History: the
    // path was default-ON pre-RESTYLE, flipped to opt-in after a decode-regression
    // finding on the pre-RESTYLE kernels, briefly shipped as a MoE-conditional
    // default (admit when the loaded model has 3D expert tensors), and is back to
    // opt-in: the RESTYLE GEMM is not wired at the repacked GEMM dispatch sites,
    // so the repacked prefill GEMM is not bit-identical to stock on the shapes the
    // gcn_kq yield declines (9..31-token prefill etc). GGML_CUDA_REPACK_Q8_0=1
    // admits only models whose own per-model shape hint qualifies (see
    // q8_0_repack_admitted above: 3D expert tensors + dense-Q8_0 floor + tensor
    // split, read per tensor); =2 forces admission ON for every model (escape
    // hatch); =0 forces it OFF. Effective only together with STOCK_ACC=0.
    //
    // The repack is a layout repack of the 8-bit weights (dp4a-friendly row +
    // scale-plane layout); without RESTYLE the accumulation order differs from stock
    // - see the STOCK_ACC gate below for the divergence map and the neutral fallback.
    // Tier-3.3 stock-accumulation gate: GGML_CUDA_REPACK_Q8_0_STOCK_ACC defaults ON
    // (the pre-ship posture): Q8_0 matmuls stay on STOCK accumulation order (Q8_0
    // stays canonical, repack suppressed). The RESTYLE mat-vec kernels reproduce the
    // stock accumulation order and granularity on the repacked layout for the
    // single-token MMV and nc verify at widths 2-5 (nc widths 6-8 diverge: stock
    // GCN mmvq runs 1 warp there, the repacked nc kernel runs 2 - different
    // reduce-tree geometry), but the GEMM restyle is not wired at the dispatch
    // sites, so the repacked prefill GEMM is not bit-identical to stock on the
    // shapes the gcn_kq yield declines - the default stays stock until that lands.
    // Note: wiring the existing RESTYLE template branch at the GEMM dispatch sites
    // is NOT that fix - the branch only reorders the 8 dp4a ops of a 32-value block
    // inside an exact int32 accumulator (order-invariant); the real divergence is
    // the fp32 K-accumulation geometry below (serial full-K per thread vs stock's
    // tx-lane K split + shared reduce), which needs a re-derivation of the stock
    // kernel geometry on the repacked layout. =0 opts into the repack path (together
    // with GGML_CUDA_REPACK_Q8_0=1) for A/B and for the validated MoE decode win.
    //
    // Divergence map (adopt-gdn-rows): the weight repack itself is lossless -
    // block_q8_0 carries a ggml_half d (ggml-common.h:253) and both planes are
    // a byte-for-byte permutation of the canonical blocks (repack_q8_0_kernel,
    // q8_repack/buffer.cu:127-148) - and both engines feed the same q8_1
    // activation quantization, so the drift is purely fp32 accumulation order:
    //  * decode MMV: stock mul_mat_vec_q assigns (block, quarter) pairs across
    //    the whole block thread space (mmvq.cu:727-731: kbx = tid/(qi/vdr),
    //    kqs = vdr*(tid%(qi/vdr))) and applies the d*dx scale per 8-value
    //    chunk (vec_dot_q8_0_q8_1, vecdotq.cuh:852-865, VDR_Q8_0_Q8_1_MMVQ=2),
    //    then sums cross-warp partials in shared memory and a 32-lane shfl
    //    tree (mmvq.cu:793-807). The repacked MMV applies the scale per
    //    16-value half-block across 64 lanes (repack-kernels.cuh:299,346) and
    //    reduces with a 64-lane DPP tree (repack-kernels.cuh:439).
    //  * prefill GEMM: stock mul_mat_q splits K across the tile's tx lanes
    //    (per-lane partial sums, shared reduce at tile end,
    //    mmq-vec-dot.cuh:137-148); the repacked GEMM accumulates the full K
    //    serially per thread into acc[r][n] (repack-kernels.cuh:1784).
    // Different chunk granularity + lane partition + reduce tree => different
    // fp32 rounding => greedy divergence (q8n1r KL 0.0035 vs T0; the
    // q40-longpp r02 flip attributed to a2f211cb0).
    //
    // A bit-identical in-kernel restyle must reproduce stock's exact launch
    // geometry (the mmvq nwarps/rows_per_block tables are internal to
    // mmvq.cu) and MMQ's tx K-split - a re-derivation of both stock kernels
    // on the repacked layout, not a restyle. Until that lands this gate is
    // the numerics-neutral fallback: the stock kernels cannot read the
    // two-plane layout, so admission is the only point where a stock-kernel
    // fallback is possible. With the gate ON, Q8_0 tensors stay canonical:
    // the repack buft degrades cleanly for un-admitted tensors (canonical
    // get_alloc_size + plain memcpy upload, buffer.cu:611,740) and
    // should_fire() stays false, so Q8_0 matmuls run the stock kernels
    // bit-identically to a stock build while every other repacked type
    // (Q4_0/Q4_1/K-quant/MXFP4) keeps its speed. Same kernel count per
    // matmul as stock - no extra passes.
    static const bool s_q8_0_stock_acc = [] {
        const char * e = getenv("GGML_CUDA_REPACK_Q8_0_STOCK_ACC");
        const bool on = e == nullptr || atoi(e) != 0;
        if (on) {
            GGML_LOG_WARN("CUDA repack: GGML_CUDA_REPACK_Q8_0_STOCK_ACC gate ACTIVE: Q8_0 stays canonical, stock accumulation (repack suppressed for Q8_0)\n");
        }
        return on;
    }();
    switch (t->type) {
        case GGML_TYPE_Q8_0: {
            const bool adm = q8_0_repack_admitted(t) && !s_q8_0_stock_acc;
            // The gate decision is logged whenever it CHANGES (per-model shape hints
            // can differ between models in one process: MoE main + small spec-decode
            // draft), so the log shows each model's own admission instead of a
            // once-per-process banner that would only ever show the first decision.
            static std::atomic<int> s_banner_adm(-1);
            const int adm_now = adm ? 1 : 0;
            if (s_banner_adm.exchange(adm_now) != adm_now) {
                GGML_LOG_WARN("CUDA repack: GGML_CUDA_REPACK_Q8_0 gate %s (tensor '%s', env mode %d): Q8_0 %s the gfx906 repack path (mode 1 = forced for all, mode 2 = qualifying models only by per-model shape hint)\n",
                              adm ? "ACTIVE" : "inactive", ggml_get_name(t), q8_0_repack_env_mode(),
                              adm ? "admitted to" : "stays canonical for");
            }
            return adm && t->ne[0] % 32 == 0;
        }
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_IQ4_NL: {
            return t->ne[0] % 32 == 0;
        }
        case GGML_TYPE_Q4_0: {
            return s_q4_0 && t->ne[0] % 32 == 0;
        }
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K: {
            // whole super-blocks, and no views: the multi-plane layouts have no view splice yet
            return !s_kquant_off && t->ne[0] % 256 == 0 && t->view_src == nullptr;
        }
        case GGML_TYPE_Q5_1: {
            // 32-value blocks, multi-plane like the K-quants, so no view splice either
            return t->ne[0] % 32 == 0 && t->view_src == nullptr;
        }
        case GGML_TYPE_Q4_1: {
            // Gated on GGML_CUDA_W4A4_DOT8=4 (s_q4_1 above). 32-value blocks,
            // two-plane (nibble rows + half2 {d,m} plane) like Q5_1, so no view
            // splice either. With the gate unset this returns false: the tensor
            // is not repacked at all and runs the canonical stock path.
            return s_q4_1 && t->ne[0] % 32 == 0 && t->view_src == nullptr;
        }
        default:             return false;
    }
}

bool ggml_cuda_repack_mmv_fusion_width_ok(int64_t n_tokens, bool has_ids, ggml_type wt) {
    // MoE fuses across the whole narrow range: its grid is one block per
    // assignment, so at these widths launch overhead dominates and folding
    // three launches into one pays. Dense does NOT: its grid already spans
    // the full row count, there is ample parallelism to hide latency, and
    // halving the block count to double the work per block costs more than
    // the saved launches return. Measured on Qwen3.6-27B-MTP-Q8_0, pp512 at
    // that ubatch, fused against unfused: +4.6% at 2 tokens, -2.7% at 3,
    // +2.4% at 4 - and a speculative verify step is exactly 3 wide at the
    // default draft depth, so dense fusion loses where it would be used.
    const int64_t max_tokens = has_ids ? MMQ_RP_Q8_MOE_MMV_MAX_TOKENS : 1;
    GGML_UNUSED(wt);
    return n_tokens >= 1 && n_tokens <= max_tokens;
}

bool ggml_cuda_repack_mul_mat_should_fire(const ggml_tensor * src0) {
    if (src0->buffer == nullptr || !ggml_backend_buft_is_cuda_repack(ggml_backend_buffer_get_type(src0->buffer))) {
        return false;
    }
    if (ggml_cuda_repack_tensor_supported(src0)) {
        return true;              // full tensor in repacked layout
    }
    return src0->view_src != nullptr && ggml_cuda_repack_tensor_supported(src0->view_src);
}

// E4 gcn_kq yield: static half of the ggml_cuda_gcn_kq_gemm admission for dense
// Q8_0 (gcn-q8-gemm.cu:1976 area). Mirrors the env/mask logic there: default
// GGML_CUDA_GCN_KQ=1 -> mask 8191 admits bit 128 (Q8_0); GGML_GCN_GEMM_MIN_M
// default 128; K%256==0 (QK_K super-blocks); dense single-batch; not a view
// (views splice from the repacked base and have no canonical copy). The runtime
// half (N>=32, src1/dst layout, alignment, v4 checks) is decided by calling
// ggml_cuda_gcn_kq_gemm itself, so the regime split can never drift from stock.
// Consulted at load time (get_alloc_size / upload: tensor->buffer not yet set)
// AND at dispatch; identical env statics both times, so alloc and yield agree.
bool ggml_cuda_q8_yield_eligible(const ggml_tensor * t) {
    if (t->type != GGML_TYPE_Q8_0 || t->view_src != nullptr) {
        return false;
    }
    if (t->ne[2] != 1 || t->ne[3] != 1) {
        return false;
    }
    if (!ggml_cuda_repack_tensor_supported(t)) {
        return false;   // Q8_0 repack gate off (or STOCK_ACC) -> nothing to yield
    }
    static const int kq_env = [] { const char * e = getenv("GGML_CUDA_GCN_KQ"); return e ? atoi(e) : 1; }();
    const int mask = kq_env == 1 ? 8191 : kq_env;
    if (!(mask & 128)) {
        return false;   // Q8_0 bit declined by GGML_CUDA_GCN_KQ -> stock regime is mmq
    }
    static const int min_rows = [] { const char * e = getenv("GGML_CUDA_GCN_GEMM_MIN_M"); return e ? atoi(e) : 128; }();
    return t->ne[0] % 256 == 0 && t->ne[1] >= (int64_t) min_rows;
}

// The FUSED mat-vec path dispatches to mul_mat_vec_q8_0_repacked, which is Q8_0-only:
// an MXFP4 sub-block is a single uint4 of nibbles, so that kernel's half-block
// indexing (2 uint4 per sub-block) has no MXFP4 equivalent. MXFP4 reaches the GEMM
// for every token count instead, so it must not be offered the fusion.
bool ggml_cuda_repack_mmv_fusion_supported(const ggml_tensor * src0) {
    const ggml_tensor * t = src0->view_src != nullptr ? src0->view_src : src0;
    return (t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_MXFP4) &&
           ggml_cuda_repack_mul_mat_should_fire(src0);
}

// The MoE up/gate fusion has a fused mat-vec for every repacked type, so the expert path admits all of them.
// The dense fusion stays on the predicate above until its K-quant and bias variants have their own gate.
// GGML_CUDA_REPACK_KQUANT_MOE_FUSION=0 keeps the expert path on the Q8_0 and MXFP4 admission.
bool ggml_cuda_repack_mmv_id_fusion_supported(const ggml_tensor * src0) {
    static const bool kquant = [] {
        const char * env = getenv("GGML_CUDA_REPACK_KQUANT_MOE_FUSION");
        return env == nullptr || atoi(env) != 0;
    }();
    const ggml_tensor * t = src0->view_src != nullptr ? src0->view_src : src0;
    const bool base = t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_MXFP4;
    // Each of these four types has been gated on a real MoE model: deterministic, unfused path unchanged, width-1 PPL within the repack's own spread.
    const bool kq   = t->type == GGML_TYPE_Q4_K || t->type == GGML_TYPE_Q5_K || t->type == GGML_TYPE_Q6_K || t->type == GGML_TYPE_IQ4_NL;
    return (base || (kquant && kq)) && ggml_cuda_repack_mul_mat_should_fire(src0);
}

// Host repack of one Q8_0 matrix: qs plane [ne1 x nsp x 32] then f16 scale plane
// [ne1 x nsp]; padding sub-blocks are left zeroed.
void repack_q8_0_host(const block_q8_0 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    // a non-multiple ne0 would silently drop the row tail below, so fail loudly instead
    GGML_ASSERT(ne0 % 32 == 0);
    const int64_t n_blocks = ne0 / 32;
    const int64_t qs_str   = repack_qs_stride(ne0);
    const size_t  qs_len   = (size_t) ne1 * qs_str;

    memset(dst, 0, qs_len + (size_t) ne1 * n_blocks * 2);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q8_0 * b = &blocks[row * n_blocks + blk];
            memcpy(dst + (size_t) row * qs_str + (size_t) blk * 32, b->qs, 32);
            memcpy(dst + qs_len + (size_t)(row * n_blocks + blk) * 2, &b->d, 2);
        }
    }
}

// Host repack of one MXFP4 matrix: nibble rows [ne1 x qs_str, de-aliased like
// Q8_0's qs rows] then a 1-byte e8m0 plane [ne1 x ne0/32]. The nibbles are
// copied VERBATIM - the low/high split is undone on the device by
// rp_mxfp4_expand, so the packed bytes stay a single aligned uint4 load.
// Padding bytes are left zeroed. NOTE: a zero e8m0 byte decodes to 2^-127, not
// 0, so device code must never dot padding nibbles against live data - the
// kernels guard with sb < n_sub, not with the scale.
void repack_mxfp4_host(const block_mxfp4 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    GGML_ASSERT(ne0 % 32 == 0);
    const int64_t n_blocks = ne0 / 32;
    const int64_t qs_str   = repack_qs_row_stride(GGML_TYPE_MXFP4, ne0);
    const size_t  qs_len   = (size_t) ne1 * qs_str;

    memset(dst, 0, qs_len + (size_t) ne1 * n_blocks);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_mxfp4 * b = &blocks[row * n_blocks + blk];
            memcpy(dst + (size_t) row * qs_str + (size_t) blk * 16, b->qs, 16);
            dst[qs_len + (size_t)(row * n_blocks + blk)] = b->e;
        }
    }
}

// Host repack of one IQ4_NL matrix: canonical block_iq4_nl (18 B: f16 d, 16 nibble bytes) to de-aliased nibble rows + an f16 plane.
// Padding stays zero, and a zero f16 is a zero scale, so padding can never leak into a sum.
void repack_iq4nl_host(const block_iq4_nl * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    GGML_ASSERT(ne0 % 32 == 0);
    const int64_t n_blocks = ne0 / 32;
    const int64_t qs_str   = repack_qs_row_stride(GGML_TYPE_IQ4_NL, ne0);
    const size_t  qs_len   = (size_t) ne1 * qs_str;

    memset(dst, 0, qs_len + (size_t) ne1 * n_blocks * 2);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_iq4_nl * b = &blocks[row * n_blocks + blk];
            memcpy(dst + (size_t) row * qs_str + (size_t) blk * 16, b->qs, 16);
            memcpy(dst + qs_len + (size_t)(row * n_blocks + blk) * 2, &b->d, 2);
        }
    }
}

// Host repack of one Q4_0 matrix: IQ4_NL's byte-identical converter (block_q4_0
// is byte-identical to block_iq4_nl: f16 d + 16 nibble bytes), plus the SUMW
// plane fill when GGML_CUDA_W4A4_SUMW is on (sumw-plane.h). Gate off: exactly
// the repack_iq4nl_host call, byte-for-byte. Gate on: the plane is int16 per
// 32-value block right after the d plane, holding (int16_t)((int) nib_sum - 256)
// - the SIGNED semantics the epilogue requires (milestone-3b class).
void repack_q40_host(const block_q4_0 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    repack_iq4nl_host((const block_iq4_nl *) blocks, dst, ne0, ne1);
    if (!repack_sumw_plane_on()) {
        return;
    }
    GGML_ASSERT(ne0 % 32 == 0);
    const int64_t n_blocks = ne0 / 32;
    const int64_t qs_str   = repack_qs_row_stride(GGML_TYPE_Q4_0, ne0);
    uint8_t * sw_plane     = dst + rp_sumw_plane_off(ne0, ne1, qs_str);
    rp_sumw_fill_plane((const rp_sumw_block *) blocks, sw_plane, ne1, n_blocks);
}

// Host repack of one Q6_K matrix.
// Canonical block_q6_K packs 256 values as ql[128] low nibbles + qh[64] 2-bit highs in an interleaved order, 16 int8 scales (one per 16 values) and one f16 d.
// Planes per 32-value sub-block: 16 B packed lows on the de-aliased row stride (byte j = lows of values j and j+16), 8 B packed highs (byte m = 2-bit fields of values 4m..4m+3, low half first), a 2-byte int8 scale pair, and the f16 d per super-block in its own plane.
void repack_q6k_host(const block_q6_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    GGML_ASSERT(ne0 % 256 == 0);
    const int64_t n_sub  = ne0 / 32;
    const int64_t nds    = (n_sub + 7) >> 3;
    const int64_t qs_str = repack_qs_row_stride(GGML_TYPE_Q6_K, ne0);
    const size_t  hoff   = (size_t) ne1 * qs_str;
    const size_t  soff   = hoff + (size_t) ne1 * n_sub * 8;
    const size_t  doff   = soff + (size_t) ne1 * n_sub * 2;

    memset(dst, 0, doff + (size_t) ne1 * nds * 2);

    for (int64_t row = 0; row < ne1; row++) {
        const block_q6_K * brow = blocks + row * (ne0 / 256);
        for (int64_t sb = 0; sb < n_sub; sb++) {
            const block_q6_K * b = &brow[sb >> 3];
            const int k = (int)(sb & 7);
            uint8_t raw[32];
            for (int i = 0; i < 32; i++) {
                const int v = k * 32 + i;
                const int h = v >> 7;
                const int l = v & 127;
                const uint8_t * ql = b->ql + 64 * h;
                const uint8_t * qh = b->qh + 32 * h;
                uint8_t r6;
                if      (l <  32) { r6 = (ql[l]      & 0xF) | (((qh[l]      >> 0) & 3) << 4); }
                else if (l <  64) { r6 = (ql[l]      & 0xF) | (((qh[l - 32] >> 2) & 3) << 4); }
                else if (l <  96) { r6 = (ql[l - 64] >>  4) | (((qh[l - 64] >> 4) & 3) << 4); }
                else              { r6 = (ql[l - 64] >>  4) | (((qh[l - 96] >> 6) & 3) << 4); }
                raw[i] = r6;
            }
            const size_t wi = (size_t) row * n_sub + sb;
            uint8_t * lows  = dst + (size_t) row * qs_str + (size_t) sb * 16;
            uint8_t * highs = dst + hoff + wi * 8;
            for (int j = 0; j < 16; j++) {
                lows[j] = (uint8_t)((raw[j] & 0xF) | ((raw[j + 16] & 0xF) << 4));
            }
            for (int m = 0; m < 8; m++) {
                const int base = (m & 3) * 4 + (m >> 2) * 16;
                highs[m] = (uint8_t)( ((raw[base + 0] >> 4) << 0)
                                    | ((raw[base + 1] >> 4) << 2)
                                    | ((raw[base + 2] >> 4) << 4)
                                    | ((raw[base + 3] >> 4) << 6));
            }
            dst[soff + wi * 2 + 0] = (uint8_t) b->scales[2 * k + 0];
            dst[soff + wi * 2 + 1] = (uint8_t) b->scales[2 * k + 1];
            if (k == 0) {
                memcpy(dst + doff + ((size_t) row * nds + (sb >> 3)) * 2, &b->d, 2);
            }
        }
    }
}

// Host repack of one Q4_K matrix.
// Canonical block_q4_K: half d, half dmin, the 12-byte packed scale/min record, then qs[128] where sub-block 2j lives in the low nibbles of bytes 32j..32j+31 and sub-block 2j+1 in their high nibbles.
// Planes: 16 B packed nibbles per sub-block on the de-aliased row stride (byte i = value i | value (16+i) << 4), the scale record verbatim per super-block, half2 {d, dmin}.
void repack_q4k_host(const block_q4_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    GGML_ASSERT(ne0 % 256 == 0);
    const int64_t n_sub  = ne0 / 32;
    const int64_t nsb    = (n_sub + 7) >> 3;
    const int64_t qs_str = repack_qs_row_stride(GGML_TYPE_Q4_K, ne0);
    const size_t  soff   = (size_t) ne1 * qs_str;
    const size_t  doff   = soff + (size_t) ne1 * nsb * 12;

    memset(dst, 0, doff + (size_t) ne1 * nsb * 4);

    for (int64_t row = 0; row < ne1; row++) {
        const block_q4_K * brow = blocks + row * (ne0 / 256);
        for (int64_t sb = 0; sb < n_sub; sb++) {
            const block_q4_K * b = &brow[sb >> 3];
            const int k = (int)(sb & 7);
            const uint8_t * q = b->qs + 32 * (k >> 1);
            const int shift = (k & 1) * 4;
            uint8_t * lows = dst + (size_t) row * qs_str + (size_t) sb * 16;
            for (int j = 0; j < 16; j++) {
                const uint8_t v0 = (q[j]      >> shift) & 0xF;
                const uint8_t v1 = (q[j + 16] >> shift) & 0xF;
                lows[j] = (uint8_t)(v0 | (v1 << 4));
            }
            if (k == 0) {
                const size_t sbk = (size_t) row * nsb + (sb >> 3);
                memcpy(dst + soff + sbk * 12, b->scales, 12);
                memcpy(dst + doff + sbk * 4, b, 4);
            }
        }
    }
}

// Host repack of one Q5_K matrix: Q4_K's planes plus the fifth bit.
// Canonical block_q5_K keeps it in qh[32], bit (2j + (k & 1)) of qh[i] for value i of sub-block k = 2j + (k & 1).
// The plane stores one 32-bit word per sub-block, bit i = value i.
void repack_q5k_host(const block_q5_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    GGML_ASSERT(ne0 % 256 == 0);
    const int64_t n_sub  = ne0 / 32;
    const int64_t nsb    = (n_sub + 7) >> 3;
    const int64_t qs_str = repack_qs_row_stride(GGML_TYPE_Q5_K, ne0);
    const size_t  hoff   = (size_t) ne1 * qs_str;
    const size_t  soff   = hoff + (size_t) ne1 * n_sub * 4;
    const size_t  doff   = soff + (size_t) ne1 * nsb * 12;

    memset(dst, 0, doff + (size_t) ne1 * nsb * 4);

    for (int64_t row = 0; row < ne1; row++) {
        const block_q5_K * brow = blocks + row * (ne0 / 256);
        for (int64_t sb = 0; sb < n_sub; sb++) {
            const block_q5_K * b = &brow[sb >> 3];
            const int k = (int)(sb & 7);
            const uint8_t * q = b->qs + 32 * (k >> 1);
            const int shift = (k & 1) * 4;
            const int hbit  = 2 * (k >> 1) + (k & 1);
            const size_t wi = (size_t) row * n_sub + sb;
            uint8_t * lows = dst + (size_t) row * qs_str + (size_t) sb * 16;
            uint32_t h = 0;
            for (int j = 0; j < 16; j++) {
                const uint8_t v0 = (q[j]      >> shift) & 0xF;
                const uint8_t v1 = (q[j + 16] >> shift) & 0xF;
                lows[j] = (uint8_t)(v0 | (v1 << 4));
            }
            for (int i = 0; i < 32; i++) {
                h |= (uint32_t)((b->qh[i] >> hbit) & 1) << i;
            }
            memcpy(dst + hoff + wi * 4, &h, 4);
            if (k == 0) {
                const size_t sbk = (size_t) row * nsb + (sb >> 3);
                memcpy(dst + soff + sbk * 12, b->scales, 12);
                memcpy(dst + doff + sbk * 4, b, 4);
            }
        }
    }
}

// Host repack of one Q5_1 matrix: the canonical nibble bytes and fifth-bit word per block are already in plane order, so the three planes are copies.
void repack_q51_host(const block_q5_1 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    GGML_ASSERT(ne0 % 32 == 0);
    const int64_t n_sub  = ne0 / 32;
    const int64_t qs_str = repack_qs_row_stride(GGML_TYPE_Q5_1, ne0);
    const size_t  hoff   = (size_t) ne1 * qs_str;
    const size_t  doff   = hoff + (size_t) ne1 * n_sub * 4;

    memset(dst, 0, doff + (size_t) ne1 * n_sub * 4);

    for (int64_t row = 0; row < ne1; row++) {
        const block_q5_1 * brow = blocks + row * n_sub;
        for (int64_t sb = 0; sb < n_sub; sb++) {
            const block_q5_1 * b = &brow[sb];
            const size_t wi = (size_t) row * n_sub + sb;
            memcpy(dst + (size_t) row * qs_str + (size_t) sb * 16, b->qs, 16);
            memcpy(dst + hoff + wi * 4, b->qh, 4);
            memcpy(dst + doff + wi * 4, &b->dm, 4);
        }
    }
}

// Host repack of one Q4_1 matrix: the canonical nibble bytes (already in plane
// order: byte i = value i | value (16+i) << 4, Q4_0's byte order) and the half2
// {d, m} per 32-value block. Q5_1's converter minus the fifth-bit plane.
void repack_q41_host(const block_q4_1 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    GGML_ASSERT(ne0 % 32 == 0);
    const int64_t n_sub  = ne0 / 32;
    const int64_t qs_str = repack_qs_row_stride(GGML_TYPE_Q4_1, ne0);
    const size_t  doff   = (size_t) ne1 * qs_str;

    memset(dst, 0, doff + (size_t) ne1 * n_sub * 4);

    for (int64_t row = 0; row < ne1; row++) {
        const block_q4_1 * brow = blocks + row * n_sub;
        for (int64_t sb = 0; sb < n_sub; sb++) {
            const block_q4_1 * b = &brow[sb];
            const size_t wi = (size_t) row * n_sub + sb;
            memcpy(dst + (size_t) row * qs_str + (size_t) sb * 16, b->qs, 16);
            memcpy(dst + doff + wi * 4, &b->dm, 4);
        }
    }
}

void repack_host(ggml_type type, const void * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    switch (type) {
        case GGML_TYPE_Q8_0:
            repack_q8_0_host((const block_q8_0 *) blocks, dst, ne0, ne1);
            break;
        case GGML_TYPE_MXFP4:
            repack_mxfp4_host((const block_mxfp4 *) blocks, dst, ne0, ne1);
            break;
        case GGML_TYPE_IQ4_NL:
            repack_iq4nl_host((const block_iq4_nl *) blocks, dst, ne0, ne1);
            break;
        case GGML_TYPE_Q4_0:
            // block_q4_0 is byte-identical to block_iq4_nl (f16 d + 16 nibble
            // bytes): the converter is IQ4_NL's routine, wrapped so the SUMW
            // plane (GGML_CUDA_W4A4_SUMW) is filled when the gate is on.
            repack_q40_host((const block_q4_0 *) blocks, dst, ne0, ne1);
            break;
        case GGML_TYPE_Q6_K:
            repack_q6k_host((const block_q6_K *) blocks, dst, ne0, ne1);
            break;
        case GGML_TYPE_Q4_K:
            repack_q4k_host((const block_q4_K *) blocks, dst, ne0, ne1);
            break;
        case GGML_TYPE_Q5_K:
            repack_q5k_host((const block_q5_K *) blocks, dst, ne0, ne1);
            break;
        case GGML_TYPE_Q5_1:
            repack_q51_host((const block_q5_1 *) blocks, dst, ne0, ne1);
            break;
        case GGML_TYPE_Q4_1:
            repack_q41_host((const block_q4_1 *) blocks, dst, ne0, ne1);
            break;
        default:
            GGML_ABORT("unsupported repack type");
    }
}

// Persistent cache for re-packed view buffers. Pool allocs are invalid inside CUDA
// graph capture, so buffers are cudaMalloc'd once per unique view and never freed.
struct RepackViewCacheKey {
    const void * view_data;
    int64_t ne0;
    int64_t ne1;
    int64_t ne2;
    bool operator<(const RepackViewCacheKey & o) const {
        if (view_data != o.view_data) return view_data < o.view_data;
        if (ne0 != o.ne0) return ne0 < o.ne0;
        if (ne1 != o.ne1) return ne1 < o.ne1;
        return ne2 < o.ne2;
    }
};

struct RepackViewCacheEntry {
    uint8_t * d_ptr;
    size_t    size;
};

static std::map<RepackViewCacheKey, RepackViewCacheEntry> s_view_cache;
static std::mutex s_view_cache_mutex;

// Get or create a persistent device buffer for a re-packed view (source may be a
// graph-captured stream; src/dst addresses are stable because the entry persists).
const uint8_t * repack_view_get_cached(
        const ggml_tensor * view, const ggml_tensor * base,
        cudaStream_t stream) {
    // the K-quant multi-plane layouts have no view splice: supports() refuses their views
    GGML_ASSERT(base->type != GGML_TYPE_Q6_K && base->type != GGML_TYPE_Q4_K && base->type != GGML_TYPE_Q5_K && base->type != GGML_TYPE_Q5_1 && base->type != GGML_TYPE_Q4_1 && "repack: multi-plane views are not spliced");
    const int64_t ne0_v = view->ne[0];
    const int64_t ne1_v = view->ne[1];
    const int64_t ne2_v = view->ne[2];
    const int64_t ne0_b = base->ne[0];
    const int64_t ne1_b = base->ne[1];
    const int64_t ne2_b = base->ne[2];
    GGML_ASSERT(ne0_v == ne0_b);
    GGML_ASSERT(base->view_src == nullptr);

    // Per-type strides: MXFP4 rows carry packed nibbles (ne0/2 B) with 1 B
    // e8m0 scales, Q8_0 rows carry ne0 B with 2 B f16 scales.
    const int64_t qs_str  = repack_qs_row_stride(base->type, ne0_v);
    const int64_t sc_row  = repack_scale_row_bytes(base->type, ne0_v);
    const uint8_t * base_ptr = (const uint8_t *) base->data;
    const uint8_t * view_ptr = (const uint8_t *) view->data;

    const int64_t row_start = (int64_t)(view_ptr - base_ptr) / qs_str;
    GGML_ASSERT((view_ptr - base_ptr) % qs_str == 0);

    // Copy every expert in the view (ne1_v x ne2_v), not just ne1_v rows.
    const size_t qs_size = (size_t) ne1_v * ne2_v * qs_str;
    const size_t sc_size = (size_t) ne1_v * ne2_v * sc_row;
    // SUMW plane (GGML_CUDA_W4A4_SUMW): Q4_0 views splice the third plane too,
    // from after the FULL base d plane, with the same row_start offset logic.
    const int64_t n_sub   = ne0_v / 32;
    const size_t  sw_size = (base->type == GGML_TYPE_Q4_0 && repack_sumw_plane_on())
                          ? (size_t) ne1_v * ne2_v * n_sub * 2 : (size_t) 0;
    const size_t total   = qs_size + sc_size + sw_size;

    RepackViewCacheKey key{ view->data, ne0_v, ne1_v, ne2_v };

    std::lock_guard<std::mutex> lock(s_view_cache_mutex);
    auto it = s_view_cache.find(key);
    if (it != s_view_cache.end()) {
        return it->second.d_ptr;
    }

    uint8_t * d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, total));

    // QS plane: copy from the view's position in the base QS plane.
    CUDA_CHECK(cudaMemcpyAsync(d_ptr, view_ptr, qs_size,
        cudaMemcpyDeviceToDevice, stream));

    // Scale plane: starts after the FULL base QS plane.
    const uint8_t * src_scales = base_ptr + (size_t) ne1_b * ne2_b * qs_str
                                       + (size_t) row_start * sc_row;
    CUDA_CHECK(cudaMemcpyAsync(d_ptr + qs_size, src_scales, sc_size,
        cudaMemcpyDeviceToDevice, stream));

    // SUMW plane (GGML_CUDA_W4A4_SUMW): starts after the FULL base d plane,
    // same row_start splice as the scale plane. Gate off: sw_size == 0, no-op.
    if (sw_size > 0) {
        const uint8_t * src_sw = base_ptr + (size_t) ne1_b * ne2_b * (qs_str + sc_row)
                                       + (size_t) row_start * n_sub * 2;
        CUDA_CHECK(cudaMemcpyAsync(d_ptr + qs_size + sc_size, src_sw, sw_size,
            cudaMemcpyDeviceToDevice, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    s_view_cache[key] = { d_ptr, total };
    return d_ptr;
}

void repack_view_cache_purge(int device, const void * base, size_t size) {
    if (base == nullptr || size == 0) {
        return;
    }
    const uint8_t * lo = (const uint8_t *) base;
    const uint8_t * hi = lo + size;
    ggml_cuda_set_device(device);
    std::lock_guard<std::mutex> lock(s_view_cache_mutex);
    for (auto it = s_view_cache.begin(); it != s_view_cache.end(); ) {
        const uint8_t * v = (const uint8_t *) it->first.view_data;
        if (v >= lo && v < hi) {
            CUDA_CHECK(cudaFree(it->second.d_ptr));
            ggml_scratch_gen_bump("repack view cache free");
            it = s_view_cache.erase(it);
        } else {
            ++it;
        }
    }
}


// Host reference quantizer: fp32 -> block_q8_1_u4 (scheme B, per-32 asymmetric
// uint4). Vectorized reference impl: one pass for min/max, one pass to pack the
// nibble pairs. k must be a multiple of 32. NOT dispatched anywhere yet
// (milestone 2 wires it into mul_mat_vec_repacked_nc / the GEMM path).
// FIXED 2026-10-22: this reference now packs the SHIPPED device convention -
// two's-complement nibbles a=(q-8)&0xF - matching quantize.cu and the sealed
// epilogue (settled by seals/nibble_layout_harness.cc: pairing was already
// identical, only signedness differed). Safe: no dispatch caller.
void quantize_row_q8_1_u4(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    GGML_ASSERT(k % 32 == 0);

    block_q8_1_u4 * GGML_RESTRICT y = (block_q8_1_u4 *) vy;
    const int nb = (int) (k / 32);

    for (int ib = 0; ib < nb; ++ib) {
        const float * GGML_RESTRICT xb = x + ib * 32;

        float mn = xb[0], mx = xb[0];
        for (int i = 1; i < 32; ++i) {
            mn = mn < xb[i] ? mn : xb[i];
            mx = mx > xb[i] ? mx : xb[i];
        }

        const float d  = (mx - mn) / 15.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;

        uint8_t * GGML_RESTRICT q = y[ib].qs;
        // De-interleaved byte order mirroring canonical Q4_0 weights
        // (ggml-quants.c:137-141): byte j = {value j low, value j+16 high},
        // so sdot8 against the repacked weight pairs value i with value i.
        for (int i = 0; i < 16; ++i) {
            int q0 = (int) roundf((xb[i]     - mn) * id);
            int q1 = (int) roundf((xb[i + 16] - mn) * id);
            q0 = q0 < 0 ? 0 : (q0 > 15 ? 15 : q0);
            q1 = q1 < 0 ? 0 : (q1 > 15 ? 15 : q1);
            q[i] = (uint8_t) (((q0 - 8) & 0xF) | (((q1 - 8) & 0xF) << 4));   // byte j = {j low, j+16 high}, signed (q-8)&0xF
        }

        y[ib].d4 = (ggml_half) d;
        y[ib].m  = (ggml_half) mn;
    }
}
