#include "tp-allreduce.cuh"

#include <type_traits>

#if defined(GGML_USE_HIP)
#define AR_MEMCPY_FROM_SYMBOL hipMemcpyFromSymbol
#define AR_MEMCPY_TO_SYMBOL   hipMemcpyToSymbol
#else
#define AR_MEMCPY_FROM_SYMBOL cudaMemcpyFromSymbol
#define AR_MEMCPY_TO_SYMBOL   cudaMemcpyToSymbol
#endif

namespace ggml_cuda_tp {

// GGML_TP_AR_SLEEP: back off the barrier spin with s_sleep(1) instead of a bare
// spin loop. A tight spin monopolises the SM's issue slots on gfx906 and can
// starve the peer-write pipeline the same CTA still has to drain.
// Declared before the barriers because they read it.
__device__ int g_ar_sleep_on = 0;

// ============================================================================
// Barrier primitives — GPU-side synchronization across devices
// ============================================================================
//
// Protocol (per CTA block):
//   1. Writer: store incremented flag to PEER's signal buffer (cross-device write)
//   2. Reader: poll OWN signal buffer until all peers have written (local read)
//   3. __syncthreads() to synchronize within the block
//
// Signal buffers are in fine-grained/uncached memory so cross-device writes
// are immediately visible without explicit cache flushes.

#if defined(GGML_USE_MUSA)
// ---- MUSA path (compile-only) ----
// This fork does not target MUSA, and the custom AllReduce is never enabled on
// MUSA at run time (the coherence gate is AMD/NVIDIA only). These device helpers
// exist solely so ggml-musa compiles. MUSA cannot parse the PTX inline asm the
// NVIDIA path uses, so we fall back to portable system-scope fences.

static __device__ __forceinline__ void st_flag_volatile(FlagType * flag_addr, FlagType flag) {
    __threadfence_system();
    *reinterpret_cast<volatile FlagType *>(flag_addr) = flag;
}

static __device__ __forceinline__ FlagType ld_flag_volatile(FlagType * flag_addr) {
    FlagType flag = *reinterpret_cast<volatile FlagType *>(flag_addr);
    __threadfence_system();
    return flag;
}

template <int NRANKS>
static __device__ __forceinline__ void barrier_start(
        const RankSignals & sg, Signal * self_sg, int rank) {
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < NRANKS) {
        st_flag_volatile(&sg.signals[threadIdx.x]->start[blockIdx.x][rank], flag);
        while (ld_flag_volatile(&self_sg->start[blockIdx.x][threadIdx.x]) != flag)
            ;
    }
    __syncthreads();
    if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

template <int NRANKS>
static __device__ __forceinline__ void barrier_end(
        const RankSignals & sg, Signal * self_sg, int rank) {
    __syncthreads();
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < NRANKS) {
        // Release/acquire like barrier_start: the flag store must not pass the
        // preceding payload peer-writes. Hardening - the HIP path always had
        // this ordering.
        st_flag_volatile(&sg.signals[threadIdx.x]->end[blockIdx.x][rank], flag);
        while (ld_flag_volatile(&self_sg->end[blockIdx.x][threadIdx.x]) != flag)
            ;
    }
    if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

#elif !defined(GGML_USE_HIP)
// ---- NVIDIA CUDA path ----

static __device__ __forceinline__ void st_flag_volatile(FlagType * flag_addr, FlagType flag) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.u32 [%1], %0;" :: "r"(flag), "l"(flag_addr));
#else
    asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" :: "r"(flag), "l"(flag_addr));
#endif
}

static __device__ __forceinline__ FlagType ld_flag_volatile(FlagType * flag_addr) {
    FlagType flag;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
#else
    asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" : "=r"(flag) : "l"(flag_addr));
#endif
    return flag;
}

template <int NRANKS>
static __device__ __forceinline__ void barrier_start(
        const RankSignals & sg, Signal * self_sg, int rank) {
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < NRANKS) {
        // Write our flag to peer threadIdx.x's signal buffer
        st_flag_volatile(&sg.signals[threadIdx.x]->start[blockIdx.x][rank], flag);
        // Wait until peer threadIdx.x has written to our buffer
        while (ld_flag_volatile(&self_sg->start[blockIdx.x][threadIdx.x]) != flag)
            ;
    }
    __syncthreads();
    if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

template <int NRANKS>
static __device__ __forceinline__ void barrier_end(
        const RankSignals & sg, Signal * self_sg, int rank) {
    __syncthreads();
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < NRANKS) {
        // Relaxed semantics — no downstream reads depend on this barrier's data.
        asm volatile("st.volatile.global.u32 [%1], %0;" :: "r"(flag),
                     "l"(&sg.signals[threadIdx.x]->end[blockIdx.x][rank]));
        FlagType val;
        do {
            asm volatile("ld.volatile.global.u32 %0, [%1];"
                         : "=r"(val)
                         : "l"(&self_sg->end[blockIdx.x][threadIdx.x]));
        } while (val != flag);
    }
    if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

#else
// ---- AMD HIP/ROCm path ----
// On MI50/gfx906 PCIe (no XGMI), RELEASE/ACQUIRE is required for correctness —
// RELAXED produces garbage even with fine-grained staging (tested). vLLM uses
// RELAXED on MI300X where XGMI provides HW cache coherence. PCIe has none;
// we rely on the C++11 atomic ordering to emit the right fences.
//
// SYSTEM scope on both sides is required: the load must re-fetch HBM each
// iteration (DEVICE scope + RELAXED can cache in L1/L2 even for "uncached"
// memory under some access patterns).

template <int NRANKS>
static __device__ __forceinline__ void barrier_start(
        const RankSignals & sg, Signal * self_sg, int rank) {
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < NRANKS) {
        __hip_atomic_store(
            &sg.signals[threadIdx.x]->start[blockIdx.x][rank],
            flag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        while (__hip_atomic_load(
                   &self_sg->start[blockIdx.x][threadIdx.x],
                   __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < flag) {
            if (g_ar_sleep_on) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

template <int NRANKS>
static __device__ __forceinline__ void barrier_end(
        const RankSignals & sg, Signal * self_sg, int rank) {
    __syncthreads();
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if (threadIdx.x < NRANKS) {
        __hip_atomic_store(
            &sg.signals[threadIdx.x]->end[blockIdx.x][rank],
            flag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        while (__hip_atomic_load(
                   &self_sg->end[blockIdx.x][threadIdx.x],
                   __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < flag) {
            if (g_ar_sleep_on) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
    }
    if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
}

#endif // GGML_USE_HIP

// ============================================================================
// One-shot AllReduce kernel
// ============================================================================
//
// Each rank reads from ALL ranks' partial buffers via peer access, sums locally,
// writes result to its own output buffer. Uses 128-bit packed loads (float4)
// for bandwidth efficiency.
//
// Data flow:
//   barrier_start  — ensure all GEMMs have written their partials
//   packed reduce  — read float4 from each peer, sum, write to local output
//   barrier_end    — ensure all reads complete before next op overwrites partials

template <int NRANKS>
__global__ void __launch_bounds__(kThreads, 1)
k_cross_device_reduce_1stage(
        RankData * __restrict__ _dp,
        RankSignals              sg,
        Signal *   __restrict__  self_sg,
        float *    __restrict__  result,
        int                      rank,
        int64_t                  n_elements) {

    // Copy RankData to registers (64 bytes — fits in register file).
    // Passing as a pointer + loading once keeps SGPR pressure low on gfx906
    // (versus pass-by-value, which bloats the kernel arg region and hurts
    // occupancy; measured ~6% TG regression on 4×MI50).
    auto dp = *_dp;

    // Start barrier: all ranks' partial data is ready
    barrier_start<NRANKS>(sg, self_sg, rank);

    // Reduce using float4 (128-bit packed loads/stores). Use non-temporal loads
    // for peer reads on HIP — peer data is one-shot (no reuse) and normal
    // cached loads can return stale L2 entries from prior ARs on MI50 without
    // XGMI cache coherence. NT loads go direct to HBM over PCIe.
    const int64_t n_float4 = n_elements / 4;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_float4;
         idx += (int64_t)gridDim.x * blockDim.x) {

        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
#if defined(GGML_USE_HIP)
            const float * base = ((const float *)dp.ptrs[r]) + idx * 4;
            const float x = __builtin_nontemporal_load(base + 0);
            const float y = __builtin_nontemporal_load(base + 1);
            const float z = __builtin_nontemporal_load(base + 2);
            const float w = __builtin_nontemporal_load(base + 3);
#else
            const float4 v = ((const float4 *)dp.ptrs[r])[idx];
            const float x = v.x, y = v.y, z = v.z, w = v.w;
#endif
            sum.x += x;
            sum.y += y;
            sum.z += z;
            sum.w += w;
        }
        ((float4 *)result)[idx] = sum;
    }

    // Handle remainder (n_elements not divisible by 4)
    const int64_t rem_start = n_float4 * 4;
    for (int64_t idx = rem_start + (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += (int64_t)gridDim.x * blockDim.x) {

        float sum = 0.0f;
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
#if defined(GGML_USE_HIP)
            sum += __builtin_nontemporal_load(((const float *)dp.ptrs[r]) + idx);
#else
            sum += ((const float *)dp.ptrs[r])[idx];
#endif
        }
        result[idx] = sum;
    }

    // End barrier: all reads complete — safe for next AllReduce to overwrite partials
    barrier_end<NRANKS>(sg, self_sg, rank);
}

// ----------------------------------------------------------------------------
// Broadcast + reduce kernel (default for small / TG-size ARs).
// Each rank writes its input into *every peer's* staging at offset
// rank*n_elements via PCIe peer access. PCIe peer writes go directly to the
// peer's HBM (bypassing both source and destination L2), so the producer's
// data is peer-visible without the usual cudaEventRecord/StreamWaitEvent
// L2-flush handshake.
//
// Staging layout (per rank): [N_ranks][n_elements] — rank R's input from peer
// R lives at offset R*n_elements. "Self slot" (R==rank) is unused; we read
// our own contribution from `input` (local L2-coherent read).
//
// Correctness contract: the in-kernel barrier's SYSTEM-scope RELEASE atomic
// flag store is itself a PCIe peer write; PCIe ordering guarantees writes
// from the same source arrive in-order, so the flag arrives after all prior
// data writes. Peer sees our flag → all our prior data writes are in peer's
// HBM.
//
// Each rank's outbound traffic: (N-1)*S bytes of PCIe writes (posted, async,
// low latency). Same aggregate BW as one-shot reads, but writes typically
// beat reads on PCIe 3.0.
// ----------------------------------------------------------------------------
// [0]=peer-write, [1]=barrier_start wait, [2]=reduce, [3]=barrier_end,
// [4]=sample count. Ticks are shader-clock, so only RATIOS are meaningful.
__device__ unsigned long long g_ar_ktime[5] = {0, 0, 0, 0, 0};
__device__ int                g_ar_ktime_on = 0;
// GGML_TP_AR_SKIP_XBAR: broadcast-path end-barrier elision via parity
// double-buffering (see k_broadcast_reduce). Uploaded once from env.
__device__ int                g_ar_xbar_on = 0;

template <int NRANKS>
__global__ void __launch_bounds__(kThreads, 1)
k_broadcast_reduce(
        RankData * __restrict__ _dp,        // peer (and self) staging pointers
        RankSignals              sg,
        Signal *   __restrict__  self_sg,
        const float * __restrict__ input,   // our own input (local device)
        float *    __restrict__  result,    // our own output (local device)
        int                      rank,
        int64_t                  n_elements) {

    auto dp = *_dp;

    // GGML_TP_AR_SKIP_XBAR: elide barrier_end on the broadcast path by
    // double-buffering the staging inbox per parity of the per-block call
    // counter. barrier_end's only cross-call duty is preventing call N+1's
    // phase-1 peer writes from overwriting call N's phase-2 read sources
    // before every rank finished reading them. With parities alternating,
    // call N+1 writes buffer set (N+1)%2 while call N reads set N%2, so no
    // overlap exists and the barrier is redundant. Cross-call safety is
    // otherwise provided by barrier_start: a rank can only enter call N+1's
    // start barrier after all peers published call N+1's data-ready flags,
    // which requires them to have finished call N's reduce (and passed its
    // end flag store when the barrier is present). Stream ordering per rank
    // serializes its own phase-1 writes after its phase-2 reads regardless.
    // Parity is derived in-kernel from _flag[blockIdx.x] so captured graphs
    // (whole-token graph capture) alternate sets on each replay without any
    // host-side state. Reduction order is untouched: bit-exact.
    // When the flag is off (default) the barrier runs and staging is the
    // original single set, exactly stock behavior.
    const bool use_xbar = (g_ar_xbar_on != 0);
    uint32_t parity = 0;
    if (use_xbar) {
        // _flag advances +2 per stock call (once per barrier), so bit 0 never
        // flips between calls; bit 1 flips exactly once per call. In xbar mode
        // the epilogue also advances +1 (total +2), keeping the cadence, so
        // (_flag >> 1) & 1 alternates every call on every block. Mixed
        // broadcast/twoshot callers both advance +2, so parity stays sane.
        parity = (self_sg->_flag[blockIdx.x] >> 1) & 1u;
        // staging layout with XBAR: [2 sets][nranks slots][n_elements].
        // Shift every rank's staging pointer to this call's set.
        const int64_t par_off = (int64_t) parity * NRANKS * n_elements;
        for (int r = 0; r < NRANKS; r++) {
            dp.ptrs[r] = (const void *)(((float *)(uintptr_t) dp.ptrs[r]) + par_off);
        }
    }

    const bool ktm = (g_ar_ktime_on != 0) && (blockIdx.x == 0) && (threadIdx.x == 0);
    unsigned long long kt0 = 0, kt1 = 0, kt2 = 0;
    if (ktm) { kt0 = clock64(); }

    // Phase 1: write our input into each PEER's staging at offset rank*n_elements.
    // We do NOT write the self-slot. Writing the self-slot would be a local
    // store to fine-grained memory, which empirically does NOT bypass L2 on
    // gfx906 even under FGP=1 — phase 2's NT load from HBM would then miss
    // the fresh self-data. This corrupts ~1/N of the reduction inputs and
    // degrades PPL (~+2 on a 10-baseline at ubatch=32, tested). Reading our
    // own contribution directly from `input` in phase 2 (local L2-coherent
    // read) avoids that path entirely.
    const int64_t n_float4 = n_elements / 4;
    const int64_t rank_offset = (int64_t) rank * n_elements;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_float4;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const float4 v = ((const float4 *) input)[idx];
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
            if (r == rank) continue;   // self-slot is NOT peer-written; read `input` directly in phase 2
            float * dst = ((float *)(uintptr_t)dp.ptrs[r]) + rank_offset + idx * 4;
            *((float4 *) dst) = v;     // peer write (bypasses source + dest L2 via PCIe)
        }
    }
    const int64_t rem_start = n_float4 * 4;
    for (int64_t idx = rem_start + (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const float v = input[idx];
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
            if (r == rank) continue;
            ((float *)(uintptr_t)dp.ptrs[r])[rank_offset + idx] = v;
        }
    }

    // Ensure every thread in this CTA has finished its peer writes before the
    // first NRANKS threads publish the completion flag below.
    __syncthreads();

    // Force all prior peer-writes to retire before we signal via barrier.
    // The RELEASE-scope atomic inside barrier_start should do this on its own,
    // but an explicit system fence is cheap and proved necessary for a 0.18
    // PPL drift that the atomic-alone variant had at ubatch=32.
    __threadfence_system();

    if (ktm) { kt1 = clock64(); }

    // Barrier: SYSTEM-scope RELEASE flag store is itself a peer write; PCIe
    // same-source ordering guarantees our prior data writes arrive first.
    barrier_start<NRANKS>(sg, self_sg, rank);

    if (ktm) { kt2 = clock64(); }

    // Phase 2: identical reduction order on EVERY rank:
    //   sum = 0 + p0 + p1 + p2 + ... + p_{N-1}
    // For r == rank we read `input` directly (local L2-coherent); for peers we
    // NT-load from the per-peer slot in our own staging (peer-written → HBM).
    // The `if (r == rank)` branch inside a #pragma unroll + compile-time NRANKS
    // resolves at compile time per unrolled iteration, so the emitted code has
    // no runtime branch — same accumulation sequence on all ranks.
    const float * local_staging = (const float *) dp.ptrs[rank];
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_float4;
         idx += (int64_t)gridDim.x * blockDim.x) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
            float4 v;
            if (r == rank) {
                v = ((const float4 *) input)[idx];   // self contribution
            } else {
                const float * base = local_staging + (int64_t)r * n_elements + idx * 4;
#if defined(GGML_USE_HIP)
                v.x = __builtin_nontemporal_load(base + 0);
                v.y = __builtin_nontemporal_load(base + 1);
                v.z = __builtin_nontemporal_load(base + 2);
                v.w = __builtin_nontemporal_load(base + 3);
#else
                v = *((const float4 *) base);
#endif
            }
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
        }
        ((float4 *) result)[idx] = sum;
    }
    for (int64_t idx = rem_start + (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += (int64_t)gridDim.x * blockDim.x) {
        float sum = 0.0f;
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
            float v;
            if (r == rank) {
                v = input[idx];
            } else {
                const float * base = local_staging + (int64_t)r * n_elements + idx;
#if defined(GGML_USE_HIP)
                v = __builtin_nontemporal_load(base);
#else
                v = *base;
#endif
            }
            sum += v;
        }
        result[idx] = sum;
    }

    unsigned long long kt3 = 0;
    if (ktm) { kt3 = clock64(); }

    if (use_xbar) {
        // XBAR: barrier_end elided (see the parity scheme at kernel top).
        // Advance the block's call counter so the NEXT call lands on the
        // other parity set, and keep the flag arithmetic of the stock path
        // (barrier_start consumes _flag + 1 each call) intact: skipping the
        // barrier would otherwise stall the next barrier_start's spin, since
        // it waits for flag == _flag + 1 with _flag only advanced by the
        // barriers themselves. Store the counter from thread 0 only, after
        // all threads finished the reduce (they did - no code follows).
        __syncthreads();
        if (threadIdx.x == 0) {
            self_sg->_flag[blockIdx.x] = self_sg->_flag[blockIdx.x] + 1;
        }
    } else {
        barrier_end<NRANKS>(sg, self_sg, rank);
    }

    if (ktm) {
        const unsigned long long kt4 = clock64();
        atomicAdd(&g_ar_ktime[0], kt1 - kt0);
        atomicAdd(&g_ar_ktime[1], kt2 - kt1);
        atomicAdd(&g_ar_ktime[2], kt3 - kt2);
        atomicAdd(&g_ar_ktime[3], kt4 - kt3);
        atomicAdd(&g_ar_ktime[4], 1ULL);
    }
}

// ----------------------------------------------------------------------------
// bf16 wire helpers + broadcast kernel (GGML_TP_AR_BF16_THRESHOLD)
//
// The F32 broadcast path moves 4 bytes per element over PCIe. A bf16 wire
// halves that. Conversion is done with bit operations (round-to-nearest-even)
// so the code does not depend on bf16 intrinsic availability per toolchain.
// The local contribution is still read in F32 and the reduction accumulator is
// still F32; only the peer-visible copy is narrowed.
// ----------------------------------------------------------------------------
static __device__ __forceinline__ uint16_t ar_f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    // round-to-nearest-even on the 16 discarded mantissa bits
    const uint32_t r = u + 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t) (r >> 16);
}

static __device__ __forceinline__ float ar_bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

template <int NRANKS>
__global__ void __launch_bounds__(kThreads, 1)
k_broadcast_reduce_bf16(
        RankData * __restrict__ _dp,
        RankSignals              sg,
        Signal *   __restrict__  self_sg,
        const float * __restrict__ input,
        float *    __restrict__  result,
        int                      rank,
        int64_t                  n_elements) {

    auto dp = *_dp;

    const bool use_xbar = (g_ar_xbar_on != 0);
    uint32_t parity = 0;
    if (use_xbar) {
        parity = (self_sg->_flag[blockIdx.x] >> 1) & 1u;
        // parity offset in bf16 elements (staging slot = n_elements bf16 values)
        const int64_t par_off = (int64_t) parity * NRANKS * n_elements;
        for (int r = 0; r < NRANKS; r++) {
            dp.ptrs[r] = (const void *) (((uint16_t *)(uintptr_t) dp.ptrs[r]) + par_off);
        }
    }

    const bool ktm = (g_ar_ktime_on != 0) && (blockIdx.x == 0) && (threadIdx.x == 0);
    unsigned long long kt0 = 0, kt1 = 0, kt2 = 0;
    if (ktm) { kt0 = clock64(); }

    const int64_t n_quad     = n_elements / 4;
    const int64_t rank_offset = (int64_t) rank * n_elements;

    // Phase 1: narrow our input and peer-write 4 elements per 8-byte store.
    for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_quad;
         idx += (int64_t) gridDim.x * blockDim.x) {
        const float4 v = ((const float4 *) input)[idx];
        const uint32_t lo = (uint32_t) ar_f32_to_bf16(v.x) | ((uint32_t) ar_f32_to_bf16(v.y) << 16);
        const uint32_t hi = (uint32_t) ar_f32_to_bf16(v.z) | ((uint32_t) ar_f32_to_bf16(v.w) << 16);
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
            if (r == rank) continue;
            uint32_t * dst = (uint32_t *) ((uint16_t *)(uintptr_t) dp.ptrs[r] + rank_offset + idx * 4);
            dst[0] = lo;
            dst[1] = hi;
        }
    }
    const int64_t rem_start = n_quad * 4;
    for (int64_t idx = rem_start + (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += (int64_t) gridDim.x * blockDim.x) {
        const uint16_t b = ar_f32_to_bf16(input[idx]);
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
            if (r == rank) continue;
            ((uint16_t *)(uintptr_t) dp.ptrs[r])[rank_offset + idx] = b;
        }
    }

    __syncthreads();
    __threadfence_system();

    if (ktm) { kt1 = clock64(); }

    barrier_start<NRANKS>(sg, self_sg, rank);

    if (ktm) { kt2 = clock64(); }

    // Phase 2: same accumulation order on every rank; self slot comes from
    // `input` in full F32, peers are widened back from bf16.
    const uint16_t * local_staging = (const uint16_t *) dp.ptrs[rank];
    for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_quad;
         idx += (int64_t) gridDim.x * blockDim.x) {
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
            float4 v;
            if (r == rank) {
                v = ((const float4 *) input)[idx];
            } else {
                const uint32_t * base = (const uint32_t *) (local_staging + (int64_t) r * n_elements + idx * 4);
#if defined(GGML_USE_HIP)
                const uint32_t lo = __builtin_nontemporal_load(base + 0);
                const uint32_t hi = __builtin_nontemporal_load(base + 1);
#else
                const uint32_t lo = base[0];
                const uint32_t hi = base[1];
#endif
                v.x = ar_bf16_to_f32((uint16_t) (lo & 0xFFFFu));
                v.y = ar_bf16_to_f32((uint16_t) (lo >> 16));
                v.z = ar_bf16_to_f32((uint16_t) (hi & 0xFFFFu));
                v.w = ar_bf16_to_f32((uint16_t) (hi >> 16));
            }
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
        }
        ((float4 *) result)[idx] = sum;
    }
    for (int64_t idx = rem_start + (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += (int64_t) gridDim.x * blockDim.x) {
        float sum = 0.0f;
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
            const float v = (r == rank)
                ? input[idx]
                : ar_bf16_to_f32(local_staging[(int64_t) r * n_elements + idx]);
            sum += v;
        }
        result[idx] = sum;
    }

    unsigned long long kt3 = 0;
    if (ktm) { kt3 = clock64(); }

    if (use_xbar) {
        __syncthreads();
        if (threadIdx.x == 0) {
            self_sg->_flag[blockIdx.x] = self_sg->_flag[blockIdx.x] + 1;
        }
    } else {
        barrier_end<NRANKS>(sg, self_sg, rank);
    }

    if (ktm) {
        const unsigned long long kt4 = clock64();
        atomicAdd(&g_ar_ktime[0], kt1 - kt0);
        atomicAdd(&g_ar_ktime[1], kt2 - kt1);
        atomicAdd(&g_ar_ktime[2], kt3 - kt2);
        atomicAdd(&g_ar_ktime[3], kt4 - kt3);
        atomicAdd(&g_ar_ktime[4], 1ULL);
    }
}

// ----------------------------------------------------------------------------
// Chunked broadcast kernel for LARGE tensors (GGML_TP_AR_CHUNKED).
//
// Why: at TP prefill every AllReduce boundary is [n_tokens, n_embd]-sized
// (megabytes). The stock size gate routes those to the backend fallback
// (NCCL/RCCL ring), which measures ~1.75 ms per 10.5 MB AR on PCIe 3.0 x16 -
// the single largest cost in TP2 prefill. This kernel keeps the peer-write
// design but loops over the tensor in fixed PIECE-sized chunks, so staging is
// a small fixed allocation made once (no per-size realloc, no hipFree drain,
// and - unlike a lazy-growth staging - safe to run inside a stream capture).
//
// Numerics: F32 wire, same accumulation order as k_broadcast_reduce (self
// contribution read from `input`, peers summed in rank order), so results are
// bit-identical to the broadcast path. One barrier pair per piece; stream
// ordering makes single-set staging safe across pieces and across calls.
// ----------------------------------------------------------------------------
template <int NRANKS>
__global__ void __launch_bounds__(kThreads, 1)
k_broadcast_chunked(
        RankData * __restrict__ _dp,
        RankSignals              sg,
        Signal *   __restrict__  self_sg,
        const float * __restrict__ input,
        float *    __restrict__  result,
        int                      rank,
        int64_t                  n_elements,
        int64_t                  piece_elems) {

    auto dp = *_dp;

    const int64_t n_pieces = (n_elements + piece_elems - 1) / piece_elems;
    const int64_t rank_slot = (int64_t) rank * piece_elems;

    for (int64_t p = 0; p < n_pieces; p++) {
        const int64_t beg = p * piece_elems;
        const int64_t cnt = std::min(piece_elems, n_elements - beg);

        const bool ktm = (g_ar_ktime_on != 0) && (blockIdx.x == 0) && (threadIdx.x == 0) && (p == 0);
        unsigned long long kt0 = 0, kt1 = 0, kt2 = 0;
        if (ktm) { kt0 = clock64(); }

        // Phase 1: peer-write our slice of this piece (self slot NOT written -
        // local stores do not bypass L2 on gfx906; phase 2 reads `input`).
        const int64_t n_vec = cnt / 4;
        for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
             idx < n_vec;
             idx += (int64_t) gridDim.x * blockDim.x) {
            const float4 v = ((const float4 *) input)[beg / 4 + idx];
#pragma unroll
            for (int r = 0; r < NRANKS; r++) {
                if (r == rank) continue;
                ((float4 *) (((float *)(uintptr_t) dp.ptrs[r]) + rank_slot))[idx] = v;
            }
        }
        for (int64_t idx = n_vec * 4 + (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
             idx < cnt;
             idx += (int64_t) gridDim.x * blockDim.x) {
            const float v = input[beg + idx];
#pragma unroll
            for (int r = 0; r < NRANKS; r++) {
                if (r == rank) continue;
                ((float *)(uintptr_t) dp.ptrs[r])[rank_slot + idx] = v;
            }
        }

        __syncthreads();
        __threadfence_system();
        if (ktm) { kt1 = clock64(); }
        barrier_start<NRANKS>(sg, self_sg, rank);
        if (ktm) { kt2 = clock64(); }

        // Phase 2: reduce this piece. Same rank order as k_broadcast_reduce.
        const float * local_staging = (const float *) dp.ptrs[rank];
        for (int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
             idx < n_vec;
             idx += (int64_t) gridDim.x * blockDim.x) {
            float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
            for (int r = 0; r < NRANKS; r++) {
                float4 v;
                if (r == rank) {
                    v = ((const float4 *) input)[beg / 4 + idx];
                } else {
                    const float * base = local_staging + (int64_t) r * piece_elems + idx * 4;
#if defined(GGML_USE_HIP)
                    v.x = __builtin_nontemporal_load(base + 0);
                    v.y = __builtin_nontemporal_load(base + 1);
                    v.z = __builtin_nontemporal_load(base + 2);
                    v.w = __builtin_nontemporal_load(base + 3);
#else
                    v = *((const float4 *) base);
#endif
                }
                sum.x += v.x;
                sum.y += v.y;
                sum.z += v.z;
                sum.w += v.w;
            }
            ((float4 *) result)[beg / 4 + idx] = sum;
        }
        for (int64_t idx = n_vec * 4 + (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
             idx < cnt;
             idx += (int64_t) gridDim.x * blockDim.x) {
            float sum = 0.0f;
#pragma unroll
            for (int r = 0; r < NRANKS; r++) {
                const float v = (r == rank)
                    ? input[beg + idx]
                    : __builtin_nontemporal_load(local_staging + (int64_t) r * piece_elems + idx);
                sum += v;
            }
            result[beg + idx] = sum;
        }

        // barrier_end per piece: the next piece's peer-writes must not overwrite
        // this piece's staging before every rank finished reading it. Stream
        // ordering alone covers ACROSS calls; within the call the pieces are
        // back-to-back on the same stream, so the end barrier is what makes
        // single-set staging safe.
        barrier_end<NRANKS>(sg, self_sg, rank);

        if (ktm) {
            const unsigned long long kt3 = clock64();
            atomicAdd(&g_ar_ktime[0], kt1 - kt0);
            atomicAdd(&g_ar_ktime[1], kt2 - kt1);
            atomicAdd(&g_ar_ktime[2], kt3 - kt2);
            atomicAdd(&g_ar_ktime[3], 0ULL);
            atomicAdd(&g_ar_ktime[4], 1ULL);
        }
    }
}

// ----------------------------------------------------------------------------
// Two-shot F32 AllReduce (reduce-scatter + allgather via peer writes).
//
// Target: large (PP-size) messages where the broadcast kernel's (N-1)*S
// per-rank outbound saturates PCIe. Two-shot brings per-rank outbound down
// to 2*(N-1)/N * S, with parallel all-to-all peer writes instead of NCCL's
// serial ring — wins on PCIe systems where every GPU has its own root-
// complex link AND the fabric can carry parallel transfers concurrently.
//
// LOSSLESS: F32 input → F32 wire → F32 reduction (F32 accumulator) → F32
// output. No BF16 round-trip anywhere; output is bit-deterministic and a
// strictly lossless function of the inputs (modulo the standard fp non-
// associativity of summation, which is fixed across runs because all ranks
// sum in the same r-order). Costs 2× the PCIe bytes per AR vs a BF16-on-
// wire variant; the trade-off vs NCCL BF16 ring is BW for precision.
//
// Staging layout (per rank, bytes):
//   [0, n_elements * 4)                    : scatter_buf — N F32 slots, each
//                                            of slice = N_elements/N elements.
//                                            Slot r holds rank r's contribution
//                                            to OUR slice (my_rank). Filled in
//                                            stage 1 by peer writes.
//   [ag_offset * 4, (ag_offset + n_elements) * 4) : allgather_buf, N F32 slots in the same layout.
//                                            Slot r holds rank r's reduced slice, filled in stage 3 by peer writes from rank r.
// dp.ptrs[r] points to rank r's scatter_buf base and allgather_buf starts ag_offset floats above it.
// ag_offset is half of the staging allocation, not n_elements.
// A rank leaves the kernel as soon as its own stage 4 copy is done, while a peer can still be copying out of its allgather_buf.
// With the allgather region at n_elements, a larger next call scattered into [0, n_new) on top of the peer's live reads of [n_old, 2 * n_old) and that rank reduced garbage.
// The upper half lies above every scatter region the allocation can hold, so no third barrier is needed.
//
// Per-rank peer traffic: stage 1 (N-1) * slice F32 + stage 3 (N-1) * slice
// F32 = 2*(N-1)/N * S bytes outbound (S = n_elements * sizeof(float)).
//
// Requires n_elements % NRANKS == 0 and slice % 4 == 0 for float4 path. The
// caller (host dispatch) must gate on these before selecting this kernel.
// ----------------------------------------------------------------------------
template <int NRANKS>
__global__ void __launch_bounds__(kThreads, 1)
k_twoshot_f32(
        RankData * __restrict__ _dp,        // peer staging base pointers (float*)
        RankSignals              sg,
        Signal *   __restrict__  self_sg,
        const float * __restrict__ input,   // our own input (F32)
        float *    __restrict__  result,    // our own output (F32)
        int                      rank,
        int64_t                  n_elements,
        int64_t                  ag_offset) {

    auto dp = *_dp;

    const int64_t slice = n_elements / NRANKS;          // guaranteed divisible
    const int64_t slice_n_float4 = slice / 4;
    const int64_t my_slice_start = (int64_t) rank * slice;

    // Staging pointers — each rank's staging holds scatter_buf then allgather_buf.
    // rank r's scatter_buf base   = dp.ptrs[r]              (float*)
    // rank r's allgather_buf base = dp.ptrs[r] + ag_offset, the upper half of the staging allocation
    auto scat_ptr = [&](int r) -> float * { return (float *)(uintptr_t) dp.ptrs[r]; };
    auto ag_ptr   = [&](int r) -> float * { return ((float *)(uintptr_t) dp.ptrs[r]) + ag_offset; };

    // ------------------------------------------------------------------------
    // Stage 1 (peer-write scatter): for each non-self target peer, all threads
    // grid-stride over the target's slice of our input and peer-write F32x4
    // to the target's scatter_buf at OUR slot. Round-robin target order
    // (rank+t) % NRANKS spreads PCIe contention so all ranks write to
    // different targets at the same loop iteration.
    // ------------------------------------------------------------------------
    const int64_t my_slot_offset = (int64_t) rank * slice;
#pragma unroll
    for (int t = 1; t < NRANKS; t++) {
        const int target = (rank + t) % NRANKS;
        const int64_t target_slice_start_f4 = ((int64_t) target * slice) / 4;
        for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
             idx < slice_n_float4;
             idx += (int64_t)gridDim.x * blockDim.x) {
            const float4 v = ((const float4 *) input)[target_slice_start_f4 + idx];
            float4 * dst   = (float4 *)(scat_ptr(target) + my_slot_offset + idx * 4);
            *dst = v;       // 16-byte PCIe peer write
        }
    }

    // Ensure every thread in this CTA has finished its peer writes before the
    // first NRANKS threads publish the completion flag below.
    __syncthreads();
    __threadfence_system();
    barrier_start<NRANKS>(sg, self_sg, rank);

    // ------------------------------------------------------------------------
    // Stage 2 + 3 fused: reduce our slice from scatter_buf + own input, then
    // peer-write the F32 sum to every rank's allgather_buf at OUR slot.
    // Self-contribution is read directly from `input` — no precision adjust
    // needed (peers see our slice as F32 via stage-1 peer write; same value).
    // ------------------------------------------------------------------------
    const float * own_scat = scat_ptr(rank);
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < slice_n_float4;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int64_t local_pos = idx * 4;               // position within slice
        float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int r = 0; r < NRANKS; r++) {
            float4 v;
            if (r == rank) {
                v = ((const float4 *) input)[my_slice_start / 4 + idx];
            } else {
                const float4 * base = (const float4 *)(own_scat + (int64_t) r * slice + local_pos);
#if defined(GGML_USE_HIP)
                v.x = __builtin_nontemporal_load(((const float *) base) + 0);
                v.y = __builtin_nontemporal_load(((const float *) base) + 1);
                v.z = __builtin_nontemporal_load(((const float *) base) + 2);
                v.w = __builtin_nontemporal_load(((const float *) base) + 3);
#else
                v = *base;
#endif
            }
            sum.x += v.x;
            sum.y += v.y;
            sum.z += v.z;
            sum.w += v.w;
        }

        // Allgather peer-write: send our reduced F32 slice to every PEER's
        // allgather_buf[rank_slot][idx*4]. We do NOT local-write to OWN
        // allgather_buf — on gfx906/PCIe, kernel-initiated local writes to
        // fine-grained memory dwell in L2 until kernel exit, while stage-4
        // NT-load bypasses L2 and would see stale HBM. Instead we write our
        // own slice directly to `result` (lossless, no round-trip).
        const int64_t slot_off = my_slice_start;
#pragma unroll
        for (int t = 1; t < NRANKS; t++) {
            const int target = (rank + t) % NRANKS;
            float4 * dst = (float4 *)(ag_ptr(target) + slot_off + local_pos);
            *dst = sum;                                  // 16-byte PCIe peer write
        }
        // Own slice: write F32 sum directly to result.
        ((float4 *) result)[my_slice_start / 4 + idx] = sum;
    }

    __threadfence_system();
    barrier_end<NRANKS>(sg, self_sg, rank);
    // barrier_end has a __syncthreads() at its start but NOT at its end —
    // only the first NRANKS threads spin on peer flags. For kernels that READ
    // peer-written data after barrier_end (stage 4), the remaining threads
    // must wait for the spinning threads to confirm peer writes are visible.
    // Without this sync, threads >= NRANKS run stage-4 NT-loads before peers'
    // stage-3 writes have arrived, returning stale HBM and corrupting output.
    __syncthreads();

    // ------------------------------------------------------------------------
    // Stage 4 (local scatter to F32 result): peer slots only (own slice was
    // written directly to `result` in stage 3). Each peer's allgather_buf
    // slot is FGP-coherent (peer writes bypass source L2 → land in our HBM)
    // so NT loads return the freshly-arrived F32 values.
    // ------------------------------------------------------------------------
    const float * own_ag = ag_ptr(rank);
#pragma unroll
    for (int t = 1; t < NRANKS; t++) {
        const int src_slot = (rank + t) % NRANKS;
        const int64_t slot_base = (int64_t) src_slot * slice;
        for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
             idx < slice_n_float4;
             idx += (int64_t)gridDim.x * blockDim.x) {
            const float4 * base = (const float4 *)(own_ag + slot_base + idx * 4);
#if defined(GGML_USE_HIP)
            float4 v;
            v.x = __builtin_nontemporal_load(((const float *) base) + 0);
            v.y = __builtin_nontemporal_load(((const float *) base) + 1);
            v.z = __builtin_nontemporal_load(((const float *) base) + 2);
            v.w = __builtin_nontemporal_load(((const float *) base) + 3);
#else
            float4 v = *base;
#endif
            ((float4 *) result)[(slot_base / 4) + idx] = v;
        }
    }
}

// ============================================================================
// Host functions
// ============================================================================

void tp_custom_ar_init(CustomARContext * ctx, int nranks, const int * dev_ids) {
    if (ctx->initialized) return;

    ctx->nranks = nranks;
    for (int r = 0; r < nranks; r++) {
        ctx->dev_ids[r] = dev_ids ? dev_ids[r] : r;
    }

    // Enable peer access between all GPU pairs
    bool peer_access_ok = true;
    for (int i = 0; i < nranks; i++) {
        ggml_cuda_set_device(ctx->dev_ids[i]);
        for (int j = 0; j < nranks; j++) {
            if (i == j) continue;
            int can_access = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, ctx->dev_ids[i], ctx->dev_ids[j]));
            if (can_access) {
                cudaError_t err = cudaDeviceEnablePeerAccess(ctx->dev_ids[j], 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                    CUDA_CHECK(err);
                }
                // cudaDeviceEnablePeerAccess leaves the sticky error state set when access
                // was already enabled. Clear it here so the first downstream cudaGetLastError()
                // (e.g. after a MUL_MAT kernel) doesn't trip on it.
                (void) cudaGetLastError();
            } else {
                GGML_LOG_WARN("TP custom AR: peer access not available between GPU %d and %d\n",
                              ctx->dev_ids[i], ctx->dev_ids[j]);
                peer_access_ok = false;
            }
        }
    }
    if (!peer_access_ok) {
        GGML_LOG_WARN("TP custom AR: disabled because full peer access is not available\n");
        return;
    }

    // Allocate signal buffers (one per rank) in fine-grained/uncached memory
    for (int rank = 0; rank < nranks; rank++) {
        ggml_cuda_set_device(ctx->dev_ids[rank]);

#if defined(GGML_USE_HIP)
        // AMD: hipDeviceMallocUncached for cross-device visibility (matches vLLM)
        void * ptr;
        CUDA_CHECK(hipExtMallocWithFlags(&ptr, sizeof(Signal), hipDeviceMallocUncached));
        ctx->d_signals[rank] = (Signal *)ptr;
#else
        // NVIDIA: regular device memory (peer access handles visibility)
        CUDA_CHECK(cudaMalloc(&ctx->d_signals[rank], sizeof(Signal)));
#endif
        CUDA_CHECK(cudaMemset(ctx->d_signals[rank], 0, sizeof(Signal)));

        // Allocate device-side RankData (holds pointers to all ranks' buffers)
        CUDA_CHECK(cudaMalloc(&ctx->d_rank_data[rank], sizeof(RankData)));

        ctx->rank_signals.signals[rank] = ctx->d_signals[rank];

        // Event used for cross-stream handshake before each AR
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx->events[rank], cudaEventDisableTiming));
    }

    // Determine whether the broadcast path is safe on this hardware.
    // The broadcast kernel relies on kernel-initiated PCIe peer writes being
    // visible to the destination rank before the in-kernel barrier's RELEASE
    // flag arrives, i.e. cache-coherent peer writes. This is true on:
    //   - NVIDIA (any CUDA-capable device): NVLink and PCIe P2P are HW-coherent
    //   - AMD gfx90a (MI200), gfx94x (MI300), gfx95x: XGMI is HW-coherent
    //   - any AMD PCIe GPU when HSA_FORCE_FINE_GRAIN_PCIE=1 forces all device
    //     allocations to fine-grained (write-through) memory. The fine-grain
    //     mechanism is generic to AMD, but this peer-write path is validated
    //     only on gfx906 (MI50), so other AMD archs get a warning below.
    // Without FGP (and off XGMI) we stay on the one-shot path, which uses
    // cudaMemcpyAsync staging + an event handshake and is always correct.
    bool hw_peer_write_coherent = false;
#if defined(GGML_USE_HIP)
    {
        hipDeviceProp_t prop;
        CUDA_CHECK(hipGetDeviceProperties(&prop, ctx->dev_ids[0]));
        const char * arch = prop.gcnArchName;   // e.g. "gfx906:sramecc+:xnack-"
        const bool is_gfx906 = (strncmp(arch, "gfx906", 6) == 0);
        const bool is_gfx9_xgmi =
            (strncmp(arch, "gfx90a", 6) == 0) ||
            (strncmp(arch, "gfx94",  5) == 0) ||
            (strncmp(arch, "gfx95",  5) == 0);
        const char * e_fgp = getenv("HSA_FORCE_FINE_GRAIN_PCIE");
        const bool fgp_on = e_fgp && e_fgp[0] != '\0' && e_fgp[0] != '0';
        hw_peer_write_coherent = is_gfx9_xgmi || fgp_on;
        const bool peer_write_experimental = fgp_on && !is_gfx906 && !is_gfx9_xgmi;
        if (peer_write_experimental) {
            GGML_LOG_WARN("TP custom AR: peer-write enabled on %s via HSA_FORCE_FINE_GRAIN_PCIE "
                          "(validated only on gfx906, verify output correctness)\n", arch);
        }
    }
#else
    hw_peer_write_coherent = true;   // CUDA / MUSA: HW-coherent peer access
#endif
    ctx->broadcast_ok = hw_peer_write_coherent;
    ctx->initialized  = true;

    const char * path      = ctx->broadcast_ok ? "broadcast F32 + twoshot F32 (peer-write, size-adaptive, lossless)"
                              :
#if defined(GGML_USE_HIP)
                                "one-shot (PCIe without HSA_FORCE_FINE_GRAIN_PCIE=1, set it to enable fast peer-write paths)"
#else
                                "one-shot (HW not recognised as peer-write coherent)"
#endif
                              ;
    GGML_LOG_INFO("TP custom AllReduce: initialized for %d GPUs, path = %s\n", nranks, path);
}

void tp_custom_ar_destroy(CustomARContext * ctx) {
    if (getenv("GGML_TP_AR_KTIMING") != nullptr && ctx->initialized) {
        for (int r = 0; r < ctx->nranks; r++) {
            unsigned long long h[5] = {0, 0, 0, 0, 0};
            ggml_cuda_set_device(ctx->dev_ids[r]);
            if (AR_MEMCPY_FROM_SYMBOL(h, g_ar_ktime, sizeof(h)) != cudaSuccess || h[4] == 0) {
                continue;
            }
            const double tot = (double) (h[0] + h[1] + h[2] + h[3]);
            const double n   = (double) h[4];
            fprintf(stderr,
                "[ar-ktime] rank %d calls=%llu | write=%.1f%% bar_start=%.1f%% reduce=%.1f%% "
                "bar_end=%.1f%% | ticks/call: write=%.0f bar_start=%.0f reduce=%.0f bar_end=%.0f\n",
                r, (unsigned long long) h[4],
                100.0 * h[0] / tot, 100.0 * h[1] / tot, 100.0 * h[2] / tot, 100.0 * h[3] / tot,
                h[0] / n, h[1] / n, h[2] / n, h[3] / n);
        }
    }

    if (!ctx->initialized) return;

    // free the capture-safe AR pools (after this point no graph can outlive us)
    for (auto * p : ctx->csar_pools) {
        for (int rank = 0; rank < ctx->nranks; rank++) {
            ggml_cuda_set_device(ctx->dev_ids[rank]);
            if (p->staging[rank])    { cudaFree(p->staging[rank]); }
            if (p->d_signals[rank])  { cudaFree(p->d_signals[rank]); }
            if (p->d_rank_data[rank]) { cudaFree(p->d_rank_data[rank]); }
        }
        delete p;
    }
    ctx->csar_pools.clear();

    for (int rank = 0; rank < ctx->nranks; rank++) {
        ggml_cuda_set_device(ctx->dev_ids[rank]);
        if (ctx->events[rank]) {
            CUDA_CHECK(cudaEventDestroy(ctx->events[rank]));
            ctx->events[rank] = nullptr;
        }
        if (ctx->d_signals[rank]) {
#if defined(GGML_USE_HIP)
            CUDA_CHECK(hipFree(ctx->d_signals[rank]));
#else
            CUDA_CHECK(cudaFree(ctx->d_signals[rank]));
#endif
            ctx->d_signals[rank] = nullptr;
        }
        if (ctx->d_rank_data[rank]) {
            CUDA_CHECK(cudaFree(ctx->d_rank_data[rank]));
            ctx->d_rank_data[rank] = nullptr;
        }
        if (ctx->d_staging[rank]) {
#if defined(GGML_USE_HIP)
            CUDA_CHECK(hipFree(ctx->d_staging[rank]));
#else
            CUDA_CHECK(cudaFree(ctx->d_staging[rank]));
#endif
            ctx->d_staging[rank] = nullptr;
        }
    }
    ctx->staging_size = 0;

    ctx->initialized = false;
}

// ============================================================================
// Capture-safe AR (GGML_TP_AR_CSAR=1) — per-width state pools
// ============================================================================
// Port of the CV hcp_ar capture-safety idea (hc-persist.cu hcp_ar_*): all
// per-call sequencing state must be device-side so captured graphs replay
// correctly forever, and no state layout may depend on the *other* width's
// calls. CV uses a fixed-size inbox; we key a complete state pool (staging +
// signals + rank-data) by (path, n_elements) instead. Pools are created on
// first sight of a width (always during eager warmup, never inside a stream
// capture — refused there) and never freed until the context is destroyed, so
// pointers baked into captured graphs stay valid across replays of any width.
static bool csar_enabled() {
    static const bool en = []{
        const char * e = getenv("GGML_TP_AR_CSAR");
        return e && e[0] && e[0] != '0';
    }();
    return en;
}

static bool csar_any_stream_capturing(cudaStream_t * streams, int nranks) {
#if defined(GGML_USE_HIP)
    using capture_status_t = hipStreamCaptureStatus;
    const auto status_none = hipStreamCaptureStatusNone;
    const auto get_info    = hipStreamGetCaptureInfo;
#else
    using capture_status_t = cudaStreamCaptureStatus;
    const auto status_none = cudaStreamCaptureStatusNone;
    const auto get_info    = cudaStreamGetCaptureInfo;
#endif
    for (int r = 0; r < nranks; r++) {
        capture_status_t st = status_none;
        if (get_info(streams[r], &st, nullptr) == cudaSuccess && st != status_none) {
            return true;
        }
    }
    return false;
}

// Returns true when the call is fully served from a per-width pool (plan is
// filled and the caller must return). On false with plan->transient_refusal
// set, the refusal is capture-scoped: the caller must fall back for THIS call
// only and keep the custom path enabled.
static bool csar_prepare_pool(CustomARContext * ctx,
                              float ** input_ptrs, float ** output_ptrs,
                              int64_t n_elements, int nranks,
                              cudaStream_t * streams, CustomARPlan * plan,
                              bool bf16) {
    static const int s_xbar = []{
        const char * e = getenv("GGML_TP_AR_SKIP_XBAR");
        return (e && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }();
    static const int s_uncached = []{
        const char * e = getenv("GGML_TP_AR_UNCACHED");
        return (e && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }();

    ARWidthPool * pool = nullptr;
    for (auto * p : ctx->csar_pools) {
        if (p->n_elements == n_elements && p->bf16 == bf16) { pool = p; break; }
    }
    if (pool == nullptr) {
        if (csar_any_stream_capturing(streams, nranks)) {
            plan->transient_refusal = true;
            static bool warned = false;
            if (!warned) {
                warned = true;
                GGML_LOG_WARN("TP custom AR [CSAR]: width %lld first seen inside stream capture; "
                              "refusing custom AR for this call (no regression, capture fails as before)\n",
                              (long long) n_elements);
            }
            return false;
        }
        const size_t per_slot = bf16 ? (size_t) n_elements * sizeof(uint16_t)
                                     : (size_t) n_elements * sizeof(float);
        const size_t bytes    = per_slot * (size_t) nranks * (s_xbar ? 2 : 1);
        pool = new ARWidthPool();
        pool->n_elements = n_elements;
        pool->bf16 = bf16;
        bool ok = true;
        for (int rank = 0; rank < nranks && ok; rank++) {
            ggml_cuda_set_device(ctx->dev_ids[rank]);
            void * ptr = nullptr;
#if defined(GGML_USE_HIP)
            if (hipExtMallocWithFlags(&ptr, bytes,
                    s_uncached ? hipDeviceMallocUncached : hipDeviceMallocFinegrained) != hipSuccess) {
                ok = false; break;
            }
            void * sgn = nullptr;
            if (hipExtMallocWithFlags(&sgn, sizeof(Signal), hipDeviceMallocUncached) != hipSuccess) {
                ok = false; break;
            }
            pool->d_signals[rank] = (Signal *) sgn;
#else
            if (cudaMalloc(&ptr, bytes) != cudaSuccess) { ok = false; break; }
            if (cudaMalloc((void **) &pool->d_signals[rank], sizeof(Signal)) != cudaSuccess) { ok = false; break; }
#endif
            pool->staging[rank] = (float *) ptr;
            CUDA_CHECK(cudaMemset(pool->d_signals[rank], 0, sizeof(Signal)));
            CUDA_CHECK(cudaMalloc((void **) &pool->d_rank_data[rank], sizeof(RankData)));
            pool->rank_signals.signals[rank] = pool->d_signals[rank];
        }
        if (!ok) {
            for (int rank = 0; rank < nranks; rank++) {
                ggml_cuda_set_device(ctx->dev_ids[rank]);
                if (pool->staging[rank])  { cudaFree(pool->staging[rank]); }
                if (pool->d_signals[rank]) { cudaFree(pool->d_signals[rank]); }
                if (pool->d_rank_data[rank]) { cudaFree(pool->d_rank_data[rank]); }
            }
            delete pool;
            return false;  // allocation failure: caller's sticky fallback is correct here
        }
        // upload this pool's staging bases once (illegal inside capture — guarded above)
        RankData h_data;
        for (int i = 0; i < kMaxRanks; i++) { h_data.ptrs[i] = nullptr; }
        for (int i = 0; i < nranks; i++) {
            h_data.ptrs[i] = pool->staging[i];
            pool->cached_ptrs[i] = pool->staging[i];
        }
        for (int rank = 0; rank < nranks; rank++) {
            ggml_cuda_set_device(ctx->dev_ids[rank]);
            CUDA_CHECK(cudaMemcpy(pool->d_rank_data[rank], &h_data, sizeof(RankData), cudaMemcpyHostToDevice));
        }
        ctx->csar_pools.push_back(pool);
        if (getenv("GGML_TP_AR_DEBUG") != nullptr) {
            fprintf(stderr, "[csar] pool created: n_elements=%lld bf16=%d bytes=%zu\n",
                    (long long) n_elements, (int) bf16, (size_t) 0);
        }
    }

    // device-symbol flags the kernels read (same uploads the legacy path does)
    if (s_xbar) {
        static bool csar_uploaded = false;
        if (!csar_uploaded) {
            csar_uploaded = true;
            static const int one = 1;
            for (int r = 0; r < nranks; r++) {
                ggml_cuda_set_device(ctx->dev_ids[r]);
                CUDA_CHECK(AR_MEMCPY_TO_SYMBOL(g_ar_xbar_on, &one, sizeof(int)));
            }
        }
    }

    static const int s_blocks_override_csar = []{
        const char * e = getenv("GGML_TP_AR_BLOCKS");
        if (!e || !e[0]) return 0;
        int v = atoi(e);
        return (v > 0 && v <= kMaxBlocks) ? v : 0;
    }();
    const int blocks_cap = s_blocks_override_csar ? s_blocks_override_csar : kDefaultBlocks;
    const int blocks = std::min(blocks_cap,
                                std::max(1, (int) ((n_elements / 4 + kThreads - 1) / kThreads)));

    plan->ctx        = ctx;
    plan->pool       = pool;
    plan->n_elements = n_elements;
    plan->nranks     = nranks;
    plan->blocks     = blocks;
    plan->twoshot    = false;
    plan->broadcast  = true;
    plan->bf16       = bf16;
    plan->chunked    = false;
    plan->piece_elems = 0;
    plan->ag_offset  = 0;
    for (int rank = 0; rank < nranks; rank++) {
        plan->inputs [rank] = input_ptrs [rank];
        plan->outputs[rank] = output_ptrs[rank];
        plan->streams[rank] = streams    [rank];
    }
    return true;
}

bool tp_custom_ar_prepare(CustomARContext * ctx,
                          float ** input_ptrs,
                          float ** output_ptrs,
                          int64_t  n_elements,
                          int      nranks,
                          cudaStream_t * streams,
                          CustomARPlan * plan) {
    GGML_ASSERT(ctx->initialized);
    GGML_ASSERT(nranks == ctx->nranks);
    plan->transient_refusal = false;
    plan->pool = nullptr;
    if (getenv("GGML_TP_AR_DEBUG") != nullptr) {
        fprintf(stderr, "[ar-debug] prepare enter: staging_size=%zu slots=%d n_elements=%lld\n",
                ctx->staging_size, ctx->staging_slots, (long long) n_elements);
    }
    GGML_ASSERT(nranks >= 2 && nranks <= kMaxRanks);

    // Path selection. broadcast_ok was set at init based on the HW peer-write
    // coherence model (see tp_custom_ar_init).
    // Two-shot (reduce-scatter + allgather) kernel selection for large msgs.
    // F32 on the wire — fully lossless, matches the broadcast kernel's
    // precision and avoids any BF16 conversion. Per-rank outbound is
    // 2*(N-1)/N * S, so 2× the BW of an equivalent BF16-on-wire variant
    // but with parallel all-to-all peer writes (vs NCCL's serial ring).
    //
    // GGML_TP_AR_TWOSHOT can force on (=1) or off (=0); default auto picks
    // twoshot for ne ≥ ~256K elements where broadcast's (N-1)*S/N per-rank
    // outbound starts to lose to twoshot's 2*(N-1)/N*S aggregate parallel BW.
    static const int s_twoshot_env = []{
        const char * e = getenv("GGML_TP_AR_TWOSHOT");
        if (!e || !e[0]) return -1;                      // default = auto
        return (e[0] != '0') ? 1 : 0;
    }();
    const bool s_broadcast = ctx->broadcast_ok;
    // Twoshot needs slice-aligned + float4-aligned data: n_elements % (4*nranks) == 0.
    const bool twoshot_eligible =
        s_broadcast && (n_elements % (int64_t)(4 * nranks) == 0);
    const int64_t twoshot_legacy_min_ne = 262144;        // ~1 MB F32 crossover
    // Decode A/B on gfx906 selects two-shot for ranks where its reduced peer
    // traffic pays for the second phase. Four-rank communicators cross over
    // only when they are pipeline stages; standalone TP4 remains broadcast.
    const bool auto_small_twoshot =
        nranks == 3 || nranks == 5 || nranks == 8 || nranks == 10 ||
        (nranks == 4 && ctx->prefer_small_twoshot);
    // A standalone four-rank group was pinned to broadcast at every size it can reach, because the legacy crossover sits at the size where the group stops using the custom path at all.
    // Broadcast wins while a message is one token wide, and loses once it is several, which is what a speculative verify batch sends: at four ranks its outbound grows with the group while twoshot's does not.
    // Two ranks stay on broadcast whatever the size, since twoshot moves the same bytes there and adds a second phase.
    // GGML_TP_AR_TWOSHOT_MIN_NE overrides the crossover in elements for tuning.
    const int64_t twoshot_small_min_ne = nranks >= 4 ? 8192 : twoshot_legacy_min_ne;
    static const int64_t s_twoshot_min_ne_env = []{
        const char * e = getenv("GGML_TP_AR_TWOSHOT_MIN_NE");
        return e != nullptr ? (int64_t) strtoll(e, nullptr, 10) : (int64_t) -1;
    }();
    int64_t twoshot_min_ne = auto_small_twoshot ? 1 : twoshot_small_min_ne;
    if (s_twoshot_min_ne_env >= 0) {
        twoshot_min_ne = s_twoshot_min_ne_env;
    }
    const bool s_twoshot =
        twoshot_eligible &&
        (s_twoshot_env == 1 || (s_twoshot_env == -1 && n_elements >= twoshot_min_ne));
    if (s_twoshot && s_twoshot_env == -1 && n_elements < twoshot_legacy_min_ne) {
        if (!ctx->logged_small_twoshot) {
            ctx->logged_small_twoshot = true;
            fprintf(stderr,
                    "TP custom AllReduce: auto-selected two-shot for %d-rank decode-sized tensors\n",
                    nranks);
        }
    }

    const size_t bytes_f32 = (size_t) n_elements * sizeof(float);
    // Broadcast path: F32 staging, N slots per rank (one inbox per peer + unused self slot).
    // Two-shot path: F32 staging, 2 buffers (scatter_buf + allgather_buf) each
    //   sized n_elements floats. Total = 2 * n_elements * sizeof(float) bytes.
    // GGML_TP_AR_SKIP_XBAR doubles the broadcast staging (2 parity sets) so the
    // end-barrier can be elided; see k_broadcast_reduce.
    static const int s_xbar = []{
        const char * e = getenv("GGML_TP_AR_SKIP_XBAR");
        return (e && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }();
    // GGML_TP_AR_SLEEP: s_sleep(1) backoff in the barrier spins (default off).
    static const int s_sleep = []{
        const char * e = getenv("GGML_TP_AR_SLEEP");
        return (e && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }();
    // GGML_TP_AR_UNCACHED: allocate the peer-visible staging as uncached device
    // memory (hipDeviceMallocUncached) instead of fine-grained. Peer writes then
    // bypass the receiver's L2 the way CV's inbox does (default off).
    static const int s_uncached = []{
        const char * e = getenv("GGML_TP_AR_UNCACHED");
        return (e && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }();
    // GGML_TP_AR_BF16_THRESHOLD: minimum element count for a bf16 peer wire on
    // the broadcast path. 0 or unset keeps the stock F32 wire everywhere, so
    // default behaviour is byte-identical. Requires 4-element alignment for the
    // packed 8-byte store; twoshot and one-shot stay F32.
    static const int64_t s_bf16_th = []{
        const char * e = getenv("GGML_TP_AR_BF16_THRESHOLD");
        return (e && e[0]) ? (int64_t) strtoll(e, nullptr, 10) : (int64_t) 0;
    }();
    const bool s_bf16 =
        s_broadcast && !s_twoshot && s_bf16_th > 0 &&
        n_elements >= s_bf16_th && (n_elements % 4 == 0);
    // GGML_TP_AR_CHUNKED=1: route LARGE tensors (>= GGML_TP_AR_CHUNK_MIN, default
    // 32768 elements) through the piece-looped peer-write kernel with fixed
    // staging instead of the backend fallback. Decode-sized tensors keep the
    // broadcast path. Exclusive with bf16 and twoshot.
    static const bool s_chunked_env = []{
        const char * e = getenv("GGML_TP_AR_CHUNKED");
        return e && e[0] != '\0' && e[0] != '0';
    }();
    static const int64_t s_chunk_elems = []{
        const char * e = getenv("GGML_TP_AR_CHUNK_ELEMS");
        const int64_t v = e ? strtoll(e, nullptr, 10) : 0;
        return (v >= 4096) ? v : (int64_t) 131072;
    }();
    static const int64_t s_chunk_min = []{
        const char * e = getenv("GGML_TP_AR_CHUNK_MIN");
        const int64_t v = e ? strtoll(e, nullptr, 10) : 0;
        return (v > 0) ? v : (int64_t) 32768;
    }();
    const bool s_chunked =
        s_broadcast && s_chunked_env && !s_bf16;
    if (s_chunked) {
        static bool logged_chunked = false;
        if (!logged_chunked) {
            logged_chunked = true;
            fprintf(stderr, "TP custom AR: chunked peer-write ACTIVE (n_elements=%lld piece=%lld)\n",
                    (long long) n_elements, (long long) s_chunk_elems);
        }
    }
    if (s_bf16) {
        static bool logged_bf16 = false;
        if (!logged_bf16) {
            logged_bf16 = true;
            fprintf(stderr, "TP custom AR: bf16 wire ACTIVE (n_elements=%lld >= threshold=%lld)\n",
                    (long long) n_elements, (long long) s_bf16_th);
        }
    }
    const size_t per_slot   = s_chunked ? (size_t) s_chunk_elems * sizeof(float)
                          : s_bf16     ? (size_t) n_elements * sizeof(uint16_t)
                                       : bytes_f32;
    const int    need_slots = s_chunked ? ctx->nranks
                              : s_twoshot ? 2
                              : (s_broadcast ? ctx->nranks * (s_xbar ? 2 : 1) : 1);
    const size_t need_bytes = per_slot * need_slots;

    // Capture-safe AR (GGML_TP_AR_CSAR=1): the broadcast path is served from
    // per-width pools so draft/verify width transitions never share staging,
    // signal or rank-data state (see csar_prepare_pool). Non-broadcast paths
    // (twoshot/one-shot/chunked) keep the old MTP_SAFE lock semantics: once a
    // width transition has been seen they refuse, exactly as the sticky lock
    // did before CSAR existed.
    if (csar_enabled()) {
        if (s_broadcast && !s_twoshot && !s_chunked) {
            if (csar_prepare_pool(ctx, input_ptrs, output_ptrs, n_elements, nranks,
                                  streams, plan, s_bf16)) {
                return true;
            }
            return false;
        }
        if (ctx->width_transition_seen) {
            return false;
        }
    }

    if (getenv("GGML_TP_AR_DEBUG") != nullptr && (ctx->staging_slots != need_slots || ctx->staging_size < need_bytes)) {
        fprintf(stderr, "[ar-debug] staging realloc: need_bytes=%zu staging_size=%zu need_slots=%d staging_slots=%d n_elements=%lld bf16=%d twoshot=%d\n",
                need_bytes, ctx->staging_size, need_slots, ctx->staging_slots,
                (long long) n_elements, (int) s_bf16, (int) s_twoshot);
    }
    if (ctx->staging_slots != need_slots || ctx->staging_size < need_bytes) {
        for (int rank = 0; rank < ctx->nranks; rank++) {
            ggml_cuda_set_device(ctx->dev_ids[rank]);
            if (ctx->d_staging[rank]) {
#if defined(GGML_USE_HIP)
                CUDA_CHECK(hipFree(ctx->d_staging[rank]));
#else
                CUDA_CHECK(cudaFree(ctx->d_staging[rank]));
#endif
                ctx->d_staging[rank] = nullptr;
            }
            void * ptr = nullptr;
#if defined(GGML_USE_HIP)
            const cudaError_t err = hipExtMallocWithFlags(&ptr, need_bytes,
                    s_uncached ? hipDeviceMallocUncached : hipDeviceMallocFinegrained);
#else
            const cudaError_t err = cudaMalloc(&ptr, need_bytes);
#endif
            if (err == cudaErrorMemoryAllocation) {
                (void) cudaGetLastError();
                fprintf(stderr, "[ar-debug] staging alloc FAILED (hip err=%d) need_bytes=%zu - if inside stream capture, hipExtMallocWithFlags is illegal here\n", (int) err, need_bytes);
                GGML_LOG_WARN("TP custom AR: could not allocate %.2f MiB staging on device %d; "
                              "falling back to the regular AllReduce path\n",
                              need_bytes / (1024.0 * 1024.0), ctx->dev_ids[rank]);
                for (int cleanup_rank = 0; cleanup_rank < ctx->nranks; cleanup_rank++) {
                    ggml_cuda_set_device(ctx->dev_ids[cleanup_rank]);
                    if (ctx->d_staging[cleanup_rank] != nullptr) {
                        CUDA_CHECK(cudaFree(ctx->d_staging[cleanup_rank]));
                        ctx->d_staging[cleanup_rank] = nullptr;
                    }
                    ctx->cached_ptrs[cleanup_rank] = nullptr;
                }
                ctx->staging_size  = 0;
                ctx->staging_slots = 0;
                *plan = {};
                return false;
            }
            CUDA_CHECK(err);
            if (getenv("GGML_TP_AR_DEBUG") != nullptr) {
                fprintf(stderr, "[ar-debug] staging ALLOCATED need_bytes=%zu on rank %d\n", need_bytes, rank);
            }
            ctx->d_staging[rank] = (float *) ptr;
        }
        ctx->staging_size  = need_bytes;
        ctx->staging_slots = need_slots;
        // invalidate cached RankData pointers → forces re-upload below
        for (int i = 0; i < nranks; i++) ctx->cached_ptrs[i] = nullptr;
    }

    if (!s_broadcast) {
        // One-shot path: SDMA staging + event handshake. Used as a fallback
        // when peer-write coherence is unavailable.
        for (int rank = 0; rank < nranks; rank++) {
            ggml_cuda_set_device(ctx->dev_ids[rank]);
            CUDA_CHECK(cudaMemcpyAsync(ctx->d_staging[rank], input_ptrs[rank], bytes_f32,
                                       cudaMemcpyDeviceToDevice, streams[rank]));
        }
        for (int rank = 0; rank < nranks; rank++) {
            ggml_cuda_set_device(ctx->dev_ids[rank]);
            CUDA_CHECK(cudaEventRecord(ctx->events[rank], streams[rank]));
        }
        for (int rank = 0; rank < nranks; rank++) {
            for (int peer = 0; peer < nranks; peer++) {
                if (peer == rank) continue;
                CUDA_CHECK(cudaStreamWaitEvent(streams[rank], ctx->events[peer], 0));
            }
        }
    }

    // Update device-side RankData only when staging pointers change (rare —
    // staging buffers persist in ctx after first alloc, unless switching
    // between one-shot and broadcast staging layouts).
    bool ptrs_changed = false;
    for (int i = 0; i < nranks; i++) {
        if ((void *) ctx->d_staging[i] != ctx->cached_ptrs[i]) {
            ptrs_changed = true;
            break;
        }
    }
    if (ptrs_changed) {
        RankData h_data;
        for (int i = 0; i < nranks; i++) {
            h_data.ptrs[i] = ctx->d_staging[i];
            ctx->cached_ptrs[i] = ctx->d_staging[i];
        }
        for (int rank = 0; rank < nranks; rank++) {
            ggml_cuda_set_device(ctx->dev_ids[rank]);
            CUDA_CHECK(cudaMemcpy(ctx->d_rank_data[rank], &h_data,
                                  sizeof(RankData), cudaMemcpyHostToDevice));
        }
    }

    // Compute grid dimensions. Block count can be overridden via GGML_TP_AR_BLOCKS
    // for quick perf sweeps; default kDefaultBlocks is tuned per platform in .cuh.
    static const int s_blocks_override = []{
        const char * e = getenv("GGML_TP_AR_BLOCKS");
        if (!e || !e[0]) return 0;
        int v = atoi(e);
        return (v > 0 && v <= kMaxBlocks) ? v : 0;
    }();
    const int blocks_cap = s_blocks_override ? s_blocks_override : kDefaultBlocks;
    const int64_t packed_size = n_elements / 4;
    int blocks = std::min(blocks_cap, std::max(1, (int)((packed_size + kThreads - 1) / kThreads)));

    // Everything above is shared setup. Record what a per-rank launch needs so
    // the kernels can be issued one rank at a time, from any thread.
    static const int s_ktiming = []{
        const char * e = getenv("GGML_TP_AR_KTIMING");
        return (e && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }();
    if (s_ktiming || s_xbar || s_sleep) {
        static bool uploaded = false;
        if (!uploaded) {
            uploaded = true;
            for (int r = 0; r < nranks; r++) {
                ggml_cuda_set_device(ctx->dev_ids[r]);
                if (s_ktiming) {
                    CUDA_CHECK(AR_MEMCPY_TO_SYMBOL(g_ar_ktime_on, &s_ktiming, sizeof(int)));
                }
                if (s_xbar) {
                    CUDA_CHECK(AR_MEMCPY_TO_SYMBOL(g_ar_xbar_on, &s_xbar, sizeof(int)));
                }
                if (s_sleep) {
                    CUDA_CHECK(AR_MEMCPY_TO_SYMBOL(g_ar_sleep_on, &s_sleep, sizeof(int)));
                }
            }
        }
    }

    plan->ctx        = ctx;
    plan->n_elements = n_elements;
    plan->nranks     = nranks;
    plan->blocks     = blocks;
    plan->twoshot    = s_twoshot && !s_chunked;
    plan->broadcast  = s_broadcast;
    plan->bf16       = s_bf16;
    plan->chunked    = s_chunked;
    plan->piece_elems = s_chunked ? s_chunk_elems : 0;
    // The allgather region starts at half the staging allocation, above the scatter region of every call the allocation can serve.
    plan->ag_offset  = (int64_t) (ctx->staging_size / sizeof(float) / 2) & ~(int64_t) 3;
    GGML_ASSERT(!plan->twoshot || plan->ag_offset >= n_elements);
    for (int rank = 0; rank < nranks; rank++) {
        plan->inputs [rank] = input_ptrs [rank];
        plan->outputs[rank] = output_ptrs[rank];
        plan->streams[rank] = streams    [rank];
    }
    return true;
}

void tp_custom_ar_launch_rank(const CustomARPlan * plan, int rank) {
    CustomARContext * ctx = plan->ctx;
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(rank >= 0 && rank < plan->nranks);

    const int     nranks     = plan->nranks;
    const int     blocks     = plan->blocks;
    const int64_t n_elements = plan->n_elements;

    // Capture-safe AR: a pooled plan carries its own signals and rank-data; the
    // pointers are pass-by-value at launch, so a captured graph bakes the pool's
    // state and replays it unchanged (pools are never freed before destroy).
    ARWidthPool * const pool = plan->pool;
    RankData * const    rank_data = pool ? pool->d_rank_data[rank] : ctx->d_rank_data[rank];
    const RankSignals & rank_signals = pool ? pool->rank_signals : ctx->rank_signals;
    Signal * const      self_signals = pool ? pool->d_signals[rank] : ctx->d_signals[rank];

    ggml_cuda_set_device(ctx->dev_ids[rank]);

    if (plan->zero_ptr[rank] != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(plan->zero_ptr[rank], 0, plan->zero_bytes,
                                   plan->streams[rank]));
    }

    // Per-N kernel launches. The dispatcher below recursively matches `nranks`
    // against compile-time N from kMaxRanks down to 2; the compiler folds the
    // chain into a jump table at -O2. Adding ranks just means bumping
    // kMaxRanks — the templates instantiate automatically.
    auto dispatch = [&](auto self, auto N_CONST) -> void {
        constexpr int N = decltype(N_CONST)::value;
        if constexpr (N < 2) {
            GGML_ABORT("TP custom AR: unsupported nranks=%d (must be 2..%d)\n", nranks, kMaxRanks);
        } else        if (nranks == N) {
            if (!plan->broadcast) {
                k_cross_device_reduce_1stage<N><<<blocks, kThreads, 0, plan->streams[rank]>>>(
                    rank_data, rank_signals, self_signals,
                    plan->outputs[rank], rank, n_elements);
            } else if (plan->chunked) {
                k_broadcast_chunked<N><<<blocks, kThreads, 0, plan->streams[rank]>>>(
                    rank_data, rank_signals, self_signals,
                    plan->inputs[rank], plan->outputs[rank], rank, n_elements, plan->piece_elems);
            } else if (plan->twoshot) {
                k_twoshot_f32<N><<<blocks, kThreads, 0, plan->streams[rank]>>>(
                    rank_data, rank_signals, self_signals,
                    plan->inputs[rank], plan->outputs[rank], rank, n_elements, plan->ag_offset);
            } else if (plan->bf16) {
                k_broadcast_reduce_bf16<N><<<blocks, kThreads, 0, plan->streams[rank]>>>(
                    rank_data, rank_signals, self_signals,
                    plan->inputs[rank], plan->outputs[rank], rank, n_elements);
            } else {
                k_broadcast_reduce<N><<<blocks, kThreads, 0, plan->streams[rank]>>>(
                    rank_data, rank_signals, self_signals,
                    plan->inputs[rank], plan->outputs[rank], rank, n_elements);
            }
            CUDA_CHECK(cudaGetLastError());
        } else {
            self(self, std::integral_constant<int, N - 1>{});
        }
    };
    dispatch(dispatch, std::integral_constant<int, kMaxRanks>{});
}

bool tp_custom_ar_allreduce(CustomARContext * ctx,
                            float ** input_ptrs,
                            float ** output_ptrs,
                            int64_t  n_elements,
                            int      nranks,
                            cudaStream_t * streams) {
    CustomARPlan plan;
    if (!tp_custom_ar_prepare(ctx, input_ptrs, output_ptrs, n_elements, nranks, streams, &plan)) {
        return false;
    }
    for (int rank = 0; rank < nranks; rank++) {
        tp_custom_ar_launch_rank(&plan, rank);
    }
    return true;
}

bool tp_custom_ar_mtp_safe_transition(CustomARContext * ctx, int64_t n_elements) {
    // Capture-safe AR: per-width pools make the broadcast path width-transition
    // safe, so the sticky lock is not needed there. We keep tracking the
    // transition so the non-pool paths (twoshot/one-shot/chunked) refuse exactly
    // as the lock did; the lock itself is never armed while CSAR is on.
    if (csar_enabled()) {
        const int64_t prev = ctx->last_ne;
        ctx->last_ne = n_elements;
        if (prev >= 0 && prev != n_elements) {
            ctx->width_transition_seen = true;
        }
        return false;
    }
    static const int s_mtp_safe = []{
        const char * e = getenv("GGML_TP_AR_MTP_SAFE");
        if (!e || !e[0] || e[0] == '0') return 0;
        return (e[0] == '2') ? 2 : 1;
    }();
    if (!s_mtp_safe) return false;
    if (ctx->mtp_safe_locked) return true;   // once a width change is seen, custom is off for good
    const int64_t prev = ctx->last_ne;
    ctx->last_ne = n_elements;
    if (prev < 0 || prev == n_elements) return false;

    // Width transition detected: speculative decoding alternates batch
    // widths (draft batch vs verify batch). The custom kernels' staging
    // layout, parity double-buffering and shared per-block signal counters
    // are only proven safe for a single steady width per context;
    // alternating widths across asynchronously-replayed captured graphs
    // corrupt the reduction (violation repro 2026-09-25). Permanently
    // route this context through the regular AllReduce path. We
    // deliberately do NOT free the staging: graphs captured earlier still
    // reference it, and as the sole remaining custom user (constant width)
    // it stays safe.
    ctx->mtp_safe_locked = true;
    if (s_mtp_safe >= 2) {
        fprintf(stderr, "TP custom AR [MTP_SAFE]: width transition %lld -> %lld; custom AR locked off for this context\n",
                (long long) prev, (long long) n_elements);
    }
    return true;
}

} // namespace ggml_cuda_tp
