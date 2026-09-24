# W4A4 dot8 Kernel Notes — Phase 2c

Repo: `mx-Mistral-Large fine-tune` (Gitea `root/mx-llama.cpp`), branch `perf/w4a4-dot8-gfx906`.
Build tree (mi50, 192.168.50.41): `/root/build-w4a4` (`-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DCMAKE_BUILD_TYPE=Release`, TheRock venv toolchain).

## Milestone 1 — primitives + types (DONE, commit `a4643ebc8`)

Unused by any dispatch; default build is byte-identical behavior. 72 insertions across 3 files:

1. **`ggml/src/ggml-cuda/common.cuh:756-774`** — `ggml_cuda_dp8a(int a, int b, int c)`.
   - gfx906 path: `__builtin_amdgcn_sdot8(a, b, c, false)` → emits `v_dot8_i32_i4` (verified in saved-temps asm).
   - Operand layout: uint32 = 8 signed nibbles; byte k = nibble pair {2k, 2k+1} little-endian.
   - Non-gfx906 fallback: scalar nibble loop (later AMD ISAs dropped the 8-way int4 dot). Guarded `GGML_USE_HIP && __gfx906__`, mirroring `ggml_cuda_dp4a` (line 716).
2. **`ggml/src/ggml-cuda/q8_repack/repack-common.cuh:188-201`** — `block_q8_1_u4 { uint8_t qs[16]; ggml_half d4; ggml_half m; }`, `static_assert(sizeof == 20)`.
   - Per-32 asymmetric uint4 activation block, scheme B from `docs/w4a4-activation-study.md`: `d4=(max-min)/15`, `value = qs_nibble*d4 + m`.
   - Nibble order matches dp8a operand order → one 16-byte qs feeds 4 `dp8a` instrs directly.
3. **`ggml/src/ggml-cuda/q8_repack/repack-common.cu:449-483`** — host reference quantizer `quantize_row_q8_1_u4(const float * x, void * vy, int64_t k)` (k % 32 == 0): min/max pass + nibble-pack pass. Symbol verified in `libggml-hip.so`.

Build: `llama-bench` target clean (default config). No env gate added yet — nothing reads the new symbols, so the `GGML_CUDA_W4A4_DOT8` gate lands with milestone 2's dispatch wiring.

## Milestone 2a — device uint4 activation quantize kernel (DONE, unwired)

fp32 -> `block_q8_1_u4` on device. 114 insertions across 2 files, purely additive; no dispatch touched.

1. **`ggml/src/ggml-cuda/quantize.cu:560-650`** — `quantize_mmq_q8_1_u4` `__global__`.
   - Per-32 block: `mn=min`, `mx=max`, `d4=(mx-mn)/15`, `id = d4!=0 ? 1/d4 : 0`, `q = clamp(roundf((x-mn)*id), 0, 15)`.
   - Stores the **signed** nibble `(q-8)&0xF`, 2/byte low-first (byte k = {2k,2k+1} LE — the dp8a operand order). 4 vals/thread = 2 bytes = one `uint16_t` store at `qs + iqs/2`; 8 threads fill the 16-byte `qs` exactly.
   - `d4`/`m` as fp16 written as one 32-bit store at block offset 16 (matches `block_q8_1_u4` field layout, repack-common.cuh:192-196) by the `iqs==0` thread.
   - Grid/block mirrors `quantize_mmq_q8_1`: `grid=(ne1, ceil(ne0/512), ne2*ne3)`, block 128, 4 floats/thread; min/max reduce over the 8 threads per 32-block via `__shfl_xor_sync` width=32 (repo's `vendors/hip.h` maps it to raw `__shfl_xor`, so the 32-bit mask is correct on wave64).
   - Out-of-range threads (`i0` in `[ne00, ne0)`) write the zero-input pattern (nibble 8, d4=m=0) into row padding, same as the q8_1 MMQ quantizer writes zeros there; padding blocks are never read.
2. **`ggml/src/ggml-cuda/quantize.cu:705-723`** — host launcher `quantize_mmq_q8_1_u4_cuda(...)` beside `quantize_mmq_q8_1_cuda`, same stream placement. **Not called from any dispatch** — 2b switches the repacked mat-vec path onto it under the env gate.
3. **`ggml/src/ggml-cuda/quantize.cuh:31-37`** — declaration.

### Delta vs host reference `quantize_row_q8_1_u4` (repack-common.cu:449)

- **Intentional**: device stores signed nibble `(q-8)&0xF`; host stores the raw unsigned code `q`. Same q, different bias convention — the dot loop's `sdot8 + 8*SUMW` bookkeeping needs the signed form.
- **Identical**: `d4`, `id` (incl. the d4==0 -> id=0 guard), `roundf` (half-away-from-zero), clamp [0,15], fp16 RNE of d4/m.
- **Layout**: device output is interleaved like `quantize_mmq_q8_1` (`ib = z*ne1*nbpc + k_block*ne1 + row`), not per-channel-contiguous like the host row writer. Verified equivalent under that mapping.
- **Numeric verification** (standalone hipcc harness, gfx906, pre-GPU-pause): single-channel cases (ne00=4096/32, ne1=1) bit-identical incl. d4==0 constant rows and padding. Multi-channel cases initially "failed" only because the harness compared interleaved-vs-contiguous indices 1:1; harness fixed to the correct mapping but **re-run deferred** under the GPU pause (owner directive). Mapping proven by CPU-side stride check: device `stride_z == ne1*nbpc` for all ne0 % 512 == 0 cases; the two non-512-multiple harness cases (ne0=32, 4128) have single z so stride is moot.

Build: `flock /root/gpu.lock -c 'cmake --build /root/build-w4a4 -j24 --target llama-bench'` — clean, exit 0. No benching (GPU pause).

## Milestone 2b — u4-activation decode kernel variant (DONE, unwired)

Commit `f5325c1e6` on `perf/w4a4-dot8-gfx906` (pushed to Gitea). 97 insertions across 2 files; default build byte-identical behavior (variant unreachable until milestone 3 wires the dispatch).

### CPU numeric seal (ran before commit, /tmp/dp8a_dot_test.cc, g++ -O2)

```
blocks=1000  (a) naive==sdot8: fails=0 PASS
epilogue A  d*(d4*idot + m*SUMW):          fails=983 FAIL
epilogue B  d*(d4*(idot+8*SUMW) + m*SUMW): fails=0 PASS
```

**IMPORTANT — the Slot B brief's epilogue (A) is WRONG.** The activation value is
`q*d4 + m` with `q = a_tc + 8` (a_tc = the two's-complement nibble 2a stores), so
`sum(w*a) = d*[ d4*(sdot8 + 8*SUMW) + m*SUMW ]` — the `+8*d4*SUMW` term is real
and A drops it. The context pack's excess-8 form (B) is the exact one and is what
the kernel implements. (a) confirms the XOR-8 weight transform × signed activation
nibbles == naive dot of (cw-8)*(q-8) streams, 1000/1000.

### Files + lines (as committed)

1. **`ggml/src/ggml-cuda/q8_repack/repack-kernels.cuh:63-76`** — `mmvq_dp8a_u4(uint4 w, uint4 a)`: 4 chained `ggml_cuda_dp8a` over .x/.y/.z/.w; non-gfx906 fallback `GGML_UNUSED_VARS + return 0` (mirrors `mmvq_dp4a_u4`).
2. **`repack-kernels.cuh:78-88`** — `nib_sum_u4(uint4)`: sum of the 32 raw excess-8 nibble codes; `SUMW = nib_sum_u4(pk) - 256`.
3. **`repack-common.cuh:436`** — `rp_traits<GGML_TYPE_Q4_0>::u4_act = true` (only consumed under the U4ACT template variant).
4. **`repack-common.cuh:459-465`** — `rp_traits<GGML_TYPE_Q4_0>::load_w_raw`: raw packed uint4 (canonical Q4_0 bytes, excess-8 interleaved pairs) + f16 scale slot, addressing mirrored from `load_w` (`wrow*rs_u4 + sb`, dplane at `dplane_off + (wrow*n_sub + sb)*2`).
5. **`repack-kernels.cuh:1005-1006`** — `mul_mat_vec_repacked_nc_impl` gains `bool U4ACT = false` (after WT; all existing instantiations unchanged).
6. **`repack-kernels.cuh:1015-1016`** — `static_assert(!U4ACT || WT==GGML_TYPE_Q4_0)` and `static_assert(!U4ACT || !HAS_FUSION)`.
7. **`repack-kernels.cuh:1066-1110`** — U4ACT branch in the sb loop (early-`continue`, replaces the whole stock body; no ds.y affine term). Per row: `pk = load_w_raw(...)`; `w_tc = pk ^ 0x88888888` per word (excess-8 → two's-complement); `d = scale(rp_slot_keep(db, rvalid))` — dead rows zero the scale, zeroing the whole term (stock masking semantics); `sw = nib_sum_u4(pk) - 256`. Per column: `xb4 = (block_q8_1_u4*)xq + c*xs + sb`, qs read as four u32s (20 B stride keeps 4-byte alignment — no misaligned uint4 load); `idot = mmvq_dp8a_u4(w_tc, a)`; `acc += d * (d4 * (idot + 8*sw) + m * sw)`.
8. **`repack-kernels.cuh:1215-1221`** — wrapper `mul_mat_vec_repacked_nc` threads `bool U4ACT = false` through to the impl; all stock call sites (mul-mat.cu 233-280) compile unchanged.

### Build + instantiation verification

- `flock /root/gpu.lock -c 'cmake --build /root/build-w4a4 -j24 --target llama-bench'` — clean, exit 0.
- Force-instantiation check: temporary `__global__` calling `mul_mat_vec_repacked_nc_impl<2,1,4,2,false,64,GGML_TYPE_Q4_0,true>` compiled to .o with the exact build flags (clang++ 17, --offload-arch=gfx906); `--save-temps` asm shows **32 `v_dot8_i32_i4`** instructions = RPL(2)×NCOLS(4)×4 dot8s — real codegen, not a stub. Dummy removed after the check (box clean).

### Runtime state

Variant is UNREACHABLE at runtime — no dispatch selects U4ACT=true yet. Milestone 3 wires it: env gate `GGML_CUDA_W4A4_DOT8=1` at the `mul_mat_vec_repacked_nc` dispatch (mul-mat.cu ~233-280) + switching the src1 quantizer to `quantize_mmq_q8_1_u4_cuda` under the same gate. Gauntlet commands (milestone 3):

```
# bench arms (Slot A owns the bench window; coordinate via /root/gpu.lock)
GGML_CUDA_W4A4_DOT8=1 ./bin/llama-bench -m <q4_0-model> -p 512 -n 128 -r 5 -fa 0   # u4-act arm
./bin/llama-bench -m <q4_0-model> -p 512 -n 128 -r 5 -fa 0                          # q8_1-act baseline
# PPL A/B (the real go/no-go; est. deltaPPL 0.10-0.53 for scheme B)
GGML_CUDA_W4A4_DOT8=1 ./bin/llama-perplexity -m <q4_0-model> -f wiki.test.bin -p 512
./bin/llama-perplexity -m <q4_0-model> -f wiki.test.bin -p 512
```

## Milestone 3 — decode-path wiring behind GGML_CUDA_W4A4_DOT8 (DONE)

Single file touched: `ggml/src/ggml-cuda/q8_repack/mul-mat.cu` (+~60 lines).
Gate: `static const bool w4a4_dot8 = getenv("GGML_CUDA_W4A4_DOT8") != nullptr;`
(function-local static, fork pattern, mirrors `GGML_RP_GENERIC_MMV`), ANDed with
`src0->type == GGML_TYPE_Q4_0` into `u4act`. Scope: the dense narrow-batch block
(`ne11 <= MMQ_RP_Q8_MMV_MAX_TOKENS`) ONLY — GEMM (`mmq_gemm_repacked`) and MoE
paths untouched.

### Changes

1. **Gate + activation-quant switch** (`ggml_cuda_mul_mat_repacked`): under
   `u4act`, the src1 quantization bypasses `repack_quantize_src1_q8_1` (and its
   q8_1 activation cache — variant 0 is the q8_1 layout, u4 must not share it)
   and calls `quantize_mmq_q8_1_u4_cuda` into a `ggml_cuda_pool_alloc<block_q8_1_u4>`.
2. **ALLOC MATH**: `block_q8_1_u4` is 20 B vs `block_q8_1`'s 32 B, but the
   BLOCK COUNT is identical (`ne10_padded/32 = x_stride` per token row), so the
   pool alloc is `ne13*ne12*ne11*x_stride` elements of the u4 type and `xs`
   keeps its "blocks per token column" meaning — the kernel's `c*xs + sb`
   indexing needs no change. `xq` is passed as `block_q8_1*` (kernel signature)
   reinterpreted from the u4 pool pointer; the U4ACT kernel casts back.
3. **Dispatch**: `ggml_cuda_mul_mat_repacked_nc_t` gains template param
   `bool U4ACT = false` (a runtime bool cannot feed a template arg — first build
   attempt failed on exactly that), threaded into all 9 kernel launches as
   `<..., WT, U4ACT>`. Only the Q4_0 case branches: `u4act ? <Q4_0,true> :
   <Q4_0>`. Single-token (`ne11==1`) under the gate goes to
   `mul_mat_vec_repacked_nc<2,1,1,2,64,Q4_0,true>` (NCOLS=1, same 2-row/
   RPL=2/64-lane shape as widths 4-8) instead of the stock slice dispatch.
4. **Gate-off = byte-identical stock**: every new branch is inside `if (u4act)`
   / template `U4ACT=false` compiles to the previous codepath; the only
   unconditional changes are the template signature (defaulted) and pointer-init
   `= nullptr` (dead without the gate).

### Verification

- Build: `flock /root/gpu.lock -c 'cmake --build /root/build-w4a4 -j24 --target llama-bench'` — exit 0.
- Instantiation check: `nm mul-mat.cu.o | c++filt` shows
  `mul_mat_vec_repacked_nc<2,1,1,2,64,Q4_0,true>` (new single-token) and
  `<2,1,{3..8},2,64,Q4_0,true>` alongside all stock `<...,false>` kernels.
- Runtime smoke (tiny TinyLlama-1.1B Q4_0, /root/tiny-Q4_0.gguf,
  sha256 da3087fb...b556): gate ON (`GGML_CUDA_REPACK_Q4_0=1
  GGML_CUDA_W4A4_DOT8=1`, GPU, -ngl 99) and gate OFF both run clean, no
  crash/OOB. NOT benched — gauntlet is Slot A's job.

### GAUNTLET (Slot A — runs on /root/build-w4a4/bin, NOT production)

Model: Mistral-Large fine-tune 123B Q4_0 repack:
`MON=/mnt/old/home/brad/models/Mistral-Large-123B fine-tune-v2-Q4_0/Mistral-Large-123B fine-tune-v2-Q4_0-00001-of-00002.gguf`
Env (prod stack, same as baseline-plx.sh): `source /root/therock-venv/bin/activate;
export LD_LIBRARY_PATH=/opt/rocm-7.2.3/lib HSA_XNACK=0 GGML_ENABLE_CUSTOM_AR=1
HSA_FORCE_FINE_GRAIN_PCIE=1 GPU_MAX_HW_QUEUES=8 GGML_TP_AR_SKIP_XBAR=1
GGML_CUDA_REPACK_Q4_0=1` (the W4A4 gate is the ONLY arm difference).
All runs under `flock /root/gpu.lock -c '...'`. Interleaved same-day:

```
cd /root/build-w4a4
# (a) gate-off anchor
env -u GGML_CUDA_W4A4_DOT8 ./bin/llama-bench -m $MON -ngl 999 -fa 1 -sm tensor -lm dio -p 2048 -n 128 -r 5
# (b) gate-on
GGML_CUDA_W4A4_DOT8=1 ./bin/llama-bench -m $MON -ngl 999 -fa 1 -sm tensor -lm dio -p 2048 -n 128 -r 5
# (c) gate-off again (drift control)
env -u GGML_CUDA_W4A4_DOT8 ./bin/llama-bench -m $MON -ngl 999 -fa 1 -sm tensor -lm dio -p 2048 -n 128 -r 5
# PPL A/B (go/no-go)
env -u GGML_CUDA_W4A4_DOT8 ./bin/llama-perplexity -m $MON -ngl 999 -fa 1 -sm tensor -lm dio -c 2048 -b 2048 -ub 2048 --chunks 12 -f /root/ppl-test.txt
GGML_CUDA_W4A4_DOT8=1 ./bin/llama-perplexity -m $MON -ngl 999 -fa 1 -sm tensor -lm dio -c 2048 -b 2048 -ub 2048 --chunks 12 -f /root/ppl-test.txt
```

Reference: tg128 19.4-19.6 / PPL 1.0564 (Q4_0-repack, gate off).
Decision rule: adopt-candidate iff >= +2% tg128 AND PPL delta <= +0.05.
NOTE: `llama-perplexity` is not in the llama-bench-only build tree — build it
first: `flock /root/gpu.lock -c 'cmake --build /root/build-w4a4 -j24 --target llama-perplexity'`.

## Milestone 3b — decode-path correctness fix (DONE, commit `88bbc3572`)

The first gauntlet (see `w4a4-gauntlet.txt` in the workspace archive) showed the
gate-ON arm ~+3.4% tg128 but **decode emitted 100x token id 0** and the
standalone quant harness (`/tmp/u4qc`) failed every multi-channel case
(97054 mismatches; single-channel cases passed). Root cause was three
interacting index/convention bugs, all fixed in `88bbc3572` (+55/-25, 3 files):

### Root cause 1 — k-major vs token-major `ib` (quantize.cu:644-650)

`quantize_mmq_q8_1_u4` wrote activation blocks with the MMQ-style k-major
formula `ib = z*(gridDim-derived) + k_block*ne1 + blockIdx.x`. The U4ACT decode
kernel reads `xq_u4 + c*xs + sb` with `xs = nkb` — i.e. it expects **token-major**
(all 32-value blocks of one token/channel column adjacent). For `ne1 == 1`
(single-token decode) the two layouts coincide, which is why the tiny smoke tests
passed; for `ne1 > 1` the layout was transposed, and the gridDim-derived `ib0`
additionally left gaps when `ne0 % 512 != 0` (the 4100x3 harness case).
Fix: `ib = ((int64_t) blockIdx.z * ne1 + blockIdx.x) * nkb + k_block`
(`nkb = ne0/32`), matching the kernel read and the host reference's
per-channel-contiguous blocks.

### Root cause 2 — nibble interleave mismatch (quantize.cu + repack-common.cu)

The device quantizer packed values pair-interleaved (`byte k = {val 2k, val 2k+1}`)
but canonical Q4_0 **weight** bytes are de-interleaved
(`ggml-quants.c:137-141`: `byte j = {val j low, val j+16 high}`). `sdot8`
therefore paired permuted activation values against weights — wrong dot products
even with correct block placement. Fix (both device kernel and host reference
`quantize_row_q8_1_u4`): thread `t` of the 8 covering a 32-value block owns
bytes `2t, 2t+1` with low nibbles = values `2t, 2t+1` and high nibbles =
values `2t+16, 2t+17`.

### Root cause 3 — the dp8a pairing correction (MISMATCH probe finding)

`repack-kernels.cuh:1101` computed the SUMW epilogue term as
`(float)(nib_sum_u4(pk) - 256)` on **unsigned** u32s. The 32 raw excess-8
nibble codes average ~7.5 each (sum ~240), so `nib_sum_u4(pk) < 256` for
roughly HALF of all blocks — the unsigned subtraction wrapped to ~2^32,
injecting ~1e9-magnitude garbage into the accumulator on those blocks. With
roots 1+2 fixed this was the remaining corruption: the MISMATCH probe showed the
dot agreeing on ~half the blocks and wildly diverging on the rest, exactly the
`< 256` boundary. Fix: `const float sw = (float) ((int) nib_sum_u4(pk) - 256);`
— signed subtraction before the float cast.

### Verification (post-fix gauntlet v2, 2026-09-22, build 88bbc3572)

- Harness `/tmp/u4qc`: **OVERALL: OK**, 0 mismatches on ALL cases including
  the previously failing multi-channel ones (4100x3: was 6472 mismatches).
- Bench: gate-on tg128 21.20 ± 0.05 vs gate-off legs 20.62 / 20.58
  (+2.81% / +3.01%); pp2048 within noise (163.85 vs 163.88/163.44).
- PPL: 1.0564 off vs 1.0564 on — delta 0.0000.
- Greedy gate-ON (`--temp 0 --top-k 1`, output via `-o` only in this build):
  exact primes 2,3,5,7,11,13,17,19,23,29 + correct tide sentence
  (`/root/leg7-fixed-answer.txt`).
- **VERDICT: ADOPT-CANDIDATE** — all decision-rule criteria met.
  Full tables: `/root/w4a4-gauntlet-v2.txt` (mi50).

## Milestone 4 — MoE narrow-batch dispatch wiring (DONE, commit `ad123ecdc`)

Single commit, 2 files (+127/-21): `q8_repack/mul-mat-id.cu` + `q8_repack/repack-kernels.cuh`.
Same gate `GGML_CUDA_W4A4_DOT8` (function-local static, ANDed with `src0->type == GGML_TYPE_Q4_0`).

### Fusion investigation (task 1)

`ggml_cuda_repack_mmv_id_fusion_supported` (repack-common.cu:85): base set =
Q8_0/MXFP4; KQUANT_MOE_FUSION (default-on) adds Q4_K/Q5_K/Q6_K/IQ4_NL.
**Q4_0 is in NEITHER set** — a Q4_0 MoE model never reaches the fused
`id1_fused` / `*_fused` epilogues at any width. No u4 gate-epilogue (acc_gate)
needed; the unfused-only v1 the brief anticipated is already the stock routing.

### Changes

1. **`mul_mat_vec_repacked_id1`** (narrow batch, 2-8 tokens): gains `bool U4ACT =
   false`, forwarded to `mul_mat_vec_repacked_nc_impl<...,WT,U4ACT>` — inherits
   the milestone-2b/3b kernel for free. **Unit fix**: under U4ACT the per-
   assignment column offset `ids_src1[a] * xs` must be applied in 20-byte
   `block_q8_1_u4` units (cast first, then add) — stock `block_q8_1*` math
   strides 32 B/block past the data.
2. **`mul_mat_vec_rp`** (single-token MoE decode): gains `U4ACT` param + the
   same dp8a branch (load_w_raw / XOR 0x88888888 / signed SUMW / sealed
   epilogue), same u4-unit column-offset guard on the HAS_IDS path.
3. **Dispatch** (mul-mat-id.cu): under `u4act`, both the narrow-batch and
   single-token arms bypass `repack_quantize_src1_q8_1` (q8_1 cache is variant-0
   layout) and call `quantize_mmq_q8_1_u4_cuda` into a
   `ggml_cuda_pool_alloc<block_q8_1_u4>` of `n_cols * x_stride` blocks (same
   block count as q8_1, 20 B elements — alloc math mirrors dense m3). Q4_0
   cases branch `u4act ? <...,true> : <...>`; all other types untouched.
   GEMM path (n_tokens > 8) untouched.

### Verification

- Build both gate states clean (`llama-bench llama-cli llama-perplexity`, exit 0).
- `nm` shows `mul_mat_vec_repacked_id1<4,2,2,Q4_0,true>` and
  `mul_mat_vec_rp<Q4_0,64,16,true,16,false,true>` alongside all stock `<...,false>`.
- Gate-off = stock (every branch inside `if (u4act)` / `if constexpr (U4ACT)`).

### GAUNTLET — Ornith-1.5-35B-A3B Q4_0 (Huihui i1), 2-rank GPUs 1,2, build ad123ecdc

Interleaved off/on/off, `-ngl 999 -fa 1 -sm tensor -lm dio -p 2048 -n 128 -r 5`:

| arm    | pp2048 t/s     | tg128 t/s      |
|--------|---------------:|---------------:|
| off-1  | 1835.97 ± 2.89 | 110.17 ± 0.62  |
| **on** | 1829.67 ± 3.48 | **104.28 ± 0.56** |
| off-2  | 1827.94 ± 5.43 | 109.78 ± 0.52  |

PPL A/B (`-c 2048 -b 2048 -ub 2048 --chunks 30 -f /root/ppl-test.txt`):
off 2.0643 ± 0.01812 / on 2.0643 ± 0.01812 — **dPPL 0.0000**.
Greedy gate-ON primes: exact 2,3,5,7,11,13,17,19,23,29 + correct tide sentence
(`/root/orn-w4a4-answer-full.txt`).

**VERDICT: REJECT for MoE.** Correctness criteria pass (dPPL 0.0000, primes
exact) but tg128 is **-5.3%** vs both controls (decision rule: >= +2%).
pp2048 within noise confirms the gate only touches decode, as designed.

Root cause (analysis, not measured): single-token MoE decode runs
`mul_mat_vec_rp` at LANES=16/NCOLS=1, where the per-block `nib_sum_u4` SUMW
recompute is consumed exactly once per block. The dense win came from the nc
kernel's RPL=2 row amortization (and the fused path would share SUMW across
up+gate). Unfused single-token pays the full nibble-sum overhead per row.
Follow-up options if MoE u4 is ever wanted: precompute SUMW at repack (+2 B
fp16 per 32-block, context-pack option (a)), or u4-ize the fused gate
epilogue so SUMW amortizes 2x. Gate stays opt-in; default off = stock.
Full log: `mi50:/root/orn-w4a4-gauntlet.txt`.

## Milestone 2 — activation wiring (mat-vec)

- Quantizer call site: replace `quantize_row_q8_1_cuda` / `quantize_mmq_q8_1_cuda` for src1 in the repacked non-canonical path — `mul_mat_vec_repacked_nc` family (`ggml/src/ggml-cuda/q8_repack/mul-mat.cu`), gated `GGML_CUDA_W4A4_DOT8=1`.
- Kernel loop: swap `ggml_cuda_dp4a` accumulation to `ggml_cuda_dp8a` on the uint4 activation operand; weight operand stays whatever the repacked layout provides (int4 nibbles need matching sign handling — dp8a treats both operands as signed nibbles, so the weight side needs its own bias or a sign-flip trick; study the repacked weight nibble order here).
- Epilogue: asymmetric offset — `sum += m_block * weight_sum` per 32-block (the classic Q4_0-style correction; weight column sums must be precomputed alongside the repacked weights).
- Gate + fallback: env unset keeps the q8_1 int8 path exactly as today.

## Milestone 3 — GEMM path + gauntlet

- `mmq_gemm_repacked` tile loop: dp8a across the K dimension, same epilogue correction.
- Full bench+PPL gauntlet per plan: llama-bench pp/tg arms vs q8_1-activation baseline, greedy quality gate, PPL A/B (the real go/no-go from the 2b study: est. ΔPPL 0.10–0.53 for scheme B).
- Coordinate GPU queue via `/root/gpu.lock` (Slot A owns bench windows; milestone 1 compile-only used flock for compile protection only).

## Milestone G1 — GEMM u4 skeleton (DONE, commit `8473f62c3`)

Template param `U4ACT` + asserts (`WT==Q4_0`, dense-only) threaded through
`mmq_gemm_repacked_impl`; LDS/register plumbing (`sWs` sw-sums array, `pw_s`
prefetch regs, U4ACT weight-prefetch branch: `load_w_raw` -> XOR 0x88888888 +
signed `nib_sum_u4` sw, store_stage branch). Activation fetch + compute dot
loop were TODO stubs. Plan: `w4a4-gemm-u4-plan.md` (workspace `w4a4-g1`).
Note: the G1 commit message claimed the launch wrappers were already threaded;
they were not — corrected in G2.

## Milestone G2 — GEMM u4 dot loop functional (DONE, unwired)

The sealed form-B dot loop in `mmq_gemm_repacked_impl` compute_stage, GEMM
twin of the decode U4ACT branch (worked example: `mul_mat_vec_repacked_nc_impl`
at `88bbc3572`). Per 32-value sub-block:
`acc[r][n] += d * (d4 * ((float)idot + 8.0f*sw) + mu*sw)` with
`idot = mmvq_dp8a_u4(w_tc, a)` — 4 dot8 per block per (r,n) vs the stock
8 dp4a.

### Changes (repack-kernels.cuh + repack-common.cuh)

1. `rp_x_sub_u4_from_blocks(blocks, idx, s)` (repack-common.cuh): fetch one
   u4 sub-block from the token-major `block_q8_1_u4` pool —
   `blocks[tok*nkb + sb]`, nkb = ne0/32. qs assembled as 4x u32 (20 B
   stride is NOT 16 B aligned — never `reinterpret_cast<uint4*>` on it).
   `out.d` = d4, `s` = m, q1 zeroed (unused under U4ACT).
2. `prefetch_x_checked` / `prefetch_x_full` U4ACT branches: real fetch via
   the helper (was zeroed stubs). Dead token/slab -> d4=0 AND m=0 so the
   sealed epilogue collapses to 0 regardless of the weight term.
3. `prefetch_w_full` U4ACT branch added (G1 only covered the checked path;
   the full-tile path would have staged the expanded lo/hi pair).
4. `compute_stage` U4ACT branch: the sealed dot loop. Activation operands
   read straight from LDS inside the n loop — the hoisted `a[]/d4[]/mu[]`
   form spilled 178 VGPRs at launch_bounds(256,2); the LDS-direct form
   spills 4.
5. LDS budget: `sWs` shares storage with `sW_hi` via a named union
   (`rp_w_planes` + reference aliases — an anonymous block-scope union
   cannot be lambda-captured). The dot8 path never touches the hi plane,
   the stock path never touches sWs. Keeps the static LDS total at the
   stock 32768 B in BOTH gate states; the G1 `sWs[1][1]` addition had
   pushed the affine 64-wide GEMM to 32772 B and dropped
   launch_bounds(256,2) occupancy to 1 (-Wpass-failed on the Q4_0/Q5_K/
   Q6_K U4ACT=false variants).
6. Launch wrappers `mmq_gemm_repacked` / `mmq_gemm_repacked_w32`: `U4ACT`
   template param threaded to the impl (default false = byte-identical
   stock).

### Build + instantiation verification

- `flock /root/gpu.lock -c 'cmake --build /root/build-w4a4 -j24'` — clean,
  exit 0. The 5 pre-existing occupancy warnings (Q4_0/Q5_K/Q6_K 64-wide,
  present at `cd3be9981` before G1) are unchanged — the union fix restored
  parity; no new warnings introduced by G1/G2.
- Force-instantiation check (kept `#if 0` at end of repack-kernels.cuh):
  standalone compile of `mmq_gemm_repacked<false,2,4,Q4_0,true>` and
  `mmq_gemm_repacked_w32<false,1,8,Q4_0,true>` with the exact build flags
  (clang++ ROCm 7.2.3, --offload-arch=gfx906, -x hip, -O3). Device asm:
  256 `v_dot8_i32_i4` in the 64-wide body, 128 in the w32 body, zero
  `v_dot4_i32_i8` in the u4 path — real codegen, not a stub. VGPR spills:
  4 (64-wide, LDS-direct form) / 40 (w32) vs stock 20 / 0.
- Gate states: the gate is runtime (`GGML_CUDA_W4A4_DOT8`); the single
  build carries both variants. U4ACT=false is exercised by the full build;
  U4ACT=true by the force-instantiation check. No dispatch selects
  U4ACT=true yet — G3 wires it.

### G3 wiring (NOT in G2)

1. mul-mat.cu GEMM quantize block (~:153-215): under `u4act`, swap
   `quantize_mmq_q8_1_cuda` for `quantize_mmq_q8_1_u4_cuda` into a
   `ggml_cuda_pool_alloc<block_q8_1_u4>` sized
   `ne13*ne12*chunk_ne11*nkb` (nkb = ne10_padded/32; 20 B/block vs the
   q8_1_mmq 144 B/group = 36 B/block-equivalent — workspace shrinks).
   Post-3b quantizer is token-major `ib = (z*ne1 + x)*nkb + k_block` —
   matches the kernel's `[tok*nkb + sb]` read directly.
2. Launch switches (~:400-439 64-wide, ~:443-486 32-wide): Q4_0 case
   branches `u4act ? <...,true> : stock`, mirroring the nc-path pattern.
3. Reuse the existing `w4a4_dot8` gate (mul-mat.cu:52-53). Gate-off stays
   byte-identical stock.
4. Gauntlet (Slot A, /root/gpu.lock): pp2048 >= +2% vs q8_1-activation
   baseline, dPPL <= +0.05. pp2048 hits the 64-wide arm (ne11=2048);
   32-wide covers 9..127.

## Milestone M2 — per-8 u4p decode kernel (DONE, unwired)

C1/B8 (docs/design-w4a4-per8.md §4, M1 seal docs/w4a4-per8-m1.md). Adds the
per-8 asymmetric activation variant to the narrow-batch mat-vec, form-1 epilogue.
No dispatch selects it yet (M3 wires the gate).

### Changes

1. repack-common.cuh: `block_q8_1_u4p` { qs[16]; ggml_half d4[4]; ggml_half m[4]; }
   + static_assert(sizeof == 32) + layout comment (mirrors u4p_host.h M1: word k =
   values 8k..8k+7, byte j = {val 8k+2j low, 8k+2j+1 high}, SIGNED (q-8)&0xF
   nibbles matching the dp8a convention).
2. repack-kernels.cuh after mmvq_dp8a_u4: `nib_sum_word` (8 raw codes per u32)
   + `mmvq_dp8a_u4p(w_raw, a) -> mmvq_dp8a_u4p_result { int idot[4]; float sw[4]; }`
   — 4 UNCHAINED dp8a accumulators (one per sub-block; the algebra needs them
   separate, no per-32 chaining) + SW_k = nib_sum_word(raw_k) - 64 (signed before
   the float cast — the 3b root-cause class).
3. mul_mat_vec_repacked_nc_impl: new template param `U4P8 = false`, independent
   of U4ACT (both selectable; static_asserts: U4P8 => WT==Q4_0, !HAS_FUSION,
   !(U4ACT && U4P8) — the two activation layouts cannot mix). U4P8 branch in the
   sb loop: load_w_raw (weights stay in the SAME 32-value-block repack layout —
   one uint4 carries sub-blocks k=0..3), activations read as
   const block_q8_1_u4p* (32 B blocks; the 32 B stride keeps every block 16 B
   aligned so qs loads as a full uint4 — no 4xu32 workaround like the 20 B u4).
   FORM 1 epilogue per sub-block, summed:
   acc += d * ( d4_k * (idot_k + 8*SW_k) + m_k * SW_k ).
   The fp16-stored m-fold variant is NOT exact (M1 seal: 29/1504 fail) — do not
   substitute. Dead rows: zero the d slot via rp_slot_keep, as shipped.
4. Wrapper threading: mul_mat_vec_repacked_nc gained U4P8 (default false =
   byte-identical stock instantiation).

### Verification

- Force-instantiation save-temps check (clang++ ROCm 7.2.3, --offload-arch=gfx906,
  -O3, exact build flags): `mul_mat_vec_repacked_nc<2,1,4,2,64,Q4_0,false,true>`
  device asm: **32 v_dot8_i32_i4** = RPL(2) x NCOLS(4) x 4 sub-blocks — matches
  the template geometry exactly (same count as the m2b per-32 check: the dot count
  is per-32-value-block either way; per-8 only moves the epilogue granularity).
  Zero v_dot4_i32_i8 in the u4p branch. NumVgprs 64,
  vgpr_spill_count 27 (vs 19 for the shipped per-32 U4ACT twin at the same
  geometry — the 4-accumulator/4-scale epilogue pressure flagged in design §8;
  the nc kernel has no launch_bounds occupancy target, so spills add scratch
  traffic only, no occupancy cliff. A sw-hoist experiment measured 28 spills
  (worse) and was reverted; M3 gauntlet measures the real cost). Dummy removed
  after the check (box clean).
- Build both gate states clean: `flock /root/gpu.lock -c "cmake --build
  /root/build-w4a4 -j24 --target llama-bench llama-cli llama-perplexity"` —
  exit 0. Warnings: the 5 pre-existing mmq_gemm_repacked occupancy warnings
  (U4ACT=false variants, present since before G1) unchanged; zero new warnings
  from the M2 code. U4P8=false is exercised by the full build; U4P8=true by the
  force-instantiation check. No dispatch selects U4P8=true yet.

### M3 dispatch note (NOT in M2)

At the mul-mat.cu nc call sites under the w4a4_dot8 gate, value 2 selects u4p:
- Quantizer: quantize the activation pool with the u4p host/device quantizer
  (per-8 d4/m; M1 reference: w4a4_study/u4p_host.h quantize_row_q8_1_u4p)
  into ggml_cuda_pool_alloc<block_q8_1_u4p>.
- Pool type + strides: xs stays "blocks per token column" (ne0/32) — the kernel
  indexes (block_q8_1_u4p*)xq + c*xs + sb; only the element size changes
  (32 B vs 20 B per-32 u4). Pool byte size = ne1 * (ne0/32) * 32.
- Template arg: pass U4P8=true (U4ACT=false) at the mul_mat_vec_repacked_nc
  launch sites (dense nc path; id1/MoE wrappers unchanged — reject stands).
- Gate-off (=0) and per-32 (=1) dispatch stay byte-identical to shipped.
