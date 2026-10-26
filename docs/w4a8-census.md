# W4A8 Go/No-Go ASM Census - gfx906 repacked-GEMM & mat-vec (floor-u4x16)

**Date:** 2026-10-21 * **Target:** `mi50:/root/floor-u4x16/build/bin/libggml-hip.so.0` (-> `.so.0.26.0`, built 2026-10-21 16:36) * **Toolchain:** ROCm 7.2.3 llvm (`/opt/rocm-7.2.3/lib/llvm/bin/llvm-objdump`)
**Mode:** read-only static ISA census; GPU untouched, no `gpu.lock`, no benchmark interference.

## Method (per skill census discipline)

1. `roc-obj-ls` on the .so -> 144 gfx906 code objects (one ELF per kernel group in the `__CLANG_OFFLOAD_BUNDLE__` fatbin).
2. `dd` each entry at its recorded offset/size -> per-kernel ELFs (`/tmp/w4a8-census/NN.elf` on mi50). Repacked u4 kernels live in **objects 48 (gate1/b1 variants) and 49 (stock/b0 variants)**; Q4_0 = `ggml_type2`.
3. `llvm-objdump -d --mcpu=gfx906 --start-address/--stop-address` per symbol (note: `--disassemble-symbols` silently returns empty on some objects - address-range mode is the reliable path; objdump addresses print with uppercase hex).
4. Hot loops = largest backward-branch bodies per kernel; spills from `.note` YAML (`vgpr_spill_count`, `private_segment_fixed_size`) cross-checked against `buffer_store_dword` scratch ops in the disassembly.
5. Raw machine output: `census_raw.json` (same dir).

## Step 4 - toolchain emit verification (PASS)

Probe `p.cu` (4 builtins, ROCm 7.2.3 `hipcc -O2 --offload-arch=gfx906`): device code object contains **2x `v_dot4_i32_i8` + 2x `v_dot8_i32_i4`** -> the i8 dot form (`v_dot4_i32_i8`, via `__builtin_amdgcn_sdot4`) **does emit** on this toolchain. Bogus-symbol control failed to compile as required (probe is meaningful).

## Step 5 - operand-pairing semantics: **UNVERIFIED**

GPU execution is not permitted for this census and CPU emulation of dp4a/dp8a semantics would only re-confirm the documented ISA behavior, not the hardware. Per instructions: **UNVERIFIED**. (Documented expectation, untested here: `v_dot4_i32_i8` pairs signed byte k of a with signed byte k of b, 4 products + c; `v_dot8_i32_i4` pairs signed nibbles little-endian.)

## Census table (Q4_0 / ggml_type2 repacked kernels)

| Kernel (demangled, abbreviated) | obj | total instrs | v_dot4_i32_i8 | v_dot8_i32_i4 | hot-loop instrs | hot-loop dots (density) | hot-loop loads | VGPR / spills (priv seg) | scratch in hot loop |
|---|---|---|---|---|---|---|---|---|---|
| `mmq_gemm_repacked<true,2,4,Q4_0,?>` (gate1 GEMM, q8_1 act) | 48 | 2256 | 512 | 0 | 655 | 256 dot4 (**39.1%**) | 67 | 128 VGPR / **20 spills** (52 B) | 6 |
| `mmq_gemm_repacked<false,2,4,Q4_0,false,false,true,?>` (dot8/u4act GEMM) | 49 | 2159 | 0 | 256 | 647 | 128 dot8 (**19.8%**) | 42 | 127 VGPR / 0 (0 B) | 0 |
| `mmq_gemm_repacked<false,2,4,Q4_0,?>` (stock q8_1 GEMM) | 49 | 2160 | 512 | 0 | 653 | 256 dot4 (**39.2%**) | 65 | 128 VGPR / **20 spills** (52 B) | 6 |
| `mmq_gemm_repacked_w32<true,?,Q4_0,?>` | 48 | 622 | 64 | 0 | 257 | 64 dot4 (24.9%) | 33 | 83 / 0 | 0 |
| `mmq_gemm_repacked_w32<false,?,Q4_0,?>` | 49 | 549 | 64 | 0 | 255 | 64 dot4 (25.1%) | 32 | 83 / 0 | 0 |
| `mul_mat_vec_rp<Q4_0,16,16,false,64,?>` | 49 | 137 | 8 | 0 | 52 | 8 dot4 (15.4%) | 5 | 23 / 0 | 0 |
| `mul_mat_vec_rp<Q4_0,64,16,true,16,?>` | 48 | 171 | 8 | 0 | 56 | 8 dot4 (14.3%) | 5 | 28 / 0 | 0 |
| `mul_mat_vec_repacked_id1<4,2,2,Q4_0,false,?>` (dot4 mat-vec) | 48 | 291 | 16 | 0 | 103 | 16 dot4 (15.5%) | 7 | 38 / 0 | 0 |
| `mul_mat_vec_repacked_id1<4,2,2,Q4_0,true,?>` (dot8/u4act mat-vec) | 48 | 351 | 0 | 8 | 164 | 8 dot8 (**4.9%**) | 6 | 59 / 0 | 0 |
| `mul_mat_vec_repacked_nc<2,1,1,2,64,Q4_0,?>` (dot8/u4act mat-vec) | 49 | 303 | 0 | 8 | 186 | 8 dot8 (**4.3%**) | 6 | 44 / 0 | 0 |

## Reading

- **The W4A8 shape already exists in the binary** and is the heavier one. The i8-activation (dot4) form of the production w64 GEMM needs **2x the dot instructions** (256 vs 128 per hot loop) and **~1.5x the loads** (65-67 vs 42) versus the dot8/u4act form - same MACs, strictly more instructions and bytes.
- **The dot4 GEMM hot loop is dot-dense (39%), not bandwidth-dominated.** The go criterion's premise ("doubling dot instructions is nearly free") fails exactly where the program's wall time lives: the w64 tiled GEMM is the production path (w32 is not; per program notes).
- **Register pressure is already at the cliff in the dot4 form:** 128 VGPR + 20 VGPR spills (52 B private segment, 6 scratch ops inside the hot loop) for both b1 and b0 q8_1 GEMM variants. The dot8/u4act GEMM fits in 127 VGPR with **zero spills**. A W4A8 kernel would start from the spilled side of that cliff.
- **Mat-vec kernels do pass the bandwidth criterion** (4-15% dot density, zero spills): doubling their dot count (8->16 per ~160-instr loop, ~+5%) would be nearly free. But mat-vec is the narrow-batch path, not the production GEMM lever.

## Verdict: **NO-GO**

The go criterion - u4 hot loops bandwidth-dominated (low dot density, zero spills) - is **not met by the production GEMM**: its i8-activation (dot4) form runs a 39%-dot-dense hot loop with 20 VGPR spills, vs the dot8 form's 20%-density zero-spill loop. A hypothetical W4A8 Phase-1 kernel would double dot-instruction count in a dot-dense loop and inherit the spill penalty; the census says the doubling is **not** nearly free on the w64 path. The only kernels meeting the criterion (mat-vec, 4-15% density, zero spills) are not where the prefill wall time is. No Phase-1 kernel proposal is warranted on this evidence.

*Caveats: hot-loop identification is static (largest back-edge body); no timing claims. Pairing semantics UNVERIFIED (no GPU run permitted). Census objects 48/49 of 144; non-Q4_0 repack variants (iq4nl/q6k, obj 47) out of scope.*

## Addendum (2026-10-21, post-census, parent agent)

- Pairing semantics: VERIFIED EMPIRICALLY on hardware (ROCm0, sub-5s probe,
  flock'd): v_dot4_i32_i8 with known-distinct byte lanes returns the
  LITTLE-ENDIAN byte-order dot product exactly (hw=192 = {-3}x{16} +
  {1}x{32} + {7}x{48} + {-2}x{64}; reversed-order prediction 48 rejected).
  Emission re-confirmed via --save-temps (2x v_dot4_i32_i8 in the probe ISA).
- The census "UNVERIFIED" caveat above is thereby closed; the NO-GO verdict
  stands with no open caveats.

## Addendum 2 (2026-10-22, external fact-check pass)

- Independent re-verification (separate model, fresh disassembly of objects
  48/49 from the extracted ELFs): all 10 table rows reproduce exactly from
  census_raw.json; dot counts confirmed per-symbol from ISA (512 dot4 in
  BOTH gate1 and stock q8_1 GEMM variants, 256 dot8 u4act, 64 w32, 8/8/16
  mat-vecs); spill metadata corroborated by direct buffer_store_dword
  count (exactly 20 in the stock-form GEMM). Verdict stands.
- CORRECTIONS: (a) the scratch dir holds 145 extracted ELFs vs 144 listed
  code objects (extraction artifact, one extra; count in the text refers
  to roc-obj-ls entries); (b) the epilogue bit bound: 15 significant bits
  is the ACCUMULATED sub-block bound (|D| <= 16 x 127 x 12 = 24384); the
  per-term bound is ~11 bits (127 x 12). Either way it exceeds fp16's 11
  bit mantissa, so the fp32-fold conclusion is unchanged.
- NEW QUEUE ITEM from this pass: the STOCK-FORM w64 GEMM (the kernel that
  runs every floored/narrow-width fallback and every GGML_CUDA_W4A4_SKIP
  tensor) is the spilled one (128 VGPR, 20 VGPR spills, 52 B private
  segment, 6 scratch ops in-loop). A spill fix or tile restructure on that
  kernel would speed the fallback path wherever the u4 gate disengages -
  small, well-scoped kernel-tuning target, disjoint from the W4A8 verdict.
