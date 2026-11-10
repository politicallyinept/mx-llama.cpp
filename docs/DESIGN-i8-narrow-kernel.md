# DESIGN: i8narrow - ruquant-style narrow-batch Q4_0 kernel with EXACT 8-bit activations

Worktree: `/root/int4-noquant-wt`, branch `INT4-noquant` @ fb18241213.
Status: DESIGN + CPU seal DONE; HIP skeleton AUTHORED, NOT COMPILED (build pause point:
box RAM below the 12 GB build gate and GPU1/GPU2 busy - see taskC-LEDGER.md).
Companion seal: `seals/seal_i8split.cc` (CPU-only, g++ -O2, no deps, 16 invariants PASS).

---

## 1. Scope and why this is allowed by the census

The W4A8 NO-GO (docs/w4a8-census.md:44) is DECISION-FINAL for the WIDE tiled GEMM:
its i8-activation (dot4) form runs a 39%-dot-dense hot loop with 20 VGPR spills
(census table rows :26,:28). This design does NOT re-open that verdict.

The narrow mat-vec kernels pass the census's own criterion instead, quoted verbatim
from docs/w4a8-census.md:42:

> Mat-vec kernels do pass the bandwidth criterion (4-15% dot density, zero spills):
> doubling their dot count (8->16 per ~160-instr loop, ~+5%) would be nearly free.
> But mat-vec is the narrow-batch path, not the production GEMM lever.

Census rows this design builds on:
- `mul_mat_vec_repacked_nc<2,1,1,2,64,Q4_0,?>` (dot8/u4act mat-vec): hot loop 186
  instrs, 8 dot8 (4.3% density), 6 loads, 44 VGPR, 0 spills (census:35).
- `mul_mat_vec_rp<Q4_0,16,16,false,64,?>` (stock dot4 narrow): hot loop 52 instrs,
  8 dot4 (15.4%), 23 VGPR, 0 spills (census:31).

Doubling the dot8 count (4 -> 8 per 32-value block per token, see §3) is exactly the
"nearly free" case the census describes.

## 2. The narrow window in the current tree (verified at fb18241213)

- `MMQ_RP_Q8_MMV_MAX_TOKENS` = **8** - `ggml/src/ggml-cuda/q8_repack/repack-common.cuh:20`.
- `rp_narrow_max`: runtime upper bound, env `GGML_CUDA_RP_NARROW_MAX_TOKENS`, clamped
  to `[1, MMQ_RP_Q8_MMV_MAX_TOKENS]`, default 8 - `q8_repack/mul-mat.cu:320-330`.
- Narrow window: `if (ne11 >= 1 && ne11 <= rp_narrow_max)` - `mul-mat.cu:343`; the
  comment at :344-345 states widths 2..8 are the MTP speculative-verify batch and
  that "the tiled path below computes a full 32-wide tile whatever the batch is".
- **Tiled GEMM starts at `mul-mat.cu:509`** ("Multi-token: grouped MMQ Q8_1 layout
  for the tiled GEMM"), i.e. everything with `ne11 > rp_narrow_max` (>= 9 by default).
- Narrow Q4_0 dispatch: `ne11 == 1` -> `mul_mat_vec_repacked_nc<2,1,1,2,64,...>`
  launches (`mul-mat.cu:410-443`); widths 2..8 -> `ggml_cuda_mul_mat_repacked_nc_t`
  (`mul-mat.cu:444-475`; stock q8_1-activation Q4_0 call at :471-474, u4act at :462-470).
- Kernel impl: `mul_mat_vec_repacked_nc_impl` - `q8_repack/repack-kernels.cuh:2142`;
  strided block loop `for (sb = lane; sb < n_blocks; sb += LANES)` at :2216;
  u4act dot8 branch :2217-2274; stock dot4 branch :2397-2472 (8 `ggml_cuda_dp4a`
  per block per (r,c) at :2437-2444, affine epilogue `acc += dw2.x*dx*(ilo+ihi) -
  dw2.y*sx` at :2448); DPP reduction + guarded write :2475-2493.

**Width 16 is OUTSIDE the current window** (clamped at 8). Serving width 16 from this
kernel requires either raising `MMQ_RP_Q8_MMV_MAX_TOKENS` or a new env-gated dispatch
arm for the i8narrow kernel; T=1 stays on the existing paths unless the A/B says
otherwise. This is a dispatch change, not a numerics change.

## 3. The exact-8-bit scheme (the core identity)

Activation quantization is the STOCK q8_1 quantizer, unchanged: per 32-value block
`d = amax/127` stored f16, `a_i = roundf(x_i/d)` as int8 two's complement
(`ggml/src/ggml-quants.c:302` `quantize_row_q8_1_ref`). Real activations land in
`[-127, +127]`; `-128` is unreachable from real data but MUST be handled exactly by
the kernel (the seal exercises it - §8).

Weights are the existing repacked Q4_0 plane: raw canonical bytes (excess-8 codes,
interleaved pairs), read via `rp_traits<GGML_TYPE_Q4_0>::load_w_raw`
(`repack-common.cuh:581-586`), plus the f16 d plane and the int16 SIGNED weight-sum
plane `load_sw` (`repack-common.cuh:592-595`). Weight two's-complement nibbles:
`w = c - 8` obtained by `XOR 0x88888888` per word (the shipped u4act convention,
`repack-kernels.cuh:2244-2245`), `w in [-8, 7]`.

**Nibble split of each int8 activation** `a in [-128, 127]`:

```
a_hi = a >> 4        (ARITHMETIC shift; a_hi in [-8, 7] - a two's-complement nibble)
a_lo = a & 0xF       (a_lo in [0, 15]; stored biased as (a_lo ^ 8), i.e. the signed
                      nibble value the dot8 sees is a_lo - 8 in [-8, 7])
a    = 16*a_hi + a_lo              (exact, integer identity)
```

**Per 32-value block, the integer dot identity** (all quantities int32):

```
D_hi  = mmvq_dp8a_u4(w_tc, hi_plane)   = sum_i w_i * a_hi_i          (4 chained v_dot8_i32_i4)
D_lo  = mmvq_dp8a_u4(w_tc, lo_plane)   = sum_i w_i * (a_lo_i - 8)    (4 chained v_dot8_i32_i4)
SUMW  = (int) nib_sum_u4(raw) - 256    = sum_i w_i                   (SIGNED - the 3b class,
                                      repack-kernels.cuh:2247-2260; or the load_sw plane)

S = 16*D_hi + D_lo + 8*SUMW
  = sum_i w_i * (16*a_hi_i + a_lo_i - 8 + 8)
  = sum_i w_i * a_i                                (EXACT int32, no correction error)
```

The `+8*SUMW` term is the exact cancellation of the lo-plane bias; it is one scalar
FMA per (row, block), shared across all T tokens.

**Equivalence to the stock int8 path (the no-quality-cost argument).** The stock
q8_1 narrow path computes, per block, `I = (ilo + ihi)` via 8 dot4 on the masked
excess-8 nibble bytes (`repack-kernels.cuh:2437-2444`) and the affine epilogue
`d*dx*I - 8d*sx` at :2448, where `sx = dx*sum(a)`. Its integer partial sum is
`I - 8*SA` with `SA = sum_i a_i`. Since `w = c - 8`:

```
S = sum_i (c_i - 8) * a_i = I - 8*SA        (integer equality - SEALED, §8)
```

so the i8split path consumes **exactly the same integer partial sums as the stock
int8 path**. Same quantization points, same integers; the only difference is the
fp32 fold form (below), which is the same rounding class as stock. No quality cost.

**int32 headroom:** per block `|S| <= 32*8*127 = 32512` (< 2^15); `|16*D_hi| <= 32768`,
`|D_lo| <= 2048`, `|8*SUMW| <= 2048`. All int32-safe with huge margin (seal-checked).

**fp32 epilogue (documented fold):** per block, `acc += d_w * (d_a * (float)S)` -
one int->float convert, two multiplies, one add per (row, token, block). Cross-block
accumulation is fp32 in a FIXED order (per-lane sequential over blocks, then a
fixed-order DPP/LDS tree), so results are deterministic. The fp32 fold-vs-dequant
error is the same class as the stock path's (seal bar: 1e-3 relative over a 160-block
row; observed 2.3e-4).

**Dot cost:** 8 dot8 per 32-block per token vs 4 for u4act (2x), vs 8 dot4 for stock
(same instruction count, half the operand bytes on the weight side, and each dot8 is
8 MACs vs 4). Per census:42 this is nearly free in the narrow loop.

## 4. Nibble pairing spec (verified from code, not comments)

Canonical Q4_0 pairing, from the quantizer source: `quantize_row_q4_0_ref`,
`ggml/src/ggml-quants.c:113`, pairing loop :137-146 - **byte j = {value j (low
nibble), value j+16 (high nibble)}**, excess-8 codes.

The activation side MUST use the same byte order so dot8 pairs value i with value i:
- device u4 quantizer `quantize_mmq_q8_1_u4` (`quantize.cu:579-672`): comment :601-607
  records the historical bug ("the old pair-interleaved {2k,2k+1} order mismatched
  the weight nibble order, so sdot8 paired permuted values and the decode emitted
  garbage"); store at :642-650 is canonical order, signed `(q-8)&0xF`.
- host ref `quantize_row_q8_1_u4` (`repack-common.cu:767-800`): pairing :786-794,
  signed form since the 2026-10-22 fix (:763-766). NOTE the WARNING comment at
  `repack-common.cuh:206-211` ("the host reference ... packs RAW unsigned q") is
  STALE for the host ref - the code at repack-common.cu:794 stores `(q-8)&0xF`.
  Read the code, not the comments - exactly as the brief warned.
- `ggml_cuda_dp8a` operand layout: `common.cuh:755-758` - byte k holds nibble pair
  {2k, 2k+1} little-endian; the portable fallback :763-772 sign-extends nibble i
  from bits 4i..4i+3. The pairing works because BOTH operands use the same
  {j, j+16} byte order, so positional pairing == value pairing.

**i8split planes inherit the canonical order:** hi-plane byte j = {a_hi(j) low,
a_hi(j+16) high}; lo-plane byte j = {(a_lo(j)^8) low, (a_lo(j+16)^8) high}.
The seal includes a wrong-pairing detector (§8) that distinguishes {i,i+16} from
{2k,2k+1} on 64/64 randomized blocks.

## 5. Layouts

### 5.1 Weights (reuse, plus one optional new plane)

v1 reads the EXISTING repacked Q4_0 planes (`rp_traits<Q4_0>::geom`,
`repack-common.cuh:553-565`): qs plane (raw canonical, row stride `rs`,
`rs_u4 = rs/16` uint4s), f16 d plane at `dplane_off`, int16 SUMW plane at
`swplane_off` (SUMW plane must be allocated: `GGML_CUDA_W4A4_SUMW` semantics,
`repack-common.cuh:556-560`). No new weight bytes for v1 correctness.

v2 perf knob (ruquant lesson "weights read in place with one contiguous
16-byte-per-lane run per wave via interleaving"): a wave-interleaved qs variant
where, for each 64-row wave group and each K-block sb, the 64 lanes' 16 B blocks
are contiguous: `off = ((row/64)*n_blocks + sb)*1024 + (row%64)*16`. This makes
each wave's per-step weight read one contiguous 1 KB run. Template flag
`INTERLEAVED`; host repack function declared in the skeleton, fill deferred.

### 5.2 Activations (new, three split planes, token-major)

For a token row of K values (`nkb = K/32`):
- `hi[token*nkb + sb]` : 16 B - 32 signed nibbles `a>>4`, canonical pairing.
- `lo[token*nkb + sb]` : 16 B - 32 nibbles `(a&0xF)^8`, canonical pairing.
- `d [token*nkb + sb]` : f16 - the stock q8_1 scale (`amax/127`).

34 B per (token, block) total vs block_q8_1's 36 B - strictly less activation
bandwidth than the stock narrow path, and both nibble planes are 16 B-aligned
(the 20 B `block_q8_1_u4` single-plane trap of `repack-common.cuh:358-360` is
avoided by splitting planes instead of interleaving scale into the vector).
Quantizer: same int8 codes as `quantize_row_q8_1_ref` (bit-identical), then the
lossless split. Host reference + device quantizer declared in the skeleton.

### 5.3 One layout for every batch width

The planes are per (token, block) and width-independent: widths 1..16 read the
same layout with T rows (ruquant lesson: one layout for every batch width). No
MMQ k-major grouping, no per-width repack.

## 6. Kernel geometry, registers, buffers

Template: `mul_mat_i8narrow_q40<int T, int NWAVES, int KSPLIT, bool INTERLEAVED>`.

- **One lane per output row** (ruquant w4a4_skinny shape). Block = `NWAVES*64`
  threads (default 4 waves = 256 threads); wave w owns rows
  `[blockIdx.x*NWAVES*64 + w*64, +64)` in the interleaved layout (lane l = row
  base + l). Grid.x = ceil(N / (NWAVES*64)); grid.z = KSPLIT.
- **Per-lane step** (32 K-values): 1x 16 B global weight load (coalesced across the
  wave when INTERLEAVED), 4x XOR, SUMW from plane (1x 2 B, wave-coalesced), then
  per token c: 2x LDS uint4 (hi, lo), 4+4 chained dot8, 1 int fold (`16*D_hi + D_lo`
  = shl+add), 1 fp32 FMA-class update `acc[c] += d_w*(d_a[c]*(t + 8*sw_f))`.
- **Activations staged one contiguous scalar load per step** (ruquant lesson): the
  block cooperatively stages the step's `T*34 B` of hi/lo/d for the wave's K-block
  into LDS (vector loads: T x 2x16 B + T x 2 B), then each lane reads its tokens'
  staged values from LDS. LDS per block per step: `NWAVES*T*34 B` single-buffer
  (T=16, NWAVES=4: 2.2 KB) - double-buffered: 4.4 KB, trivial vs 64 KB.
- **Register plan (VGPR budget, target <= 64 => 4 waves/SIMD):**
  acc[T] fp32 (T), w_tc uint4 (4), sw (1), LDS/global pointers + row id (6),
  d_a staged (T or LDS-read), dot temps D_hi/D_lo/t (3), loop counters (4).
  T=1: ~20; T=4: ~28; T=8: ~44 (matches the shipped nc u4act kernel's 44 VGPR,
  census:35); T=16: ~60 - tight, allow `__launch_bounds__(NWAVES*64, min_blocks)`
  tuning and a 2-waves/SIMD fallback instantiation if ptxas reports spills.
  ZERO spills is a hard bar (the census spill cliff, census:41).
- **Activations in scalar registers staged one step ahead where cheap** (ruquant
  lesson): at T <= 4 the hi/lo uint4s for the next step are prefetched into
  registers; at T >= 8 rely on LDS + the barrier below. Prefetching the next
  block's activations from 24 tokens up is NOT needed at T <= 16 (the lesson's
  own threshold); knob `PREFETCH_A` left in the template.

## 7. Reduction, K-split, barriers

- **Within a 32-block:** all adds are integer (dot8 chains + the fold) - exact in
  any order; the lane->block assignment is therefore free (this is where the
  "integer adds are exact in any order" property is used).
- **Across blocks within a lane:** fp32 fold in fixed sequential order (deterministic).
- **Wave level (KSPLIT > 1):** each wave reduces a disjoint contiguous K-segment of
  the block's rows; waves reduce their fp32 per-(row,token) partials through ONE
  shared LDS buffer all waves add into (ruquant lesson: 1.25-1.76x at 32-64 tokens),
  in a FIXED tree order (wave0+wave1, wave2+wave3, then combine) - deterministic.
  Note (honest deviation): cross-block integer accumulation across waves is NOT
  possible with Q4_0 per-block f16 scales (the fold must happen per block), so the
  wave-level LDS reduction is fp32-in-fixed-order, not integer. The integer-exact
  freedom applies within a block only.
- **Workgroup level (grid.z KSPLIT):** partials go to a fp32 workspace, reduced by
  a fixed-order second pass (no fp32 atomics - nondeterministic order). KSPLIT=1 is
  the v1 default for N=17408 (68 blocks of 256 rows on 60 CUs); KSPLIT=4 recommended
  for N=5120 (20 blocks otherwise underfills 60 CUs).
- **Barriers: exactly one `__syncthreads()` per token block / K step**, placed
  between the cooperative activation staging store and its consumption (ruquant
  lesson: without it the compiler hoists the scalar loads and spills). Weight loads
  are per-lane global and need no barrier. No barrier inside the per-token loop.
- **Wave/offset policy:** avoid power-of-two run lengths - K=5120 -> 160 blocks
  (5x32), K=17408 -> 544 blocks (17x32): segment boundaries at multiples of 32
  blocks (960 values) land on odd factors 5/17, good. Rotate each wave's K start
  offset by `wave*7` blocks (odd rotation) to spread HBM channel pressure.

## 8. CPU seal (DONE - `seals/seal_i8split.cc`, 16 invariants PASS)

Single-file C++ (g++ -O2 -std=c++17, no deps), CPU-only. Emulates `v_dot8_i32_i4`
/ `v_dot4_i32_i8` with the tree's own portable fallback semantics
(`common.cuh:736-738, 763-772`), re-implements `quantize_row_q4_0_ref`
(ggml-quants.c:113) and `quantize_row_q8_1_ref` (ggml-quants.c:302) including the
f16 scale storage, and checks:

1. **Per-regime coefficient asserts** (declared expected coefficients vs realized,
   vacuity-guarded - SUMW != 0 required for coefficient regimes, degenerate regimes
   declare zero coefficients and the zero is asserted): const-0, const-127,
   const-(-128), const-8, const-(-8), const-1 (the `>>4` sign trap), const-(-1),
   varying-lows, varying-mixed.
2. **Per-nibble-lane sign sweep:** 32 lanes x a_hi in [-8,7], asserting
   `D_hi = w_L*h`, `D_lo = -8*SUMW`, `S = w_L*16h` per lane.
3. **Pairing check:** canonical {i, i+16} exact vs direct; the WRONG pairing
   {2k, 2k+1} is distinguished on 64/64 randomized blocks (a pairing regression
   cannot pass silently).
4. **Adversarial scale boundary:** stock q8_1 never emits -128 from real data
   (proved over 256 edge inputs) - the harness exercises -128/+127 mixes against
   adversarial weight packs (all-codes-0 = w -8, all-codes-15 = w +7, alternating,
   single-lane) anyway; exact vs direct and stock.
5. **Fuzz:** 200k random blocks through the real quantizers:
   `S_i8split == S_stock == direct` on every block (the integer-identity claim).
6. **fp32 epilogue:** 2000 rows x 160 blocks (K=5120), fold within 1e-3 relative
   of a double dequant reference (observed 2.3e-4 - fp32 fold class, same as stock).
7. **int32 headroom:** worst |S| observed 7860 vs bound 32512.

## 9. Shapes to serve and GPU pass/fail bar (GPU-QUEUED - see ledger)

Shapes (Qwen3-14B-class FFN geometry): `(ne00=5120, ne01=17408)` gate/up and
`(ne00=17408, ne01=5120)` down; widths **1, 2, 4, 8, 16** (16 needs the dispatch
window change of §2).

Microbench bar (later, idle card under gpu.lock; llama-bench is DIAGNOSTIC-ONLY,
authoritative claims go through the harness/server protocol):
- **Primary: 8-token kernel time <= 1.5x the 1-token kernel time** (same kernel,
  same shape) - the narrow-batch scaling claim.
- Parity bar: >= 95% of the current narrow stock nc kernel (`mul_mat_vec_repacked_nc`
  q8_1 path) at widths 1-8 on the same card; beat it at widths 4-16 to justify the
  new kernel.
- Cross-reference: ruquant `w4a4_skinny` on the same card (facts in
  `~/workspace/ruquant-bench/RESULTS-h2h.md`; h2h method per the foreign-fork notes).
- ISA gates before timing: 0 VGPR spills, <= 64 VGPR at T <= 8, dot8 count per
  hot loop == 8*T (llvm-objdump census method, docs/w4a8-census.md:6-12).
- Correctness gate before any perf claim: device integer partials bit-match the
  host emulation (extend seal harness with a device dump), then greedy seal via
  the harness/server protocol.

## 10. Integration sketch (post-GPU-validation)

- New env gate `GGML_CUDA_I8NARROW=1` (default off => byte-identical dispatch),
  narrow-window arm in `mul_mat.cu` ahead of the existing nc dispatch for
  `src0->type == GGML_TYPE_Q4_0`; activation quantizer call mirrors
  `quantize_mmq_q8_1_u4_cuda` (`mul-mat.cu:372-382`) with the split-plane alloc
  math (`hi/lo: ne13*ne12*ne11*x_stride` 16 B blocks; `d`: same count x 2 B).
- Width-16 window: raise `MMQ_RP_Q8_MMV_MAX_TOKENS` or gate a separate window
  `ne11 <= 16` for i8narrow only; T=1 numerics untouched unless A/B says otherwise.
- Files: `ggml/src/ggml-cuda/q8_repack/w4a4-i8narrow.cuh` (layouts, static asserts,
  SIZEOF_PROBE idiom), `w4a4-i8narrow.cu` (kernel + quantizer + launchers).
  AUTHORED, NOT COMPILED - pause point (no cmake/make on this box while RAM < gate).
