# E4 GEMM Restyle — Implementation Plan (item #5, scoped 2026-11-12)

## What the item actually is
Wire-and-replace the GEMM half of the E4 RESTYLE so the repacked Q8_0 prefill GEMM is
bit-identical to stock on every shape the gcn_kq yield declines (9..31-token prefill;
N>=32 with M<128 or K%256!=0). That un-gates REPACK_Q8_0=2 + STOCK_ACC=0 default-ON and
the measured +1.3-1.5% MoE decode win. The mat-vec half already ships bit-identical
(widths 2-5); long prefill is already neutralized by the gcn_kq yield.

## The rejected RESTYLE branch (why it was a misfire)
- What it did: reordered the 8 dp4a of a 32-value block into stock int order
  (repack-kernels.cuh:1981-1994), matching vec_dot_q8_0_q8_1_impl.
- Rejected grounds: (1) never wired — dispatch sites hardcode RESTYLE=false
  (mul-mat.cu:1039, :1149); the section-92 MoE-conditional default was not bit-safe on
  yield-declined shapes, STOCK_ACC was restored ON. (2) Changes zero bytes even if wired:
  int32 dp4a chain reorder is order-invariant inside an exact int32 accumulator.
- The mat-vec half WAS adopted and ships default-ON (bit-identical widths 2-5).

## Ground truth BEFORE any kernel code (mandatory step 0)
The fork divergence map claims "stock: K split across tile lanes + shared reduce" — but the
dp4a MMQ path gfx906 actually runs uses a single per-thread fp32 register accumulator with
serial ascending K and NO shared reduce (mmq.cuh:912; the shared-reduce path is stream-K/MMA,
not selected on gfx906). Both kernels form one fp32 term per 32-value block in ascending K.
True mechanism candidates: FMA contraction of acc[r][n] += d*dx[n]*(float)idot
(repack-kernels.cuh:2010) vs stock's emitted mul/add sequence, or scale-staging rounding
(stock stages float scales in MMQ tiles; repacked converts half->float at compute).
- (a) standalone microbench: identical q8_1 activation bytes + identical weights through
  stock MMQ vs repacked GEMM at a diverging shape; per-output fp32 diff.
- (b) ISA forensics of both epilogues with /opt/rocm/llvm/bin/llvm-objdump (gnu objdump
  emits empty AMDGPU disasm silently).

## The restyle (step 1, repack-kernels.cuh)
Replace the dead int32-reorder branch (:1981-1994) with the real fp32-geometry fix in the
Q8_0 compute stage (:1905-2010): per-32-block term = (d_x*d_w)*(float)sumi with explicit
non-contracted intrinsics (__fmul_rn/__fadd_rn) reproducing stock's exact emitted sequence,
accumulated ascending-K; if ground truth shows grouping differences, reproduce stock's
grouping (per-tile partial + tile-sum, or KSPLIT partial-plane machinery :993-1010/:2737
with matching slice boundaries/reduce order). dp4a unchanged (int32 exact). Cover BOTH
instantiations (64-wide :2112, w32 :2714) and BOTH dense and HAS_IDS (expert mul_mat_id at
9..31 routed tokens — same impl, un-proven today; section-92's 6/6 MoE gate did not exercise
them).

## Wiring (step 2, mul-mat.cu)
Select RESTYLE=true at :1039 and :1149 under rp_q8_0_restyle_on() (default ON); update gate
banner (drop "GEMM restyle not wired"), README, repack-common divergence map. STOCK_ACC stays
default-ON as rollback until the full gate battery passes; the admission flip is a separate
ship commit.

## Optional step 3 (same file, cheap)
nc verify widths 6-8 warp-geometry fix: instantiate repacked nc at 1 warp for ncols_dst 6-8
to match stock mmvq — otherwise full default-ON stays blocked for spec-verify shapes
(width 8 = dflash).

## Explicitly out of scope
i8narrow reuse (closed negative — stock nc kernels win every width <=8); layer-split direct
full-tensor repack path (separate corruption class — tensor-split-only admission must
survive); re-pinning gcn_kq seal baselines.

## Risks
1. Misdiagnosis: the documented mechanism wording does not match the code path — step 0 may
   pivot the design (FMA contraction or scale staging rather than K grouping). Budgeted.
2. FMA contraction is compiler-dependent (TheRock 7.15 clang) — pin with __fmul_rn/__fadd_rn,
   re-verify bit-identity after any toolchain bump.
3. HAS_IDS expert GEMMs un-proven; if missed, MoE un-gating is not actually bit-safe.
   Canonical-copy yield is not an option for experts (VRAM).
4. Perf erosion: non-contracted FP ops may cost the repacked GEMM its ~2.4x class on
   9..127-token shapes — mandatory prefill confirmatory leg; yielding long prefill costs ~1% pp.
5. Full default-ON needs THREE closures (GEMM restyle + nc 6-8 + tensor-split-only) — do not
   conflate in the ship decision.
6. Dual-layout VRAM: canonical copy adds ~1x dense Q8_0 bytes (+2.6-9.7 GB); dense 27B stays
   excluded via =2 shape hint + MIN_DENSE floor.
7. Template cross-product build time — prune to served shapes.
8. First-model-wins latch: keep per-model/per-tensor gate reads (process-global statics broke
   multi-model load order before).

## Validation plan (all legs: mi50-doctor preflight, gpu.lock held, banner witnessed,
TERM teardown PID-verified, interleaved median-of-3, never llama-bench for reported numbers)
1. Microbench bit-match: stock MMQ vs restyled repacked GEMM, identical weights+activations;
   N in {9,16,31,32,33,64,127,128,256}, K in {1024,2048,4096,5120,6144}, M in {2048,4096,5120},
   dense + HAS_IDS scatter; per-output fp32 bit-identical every shape + launch-deterministic.
2. Ground-truth dump before AND after the kernel change — prove the sequences match.
3. Greedy bit-id probes (temp-0 seed-42, sha256): prompts engineered for 9..31-token prefill
   GEMMs + standard suite, on 35B-A3B MoE Q8_0 TP2 and UD-Q4_K_XL TP2,
   REPACK_Q8_0=1 STOCK_ACC=0 gate-ACTIVE vs stock control — must be bit-identical.
4. Seal suites: harness-adopt pinned baselines (ud-tp1 6/6 incl long-pp-r03 f2d63e2937a4) gate-ON.
5. KLD ladder stock vs restyled: expect 0.000000; red lines mean<=0.0125/max<=0.065 are the
   outer bound, not the acceptance bar.
6. Width coverage 2..8 greedy unchanged (6-8 only if step 3 folded in).
7. Perf A/B: interleaved median-of-3 tg256 + pp4096 (+pp2048) on 35B-A3B MoE Q8_0 TP2 and
   UD-Q4_K_XL TP2 (fleet flags, MTP per convention); acceptance: +1.3..1.5% tg class preserved,
   pp within noise; dense 27B control confirms shape-hint keeps dense stock.
8. VRAM/doctor leg: dual-layout footprint fits 32 GB cards on served MoE configs.
Ship gate: all green -> flip STOCK_ACC default + README in a separate commit. Rollback = STOCK_ACC=1.

## Effort
~2.5-3.5 engineering weeks: ground-truth dump + ISA forensics 2-3 d; kernel restyle across
2 widths x dense/HAS_IDS 4-7 d; wiring + gate/banner/docs 1 d; microbench bring-up 2-3 d;
full gate battery 3-4 d of GPU legs (gpu.lock-serialized). Optional nc 6-8 fold-in: +1-2 d.
GPU legs unavoidable (bit-identity is a runtime property); scoping needs none.

## Step 0 verdict (2026-11-11, task E1): the restyle premise is REFUTED
Microbench + ISA forensics (see taskE1-LEDGER.md, docs/E4-step0-results.csv,
docs/E4-step0-isa-evidence.txt): stock MMQ and the repacked Q8_0 GEMM are BIT-IDENTICAL
on identical input bytes at 171/171 grid shapes (incl. M<128 fallback and K%256!=0
classes), both launch-deterministic, and both bit-identical to an independent ascending-K
fmaf CPU reference. ISA: both emit v_cvt_f32_i32 -> v_mul_f32(d_w,d_x) -> v_fmac_f32 with
8:1 dp4a:fp32 ratio and zero cross-lane ops; stock stores its per-thread fp32 accumulator
registers directly. The maps "stock K-split + shared reduce" mechanism does not exist in
the dp4a path. Step 1 (epilogue restyle) is unnecessary for bit-identity; re-scope to the
byte-generation checks listed in the ledger (GPU-vs-host repack equivalence, repack vs
gcn_kq family, HAS_IDS).

## Step 1a verdict (2026-11-11, task E4-bytegen): GPU-vs-host repack bytes are EQUIVALENT
Microbench mb_e4_bytegen_test.cu (see taskE4-bytegen-LEDGER.md,
docs/E4-step1a-bytegen-results.csv): repack_q8_0_kernel (GPU, runtime async path) and
repack_q8_0_host (used by E1 microbench) produce BYTE-IDENTICAL records at 26/26 shapes
(E1 dense grid, MoE 64-expert incl. the 65535-grid-cap shape, no-gap K, tiny edges),
launch-deterministic, FNV-equal records. Sole difference: the 16 B de-alias stride gap
(ne0%128==0) is memset-zeroed by the host but never written by the kernel (runtime leaves
it uninitialized); consumer kernels guard sb < n_sub so gap bytes are never read --
cosmetic, comment at buffer.cu:136 should be fixed. The item-#5 defect is NOT in
GPU-vs-host byte generation; remaining suspects: gcn_kq-baseline identity, HAS_IDS.
