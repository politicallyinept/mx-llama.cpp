# taskD1  -  VEC_DOT ALU REWRITE for the mmvq family (gfx906)

Branch `int4-d1-vecdot` from `int4-fix4` (e3dac6d835; code-identical to 468f0bbd84,
delta is ledger-only). Big-lane D1: replace the "scalar nibble-decode" in
`vec_dot_q6_K_q8_1` / `vec_dot_q4_0_q8_1` with PRMT/v_dot8-style vectorized decode,
bit-identical by construction, go bar >= 25% vs generic verbatim on BOTH the Q6_K
head shape (5120x248320) and the Q4_0 attn_q shape (5120x6144) at widths 1-8.

## Status

**REFUTED (2026-11-16).** The rewrite is bit-exact everywhere (CPU seal PASS;
32/32 shape x width combos, every arm bit=1 det=1; GPU-vs-CPU oracle 12/12 PASS)
but the go bar is not met: best bit-exact arm per width is **+2.7%..+21.5%** on the
Q6_K head and **+1.5%..+26.3%** on Q4_0 attn_q (bar: >= 25% at EVERY width on both).
No e2e legs, no KLD, no production routing. The lane premise  -  "the shortfall is
ALU: the scalar nibble-decode"  -  is refuted by ISA + roofline: the compiler already
emits `v_dot4_i32_i8` for every integer product (no scalar nibble loop exists in
this codebase; `v_dot8` does not exist on GCN), the decode+saturation lowering is
[CORRECTION R-review C1 (2026-11-24): THE CLAIM IN THIS PARAGRAPH IS FALSE. v_dot8 DOES exist on this ISA. It is present in this tree at ggml/src/ggml-cuda/common.cuh lines 755-772 (ggml_cuda_dp8a, __builtin_amdgcn_sdot8 -> v_dot8_i32_i4), guarded by `#if defined(GGML_USE_HIP) && defined(__gfx906__)` because LATER AMD ISAs dropped the 8-way int4 dot - the opposite of the original claim below. The original text is retained unstruck for the record; the strike is authoritative. Standing rule: grep common.cuh before asserting any instruction is absent on this arch. Full strike and consequence: docs/reviews/REVIEW-LEDGER.md, taskD1 CORRECTIONS C1.]
~24 of 239 kernel instructions at Q6_K T=1, and removing ALL of it yields <= 16%.
The mmvq family on gfx906 is bounded by **load latency + L1 activation re-read
traffic**, not decode ALU. See "Ceiling analysis".

## Design

`taskD1-mb/d1-mmvq.cuh`  -  bit-identical twins `mul_mat_vec_q_q6k_d1<T, nwarps, rpb,
D1_DECODE, UNROLL>` / `mul_mat_vec_q_q40_d1<T, nwarps, rpb, UNROLL>`:

- Geometry mirrors `mul_mat_vec_q` on the GCN table verbatim (review rule 2,
  mmvq.cu @ int4-fix4): `calc_nwarps(GCN)`: ncols 1..4 -> 2 warps, 5..8 -> 1 warp
  (Q8_0@5 -> 2); `calc_rows_per_block(GCN)`: ncols 1 -> 1 (2 with
  GGML_CUDA_MMVQ_GCN_RPB1=2 production lever), 2..8 -> 2; warp_size 64
  (`__GFX9__` -> `ggml_cuda_get_physical_warp_size()==64`, verified on the pinned
  TheRock clang). Q6_K: qi=QI6_K=32, vdr=1 -> kbx=tid/32, kqs=tid%32, stride
  2*nwarps. Q4_0: qi=QI4_0=4, vdr=2 -> kbx=tid/2, kqs=2*(tid%2), stride 32*nwarps.
- **Hoist** (D1_DECODE=0): weight nibble-decode hoisted out of the column loop
  (invariant of j), activation words + f16 scales hoisted out of the row loop
  (invariant of i). Every per-(j,i,kbx) integer product and the fp32 sequence
  (`sumf=0; sumf += d8[0]*(dot0*sc0); sumf += d8[1]*(dot1*sc1); d*sumf;
  tmp[j][i] += ...`) is preserved EXACTLY; same kbx chain, kqs split, reduce tree.
- **Raw decode** (D1_DECODE=1, Q6_K): drop `__vsubss4(vi, 0x20202020)` and correct
  in the integer domain: `dot(raw-32,u) == dot(raw,u) - (dp4a(0x01010101,u,0) << 5)`
  as EXACT integers (raw bytes in [0,63], no saturation, no overflow:
  |sum| <= 4*63*128). Also folds `((vh>>4)<<4)&0x30303030 -> vh&0x30303030`.
  Downstream fp sequence untouched => bit-identical.
- **UNROLL sweep**: `#pragma unroll U` on the kbx loop (generic has it commented
  out); pure scheduling change, per-lane kbx chain order unchanged.
- **roofline-read** diagnostic kernel: same geometry + same weight-load pattern,
  no activations, no dots  -  the weight-read ceiling of the access pattern.

## Verification (template, review rule 6)

1. **CPU simulation seal, independent of GPU** (`taskD1-mb/d1-cpu-seal.cpp`, plain
   C++, own f16 decode, -ffp-contract=off): S1 raw-decode identity 1,228,800 cases
   0 mismatches; S2 vih-fold identity 1e6 cases 0; S3 Q6_K twin==generic 64,000
   cases 0; S4 Q4_0 twin==generic 4,000 cases 0. **PASS** (run before any GPU leg).
2. **Verbatim generic baseline**: extracted from mmvq.cu at build time
   (d1-build-mb.sh, dynamic range, line-E guard), same TU, same pinned TheRock
   amdclang++ `-x hip --offload-arch=gfx906 -O3 -ffp-contract=off` both sides.
3. **GPU oracle**: generic kernel vs CPU mirror of the generic arithmetic, 512 rows
   x T at T=2,5,8, all 4 shapes: 12/12 PASS.
4. **Determinism**: every arm run twice, outputs byte-compared: det=1 everywhere.
5. **Bit-exactness**: every twin arm vs generic verbatim, full output vectors,
   32/32 shape x width combos x all arms: bit=1.

## Results (mb-full.log, GPU1, flock, w4a4-env; medians of 30)

roof = roofline-read (weight-only ceiling of the same access pattern).
best = best bit-exact arm at that width. raw = d1-raw (Q6_K) / d1-hoist u1 (Q4_0).

```
shape                    T  generic us   roof us  roof GB/s   best arm      best%   raw%
Q6_K attn_q  5120x6144   1       66.2      40.2      642     d1-raw-u2    +46.8  +43.7
                         2       85.8      77.3      334     d1-raw       +56.7  +56.7
                         3      101.1      42.7      604     d1-raw       +24.9  +24.9
                         4      118.4      42.7      604     d1-raw       +23.3  +23.3
                         5      120.6      98.1      263     d1-raw       +11.9  +11.9
                         6      173.9      71.7      360     d1-hoist-u2  +51.8   +4.5
                         7      198.1      65.8      392     d1-hoist-u2  +29.9   +1.1
                         8      248.6      80.8      319     d1-raw       +14.8  +14.8
Q6_K head 5120x248320    1     2469.0    1567.0      666     d1-raw-u2    +18.4  +16.1
                         2     3111.8    1579.0      661     d1-raw       +21.5  +21.5
                         3     3697.8    1611.4      647     d1-raw        +9.4   +9.4
                         4     4368.5    1587.7      657     d1-raw        +9.2   +9.2
                         5     4542.9    1625.8      642     d1-hoist      +2.7   +0.8
                         6     5821.9    1656.2      630     d1-hoist-u2   +8.0   -6.9
                         7     6656.5    1643.7      635     d1-hoist-u2   +6.3   -7.6
                         8     8152.5    1632.6      639     d1-hoist      +6.2   +2.3
Q4_0 attn_q 5120x6144    1       50.9      37.9      467     d1-hoist-u4  +16.9  +11.2
                         2       52.8      38.9      455     d1-hoist-u2   +8.2   +3.8
                         3       66.2      38.6      459     d1-hoist-u4  +22.5   +8.7
                         4       82.2      32.6      542     d1-hoist-u4  +26.3  +13.7
                         5       85.4      72.5      244     d1-hoist      +6.0   +6.0
                         6       96.5      38.6      459     d1-hoist-u4   +1.5   +0.2
                         7      109.1      70.1      253     d1-hoist-u2   +4.0   +3.5
                         8      124.0      73.3      242     d1-hoist-u2   +5.9   +1.7
Q4_0 ffn   5120x9216     1       69.6      42.4      626     d1-hoist-u4  +25.4   +9.8
                         ...    (T=2..8: +3.4..+30.3, full table in mb-full.log)
```

Run-to-run variance note: a prior quick run (15 iters, different box power state  -
GPU0 burn-in + sibling lanes share the power budget) gave head T=1 raw +7.8/+10.0%
vs full-run +16.1/+18.4%. Small-shape attn_q deltas swing widely (+21..+57%); the
head numbers are the stable, decision-relevant ones. No width on either target
shape reached +25% in either run.

## Ceiling analysis (why the family cannot hit 25% via decode ALU)

ISA (pinned compiler, --save-temps, per-kernel counts):
- **v_dot4_i32_i8 is already emitted for every integer product** in the generic
  kernel (2/4 per kbx at Q6_K T=1, scaling with T; 64 at Q4_0 T=8). gfx906 has
  dot4 (correctness proven by the CPU oracle at T=2,5,8); **v_dot8 does not exist
  on GCN**  -  the task's v_dot8 option is not on this ISA, and the PRMT-style
  [CORRECTION R-review C1 (2026-11-24): THE CLAIM IN THIS PARAGRAPH IS FALSE. v_dot8 DOES exist on this ISA. It is present in this tree at ggml/src/ggml-cuda/common.cuh lines 755-772 (ggml_cuda_dp8a, __builtin_amdgcn_sdot8 -> v_dot8_i32_i4), guarded by `#if defined(GGML_USE_HIP) && defined(__gfx906__)` because LATER AMD ISAs dropped the 8-way int4 dot - the opposite of the original claim below. The original text is retained unstruck for the record; the strike is authoritative. Standing rule: grep common.cuh before asserting any instruction is absent on this arch. Full strike and consequence: docs/reviews/REVIEW-LEDGER.md, taskD1 CORRECTIONS C1.]
  nibble unpack is already 2-4 VALU ops (and/shift/or), not a scalar loop.
- Generic Q6_K T=1 kernel: 239 instructions; the `__vsubss4` saturation lowering
  (v_add_u16 + v_add_co/v_addc + v_cmp_lt + v_cndmask x2 dots) is ~24; total
  decode+saturation ~80. The twin removes most of it (159 instr, -33%) yet gains
  only +7.8..+18.4% on the head => **not issue-bound**. `s_waitcnt` is the single
  largest op class (20/239): the waves stall on load latency.
- No spills anywhere (privseg=0, VGPR 22-64) => occupancy is not register-limited;
  unroll variants only thrash L1 (u2/u4 regress -10..-21% at T>=4).
- **Roofline-read** (same geometry, weights only): head sustains 630-666 GB/s
  (~73-78% of the 858 GB/s practical flat-read figure  -  the 210 B block / 5-load
  pattern is near its own ceiling). Generic sits at 34-63% of that roof; the gap
  is activation traffic + f16 cvt + fp accumulate, all MANDATORY for bit-exactness:
  at T=8 each block re-reads all T columns' q8_1 blocks (~14x weight bytes in L1
  traffic), which is why T=8 generic runs at 17.5% of the weight roof and no
  decode rewrite can touch it.
- => mmvq family ceiling on gfx906: DRAM-bound at T=1 (generic already at 63% of
  the access-pattern roof; residual gap is activation interleaving), L1
  activation-re-read-bound at T>=4. The 226-380 GB/s production figure is
  consistent with this, and LDS staging being refuted twice (K3) fits the same
  picture: the bytes are not the problem, the re-read pattern is.

## Partial-win candidates (below bar, bit-exact, NOT routed)

- `d1-raw` (Q6_K): head T=1 +16.1/+18.4% (full run), T=2 +21.5%, attn_q T=1-4
  +23..+57% (noisy); consistently positive at T<=4, ~0 at T>=5.
- `d1-raw-u2`: head T=1 best (+18.4%) but regresses at T>=6.
- `d1-hoist-u4` (Q4_0): attn_q T=1/T=3/T=4 +16.9/+22.5/+26.3%, ffn T=1 +25.4%;
  ~0..-24% at T>=5 with unroll.
All are bit-exact (no KLD seal needed per standing rule), but none clears the
25%-at-every-width bar, so per step 4 no e2e leg was run and nothing is routed.
If the parent wants a T<=2-only decode lever later, d1-raw is the candidate; it
would still need the e2e paired-interleaved leg before adoption.

## Repro

```
ssh mi50; worktree /root/wt-d1 (branch int4-d1-vecdot)
bash /root/wt-d1/taskD1-mb/d1-build-mb.sh /root/wt-d1   # CPU seal + build
source /root/w4a4-env.sh; HIP_VISIBLE_DEVICES=1 flock -w 2700 /root/gpu.lock -c \
  '/root/wt-d1/taskD1-mb/rp_d1_vecdot_test > mb-full.log'   # ~25 min
# quick: rp_d1_vecdot_test --quick [shape-substring]
```
Files: taskD1-mb/{d1-mmvq.cuh, rp_d1_vecdot_test.cu, d1-cpu-seal.cpp,
d1-build-mb.sh, mb-quick.log, mb-quick2.log, mb-full.log}. Logs also at
mi50:/root/wt-d1/taskD1-mb/.

## Verdict

D1 closes the lane: the mmvq family's shortfall vs the 858 GB/s flat-read figure is
structural (activation re-read traffic + load latency at 210 B/18 B block
granularity), not decode ALU. A bit-exact decode rewrite tops out around +18% on
the head at T=1 and ~0 at T>=5. Any future attempt must change the data layout or
the re-read pattern (repack/fattn-style), which is outside D1's bit-exact scope
and would carry full seal debt.
