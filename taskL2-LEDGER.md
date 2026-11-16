# taskL2 LEDGER -- grouped expert accumulation order: audit, numerics, perf ceiling

Branch `int4-l2-exporder` from `INT4-noquant` tip `e5c91714e3`. Worktree `/root/wt-l2-exporder`.
Claim under investigation (Discord): "grouped expert execution accumulates experts in a
different order from serial execution" -- and whether exploiting that reordering buys
spec-decode (DFlash2/MTP) performance on the 35B MoE (Qwen_Qwen3.6-35B-A3B-Q8_0,
qwen35moe: n_embd 2048, 256 experts, top-8, n_ff_exp 512).

## 1. AUDIT -- how expert accumulation is ordered in this tree

**The claim is structurally inapplicable to this fork: we already group, and the
cross-expert accumulation order is fixed by the graph, not by any kernel.**

Graph level (`src/llama-graph.cpp`, `build_ffn`, ~L2330-2620):
1. Router: `argsort_top_k` -> `selected_experts [k, n_tokens]`, softmax weights (norm_w).
2. Per-expert matmuls: `ggml_mul_mat_id` (gate_up / down) write each (token, slot)
   contribution into its OWN slice of a 3D buffer `experts [n_embd, k, n_tokens]`.
3. `experts = ggml_mul(experts, weights)` (per-slot scale).
4. Cross-expert sum: **left fold of `ggml_add` over slots 0..k-1**
   (`moe_out = cur_experts[0]; for i in 1..k-1: moe_out = add(moe_out, cur_experts[i])`).
   The order is fixed by graph topology (top-k slot order). No kernel participates in
   cross-expert accumulation at all.

Kernel level -- every MUL_MAT_ID kernel family writes each output slice exactly once
(single scatter store, no beta-accumulation, no atomics):
- `mul_mat_vec_q_moe` (`mmvq.cu:889`, fork's verify-width MoE matvec, serves the 35B
  canonical Q8_0 experts at ncols_dst>1 since `GGML_CUDA_MOE_CANONICAL` defaults ON for
  non-Q4_0): grid.y = all (slot, token) pairs; store
  `dst[channel_dst*stride + token*stride_col + row] = result` -- one write per element.
- repack HAS_IDS mmv `mul_mat_vec_q8_0_repacked<...,HAS_IDS,...>`
  (`q8_repack/repack-kernels.cuh:181`): `y += a*dst_s1` per block, single write.
- repack HAS_IDS GEMM `mmq_gemm_repacked_impl<HAS_IDS>` (`repack-kernels.cuh:934`):
  tiles partitioned per expert via `tile_meta`/`expert_bounds`; epilogue scatters
  `dst = y + ids_dst[a]*dst_s1 + rbase` (L2080) -- one write per element.
- stock mmvq HAS_IDS path: same single-store form.

Grouping itself (`ggml/src/ggml-cuda/mmid.cu`): `mm_ids_helper` (or the fork's parallel
counting sort `mm_ids_count/offsets/scatter`, `GGML_MM_IDS_NO_SORT=0` default ON) sorts
token slots by expert into a compact src1; `ids_dst` scatters results back. Grouping
only decides **which token slots share a kernel launch/block** -- it cannot change any
summation order:
- K-dim reduction order inside one expert's matmul: fixed by the kernel's k-loop
  (unchanged by grouping).
- Cross-expert order: the `ggml_add` fold (unchanged by grouping).

So "grouped vs serial" is not an accumulation-order knob in this tree. The only way to
change cross-expert order is to change the fold (graph-level), e.g. a fused weighted-sum
kernel that folds in expert-id order instead of slot order.

## 2. NUMERICS -- what a grouped-order fold would cost (if ever fused)

Harness `pocs/l2-exporder-num_order.py` (numpy fp32, IEEE-exact adds; run on mi50):
shapes n_embd=2048, k=8, verify width W in {1,5,8}; expert outputs ~N(0,sigma) sweep,
weights = softmax(randn(8)) (norm_w semantics); A = current slot-order fold (mul then
left-fold adds, exactly the graph), B = fold in ascending expert-id order (grouped),
C = pairwise tree, D = fp64 truth, E = fused-kernel FMA variant (fma(w,s,acc)).

max abs delta (max|A| in last col for scale):

| sigma  | W | max|A-B| | max|A-C| | max|A-D| | max|A-E| | max|A| |
|--------|---|----------|----------|----------|----------|--------|
| 0.0316 | 1 | 3.7e-09  | 3.7e-09  | 4.4e-09  | 5.6e-09  | 0.04   |
| 0.316  | 5 | 6.0e-08  | 1.2e-07  | 6.1e-08  | 7.5e-08  | 0.58   |
| 3.16   | 5 | 9.5e-07  | 4.8e-07  | 5.4e-07  | 4.8e-07  | 6.71   |
| 31.6   | 8 | 7.6e-06  | 7.6e-06  | 7.4e-06  | 7.6e-06  | 76.4   |

Verdict: the grouped-vs-serial fold delta is ~1e-7..1e-6 abs at realistic activation
magnitudes (relative ~1e-7) -- **two orders of magnitude inside the WSPLIT envelope
(1.14e-4, accepted with KLD seal)**. If a fused grouped-order epilogue were ever
adopted it is seal-class. Note the FMA trap: a fused kernel MUST use explicit
`__fmul_rn`/`__fadd_rn` (or `-ffp-contract=off` for the expression) to stay
bit-identical to the current mul-then-add chain; column A-E shows the FMA variant
lands in the same tiny class but is NOT byte-identical.

## 3. PERF HYPOTHESIS -- what a different order actually buys

(a) **Fuse the per-slot materialization + weighted sum** (the only real candidate):
    replace `mul + (k-1) adds` per layer with one kernel reading the k slices + weights
    and writing one output. Saves per layer: ~1.5-2 MB HBM round-trips (width 5) and
    k-1+1 kernel launches (k=8 -> 8 launches/layer x 41 layers ~ 330 launches/forward).
    The kernel structure FORECLOSES the stronger version (accumulate inside the down
    proj epilogue): the k contributions for one token are computed in k different
    expert-group blocks -- cross-block accumulation would need atomics (arrival-order =
    run-to-run nondeterminism, unsealable by greedy byte-identity). A standalone fused
    sum kernel in slot order is bit-identical (with FMA discipline) -- no seal needed.
    This is upstream's `GGML_OP_MUL_MOE` (added upstream after this fork's base; absent
    here -- `grep MUL_MOE ggml.h` empty, no `moe_sum` kernel in tree).
(b) Wave packing via token-grouped order: **already done** (mm_ids_helper / parallel
    counting sort). At decode width 1 grouping is a no-op (8 slots, <=1 token/expert);
    at verify width 5 (40 slots) it already merges shared-expert slots.
(c) Expert k+1 weight streaming overlap: already free -- one launch, blocks across
    experts run concurrently on the GPU scheduler; serial per-expert launches would be
    strictly worse and we do not have them.

Ceiling probe (this task): env `GGML_L2_SKIP_MOE_SUM=1` (default OFF) removes the
`mul + add` chain entirely (numerics WRONG by design -- drops slots 1..k-1; probe only,
never ship). Measures the upper bound of (a) on plain decode tg256 and DFlash2 d4.

## 4. A/B -- ceiling probe on the 35B MoE (GPU pair 1,3; production repack env)

Env: w4a4-env + GGML_CUDA_ALLREDUCE=internal GGML_CUDA_REPACK_KQUANT=0
GGML_GDN_COLTHREAD=1 + repack pair (REPACK_Q8_0=1 STOCK_ACC=0); HIP_VISIBLE_DEVICES=1,3;
flock /root/gpu.lock -w 2700; TERM-only teardown. Driver /root/taskL2-drive.sh,
outputs /root/taskL2/. Probe engagement witnessed in d4-skip server log:
"build_moe_ffn: GGML_L2_SKIP_MOE_SUM ACTIVE - MoE weighted-sum chain SKIPPED" +
/proc/pid/environ. (llama-bench suppresses llama logs, so plain-arm engagement is
attested by the identical arm_env export path (same function as the witnessed d4 arm)
plus the behavioral delta on the d4 arm; see sec 4b for the plain re-runs.)

| leg | metric | base | skip (ceiling) | delta |
|-----|--------|------|----------------|-------|
| d4  | spec tg t/s (median of 24) | 67.84 | 30.89 | acceptance collapse, see below |
| d4  | accept rate / mean_accept_len | 0.373 / 2.4919 | 0.036 / 1.1423 | (numerics broken by design) |
| d4  | **verify forward time** = mean_accept_len / t/s | **36.73 ms** | **36.98 ms** | **+0.7% (noise)** |
| plain | tg256 t/s run-1 | 115.99 +/- 1.46 | 108.54 +/- 0.69 | -6.4% (see sec 4b) |

d4 baseline reproduces the taskW2 sealed numbers (68.04-68.85 t/s, accept 0.373).
The skip arm's t/s collapse is the expected acceptance collapse (broken MoE output);
the honest ceiling metric is the derived verify-forward time: removing the ENTIRE
mul+add chain changes it by +0.7% -- i.e. **0% win, the chain is fully hidden behind
weight streaming at verify width 5**. (pp4096 median 1763 -> 1297 on the skip arm is a
probe artifact of the deliberately-broken numerics, not a property of a correct fused
kernel, which preserves values and buffer lifetimes.)

### 4b. plain-decode re-runs (llama-bench tg256, -r 3, interleaved arms x5)

Rig noise during the measurement window was large (base arm spread 101.9-117.6,
~15%; taskV2's 0.35-0.64% spreads were on an idle rig). Per-leg medians:

| pair | base t/s | skip t/s |
|------|----------|----------|
| run1 | 115.99   | 108.54   |
| run2 | 117.57   | 95.60    |
| run3 | 106.26   | 107.19   |
| run4 | 117.48   | 97.18    |
| run5 | 101.85   | 108.68   |

base median 115.99 / mean 111.8; skip median 107.19 / mean 103.4. The arms overlap
(the last two interleaved pairs have skip >= base); no leg shows the skip arm FASTER
by the serial ceiling, and the probe arm never shows a sustained win. Plain-decode A/B
is noise-dominated; the d4 forward-time result above is the decisive measurement.

### 4c. microbench (pocs/mb_moesum.hip, GPU3, under flock)

Serial cost of the chain vs one fused weighted-sum kernel, per layer:
W=1: chain (1 mul + 7 adds) 18.59 us, fused 3.92 us -> saving 14.67 us x41 = 0.60 ms/forward
W=5: chain 18.92 us, fused 3.92 us -> saving 15.00 us x41 = 0.62 ms/forward

Serial ceiling = 0.60 ms = 7.0% of the 8.6 ms plain decode step, 1.7% of the 36.7 ms
d4 verify forward. The A/B shows this ceiling is NOT reachable: the chain's kernels and
launches are hidden behind expert-weight streaming (same launch count at width 1 and
width 5 - MoE ops are per-slot, k=8 both - and the width-5 forward, which is MORE
launch-heavy, measured +0.7% when the chain was removed entirely).

## 5. VERDICT

**REFUTE the claim as inapplicable; REJECT grouped-order exploitation and the fused
weighted-sum epilogue.**
1. The order claim does not describe our kernel structure: we already group (expert-sorted
   compact mul_mat_id via mm_ids_helper / parallel counting sort), and no kernel
   accumulates across experts - every HAS_IDS kernel family (mul_mat_vec_q_moe, repack
   HAS_IDS mmv incl. the nc 6-8 fold variants, repack HAS_IDS GEMM, stock mmvq HAS_IDS)
   writes each output slice exactly once. The cross-expert order is forced by the graph's
   `ggml_add` left-fold over top-k slots (llama-graph.cpp build_ffn), independent of
   grouping. Code evidence in sec 1. The claim describes implementations where the expert
   loop accumulates into the output (atomic/beta style) - not this tree.
2. Grouped-order accumulation would be numerically sealable (<=1e-6 abs, two orders
   inside the WSPLIT 1.14e-4 envelope) but buys nothing by itself - order is not a perf
   knob here.
3. The only real optimization in the vicinity (candidate a: fuse mul+(k-1) adds into one
   weighted-sum kernel = upstream MUL_MOE port, bit-identical in slot order with
   __fmul_rn/__fadd_rn) has a serial ceiling of 0.6 ms/forward and a MEASURED ceiling of
   ~0% on the verify forward (d4: 36.73 -> 36.98 ms with the chain fully removed) and
   no measurable win on plain decode. Candidates (b) wave packing and (c) cross-expert
   streaming overlap are already provided by the existing grouped launch structure.
   The ggml op surgery is NOT justified. No prototype shipped beyond the ceiling probe.
4. Keep: `GGML_L2_SKIP_MOE_SUM` ceiling probe (default OFF, byte-identical when unset -
   the else-branch is the untouched original code path). It is a wrong-output probe;
   never enable in a sealed arm.


## Files
- branch `int4-l2-exporder`: `src/llama-graph.cpp` (GGML_L2_SKIP_MOE_SUM ceiling probe,
  default OFF), `pocs/l2-exporder-num_order.py` (numerics harness),
  `pocs/mb_moesum.hip` (chain-vs-fused microbench), this ledger.
- mi50: /root/taskL2-drive{,2,3}.sh, /root/taskL2/ (legs, witnesses, bench jsonl),
  /root/wt-l2-exporder/.
