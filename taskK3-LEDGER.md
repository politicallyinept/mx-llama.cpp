# taskK3 — Q6_K lm_head LDS TWIN (X1: target-side head coverage)

Branch `int4-k3-q6lds` from tip `e5c91714e3`. Target: `Qwen3.8-27B-Q4_0.gguf`
`output.weight` = Q6_K 5120x248320 (9.4% of verify-width weight bytes, 1.04 GB/verify
pass full / 0.52 GB per TP2 card, the largest single matvec, outside all LDS/repack
coverage). Champion context: f128q4kv-p-d4 58.20 tg; verify pass ~33-42 ms at width 5-8.

## Status

**REFUTED (2026-11-16, post-reboot).** Microbench complete: twin is BIT-EXACT at every
width/arm (96/96 arms bit=1 det=1, oracle 6/6 PASS) but the shipped gate config
(STEP=4, STAGE_W=true) is 27-41% SLOWER than generic at every width 5..8 and every Q6_K
shape. Best arm (act-staging only, W4 STEP=2 STAGE_W=false) wins at every width but peaks
at +0.87 ms/verify only at T=8, which the champion (draft n_max 4 => verify width <= 5)
never reaches; at the champion's max reachable width 5 it saves +0.09 ms. Go/no-go bar
(>= 0.8 ms/verify at champion verify widths) NOT met => no e2e legs, no KLD. Gate stays
default OFF; branch not merged. See Results.

## Design

`ggml/src/ggml-cuda/q6k-lds.cuh` — `rp_q6k_nc_lds<T, W, STEP, STAGE_W, RPL=2>`:
- Bit-exact twin of `mul_mat_vec_q<GGML_TYPE_Q6_K, T, false>` GCN geometry (widths 5..8:
  nwarps=1, rows_per_block=2, physical warp 64). Same per-lane `kbx = lane/32 + iter*2`,
  same `vec_dot_q6_K_q8_1(vbase, ybase + kbx*8, kbx, kqs)` calls, same accumulation order,
  same `warp_reduce_sum<64>` tree => bit-identical outputs (the claim the microbench seals).
- Win mechanism 1 (act staging): each step's q8_1 activation blocks staged in LDS once
  per workgroup (STEP=4 x 256-value blocks), shared by all W warps. The generic path
  re-reads every token's activation blocks from L2 once per row.
- Win mechanism 2 (STAGE_W, weight staging): Q6_K row chunks staged in LDS with coalesced
  4-byte word copies. The generic path reads ql/qh/scales/d through strided 2/4-byte
  accesses at a 210 B/256-value block stride (sector-hostile: odd blocks are 2-byte
  aligned so every tile load splits). LDS layout keeps the exact 210 B block layout so
  vec_dot indexes it identically.
- LDS budget: sq[T][STEP*8] q8_1 (max 9216 B) + sw[W][RPL][STEP*210/4] (max 13440 B)
  ~< 23 KB, well under the 64 KB gfx906 budget.
- Gate: `GGML_CUDA_RP_Q6K_LDS=<2|4|8>` = workgroup wave count W (mirrors
  GGML_CUDA_RP_Q4_NC_LDS semantics), default 0 = OFF. Fires only for Q6_K, plain 2D
  (nchannels=nsamples=1), widths 5..8, ne0%512==0, even row stride, no ids/fusion.
  Hooked at the single choke point `mul_mat_vec_q_switch_type` so it covers both the
  single-GPU and the TP-per-device (`ggml_cuda_op_mul_mat_vec_q`) dispatch paths.
  Gate-off: `rp_q6k_lds_try_launch` returns false with no side effects; generic dispatch
  untouched => byte-identity by construction.
- Bug fixed pre-measurement: initial staging copied `BS/4 = 52` words per block, dropping
  the trailing 2-byte `d` half of every staged Q6_K block (commit cd5bb1968e).

## Microbench harness

`ggml/src/ggml-cuda/q8_repack/rp_q6k_lds_test.cu` + `/root/k3-build-mb.sh` ->
`/root/mb_q6klds_test` (amdclang++ -x hip --offload-arch=gfx906 -ffp-contract=off;
generic kernel extracted VERBATIM from mmvq.cu at build time, range auto-located).
Bars: BIT (twin vs generic, widths 5..8, both STAGE_W arms, 8 configs: W2/W4/W8 S2 +
W4 S4, act-only + full), ORACLE (CPU mirror of the exact accumulation, widths 5+8,
512 rows), DETERM (every arm twice), PERF (median us/call + eff GB/s vs 1024 GB/s HBM
peak at attn_q 5120x6144, head/card 5120x124160 (TP2), head 5120x248320; T=1/2/4
generic-only reference rows, T=1 at serving default rpb=1).

## Results

### CPU LDS-staging seal (no GPU required) — PASS

`q8_repack/k3-stagecheck.cpp` (g++ -O2 -ffp-contract=off): simulates the kernel's
LDS copy loops, kbx_loc indexing, per-lane accumulation order and reduction tree
in plain host arrays and compares bit-exactly against the generic-path CPU oracle.
**768/768 comparisons PASS** (T=8, K=5120, 64 rows, both rows of each staged pair).
This seals the staging layout/indexing (the class of bug the pre-measurement
`d`-half drop was) independent of the GPU. Commit 6405f6a097.
Note: the sim initially disagreed due to a wrong `QI8_1` in the standalone file
(16 vs the real 8 = QK8_1/(4*QR8_1)); the kernel and mb harness use the real
header and were never affected.

### GPU legs — microbench (`/root/mb_q6klds_test`, full sweep, 2026-11-16, GPU1 under flock)

Provenance: binary built 00:24 post-cd5bb1968e; libggml-hip md5 00e8ca428608027a70bc0d08d9067b52
(gate strings present); raw logs `taskK3-mb-quick.log` / `taskK3-mb-full.log` in this dir.
BIT: 96/96 arms bit-exact vs generic at widths 5..8, det=1 everywhere. ORACLE: 6/6 PASS.

**head/card 5120x124160 (TP2 serving shape)**

| arm | T=5 | T=6 | T=7 | T=8 |
|---|---|---|---|---|
| generic us (GB/s) | 2251 (232) | 2909 (179) | 3348 (156) | 4099 (127) |
| W2 S2 act | -31.5% (-1.03 ms) | -22.4% (-0.84 ms) | -21.8% (-0.93 ms) | -39.8% (-2.72 ms) |
| W4 S2 act | +4.2% (+0.09 ms) | +12.1% (+0.31 ms) | +15.2% (+0.44 ms) | +27.0% (+0.87 ms) |
| W8 S2 act | -3.5% (-0.08 ms) | +1.8% (+0.05 ms) | +6.3% (+0.20 ms) | -5.7% (-0.25 ms) |
| W4 S4 act | -27.7% (-0.86 ms) | -13.4% (-0.45 ms) | -38.8% (-2.13 ms) | -17.4% (-0.87 ms) |
| W2 S2 full | -48.1% (-2.09 ms) | -40.5% (-1.98 ms) | -52.5% (-3.71 ms) | -48.8% (-3.91 ms) |
| W4 S2 full | -30.2% (-0.98 ms) | -12.4% (-0.41 ms) | -17.0% (-0.68 ms) | +8.7% (+0.33 ms) |
| W8 S2 full | -34.4% (-1.18 ms) | -18.0% (-0.64 ms) | -24.2% (-1.07 ms) | +1.7% (+0.07 ms) |
| W4 S4 full | -37.4% (-1.35 ms) | -26.8% (-1.07 ms) | -35.0% (-1.80 ms) | -26.8% (-1.50 ms) |

**head 5120x248320 (single-GPU full)**

| arm | T=5 | T=6 | T=7 | T=8 |
|---|---|---|---|---|
| generic us (GB/s) | 4545 (230) | 5844 (178) | 6672 (156) | 8167 (128) |
| W2 S2 act | -30.2% (-1.96 ms) | -21.3% (-1.59 ms) | -20.8% (-1.75 ms) | -39.9% (-5.42 ms) |
| W4 S2 act | +7.0% (+0.30 ms) | +13.0% (+0.67 ms) | +15.1% (+0.87 ms) | +27.3% (+1.75 ms) |
| W8 S2 act | -1.9% (-0.09 ms) | +2.8% (+0.16 ms) | +6.7% (+0.42 ms) | -5.6% (-0.49 ms) |
| W4 S4 act | -26.9% (-1.67 ms) | -12.8% (-0.86 ms) | -38.3% (-4.15 ms) | -17.5% (-1.73 ms) |
| W2 S2 full | -47.9% (-4.18 ms) | -40.2% (-3.92 ms) | -52.7% (-7.44 ms) | -48.6% (-7.73 ms) |
| W4 S2 full | -28.7% (-1.83 ms) | -12.0% (-0.80 ms) | -16.4% (-1.31 ms) | +9.0% (+0.67 ms) |
| W8 S2 full | -32.3% (-2.17 ms) | -17.3% (-1.22 ms) | -24.2% (-2.12 ms) | +0.9% (+0.07 ms) |
| W4 S4 full | -36.6% (-2.63 ms) | -26.2% (-2.08 ms) | -34.7% (-3.55 ms) | -26.9% (-3.00 ms) |

**attn_q 5120x6144**

| arm | T=5 | T=6 | T=7 | T=8 |
|---|---|---|---|---|
| generic us (GB/s) | 121 (213) | 175 (148) | 199 (130) | 250 (103) |
| W2 S2 act | -49.6% (-0.12 ms) | -23.0% (-0.05 ms) | -25.8% (-0.07 ms) | -37.8% (-0.15 ms) |
| W4 S2 act | +27.1% (+0.03 ms) | +33.0% (+0.04 ms) | +39.9% (+0.06 ms) | +70.6% (+0.10 ms) |
| W8 S2 act | +6.5% (+0.01 ms) | +19.9% (+0.03 ms) | +17.9% (+0.03 ms) | +14.1% (+0.03 ms) |
| W4 S4 act | -28.9% (-0.05 ms) | -4.6% (-0.01 ms) | -29.6% (-0.08 ms) | -13.6% (-0.04 ms) |
| W2 S2 full | -50.7% (-0.12 ms) | -37.5% (-0.10 ms) | -48.7% (-0.19 ms) | -43.3% (-0.19 ms) |
| W4 S2 full | -33.2% (-0.06 ms) | -6.3% (-0.01 ms) | -13.5% (-0.03 ms) | +22.0% (+0.04 ms) |
| W8 S2 full | -33.5% (-0.06 ms) | -17.4% (-0.04 ms) | -21.5% (-0.05 ms) | +5.4% (+0.01 ms) |
| W4 S4 full | -40.9% (-0.08 ms) | -24.9% (-0.06 ms) | -33.9% (-0.10 ms) | -22.7% (-0.07 ms) |

Arm names: `W{waves} S{STEP} act|full` (full = STAGE_W=true weight staging). The library
gate `rp_q6k_lds_try_launch` launches `<T,W,4,true,2>` = the **S4 full** arm — the worst
variant measured. The brief's premise (generic ~302 GB/s, restore coalescing on the 210 B
layout) is refuted: generic head/card runs 232 GB/s at T=5 falling to 127 GB/s at T=8, and
NO arm exceeds 249 GB/s at any width. Coalescing the weight reads via LDS (STAGE_W) is
actively harmful (-12% to -53%): the generic strided 2/4 B reads are already absorbed by
L1/L2; the staging pass serializes the workgroup and doubles LDS pressure. The only real
waste the twin recovers is the per-row activation re-read (act staging): W4 S2 act wins at
every width (+4.2% to +27.0% at head/card) but the win scales with T and never approaches
memory-bound.

### Verdict vs go/no-go bar (>= 0.8 ms/verify at champion verify widths)

Champion config runs `--spec-draft-n-max 4` => verify width <= 5; widths 6..8 never occur.
At the TP2 serving shape (head/card), per verify pass per card:

| width | shipped gate (S4 full) | best arm (W4 S2 act) |
|---|---|---|
| T=5 (champion max) | **-1.35 ms** | **+0.09 ms** |
| T=6 | -1.07 ms | +0.31 ms |
| T=7 | -1.80 ms | +0.44 ms |
| T=8 (unreachable) | -1.50 ms | +0.87 ms |

NOT MET at any champion-reachable width, even after re-gating to the best arm. => **REFUTED,
accepted outcome (step 3). No e2e legs run, no KLD, probe sha / acceptance / tg gates not
exercised.** Gate stays default OFF; branch stays unmerged.

### Width coverage (for parent reconciliation with K2)

- K3 gate fires: Q6_K, plain 2D (nchannels=nsamples=1), widths (ncols_dst) **5..8**,
  ne0%512==0, even row stride, no ids/fusion; single-GPU and TP-per-device dispatch paths.
- K2 sibling: Q6_K head kernel widths 2..5 => overlap at width 5 only; union 2..8.
- Champion (draft n_max 4) reaches widths <= 5 => K3's effective champion coverage is
  width 5 only; K2 owns the widths that actually matter.

### Follow-up candidates (not shipped, not measured e2e)

1. Re-gating the twin to STEP=2/STAGE_W=false (act-only) would make width 5..8 strictly
   non-negative (+0.09..+0.87 ms) — still below bar at champion widths; only worth
   revisiting if draft n_max is raised to >= 7.
2. The STAGE_W result is a transferable negative for K2's design: do not LDS-stage Q6_K
   weight tiles on gfx906 for this access pattern.

## Blocker log

- 2026-11-16 ~07:00: RESOLVED — host reboot cleared the wedge. Post-reboot: GPU0
  ECC-FAILED quarantined (62311 correctable, never device 0); GPU2 recovered; mb legs ran
  on GPU1 (ECC clean per :9400 metrics). e2e pair for any future legs: 1,3.
- 2026-11-16 ~00:05: GPU 0000:1a:00.0 (rocm-smi Device 2, kfd node 3) SMC died —
  `dmesg`: 949+ `amdgpu: failed to send message: 0x2a ... response: 0xfc` (all on
  1a:00.0), card shows Temp/SCLK/PwrCap N/A + 1.03 GB ghost VRAM. `hipGetDeviceCount`
  => rc 100 (hipErrorNoDevice), count 0 even with TheRock runtime; one dead agent
  poisons whole HSA enumeration => zero usable GPUs box-wide. This is the recorded
  SMC-wedge family; recovery is out-of-container (host reboot; container restart may
  not clear a host-driver-level wedge). Health poller: /root/k3-gpu-health.log.
- Disk: / at 98% (16G free) — mi50-build.sh gate needs >=20G; build dir moved to
  /mnt/old/build-k3-q6lds (149G free). Full-tree build of tip+K3: BUILD-RC=0,
  gate strings verified present in bin/libggml-hip.so.0.26.0 (md5 00e8ca428608027a70bc0d08d9067b52),
  log /root/build-k3-q6lds.log.

## Files

- `ggml/src/ggml-cuda/q6k-lds.cuh` (kernel + gate) — commits dafd421c31, cd5bb1968e
- `ggml/src/ggml-cuda/mmvq.cu` (gate hook in mul_mat_vec_q_switch_type)
- `ggml/src/ggml-cuda/q8_repack/rp_q6k_lds_test.cu` (microbench harness)
- `/root/k3-build-mb.sh`, `/root/k3-drive.sh`, `/root/k3-gpu-health.sh` (ops scripts)
- `taskK3-mb-quick.log`, `taskK3-mb-full.log` (raw microbench output, 2026-11-16)
- `taskK3-LEDGER.md` (this file)
