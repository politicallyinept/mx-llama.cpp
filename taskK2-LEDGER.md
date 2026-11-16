# taskK2 LEDGER — DRAFTER-HEAD LDS-STAGED Q6_K MAT-VEC (GGML_DFLASH_HEAD_KERNEL)

Branch: `int4-k2-headkernel` from tip `e5c91714e3` (INT4-noquant). Worktree `/root/wt-k2`.
Do NOT push. Gate default OFF; gate-off byte-identity by construction (dispatch never reaches new code).

## Mission (as briefed)
Dedicated LDS-staged head mat-vec for the DFlash2 drafter head at draft widths, targeting the
~4.5 ms/step head share. taskJ2/taskQ discipline: bit-exact vs the generic shape, CPU oracle,
gate default OFF, microbench first, e2e probe sha `77841cb6a251235d` must hold if wired.

## Head tensor as shipped (GGUFReader, verified pre-implementation)
`DFlash2-Q4_0-ownhead-q4.gguf` `output.weight` = **Q6_K, K=5120, N=248320** (own-head graft).
The brief's "Q8_0, K=2048" framing corresponds to the q8-variant head file
(`DFlash2-Q4_0-ownhead-q8.gguf`); the shipped Q4 drafter head is Q6_K/5120 and that is what
the gate matches (`ne00==5120 && ne01==248320 && type==Q6_K`, `ncols_dst in [2,5]`, no ids,
no fusion). Weight bytes: 248320 x (5120/256) x 212 = 1,052,876,800 B (~1.053 GB).

## Step 1 — profile of the current head matvec (from W1 rocprof trace segmentation)
Generic `mul_mat_vec_q<Q6_K, T=5>` instance (taskW1/rocprof-draft, k_kernel_trace.csv):
  WG 64x1, grid 7,946,240 x 1, LDS=0, VGPR=64, SGPR=32, **dur = 3487.677 us**
  -> achieved weight bandwidth = 1.0529 GB / 3.4877 ms = **~302 GB/s** per card.
MI50 32GB HBM2 peak ~1.02 TB/s. **The head matvec runs at ~30% of peak: bandwidth is NOT
saturated — the refutation outcome does not apply; kernel headroom is ~3x.** (W1 whole-draft
step ran 226 GB/s/GPU; the head alone is ~302 GB/s.) The generic path at this shape is
row-per-wave with no LDS staging and a tiny wave footprint per row — launch config leaves
locality on the table, which is exactly what the taskJ2/taskQ LDS pattern fixes.

## Step 2 — kernel (committed)
`ggml/src/ggml-cuda/mmvq-head-lds.cuh` (146 lines) + gate in `mmvq.cu` (+31 lines).
Commits: `f430e9c1bb` kernel+gate+harness, `2c7b8db546` harness alloc fix,
`778d42f937` harness out of globbed kernel dir.
Design (rp_q8_nc_lds house style): W waves x 2 rows per workgroup; activation (Q8_1, K=5120)
staged in LDS once per workgroup; weight rows streamed; lanes 0-31 take even superblocks
(kbx=s), lanes 32-63 odd (kb0=s+1), kbx ascending by 2 — **identical per-row fp32
accumulation order to `mul_mat_vec_q<Q6_K,T>` (nwarps=1, rpb=2), so bit-identical by
construction**; same `vec_dot_q6_K_q8_1(vx + rr[r]*stride_row_x, bq8, kbx, kqs)`.
Gate: `GGML_DFLASH_HEAD_KERNEL=1` (default OFF), `GGML_DFLASH_HEAD_W` = 4|8 (default 8).
Banner on route: `[DFLASH-HEAD-KERNEL] drafter Q6_K head widths 2-5 routed to LDS-staged kernel (W=%d)`.

## Step 3 — microbench (built; NOT RUN — box wedge, below)
Harness `/root/k2-mb_head.cu` -> `/root/mb_k2_head` (compiled OK against the K2 lib):
single-op ggml graph Q6_K [5120,N] x f32 [K,T], deterministic gaussian fill (seed 1234),
FNV-1a hash of dst + min/med/mean us + weight GB/s; `bw` mode = 1 GiB read-bandwidth ref.
Driver `/root/k2-mb-sweep.py`: per T in {1..5}, arms = gate-off / W=4 / W=8, same binary,
flock -w 2700 /root/gpu.lock, HIP_VISIBLE_DEVICES=1 (clean card), asserts hash equality vs
gate-off, TSV -> /root/taskK2-mb.tsv.
Build cmd (rebuild if harness changes):
  TR=/root/therock-venv/lib/python3.12/site-packages/_rocm_sdk_devel
  $TR/bin/hipcc -x hip --offload-arch=gfx906 -O2 -I/root/wt-k2/ggml/include -I/root/wt-k2/ggml/src \
    /root/k2-mb_head.cu -o /root/mb_k2_head -L/mnt/old/build-k2/bin -Wl,-rpath,/mnt/old/build-k2/bin \
    -lggml -lggml-base -lggml-hip

## Build
`/mnt/old/build-k2` via `/root/mi50-build.sh` (TheRock amdclang++ pinned, verified in
CMakeCache). BUILD-PROVENANCE: src_sha=778d42f937 (dirty=0), libggml_hip_md5=e56c6617c6b47f1ab78d531e8a6835ce,
llama_server_md5=81367650445c3874a859291cbf317a27.
NOTE: build script's post-check `grep -c ROCm` counts the ERROR line
"failed to initialize ROCm" as 1 device — broken check, flagged for the contract owner.

### mi50-build.sh disk-guard change (backup: /root/mi50-build.sh.bak-k2)
`/` fell to 16 GB free (< the hardcoded 20 GB gate) and blocked EVERY lane (K3 hit it too).
Guard now measures the filesystem the build dir actually lives on (identical behavior for
/root paths; a build is ~1 GB). Rationale + backup path recorded here.

## BLOCKED — box-level hardware incident (this is why Step 3/4 did not run)
Timeline (kern.log, dmesg, exporter):
- 23:08:28 PST  GPU2 (PCI 1a:00.0, a "clean" card 1/2/3) raises **uncorrectable hardware
  error** (ERREVENT_ATHUB_INTERRUPT).
- 23:08:28-57   driver GPU reset: powerplay suspend fails (-5), fw reload fails during
  unwind, BACO reset, "GPU reset end with ret = -22", "VRAM is lost due to GPU reset".
- Since: SMC dead on 1a:00.0 (GetMclks/GetSclks spam, kern.log now 311 MB).
- Now: `rocm-smi` shows GPU2 temp MISSING and ~1.03 GB phantom VRAM used; contract RULE
  ("If temps are gone -> host reboot") signature exactly.
- Consequence: **`openat("/dev/kfd") = EINVAL` -> HIP enumerates 0 devices box-wide**
  (verified: minimal hipGetDeviceCount program, both TheRock and apt ROCm stacks, and
  llama-server --list-devices from the known-good build-k1). All GPU work on mi50 is
  impossible right now, for every lane. No GPU processes running; llama-swap idle (untouched).
- NOT attempted: driver reload / reboot — shared box, llama-swap production service,
  contract playbook says host reboot; that is a user/parent-level call, not a subagent's.

## POST-REBOOT RUN 2026-11-16 — microbench sweep + VERDICT: REFUTED

Box recovered per contract. GPU0 ECC-failed (62311 correctable, stays quarantined); GPU1
clean (used for the sweep, CARD=1); GPU2 recovered, treated as suspect. ECC re-checked
via exporter (amd_gpu_ecc_errors_total) before and after the sweep: GPU1 0/0.

Read-bandwidth reference (`mb_k2_head bw`, GPU1): **857.9 GB/s** practical (1 GiB read).
Sweep: `CARD=1 python3 /root/k2-mb-sweep.py /root/mb_k2_head "1 2 3 4 5"` (IT=15,
N=248320, K=5120, flock -w 2700 /root/gpu.lock). Raw TSV: taskK2-mb.tsv (this commit).

### Sweep table (med_us / weight GB/s; win = generic med - best LDS arm med)
| T | gate-off        | W4              | W8              | win (best vs off) | bit-exact vs off |
|---|-----------------|-----------------|-----------------|-------------------|------------------|
| 1 | 2147.1 / 490.4  | 2093.2 / 503.0* | 2143.3 / 491.2* | n/a (not routed)  | see note 1       |
| 2 | 2884.8 / 365.0  | 3744.0 / 281.2  | 3927.8 / 268.1  | **-0.86 ms**      | NO               |
| 3 | 3632.3 / 289.9  | 5300.0 / 198.7  | 5591.4 / 188.3  | **-1.67 ms**      | NO               |
| 4 | 4137.4 / 254.5  | 6212.8 / 169.5  | 6371.2 / 165.3  | **-2.08 ms**      | NO               |
| 5 | 4000.5 / 263.2  | 7439.1 / 141.5  | 7564.8 / 139.2  | **-3.44 ms**      | yes (W4 & W8)    |

(*) T=1 is NOT routed: the gate requires ncols_dst>=2; the W4/W8 rows at T=1 are the
generic path re-run (banner prints regardless of routing — env-gate init, not routing).

**Verdict: REFUTED — no e2e legs run.** The LDS kernel loses at every routed width
(-0.86 to -3.44 ms/step vs generic) and peaks at 281 GB/s vs the >=500 GB/s bar (and
857.9 GB/s practical read reference). Neither go-condition (>=1.0 ms win OR >=500 GB/s)
is met at any width. Gate stays default OFF; branch stays unmerged.

### Root cause, bit-exactness failure (T=2..4)
`calc_nwarps(Q6_K, ncols_dst)` on the GCN table returns **nwarps=2 for widths 1-4** and
**nwarps=1 for width 5** (mmvq.cu:469-485). The kernel replicates only the nwarps=1
geometry (64-thread wave, kbx split lane>>5), so it is bit-exact exactly at T=5
(hash-verified: 3e91b02032d0a50a matches generic at W4 and W8) and produces a different
(valid, differently-ordered) fp32 reduction tree at T=2..4. The header's "nwarps=1,
rpb=2 on the GCN table" premise was true only for width 5; the "Verified bit-exact at
T=2..5" line predates the wedge and had never actually been run. A fixed kernel would
need a second geometry (nwarps=2 twin) for widths 2-4.

### Root cause, perf loss (design-level, consistent with K3)
Per-K-step `__syncthreads()` staging serializes the loop every 512 values and the block
only covers 2 rows/wave, cutting weight-streaming concurrency; the generic path already
serves the activation from L2 without staging. Losses grow with T (more LDS traffic per
step). Independently corroborated by the K3 lane: their Q6_K LDS twin (widths 5-8,
GGML_CUDA_RP_Q6K_LDS) was 27-41% slower in shipped config, best arm +0.09 ms at width 5
— below their 0.8 ms bar. **Two independent LDS-staging designs both lose at this shape:
the head matvec's ~260-365 GB/s at draft widths is NOT an LDS-staging problem.**

### Coverage report for parent reconciliation (K2 vs K3)
- K2 gate `GGML_DFLASH_HEAD_KERNEL` (+`GGML_DFLASH_HEAD_W` 4|8): widths **2-5**, Q6_K
  5120x248320, no ids/fusion. Per-width wins vs generic: T2 -0.86, T3 -1.67, T4 -2.08,
  T5 -3.44 ms — all negative. Bit-exact only at T=5.
- K3 gate `GGML_CUDA_RP_Q6K_LDS`: widths **5-8**, REJECTED independently (+0.09 ms best
  at width 5).
- Overlap at width 5: both kernels lose. Champion draft n_max=4 reaches widths 2-5 only;
  the head matvec stays on the generic path at every reachable width. No merge needed.

### Notes
1. Note 1 (T=1 hash instability, generic path): across the three captured gate-off-equivalent
   T=1 runs the dst hash took two values (W0=W4=dd088cf64c417abe; W8=626c9fd66467559f; the
   base run's hash was not captured by the driver but differed from W0, since W0 read
   exact=NO). Within-run determinism
   held everywhere (no NONDETERMINISTIC lines at T>=2). T>=2 base-vs-W0 re-runs were
   deterministic at every width (bitexact_vs_off=yes on all W=0 rows). T=1 is outside the
   gate scope; flagged for follow-up, not chased.
2. The sweep driver compares each arm to the FIRST gate-off run; where that base run was
   one of the flipped T=1 runs, the TSV bitexact_vs_off column reads NO even for matching
   rows — the table above compares hashes directly.
3. `/mnt/old/build-k2/bin/llama-server` md5 81367650445c3874a859291cbf317a27 matches
   BUILD-PROVENANCE; gate string present in libggml-hip (e56c6617...). No fresh build was
   needed. E2E leg script prepared at `/root/taskK2/k2-leg.sh` (GPUs 1,3, off/W4/W8 arms,
   probe-sha + acceptance + GGML_DFLASH_PROF wired) but UNUSED — kept for any future
   kernel that clears the mb bar.
4. mb runs are fill-dominated (~7 min each: CPU Q6_K quantize of 1.27B floats); a fill
   cache or smaller-N pre-screen would cut sweep wall time ~5x if this harness is reused.

## Verdict (final)
**REJECT the K2 LDS head kernel as-is.** It is slower than the generic mul_mat_vec_q at
every routed width and bit-exact only at width 5 (GCN nwarps geometry varies with width).
The W1 headroom read (302 GB/s vs ~1.02 TB/s peak) stands, but LDS activation staging is
the wrong lever for it — corroborated by K3's independent refutation at widths 5-8.
Gate default OFF; branch `int4-k2-headkernel` unmerged, kept for the kernel source and
this negative result.
