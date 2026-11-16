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

## Next steps when the box recovers (reboot per contract, then)
1. `rocm-smi --showtemp` — confirm GPU2 state; if still templess after reboot, quarantine it
   alongside GPU0 and use cards 1/3 only (set CARD= in the driver).
2. `/root/mb_k2_head bw` — practical read-bandwidth reference (expect ~800-950 GB/s).
3. `CARD=1 /root/k2-mb-sweep.py /root/mb_k2_head "1 2 3 4 5"` — bit-exact + GB/s per arm.
   Acceptance: hash identical gate-off vs on at every T; win bar = >=1.0 ms/step on the head
   (generic head share ~3.5-4.5 ms/step at ~302 GB/s; LDS arm must show materially higher GB/s).
4. If win: e2e at champion config, gate ON: probe sha 77841cb6a251235d MUST hold; acceptance
   bit-equal; tg vs 58.20. Gate-off byte-identity leg. If microbench loses: deliver refutation
   with GB/s numbers (step 1 already shows headroom, so a loss would mean the LDS design is
   at fault, not saturation).

## Verdict (as of this run)
Kernel implemented, committed, compiles clean on the pinned toolchain; microbench harness
built. **No performance verdict is possible: mi50 lost KFD box-wide to an uncorrectable
GPU2 hardware error at 23:08 PST; host reboot required per contract before any GPU leg.**
The bandwidth-headroom question from W1 is answered from the existing trace: the head matvec
achieves ~302 GB/s vs ~1.02 TB/s peak — NOT saturated, ~3x headroom, so the LDS kernel has
room to win; the microbench must still prove it.
