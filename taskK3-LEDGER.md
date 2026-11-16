# taskK3 — Q6_K lm_head LDS TWIN (X1: target-side head coverage)

Branch `int4-k3-q6lds` from tip `e5c91714e3`. Target: `Qwen3.8-27B-Q4_0.gguf`
`output.weight` = Q6_K 5120x248320 (9.4% of verify-width weight bytes, 1.04 GB/verify
pass full / 0.52 GB per TP2 card, the largest single matvec, outside all LDS/repack
coverage). Champion context: f128q4kv-p-d4 58.20 tg; verify pass ~33-42 ms at width 5-8.

## Status

**BLOCKED on box-level GPU wedge (2026-11-16 ~00:05).** All code complete + compile-verified;
measurement legs cannot run until the box is recovered (see Blocker log).

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

PENDING — GPU wedge prevented every measurement leg. Planned table:

| shape | T | generic us | GB/s (% peak) | best LDS arm | us | GB/s | delta | bit |
|---|---|---|---|---|---|---|---|---|
| (fill from /root/mb_q6klds_test) | | | | | | | | |

e2e (only if mb win >= 0.8 ms/verify): `/root/k3-drive.sh` arms ctrl / q6lds8 / q6lds4
at champion config, probe sha 77841cb6a251235d, acceptance 0.4377/2.7508, tg vs 58.20,
narrow-window KLD (taskQ protocol: llama-perplexity -b 8, ctrl captures base logits,
gate-on arms scored vs ctrl).

## Blocker log

- 2026-11-16 ~00:05: GPU 0000:1a:00.0 (rocm-smi Device 2, kfd node 3) SMC died —
  `dmesg`: 949+ `amdgpu: failed to send message: 0x2a ... response: 0xfc` (all on
  1a:00.0), card shows Temp/SCLK/PwrCap N/A + 1.03 GB ghost VRAM. `hipGetDeviceCount`
  => rc 100 (hipErrorNoDevice), count 0 even with TheRock runtime; one dead agent
  poisons whole HSA enumeration => zero usable GPUs box-wide. This is the recorded
  SMC-wedge family; recovery is out-of-container (host reboot; container restart may
  not clear a host-driver-level wedge). Health poller: /root/k3-gpu-health.log.
- Disk: / at 98% (16G free) — mi50-build.sh gate needs >=20G; build dir moved to
  /mnt/old/build-k3-q6lds (149G free). Full-tree build of tip+K3 launched 00:22,
  log /root/build-k3-q6lds.log.

## Files

- `ggml/src/ggml-cuda/q6k-lds.cuh` (kernel + gate) — commits dafd421c31, cd5bb1968e
- `ggml/src/ggml-cuda/mmvq.cu` (gate hook in mul_mat_vec_q_switch_type)
- `ggml/src/ggml-cuda/q8_repack/rp_q6k_lds_test.cu` (microbench harness)
- `/root/k3-build-mb.sh`, `/root/k3-drive.sh`, `/root/k3-gpu-health.sh` (ops scripts)
- `taskK3-LEDGER.md` (this file)
