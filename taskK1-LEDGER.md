# taskK1 LEDGER — draft-head split: vocab-parallel output.head across draft-dev + head-dev

Date: 2026-11-15 (PST). Box: mi50. Branch `int4-k1-headsplit` off INT4-noquant tip e5c91714e3
(NOT pushed). Gate commit: 362542247a. Build: /root/build-k1 (BUILD-PROVENANCE:
src_sha=362542247a, dirty=0, llama-server md5 fe8143ef28839ddea8936dbc80b936c5,
libllama md5 in k1-*.witness). Canonical greedy probe sha16: 77841cb6a251235d.
Pair: GPUs 1,2 (rocm0 quarantined). NOTE: /root/build-uwave provenance is e9b3e3197f,
NOT the tip — tip adds taskP2 i8narrow + W1 instrumentation code; all K1 legs use build-k1.

## 0. Goal

Row-shard the DFlash2 drafter's output head matvec across the draft device AND the
second GPU (idle 9.26 ms/step during the draft slice, taskX3), targeting the head
share of the 9.26 ms draft device time. Env gate GGML_DFLASH_HEAD_SPLIT=<dev>,
default OFF, gate-off byte-identical, numerics-neutral (each logit row computed
exactly once by exactly one device, same K-dim summation order — no cross-device
fp32 reduction).

## 1. Head execution path (step 1)

- Head = `build_lora_mm(output, cur, output_s)` at src/models/dflash.cpp:815 —
  MUL_MAT, src0 = output.weight Q8_0 [2048, 248320] (ownhead drafter; 0.50 GiB),
  src1 = result_norm hs f32 [2048, 5]. Runs on the draft device (pinned -devd),
  INSIDE the single 690-node whole-draft HIP graph (taskW1: replay 3291/3421 = 96%,
  host launch 8-84 us). The DFlash2 selector (top_k -> get_rows -> score matmuls)
  consumes the logits in-graph; t_h_nextn (packed lattice) is the graph output the
  host walk reads via llama_get_embeddings_nextn (the sync point).
- Capture boundary at the head input: splitting the head off the graph would force
  two llama_process passes or a manual second graph + relocated selector — heavy
  surgery, capture-fragile. Chosen instead: keep ONE cgraph and let the fork's meta
  backend split it per-lane (per-lane subgraph HIP capture is proven on the
  borrowed 2-device draft path, taskR H3).
- Fork machinery already present: under SPLIT_MODE_TENSOR the loader routes
  output.weight to GGML_BACKEND_SPLIT_AXIS_1 (vocab-parallel, the deepseek4 head
  rule, llama-model.cpp ~921) and the meta backend GATHERS axis-0-split sources
  before whole-row ops incl. TOP_K/ARGMAX (ggml-backend-meta.cpp:867-884,
  handle_per_row:957). The DSpark ARGMAX-abort comment predates that gather.
  LLAMA_DRAFT_SPLIT=1 (whole-draft split) = the borrowed path = allreduce storm
  (11.5 vs 9.26 ms, taskR H3) — K1 is "split ONLY the head".

## 2. Design + implementation (steps 2-3) — commit 362542247a

GGML_DFLASH_HEAD_SPLIT=<dev> (unset or "0" = OFF). Two paired gates:
1. common/speculative.cpp common_base_params_to_speculative: when set and the draft
   is a single-device DFlash/DSpark sidecar, resolve <dev> (device name or bare
   GPU index), append it to the draft device list, force draft split_mode=TENSOR +
   tensor_parallel_size=2. The head device joins the draft TP group; hs replication
   to head-dev is the meta MIRRORED broadcast of result_norm (40 KB/step at width 5).
2. src/llama-model.cpp get_tensor_config (arch DFLASH): gate ON -> every tensor
   EXCEPT output.weight/bias returns MIRRORED (whole, lane-local -> no per-layer
   allreduce); output.weight falls through to the AXIS_1 vocab-parallel rule.
   Gate OFF -> existing LLAMA_DRAFT_SPLIT/dflash_dsv4_split logic untouched.
Numerics: vocab-parallel row split = each logit row once, same mmvq template (K and
ncols_dst unchanged), gather is memcpy -> byte-identity expected (e2e gate, step 5).
Graph capture: draft graph becomes 2-lane; per-lane subgraph capture unchanged
(borrowed-path precedent); no per-step re-capture expected — to be witnessed via
DFLASH-PROF cuda-graph mode histogram on the ON legs.
VRAM watch: MIRRORED replicates the drafter on both lanes (~2.65 GiB/card) + half
head; GPU2 also holds its target TP share — fits 32 GiB cards, witness via server log.

## 3. Microbench (step 4) — /root/taskK1/head_bench, GPUs 1,2 under gpu.lock

Head matvec alone (Q8_0 [2048,248320] x f32 [2048,5], fake data, 20 iters):

| arm                       | ms/iter |
|---------------------------|---------|
| full head on GPU1         | 1.427   |
| full head on GPU2         | 1.442   |
| half head on GPU1 alone   | 0.713   |
| half head on GPU2 alone   | 0.734   |
| sharded GPU1+GPU2 concur. | 0.789   |

- **The head matvec is ~1.43 ms/step, NOT the ~4.5 ms the W1 tensor-census estimate
  implied.** The census counted token_embd+output.weight (0.97 GiB) at the drafter's
  226 GB/s average; token_embd is gather-only and the Q8_0 head actually streams at
  ~378 GB/s (0.50 GiB / 1.43 ms). Head share of the 9.26 ms draft slice = ~15%, not 37%.
- Sharding physics: 1.43 -> 0.79 ms (-44%), sync/launch delta ~0.08 ms over the
  0.71 ms half. CEILING of K1 = ~0.64 ms/step = ~1.4% of the 44.8 ms step
  (~+0.8 t/s on 58.20) — BELOW the 2.9% tg256 noise floor; the draft-phase
  readnx timer is the deciding metric, not tg256.
- The whole-head-on-GPU2 variant (4.5 ms premise) is moot at 1.43 ms: row-split
  (0.79) beats whole-head-on-dev1 (1.44 concurrent with 7.8 ms of lane-0 work —
  lane 0 remains the critical path either way).

## 4. BLOCKER — box wedged before the e2e legs (step 5 UNTESTED)

The microbench's exploratory raw cross-device copy arm (ggml_backend_tensor_copy
between two CUDA backends in one process) crashed with "ROCm error: unspecified
launch failure" in ggml_backend_cuda_buffer_cpy_tensor. ~9 min later GPU2
(PCI 0000:1a:00.0) entered a continuous SMC fault loop (dmesg "Failed to send
message 0x2a/0x2b, response 0xfc" x949+, ongoing), GPU2 holds 1.03 GB ghost VRAM,
and HIP enumeration is dead BOX-WIDE on both runtimes (rocminfo 0 gfx906;
llama-server --list-devices = none on TheRock AND apt runtimes). All four e2e legs
(k1-off-r1/on-r1/on-r2/off-r2) died at LEG-LOADFAIL with zero GPU work — no
contamination, lock released, no orphan servers (pgrep -x llama-server empty).
Recovery per skill: container restart from the PVE host (`pct reboot`); if TP2
re-wedges on a fresh container, host reboot. Nothing in-container fixes it.
LESSON (now encoded in head_bench.cpp + skill): never issue raw cross-backend
tensor copies in a standalone mi50 process; measure staged get/set instead.

## 5. Status + re-run protocol after recovery

- [x] gate implemented + committed (362542247a), build-k1 ready (provenance clean)
- [x] microbench: physics supports the split (-0.64 ms/step ceiling)
- [x] gate-OFF byte-identity to 77841cb6a251235d on build-k1 (k1e-off-r1/r2: PASS, sec 7)
- [x] gate-ON: probe byte-identity PASS 4/4; acceptance bit-equal PASS r1 / MISS r2 (sec 7)
      + tg256 vs 58.20 + draft-phase before/after (readnx bucket, DFLASH-PROF)
      + cuda-graph mode histogram (no per-step re-capture) + GPU2 VRAM witness
- Re-run: `/root/taskK1/k1-leg.sh <tag> off 8461` then `<tag> ROCm1 8462` (paired
  x2 recommended; refuse-if-exists .done markers already set up; legs flock -w 2700,
  TERM-only teardown, production-env via mi50-run.sh, gate witnessed via proc environ).
- Verdict: REJECT (sec 7). Was: PENDING e2e. If readnx drops ~0.5-0.6 ms and probe/acceptance hold ->
  adopt (win is real but sub-noise on tg256; value = draft-phase reduction + free
  GPU2 utilization). If meta-path overhead (hs broadcast + logits gather + extra
  subgraph sync) eats the 0.64 ms -> reject with the decomposition.

## 6. Files

- /root/taskK1/: k1-leg.sh, head_bench{.cpp,}, microbench.log, k1-*.{server.log,
  run.out,witness} (4 LOADFAIL legs, evidence of the wedge timeline), chain.log
- /root/build-k1 (BUILD-PROVENANCE.txt), /root/build-k1.log, /root/mi50-build-k1.sh
  (copy of mi50-build.sh with the / disk guard at 10 GB — / is at 98%, 17 GB free;
  the sanctioned script untouched)
- Branch int4-k1-headsplit: 362542247a (gate) + this ledger. NOT pushed.
- ECC health: amd_gpu_ecc_errors_total all-zero at 23:19 (exporter re-zeroed state;
  GPU0 quarantine stands). GPU2 SMC loop is a NEW fault class from the wedge, not ECC.

## 7. e2e RESULTS (2026-11-16, post-reboot) — VERDICT: REJECT

Device enumeration re-verified post-reboot: rocm-smi GPU2 = PCI 0000:1a:00.0 = KFD node 3;
HIP enumeration follows PCI order, so with HIP_VISIBLE_DEVICES=1,2 the internal ROCm1 IS
physical GPU2. Gate value ROCm1 correct; GPU0 never in any leg's visible set.
Legs k1e-{off,on}-r{1,2}, all rc=0, flock+TERM teardown clean, GPU0 untouched.

| leg        | gate  | probe sha16         | accept agg        | tg256 t/s | readnx med us | fwd med us | draft total med us |
|------------|-------|---------------------|-------------------|-----------|---------------|------------|--------------------|
| k1e-off-r1 | OFF   | 77841cb6a251235d OK | 0.4377 / 2.7508   | 54.31     | 9269          | 533        | 9822               |
| k1e-off-r2 | OFF   | 77841cb6a251235d OK | 0.4377 / 2.7508   | 54.31     | 9285          | 539        | 9844               |
| k1e-on-r1  | ROCm1 | 77841cb6a251235d OK | 0.4377 / 2.7508   | 51.54     | 116           | 12132      | 12259              |
| k1e-on-r2  | ROCm1 | 77841cb6a251235d OK | 0.4364 / 2.7456 MISS | 51.50  | 114           | 12061      | 12187              |

- Gate-OFF control: probe byte-identity + acceptance bit-identity both hold, both reps;
  OFF legs bit-reproducible against each other (all 32 bench records equal).
- Gate-ON numerics: probe byte-identity 4/4 (target output neutral, as designed).
  Acceptance bit-identity: r1 PASS; r2 MISS on 1 of 48 tg256 reps (dialogue rep3:
  draft_n 285->302, accepted 183->179; agg 0.4364/2.7456). Target text unchanged
  (predicted_n identical); only the drafter PROPOSAL stream diverged. Repro probe
  (fresh ON server, dialogue tg256 x5): 5/5 bit-equal canonical (dn 285/da 183, text
  sha16 fb93e4602dc33444) -> rare one-off, not systematic; root cause not identified
  (candidate: 2-lane meta sync race). Logged as latent hazard; moot given perf verdict.
- readnx timer: 9269 -> 116 us is a SYNC-POINT ARTIFACT, not a win. Gate-ON moves the
  blocking wait from llama_get_embeddings_nextn (readnx) into the draft-fwd call
  (fwd_us 533 -> 12132): the 2-lane meta graph synchronizes inside fwd. Honest draft-phase
  total_us: 9822/9844 -> 12259/12187 = +2.4 ms/step. Meta overhead (hs broadcast +
  in-graph logits gather + 2-lane dispatch) eats the 0.64 ms ceiling ~4x over.
- cuda-graph histogram: OFF draft graph = single 690-node HIP graph, capture 66 / replay
  3067 (~2% shape-change re-captures, matches taskW1 96% replay; no per-step re-capture).
  ON: the 690-node graph is GONE - draft shapes dispatch via meta whole-graph capture
  (tg-capture lanes=2 ok=1 x121) plus tg-not-eligible shapes permanently on per-subgraph
  dispatch; draft subgraph replay collapses (nodes=58 dev0: replay 3237 -> 4). Overall
  replay 10130 -> 5120, direct 46620 -> 64654. The 2-lane split forfeits the
  single-graph draft replay that was the W1 win.
- tg256: 54.31 -> 51.52 (-5.1%), consistent with +2.4 ms/step; moves the WRONG way past
  the 2.9% noise floor. (OFF legs read 54.31 not canonical 58.20 - post-reboot
  environment; paired OFF legs are the control, not the pre-wedge champion number.)
- GPU2 witness: VRAM 10.47 GB (OFF) -> 13.43-13.50 GB (ON) = +2.97 GB (mirrored drafter
  ~2.65 + half head ~0.25 + meta scratch); ECC correctable/uncorrectable 0/0 across all
  legs and the repro probe; mem temp peak 61 C, hotspot 82 C - light load as predicted,
  no anomaly. GPU2 stays healthy and usable for light loads.
- draft-gather (host side): identical OFF/ON (med 37 vs 33 us) - the overhead is inside
  fwd (meta graph dispatch), not the host gather.

VERDICT: REJECT. The decision criterion (readnx drops >=0.5 ms with meta overhead not
eating it) FAILS: the apparent readnx drop is a sync-point artifact; the draft phase is
net +2.4 ms/step and tg256 -5.1%. Even a perfect head split caps at -0.64 ms (below the
noise floor); the actual implementation is net-negative because the meta 2-lane machinery
forfeits the single-graph draft replay that was the W1 win. The X3 survivor (monetizing
GPU2 idle draft-slice time via head split) is CLOSED: the offload target (1.43 ms head
matvec) is smaller than the cost of splitting it across devices. Gate
GGML_DFLASH_HEAD_SPLIT stays default OFF; branch NOT merged.
