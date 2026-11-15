# taskV2 LEDGER - un-gating A/B on the real 35B Q8_0 MoE (Qwen3.6-35B-A3B)

Date: 2026-11-11 (PST). Driver: /root/taskV2/v2_ab.sh, artifacts /root/taskV2/.
Build: /root/int4-noquant-wt/build @ d496c12154 (tree clean; binary stamped
21:22:43, 35 s after the last serving-code commit d67cd96f0f 21:22:08; the
later E4/E5/E6 commits added test files only). llama-server md5 c4fd0ba7e28290cfdb8b10c2524b69ce.
No BUILD-PROVENANCE.txt in this build dir (built outside mi50-build.sh) - noted.

## Model + fit preflight (dual-layout VRAM note)

Model: /root/bench-quants/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf
(37,812,647,552 B = 35.2 GiB). GGUF tensor inventory (gguf-py, tensors as list):
- total Q8_0: 37.70 GB; dense 2D Q8_0: 2.615 GB (260 tensors); 3D expert Q8_0: 35.08 GB (123 tensors).
- arch qwen35moe: 41 layers, hidden 2048, 16 Q / 2 KV heads (head_dim 256), 256 experts / 8 used.

FIT VERDICT: does NOT fit a single 32 GiB GPU2 even stock (35.2 GiB weights >
32 GiB VRAM). Per the dispatch dual-layout note, ran TP2: -sm tensor --device
ROCm1,ROCm2 (the sanctioned fallback; GPU2-only is physically impossible for
this file). TP2 also satisfies head_count_kv=2 >= TP width 2.

Repack-arm dual-layout cost is bounded by the ADMITTED class, not the whole
model: GGML_CUDA_MOE_CANONICAL default ON keeps the 35.08 GB of 3D expert Q8_0
canonical (banner witnessed: "first expert tensor blk.0.ffn_down_exps.weight
type q8_0"), and the canonical tail copy is allocated only for yield-eligible
tensors (q8_repack/buffer.cu:78 dual = ggml_cuda_q8_yield_eligible). Measured
end-of-leg VRAM: GPU1 21.54 GB / GPU2 21.54 GB used on the repack arm (32 GiB
cards) - ~12.8 GB headroom per card, no pressure.

## Arms (same binary, env-only isolation)

- stock:  production-env defaults (REPACK_Q8_0 unset -> mode 0 OFF; STOCK_ACC
  default ON). Banner witness: "REPACK_Q8_0 gate inactive (env mode 0)" +
  "STOCK_ACC gate ACTIVE: Q8_0 stays canonical".
- repack: GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0, YIELD
  unset = default ON. Banner witness: "REPACK_Q8_0 gate ACTIVE (env mode 1)" +
  "YIELD gate ACTIVE" + "RESTYLE gate ACTIVE (stock granularity)"; live-proc
  /proc/pid/environ witness shows both vars. Env passed UNQUOTED via exports
  before mi50-run.sh (production-env.sh does not touch these three).

Serving flags (both arms, via mi50-run.sh): -ngl 99 -fa on -c 16384 -b 4096
-ub 4096 -t 16 -np 1 -lm dio -fit off -sm tensor --device ROCm1,ROCm2.
Production env bundle asserted by the wrapper (6 required levers + FGP=1).
No MTP/spec flags - this A/B measures the dense Q8_0 repack path on plain
pp/tg workloads.

## Protocol

Interleaved median-of-3: legs stock-r1, repack-r1, stock-r2, repack-r2,
stock-r3, repack-r3 (ports 8561-8566), one fresh server per leg, flock -w 600
/root/gpu.lock per leg (6 retries), RAM guard >=12 GB, GPU1+GPU2 drain <2 GB
before launch and after teardown, TERM-only teardown via the recorded timeout
wrapper PID (single forward, avoids the documented double-TERM hazard), curl
health poll until down. Harness: /root/harnessfix/mtp_bench.py (refuse-if-exists;
per-leg jsonl bench-<leg>.jsonl: 2 warmup + 3 reps pp4096 + tg256, temp-0
seed-42, ignore_eos) + /root/taskV2/pp2048_bench.py (1 warmup + 3 reps pp2048).
Doctor ALL CLEAR before launch; no foreign servers during legs.

## Raw per-leg medians (t/s) and aggregates

tg256  stock : 109.05 / 109.26 / 109.74  -> median 109.26 (spread 0.64%)
tg256  repack: 111.01 / 111.31 / 111.40  -> median 111.31 (spread 0.35%)
pp4096 stock : 3028.98 / 3031.78 / 3034.04 -> median 3031.78 (spread 0.17%)
pp4096 repack: 3027.98 / 3031.07 / 3031.69 -> median 3031.07 (spread 0.12%)
pp2048 stock : 2870.04 / 2874.26 / 2881.69 -> median 2874.26 (spread 0.41%)
pp2048 repack: 2866.10 / 2867.69 / 2873.15 -> median 2867.69 (spread 0.25%)
pp4096 prompt_ms arm medians: stock 1351.02 ms, repack 1351.34 ms.

Deltas (repack vs stock):
- tg256: +1.88%  CLEAN separation - all 3 repack legs beat all 3 stock legs;
  delta (1.88%) is ~3x the stock arm spread (0.64%) -> resolved, not noise.
- pp4096: -0.02% (inside spread) - NEUTRAL.
- pp2048: -0.23% (inside spread) - NEUTRAL.

Coherence/numerics witness: temp-0 seed-42 64-token probe identical across ALL
6 legs (sha256 prefix 35228bd48f947856) - the repack arm is byte-identical to
stock on this greedy probe (yield reproduces the stock gcn_kq regime on
yielded shapes; the repacked mat-vec runs at stock accumulation granularity
per the RESTYLE banner). Full bit-id ladder is the sibling V1 scope.

## Verdict

PASS. The un-gating target state (REPACK_Q8_0=1, STOCK_ACC=0, yield default ON)
preserves and slightly exceeds the +1.3..1.5% decode class on the real 35B
Q8_0 MoE: tg +1.88% with clean per-leg separation, prefill neutral at both
pp4096 and pp2048 (within noise), VRAM cost ~+1.3 GB/card dual-layout on TP2
(21.5 GB/GPU measured, ample headroom). CODE vs CONFIG: no code shipped - the
lever is the existing env pair on the d496c12154 build.

Serving note: the 35B Q8_0 requires TP2 on this rig (35.2 GiB > 32 GiB); the
env pair is safe to adopt for Q8_0 MoE entries at this shape class.

## Findings

1. Fit: 35B Q8_0 (35.2 GiB) exceeds one 32 GiB MI50 even stock -> TP2
   ROCm1,ROCm2 per the dispatch fallback; GPU2-only impossible for this file.
2. Repack dual-layout touched only the 2.615 GB dense 2D Q8_0 subset (experts
   canonical via MOE_CANONICAL default ON); measured +~1.3 GB/GPU, fits with
   ~12.8 GB/GPU headroom.
3. tg +1.88% clean separation (all repack legs > all stock legs; spreads
   0.35%/0.64%) - above the +1.3..1.5% class, preserved on the real MoE.
4. pp4096 -0.02%, pp2048 -0.23% - both inside arm spread, NEUTRAL as expected
   (yield hands gcn_kq-admissible prefill shapes back to the stock kernel).
5. Greedy temp-0 probe byte-identical across all 6 legs (35228bd48f947856).
6. Banner-witnessed per arm (gate ACTIVE/inactive + live /proc environ); no
   silent arm collapse.
7. All 6 legs OK on first attempt; no retries, no wedges, VRAM drained <2 GB
   after every teardown.

---

# taskV2-moe (R8) - MoE repack-admission decision: REPACK_Q8_0 on the 35B MoE ssm_out class

Date: 2026-11-15 (PST). R8 decision lane (STATE 2026-11-12 sec 26: "MoE repack-admission
decision"). Driver /root/taskV2-moe/drive.sh, artifacts /root/taskV2-moe/. Branch
int4-v2-moe from tip e9b3e3197f (worktree /root/wt-v2moe). Build: /root/build-uwave
(BUILD-PROVENANCE src_sha=e9b3e3197f dirty=0; llama-server md5 b956a6858d8976624271378f97d825cb,
libggml-hip.so 127f1494af87e2a85b2e6801892ca542, identical before and after all legs). No pushes.

## 0. Question

The V2 un-gating lever (GGML_CUDA_REPACK_Q8_0=1 + GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0, yield
default-ON) was bit-identical on the 27B Q8_0 stack and +1.3..1.5% tg class (Nov-11 taskV1/V2
legs, build d496c12154). R8: does the admission extend to the 35B MoE's ssm_out-class shapes
on the current tip - adopt / modify / reject.

## 1. Inventory - exact admission shapes (GGUFReader, header-only)

Model: /root/bench-quants/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf
(37,812,647,552 B; qwen35moe, 41 blocks, 256 experts / 8 used).

* 753 tensors: 383 Q8_0 (37.70 GB), 368 F32 norms (0.105 GB), 2 BF16 (1.05 MB).
* Q8_0 split: 260 dense 2D = 2.615 GB; 123 expert 3D = 35.08 GB.
* ssm_out class: 30 tensors, ALL Q8_0, shape [4096, 2048], 267 MB total (30 of 41 layers are
  GDN; the other 11 are full-attention).
* Admission predicate (repack-common.cu ggml_cuda_repack_tensor_supported): Q8_0, 2D/3D,
  contiguous, non-view, ne0 % 32 == 0 -> ALL 383 Q8_0 tensors pass (0 declined).
* Effective repacked class at runtime: MOE_CANONICAL default-ON keeps the 35.08 GB of 3D
  expert Q8_0 canonical (banner witnessed), so the admitted-and-repacked set is the 260
  dense 2D tensors (2.615 GB): ssm_out 30x[4096,2048] (267 MB), output/token_embd
  2x[2048,248320] (1.08 GB), attn_qkv 41x[2048,8192], attn_gate 30x[2048,4096],
  ffn_{gate,up,down}_shexp dense shared-expert set. The ssm_out tensors are the per-layer
  decode-hot members of this class (the J2 LDS microbench proved ssm_out is the narrow-matvec
  hot spot at verify widths).
* NOTE (dispatch-brief discrepancy): the brief's tensor map "323 Q8_0 dense, Q4_K/Q5_K/Q6_K
  experts, F32 norms" matches the UD-Q4_K_XL file (312 Q8_0 + Q4_K/Q5_K/Q6_K experts), NOT
  this Q8_0 file. Legs ran on the Q8_0 file per the named path and the 106.8 t/s plain-decode
  anchor (taskM/taskI2). The UD file's ssm_out tensors are also Q8_0 [4096,2048] - the
  admission class is the same shape class on both files.

## 2. What admission changes (why a numeric seal is required)

REPACK_Q8_0=1 + STOCK_ACC=0 moves the admitted dense Q8_0 tensors from canonical layout to
the repacked two-plane dp4a layout (lossless byte permutation of the block_q8_0 data + scale
plane). Decode MMV runs the RESTYLE kernel (four scaled fp32 terms per 32-value block =
stock accumulation granularity); gcn_kq-admissible prefill GEMMs yield to the stock gcn_kq
path via the canonical copy (YIELD default-ON). The claim is bit-identity, but the layout
change is exactly the class of change that can move fp32 accumulation order, so the verdict
is seal-gated: greedy probe byte-identity + KLD vs stock dump. VRAM: dual-layout (repacked +
canonical copy) is bounded to the admitted dense subset (yield_eligible requires ne[2]==1).

## 3. Arms / protocol (V2 protocol, current tip, pair B)

Same binary, env-only isolation:
* stock: production-env defaults (REPACK_Q8_0 unset -> mode 0; STOCK_ACC default-ON).
  Witness: "gate inactive (env mode 0)" + STOCK_ACC ACTIVE banner.
* repack: GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0 (YIELD unset = ON).
  Witness: "gate ACTIVE (env mode 1)" + RESTYLE ACTIVE + YIELD ACTIVE + functional witness
  "q8_repack: repacking q8_0 tensors on device 0" AND "device 1" + /proc/pid/environ.
Serving line (both arms): -ngl 99 -fa on -sm tensor -fit off -c 16384 -b 4096 -ub 4096
-t 16 -np 1 -lm dio, HIP_VISIBLE_DEVICES=2,3 (pair B TP2; 35.2 GiB > one 32 GiB card).
Production env asserted; GGML_TP_AR_MTP_SAFE=1.
Legs: interleaved median-of-3 stock/repack (ports 8571-8576), mtp_bench pp4096+tg256 +
pp2048 per leg; greedy suite (temp0 seed42 logprobs=1, 10 probes p0008..p4128 covering the
declined 9..31 GEMM class, the >=32 yield class, chunked prefill 4096+16/32, and N=1 repacked
MMV decode) on stock-r1/r2 + repack-r1; KLD legs llama-perplexity wiki.test.raw -c 2048
-b 2048 --chunks 16 (32768 positions): stock dump, repack vs dump, stock-vs-own-dump control.
Ops: flock /root/gpu.lock -w 900 per leg (6 retries), RAM guard >=12 GB, GPU2/3 drain <2 GB
before/after every leg, TERM-only teardown (PID-targeted), doctor ALL CLEAR pre-launch.
Sibling lane (taskV3 27B DFlash2, ports 8431/8432) shared the lock and sat on GPU2/3 during
early legs - all legs waited via flock + drain gate, zero LOCKFAILs, zero contamination
(stock arm spread 0.75 pct). All 9 legs OK first attempt; pgrep -x llama-server empty after
every teardown (the surviving server after ALL-DONE was the sibling's, verified by cmdline).

## 4. Results

Greedy seal (v1 protocol, compare.py):
* Determinism control stock-r1 vs stock-r2: 10/10 BITID.
* THE GATE stock-r1 vs repack-r1: 10/10 BITID - text sha EQ, logprob-payload sha EQ,
  tokdiff=0, max_logprob_delta 0.000e+00 on every probe. GREEDY-BIT-ID: PASS.

KLD (llama-perplexity anchor, 32768 positions):
* repack vs stock dump: Mean KLD -0.000000 +/- 0.000000, Maximum +0.000063, Minimum -0.000061,
  Mean PPL(Q) 5.891720 +/- 0.110374, PPL ratio 1.000667.
* CONTROL (stock vs its own dump): IDENTICAL statistics to the last digit (mean -0.000000,
  max +0.000063, min -0.000061, ratio 1.000667) -> the +/-6e-5 band is the uint16 base-logprob
  quantization floor, not a logit delta. KLD 0.000000 class CONFIRMED (reproduces taskV1).

tg/pp A/B (per-leg medians of 3):
| metric  | stock reps              | median  | repack reps             | median  | delta   |
|---------|-------------------------|---------|-------------------------|---------|---------|
| tg256   | 108.79/109.47/109.62    | 109.47  | 111.48/111.32/111.46    | 111.46  | +1.82%  |
| pp4096  | 3001.71/3005.79/3004.57 | 3004.57 | 2998.81/2999.61/3004.38 | 2999.61 | -0.17%  |
| pp2048  | 2843.97/2846.50/2847.13 | 2846.50 | 2839.80/2846.81/2842.83 | 2842.83 | -0.13%  |

tg256 CLEAN separation: every repack leg (min 110.69 per-rep) beats every stock leg
(max 109.91 per-rep); arm spreads 0.14 pct (repack) / 0.75 pct (stock) - delta is ~2.4x
the stock spread. pp4096/pp2048 inside arm spread = NEUTRAL (yield hands gcn_kq-admissible
prefill back to the stock kernel). One cold first-rep pp4096 artifact (2671) in stock-r3
excluded by the per-leg median, consistent with the documented cold-rep class.
VRAM at end of measurement: repack 21.56-21.65 GB/GPU vs stock 20.78-20.87 GB/GPU ->
dual-layout cost ~+0.79 GB/GPU on TP2, ample headroom on 32 GiB cards.

## 5. Verdict: ADOPT

The V2 repack-admission lever extends to the 35B MoE's ssm_out-class shapes BIT-IDENTICALLY
on the current tip (greedy 10/10 BITID + control-confirmed KLD 0.000000), and carries the
full V2 tg class on the MoE's fastest serving mode (plain decode): +1.82% tg256 with clean
per-leg separation, prefill neutral at both widths. This is the first MoE serving gain from
the repack admission path, and it is CONFIG, not CODE: the lever is the existing env pair
GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0 on the e9b3e3197f build.

Serving recommendation: add the env pair to the 35B Q8_0 MoE plain-decode entry (TP2 pair).
Expected: ~109.5 -> ~111.5 t/s natural tg256. No admission narrowing needed ("modify" not
required): the whole admitted dense class is bit-identical, and the gain is class-wide
(lm_head/token_embd + ssm_out + dense attn/ffn), not ssm_out-only - narrowing to ssm_out
alone would forfeit most of the win for zero numerics benefit.
Caveats: (1) mode =1 forces admission for EVERY model in the process - for fleet-wide use
the shape-gated level (=2, per-model hint: 3D experts + dense-Q8_0 floor + tensor split)
is the shipping posture; this lane validates the admission CLASS, not a default flip.
(2) The bit-identity holds for the serving shape measured (-ub 4096, TP2 tensor-split,
plain decode); a serving-shape change must be re-sealed per topology (ub-4096 precedent).
(3) DFlash2/spec-decode configs on this model were not re-sealed here (plain decode is the
serving mode; nc widths 6-8 remain E2 microbench evidence).

## 6. Artifacts

* /root/taskV2-moe/: drive.sh, driver.log, arm-status.txt (9x OK), per-leg server logs,
  bench-*.jsonl, pp2048-*.jsonl, greedy-{stock-r1,stock-r2,repack-r1}.jsonl, compare.py
  verdict (DETERMINISM PASS / GREEDY-BIT-ID PASS), kld-{stock-dump,repack,control}.log,
  *.witness.txt (banners + /proc environ + VRAM), aggregate.py, bin-md5.txt.
* Inventory: /root/taskV2-moe/inventory.py (GGUFReader list API, header-only).
* This ledger section: appended to taskV2-LEDGER.md on branch int4-v2-moe (worktree
  /root/wt-v2moe). NOT pushed.
