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
