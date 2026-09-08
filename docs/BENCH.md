# BENCH.md — Mistral-Large-123B fine-tune-v2 q4_0 on 3x MI50 32GB (gfx906, TheRock ROCm 7.2.3)

All runs: -lm dio, env GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1
HSA_XNACK=0 GPU_MAX_HW_QUEUES=8, HEAD build 50d5fdeba (no RCCL) unless noted.
Model 64.56 GiB, r=3 unless noted. Raw tables: mi50:/root/bench-results/*.md

## 1. Split mode — the headline (2026-09-07)

| config | pp512 | pp1024 | pp2048 | pp4096 | tg16/32 | tg128 |
|---|---:|---:|---:|---:|---:|---:|
| layer, fa1 (deployed config class) | 70.9 | 104.8 | 134.6 | 151.7 | 7.57 | 7.62 |
| tensor, fa1 | 88.9-90.5 | 89.8 | 89.7 | 88.6-91.3 | 16.75 | 16.49 |

- **Decode: tensor +117% (7.6 -> 16.5-16.8 t/s).** This is the whole ballgame for
  a chat/RP model: generation dominates wall-clock.
- **Prefill: layer scales with batch (70->152), tensor flat ~90 (comm-bound).**
  Crossover ~700-800 prompt tokens. Worst case for tensor (8k prompt + 500 gen):
  layer ~81s vs tensor ~107s; typical RP turn (500 prompt + 500 gen):
  layer ~72s vs tensor ~40s. Decode-heavy = tensor wins.
- Wall-clock at 2k prompt + 500 gen: layer ~81s, tensor ~53s.

## 2. Env toggles (tensor, pp2048/tg128)

| env | pp2048 | tg128 | verdict |
|---|---:|---:|---|
| full (AR=1+graphs+dispatch) | 89.75 | 16.68 | keep |
| GGML_ENABLE_CUSTOM_AR=0 | 65.00 | 13.30 | **custom AR = +25% tg, keep ON always** |
| GGML_META_TOKEN_GRAPH=0 | 89.75 | 16.09 | token graph +3.7% tg, keep ON |
| GGML_META_PARALLEL_DISPATCH=0 | 89.35 | 16.06 | parallel dispatch +3.9% tg, keep ON |

## 3. KV quant (tensor, pp2048/tg128)
f16: 89.66 / 16.49. q8_0 both: 89.50 / 15.91. **q8 KV costs 3.5% decode, saves
~1GB total (GQA-8 KV is tiny) — use f16.** (Deployed config used q8_0 — change it.)

## 4. ubatch (tensor, pp4096/tg16)
ub512: 88.31/16.42. ub1024: 90.46/16.43. ub2048: 91.26/16.47. Keep -ub 2048.

## 5. fa
layer: fa1 >= fa0 (71.07/7.62 vs 70.94/7.21). tensor+fa0: **context creation
FAILS** (KQ score buffers exceed VRAM with weights+KV mirrored). fa=on required
for tensor mode. Keep fa on.

## 6. Build/platform findings (root-caused)
- **TheRock ROCm 7.2.3 RCCL segfaults on gfx906** in tensor mode
  (bt: ggml_backend_cuda_comm_try_allreduce_nccl -> librccl -> libamdhip64).
  Fix: build WITHOUT -DGGML_HIP_RCCL=ON; custom AR (GGML_ENABLE_CUSTOM_AR=1)
  covers the per-layer 48KB allreduce. Deployed build never had RCCL — works.
- Build flags that work: GGML_HIP=ON GRAPHS=ON MMQ_MFMA=ON NO_VMM=ON,
  AMDGPU_TARGETS=gfx906, clang, -O3. 4.5 min build on the 5950X.
- llama-bench has no -c flag (ctx from -p/-n); llama-cli defaults huge ctx ->
  KV OOM under tensor — ALWAYS pass -c.
- llama-cli: -no-cnv removed upstream; use -st (single-turn) for scripted gens.
- llama-server --fit: "not implemented for SPLIT_MODE_TENSOR" warning + aborts
  fit only — pass --fit off explicitly for tensor configs.
- GPU contention: llama-swap spawns on 8181 requests; benching needs an idle
  window (two of my runs died to this; also each llama-bench exit leaves ~64GB
  VRAM draining — the fixed harness waits for drain between runs).

## 7. Quality gate — PASSED
Greedy (-c 32768, -st, temp 0): layer and tensor both produce correct coherent
output; wording differs slightly (summation-order class, same as fork documents
for its bit-exact-except-reduction-order paths). No degeneration.

## 8. Recommended deployed config (vs current)

Current llama-swap Mistral-Large fine-tune: -sm layer, -ts 30/29/30, -ctk/-ctv q8_0, -fit absent.
Recommended:

```yaml
"Mistral-Large-123B fine-tune-v2":
  cmd: >
    ${llama-server}
    --model ${model_path}/Mistral-Large-123B fine-tune-v2-Q4_0/Mistral-Large-123B fine-tune-v2-Q4_0-00001-of-00002.gguf
    --port ${PORT}
    -ngl 999 -fa on -sm tensor --fit off
    --device ROCm0,ROCm1,ROCm2
    -ctk f16 -ctv f16
    -b 2048 -ub 2048 -c 32768
    -lm dio
```
(box env already exports the AR/PCIe vars; -ts not needed — tensor mode splits evenly)

Expected: tg 7.6 -> ~16.5 t/s (+117%), pp512 72 -> ~89; pp2048+ slower than
layer (135 -> 90) — acceptable for decode-dominant RP use.

## 9. PENDING (next session)
- llama-server tensor validation end-to-end (was blocked by active llama-swap
  workloads — Qwen3.8-27B on ROCm1+2, Ornith on ROCm0).
- Switch /root/mx-Mistral-Large fine-tune checkout to Gitea gfx906-w4a4 branch (currently
  GitHub detached at 50d5fdeb + no-RCCL rebuild).
- Tuning: L2 q4_0 MMQ tile entry (mmq-config-gfx906.cuh has Q8_0/MXFP4 only),
  L1 q4_0 repack traits entry (fork admits K-quants + IQ4_NL, not Q4_0).
  Decode ceiling ~46 t/s aggregate; currently 36% of it.
- Optional: quantify llama-swap co-residency effects (Mistral-Large fine-tune tensor + small
  model on leftover VRAM).
