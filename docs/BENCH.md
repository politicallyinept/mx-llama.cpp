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

## 10. Kernel profile — tensor decode (rocprof, 2026-09-08)

tg96 run (64p+96g), instrumented 14.0 t/s (~19% overhead vs 16.75 clean). Raw:
mi50:/root/prof-tensor/tg.stats.csv. Decode-time attribution per token:

| consumer | share of decode | detail |
|---|---:|---|
| mul_mat_vec_q Q4_0 (type 2) | **~72%** | 2 shape classes: 52.7us x 124,845 (rows 12288-class) + 226us x 25,614 (need_sum=true, 1/layer/GPU — big rows: MLP/vocab class) |
| custom AR (k_broadcast_reduce<3>) | **~25%** | 176 calls/token (2/layer), 88us each — latency-bound, 48KB msgs over 2-hop PCIe |
| rms_norm + quantize_q8_1 | ~4% | 52569 + 102741 calls, 12us/4.4us |
| rest | <2% | |

Prefill-only in profile: mul_mat_q MMQ 14.3%, k_twoshot_f32 AR 8.5% (2.3ms each),
rocclr_copyBufferRectAligned 7.6% (4.7ms x 451 — KV/layer-seam copies).

**Implication: mat-vec (72%) >> AR latency (25%) >> everything else. AR at 88us/call
is ~3x pure-transfer time — a sync-protocol cost; candidate: fewer/lazier barriers,
but kernel-level AR surgery is higher risk than mat-vec work.**

## 11. Tuning experiments — mvq warp sweep (2026-09-08)

Target: `mul_mat_vec_q` Q4_0 (72% of decode time per profile). Tested GCN-table
nwarps for Q4_0 batch-1 (stock=2; RDNA4/DGX-Spark uses 8 for same shape, #26843).
Method: edit mmvq.cu calc_nwarps GCN table -> Q4_0 special case -> incremental
rebuild (~1 min) -> same bench params as T1 (r3).

| variant | tensor tg128 | layer tg128 | verdict |
|---|---:|---:|---|
| stock nwarps=2 | 16.68 | 7.62 | **best — keep** |
| nwarps=4 | 16.48 | 7.26 | -4.7% layer — reject |
| nwarps=1 | 15.16 | 7.25 | -9% tensor — reject |

**Negative result: RDNA4's 8-warp finding does NOT transfer to GCN5. Fork stock
warp config is optimal at this level.** (Thermal note: post-sweep stock re-verify
read 16.48 vs 16.68 earlier — sustained-bench drift; relative ordering valid.)

Remaining decode levers (mapped, not yet attempted):
1. AR latency: k_broadcast_reduce 88us x 176/token = 25% of decode. Pure transfer
   is ~25us for 48KB over 2-hop PCIe -> ~3x sync-protocol overhead. Needs barrier
   laziness/overlap work in tp-allreduce.cu (high risk, high value ~+15-20% tg).
2. Fused dequant+dot inside mul_mat_vec_q for Q4_0 (the 226us need_sum class).
3. MMVQ->MMQ crossover knobs are MoE-only (MUL_MAT_ID batch caps) — N/A dense.
4. rms_norm/quantize are 4% combined — not worth fusing first.

## 12. Subagent delegation status (2026-09-08)
Delegation model degraded (qwen3.8-flash-next: 10 API calls/900s, 4/5 dispatches
timed out). Parent did analysis + patches directly. Partial transcripts at
~/.hermes/cache/delegation/live/{deleg_970b9ec5,deleg_7d8c0a0d}/ — the mmvq
parameter-table map and vecdotq path study were recovered and used.
- ~~llama-server tensor validation end-to-end~~ **DONE 2026-09-08**: ready in ~20s,
  short-prompt gen 16.18 t/s (matches bench), 613-tok prompt eval 86 t/s, 0 errors,
  coherent output. Config validated: -sm tensor --fit off -ctk f16 -ctv f16 -lm dio.
- Switch /root/mx-Mistral-Large fine-tune checkout to Gitea gfx906-w4a4 branch (currently
  GitHub detached at 50d5fdeb + no-RCCL rebuild).
- Tuning: L2 q4_0 MMQ tile entry (mmq-config-gfx906.cuh has Q8_0/MXFP4 only),
  L1 q4_0 repack traits entry (fork admits K-quants + IQ4_NL, not Q4_0).
  Decode ceiling ~46 t/s aggregate; currently 36% of it.
- Optional: quantify llama-swap co-residency effects (Mistral-Large fine-tune tensor + small
  model on leftover VRAM).

## 14. Mistral-Large-123B fine-tune-v3 Q4_0 validation (2026-09-08, same rig/method)

125B params, 67.71 GiB. Quality gate passed (coherent, mode-wording differs as expected).

| config | pp512 | pp2048 | pp4096 | tg32 | tg128 |
|---|---:|---:|---:|---:|---:|
| tensor + f16 KV | 90.05 | 89.33 | 88.35 | 16.04 | 15.93 |
| tensor + q8_0 KV | — | 89.13 | — | — | 15.58 |
| layer + f16 KV | 70.65 | 132.64 | 148.80 | 7.45 | 7.44 |

Mistral-Large fine-tune findings REPLICATE on this independent finetune: tensor decode +114%,
same flat-PP ~89 comm-bound signature, same ~800-token crossover, KV q8_0 costs
2.2% tg. Conclusion: these are properties of the Mistral-Large arch + q4_0 +
3xMI50 tensor split, not model-specific.

### Config bug found + fixed in llama-swap Mistral-Large fine-tune entry (root cause)
Symptom: entry 502'd on spawn ("upstream exited prematurely").
Chain: `--fit on` aborts under SPLIT_MODE_TENSOR (known) -> `--fit-ctx 65536`
is part of the fit machinery and was silently DROPPED with it -> ctx defaulted
to model max 262144 -> with the original q8_0 KV the 256k KV (~4.3 GiB/GPU)
barely fit, masking the bug; moving KV to f16 doubled it (~8.6 GiB/GPU) ->
graph-reserve alloc failure (ggml_backend_meta_buffer_type_alloc_buffer abort,
exit 134). The entry had been unspawnable-in-principle since -sm tensor was
added; nobody noticed because q8 KV + luck kept it just inside VRAM.
Fix deployed: --fit off, explicit -c 32768, f16 KV. Live-verified via proxy
(ready, coherent output). Backups: config.yaml.bak-20260908{-Mistral-Large fine-tune,}.
NOTE: 64k f16 KV does NOT fit on this model (1 GiB/GPU more weights than
Mistral-Large fine-tune); 32k is the max comfortable ctx.

## 15. Delegation timeout fixed
config had delegation.child_timeout_seconds: 900 (explicitly set; code default
is no cap) — raised to 3600 via hermes config set. Local qwen3.8-flash-next
turns run 49-110s; the 900s cap was killing healthy runs.

## 16. llama-swap full config sync (2026-09-09, owner-directed)

Backup: config.yaml.bak-20260908-listsync. Applied via text-preserving transform
(not yaml round-trip). Result: 26 entries, all validated.

- REMOVED (9): model files gone from disk (Laguna-S-2.1, Qwen3.5-122B-heretic-v2,
  RavenX v5.1/v6.2, Qwen3.6-35B-heretic-MTP, ace-saber x2, Fable-Fusion-iq4-amd,
  Qwen3.6-27B-heretic-v2-MTP).
- ADDED (8): Huihui-Qwen3.8-27B-abliterated(27.1G), Qwen3.8-27B base Q8/Q4(27.1/15.0G),
  Muse-Glimmer-30B(27.6G, gemma preset), Ornith-1.5-35B Q8/Q4(34.4/19.7G),
  endless-frontier-BigBang-v1(35.2G), Mistral-7B-v0.3-abliterated(4.1G, spec-draft
  utility). my-stories-lora.gguf excluded (LoRA adapter, not servable).
- DEVICE RULE applied per owner: <=35GB -> --device ROCm1,ROCm2 (low-latency pair);
  >35GB -> all three GPUs. 15 existing entries migrated.
- LANDMINE SWEEP: 0 remaining -sm layer / --fit on / --fit-ctx; every entry has
  explicit -c; ${default_params} macro now has --fit on removed.
- Mistral-Large fine-tune + Mistral-Large fine-tune entries preserved byte-identical (content-level; Mistral-Large fine-tune's
  --fit off diff vs old backup is my earlier validated fix).
- VALIDATION: yaml parse OK; llama-swap hot-reload OK ("Configuration Reloaded");
  26 models listed; spawn test Qwen3.8-27B-Q4_0 PASS (content correct, reasoning
  in reasoning_content, cache-reuse active 42 tokens).
- PITFALL logged: python f-string escaping of ${...} llama-swap macros double-braced
  them (${{PORT}}) and a blind sed half-fix left trailing }} - llama-swap rejected
  the reload ("proxy uses ${PORT} but cmd does not"). Always grep for }} and
  verify "Configuration Reloaded" (not "Unable to reload") in journalctl after edits.

## 17. Speculative decode A/B/C — Mistral-Large fine-tune (2026-09-09): CLOSED, negative

Via llama-server (only spec-capable path), validated tensor config, warm t/s:
- A baseline: 16.83 t/s
- B draft-simple (Mistral-7B-v0.3-abliterated Q4_K_M, uncensored): 17.03 (+1.2%,
  within noise; not worth 4GB VRAM + drafter overhead)
- C ngram-cache: 15.57 (-7.5%; acceptance 0.17-0.25, mean accepted len 1.7-3.0)
Conclusion: spec decode does not pay for creative RP on this target. Do not
retry with other drafts unless acceptance >0.5 is achievable (EAGLE3-class
target-trained drafts do not exist for Mistral-Large). Raw: mi50:/root/spec-ab/.
Remaining lever: AR latency surgery (tp-allreduce.cu, ~25% of decode).

## 18. Subagent provider status (2026-09-09)
Worse than 09-08: both dispatches died to hard API timeouts (180s no-response,
3 retries) + iteration cap at 3 API calls. LiteLLM qwen3.8-flash-next route
unusable for delegation right now. Config-sync + AR analysis were completed
in-parent instead (sections 16 and pending).
