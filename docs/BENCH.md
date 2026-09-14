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

## 19. AllReduce latency work (2026-09-09): twoshot found + deployed (+3.3% tg)

Method: fork's built-in GGML_TP_AR_KTIMING=1 phase profiler (write/bar_start/
reduce/bar_end per call), no code changes needed for diagnosis.

Phase profile (22,706 decode AR calls, broadcast path): ~65us/call total;
phases anti-correlated across ranks (rank0: write 31k ticks + WAIT 71k;
rank2: write 62k + wait 27k) => pacing item = peer-write phase; reduce trivial
(2.5%); rank-dependent write times = asymmetric PCIe paths.

Root cause of missed win: auto-twoshot selection (5f65f9fa3) lists nranks in
{5,8,10,4-conditional} but NOT 3. GGML_TP_AR_TWOSHOT=1 forces it (12288 %
(4*3) == 0: alignment-eligible). Two-shot cuts per-rank wire 2x48KB -> 2x16KB.

| config | tg128 | notes |
|---|---:|---|
| broadcast (auto) | 16.95 / 16.49-16.75 (r1/r3 class) | baseline |
| GGML_TP_AR_TWOSHOT=1 | **17.42 ± 0.07 (r3)**, 17.57 (r1) | +3.3% decode, PP unchanged |
| 2-rank Qwen3.8-27B Q4_0 | 43.23 vs 43.35 | CORRECTED: neutral on 2 ranks. (First measurement 27.2/26.8 was INVALID - llama-bench treats --device A,B as a cycle list of single-GPU runs, not one 2-GPU run; must HIP_VISIBLE_DEVICES-pin for multi-GPU bench. llama-server is unaffected - its --device takes a set.) |
| quality gate (greedy) | PASS | coherent output, 17.5 t/s |

DEPLOYED: GGML_TP_AR_TWOSHOT=1 added to /llama-swap/start.sh env (service
restarted, live-verified via proxy). Cumulative Mistral-Large fine-tune decode: 7.6 -> 17.4 t/s.
Remaining AR headroom: write phase still paces (asymmetric peer paths);
next lever would be per-rank-pair tuning or barrier overlap - diminishing returns.

## 20. Repack calibration (2026-09-09): dense K-quant repack = +8.7% DECODE on 3-rank tensor

A/B on L3.3-70B-Euryale Q4_K_M (dense, 3-GPU tensor, twoshot on):
- repack ON (default): pp512 109.80 / pp2048 110.53 / tg128 26.87
- repack OFF (-nr 1):  pp512 108.90 / pp2048 108.29 / tg128 24.73
=> +8.7% tg, +1-2% pp. Under 3-rank tensor, PP is AR-bound so kernel gains
barely move it; decode reads via the repack buffer are the real payoff.
(Fork author's +50-133% pp claims are MoE/full-row shapes - not our regime.)
Implication: Q4_0 repack traits entry (subagent authoring, perf/q4_0-repack-gfx906,
env GGML_CUDA_REPACK_Q4_0) is worth ~+5-9% decode on Mistral-Large fine-tune if it lands in
the same class => potential 17.4 -> ~18.5-19 t/s.

## 21. Q4_0 repack LANDED (2026-09-09): +16.6% decode, production-deployed

Patch: branch perf/q4_0-repack-gfx906 @ 3b884f00a (Gitea). 154 insertions, 6 files.
Design (in-parent after 2 subagent attempts died to provider timeouts):
- block_q4_0 byte-identical to block_iq4nl -> converter + layout REUSED from IQ4_NL
- traits: affine decode d*(q-8), scale2 = {d, 8d}, -8 term folded via DS4 activation
  sums (mmq_get_q8_1_ds_layout ALREADY mapped Q4_0->DS4; zero GEMM-side changes)
- admission env-gated GGML_CUDA_REPACK_Q4_0=1 in BOTH gates (tensor_supported +
  meta buft-selection), so off = exactly stock behavior
- numerical class: reduction-order + f16 sum rounding, same as K-quant repack

A/B (Mistral-Large fine-tune, tensor, twoshot on, r3):
| config | pp512 | pp2048 | tg32 | tg128 |
|---|---:|---:|---:|---:|
| repack OFF | 90.13 | 89.77 | 17.13 | 17.12 |
| repack ON | 94.67 | 93.89 | 20.12 | 19.96 |

=> +16.6% tg128, +5% pp. (Q4_K_M calibration predicted +5-9% tg; actual beat it.)
Quality gate: PASS — greedy tides paragraph coherent, primes 2,3,5,7,11,13,17,19
exact; inline gen 20.3 t/s. Live proxy verify PASS.

DEPLOYED: start.sh += GGML_CUDA_REPACK_Q4_0=1; ${llama-server} macro now points
at /root/mx-Mistral-Large fine-tune/build/bin (perf branch build; old tree untouched; backups:
config.yaml.bak-20260909-repack, start.sh.bak-20260909-repack).

CUMULATIVE Mistral-Large fine-tune decode on 3x MI50: 7.62 (layer, deployed Sept 7 AM)
-> 16.7 (tensor, f16 KV) -> 17.42 (twoshot AR) -> 19.96 (Q4_0 repack)
= +162% total vs the original production config.

## 22. Fleet regression suite + AR xbar (2026-09-10): ALL CLEAR, +3.9% 2-rank

Regression suite (5 models, prod binary, twoshot on, gates greedy):
| model | type | ranks | tg128 | gate |
|---|---|---|---:|---|
| Mistral-Large-123B fine-tune | Q4_0 (patch) | 3 | 17.11 -> 19.66 (+14.9%) | PASS primes |
| Qwen3.8-27B | Q4_0 (patch) | 2 | 43.59 -> 46.85 (+7.5%) | PASS |
| Qwen3.8-27B | Q8_0 (stock) | 2 | 36.53 (nr: 35.50) | PASS |
| Ornith-35B MoE | Q4_K_M (stock) | 2 | 96.81 (nr: 93.18) | PASS |
| Ornith-35B MoE | Q8_0 (stock) | 2 | 102.05 | PASS |
No regressions anywhere; stock repack types clean on patched binary.

## 23. AR xbar: broadcast end-barrier elision (perf/ar-latency-gfx906 @ 7de6af95e)

Analysis (in-parent; subagent died to provider timeouts - 3rd strike):
- broadcast kernel: peer-write flush + __threadfence_system (rank-asymmetric,
  31k vs 62k ticks) then TWO barrier RTs; reduce 2.5%.
- barrier_end is cross-call-only (protects staging overwrite). Under parity
  double-buffering (2 staging sets, alternated via _flag>>1&1 in-kernel; flag
  advances +2/call in stock AND xbar mode) consecutive calls never touch the
  same set => barrier provably redundant => elided. Bit-exact (reduce order
  untouched). Twoshot's barrier_end is intra-call (allgather visibility),
  NOT elided.
- Merged to production @ 2094f4db5. Results: 2-rank broadcast path
  43.12 -> 44.81 (+3.9%, variance 10x tighter); 3-rank twoshot no-op (as
  designed). Mistral-Large fine-tune merged-build stack: 20.35 t/s. GGML_TP_AR_SKIP_XBAR=1
  deployed fleet-wide in start.sh. Live proxy verify PASS.
Cumulative Mistral-Large fine-tune: 7.62 -> 20.35 t/s (+167%). Remaining AR cost is the
two PCIe-flush RTs per twoshot call (structural) - next lever would be
fusing the two per-layer ARs, unexplored.

## 24. Q4_K_M lever analysis + twoshot auto-fix (2026-09-10)

Question: can K-quant (Q4_K_M) models be improved like Q4_0 was?
Code + profile findings (rocprof decode, Euryale-70B 3rk / Ornith-35B 2rk):
- Dense Q4_K already rides the repacked mat-vec (mul_mat_vec_rp<12> = 44% of
  GPU time; canonical path gone). MoE GLU-fusion on by default. Dense fused
  mat-vec is Q8_0/MXFP4-only AND measured-losing by the fork author even for
  Q8_0. MMQ tile tables are bypassed under repack. => K-quant kernels are
  already well-served; the real lever was elsewhere:
- PROFILE BOMBSHELL: Ornith 2rk spent 39.6% of GPU time in k_twoshot_f32<2>
  (98.9us/call) — our fleet-wide GGML_TP_AR_TWOSHOT=1 forced twoshot on 2-rank
  models where the fork's auto-select correctly keeps broadcast.
- Measured (2rk): Ornith Q4_K_M 99.15 forced -> 100.53 auto (+1.4%),
  Qwen27B-Q4_0 46.07 -> 47.31 (+2.7%). 3rk unchanged (twoshot both ways).
FIX: one-line auto-select correction (nranks==3 added; perf/tp-ar3-autotwoshot
@ b6fa30224). GGML_TP_AR_TWOSHOT removed from prod start.sh (auto now does
the right thing per rank count). Verified: Mistral-Large fine-tune 3rk auto 20.29 t/s (==
forced), Ornith 2rk auto 100.08, gates pass, live proxy OK.
Per-model AR policy now: 3rk=twoshot(auto), 2rk=broadcast+xbar(auto+flag).
Euryale dense Q4_K 3rk baseline for the record: 21.06 t/s (profiled run).

## 25. Accuracy audit (2026-09-14): optimized stack PRESERVES PPL — PASS

Method: llama-perplexity, same binary (prod build), same corpus (Dickens
771KB, c=2048, fixed chunks), env-flip only per leg. Full logs /root/ppl-logs/.

| model | leg | PPL | verdict |
|---|---|---|---|
| Mistral-Large-123B fine-tune Q4_0 3rk | A stock | 1.0573 ± 0.0035 | baseline |
| | B +Q4_0-repack | 1.0564 ± 0.0035 | delta 0.0009 (4x inside SE) PASS |
| | C B+broadcast-AR | 1.0564 ± 0.0035 | identical to B: AR paths equal PASS |
| Qwen3.8-27B Q4_0 2rk | D stock | 4.9039 ± 0.0745 | baseline |
| | D2 +repack | 4.8990 ± 0.0744 | delta 0.005 (15x inside SE) PASS |
| | E +repack+xbar (prod) | 4.8990 ± 0.0744 | identical to D2: xbar bit-exact PASS |
| Ornith-35B Q4_K_M 2rk | F xbar-off | 1.9495 ± 0.0160 | baseline |
| | G xbar-on (prod) | 1.9495 ± 0.0160 | identical: xbar bit-exact PASS |

Conclusions:
- Q4_0 repack entry: PPL-neutral on both architectures (deltas negative but
  far inside error = zero within measurement).
- AR xbar (my patch): bit-exact confirmed on dense AND MoE (4-decimal match).
- Twoshot vs broadcast AR: identical PPL — fork's lossless-AR claim verified.
- f16 KV: not re-tested here (baked in config, predates patches; standard
  practice, was the chosen non-degrading option vs q8 in section 6).

## 26. MTP speculative decode — tested, NOT deployable (2026-09-14)

The fork ships a modern spec zoo (draft-mtp/eagle3/dflash/dspark, ngram-*).
Tested Ornith-1.5-35B + its MTPv2 head (mtp-Ornith...gguf), 2rk, prod config:
- Acceptance profile good: mean len 2.2-3.8, pos0 rate 0.94.
- BUT throughput 23-24 t/s vs 82-94 t/s control = 4x REGRESSION. Server log:
  "backend offload failed for seq_id; using CPU sampler" x4 — draft sampler
  falls back to CPU per round (root-cause lead: the -sm tensor draft backend
  offload path; likely meta-backend gap for draft contexts).
- AND greedy MTP output != greedy control (both MTP runs skip prime 23;
  control correct). Lossless contract violated -> correctness bug candidate.
VERDICT: closed until the fork's draft-offload bug is fixed upstream. The
Qwen3.6-family MTP-preserved models on disk remain dormant assets for then.
