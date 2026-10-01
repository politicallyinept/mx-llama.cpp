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

## 27. Upstream sync 2026-09-16 (32 commits) merged: CLEAN, deployed

Merge: perf/upstream-20260916 @ f35c1bd3e (base b6fa30224 + upstream 62d4be47d).
Conflicts (Q4_0 x Q5_1 in q8_repack) resolved keeping both types; 3 build-fix
passes. Highlights: custom AR default-on, twoshot allgather-region fix
(MEASURED on our rig: 76.9 -> 69.2 us/call, -10%), Q5_1 repack entry,
DeepSeek-V4.1 arch, DSpark/MTP spec work, -tps mirrored weights.

REGRESSION SCARE, RESOLVED: first post-merge suite read Mistral-Large fine-tune 19.56 (was
20.29) and Qwen 44.37 (was 47.31). Interleaved same-day A/B against a fresh
build of the PRE-MERGE commit: pre 19.37/19.56 + 44.38/44.27 vs merged
19.53/19.32 + 44.37/44.47 => IDENTICAL. The Sep-10 reference numbers were
simply from a faster box-day (~4-6% day-drift). Kernel profiles agree: merged
GPU time LOWER (11.23 vs 11.56s), AR improved, repack gain intact
(16.49 repack-off vs 19.38 on same day). PPL bit-identical. Gates pass.
LESSON (now policy): never compare t/s across days; interleave a control.

MTP RETEST on merged build (Ornith + MTPv2 head): still broken the same way —
23-25 t/s vs 82-94 control, CPU-sampler fallback x4 in log, greedy output
still skips prime 23 (lossless violation). Upstream's spec fixes did not
address it; root cause still open (draft backend offload under -sm tensor).
MTP stays closed. Reference-build trick retained: /root/mx-premerge (b6fa30224)
for future interleaved controls.

## 28. MXFP4 Mistral-Large fine-tune experiment — first leg complete (2026-09-18)

Setup: quantized dense-MXFP4 from bartowski Q8_0 source (no imatrix) using our patched
quantizer (perf/mxfp4-quant @ f6d96e052: FTYPE 42, 2D weights -> MXFP4). 65.4GB out.

| Metric | Q4_0+repack (ref) | MXFP4 no-imat | Delta |
|---|---|---|---|
| tg128 | 19.4-19.6 | 20.10 ± 0.05 | +3% |
| pp2048 | 90.6 | 86.14 | -5% |
| PPL (Dickens 12ck) | 1.0564 | 1.1184 (+5.9%) | quality cost |
| Greedy gate | pass | pass (primes exact) | |

Read: MXFP4's fast path delivers the expected decode win (+3% over Q4_0-repack,
~+21% over stock Q4_0). PPL cost is real but modest; imatrix leg running to see
how much recovers. NOTE: overnight-run lessons logged — (1) quantizer needs
--allow-requantize for Q8_0 sources, (2) script idempotency must size-check
outputs (409MB partial poisoned the first auto-resume).

## 28b. MXFP4 imatrix leg (2026-09-18): imatrix is a NO-OP — zero recovery

Ran the second leg with the official importance matrix
(MarsupialAI/Mistral-Large-123B fine-tune_iMat_GGUF/Mistral-Large fine-tune-imatrix.dat, 36MB, sha256
8d94fc64… verified against repo LFS oid; quant log confirms: "loaded 616
importance matrix entries … computed on 105 chunks"). Quantized with
--allow-requantize --leave-output-tensor --imatrix on the same Q8_0 source
(perf/mxfp4-quant @ f6d96e052). Output 65,358,845,472 B (4.26 BPW), 3.2 min.

| Metric | Q4_0+repack (ref) | MXFP4 no-imat | MXFP4 + imatrix |
|---|---|---|---|
| tg128 | 19.4-19.6 | 20.10 ± 0.05 | 20.10 ± 0.04 |
| pp2048 | 90.6 | 86.14 | 86.23 ± 0.04 |
| PPL (Dickens 12ck) | 1.0564 | 1.1184 | 1.1184 ± 0.0058 |
| Greedy gate | pass | pass | pass (primes exact) |

ROOT CAUSE of zero delta: `quantize_row_mxfp4_ref` (ggml/src/ggml-quants.c)
computes per-block E8M0 scales from amax only — the function signature takes no
importance-matrix argument, so the .dat is loaded, validated, and then never
consulted on the dense MXFP4 path. The imat file is bit-equivalent in behavior
to no-imat. This is NOT a "imatrix didn't help MXFP4" result; it's "our MXFP4
quantizer ignores imatrix."

Decision: imatrix cannot close the +5.9% PPL gap until the quantizer is patched
to consume per-column importances (e.g. importance-weighted scale selection /
rounding like the k-quant paths). Until then MXFP4-imat == MXFP4-noimat; no
reason to prefer the imat artifact. Trade vs Q4_0-repack unchanged:
+3% decode / -5% prefill / +5.9% PPL. KEEP Q4_0-repack in production.
MXFP4 stays experimental; revisit only with an imatrix-aware quantizer.

Ops notes: imatrix file is 36MB (not ~1GB as expected) — verified correct via
repo tree API + sha256. Q8_0 source shards deleted after all gates passed
(/mnt/old back to ~264G free); both MXFP4 ggufs + imatrix retained.

## 29. BF16-on-wire AllReduce — deploy-worthy (2026-09-18)

Env-gated twoshot AR with BF16 wire format: GGML_TP_AR_WIRE=bf16s (scatter only)
or bf16 (scatter+allgather). Unset/f32 = stock byte-identical. Halves peer bytes
per hop (8 bf16 per 16B store). Accumulation stays F32, same r-loop order.
Branch perf/ar-bf16-wire-gfx906 (7dbc60505).

Interleaved bench (Mistral-Large fine-tune 3rk, tg128, r=5, same-day):
| leg | config | tg128 |
|---|---|---|
| A | f32 control | 16.55 |
| B | bf16s | 15.08 (-8.9%) |
| C | bf16 | 16.97 (+2.5%) |
| D | f32 drift ctrl | 16.56 (+0.06%) |

PPL (Dickens 12ck, after addressing fix):
| config | PPL | vs f32 |
|---|---|---|
| f32 | 1.0573 | — |
| bf16s | 1.1011 | +0.044 |
| bf16 | 1.0588 | +0.0015 |

Gate: bf16 greedy primes exact (2..29). PASS.

Verdict: bf16 (scatter+allgather) clears both bars — +3.3% tg128 (confirmed
16.97/16.99 across two runs) AND PPL +0.0015 (inside +0.010 tol). bf16s is a
net loss: converting the scatter (non-barrier-gating phase) costs ALU with no
wire benefit where it matters; the win comes from halving the ALLGATHER wire.
gfx906 has no native bf16 (emulated conversion), which is why wasted
conversion in bf16s hurts.

Bug caught by PPL gate: initial patch had 2x slot-stride error (readers used
slice*1 instead of slice*4) -> PPL 13671. Fixed in 7dbc60505. Lesson: bench
never validates output; the PPL gate is what makes a wire-format change safe.

NOT deployed — production stays f32 until Brad opts in. To enable: set
GGML_TP_AR_WIRE=bf16 in the llama-swap env for Mistral-Large fine-tune.

### 29b. bf16-wire stacked on full production (repack ON) — 2026-09-18

Interleaved 4-leg (r=3 each): f32 19.62/19.63 | bf16 20.23/19.90.
Stacked gain +2.2% (vs +3.3% measured on the no-repack baseline) — matches
the multiplicative model: faster mat-vec shrinks AR's share of the critical
path. Deploy expectation: Mistral-Large fine-tune ~20.1 t/s, PPL unchanged (+0.0015;
repack and AR-wire are independent, both individually PPL-validated).
Note: the MXFP4-table "Q4_0+repack 19.4-19.6" vs this experiment's f32
control 16.4-16.6 difference = the repack flag itself (+17.5%, measured
Sep 17: 16.49->19.38). Different baselines, no contradiction.

## 30. bf16-wire full regression suite (2026-09-18/19) — NO REGRESSIONS

Full production stack in every leg (repack+xbar+custom-AR+FA+tensor); only
GGML_TP_AR_WIRE toggled. Interleaved f32/bf16/f32 same-day per model.

| Model | f32 | bf16 | delta | PPL f32->bf16 | Gate |
|---|---:|---:|---:|---|---|
| Mistral-Large-123B fine-tune Q4_0 3rk | 18.82 | 19.47 | +3.5% | — | PASS |
| Qwen3.8-27B Q4_0 2rk | 43.41 | 44.41 | +2.4% | 4.8277 -> 4.8378 (+0.010, inside SE) | PASS |
| Qwen3.8-27B Q8_0 2rk | 34.72* | 35.00 | +0.8% | — | PASS |
| Ornith-35B Q4_K_M 2rk | 100.74 | 100.68 | ~0% | 1.9212 -> 1.9190 (-0.002) | PASS** |
| Ornith-35B Q8_0 2rk | 103.34 | 103.82 | +0.4% | — | PASS** |

* first f32 Q8_0 leg contaminated (GPU not idle); clean repeat used.
** Ornith is a thinking model: -n 100 burns out in the reasoning preamble
   (reasoning itself was correct). -n 512 required for the gate to reach the answer.

Notes:
- 2-rank models DO ride k_twoshot_f32<2> (large tensors exceed legacy min),
  so the bf16 patch reaches them; MoE models are AR-amortized -> neutral.
- Both models read slightly better than Sep-14 PPL baselines on BOTH legs
  (upstream merge drift, not bf16).
- Harness lessons: parse PPL with 'sed s/.*PPL = //' (regex-first-number grabs
  log timestamps); strip build banner before gate parsing; thinking models need
  -n 512 gates.
- Deployment: flag currently on Mistral-Large fine-tune only. Mistral-Large fine-tune 3rk is the best next
  candidate (+3.5%, same topology as Mistral-Large fine-tune). Config untouched per request.

## 31. Upstream sync 2026-09-19 (11 commits) merged + validated

Merge: perf/upstream-20260919 @ e721412ab (our line 7dbc60505 + upstream
5542318e7). Auto-merge CLEAN, zero conflicts.

FEATURES.md review (new since 62d4be47d):
- DFlash/DFlash2 drafters under -sm tensor: axis-0 gather so a drafter reading
  the target's split lm head sees whole vocab rows. Upstream: Qwen3.8-27B-Q8 +
  DFlash2 on 2xMI50 43.8 -> 52.1 t/s decode. UNLOCKS spec-decode under TP for
  our Qwen fleet. -tpsd runs drafter on fewer lanes.
- Tied output head split (gemma-4 family): head no longer falls to mirrored
  default; +9-11% decode. We run Gemma 4 models.
- Verify-batch twoshot crossover: nranks>=4 crosses at 8192 elements
  (GGML_TP_AR_TWOSHOT_MIN_NE tunable). AUDITED: only nranks>=4 semantics
  change; our 3-rank Mistral-Large fine-tune + 2-rank Qwen/Ornith gates unchanged.
- Token-graph stage-seam capture (multi-stage pipelines), lane graph-cache off
  under token graph, fused 384-expert MoE router, DSV4 hc-gate fusion,
  full-device report at load.

Validation (merged build, full prod stack, vs same-day Sep-18 baselines):
| Model | merged f32 | baseline | delta |
|---|---:|---:|---:|
| Mistral-Large-123B fine-tune Q4_0 3rk | 18.83 | 18.82 | +0.05% |
| Qwen3.8-27B Q4_0 2rk | 44.43 | 43.41 | +2.35% |
| Qwen3.8-27B Q8_0 2rk | 35.03 | 34.72 | +0.89% |
| Ornith-35B Q4_K_M 2rk | 101.14 | 100.74 | +0.40% |
| Ornith-35B Q8_0 2rk | 104.77 | 103.34 | +1.38% |
PPL: Qwen 4.8277 (bit-identical to baseline), Ornith 1.9212 (bit-identical).
Gate: 5/5 PASS.
bf16-wire spot-check post-merge: Mistral-Large fine-tune 19.48 (baseline 19.47). Qwen bf16
interleaved A/B: 44.54/44.53 vs f32 43.05/43.49 — bf16 wins both pairs with
10x tighter variance. bf16 path intact and healthy post-merge.

VERDICT: clean merge, no regressions, all features intact. Deploy-ready.
Production still on prior binary pending go-ahead.

## 32. DFlash vs MTP speculative-decode shootout (2026-09-19)

Same-tool interleaved legs via llama-cli (temp 0, greedy, -n 256, realistic
RP creative-writing prompt), full prod env + LLAMA_ENABLE_MTP_OPT=1.
DFlash drafter: z-lab/Anbeeld Q8_0. MTP: native blk.64.nextn head.

### A: Qwen3.8-27B Q4_0 (actual prod target — no MTP head)
| leg | gen t/s | vs baseline |
|---|---:|---:|
| baseline (no spec) | 42.2 / 44.4 (mean 43.3) | — |
| DFlash2 (n-max 5) | 36.9 | **−14.8%** |
DFlash acceptance: 51.5% (183/355), mean len 3.54.

### B: Qwen3.6-27B-Native-MTP Q8_0 (same target, engine isolated)
| leg | gen t/s | vs baseline |
|---|---:|---:|
| baseline (no spec) | 35.5 / 35.5 | — |
| MTP (native head) | 41.4 | **+16.6%** |
| DFlash (drafter) | 37.1 | +4.5% |
MTP acceptance: 51.4% (183/356), mean len 3.54.

### Findings
- Acceptance is ~51% for BOTH engines — acceptance is NOT the differentiator.
- The differentiator is DRAFT COST. MTP head is embedded in the target and
  drafts off already-computed activations (near-free, shares KV). DFlash runs
  a separate 5-block transformer with its own KV cache every step.
- On the FAST Q4_0 target (3.8, 43 t/s), DFlash's separate draft pass can't
  pay for itself -> net −14.8%. On the slower Q8 target (3.6, 35 t/s) the
  draft overhead is a smaller fraction -> DFlash still nets +4.5%.
- Same-target (B): MTP (+16.6%) clearly beats DFlash (+4.5%). MTP is the
  better engine when the target has a native head.

### Verdict
- Prod Qwen3.8-27B Q4_0: neither spec method helps (DFlash net loss, MTP
  unavailable). Keep no-spec. Fast Q4_0 targets don't amortize draft overhead.
- If a Qwen3.8 MTP-head build is ever made, MTP is the path to try (+16% class).
- DFlash is only worth it on SLOW targets where draft cost is amortized; not
  on our fast Q4_0 fleet.

## 34. Post-PLX link upgrade: baselines + bf16-wire REVERSAL (2026-09-21)

Hardware change: inter-GPU links moved to a PLX switch card. All 6 models
re-baselined on ALL 3 CARDS, full prod stack (repack+xbar+custom AR+FA), r=3.

| Model | pp2048 | tg128 | prior tg128 | note |
|---|---:|---:|---:|---|
| Mistral-Large-123B fine-tune Q4_0 (3rk) | 176.4 | 21.39 | 19.63 | +8.9% (clean link effect, already 3rk) |
| Mistral-Large-123B fine-tune Q4_0 (3rk) | 173.1 | 20.49 | 18.82 | +8.9% (clean link effect) |
| Qwen3.8-27B Q4_0 (3rk) | 567.0 | 54.83 | 43.41 (2rk) | conflates link + rank change |
| Qwen3.8-27B Q8_0 (3rk) | 657.5 | 45.53 | 34.72 (2rk) | conflates link + rank change |
| Ornith-35B Q4_K_M (3rk) | 1543.8 | 103.92 | 100.74 (2rk) | +3.2% |
| Ornith-35B Q8_0 (3rk) | 1900.7 | 109.14 | 103.34 (2rk) | +5.6% |

PPL (improved or flat): Mistral-Large fine-tune 1.0564, Qwen3.8-27B 4.8235, Ornith 1.9021.
Gates: 3/3 PASS.

### bf16-wire is now a LOSS — the mechanism reversed

| Mistral-Large fine-tune tg128 | pre-PLX | post-PLX |
|---|---:|---:|
| f32 | 19.63 | 20.86 |
| bf16 (deployed) | 20.07 (+2.2%) | 20.62 (-1.2%) |

Interleaved 4-leg (f32/bf16/bf16/f32), variance +/-0.05: both bf16 legs below
both f32 legs. Qwen shows the same trend (+2.4% pre -> -0.3% post).

Mechanism: bf16-wire's entire value was halving bytes across a SLOW PCIe link.
The PLX card removed the bottleneck, so the byte savings no longer cover the
gfx906 bf16 conversion cost. This is the same effect that made bf16s (scatter
only) a loss before — conversion cost is fixed, wire savings are conditional.

ACTION: remove GGML_TP_AR_WIRE=bf16 from the Mistral-Large fine-tune llama-swap profile.
f32 is the correct setting on the upgraded links.

### Link-width observation (unresolved)
sysfs reports all 3 GPUs at x16 @ 16 GT/s (Gen4), each upstream bridge also
x16 Gen4; no PLX device (vendor 10b5) visible in the container's PCI
namespace. The +8.9% is real and reproducible, but does not match an "x4"
report — possibly the x4 reading was the PLX->CPU uplink rather than GPU-side.

## 30. Phase 0a + Phase 1 of the kernel/compute plan (2026-09-22)

### 0a. Perf-level pin (high vs auto): WASH — closed
Host pinned all 3 GPUs high (sysfs; LXC sysfs read-only, host-owned). Clock log PROVED
auto was already ramping to the pin's clocks (1850/1850/1725 sclk) pre-pin; same-build
controls match pinned legs within noise (17.36 vs 17.26 tg128, 149.5 vs 148.9 pp2048).
tg32 17.41 / tg64 17.25 / tg128 17.26 — no bursty-request ramp signature. No host A/B needed.
NOTE: legs ran on build/ (non-RCCL branch build) so absolute levels are lower than §6 —
the §6 anchors (176.4/21.39 post-PLX) live in BENCH.md §34 with the build-rccl binary.
Perf pin stays (harmless, already host-persistent).

### 1. Gate-threshold retune: KEEP 131072 — closed
Sweep on build-rccl (hybrid / no-gate / RCCL-only, Mistral-Large fine-tune 3rk, pp8..pp2048 + tg32/64):
- pp8 (ne=98K, routed custom AR): 49.92 / 49.90 / 47.94 — gate correct, custom AR +4% over RCCL
- pp16 (ne=197K, routed RCCL): 66.12 / 66.90 / 66.21 — ONLY misrouted size; custom AR +1.2%
- pp32+: hybrid wins (86.89 vs 85.95 at pp32; 143.35 vs 136.89 at pp256)
- tg32/tg64: hybrid = no-gate (17.40 flat); RCCL-only -11% (15.55) — decode stays custom
Empirical crossover sits inside the current threshold's routing except pp16 (+1.2%).
Not worth a code change. Threshold 131072 (3rk) confirmed optimal-enough; retune closed.
Raw: /root/phase1-raw/leg1-10 on mi50. Threshold is compile-time (ggml-cuda.cu:1382,
per-rank-count constants); NO_GATE=1 is the only runtime knob.

## 31. W4A4 dot8 gauntlet: perf passes, correctness FAILS — DO NOT ADOPT (2026-09-22)

Full program (commits a4643ebc8 -> cd96ad6475 -> f5325c1e6 -> 1ec20cd54 on perf/w4a4-dot8-gfx906)
wired behind GGML_CUDA_W4A4_DOT8, gauntleted on Mistral-Large fine-tune 3rk, interleaved, flock-serialized:

| Leg | Gate | pp2048 | tg128 | PPL |
|---|---|---|---|---|
| anchor | off | 163.97 | 20.62 ± 0.04 | — |
| gate-on | on | 163.85 | **21.33 (+3.44%)** | — |
| drift ctrl | off | 163.84 | 20.58 (0.19% drift) | — |
| PPL | off/on | — | — | 1.0564 / 1.0564 (Δ=0.0000) |
| greedy primes | on | — | 21.0 t/s | **FAIL: 100x token id 0, empty** |

Verdict: perf passes (+3.44%/+3.65% vs both controls, ΔPPL=0.0000) but greedy gate FAILS —
gate-on decode emits garbage (token id 0 x100). The +3.4% is speed on a broken path.

Root-cause localization (bonus harness): u4_quant_check FAILS on ALL multi-channel cases
(ne>1: 1024x8x2, 5120x16x2, 4100x3; 97,054 mismatches) and PASSES single-channel ->
broken multi-channel index mapping in the u4 path (quantize kernel interleaved-vs-deinterleaved
or the xs stride math in dispatch). PPL structurally blind (pp-only; bug is decode-only).

Program value: dp8a codegen verified (32x v_dot8_i32_i4), +3.4% tg achievable IF the
multi-channel mapping is fixed. The fix is localized and the harness gives exact
reproduction cases. Deferred pending decision: fix-forward (targeted debug dispatch)
or park as documented negative with the fix path recorded.

## 32. W4A4 fix-forward: all gates PASS — ADOPT-CANDIDATE (2026-09-22)

Root cause was THREE interacting bugs (fix commit 88bbc3572, +55/-25, 3 files):
1. quantize.cu ib formula transposed activation blocks for ne1>1 (token-major fix);
   gridDim-derived ib0 also gapped on ne0%512.
2. pair-interleaved nibble packing mismatched canonical Q4_0 bytes -> sdot8 paired
   permuted values. Device kernel + host reference both de-interleaved now.
3. nib_sum_u4(pk)-256 computed UNSIGNED: raw excess-8 sums (~240 avg) wrapped to ~2^32,
   injecting garbage -> the token-0 decode. Fixed: signed cast.
Plus the dp8a operand-convention correction (MISMATCH hardware probe: hw pairing differs
from doc assumption; probe-verified before the fix).

Gauntlet v2 (build 88bbc3572, Mistral-Large fine-tune 3rk, flock+perflevel-high+bench-logged, r=5):
| Leg | pp2048 | tg128 |
|---|---|---|
| gate-off | 163.88 | 20.62 ± 0.04 |
| gate-ON | 163.85 | **21.20 ± 0.05 (+2.81%/+3.01%)** |
| drift ctrl | 163.44 | 20.58 (0.19%) |
PPL: 1.0564 both (Δ=0.0000). Greedy gate-ON: primes 2..29 EXACT + coherent tides.
Harness: OVERALL OK, 0 mismatches all shapes (4100x3 was 6472).

DECISION RULE: PASS all three gates -> ADOPT-CANDIDATE.
Deploy path (not done): merge perf/w4a4-dot8-gfx906 into the production build
(build-rccl) + set GGML_CUDA_W4A4_DOT8=1 (global or per-Mistral-Large fine-tune entry). PP cost
of uncached u4 quant: not visible (pp2048 within noise). GEMM (mmq_gemm_repacked)
extension = future work, decode win banked first.

## 33. W4A4 MoE extension: correct but -5.3% tg — REJECT for MoE (2026-09-23)

Milestone 4 (commit ad123ecdc): wired u4 path into MoE narrow-batch dispatch
(mul_mat_vec_repacked_id1 + mul_mat_vec_rp, single-token) — Q4_0 MoE never reaches the
fused epilogue (Q4_0 absent from both fusion admission sets), so unfused wiring = stock routing.
Caught a unit bug pre-deploy: MoE column offsets ids_src1[a]*xs must be in 20B u4 units.

Ornith-1.5-35B-A3B Huihui-Q4_0 gauntlet (2rk, interleaved off/on/off, r=5):
| arm | pp2048 | tg128 |
|---|---:|---:|
| off-1 | 1835.97 | 110.17 |
| ON | 1829.67 | **104.28 (-5.3%)** |
| off-2 | 1827.94 | 109.78 |
PPL 2.0643 both (d=0.0000); greedy primes exact (n=400 for thinking model).

Verdict: tg fails >=+2% -> REJECT for MoE. Why: single-token MoE rides mul_mat_vec_rp
at LANES=16/NCOLS=1 — the per-block nib_sum_u4 SUMW recompute is consumed once per block
(no RPL=2 amortization like the dense winner). Fix paths if revisited: precompute SUMW
at repack (+2B fp16/block) or u4-ize the fused gate epilogue. Dense m3 remains the
adopt-candidate; MoE env flag should stay OFF in any deploy.

## 34. W4A4 GEMM gauntlet: +32% pp2048, ADOPT-CANDIDATE (2026-09-23, post-reboot rerun)

G3 fix verified committed (05bae03af: wiring + u4 quantizer stride fix) before the host restart; only the gauntlet was missing. Rerun complete (build-w4a4, Mistral-Large fine-tune 3rk, flock, perflevel high survived reboot):

| Leg | pp2048 | PPL |
|---|---|---|
| gate-off (a) | 157.40 ± 0.28 | 1.0564 ± 0.00352 |
| gate-ON (b) | **207.92 ± 0.30 (+32.1%)** | 1.1035 ± 0.00525 |
| gate-off (c) | 157.20 ± 0.27 | — |
| greedy primes gate-on | EXACT 2..29 + tides | |

DECISION: pp +32.10%/+32.26% (>= +2% x2) PASS; dPPL +0.0471 <= +0.05 PASS but within 6% of the limit (activation 4-bit cost is real at prefill scale - the 2-chunk smoke 1.0185 flattered it); primes PASS. -> ADOPT-CANDIDATE.

Caveats: (1) dPPL is the binding constraint - do not stack further u4-path changes without re-gauntlet; (2) binaries report G2 SHA (build-metadata lag) but gate toggles +32% pp AND the PPL delta, wiring demonstrably live; (3) GEMM covers prefill only - MoE stays REJECTED (§33).

Full program: W4A4 = dense decode +2.8-3% tg + prefill +32% pp (both PPL-gated), MoE reject. Deploy = merge perf/w4a4-dot8-gfx906 into build-rccl + per-model GGML_CUDA_W4A4_DOT8=1 on Q4_0 dense entries (NOT MoE).

## 35. 175W re-baseline + long-context matrix — COMPLETE (2026-09-24)

Envelope: 175W/card (upp, persists), ALL models 3-rank -sm tensor (topology standardized),
binary build-w4a4 (full stack). Prior §30-34 numbers INVALID (power + topology changed).
Matrices: Mistral-Large fine-tune@64k, Mistral-Large fine-tune@32k, Qwen-Q4_0/Q8_0+Ornith@131k.

### Mistral-Large fine-tune 123B (3rk, 175W) — gate off / on
pp2048: 157.70 / 207.72 (+31.8%) | pp8192: 125.98 / 186.14 (+47.7%) | pp16384: 94.67 / 163.75 (+72.9%!)
tg@8k: 20.47 / 21.17 (+3.4%) | PPL@8k: 1.0723 / 1.1504 (d=+0.078, OVER the +0.05 gate at depth!)
greedy@64k-depth: primes exact 2-29.

### Mistral-Large fine-tune 128B (3rk, ctx 32768)
pp2048: 154.87 / 199.62 (+28.9%) | pp8192: 142.67 / 179.69 (+25.9%) | tg@8k: 19.69 / 20.22 (+2.7%)
PPL@8k: 1.5436 / 1.7339 (d=+0.190 — OVER the +0.05 gate!)

### Qwen3.8-27B (3rk, ctx 131072)
Q4_0: pp2048 529.77 / 630.72 (+19.0%) | pp8192 511.56 / 606.32 (+18.5%) | pp16384 486.65 / 571.91 (+17.5%)
tg@131k-depth: 52.18 / 52.53 (+0.7% — flat) | PPL@8k: 5.0055 / 5.7254 (d=+0.72!! OVER gate)
Q8_0 control: pp2048 594.27, pp16384 540.39, tg@131k 44.26, PPL@8k 4.3910
NOTE: Q8_0 is FASTER at pp than Q4_0+W4A4 (594 vs 630 close, but 571 vs 606 at 16k) — dequant tax analysis revisited.

### Ornith 35B MoE (3rk, ctx 131072)
pp2048: 1808.46 / 1929.73 (+6.7%) | pp8192: 1736.56 / 1848.04 (+6.4%) | tg@131k: 103.99 / 98.27 (-5.5%, regression persists)
PPL@8k: 1.9113 / 2.2150 (d=+0.30 — OVER gate)
greedy@131k: primes 2..23 listed (29 truncated by -n 100 budget on thinking model)

### KEY FINDINGS
1. W4A4 pp win GROWS with depth on Mistral-Large fine-tune: +32%/+48%/+73% at 2k/8k/16k. The 642-instr
   GEMM loop's dequant tax dominates harder as KV grows.
2. BUT: PPL-at-depth delta BLOWS the +0.05 gate on all models (Mon +0.078@8k, Beh +0.190,
   Qwen +0.72). c=2048 audits (d=0.0000/+0.0471) were measured at shallow context; at
   8k+ context the activation-4bit cost is REAL and grows with depth.
3. Ornith@3rk: decode regression persists (-5.5%), pp gains +6.4-6.7% — still net reject
   for tg, but pp win is real for MoE prefill.
4. Qwen Q8_0 control: pp2048 594 vs Q4_0+W4A4 630 — Q4_0+full-stack now BEATS Q8_0 prefill
   on this rig (was: Q8 won prefill per findings §3).
5. Greedy-at-depth: correct on all models tested (primes listed correctly at 32k-64k depth).

## 36. C1/B8 milestone M3: per-8 u4p decode — REJECT on perf; depth-PPL lens corrected (2026-09-24)

Commit 59eefc51f wired GGML_CUDA_W4A4_DOT8=2 (per-8 u4p) into dense decode dispatch.
Gauntlet (Mistral-Large fine-tune 3rk, 175W, interleave gate 0/1/2, r=3):

| Leg | gate-0 stock | gate-1 per-32 | gate-2 per-8 |
|---|---:|---:|---:|
| pp2048 | 157.70 | 203.25 (+29%) | 157.65 (0%) |
| pp8192 | 144.97 | 185.95 (+28%) | 144.70 (0%) |
| pp16384 | 130.87 | 163.53 (+25%) | 130.62 (0%) |
| tg64@8k | 20.47 | 21.20 (+3.6%) | **18.58 (-9.2%)** |
| PPL@8k | 1.0751 | 1.1572 (+0.082) | 1.0751 (0.0000) |
| PPL@2k | 1.0564 | — | 1.0564 (0.0000) |

VERDICT: REJECT for decode (tg -9.2% blows the <=1% cost rule; the 27-VGPR spill +
4x epilogue of the per-8 kernel is real cost, matching design §8's risk register).

CRITICAL LENS CORRECTION: gate-2's PPL deltas of 0.0000 are STRUCTURAL BLINDNESS,
not quality proof — perplexity is prefill/GEMM-dominated, and u4p was wired
decode-only (pp identical to stock confirms it never touched GEMM). The +0.082
depth-PPL cost we set out to fix lives in the GEMM path (gate-1: GEMM u4 wired,
PPL moves; gate-2: decode only, PPL frozen). Same lesson as §31, re-confirmed.

C1 REFRAME: per-8 recovery, if wanted, belongs in M4 = GEMM u4p (prefill side),
where the depth-PPL cost is actually created. Decode stays per-32 (its +3.6% tg
and PPL-neutrality at 2k stand). Greedy@64k gate-2: output-file capture truncated
(empty after 'Assistant:') - noted, not re-run given the perf verdict.

## 37. C1/B8 milestone M4 complete: GEMM per-8 quality recovery CONFIRMED, perf cost REAL (2026-09-25)

Fix commit 2e8499bb6 (GEMM u4p + best-of gate=2: decode per-32, GEMM per-8).
Post-PSU completion run (2 crashes interrupted earlier attempts):

| Model / leg | gate-0 stock | gate-1 per-32 | gate-2 per-8 GEMM |
|---|---:|---:|---:|
| Mistral-Large fine-tune PPL@8k | 1.0751 | 1.1572 (+0.082) | 1.0943 (+0.019) - 75% recovery |
| Qwen PPL@8k | 4.9847 | 5.7148 (+0.730) | 5.1380 (+0.153) - 79% recovery |
| Ornith PPL@8k | 1.9525 | 2.2612 (+0.309) | 2.0157 (+0.063) - 80% recovery |
| Mistral-Large fine-tune pp2048 | 154.79 | 205.96 (+33%) | 98.11 (-37%!!) |
| Mistral-Large fine-tune pp8192 | 144.16 | 185.19 (+29%) | 93.09 (-35%) |
| Qwen pp2048 | 530.07 | 629.54 (+19%) | 389.36 (-27%) |
| Qwen tg@8k | 52.59 | 52.59 | 52.42 (flat, decode=per-32 as designed) |
| Mistral-Large fine-tune tg@8k | 20.47 | 20.48 | 20.85 (identical class) |
| Mistral-Large fine-tune greedy@64k gate2 | - | - | primes EXACT 2..29 + tides |

KEY: 1) QUALITY THESIS CONFIRMED - per-8 recovers 75-80% of depth-PPL on ALL models.
2) PERF: per-8 GEMM as implemented REGRESSES prefill hard (-27..-37% vs stock) -
epilogue/LDS overhead costs more than dp8a saves on GEMM tile shapes.
DEPLOY SHAPE: gate=1 for RP models at RP contexts; gate=2 = long-context-quality
mode if Qwen/Ornith entries need it; stock for everything else; MoE decode stock.

## 38. W4A4 step 2 (spill fix): hot-loop spills ELIMINATED, pp recovery partial (2026-09-25)

Profile-guided fix (60c627733): U4P8 GEMM scale handling restructured -
spills 160 -> 13 dwords (hot loop ZERO; remaining 13 cold checked-edge only),
w32 52 -> 2. Root cause of M4 blowup: forced #pragma unroll on kk slab loop
quadrupled static dot8s (1024) causing the spill storm; rolled form restores
256. Plus SW packing: 4 per-8 SW_k fit int8 (-64..+56), packed 1 uint32,
cuts 3 VGPRs. Re-gauntlet: pp2048 116.58 (was 98.4, +19% recovered), pp8192
109.26, PPL@8k 1.0943 (+0.019 preserved EXACTLY), greedy@64k primes EXACT.
Remaining -26% vs stock is STRUCTURAL: 4 unchained accumulators + 8 LDS scale
reads per block = epilogue-FMA-bound, inherent to per-8. C1 closes: per-8
GEMM = documented quality-mode trade, not deployable universal mode.
NOTE: step2 commits 60c627733/2570c9a5a were on-box only; pushed separately.

## 42. C1 follow-up: H-form per-8 epilogue rework - REJECT; per-8 frontier now fully mapped (2026-09-27)

Branch `perf/w4a4-epilogue-h` (results: docs/w4a4-epilogue-h-results.md on that branch). Hypothesis:
the affine u4p epilogue cost (4 FMA m-term + sWs8 LDS array + SW unpack) could
be deleted by re-parameterizing activations as hierarchical per-8 (one per-32
center c32 + symmetric per-8 signed scales d4[k], dot = SUM d4*dp8a + c32*SW32).
H1 CPU seal PASS 144/144 maxrel 2.5e-4 after fixing the host reference to the
device's strided per-8 groups (grouping is pinned by the nibble word layout -
same M3b lesson, third occurrence). H3 codegen PASS (256 static dot8, spills
within step2 bounds).

| Metric | gate0 stock | gate1 per-32 | gate2 H-form | affine per-8 ref |
|---|---:|---:|---:|---:|
| pp2048 | 157.43 | 208.30 (+32%) | 142.65 (-9.4%) | 116.58 (-26%) |
| pp8192 | 145.02 | 186.42 | 132.19 (-8.8%) | 109.26 |
| PPL@2k | 1.0564 | 1.0564 (0) | 1.0994 (+0.0430) | 1.0564 (0) |
| PPL@8k | 1.0751 | 1.1572 (+0.082) | 1.1359 (+0.0608) | 1.0943 (+0.019) |
| Qwen PPL@8k | 4.9847 | 5.7148 (+0.730) | 5.5813 (+0.597) | 5.1380 (+0.153) |
| tg128@8k | 20.45 | 21.20 | 21.22 (decode=per-32, flat) | 20.85 |
| greedy@64k | - | - | FAIL: degenerate - only newlines after "Assistant:", deterministic across retry | primes EXACT |

ADOPTION BAR vs RESULT: pp within 10% of stock = PASS (-9.4%); PPL recovery
>= 60% of gate1 cost = FAIL (Mistral-Large fine-tune 26%, Qwen 18%); greedy@64k exact = FAIL.

VERDICT: REJECT. H-form is dominated: slower than stock AND worse quality than
stock (PPL +0.061@8k), and it pays a PPL cost already at 2k (+0.043) where
affine is exact - the coarser level set (7-15 symmetric levels around a block
center vs affine's 16 asymmetric levels per group) costs real quality, and at
64k the model collapses into newline loops on the probe prompt. The epilogue
thesis is VALIDATED (pp recovered -26% -> -9.4% by deleting the m-term/LDS
machinery), but the quality price disqualifies it for the only role gate-2
has (long-context quality mode).

C1 CLOSE-OUT: the per-8 frontier is now mapped end to end -
  stock (pp 100%, PPL base) -> H-form (pp 91%, PPL +0.061, depth-collapse)
  -> affine per-8 (pp 74%, PPL +0.019, exact) -> per-32 u4 (pp +32%, PPL +0.082).
Affine per-8 remains the quality corner and stays the shipped gate-2; per-32
remains the speed mode (gate-1); H-form is parked as a documented frontier
point. No further per-8 epilogue work is warranted; the remaining depth-PPL
cost of gate-1 has no cheaper fix on this branch.

## 39. MTP x W4A4 on Qwen3.6 Native-MTP: lossless FAIL, root cause IDENTIFIED (2026-09-25)

Model: Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q8_0 (29GB), 3rk.
This was the documented-good MTP head (+16% pre-W4A4-era). Result: REJECT.

| Arm | tg128 median | acceptance | lossless |
|---|---:|---|---|
| baseline | 45.64 | - | - |
| strict MTP n6 | 49.07 (+7.5%) | 0.73-0.78 greedy | FAIL |
| MTP+W4A4 gate1 | 43.89 (variance) | - | arm INVALID (Q8_0 target, gate needs Q4_0) |

Lossless: greedy MTP output differs from no-MTP control (determinism control
passed: no-MTP byte-identical x3; divergence MTP-induced in reasoning stream).
ROOT CAUSE (same on BOTH MTP models): server logs "spec common_specu: backend
offload failed for seq_id=0..3; using CPU sampler" x4 per MTP leg, 0 on
baseline - draft sampler falls back to CPU under -sm tensor, verify is not
strict-greedy-equivalent in this build. FORK BUG, not model-specific.
Per-position acceptance n6: 0.75/0.53/0.30/0.21/0.18/0.15 - head drafts well,
infrastructure breaks it. W4A4xMTP stacking test needs Q4_0 native-MTP model.
CLOSED: fork-bug-blocked (draft sampler offload under tensor). Revisit if
upstream fixes offload path.

## 40. W4A4 DEPLOYED to production: Mistral-Large fine-tune + Mistral-Large fine-tune (2026-09-26)

Deploy sequence:
1. build-rccl rebuilt on perf/w4a4-dot8-gfx906 (includes upstream merge + full W4A4
   family + MTP guard). Verified: 231.6 pp2048 / 21.13 tg64 gate=1 pre-deploy.
2. Per-entry env (llama-swap env: list format) added to Mistral-Large-123B fine-tune-v2 and
   TheDrummer-Mistral-Large-123B fine-tune-v3: GGML_CUDA_W4A4_DOT8=1 (REPACK inherited from
   start.sh global env but also set per-entry for belt-and-suspenders).
3. Config format lesson: llama-swap env field is a LIST of KEY=VALUE strings,
   not a map (map form: "cannot unmarshal !!map into []string" crash). Inline
   env tokens in cmd: FAIL (llama-server sees "env" as unknown arg). The list
   field is the correct mechanism.
4. Live verified: server process /proc/PID/environ carries both vars; chat
   completion primes exact (2..29) + coherent tides at 82 tok.

All other entries (Qwen Q4_0, Q8_0, Ornith, MoE) remain stock — flag off.
Rollback: config backup bak-w4a4deploy-20260926; binary revert = rebuild from
pre-merge commit on build-rccl.

## 41. Post-upstream-merge floor (e4f26e4cb) - NEW BASELINE, supersedes §35 absolute values (2026-09-27)

Envelope: identical to §35 (175W/card, 3rk -sm tensor, same datasets/scripts; only change = merged binary,
ggml lib 0.22 -> 0.25.3). Full raw logs: /root/rebase-merged-{big,small}.{txt,log}.

### Mistral-Large fine-tune 123B (ctx 65536)
pp2048: 170.45 / 232.80 (+36.6%) | pp8192: 155.45 / 206.00 (+32.5%) | pp16384: 139.18 / 178.97 (+28.6%)
tg64@64k: 20.43 / 21.08 (+3.2%) | PPL@8k: 1.0722 / 1.1530 (d=+0.0808)
greedy@64k: primes exact 2-29 + coherent tides sentence.

### Mistral-Large fine-tune 128B (ctx 32768)
pp2048: 167.19 / 222.65 (+33.2%) | pp8192: 152.85 / 197.94 (+29.5%)
tg64@32k: 19.61 / 20.18 (+2.9%) | PPL@8k: 1.5554 / 1.7329 (d=+0.1775)
greedy@32k: primes exact 2-29.

### Qwen3.8-27B (ctx 131072)
Q4_0 shallow: pp2048 615.00 / 777.13 (+26.4%) | pp8192 589.84 / 738.55 (+25.2%) | pp16384 556.31 / 687.44 (+23.6%) | tg128 52.56 / 52.56 (flat)
Q4_0 @131k: pp2048 198.92 / 219.04 (+10.1%) | pp8192 208.96 / 225.14 (+7.7%) | pp16384 213.98 / 230.97 (+8.0%)
tg64@131k: 34.75 / 34.70 (-0.1%, flat) | PPL@8k: 4.9912 / 5.7102 (d=+0.7190, over gate as before)
Q8_0 control: pp2048 724.20, pp16384 645.28 | tg64@131k 30.70-31.99 | PPL@8k 4.3902

### Ornith 35B MoE (ctx 131072, Huihui imatrix Q4_0)
shallow: pp2048 1972.99 / 2139.33 (+8.4%) | pp8192 1887.32 / 2038.09 (+8.0%) | tg128 104.61 / 98.11 (-6.2%)
@131k: pp2048 540.99 / 552.28 (+2.1%) | pp8192 551.79 / 564.29 (+2.3%) | tg64: 89.84-91.22 off / 85.37-86.18 on (-5.8%)
PPL@8k: 1.9070 / 2.2064 (d=+0.2994, over gate as before)
greedy@131k budget-cut mid-think but primes 2-29 visible in trace.

### KEY FINDINGS vs §35
1. Everything faster: pp +7-47% vs §35 across the board (upstream graph/fusion/allocator wins compound with W4A4).
2. W4A4 pp gain GREW: +36.6% (Mon) / +33.2% (Beh) at pp2048 vs §35 +31.8/+28.9. The dequant-tax analysis holds.
3. Quality UNCHANGED: PPL deltas match §35 within noise (Mon +0.081 vs +0.078, Beh +0.178 vs +0.190, Qwen +0.719 vs +0.72, Orn +0.299 vs +0.30); all greedy seals exact.
4. Ornith decode regression persists (-6.2% tg shallow with W4A4 on) - MoE W4A4 stays REJECT; gate stays OFF for MoE entries.
5. Qwen shallow pp16384: Q4_0+W4A4 (687) now BEATS Q8_0 (645) - inversion vs §35 era; at 16k@131k depth W4A4 (231) also beats Q8_0 (226).
6. Verdict: upstream merge is a strict win. Adopt e4f26e4cb as the new production floor.

## 42. RVN-Qwen3.8-Flash-Next (qwen4exp MoE): prod-stack A/B + W4A4 check (2026-09-29)

Model: 0bserverx RVN-Qwen3.8-Flash-Next-Abliterated-Uncensored, Q4_0, 8 shards (94G).
Architecture qwen4exp: 48 layers, 512 experts @ 10 active, PLE n-gram embedding table
(per_layer_token_embd.weight, 27465 MiB) host-resident mmap, demand-paged (lazy read, no VRAM).
Rig: 3x MI50, -sm tensor, -c 32768, 175W. Server bench via /completion, greedy temp=0.
Prefill = two distinct ~11.3k-token prompts per config; decode = 256 tok @ 56-token prompt.
Single server per config + 2 passes; prefill spread +/-8% (page-cache warmth), decode is the
decision-grade number.

| Config | prefill pass1/pass2 | decode run1/run2 |
|---|---|---|
| A: repack ON, custom AR OFF | 770 / 809 t/s | 51.5 / 52.1 t/s |
| B: prod stack (repack + custom AR + skip-xbar) | 766 / 913 t/s | 56.7 / 57.3 t/s |
| C: B + GGML_CUDA_W4A4_DOT8=1 | 817 / 975 t/s | 51.9 / 54.0 t/s |

### FINDINGS
1. Custom AR = +10% decode (52.1 -> 57.3 avg); prefill unchanged (twoshot targets decode-sized tensors, as in section 35-era data).
2. MoE W4A4 REJECT reproduced on qwen4exp: decode -5.4% vs B (54.0 vs 57.3). Matches Ornith verdict (-5.3%, SUMW recompute per block, no amortization). Gate stays OFF for all MoE entries.
3. NEW: W4A4 PREFILL is a WIN on MoE: +6.8% vs B (975 vs 913 avg pass2; pass1 also up: 817 vs 766). Multi-token batches amortize the per-8 epilogue where single-token decode cannot. A batch-size-gated W4A4 dispatch (W4A4 when ne>1, stock decode path when ne==1) is the candidate if a MoE prefill mode is ever wanted. Not implemented; decode rules RP workloads.
4. PLE demand-paged table costs ~nothing warm: warm prefill 913-975 t/s at 11k tokens on a 27.5G table that never enters VRAM or RSS (server RSS ~4.3G).
5. BUG: fully-stock config (GGML_CUDA_REPACK_Q4_0=0) HANGS at warmup on qwen4exp under -sm tensor 3-GPU. Two deterministic repros (10+ min after buffer alloc, no warmup progress, 0% GPU). Repack is functionally REQUIRED for this arch, not just faster. Root cause not investigated.

Prod stack confirmed running on the Flash-Next server (repack Meta buffers + custom AR twoshot F32 3-rank); W4A4 gate OFF.

## 43. RVN Flash-Next: prefault + ub4096, W4A4 gate-mode-3, MTP draft (2026-09-29)

Server-bench followups on the qwen4exp MoE (same rig/flags as section 42 unless noted).

### 43a. LLAMA_PLE_PREFAULT=1 + -ub 4096 (config D vs section 42 config B)
Prefault touches the full 27465 MiB lazy PLE table at load: 43.7 s, 8 threads, zero VRAM (page cache only).
| metric | B (base) | D (prefault + ub4096) | delta |
|---|---|---|---|
| prefill pass1 (cold table) | 766 t/s | 941 t/s | +23% |
| prefill warm passes | 913 t/s | 993-997 t/s | +6-9% |
| decode | 57.3 t/s | 57.4 t/s | - |

Cold-start penalty fully eliminated: pass1 no longer lags pass2. Prefault is recommended for
any llama-swap entry serving a qwen4exp/PLE model (fixes first-request slowness after TTL reload).

### 43b. W4A4 gate mode 3: batch-gated dispatch (commit 3b1079b84, branch perf/w4a4-dot8-gfx906)
Design: u4act = mode==3 && activations_rows > 1 (mul-mat.cu: src1->ne[1]; mul-mat-id.cu: ids->ne[1]).
W4A4 kernel for multi-token GEMMs (prefill/batch), stock repack path for single-token decode.
Modes 1/2 byte-identical to prior behavior.
Seals: first-token identity off==m1==g3 PASS (mechanism verified); greedy primes exact in trace PASS;
byte-identity vs off diverges as expected for any W4A4 prefill (same standard as modes 1/2).
| config | pp (3-run mean, 6k-byte prompt) | tg (3-run mean) |
|---|---|---|
| off | 192.4 t/s | 47.57 t/s |
| mode 3 | 197.3 t/s | 47.53 t/s |
| delta | +2.5% | -0.1% (neutral) |
Decode fully protected (no SUMW tax, vs -5.4% always-on on MoE, section 42). pp gain diluted vs the
+7% server-bench observation: expert GEMMs run w32 tiles with per-expert column counts, so the dense
W4A4 prefill win shrinks on MoE-dominant workloads. Large-prompt diag (~7.1k tok): g3 429.8 = m1 423.8
> off 401.0 t/s (full prefill engagement confirmed). Mode 3 = correct default for dense entries that
want W4A4 prefill without decode exposure; on MoE entries it is safe but low-value.

### 43c. MTP draft (drluoto Qwen3.8-Flash-Next-MTP-GGUF, Q8_0 4.1G draft, llama-cli 3rk)
| config | tg @ temp 0.1 | greedy tg |
|---|---|---|
| base | 47.3 / 49.3 t/s | 51.5 t/s |
| MTP | 57.3 / 53.4 t/s | 64.1 t/s |
| MTP + MTP_SAFE | - | 63.6 t/s |
MTP = +14.7% avg tg (ABOVE the +10% bar, unlike Qwen3.6 Native-MTP +7.5%). SAFE guard costs ~0 here
(vs -10.4% on Ornith). LOSSLESS: FAIL for both plain and MTP_SAFE - greedy thinking traces diverge
from base (respond/answer word-level divergence, same set_sampler CPU-fallback signature as section
39; NOT model-specific). Final response text converged by chance in one seal; treat as non-lossless.
Deployed 2026-09-29 as llama-swap entry RVN-Flash-Next-MTP with GGML_TP_AR_MTP_SAFE=1 +
LLAMA_PLE_PREFAULT=1 (accepting trace divergence, same regime as the other MTP entries).
Methodology note: an initial 'LOSSLESS PASS' verdict from tail-compare was an artifact of differing
timing footprints; seals must compare response text with timing lines stripped, and a full-trace
compare is required for the lossless claim (the tail-only pass hid a diverging thinking trace).

## 44. MTP sampler guard T1 fix + qwen4exp stock-load root cause (2026-09-30)

### 44a. set_sampler T1: capability test replaces hard refusal (fix/mtp-sampler-guard-t1, f83423ec2)
Background: the guard is upstream-identical (PR #23287, deliberate CPU fallback under -sm tensor).
Investigation (mtp-sampler-bug-analysis.txt) designed a tiered fix; T1 implemented:
device sampling allowed iff logits row replicated (tensor_mirror_output / small dense DFLASH
drafter) or TP width 1 (get_split_state_ud-derived); CPU fallback kept for vocab-parallel heads;
one-shot warning suppression removed.

Seals (greedy, CUSTOM_AR=0, n=768, response-anchored compare, determinism control IDENTICAL):
| pair | verdict | guard state |
|---|---|---|
| DFlash2 base vs MTP | DIFFERS (near-tie wording) | 0 warnings: device sampling ENGAGED |
| RVN Flash-Next base vs MTP | DIFFERS (list formatting) | 0 warnings: device sampling ENGAGED |
| RVN perf @8k | base 36.5 vs MTP 59.1 t/s gen (+62%) | no regression |

CLASSIFICATION: both pairs diverge with device sampling engaged AND custom AR off. 43c had RVN
diverging with CPU sampling too. So the losslessness problem for RVN MTP and DFlash2 is the
upstream #25618 batch-invariance class (parallel verify kernels not bitwise-identical to serial
decode; near-tie flips), NOT the sampler guard. The investigation prediction (guard fix = DFlash2
fix) is REFUTED. Qwen3.6/Ornith remain the custom-AR reordering class (43c/44: AR=0 restores identity).

Verdict on T1: correct and worth keeping - scopes the guard exactly where backend argmax is safe
and removes the per-draft-token full-vocab logits D2H for mirrored/single-lane drafts - but it is
NOT a losslessness fix on this rig. Spec-decode losslessness on -sm tensor stays blocked on the
upstream #25618 class. Full data: /root/mtp-t1-fix-results.txt.

### 44b. qwen4exp stock-Q4_0 "hang" RESOLVED: mmap loader pathology, -lm dio avoids it
User catch: every repro ran the default mmap loader; production entries use -lm dio (direct I/O).
Identical config (GGML_CUDA_REPACK_Q4_0=0, custom AR off, -sm tensor, same model/flags):
- mmap: never completes within observation windows (90+ min). /proc counters during the "hang":
  read_bytes 61.7->80.3 GB over 60 s (312 MB/s sustained), majflt +80/s, RSS +1.1 GB - reads
  outpace residency ~100:1: the meta set_tensor 2D-copy pattern over mmap'd source pages thrashes
  the page cache instead of retaining data.
- -lm dio: load complete + listening in ~14 SECONDS (hang-dio-test.log).
Root cause: interaction of the mmap loader with meta-buffer set_tensor 2D copies on this tensor
geometry (small odd-sized lane copies, e.g. row=126/108 at pitch 360 for the 3-lane axis-0 ffn
split; measured 0.4-5.8 GB/s per-copy from cold mmap vs 5.3-8.9 GB/s aligned/pinned - bench files
memcpy2d_bench*.log). NOT a deadlock, NOT repack-dependency (repack merely takes a different
staging path), NOT qwen4exp-specific quant handling. Mistral-Large fine-tune Q4_K_M stock tolerance under mmap
remains unexplained (probable: its tensor shapes avoid the tiny-copy regime) - open, low priority
since production uses -lm dio everywhere.
Actions: -lm dio added to the RVN-Flash-Next-MTP llama-swap entry (consistency with the rest of
the fleet). Optional real fix (not implemented): staging/bulk-copy for mmap sources in the meta
set_tensor path.

### 44b-addendum: full mechanism of the mmap crawl (qwen4exp-hang-analysis.txt, 2026-09-30)
Final agent analysis (hang_final8 + raw-stack pointer resolver; note prior frame-15 $rsi captures
were invalid, offsetof(ggml_tensor,name)=256). Corrections to 44b: VRAM "growth" during the stall
was the buffer-alloc phase - VRAM is flat during the copy stall; the stuck point MOVES (~10
layers/60 s: it is a crawl, not a deadlock; stuck tensors observed: blk.25/35 ffn_up_exps /
ffn_gate_exps, the AXIS_1 strided path).

Three compounding costs in llama_model_loader::load_all_data mmap branch ->
ggml_backend_meta_buffer_set_tensor (per-lane 2D copies -> ggml_backend_cuda_buffer_set_tensor_2d,
cudaMemcpy2DAsync + cudaStreamSynchronize PER CALL):
1. strided DMA from cold pageable pages: 431 MB/s at the exact ffn_down_exps lane shapes
   (warm: 8.4 GB/s; 1D contiguous: 14.3 GB/s) - bench3
2. HIP pins+GART-maps the WHOLE pageable source span per call (hsaKmtMapMemoryToGPUNodes in the
   captured stack); 93 GiB working set vs 64 GiB RAM => ~26x disk read amplification
   (310 MB/s disk for ~12 MB/s useful)
3. per-call stream sync serializes map/fault/disk/DMA across all 3 lanes on one thread

Q4_K_M paradox resolved: Mistral-Large-123B fine-tune-v2 is dense llama-arch (no *_exps tensors, GGUF-verified)
and Q4_K 256-block cannot take the tiny-row path anyway (AXIS_0 granularity assert,
ggml-backend-meta.cpp:1234, for ne0=640). Exposure surface: block-32 quants (Q4_0/Q5_0/Q8_0/IQ4_NL)
on ffn_down_exps AXIS_0; the AXIS_1 strided path (gate/up_exps) hits ANY quant on 3D per-expert
MoE tensors. Lazy PLE table not implicated (CPU_Mapped).

Why repack/dio dodge it: repack buft nulls iface.set_tensor_2d -> per-row set through a persistent
PINNED staging buffer + ONE bulk async H2D + repack kernel (q8_repack/buffer.cu:524-599); dio uses
the same pinned-staging pattern (llama-model-loader.cpp:1706-1755).
Minimal fix (described, not implemented): in ggml_backend_cuda_buffer_set_tensor_2d, stage
non-pinned strided sources through a small reusable pinned buffer + one contiguous async copy,
sync deferred to end-of-tensor; or route GPU-bound mmap tensors through the dio upload_backend.

## 45. NVFP4 imatrix leg - the scale answer to the E2M1 wall (2026-10-01, branch perf/nvfp4-imatrix)

Question left by section 28b/28c (MXFP4+imatrix REJECT, PPL@8k 2.1626 vs Q4_0 1.9455, residual
blamed on the E2M1 mantissa wall): is the residual scale poverty (E8M0 per-32) or mantissa
poverty (2-bit)? Test: M1-style imatrix-aware NVFP4 quantizer - per-16-sub-block UE4M3 scale
search weighted by imatrix (window around ref amax/6 code), vs MXFP4's one E8M0 scale per 32.

### Build + seal
quantize_row_nvfp4_impl in ggml-quants.c; quantize_nvfp4 branches on quant_weights; NVFP4 ftype
wired in llama-quantize (dense 2D ne0%64 branch). CPU seal: no-imatrix bit-identical to ref;
synthetic weighted-error ratios 0.825-0.957; REAL Ornith expert data w_err 0.629 (37% error cut,
vs MXFP4's 5.7% - the finer scale grid actually bites on real weights). Commits ff31bc858 + 1f6be1d1a.

### Numbers (Ornith-35B, same harness as M3, Q4 control same-run)
| quant | PPL@8k | vs Q4_0 |
|---|---|---|
| Q4_0 (control, same run) | 1.9525 | - |
| Q4_0 (M3 ref) | 1.9455 | - |
| **NVFP4 + imatrix (this leg)** | **1.9778** | **+1.66%** |
| MXFP4 + imatrix | 2.1626 | +11.2% |
| MXFP4 no-imatrix | 2.4418 | +25.5% |

NVFP4+imatrix closes 85% of the gap MXFP4+imatrix left (2.1626 -> 1.9778). Greedy seal PASS
(primes 2-53 exact). GGUF: 4.54 BPW, 18.8G, token_embd overridden to mxfp4 (no NVFP4 CUDA
get_rows kernel), output.weight Q6_K prod-matched.

### Perf: non-starter on gfx906 today
NVFP4 has no repack fast path - generic MMQ: pp8192 972.5 vs MXFP4 1465.2 (-33.6%), tg128@8k
82.5 vs 106.5 (-22.6%), same session/build. A q8_repack/MoE-fusion port to the NVFP4 block
layout is the follow-up lever if the quality verdict ever justifies it.

### Verdict: REJECT for adoption (bar: PPL <= Q4_0). Science answer: the M3 "mantissa wall"
framing was too strong - scale granularity was worth ~0.18 PPL; the irreducible 2-bit-mantissa
floor is ~+1.7% over Q4_0 at this point size. 4-bit block formats (MXFP4, NVFP4) are both below
the RP adoption bar on quality AND (for NVFP4) speed on gfx906. The quantizer-side exploration
of this family is COMPLETE: MXFP4 raw, MXFP4+imatrix, NVFP4+imatrix all measured; no further
quantizer-side path remains (next rung would be mantissa width, i.e. not MXFP4/NVFP4).

## 46. Qwen3.5-9B upstream-data matrix (2026-10-01, branch bench/qwen35-9b-upstream-data)

Purpose: reviewer-reproducible data for the upstream PRs. Model bartowski/Qwen_Qwen3.5-9B-GGUF
bf16 (full-precision file in that repo; no F16), Q4_0 + Q4_K_M quantized locally with stock
llama-quantize. Dataset wikitext-2-raw-v1 test set (wiki.test.raw, scripts/get-wikitext-2.sh from
ggml-org/ci). Build 3b1079b84, 3x MI50, -t 16, -lm dio, PPL -c 512 -ub 512 --chunks 256 (KLD legs
--chunks 128). Full data: /root/qwen35-9b-upstream-results.txt, scripts committed on the branch.

### 46a. Accuracy vs F16 (single-GPU, no -sm tensor)
| quant | PPL | ratio vs F16 | KLD mean (5pctl/median/95pctl) |
|---|---|---|---|
| F16 (bf16) | 7.9504 | - | - |
| Q4_K_M | 7.9546 | 1.004 | 0.0577 (0.0004/0.024/0.162) |
| Q4_0 | 8.4203 | 1.054 | 0.0683 (0.0005/0.033/0.196) |

### 46b. Q4_0 repack identity (PR2 claim, verified)
Same Q4_0 file, GGML_CUDA_REPACK_Q4_0=1 vs 0: PPL 8.4203+-0.0858 vs 8.4205+-0.0858 (delta 2e-4,
uncertainty 8.6e-2); KLD 0.06832+-0.00168 vs 0.06897+-0.00179 (delta 6.6e-4). IDENTICAL within
noise; repack log line present only in ON runs. Repack is layout-only.

### 46c. Perf
Single-GPU Q4_0 repack on vs off (llama-bench -r 3, mean+-spread):
pp512 948.5+-4.1 vs 897.6+-4.1 (+5.7%) | pp2048 942.2+-0.5 vs 892.9+-0.3 (+5.5%)
tg32 76.0+-1.8 vs 72.5+-2.1 (+4.8%) | tg128 81.0+-0.2 vs 75.7+-0.1 (+7.0%)

3-GPU -sm tensor Q4_0 custom AR on vs off (GGML_ENABLE_CUSTOM_AR):
pp512 1523.6+-0.7 vs 1332.5+-2.2 (+14.3%) | pp2048 1509.5+-0.1 vs 1318.1+-1.0 (+14.5%)
tg32 124.5+-5.3 vs 81.8+-6.0 (+52.2%) | tg128 145.6+-1.7 vs 87.2+-0.9 (+66.9%)

PR mapping: PR1 (nranks-3 twoshot AR) = 46c second table; PR2 (Q4_0 repack entry) = 46b identity
+ 46c first table. Note: llama-bench -dev does not propagate to TP device selection; 3-GPU legs
used default all-GPU selection.

## 47. W4A4 accuracy on Qwen3.5-9B: KLD unavailable (kernel crash at small ubatch), PPL@8k +2.66% (2026-10-01)

Purpose: extend the upstream-data matrix (section 46) with W4A4 gate=1 accuracy. Reused the 9B
F16 logits base (f16-c512-ch128.logits) + wikitext-2. Single GPU, -lm dio, w4a4-env environment.

### 47a. Harness verified
Stock Q4_0 KLD vs F16: KLD 0.068316 +- 0.001681 - EXACTLY reproduces the section 46 agent number
(0.068316). Harness and base file valid.

### 47b. FINDING: W4A4 gate=1 crashes at c512-class prefill shapes
llama-perplexity with GGML_CUDA_W4A4_DOT8=1 aborts (rc=134, ROCm illegal memory access in
mmq_gemm_repacked) at c512 -b 2048 -ub 512, deterministic, independent of --kl-divergence
(reproduces with plain PPL c512) and of --chunks (128 and 32). SAME binary that ran the
section 46 matrix (which did not enable W4A4). Plain PPL at c8192 -ub 2048 runs fine, and the
W4A4 PPL delta is deterministic across sessions (7.9541 reproduced exactly).
Shape boundary: production entries run -ub 2048 (M=2048 ubatch GEMMs) + decode (M=1), both fine.
c512/-ub 512 (M=512 prefill GEMM) crashes. Not KLD-specific, not chunk-count-specific: it is the
ubatch M-dimension. Suspected u4act W4A4 kernel tile-shape hole at M=512 on Q4_0 9B shapes;
not root-caused yet. KLD with W4A4 is therefore UNAVAILABLE at c512 (the KLD base is c512).
Action items: (1) reproduce standalone with llama-mmap-perplexity-ish minimal M=512 GEMM,
(2) root-cause in mul-mat.cu u4act dispatch, (3) add a guard or fix before any entry runs
non-2048 ubatches with W4A4.

### 47c. W4A4 PPL@8k on 9B Q4_0 (works at production shapes)
| config | PPL@8k (wikitext-2, -fa on) | delta |
|---|---|---|
| W4A4 off | 7.7476 +- 0.0532 | - |
| W4A4 gate=1 | 7.9541 +- 0.0542 | +0.2065 (+2.66%) |

Context for the dense-W4A4 quality picture: Mistral-Large-123B fine-tune +0.082@8k (+7.7% rel), this 9B dense
+0.2065@8k (+2.66% rel) - relative cost smaller on the smaller model, absolute cost larger.
Reproducibility note: an earlier buggy harness run (both legs accidentally W4A4-on due to a
hardcoded env) produced 7.9541 for the "off" leg - which exactly matches today's true ON value;
the W4A4 path is deterministic session-to-session.

### 47d. Crash bisect: batch-sequence boundary + gate2 silent corruption (2026-10-01)
Flag matrix (all GGML_CUDA_W4A4_DOT8=1 unless noted, 9B Q4_0, single GPU):
| config | result |
|---|---|
| c512 -b 2048 -ub 512 (n_seq=4) | CRASH (mmq_gemm_repacked illegal access) |
| c512 -b 2048 -ub 512 -fa on | CRASH (fa not a factor) |
| c512 -b 512 -ub 512 (n_seq=1) | OK, PPL 9.5196 |
| c512 -b 2048 -ub 2048 | CRASH |
| c512 mmap loader | CRASH (loader not a factor) |
| c2048 -b 2048 -ub 512 (n_seq=1) | OK, PPL 6.9811 |
| c2048 -b 2048 -ub 2048 | OK, PPL 7.0322 |
Bisect (gate x batch x chunks):
| config | result |
|---|---|
| gate1 b2048 c512 (n_seq=4) ch8 | CRASH |
| gate1 b1024 c512 (n_seq=2) | OK |
| gate1 b512 c512 (n_seq=1) | OK, 9.5196 |
| gate2 b2048 c512 (n_seq=4) | NO crash but PPL 295.58 = SILENT CORRUPTION |

Reading: the discriminator is llama-perplexity's multi-sequence batching (n_seq = b/c):
n_seq<=2 clean, n_seq=4 breaks both W4A4 gates - gate1 hard fault, gate2 silent wrong
numbers (295 vs ~9.5 at n_seq=1; 8-chunk sample but ~30x off is not sampling noise).
Single-sequence ne11=2048 is fine (c2048 legs), so it is NOT raw row count. Production
entries (server, -ub 2048, single-sequence continuous batching) are unaffected.
Upstream stock repack path handles n_seq=4 fine (section 46 KLD legs ran it) - this is
a W4A4 u4/u4p activation-path bug specific to multi-sequence src1 handling, two
different failure modes (u4act: OOB fault; u4p: silent corruption). KLD-with-W4A4 stays
unavailable until fixed; root cause next (u4 quantize kernel z-dim/grid or x_stride
handling under ne12>1 in the tiled branch, mul-mat.cu:84-113 + quantize.cu:574-651).

### 47e. ROOT CAUSE + FIX: plane-slice pointer arithmetic in wrong element units (51a43756f, 2026-10-01)
Bug: both GEMM slice loops (mul-mat.cu narrow ~115 and tiled ~293) offset the activation pool in
BLOCKS cast to block_q8_1* (34 B equivalent) while the u4act/u4p pools hold block_q8_1_u4 (20 B)
/ block_q8_1_u4p (32 B). Same block count, wrong size: every (i2,i3) plane after the first was
misplaced. gate1 (20 B): offsets overshoot 1.7x -> OOB fault. gate2 (32 B): in-pool shifted
garbage -> silent corruption. ne12=1: offset 0 -> always clean, which hid the bug through every
prior gauntlet (all ran single-sequence).

Framing correction to 47c/47d: the discriminator is ne12>1, NOT ne11=128. Sweep: 128/ne12=1
clean (9.3680), 128/ne12=2 NaN, 128/ne12=4 fault, 192+256 at ne12=1 clean. The "ne11=128 only"
framing was a harness artifact (128 only occurred at n_seq=4). Also: the "clean" ne11=256 b1024
control was itself ne12=2 and NaN-corrupt with rc=0 - W4A4 KLD legs were silently invalid before
this fix. Quantizer and kernel indexing verified correct; purely host-side slicing.

Fix: plane offsets taken in BYTES of the pool's own element type at both sites (51a43756f).
Provably no-op at ne12=1; A/B-identical PPL on identical builds pre/post.

### 47f. Validation matrix + W4A4 KLD (the original goal of this leg)
| config | before | after fix |
|---|---|---|
| gate1 ne11=128 x ne12=4 | fault | rc=0, PPL 9.5196 |
| gate2 same shape | 295.6 (corrupt) | rc=0, PPL 9.0520 |
| gate1 ne11=128 x ne12=2 | NaN | rc=0, PPL 9.4845 |
| gate1 b512 single-seq | 9.5196 | 9.5196 (exact) |
| gate1 c2048/b2048/ub2048 | 7.0322 | 7.0322 (exact) |
| greedy seal | - | coherent, primes 2-29 |

W4A4 KLD (c512 -b 2048 -ub 512 --chunks 128 vs F16 base, now runnable):
Mean KLD 0.122631 +- 0.002149 (two runs 0.12217/0.12263; stock Q4_0 ref 0.068316 -> W4A4
~1.8x stock mean). Median 0.0643, 99% 0.959, 99.9% 4.29, max 24.2 - the gap is upper-tail
concentrated (most tokens nearly exact, rare tokens diverge hard), consistent with the
depth-PPL growth pattern. Mean PPL(Q) 8.845 vs base 8.021 (ratio 1.103).
Note: single-seq PPL shows ~1% run-to-run wobble on this rig (repack nondeterminism);
present with fix stashed - environmental, not a regression.
Full report: mi50:/root/w4a4-nseq-fix.txt; sweep logs /root/w4sweep-results.txt.

## 48. W4A4 accuracy diagnostic: outlier structure + phase-2 selection (2026-10-01, branch perf/w4a4-attrib)

Methods: per-tensor-class KLD via temporary GGML_W4A4_ONLY=attn|ffn dispatch gate; per-layer
activation dumps (16 chunks x 512 tok, blk.L.ffn_gate, L=0..31) + requant simulation for the
fix candidates. Results file: mi50:/root/w4a4-attrib-results.txt (Tables 1-3). Debug
instrumentation committed on perf/w4a4-attrib (3cd5f0fb9), marked debug.

### 48a. Attribution
| variant | Mean KLD vs bf16 |
|---|---|
| control (no W4A4) | 0.068316 |
| attn-only W4A4 | 0.083501 |
| ffn-only W4A4 | 0.096451 |
| full W4A4 | 0.122631 |
FFN ~52% of the delta, attn ~28%, ~25% superadditive via the residual stream.

### 48b. Outlier structure (the mechanism)
Outliers are CHANNEL-persistent, not token-persistent: fixed channel offsets ~3968-3999 and
3456 carry range 35-60 vs median ~2-4, from layer 4 onward; token 0/BOS worst. ~0.8% of
per-32 blocks (range > 8x median) carry 55-80% of quant MSE in layers 4-18 on ffn_gate and
attn_qkv inputs. ffn_down (post-SwiGLU) input is clean (12% concentration).
This is the classic per-channel activation-outlier failure of asymmetric per-32 u4 scales:
one big value crushes the other 31 in the block.

### 48c. Fix-candidate simulation
| candidate | activation-MSE cut | notes |
|---|---|---|
| per-16 sub-block scales | 42-54% (avg ~50%) | half the scale-plane bandwidth of per-8 |
| per-8 (gate2 u4p, EMPIRICAL) | 68-87% | recovers 84.5% of the KLD gap: 0.1226 -> 0.0768, max 24.2 -> 14.9; but -9.2% decode (sec 36) |
| SmoothQuant a=0.75 | 75-86% on foldable inputs | s-vector NOT sparse (top-1% channels carry only 2.4-5.0%); ffn_down unfoldable |
| clipping 3-6 sigma | REJECT | 3-4 sigma INCREASES MSE 3-13x; 6-sigma no-op |

### 48d. Phase-2 decision: build per-16 sub-block activation scales for the W4A4 GEMM
Expected (simulation + gate2 anchor): KLD mean ~0.090 (control 0.0683), 99.9pct ~3.5,
max <18. Direct derivative of the existing per-8 u4p kernel (2 sub-blocks instead of 4);
no calibration, no requant, no decode-path change. SmoothQuant = phase-3 fallback only.
Empirical anchor is strong: per-8 sub-blocks already recover 84.5% of the gap in this fork.
