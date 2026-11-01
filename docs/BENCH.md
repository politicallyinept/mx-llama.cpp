# BENCH.md - fork benchmark campaign log on 3x MI50 32GB (gfx906, TheRock ROCm 7.2.3)

All runs: -lm dio, env GGML_ENABLE_CUSTOM_AR=1 HSA_FORCE_FINE_GRAIN_PCIE=1
HSA_XNACK=0 GPU_MAX_HW_QUEUES=8, HEAD build (no RCCL) unless noted.
r=3 unless noted. Raw tables: retained on the measurement rig.

Env profile: unless a section says otherwise, published results were measured with
the production env bundle `GGML_CUDA_REPACK_Q4_0=1 GGML_TP_AR_SKIP_XBAR=1
GGML_CUDA_REPACK_KQUANT=0 GGML_CUDA_MOE_VEC=1 GGML_CUDA_MOE_FUSE=1
GGML_GDN_COLTHREAD=1` on top of the line above. `GGML_CUDA_REPACK_Q4_0` defaults
OFF in source, so bare-default runs do not engage the tiled/W4A4 repack paths;
where engaged, the tiled path is seal-verified bit-identical to the default path
at validated widths (batch 2-15 unsealed). Gate-truth record:
`benchmarks/defaults-vs-env.md`.

## 1. Split mode - the headline (2026-09-07)

| config | pp512 | pp1024 | pp2048 | pp4096 | tg16/32 | tg128 |
|---|---:|---:|---:|---:|---:|---:|
| layer, fa1 (deployed config class) | 70.9 | 104.8 | 134.6 | 151.7 | 7.57 | 7.62 |
| tensor, fa1 | 88.9-90.5 | 89.8 | 89.7 | 88.6-91.3 | 16.75 | 16.49 |

- **Decode: tensor +117% (7.6 -> 16.5-16.8 t/s).** This is the whole ballgame for
 a chat/creative model: generation dominates wall-clock.
- **Prefill: layer scales with batch (70->152), tensor flat ~90 (comm-bound).**
 Crossover ~700-800 prompt tokens. Worst case for tensor (8k prompt + 500 gen):
 layer ~81s vs tensor ~107s; typical creative turn (500 prompt + 500 gen):
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
~1GB total (GQA-8 KV is tiny) - use f16.** (Deployed config used q8_0 - change it.)

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
 covers the per-layer 48KB allreduce. Deployed build never had RCCL - works.
- Build flags that work: GGML_HIP=ON GRAPHS=ON MMQ_MFMA=ON NO_VMM=ON,
 AMDGPU_TARGETS=gfx906, clang, -O3. 4.5 min build on the 5950X.
- llama-bench has no -c flag (ctx from -p/-n); llama-cli defaults huge ctx ->
 KV OOM under tensor - ALWAYS pass -c.
- llama-cli: -no-cnv removed upstream; use -st (single-turn) for scripted gens.
- llama-server --fit: "not implemented for SPLIT_MODE_TENSOR" warning + aborts
 fit only - pass --fit off explicitly for tensor configs.
- GPU contention: llama-swap spawns on 8181 requests; benching needs an idle
 window (two of my runs died to this; also each llama-bench exit leaves ~64GB
 VRAM draining - the fixed harness waits for drain between runs).

## 7. Quality gate - PASSED
Greedy (-c 32768, -st, temp 0): layer and tensor both produce correct coherent
output; wording differs slightly (summation-order class, same as fork documents
for its bit-exact-except-reduction-order paths). No degeneration.

## 10. Kernel profile - tensor decode (rocprof, 2026-09-08)

tg96 run (64p+96g), instrumented 14.0 t/s (~19% overhead vs 16.75 clean). Raw:
rig/prof-tensor/tg.stats.csv. Decode-time attribution per token:

| consumer | share of decode | detail |
|---|---:|---|
| mul_mat_vec_q Q4_0 (type 2) | **~72%** | 2 shape classes: 52.7us x 124,845 (rows 12288-class) + 226us x 25,614 (need_sum=true, 1/layer/GPU - big rows: MLP/vocab class) |
| custom AR (k_broadcast_reduce<3>) | **~25%** | 176 calls/token (2/layer), 88us each - latency-bound, 48KB msgs over 2-hop PCIe |
| rms_norm + quantize_q8_1 | ~4% | 52569 + 102741 calls, 12us/4.4us |
| rest | <2% | |

Prefill-only in profile: mul_mat_q MMQ 14.3%, k_twoshot_f32 AR 8.5% (2.3ms each),
rocclr_copyBufferRectAligned 7.6% (4.7ms x 451 - KV/layer-seam copies).

**Implication: mat-vec (72%) >> AR latency (25%) >> everything else. AR at 88us/call
is ~3x pure-transfer time - a sync-protocol cost; candidate: fewer/lazier barriers,
but kernel-level AR surgery is higher risk than mat-vec work.**

## 11. Tuning experiments - mvq warp sweep (2026-09-08)

Target: `mul_mat_vec_q` Q4_0 (72% of decode time per profile). Tested GCN-table
nwarps for Q4_0 batch-1 (stock=2; RDNA4/DGX-Spark uses 8 for same shape, #26843).
Method: edit mmvq.cu calc_nwarps GCN table -> Q4_0 special case -> incremental
rebuild (~1 min) -> same bench params as T1 (r3).

| variant | tensor tg128 | layer tg128 | verdict |
|---|---:|---:|---|
| stock nwarps=2 | 16.68 | 7.62 | **best - keep** |
| nwarps=4 | 16.48 | 7.26 | -4.7% layer - reject |
| nwarps=1 | 15.16 | 7.25 | -9% tensor - reject |

**Negative result: RDNA4's 8-warp finding does NOT transfer to GCN5. Fork stock
warp config is optimal at this level.** (Thermal note: post-sweep stock re-verify
read 16.48 vs 16.68 earlier - sustained-bench drift; relative ordering valid.)

Remaining decode levers (mapped, not yet attempted):
1. AR latency: k_broadcast_reduce 88us x 176/token = 25% of decode. Pure transfer
 is ~25us for 48KB over 2-hop PCIe -> ~3x sync-protocol overhead. Needs barrier
 laziness/overlap work in tp-allreduce.cu (high risk, high value ~+15-20% tg).
2. Fused dequant+dot inside mul_mat_vec_q for Q4_0 (the 226us need_sum class).
3. MMVQ->MMQ crossover knobs are MoE-only (MUL_MAT_ID batch caps) - N/A dense.
4. rms_norm/quantize are 4% combined - not worth fusing first.

## 14. production-128B-v3 Q4_0 validation (2026-09-08, same rig/method)

125B params, 67.71 GiB. Quality gate passed (coherent, mode-wording differs as expected).

| config | pp512 | pp2048 | pp4096 | tg32 | tg128 |
|---|---:|---:|---:|---:|---:|
| tensor + f16 KV | 90.05 | 89.33 | 88.35 | 16.04 | 15.93 |
| tensor + q8_0 KV | - | 89.13 | - | - | 15.58 |
| layer + f16 KV | 70.65 | 132.64 | 148.80 | 7.45 | 7.44 |

dense-prod-model findings REPLICATE on this independent finetune: tensor decode +114%,
same flat-PP ~89 comm-bound signature, same ~800-token crossover, KV q8_0 costs
2.2% tg. Conclusion: these are properties of the Mistral-Large arch + q4_0 +
3xMI50 tensor split, not model-specific.

### Config bug found + fixed in llama-swap production-128B entry (root cause)
Symptom: entry 502'd on spawn ("upstream exited prematurely").
Chain: `--fit on` aborts under SPLIT_MODE_TENSOR (known) -> `--fit-ctx 65536`
is part of the fit machinery and was silently DROPPED with it -> ctx defaulted
to model max 262144 -> with the original q8_0 KV the 256k KV (~4.3 GiB/GPU)
barely fit, masking the bug; moving KV to f16 doubled it (~8.6 GiB/GPU) ->
graph-reserve alloc failure (ggml_backend_meta_buffer_type_alloc_buffer abort,
exit 134). The entry had been unspawnable-in-principle since -sm tensor was
added; nobody noticed because q8 KV + luck kept it just inside VRAM.
Fix deployed: --fit off, explicit -c 32768, f16 KV. Live-verified via proxy
(ready, coherent output). Backups: config.yaml.bak-20260908{-production-128b,}.
NOTE: 64k f16 KV does NOT fit on this model (1 GiB/GPU more weights than
dense-prod-model); 32k is the max comfortable ctx.

## 15. Delegation timeout fixed
config had delegation.child_timeout_seconds: 900 (explicitly set; code default
is no cap) - raised to 3600 via hermes config set. Local delegation LLM
turns run 49-110s; the 900s cap was killing healthy runs.

## 16. llama-swap full config sync (2026-09-09, owner-directed)

Backup: config.yaml.bak-20260908-listsync. Applied via text-preserving transform
(not yaml round-trip). Result: 26 entries, all validated.

- REMOVED (9): model files gone from disk (community finetune quants, names withheld).
- ADDED (8): a community-abliterated Qwen3.8-27B quant (27.1G), Qwen3.8-27B base Q8/Q4(27.1/15.0G),
 Muse-Glimmer-30B(27.6G, gemma preset), community-FT-1.5-35B Q8/Q4(34.4/19.7G),
 endless-frontier-BigBang-v1(35.2G), Mistral-7B-v0.3-abliterated(4.1G, spec-draft
 utility). my-stories-lora.gguf excluded (LoRA adapter, not servable).
- DEVICE RULE applied per owner: <=35GB -> --device ROCm1,ROCm2 (low-latency pair);
 >35GB -> all three GPUs. 15 existing entries migrated.
- LANDMINE SWEEP: 0 remaining -sm layer / --fit on / --fit-ctx; every entry has
 explicit -c; ${default_params} macro now has --fit on removed.
- dense-prod-model + production-128B entries preserved byte-identical (content-level; production-128B's
 --fit off diff vs old backup is my earlier validated fix).
- VALIDATION: yaml parse OK; llama-swap hot-reload OK ("Configuration Reloaded");
 26 models listed; spawn test Qwen3.8-27B-Q4_0 PASS (content correct, reasoning
 in reasoning_content, cache-reuse active 42 tokens).
- PITFALL logged: python f-string escaping of ${...} llama-swap macros double-braced
 them (${{PORT}}) and a blind sed half-fix left trailing }} - llama-swap rejected
 the reload ("proxy uses ${PORT} but cmd does not"). Always grep for }} and
 verify "Configuration Reloaded" (not "Unable to reload") in journalctl after edits.

## 18. Subagent provider status (2026-09-09)
Worse than 09-08: both dispatches died to hard API timeouts (180s no-response,
3 retries) + iteration cap at 3 API calls. LiteLLM local-delegation route
unusable for delegation right now. Config-sync + AR analysis were completed
in-parent instead (sections 16 and pending).

## 19. AllReduce latency work (2026-09-09): twoshot found + deployed (+3.3% tg)

Method: fork's built-in GGML_TP_AR_KTIMING=1 phase profiler (write/bar_start/
reduce/bar_end per call), no code changes needed for diagnosis.

Phase profile (22,706 decode AR calls, broadcast path): ~65us/call total;
phases anti-correlated across ranks (rank0: write 31k ticks + WAIT 71k;
rank2: write 62k + wait 27k) => pacing item = peer-write phase; reduce trivial
(2.5%); rank-dependent write times = asymmetric PCIe paths.

Root cause of missed win: auto-twoshot selection lists nranks in
{5,8,10,4-conditional} but NOT 3. GGML_TP_AR_TWOSHOT=1 forces it (12288 %
(4*3) == 0: alignment-eligible). Two-shot cuts per-rank wire 2x48KB -> 2x16KB.

| config | tg128 | notes |
|---|---:|---|
| broadcast (auto) | 16.95 / 16.49-16.75 (r1/r3 class) | baseline |
| GGML_TP_AR_TWOSHOT=1 | **17.42 +- 0.07 (r3)**, 17.57 (r1) | +3.3% decode, PP unchanged |
| 2-rank Qwen3.8-27B Q4_0 | 43.23 vs 43.35 | CORRECTED: neutral on 2 ranks. (First measurement 27.2/26.8 was INVALID - llama-bench treats --device A,B as a cycle list of single-GPU runs, not one 2-GPU run; must HIP_VISIBLE_DEVICES-pin for multi-GPU bench. llama-server is unaffected - its --device takes a set.) |
| quality gate (greedy) | PASS | coherent output, 17.5 t/s |

DEPLOYED: GGML_TP_AR_TWOSHOT=1 added to /llama-swap/start.sh env (service
restarted, live-verified via proxy). Cumulative dense-prod-model decode: 7.6 -> 17.4 t/s.
Remaining AR headroom: write phase still paces (asymmetric peer paths);
next lever would be per-rank-pair tuning or barrier overlap - diminishing returns.

## 20. Repack calibration (2026-09-09): dense K-quant repack = +8.7% DECODE on 3-rank tensor

A/B on L3.3-70B-community-70B Q4_K_M (dense, 3-GPU tensor, twoshot on):
- repack ON (default): pp512 109.80 / pp2048 110.53 / tg128 26.87
- repack OFF (-nr 1): pp512 108.90 / pp2048 108.29 / tg128 24.73
=> +8.7% tg, +1-2% pp. Under 3-rank tensor, PP is AR-bound so kernel gains
barely move it; decode reads via the repack buffer are the real payoff.
(Fork author's +50-133% pp claims are MoE/full-row shapes - not our regime.)
Implication: Q4_0 repack traits entry (subagent authoring, perf/q4_0-repack-gfx906,
env GGML_CUDA_REPACK_Q4_0) is worth ~+5-9% decode on dense-prod-model if it lands in
the same class => potential 17.4 -> ~18.5-19 t/s.

## 21. Q4_0 repack LANDED (2026-09-09): +16.6% decode, production-deployed

Patch: branch perf/q4_0-repack-gfx906 (Gitea). 154 insertions, 6 files.
Design (in-parent after 2 subagent attempts died to provider timeouts):
- block_q4_0 byte-identical to block_iq4nl -> converter + layout REUSED from IQ4_NL
- traits: affine decode d*(q-8), scale2 = {d, 8d}, -8 term folded via DS4 activation
 sums (mmq_get_q8_1_ds_layout ALREADY mapped Q4_0->DS4; zero GEMM-side changes)
- admission env-gated GGML_CUDA_REPACK_Q4_0=1 in BOTH gates (tensor_supported +
 meta buft-selection), so off = exactly stock behavior
- numerical class: reduction-order + f16 sum rounding, same as K-quant repack

A/B on the dense production model (tensor, twoshot on, r=3; per-model raw numbers
withheld from the public record along with that model's other benches):
repack ON measured +16.6% tg128, +5% pp. (Q4_K_M calibration predicted +5-9% tg; actual beat it.)
Quality gate: PASS - greedy output coherent, prime-list primes exact; inline gen ~20 t/s.
Live proxy verify PASS.

DEPLOYED: start.sh += GGML_CUDA_REPACK_Q4_0=1 (perf branch build; old tree untouched;
config/start.sh backups taken).

Cumulative decode improvement on the production dense model on 3x MI50: 7.62 (layer,
deployed Sept 7 AM) -> 16.7 (tensor, f16 KV) -> 17.42 (twoshot AR) -> 19.96 (Q4_0 repack)
= +162% total vs the original production config.

## 22. Fleet regression suite + AR xbar (2026-09-10): ALL CLEAR, +3.9% 2-rank

Regression suite (5 models, prod binary, twoshot on, gates greedy):
| model | type | ranks | tg128 | gate |
|---|---|---|---:|---|
| production-128B-128B | Q4_0 (patch) | 3 | 17.11 -> 19.66 (+14.9%) | PASS primes |
| Qwen3.8-27B | Q4_0 (patch) | 2 | 43.59 -> 46.85 (+7.5%) | PASS |
| Qwen3.8-27B | Q8_0 (stock) | 2 | 36.53 (nr: 35.50) | PASS |
| community-MoE-35B | Q4_K_M (stock) | 2 | 96.81 (nr: 93.18) | PASS |
| community-MoE-35B | Q8_0 (stock) | 2 | 102.05 | PASS |
No regressions anywhere; stock repack types clean on patched binary.

## 23. AR xbar: broadcast end-barrier elision (perf/ar-latency-gfx906)

Analysis (in-parent; subagent died to provider timeouts - 3rd strike):
- broadcast kernel: peer-write flush + __threadfence_system (rank-asymmetric,
 31k vs 62k ticks) then TWO barrier RTs; reduce 2.5%.
- barrier_end is cross-call-only (protects staging overwrite). Under parity
 double-buffering (2 staging sets, alternated via _flag>>1&1 in-kernel; flag
 advances +2/call in stock AND xbar mode) consecutive calls never touch the
 same set => barrier provably redundant => elided. Bit-exact (reduce order
 untouched). Twoshot's barrier_end is intra-call (allgather visibility),
 NOT elided.
- Merged to production on the perf/ar-latency-gfx906 branch lineage. Results: 2-rank
  broadcast path
 43.12 -> 44.81 (+3.9%, variance 10x tighter); 3-rank twoshot no-op (as
 designed). dense-prod-model merged-build stack: 20.35 t/s. GGML_TP_AR_SKIP_XBAR=1
 deployed fleet-wide in start.sh. Live proxy verify PASS.
Cumulative dense-prod-model: 7.62 -> 20.35 t/s (+167%). Remaining AR cost is the
two PCIe-flush RTs per twoshot call (structural) - next lever would be
fusing the two per-layer ARs, unexplored.

## 24. Q4_K_M lever analysis + twoshot auto-fix (2026-09-10)

Question: can K-quant (Q4_K_M) models be improved like Q4_0 was?
Code + profile findings (rocprof decode, community-70B 3rk / community-MoE-35B 2rk):
- Dense Q4_K already rides the repacked mat-vec (mul_mat_vec_rp<12> = 44% of
 GPU time; canonical path gone). MoE GLU-fusion on by default. Dense fused
 mat-vec is Q8_0/MXFP4-only AND measured-losing by the fork author even for
 Q8_0. MMQ tile tables are bypassed under repack. => K-quant kernels are
 already well-served; the real lever was elsewhere:
- PROFILE BOMBSHELL: community-FT 2rk spent 39.6% of GPU time in k_twoshot_f32<2>
 (98.9us/call) - our fleet-wide GGML_TP_AR_TWOSHOT=1 forced twoshot on 2-rank
 models where the fork's auto-select correctly keeps broadcast.
- Measured (2rk): community-FT Q4_K_M 99.15 forced -> 100.53 auto (+1.4%),
 Qwen27B-Q4_0 46.07 -> 47.31 (+2.7%). 3rk unchanged (twoshot both ways).
FIX: one-line auto-select correction (nranks==3 added; perf/tp-ar3-autotwoshot). GGML_TP_AR_TWOSHOT removed from prod start.sh (auto now does
the right thing per rank count). Verified: dense-prod-model 3rk auto 20.29 t/s (==
forced), community-FT 2rk auto 100.08, gates pass, live proxy OK.
Per-model AR policy now: 3rk=twoshot(auto), 2rk=broadcast+xbar(auto+flag).
community-70B dense Q4_K 3rk baseline for the record: 21.06 t/s (profiled run).

## 25. Accuracy audit (2026-09-14): optimized stack PRESERVES PPL - PASS

Method: llama-perplexity, same binary (prod build), same corpus (Dickens
771KB, c=2048, fixed chunks), env-flip only per leg. Full logs rig/ppl-logs/.

| model | leg | PPL | verdict |
|---|---|---|---|
| Qwen3.8-27B Q4_0 2rk | D stock | 4.9039 +- 0.0745 | baseline |
| | D2 +repack | 4.8990 +- 0.0744 | delta 0.005 (15x inside SE) PASS |
| | E +repack+xbar (prod) | 4.8990 +- 0.0744 | identical to D2: xbar bit-exact PASS |
| community-MoE-35B Q4_K_M 2rk | F xbar-off | 1.9495 +- 0.0160 | baseline |
| | G xbar-on (prod) | 1.9495 +- 0.0160 | identical: xbar bit-exact PASS |

Conclusions:
- Q4_0 repack entry: PPL-neutral on both architectures (deltas negative but
 far inside error = zero within measurement).
- AR xbar (my patch): bit-exact confirmed on dense AND MoE (4-decimal match).
- Twoshot vs broadcast AR: identical PPL on the dense production model (per-model
 numbers withheld from the public record) - fork's lossless-AR claim verified.
- f16 KV: not re-tested here (baked in config, predates patches; standard
 practice, was the chosen non-degrading option vs q8 in section 6).

## 26. MTP speculative decode - tested, NOT deployable (2026-09-14)

The fork ships a modern spec zoo (draft-mtp/eagle3/dflash/dspark, ngram-*).
Tested community-FT-1.5-35B + its MTPv2 head (mtp-community-FT...gguf), 2rk, prod config:
- Acceptance profile good: mean len 2.2-3.8, pos0 rate 0.94.
- BUT throughput 23-24 t/s vs 82-94 t/s control = 4x REGRESSION. Server log:
 "backend offload failed for seq_id; using CPU sampler" x4 - draft sampler
 falls back to CPU per round (root-cause lead: the -sm tensor draft backend
 offload path; likely meta-backend gap for draft contexts).
- AND greedy MTP output != greedy control (both MTP runs skip prime 23;
 control correct). Lossless contract violated -> correctness bug candidate.
VERDICT: closed until the fork's draft-offload bug is fixed upstream. The
Qwen3.6-family MTP-preserved models on disk remain dormant assets for then.

## 27. Upstream sync 2026-09-16 (32 commits) merged: CLEAN, deployed

Merge: perf/upstream-20260916 (base + upstream ).
Conflicts (Q4_0 x Q5_1 in q8_repack) resolved keeping both types; 3 build-fix
passes. Highlights: custom AR default-on, twoshot allgather-region fix
(MEASURED on our rig: 76.9 -> 69.2 us/call, -10%), Q5_1 repack entry,
DeepSeek-V4.1 arch, DSpark/MTP spec work, -tps mirrored weights.

REGRESSION SCARE, RESOLVED: first post-merge suite read dense-prod-model 19.56 (was
20.29) and Qwen 44.37 (was 47.31). Interleaved same-day A/B against a fresh
build of the PRE-MERGE commit: pre 19.37/19.56 + 44.38/44.27 vs merged
19.53/19.32 + 44.37/44.47 => IDENTICAL. The Sep-10 reference numbers were
simply from a faster box-day (~4-6% day-drift). Kernel profiles agree: merged
GPU time LOWER (11.23 vs 11.56s), AR improved, repack gain intact
(16.49 repack-off vs 19.38 on same day). PPL bit-identical. Gates pass.
LESSON (now policy): never compare t/s across days; interleave a control.

MTP RETEST on merged build (community-FT + MTPv2 head): still broken the same way -
23-25 t/s vs 82-94 control, CPU-sampler fallback x4 in log, greedy output
still skips prime 23 (lossless violation). Upstream's spec fixes did not
address it; root cause still open (draft backend offload under -sm tensor).
MTP stays closed. Reference-build trick retained: rig/mx-premerge
for future interleaved controls.

## 29. BF16-on-wire AllReduce - deploy-worthy (2026-09-18)

Env-gated twoshot AR with BF16 wire format: GGML_TP_AR_WIRE=bf16s (scatter only) [2026-10-22 NOTE: GGML_TP_AR_WIRE lives on branch perf/ar-bf16-wire-gfx906, NOT in this tree - these entries are historical records from that lineage.]
or bf16 (scatter+allgather). Unset/f32 = stock byte-identical. Halves peer bytes
per hop (8 bf16 per 16B store). Accumulation stays F32, same r-loop order.
Branch perf/ar-bf16-wire-gfx906 .

Interleaved bench (dense-prod-model 3rk, tg128, r=5, same-day):
| leg | config | tg128 |
|---|---|---|
| A | f32 control | 16.55 |
| B | bf16s | 15.08 (-8.9%) |
| C | bf16 | 16.97 (+2.5%) |
| D | f32 drift ctrl | 16.56 (+0.06%) |

PPL (Dickens 12ck, after addressing fix):
| config | PPL | vs f32 |
|---|---|---|
| f32 | 1.0573 | - |
| bf16s | 1.1011 | +0.044 |
| bf16 | 1.0588 | +0.0015 |

Gate: bf16 greedy primes exact (2..29). PASS.

Verdict: bf16 (scatter+allgather) clears both bars - +3.3% tg128 (confirmed
16.97/16.99 across two runs) AND PPL +0.0015 (inside +0.010 tol). bf16s is a
net loss: converting the scatter (non-barrier-gating phase) costs ALU with no
wire benefit where it matters; the win comes from halving the ALLGATHER wire.
gfx906 has no native bf16 (emulated conversion), which is why wasted
conversion in bf16s hurts.

Bug caught by PPL gate: initial patch had 2x slot-stride error (readers used
slice*1 instead of slice*4) -> PPL 13671. Fixed in a follow-up commit on the
perf/ar-bf16-wire-gfx906 branch lineage (readers switched to the correct slice*4 slot
stride). Lesson: bench
never validates output; the PPL gate is what makes a wire-format change safe.

NOT deployed - production stays f32 until Brad opts in. To enable: set
GGML_TP_AR_WIRE=bf16 in the llama-swap env for dense-prod-model. [2026-10-22 NOTE: GGML_TP_AR_WIRE lives on branch perf/ar-bf16-wire-gfx906, NOT in this tree - these entries are historical records from that lineage.]

### 29b. bf16-wire stacked on full production (repack ON) - 2026-09-18

Interleaved 4-leg (r=3 each): f32 19.62/19.63 | bf16 20.23/19.90.
Stacked gain +2.2% (vs +3.3% measured on the no-repack baseline) - matches
the multiplicative model: faster mat-vec shrinks AR's share of the critical
path. Deploy expectation: dense-prod-model ~20.1 t/s, PPL unchanged (+0.0015;
repack and AR-wire are independent, both individually PPL-validated).
Note: the MXFP4-table "Q4_0+repack 19.4-19.6" vs this experiment's f32
control 16.4-16.6 difference = the repack flag itself (+17.5%, measured
Sep 17: 16.49->19.38). Different baselines, no contradiction.

## 30. bf16-wire full regression suite (2026-09-18/19) - NO REGRESSIONS

Full production stack in every leg (repack+xbar+custom-AR+FA+tensor); only
GGML_TP_AR_WIRE toggled. Interleaved f32/bf16/f32 same-day per model. [2026-10-22 NOTE: GGML_TP_AR_WIRE lives on branch perf/ar-bf16-wire-gfx906, NOT in this tree - these entries are historical records from that lineage.]

| Model | f32 | bf16 | delta | PPL f32->bf16 | Gate |
|---|---:|---:|---:|---|---|
| production-128B-128B Q4_0 3rk | 18.82 | 19.47 | +3.5% | - | PASS |
| Qwen3.8-27B Q4_0 2rk | 43.41 | 44.41 | +2.4% | 4.8277 -> 4.8378 (+0.010, inside SE) | PASS |
| Qwen3.8-27B Q8_0 2rk | 34.72* | 35.00 | +0.8% | - | PASS |
| community-MoE-35B Q4_K_M 2rk | 100.74 | 100.68 | ~0% | 1.9212 -> 1.9190 (-0.002) | PASS** |
| community-MoE-35B Q8_0 2rk | 103.34 | 103.82 | +0.4% | - | PASS** |

* first f32 Q8_0 leg contaminated (GPU not idle); clean repeat used.
** community-FT is a thinking model: -n 100 burns out in the reasoning preamble
 (reasoning itself was correct). -n 512 required for the gate to reach the answer.

Notes:
- 2-rank models DO ride k_twoshot_f32<2> (large tensors exceed legacy min),
 so the bf16 patch reaches them; MoE models are AR-amortized -> neutral.
- Both models read slightly better than Sep-14 PPL baselines on BOTH legs
 (upstream merge drift, not bf16).
- Harness lessons: parse PPL with 'sed s/.*PPL = //' (regex-first-number grabs
 log timestamps); strip build banner before gate parsing; thinking models need
 -n 512 gates.
- Deployment: flag currently on dense-prod-model only. production-128B 3rk is the best next
 candidate (+3.5%, same topology as dense-prod-model). Config untouched per request.

## 31. Upstream sync 2026-09-19 (11 commits) merged + validated

Merge: perf/upstream-20260919 (our line + upstream
). Auto-merge CLEAN, zero conflicts.

FEATURES.md review (new since ):
- DFlash/DFlash2 drafters under -sm tensor: axis-0 gather so a drafter reading
 the target's split lm head sees whole vocab rows. Upstream: Qwen3.8-27B-Q8 +
 DFlash2 on 2xMI50 43.8 -> 52.1 t/s decode. UNLOCKS spec-decode under TP for
 our Qwen fleet. -tpsd runs drafter on fewer lanes.
- Tied output head split (gemma-4 family): head no longer falls to mirrored
 default; +9-11% decode. We run Gemma 4 models.
- Verify-batch twoshot crossover: nranks>=4 crosses at 8192 elements
 (GGML_TP_AR_TWOSHOT_MIN_NE tunable). AUDITED: only nranks>=4 semantics
 change; our 3-rank dense-prod-model + 2-rank Qwen/community-FT gates unchanged.
- Token-graph stage-seam capture (multi-stage pipelines), lane graph-cache off
 under token graph, fused 384-expert MoE router, DSV4 hc-gate fusion,
 full-device report at load.

Validation (merged build, full prod stack, vs same-day Sep-18 baselines):
| Model | merged f32 | baseline | delta |
|---|---:|---:|---:|
| production-128B-128B Q4_0 3rk | 18.83 | 18.82 | +0.05% |
| Qwen3.8-27B Q4_0 2rk | 44.43 | 43.41 | +2.35% |
| Qwen3.8-27B Q8_0 2rk | 35.03 | 34.72 | +0.89% |
| community-MoE-35B Q4_K_M 2rk | 101.14 | 100.74 | +0.40% |
| community-MoE-35B Q8_0 2rk | 104.77 | 103.34 | +1.38% |
PPL: Qwen 4.8277 (bit-identical to baseline), community-FT 1.9212 (bit-identical).
Gate: 5/5 PASS.
bf16-wire spot-check post-merge: production-128B 19.48 (baseline 19.47). Qwen bf16
interleaved A/B: 44.54/44.53 vs f32 43.05/43.49 - bf16 wins both pairs with
10x tighter variance. bf16 path intact and healthy post-merge.

VERDICT: clean merge, no regressions, all features intact. Deploy-ready.
Production still on prior binary pending go-ahead.

## 32. DFlash vs MTP speculative-decode shootout (2026-09-19)

Same-tool interleaved legs via llama-cli (temp 0, greedy, -n 256, realistic
creative-writing prompt), full prod env + LLAMA_ENABLE_MTP_OPT=1.
DFlash drafter: z-lab/Anbeeld Q8_0. MTP: native blk.64.nextn head.

### A: Qwen3.8-27B Q4_0 (actual prod target - no MTP head)
| leg | gen t/s | vs baseline |
|---|---:|---:|
| baseline (no spec) | 42.2 / 44.4 (mean 43.3) | - |
| DFlash2 (n-max 5) | 36.9 | **-14.8%** |
DFlash acceptance: 51.5% (183/355), mean len 3.54.

### B: Qwen3.6-27B-Native-MTP Q8_0 (same target, engine isolated)
| leg | gen t/s | vs baseline |
|---|---:|---:|
| baseline (no spec) | 35.5 / 35.5 | - |
| MTP (native head) | 41.4 | **+16.6%** |
| DFlash (drafter) | 37.1 | +4.5% |
MTP acceptance: 51.4% (183/356), mean len 3.54.

### Findings
- Acceptance is ~51% for BOTH engines - acceptance is NOT the differentiator.
- The differentiator is DRAFT COST. MTP head is embedded in the target and
 drafts off already-computed activations (near-free, shares KV). DFlash runs
 a separate 5-block transformer with its own KV cache every step.
- On the FAST Q4_0 target (3.8, 43 t/s), DFlash's separate draft pass can't
 pay for itself -> net -14.8%. On the slower Q8 target (3.6, 35 t/s) the
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

Hardware change: inter-GPU links moved to a PLX switch card. All models
re-baselined on ALL 3 CARDS, full prod stack (repack+xbar+custom AR+FA), r=3.

| Model | pp2048 | tg128 | prior tg128 | note |
|---|---:|---:|---:|---|
| production-128B-128B Q4_0 (3rk) | 173.1 | 20.49 | 18.82 | +8.9% (clean link effect) |
| Qwen3.8-27B Q4_0 (3rk) | 567.0 | 54.83 | 43.41 (2rk) | conflates link + rank change |
| Qwen3.8-27B Q8_0 (3rk) | 657.5 | 45.53 | 34.72 (2rk) | conflates link + rank change |
| community-MoE-35B Q4_K_M (3rk) | 1543.8 | 103.92 | 100.74 (2rk) | +3.2% |
| community-MoE-35B Q8_0 (3rk) | 1900.7 | 109.14 | 103.34 (2rk) | +5.6% |

PPL (improved or flat): Qwen3.8-27B 4.8235, community-FT 1.9021 (dense production model also flat; numbers withheld).
Gates: 3/3 PASS.

### bf16-wire is now a LOSS - the mechanism reversed

On the dense production model, tg128 flipped from +2.2% (bf16 vs f32) pre-PLX to
-1.2% post-PLX (interleaved 4-leg f32/bf16/bf16/f32, variance +/-0.05: both bf16
legs below both f32 legs; per-model raw numbers withheld from the public record).
Qwen shows the same trend (+2.4% pre -> -0.3% post).

Mechanism: bf16-wire's entire value was halving bytes across a SLOW PCIe link.
The PLX card removed the bottleneck, so the byte savings no longer cover the
gfx906 bf16 conversion cost. This is the same effect that made bf16s (scatter
only) a loss before - conversion cost is fixed, wire savings are conditional.

ACTION: keep f32 as the wire setting on the upgraded links. [2026-10-22 NOTE: GGML_TP_AR_WIRE lives on branch perf/ar-bf16-wire-gfx906, NOT in this tree - these entries are historical records from that lineage.]
f32 is the correct setting on the upgraded links.

### Link-width observation (unresolved)
sysfs reports all 3 GPUs at x16 16 GT/s (Gen4), each upstream bridge also
x16 Gen4; no PLX device (vendor 10b5) visible in the container's PCI
namespace. The +8.9% is real and reproducible, but does not match an "x4"
report - possibly the x4 reading was the PLX->CPU uplink rather than GPU-side.

## 30. Phase 0a + Phase 1 of the kernel/compute plan (2026-09-22)

### 0a. Perf-level pin (high vs auto): WASH - closed
Host pinned all 3 GPUs high (sysfs; LXC sysfs read-only, host-owned). Clock log PROVED
auto was already ramping to the pin's clocks (1850/1850/1725 sclk) pre-pin; same-build
controls match pinned legs within noise (17.36 vs 17.26 tg128, 149.5 vs 148.9 pp2048).
tg32 17.41 / tg64 17.25 / tg128 17.26 - no bursty-request ramp signature. No host A/B needed.
NOTE: legs ran on build/ (non-RCCL branch build) so absolute levels are lower than sec 6 -
the sec 6 anchors (176.4/21.39 post-PLX) live in BENCH.md sec 34 with the build-rccl binary.
Perf pin stays (harmless, already host-persistent).

### 1. Gate-threshold retune: KEEP 131072 - closed
Sweep on build-rccl (hybrid / no-gate / RCCL-only, dense-prod-model 3rk, pp8..pp2048 + tg32/64):
- pp8 (ne=98K, routed custom AR): 49.92 / 49.90 / 47.94 - gate correct, custom AR +4% over RCCL
- pp16 (ne=197K, routed RCCL): 66.12 / 66.90 / 66.21 - ONLY misrouted size; custom AR +1.2%
- pp32+: hybrid wins (86.89 vs 85.95 at pp32; 143.35 vs 136.89 at pp256)
- tg32/tg64: hybrid = no-gate (17.40 flat); RCCL-only -11% (15.55) - decode stays custom
Empirical crossover sits inside the current threshold's routing except pp16 (+1.2%).
Not worth a code change. Threshold 131072 (3rk) confirmed optimal-enough; retune closed.
Raw: rig/phase1-raw/leg1-10 on mi50. Threshold is compile-time (ggml-cuda.cu:1382,
per-rank-count constants); NO_GATE=1 is the only runtime knob.

## 31. W4A4 dot8 gauntlet: perf passes, correctness FAILS - DO NOT ADOPT (2026-09-22)

Full program (commits -> -> -> on perf/w4a4-dot8-gfx906)
wired behind GGML_CUDA_W4A4_DOT8, gauntleted on dense-prod-model 3rk, interleaved, flock-serialized:

| Leg | Gate | pp2048 | tg128 | PPL |
|---|---|---|---|---|
| anchor | off | 163.97 | 20.62 +- 0.04 | - |
| gate-on | on | 163.85 | **21.33 (+3.44%)** | - |
| drift ctrl | off | 163.84 | 20.58 (0.19% drift) | - |
| PPL | off/on | - | - | 1.0564 / 1.0564 (?=0.0000) |
| greedy primes | on | - | 21.0 t/s | **FAIL: 100x token id 0, empty** |

Verdict: perf passes (+3.44%/+3.65% vs both controls, ?PPL=0.0000) but greedy gate FAILS -
gate-on decode emits garbage (token id 0 x100). The +3.4% is speed on a broken path.

Root-cause localization (bonus harness): u4_quant_check FAILS on ALL multi-channel cases
(ne>1: 1024x8x2, 5120x16x2, 4100x3; 97,054 mismatches) and PASSES single-channel ->
broken multi-channel index mapping in the u4 path (quantize kernel interleaved-vs-deinterleaved
or the xs stride math in dispatch). PPL structurally blind (pp-only; bug is decode-only).

Program value: dp8a codegen verified (32x v_dot8_i32_i4), +3.4% tg achievable IF the
multi-channel mapping is fixed. The fix is localized and the harness gives exact
reproduction cases. Deferred pending decision: fix-forward (targeted debug dispatch)
or park as documented negative with the fix path recorded.

## 32. W4A4 fix-forward: all gates PASS - ADOPT-CANDIDATE (2026-09-22)

Root cause was THREE interacting bugs (fix commit , +55/-25, 3 files):
1. quantize.cu ib formula transposed activation blocks for ne1>1 (token-major fix);
 gridDim-derived ib0 also gapped on ne0%512.
2. pair-interleaved nibble packing mismatched canonical Q4_0 bytes -> sdot8 paired
 permuted values. Device kernel + host reference both de-interleaved now.
3. nib_sum_u4(pk)-256 computed UNSIGNED: raw excess-8 sums (~240 avg) wrapped to ~2^32,
 injecting garbage -> the token-0 decode. Fixed: signed cast.
Plus the dp8a operand-convention correction (MISMATCH hardware probe: hw pairing differs
from doc assumption; probe-verified before the fix).

Gauntlet v2 (build , dense-prod-model 3rk, flock+perflevel-high+bench-logged, r=5):
| Leg | pp2048 | tg128 |
|---|---|---|
| gate-off | 163.88 | 20.62 +- 0.04 |
| gate-ON | 163.85 | **21.20 +- 0.05 (+2.81%/+3.01%)** |
| drift ctrl | 163.44 | 20.58 (0.19%) |
PPL: 1.0564 both (?=0.0000). Greedy gate-ON: primes 2..29 EXACT + coherent tides.
Harness: OVERALL OK, 0 mismatches all shapes (4100x3 was 6472).

DECISION RULE: PASS all three gates -> ADOPT-CANDIDATE.
Deploy path (not done): merge perf/w4a4-dot8-gfx906 into the production build
(build-rccl) + set GGML_CUDA_W4A4_DOT8=1 (global or per-dense-prod-model entry). PP cost
of uncached u4 quant: not visible (pp2048 within noise). GEMM (mmq_gemm_repacked)
extension = future work, decode win banked first.

## 33. W4A4 MoE extension: correct but -5.3% tg - REJECT for MoE (2026-09-23)

Milestone 4 (commit ): wired u4 path into MoE narrow-batch dispatch
(mul_mat_vec_repacked_id1 + mul_mat_vec_rp, single-token) - Q4_0 MoE never reaches the
fused epilogue (Q4_0 absent from both fusion admission sets), so unfused wiring = stock routing.
Caught a unit bug pre-deploy: MoE column offsets ids_src1[a]*xs must be in 20B u4 units.

community-MoE-35B-A3B community-imatrix Q4_0 gauntlet (2rk, interleaved off/on/off, r=5):
| arm | pp2048 | tg128 |
|---|---:|---:|
| off-1 | 1835.97 | 110.17 |
| ON | 1829.67 | **104.28 (-5.3%)** |
| off-2 | 1827.94 | 109.78 |
PPL 2.0643 both (d=0.0000); greedy primes exact (n=400 for thinking model).

Verdict: tg fails >=+2% -> REJECT for MoE. Why: single-token MoE rides mul_mat_vec_rp
at LANES=16/NCOLS=1 - the per-block nib_sum_u4 SUMW recompute is consumed once per block
(no RPL=2 amortization like the dense winner). Fix paths if revisited: precompute SUMW
at repack (+2B fp16/block) or u4-ize the fused gate epilogue. Dense m3 remains the
adopt-candidate; MoE env flag should stay OFF in any deploy.

## 34. W4A4 GEMM gauntlet: +32% pp2048, ADOPT-CANDIDATE (2026-09-23, post-reboot rerun)

G3 fix verified committed (: wiring + u4 quantizer stride fix) before the host restart; only the gauntlet was missing. Rerun complete (build-w4a4, dense-prod-model 3rk, flock, perflevel high survived reboot):

| Leg | pp2048 | PPL |
|---|---|---|
| gate-off (a) | 157.40 +- 0.28 | 1.0564 +- 0.00352 |
| gate-ON (b) | **207.92 +- 0.30 (+32.1%)** | 1.1035 +- 0.00525 |
| gate-off (c) | 157.20 +- 0.27 | - |
| greedy primes gate-on | EXACT 2..29 + tides | |

DECISION: pp +32.10%/+32.26% (>= +2% x2) PASS; dPPL +0.0471 <= +0.05 PASS but within 6% of the limit (activation 4-bit cost is real at prefill scale - the 2-chunk smoke 1.0185 flattered it); primes PASS. -> ADOPT-CANDIDATE.

Caveats: (1) dPPL is the binding constraint - do not stack further u4-path changes without re-gauntlet; (2) binaries report G2 SHA (build-metadata lag) but gate toggles +32% pp AND the PPL delta, wiring demonstrably live; (3) GEMM covers prefill only - MoE stays REJECTED (sec 33).

Full program: W4A4 = dense decode +2.8-3% tg + prefill +32% pp (both PPL-gated), MoE reject. Deploy = merge perf/w4a4-dot8-gfx906 into build-rccl + per-model GGML_CUDA_W4A4_DOT8=1 on Q4_0 dense entries (NOT MoE).

## 35. 175W re-baseline + long-context matrix - COMPLETE (2026-09-24)

Envelope: 175W/card (upp, persists), ALL models 3-rank -sm tensor (topology standardized),
binary build-w4a4 (full stack). Prior sec 30-34 numbers INVALID (power + topology changed).
Matrices: production-128B@32k, Qwen-Q4_0/Q8_0+community-FT@131k (dense production model@64k also re-baselined; numbers withheld).

### production-128B 128B (3rk, ctx 32768)
pp2048: 154.87 / 199.62 (+28.9%) | pp8192: 142.67 / 179.69 (+25.9%) | tg@8k: 19.69 / 20.22 (+2.7%)
PPL@8k: 1.5436 / 1.7339 (d=+0.190 - OVER the +0.05 gate!)

### Qwen3.8-27B (3rk, ctx 131072)
Q4_0: pp2048 529.77 / 630.72 (+19.0%) | pp8192 511.56 / 606.32 (+18.5%) | pp16384 486.65 / 571.91 (+17.5%)
tg@131k-depth: 52.18 / 52.53 (+0.7% - flat) | PPL@8k: 5.0055 / 5.7254 (d=+0.72!! OVER gate)
Q8_0 control: pp2048 594.27, pp16384 540.39, tg@131k 44.26, PPL@8k 4.3910
NOTE: Q8_0 is FASTER at pp than Q4_0+W4A4 (594 vs 630 close, but 571 vs 606 at 16k) - dequant tax analysis revisited.

### community-FT 35B MoE (3rk, ctx 131072)
pp2048: 1808.46 / 1929.73 (+6.7%) | pp8192: 1736.56 / 1848.04 (+6.4%) | tg@131k: 103.99 / 98.27 (-5.5%, regression persists)
PPL@8k: 1.9113 / 2.2150 (d=+0.30 - OVER gate)
greedy@131k: primes 2..23 listed (29 truncated by -n 100 budget on thinking model)

### KEY FINDINGS
1. W4A4 pp win GROWS with depth (largest relative gain on the dense production model at
 16k depth). The GEMM loop's dequant tax dominates harder as KV grows.
2. BUT: PPL-at-depth delta BLOWS the +0.05 gate on all models tested (Qwen +0.72@8k; the
 dense production models also exceeded the gate). c=2048 audits (d=0.0000/+0.0471) were
 measured at shallow context; at 8k+ context the activation-4bit cost is REAL and grows
 with depth.
3. community-FT@3rk: decode regression persists (-5.5%), pp gains +6.4-6.7% - still net reject
 for tg, but pp win is real for MoE prefill.
4. Qwen Q8_0 control: pp2048 594 vs Q4_0+W4A4 630 - Q4_0+full-stack now BEATS Q8_0 prefill
 on this rig (was: Q8 won prefill per findings sec 3).
5. Greedy-at-depth: correct on all models tested (primes listed correctly at 32k-64k depth).

## 36. C1/B8 milestone M3: per-8 u4p decode - REJECT on perf; depth-PPL lens corrected (2026-09-24)

Commit wired GGML_CUDA_W4A4_DOT8=2 (per-8 u4p) into dense decode dispatch.
Gauntlet (dense-prod-model 3rk, 175W, interleave gate 0/1/2, r=3):

| Leg | gate-0 stock | gate-1 per-32 | gate-2 per-8 |
|---|---:|---:|---:|
| pp2048 | 157.70 | 203.25 (+29%) | 157.65 (0%) |
| pp8192 | 144.97 | 185.95 (+28%) | 144.70 (0%) |
| pp16384 | 130.87 | 163.53 (+25%) | 130.62 (0%) |
| tg64@8k | 20.47 | 21.20 (+3.6%) | **18.58 (-9.2%)** |
| PPL@8k | 1.0751 | 1.1572 (+0.082) | 1.0751 (0.0000) |
| PPL@2k | 1.0564 | - | 1.0564 (0.0000) |

VERDICT: REJECT for decode (tg -9.2% blows the <=1% cost rule; the 27-VGPR spill +
4x epilogue of the per-8 kernel is real cost, matching design sec 8's risk register).

CRITICAL LENS CORRECTION: gate-2's PPL deltas of 0.0000 are STRUCTURAL BLINDNESS,
not quality proof - perplexity is prefill/GEMM-dominated, and u4p was wired
decode-only (pp identical to stock confirms it never touched GEMM). The +0.082
depth-PPL cost we set out to fix lives in the GEMM path (gate-1: GEMM u4 wired,
PPL moves; gate-2: decode only, PPL frozen). Same lesson as sec 31, re-confirmed.

C1 REFRAME: per-8 recovery, if wanted, belongs in M4 = GEMM u4p (prefill side),
where the depth-PPL cost is actually created. Decode stays per-32 (its +3.6% tg
and PPL-neutrality at 2k stand). Greedy@64k gate-2: output-file capture truncated
(empty after 'Assistant:') - noted, not re-run given the perf verdict.

## 37. C1/B8 milestone M4 complete: GEMM per-8 quality recovery CONFIRMED, perf cost REAL (2026-09-25)

Fix commit (GEMM u4p + best-of gate=2: decode per-32, GEMM per-8).
Post-PSU completion run (2 crashes interrupted earlier attempts):

| Model / leg | gate-0 stock | gate-1 per-32 | gate-2 per-8 GEMM |
|---|---:|---:|---:|
| dense-prod-model PPL@8k | 1.0751 | 1.1572 (+0.082) | 1.0943 (+0.019) - 75% recovery |
| Qwen PPL@8k | 4.9847 | 5.7148 (+0.730) | 5.1380 (+0.153) - 79% recovery |
| community-FT PPL@8k | 1.9525 | 2.2612 (+0.309) | 2.0157 (+0.063) - 80% recovery |
| dense-prod-model pp2048 | 154.79 | 205.96 (+33%) | 98.11 (-37%!!) |
| dense-prod-model pp8192 | 144.16 | 185.19 (+29%) | 93.09 (-35%) |
| Qwen pp2048 | 530.07 | 629.54 (+19%) | 389.36 (-27%) |
| Qwen tg@8k | 52.59 | 52.59 | 52.42 (flat, decode=per-32 as designed) |
| dense-prod-model tg@8k | 20.47 | 20.48 | 20.85 (identical class) |
| dense-prod-model greedy@64k gate2 | - | - | primes EXACT 2..29 + tides |

KEY: 1) QUALITY THESIS CONFIRMED - per-8 recovers 75-80% of depth-PPL on ALL models.
2) PERF: per-8 GEMM as implemented REGRESSES prefill hard (-27..-37% vs stock) -
epilogue/LDS overhead costs more than dp8a saves on GEMM tile shapes.
DEPLOY SHAPE: gate=1 for creative models at creative contexts; gate=2 = long-context-quality
mode if Qwen/community-FT entries need it; stock for everything else; MoE decode stock.

## 38. W4A4 step 2 (spill fix): hot-loop spills ELIMINATED, pp recovery partial (2026-09-25)

Profile-guided fix : U4P8 GEMM scale handling restructured -
spills 160 -> 13 dwords (hot loop ZERO; remaining 13 cold checked-edge only),
w32 52 -> 2. Root cause of M4 blowup: forced #pragma unroll on kk slab loop
quadrupled static dot8s (1024) causing the spill storm; rolled form restores
256. Plus SW packing: 4 per-8 SW_k fit int8 (-64..+56), packed 1 uint32,
cuts 3 VGPRs. Re-gauntlet: pp2048 116.58 (was 98.4, +19% recovered), pp8192
109.26, PPL@8k 1.0943 (+0.019 preserved EXACTLY), greedy@64k primes EXACT.
Remaining -26% vs stock is STRUCTURAL: 4 unchained accumulators + 8 LDS scale
reads per block = epilogue-FMA-bound, inherent to per-8. C1 closes: per-8
GEMM = documented quality-mode trade, not deployable universal mode.

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
>= 60% of gate1 cost = FAIL (dense-prod-model 26%, Qwen 18%); greedy@64k exact = FAIL.

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

Model: a 27B community uncensored MTP-preserved finetune, Q8_0 (29GB), 3rk.
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

## 41. Post-upstream-merge floor - NEW BASELINE, supersedes sec 35 absolute values (2026-09-27)

Envelope: identical to sec 35 (175W/card, 3rk -sm tensor, same datasets/scripts; only change = merged binary,
ggml lib 0.22 -> 0.25.3). Full raw logs: rig/rebase-merged-{big,small}.{txt,log}.

### production-128B 128B (ctx 32768)
pp2048: 167.19 / 222.65 (+33.2%) | pp8192: 152.85 / 197.94 (+29.5%)
tg64@32k: 19.61 / 20.18 (+2.9%) | PPL@8k: 1.5554 / 1.7329 (d=+0.1775)
greedy@32k: primes exact 2-29.

### Qwen3.8-27B (ctx 131072)
Q4_0 shallow: pp2048 615.00 / 777.13 (+26.4%) | pp8192 589.84 / 738.55 (+25.2%) | pp16384 556.31 / 687.44 (+23.6%) | tg128 52.56 / 52.56 (flat)
Q4_0 @131k: pp2048 198.92 / 219.04 (+10.1%) | pp8192 208.96 / 225.14 (+7.7%) | pp16384 213.98 / 230.97 (+8.0%)
tg64@131k: 34.75 / 34.70 (-0.1%, flat) | PPL@8k: 4.9912 / 5.7102 (d=+0.7190, over gate as before)
Q8_0 control: pp2048 724.20, pp16384 645.28 | tg64@131k 30.70-31.99 | PPL@8k 4.3902

### community-FT 35B MoE (ctx 131072, community imatrix Q4_0)
shallow: pp2048 1972.99 / 2139.33 (+8.4%) | pp8192 1887.32 / 2038.09 (+8.0%) | tg128 104.61 / 98.11 (-6.2%)
@131k: pp2048 540.99 / 552.28 (+2.1%) | pp8192 551.79 / 564.29 (+2.3%) | tg64: 89.84-91.22 off / 85.37-86.18 on (-5.8%)
PPL@8k: 1.9070 / 2.2064 (d=+0.2994, over gate as before)
greedy@131k budget-cut mid-think but primes 2-29 visible in trace.

### KEY FINDINGS vs sec 35
1. Everything faster: pp +7-47% vs sec 35 across the board (upstream graph/fusion/allocator wins compound with W4A4).
2. W4A4 pp gain GREW again at pp2048 on the dense production models vs section 35. The dequant-tax analysis holds.
3. Quality UNCHANGED: PPL deltas match section 35 within noise (Qwen +0.719 vs +0.72, Orn +0.299 vs +0.30; dense production models likewise unchanged); all greedy seals exact.
4. community-FT decode regression persists (-6.2% tg shallow with W4A4 on) - MoE W4A4 stays REJECT; gate stays OFF for MoE entries.
5. Qwen shallow pp16384: Q4_0+W4A4 (687) now BEATS Q8_0 (645) - inversion vs sec 35 era; at 16k@131k depth W4A4 (231) also beats Q8_0 (226).
6. Verdict: upstream merge is a strict win. Adopt the merged binary as the new production floor.

## 42. qwen4exp MoE community finetune (unnamed): prod-stack A/B + W4A4 check (2026-09-29)

Model: unnamed community abliteration of the qwen4exp base, Q4_0, 8 shards (94G).
Architecture qwen4exp: 48 layers, 512 experts 10 active, PLE n-gram embedding table
(per_layer_token_embd.weight, 27465 MiB) host-resident mmap, demand-paged (lazy read, no VRAM).
Rig: 3x MI50, -sm tensor, -c 32768, 175W. Server bench via /completion, greedy temp=0.
Prefill = two distinct ~11.3k-token prompts per config; decode = 256 tok 56-token prompt.
Single server per config + 2 passes; prefill spread +/-8% (page-cache warmth), decode is the
decision-grade number.

| Config | prefill pass1/pass2 | decode run1/run2 |
|---|---|---|
| A: repack ON, custom AR OFF | 770 / 809 t/s | 51.5 / 52.1 t/s |
| B: prod stack (repack + custom AR + skip-xbar) | 766 / 913 t/s | 56.7 / 57.3 t/s |
| C: B + GGML_CUDA_W4A4_DOT8=1 | 817 / 975 t/s | 51.9 / 54.0 t/s |

### FINDINGS
1. Custom AR = +10% decode (52.1 -> 57.3 avg); prefill unchanged (twoshot targets decode-sized tensors, as in section 35-era data).
2. MoE W4A4 REJECT reproduced on qwen4exp: decode -5.4% vs B (54.0 vs 57.3). Matches community-FT verdict (-5.3%, SUMW recompute per block, no amortization). Gate stays OFF for all MoE entries.
3. NEW: W4A4 PREFILL is a WIN on MoE: +6.8% vs B (975 vs 913 avg pass2; pass1 also up: 817 vs 766). Multi-token batches amortize the per-8 epilogue where single-token decode cannot. A batch-size-gated W4A4 dispatch (W4A4 when ne>1, stock decode path when ne==1) is the candidate if a MoE prefill mode is ever wanted. Not implemented; decode rules creative workloads.
4. PLE demand-paged table costs ~nothing warm: warm prefill 913-975 t/s at 11k tokens on a 27.5G table that never enters VRAM or RSS (server RSS ~4.3G).
5. BUG: fully-stock config (GGML_CUDA_REPACK_Q4_0=0) HANGS at warmup on qwen4exp under -sm tensor 3-GPU. Two deterministic repros (10+ min after buffer alloc, no warmup progress, 0% GPU). Repack is functionally REQUIRED for this arch, not just faster. Root cause not investigated.

Prod stack confirmed running on the qwen4exp prod server (repack Meta buffers + custom AR twoshot F32 3-rank); W4A4 gate OFF.

## 43. qwen4exp MoE finetune: prefault + ub4096, W4A4 gate-mode-3, MTP draft (2026-09-29)

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

### 43b. W4A4 gate mode 3: batch-gated dispatch (commit , branch perf/w4a4-dot8-gfx906)
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

### 43c. MTP draft (community MTP-draft GGUF, Q8_0 4.1G draft, llama-cli 3rk)
| config | tg temp 0.1 | greedy tg |
|---|---|---|
| base | 47.3 / 49.3 t/s | 51.5 t/s |
| MTP | 57.3 / 53.4 t/s | 64.1 t/s |
| MTP + MTP_SAFE | - | 63.6 t/s |
MTP = +14.7% avg tg (ABOVE the +10% bar, unlike Qwen3.6 Native-MTP +7.5%). SAFE guard costs ~0 here
(vs -10.4% on community-FT). LOSSLESS: FAIL for both plain and MTP_SAFE - greedy thinking traces diverge
from base (respond/answer word-level divergence, same set_sampler CPU-fallback signature as section
39; NOT model-specific). Final response text converged by chance in one seal; treat as non-lossless.
Deployed 2026-09-29 as llama-swap entry qwen4exp-MTP with GGML_TP_AR_MTP_SAFE=1 +
LLAMA_PLE_PREFAULT=1 (accepting trace divergence, same regime as the other MTP entries).
Methodology note: an initial 'LOSSLESS PASS' verdict from tail-compare was an artifact of differing
timing footprints; seals must compare response text with timing lines stripped, and a full-trace
compare is required for the lossless claim (the tail-only pass hid a diverging thinking trace).

## 44. MTP sampler guard T1 fix + qwen4exp stock-load root cause (2026-09-30)

### 44a. set_sampler T1: capability test replaces hard refusal (fix/mtp-sampler-guard-t1, )
Background: the guard is upstream-identical (PR #23287, deliberate CPU fallback under -sm tensor).
Investigation (mtp-sampler-bug-analysis.txt) designed a tiered fix; T1 implemented:
device sampling allowed iff logits row replicated (tensor_mirror_output / small dense DFLASH
drafter) or TP width 1 (get_split_state_ud-derived); CPU fallback kept for vocab-parallel heads;
one-shot warning suppression removed.

Seals (greedy, CUSTOM_AR=0, n=768, response-anchored compare, determinism control IDENTICAL):
| pair | verdict | guard state |
|---|---|---|
| DFlash2 base vs MTP | DIFFERS (near-tie wording) | 0 warnings: device sampling ENGAGED |
| Finetune base vs MTP | DIFFERS (list formatting) | 0 warnings: device sampling ENGAGED |
| Finetune perf @8k | base 36.5 vs MTP 59.1 t/s gen (+62%) | no regression |

CLASSIFICATION: both pairs diverge with device sampling engaged AND custom AR off. 43c had the finetune
diverging with CPU sampling too. So the losslessness problem for the finetune MTP and DFlash2 is the
upstream #25618 batch-invariance class (parallel verify kernels not bitwise-identical to serial
decode; near-tie flips), NOT the sampler guard. The investigation prediction (guard fix = DFlash2
fix) is REFUTED. Qwen3.6/community-FT remain the custom-AR reordering class (43c/44: AR=0 restores identity).

Verdict on T1: correct and worth keeping - scopes the guard exactly where backend argmax is safe
and removes the per-draft-token full-vocab logits D2H for mirrored/single-lane drafts - but it is
NOT a losslessness fix on this rig. Spec-decode losslessness on -sm tensor stays blocked on the
upstream #25618 class. Full data: rig/mtp-t1-fix-results.txt.

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
staging path), NOT qwen4exp-specific quant handling. dense-prod-model Q4_K_M stock tolerance under mmap
remains unexplained (probable: its tensor shapes avoid the tiny-copy regime) - open, low priority
since production uses -lm dio everywhere.
Actions: -lm dio added to the qwen4exp-MTP llama-swap entry (consistency with the rest of
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

Q4_K_M paradox resolved: the dense production model is llama-arch dense (no *_exps tensors, GGUF-verified)
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
synthetic weighted-error ratios 0.825-0.957; REAL community-FT expert data w_err 0.629 (37% error cut,
vs MXFP4's 5.7% - the finer scale grid actually bites on real weights). Code for this
leg lives on branch perf/nvfp4-imatrix.

### Numbers (community-MoE-35B, same harness as M3, Q4 control same-run)
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
the creative adoption bar on quality AND (for NVFP4) speed on gfx906. The quantizer-side exploration
of this family is COMPLETE: MXFP4 raw, MXFP4+imatrix, NVFP4+imatrix all measured; no further
quantizer-side path remains (next rung would be mantissa width, i.e. not MXFP4/NVFP4).

## 46. Qwen3.5-9B upstream-data matrix (2026-10-01, branch bench/qwen35-9b-upstream-data)

Purpose: reviewer-reproducible data for the upstream PRs. Model bartowski/Qwen_Qwen3.5-9B-GGUF
bf16 (full-precision file in that repo; no F16), Q4_0 + Q4_K_M quantized locally with stock
llama-quantize. Dataset wikitext-2-raw-v1 test set (wiki.test.raw, scripts/get-wikitext-2.sh from
ggml-org/ci). Build , 3x MI50, -t 16, -lm dio, PPL -c 512 -ub 512 --chunks 256 (KLD legs
--chunks 128). Full data: rig/qwen35-9b-upstream-results.txt, scripts committed on the branch.

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

Context for the dense-W4A4 quality picture: the larger dense production model measured
+7.7% rel at 8k (per-model numbers withheld from the public record); this 9B dense
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

### 47e. ROOT CAUSE + FIX: plane-slice pointer arithmetic in wrong element units (, 2026-10-01)
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

Fix: plane offsets taken in BYTES of the pool's own element type at both sites .
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
Full report: rig/w4a4-nseq-fix.txt; sweep logs rig/w4sweep-results.txt.

## 48. W4A4 accuracy diagnostic: outlier structure + phase-2 selection (2026-10-01, branch perf/w4a4-attrib)

Methods: per-tensor-class KLD via temporary GGML_W4A4_ONLY=attn|ffn dispatch gate; per-layer
activation dumps (16 chunks x 512 tok, blk.L.ffn_gate, L=0..31) + requant simulation for the
fix candidates. Results file: rig/w4a4-attrib-results.txt (Tables 1-3). Debug
instrumentation committed on perf/w4a4-attrib , marked debug.

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

## 49. W4A4 phase 2: per-16 sub-block activation scales - gate 3 IMPLEMENTED (, 2026-10-02)

Gate 3 = GGML_CUDA_W4A4_DOT8=3: GEMM-path activations quantized per-16 sub-blocks (2x fp16
{d4,m} pairs + 16B nibbles = 28B/32-values, aligned(4); note: struct natural alignment is
2 from ggml_half - the explicit aligned(4) + 2B pad is REQUIRED, natural packing is 26B).
Decode/narrow paths untouched (per-32 u4act). Gate 2 (per-8) unchanged. Derived from the
per-8 u4p kernel (2 sub-blocks instead of 4). Layout note: 28B stride is NOT 16B-aligned -
scale pairs load as u32, never uint4.

### 49a. Seals + accuracy (Qwen3.5-9B Q4_0, single GPU, -lm dio)
| leg | gate1 | gate3 (per-16) | per-8 ref |
|---|---|---|---|
| GPU seal (vs CPU ref, 4096 cells) | - | maxrel 2.6e-6 PASS | - |
| PPL c512 (n_seq=1) | 9.5196 | **8.5422** | 9.0520 |
| KLD mean (n_seq=4, vs bf16) | 0.1226 | **0.0902** | 0.0768 |
| PPL c2048/-b2048/ub2048 | 7.0322 | **7.4332** | - |
| greedy seal (-n 700) | primes exact | primes exact, coherent | - |

KLD 0.1226 -> 0.0902 = 73.6% of the way to stock (0.0683); lands between the simulation
prediction (~0.090 - dead on) and short of per-8's 0.0768 as designed.
**CORRECTION (2026-10-08, pre-reboot audit, unverified pending re-derivation): the recorded
"73.6% of the gap" does NOT recompute from the recorded numbers above -
(0.1226-0.0902)/(0.1226-0.0683) = 59.7%. Either the 73.6% came from different inputs or it
is a transcription error. FLAGGED FOR RE-DERIVATION at next BENCH pass; until then treat the
gap-closure as ~60%, not ~74%.** c512 PPL beats BOTH
per-32 (9.52) and per-8 (9.05). Production-shape PPL is WORSE (7.03 -> 7.43) - at ub2048 the
per-16 scale plane costs more than the outlier recovery gains at that depth; see 49b.

### 49b. Perf
Quick pass: decode tg128 IDENTICAL (82.0 vs 82.2 - decode path untouched, as designed).
pp512 ~equal (1003.9 vs 1032.3, gate1 noisy +-304). pp2048 gate3 956.8 vs gate1 1211.3
(-21%) BUT gate1 spread was +-95 - flagged for the clean interleaved A/B (49c).
### 49c. Clean interleaved A/B (9 rounds/gate, trimmed means): pp -21% FLAT, tg unchanged
| cell | gate1 | gate3 | delta |
|---|---|---|---|
| pp512 | 1276.0 | 1003.9 | -21.3% |
| pp1024 | 1280.4 | 1006.7 | -21.4% |
| pp2048 | 1270.1 | 1000.1 | -21.3% |
| pp4096 | 1244.6 | 983.3 | -21.0% |
| tg128 | 82.19 | 82.25 | +0.07% |
| tg256 | 82.33 | 82.39 | +0.07% |
3-GPU TP (pp2048/n256, spread <0.1%): pp2048 2380.8 vs 1890.7 (-20.6%); tg256 88.37 vs
88.25 (-0.14%). The quick-pass "pp512 ~equal" was wrong; the penalty is flat ~-21% at
every pp size. Same-gate swings were sporadic one-off samples (5 dropped, both gates),
not thermal (42-55 C stable). Raw: rig/w4a4-ab/*.jsonl.

### 49d. VERDICT: gate 3 = REJECT for default; env-gated experimental, kept
Adoption bar was KLD <= 0.095 AND perf within ~2% of gate1. Accuracy: MET (0.0902, 73.6%
of the gap to stock; c512 PPL 9.52 -> 8.54, beats both gate1 and per-8). Perf: FAILED -
flat ~-21% pp at every size, both single-GPU and 3-GPU TP. Depth split is real: per-16
helps shallow (c512 8.54 < 9.52) and hurts deep (c2048 7.43 > 7.03) - long-context creative
(the production use) sits on the wrong side. Production default stays gate1; gate3 remains
available as GGML_CUDA_W4A4_DOT8=3 for short-context/quality-sensitive experiments.
Future lever: the -21% is a per-16 read-path cost (scale-pair loads + sub-block indexing
in mmq_gemm_repacked_impl), not fundamental - a tuning pass on the U4X16 fetch (vectorized
28B-stride handling) could narrow it; not scheduled.

Upstream framing (unchanged): the multi-seq plane-offset fix is the
PR-ready piece; sections 47-49 are the supporting story (bug, diagnostic methodology,
sub-block scale exploration with measured KLD/perf tradeoffs) if upstream wants the
accuracy context.

### 49e. Selective gating REFUTED + depth-table chunk-confound CORRECTED (2026-10-02, branch perf/w4a4-per16 )

Selective per-16 (GGML_W4A4_ONLY filter extended to the gate-3 u4x16 selection, commit
 - the phase-1 filter only covered gate-1 u4act): ffn-per16 retains 40.9% of
gate3-full's KLD recovery at 75.1% of its pp cost; attn-per16 30.9% at 48.6%. Both strictly
dominated by full per-16 (recoveries subadditive 71.7%, costs superadditive 123.7%, matching
the 48a residual-stream superadditivity). SELECTIVE DIRECTION DROPPED; the filter extension
stays as a debug/attribution tool. Gate wiring note: the phase-1 filter never covered gate-3.

CHUNK-CONFOUND CORRECTION (agent-verified stock controls): the 7.0322 "c2048" values in
47/49 were run at --chunks 8 (16k tokens) while the stock control and gate3 legs used
--chunks 64 (131k tokens) - different test slices, invalid comparison. Matched-config remeasure:
| depth leg (c2048 -b2048 -ub2048 -fa on, chunks 64) | PPL |
|---|---|
| stock Q4_0 | 7.4939 +- 0.0753 |
| W4A4 gate1 | 7.7364 +- 0.0775 (remeasured this session) |
| W4A4 gate3 per-16 | 7.4332 +- 0.0736 |

CORRECTED depth story: at production depth gate3 is -0.06 BELOW stock (parity), gate1 is
+0.24 ABOVE (+3.2%). The 49d "gate3 hurts deep" conclusion was an artifact; per-16 is
parity-or-better vs stock at BOTH depths while keeping the W4A4 prefill win (+6% single,
+21% TP over stock at pp2048 with AR). c512 stock re-confirmed 8.4570 +- 0.1217 (ref 8.4203,
within CI). Effective summary: gate1 = +pp but +3.2% PPL at depth; gate3 = +pp, stock-parity
accuracy at depth, +1.4% at shallow.

### 49f. 32B layout tuning: negative result + profiling caveat (2026-10-02)
Tested u4x16 28B -> 32B padded stride (alignment hypothesis: 28B stride forces scattered
u32 loads). Numerics bit-identical (PPL 8.5422, KLD 0.090164 - pure layout move confirmed).
PERF NOT RECOVERED: interleaved bench gate3-32B pp512 986.7 / pp2048 979.6 / pp4096 962.2
vs gate1 1268-1282 = -22..-23%, same as 28B. Alignment was not the cost; the per-16 scale
application (2x decode/apply per 32-value group in the tiled inner loop) is intrinsic to
the current kernel design. Layout reverted; per-16 stays env-gated (DOT8=3), production
remains gate1. Future option: scale-application restructure (hoist/pack) - deeper rework,
not scheduled.

PROFILING CAVEAT: rocprofv3 (system 7.2.3) cannot attach to TheRock-nightly-runtime
binaries ("register fat binary failed") - kernel-time attribution of the -21% is
unmeasured. Correction: the "98.9s mmq_gemm / 1.5s quantize" figures quoted mid-session
were from a STALE Sep-25 results.csv (M4-era), not this program; do not cite them.

### 49g. Kernel-time attribution CLOSED (rocprofv3 per-gate-cwd workaround, 2026-10-02)
Profiling succeeded after the agent fixed the invocation (rocprofv3 writes to CWD; per-gate
directories + fresh CWD per run). Matched workload, 1984 GEMM calls per gate, device ROCm1:
| kernel | total | per-call |
|---|---|---|
| mmq_gemm_repacked U4ACT=true (gate1, per-32) | 2718.5 ms | 1.37 ms |
| mmq_gemm_repacked U4X16=true (gate3, per-16) | 3343.7 ms | 1.69 ms |
Per-call delta +22.9% inside the GEMM kernel itself = the -21.3% wall clock. Attribution
CLOSED: the per-16 cost is intrinsic to u4x16 kernel execution (2x scale decode/apply per
32-value group in the tiled inner loop + 32B LDS footprint); zero contribution from
dispatch, quantizer, or AR. Recovering it requires restructuring the scale-application
math, not layout or launch changes. Traces: rig/w4a4-prof32/{gate1,gate3}/prof_kernel_trace.csv.

### 49h. 49g refinement from the tuning agent's fresh profiles (no U4P8 aliasing; epilogue ALU is the cost)
Fresh rocprofv3 traces (per-gate cwd, both gates captured cleanly) refine 49g on two points:
1. NO ALIASING: gate3 dispatches its own <false,2,4,Q4_0,false,false,true> U4X16 instantiation
 with quantize_mmq_q8_1_u4x16 - the "U4P8=true template" reading came from the stale Sep-25
 results.csv. GEMM totals: gate1 2.719 s vs gate3 3.344 s (+23.0%), quantizer delta noise.
2. COST LOCATION: same VGPR/SGPR, no spill, no occupancy difference between gates - the +23%
 is the per-16 EPILOGUE ALU (2x {half2->float2 converts, sWs16 float2 loads, FMA chain} vs
 gate1's single per-32 fold). 28B->32B padding confirmed irrelevant (fetch was never the
 bottleneck; -20.5/-22.4/-22.6% across pp512/2048/4096, unchanged from 28B).
Follow-up path if ever funded: pre-fold weight scale d into d4/m at prefetch (sWs16 extension)
and/or fp32 scale storage - removes the delta ALU ops from the epilogue; layout changes are a
dead end. Patched sources preserved at rig/w4a4-32b-patch/; tree reverted clean after
the experiment.

## 50. gfx906 wave64 idiom comparison vs Strata (2026-10-04, branch bench/gfx906-idiom-compare )

Standalone microbenchmarks (rig/w4-bench/*.cu), interleaved A/B, 7 reps, median+spread,
MI50. Full report: rig/w4-kernel-compare.md; branch has bench sources.

| idiom | verdict | numbers |
|---|---|---|
| wavefront-per-row MMVQ butterfly (Strata) | PARITY - ours 4.6x faster at K=4096/rows=2048 (449 vs 97 GB/s: our repacked layout streams, their 18B interleaved stride caps ~145 GB/s); theirs wins 1.3x only at rows=128 launch-bound | PARITY |
| dp4a(g^m,u)-dp4a(m,u) sign trick | N/A - Q4_0 repack has no sign nibble to mask; where both apply ours = 0.76x of the trick's ALU cost | N/A |
| router top-k register argmax (one wavefront) | PARITY - our topk_moe 1.75x faster at E=128/256/512 k=10 (11.2 vs 19.6us) | PARITY |
| skinny-GEMM K-split + reduce (Strata fused_gr.cu:969 observation) | PORT - at ne11=16-32 rows=1536: K-split 2.6-3.0x faster; 1.1-1.4x at 128 blocks; CAVEAT: K-split reduction reorders fp32 sums (maxrel up to 1.4 at ne11>=32 rows=1536, 1.0e-5 at 16) - numerics-changing, needs a determinism decision before wiring | PORT (conditional) |
| v_perm_b32 IQ4 lookup | PARITY - identical 5-perm sequence both sides; __byte_perm confirmed 14-17x slower on this ROCm (avoid) | PARITY |

Key positive: idiom 4 (K-split for narrow-token GEMMs) measured 2.6-3.0x at ne11=16-32 -
the spec-decode verify-window and small-prefill-chunk regime. Env-gated port sketch in the
report; numerics caveat must be resolved (fp32 sum reordering vs our byte-identical
standard - possible middle ground: K-split only where a tolerance gate allows).
Also confirmed: HIP 7.2 __shfl_*_sync static-asserts on 32-bit masks; our vendors/hip.h
shim is required for these idioms (cost one false-alarm HW exception during bring-up).


## 51. K-split GEMM + async PLE reader (2026-10-05, branches perf/w4a4-dot8-gfx906 , perf/ple-async )

### 51a. K-split narrow-token GEMM (GGML_RP_KSPLIT=1, default OFF)
grid.z K-slices + fixed-order reduce kernel; fires only for Q4_0 w32 path, 16<=ne11<=64;
KSPLIT unset folds to stock at compile time.
Numerics: q8_1-activation path bit-exact vs stock at all split factors/shapes. u4act path:
the u4act kernel is itself approximate (stock-vs-fp64-ref maxrel up to 2.45e-02); K-split is
error-neutral-to-positive vs that reference (one shape 10x closer). Model-level: gate-inert
control pair bit-identical KLD 0.068316; gate-firing shape (ub32) KLD +0.0008 (+1.1%, inside
the 0.1226 envelope); PPL c512 +0.09%, c2048 -fa +0.005%. DOT8=3+KSPLIT combo: no crash.
Perf (llama-bench, interleaved, 2 reps x r3): pp512 +26.5%, pp2048 +22.4% at ub32; +8.3% at
ub64; decode (ne11=1) parity by design. KSPLIT also cuts run spread 10-30x in fired cells.
Verdict: SHIP as env-gated. Spec-decode verify windows (ne11 16-64) are the target regime.

### 51b. Async PLE reader (Strata PleReader+DirectFile port, perf/ple-async )
Prototype: GGML_PLE_ASYNC=1 swaps the PLE table pointer to a 5.9MB staging buffer; per-ubatch
O_DIRECT prefetch + index remap (byte-identical by construction); mmap memcpy fallback.
Bench (Kioxia NVMe, 1500 tokens, 16 rows/token): mmap warm-resident (prod) 2289.5us p50 vs
dio-async with collect hidden behind 300us compute 303.1us p50 (7.5x), p95 ~305; + 1M-row
cache 303.6us. Cold-mmap reference 4013us/token (confirms 44b). Reader byte-exact vs mmap:
320,000 row checks, 0 fails.
CAVEATS: (1) warm-resident mmap is 2.3us/token - async only pays when the table cannot stay
page-cache-resident (our prod PREFAULT keeps it warm, so no immediate deploy); (2) a live
mmap section on the same file serializes O_DIRECT reads - the loader must drop the mapping
when the gate is on (ours does); (3) E2E model-level check pending a host GPU-stack recovery
cycle (host power cycle needed; the wedge predates this work) - reader-level byte-exactness
proven, graph-level integration gated but unexercised.
Verdict: CONTINUE (scoped) - mechanism proven, port clean, ship later behind the env gate
when a use case needs the table off a slow/resident-constrained disk.
Files: rig/ple-port-analysis.txt (phase-A mapping: gather site qwen4exp.cpp:1465, row-index
source llm_graph_input_ple::set_input qwen4exp.cpp:1241-1300, runs pre-graph at
llama-context.cpp:1778), rig/ple-port-stage/correctness.sh (awaiting host recovery),
prototype on perf/ple-async (11 files, +1141).

## 52. Standard upstream-PR benchmark suite: first run on merged upstream (, 2026-10-06)

New committed suite (rig/bench_suite.sh): one command produces every table a PR draft needs -
pp/tg sweep x 3 quants x repack on/off, K-split on/off, PPL+KLD per quant vs bf16 base, greedy
seal. Model: bartowski Qwen3.5-9B (bf16 + locally quantized Q4_0/Q4_K_M/Q8_0). Dataset:
wikitext-2-raw test. Single GPU ROCm1, -t 16, -lm dio.

### Results (build , post-upstream-merge)
Perf pp512 (t/s): Q4_0 939.9/937.6 (repack=1/0, +0.2% - NOT significant post-merge)
Perf tg128 (t/s): Q4_0 80.87 vs 75.46 (repack=1/0, **+7.2%** - repack gain is tg-side)
Q4_K_M repack: pp ~parity, tg 68.6 vs 68.6 (no repack path for Q4_K on this build)
Q8_0 repack: pp ~parity, tg128 58.9 vs 59.1 (parity; Q8_0 takes stock path)
K-split (ne11 16/32/64 pp): 365/490/553 vs 366/493/553 = PARITY at these shapes
KLD mean: Q4_0 0.0687, Q4_K_M 0.0573, Q8_0 0.0012 (vs bf16; consistent with sec-46)
Seal: PASS (verbose enumeration, primes 2-29 in order)

### FINDING: repack gain is TG-SIDE ONLY on merged upstream
Pre-merge (sec-46) showed pp512 +5.7% from repack; post-merge it is gone (+0.2%, noise).
Upstream master's own changes (fused router, hc gates, etc.) appear to have absorbed the pp
side. tg128 +7.2% remains - the PR-1 claim should be reworded from "+pp and +tg" to
"+tg128 7% at zero pp cost, numerics-identical".

### K-split PARITY note
The 2.6-3x win from the standalone bench (bench4_skinny) does NOT appear at model level on
these shapes - the standalone's synthetic narrow shape was not representative, or upstream
master already improved small-batch GEMM. K-split stays env-gated OFF; re-test at true
verify-window shapes (ne11 2-8) before further investment.

**SCOPE CORRECTION (2026-10-08, verified - kernel-level rocprofv3, traces
rig/ppmax/ksplit_probe/, resolution rig/ppmax/ksplit-resolution.md): k-split positives
hold ONLY on the stock q8_1 path (GGML_CUDA_W4A4_DOT8 unset). Under gate1-u4act (DOT8=1)
k-split is NEGATIVE on both lineages ( and integrated): kernel -29.8%/call
stock-path vs +13.2%/call gate1-path, measured 2026-10-08. The sign follows the activation
gate, not the lineage - the u4act kernel is dp4a/ALU-bound so K-slicing does not shrink
per-slice cost, and reduce + doubled dispatch make it net-worse. Keep GGML_RP_KSPLIT
default-off; NEVER combine it with DOT8=1.**

## 54. Layer-selective W4A8: GGML_CUDA_W4A4_SKIP frontier (2026-10-08, branch perf/w4a4-skip-layers )

Env `GGML_CUDA_W4A4_SKIP` = comma-separated tensor-NAME prefixes; any Q4_0 GEMM input whose
name starts with a prefix is excluded from u4act entirely and runs the stock q8_1 activation
path (bitwise-stock for those GEMMs - selection, zero new numerics). Unset/empty =
bit-identical gate1 dispatch. Composes with GGML_W4A4_ONLY (ONLY applies first, SKIP
subtracts); under gate 3 an excluded tensor also loses the per-16 u4x16 GEMM. Dense
(q8_repack/mul-mat.cu) + MoE (mul-mat-id.cu) dispatch sites gated; static-once parse; one
WARN audit line at startup lists the active prefix count (WARN not INFO: llama-perplexity
default verbosity filters INFO). Helper scripts/w4a4_skip_expand.sh expands "4-18" ->
blk.4.,...,blk.18. Build rig/build-skip; raw data rig/w4a4-skip/.

PHASE-A SEALS (all PASS): (a) gate-off parity - SKIP unset chunks-32 PPL 12.5836 +- 0.36763
== gate1 reference, identical across two runs, audit marker absent = gate provably inert;
(b) mechanism proof - SKIP=late chunks-2 PPL 4.8276 vs gate-off chunks-2 5.2859: dispatch
differs and sane (not ~1e5 garbage), marker present; (c) determinism - gate-on chunks-2 legs
byte-identical PPL lines across reruns.

PHASE-B FRONTIER (full-corpus wiki.test, KLD vs f16-c512-ch128 base; anchors: stock-repack
floor KLD 0.0684 / pp512 938.8, gate1 0.12215 / 1249.75 / tg 82.54, per-16 0.090147 / 984 /
82.56):

| arm (SKIP set) | KLD | % gap closed (gate1->stock) | pp512 vs stock | tg128 | depth PPL (c2048 ch64) |
|---|---|---|---|---|---|
| SKIP-early L4-18 (15 prefixes) | 0.097917 +- 0.001819 | 45.1% | +14.4% (1073.65) | 79.62 | 7.6130 |
| SKIP-late L19-31 (13 prefixes) | 0.103386 +- 0.002009 | 34.9% | +18.4% (1111.47) | 70.02 SUSPECT | 7.5888 |
| SKIP-narrow27 (h32-sel set) | 0.116919 +- 0.002171 | 9.7% | +25.7% (1180.56) | 79.67 | 7.7019 |
| full-skip L4-31 (bonus, ch32 shallow only) | - | - | - | - | shallow 7.2532 (<= per-16 ref 7.2774) |

MECHANISM FINDING: coverage, not granularity. SKIP-narrow27 and h32-sel cover the IDENTICAL
27-tensor outlier set; replacing those GEMMs with stock q8_1 recovers 9.7% of the
gate1->stock KLD gap vs 6.0% for finer H32 WHT scales on the same tensors (16.3% vs 10.1%
of the per-16 gap). Fixing the 27 by ANY mechanism recovers <=16% of the per-16 gap - the
per-16 win is an all-tensor coverage effect, not a 27-tensor effect. Skipping ALL of L4-31
approaches the stock floor (ch32 PPL 7.2532, direction only): quality scales with skipped
coverage and skip is the cheapest mechanism (pure selection, no new numerics).

EFFICIENCY (%gap closed per %pp cost vs gate1): SKIP-early 45.1/14.1 = 3.20 vs per-16
59.5/21.3 = 2.80 - skip trades a lower ceiling for a better exchange rate.

FLAGS: SKIP-late tg128 70.02 +- 2.13 (-15.2% vs gate1) is SUSPECT and UNCONFIRMED - early
and narrow27 hold tg at -3.5%/-1.0% with equal or larger prefix counts; 13 prefixes should
not cost 15% tg. Possible slot contention (ROCm0) or a real decode-path effect; not re-run.
Late arm's accuracy legs are clean (marker present, seal deterministic).

DISPOSITION: SKIP-early = ADOPT-CANDIDATE (best quality arm: KLD -19.8% vs gate1, shallow
8.5739 / depth 7.6130 both improved, seal PASS primes 2-29, tg near parity) pending
interleaved A/B vs gate1, matched dPPL vs stock, and 64k-context legs. Gate stays env-gated
OFF by default; unset = bit-identical gate1. Raw: rig/w4a4-skip/phaseA-kernel.md,
rig/w4a4-skip/phaseB/phaseB-leg{0,1,2}.{md,json}.


## 55. Review-fix verification: stride-union Q5_1, gate semantics, engagement markers (2026-10-10, branch fix/review-2026-10-10, commit )

Six review fixes on top of (B1 stride-union Q5_1+Q4_0, B2 MoE DOT8=0 semantics,
B3 engagement markers, Q2 K-split stock-path guard, Q4 MoE fused Q4_0 loud abort, N1 34->36 B
comment fix). Host-side regression test for B1: pre-fix 3 FAIL, post-fix ALL PASS. GPU re-verify
on rig/build-skip-rebase/bin (build (11504) + fixed tree), single GPU ROCm2,
serialized on rig/gpu.lock, VRAM baseline 10.87 MB/GPU before and after every leg.

### 55a. Parity sweep (llama-bench -dev ROCm2 -ngl 99 -fa 1 -t 128 -r 1, all with REPACK_Q4_0=1)
| config | pp512 | tg128 | ref ( table) | delta pp/tg |
|---|---|---|---|---|
| DOT8 unset (stock-repack) | 946.88 | 81.41 | 948.6 / 82.3 | -0.2% / -1.1% |
| DOT8=1 (gate1) | 1244.02 | 82.25 | 1249.75 / 82.3 | -0.5% / -0.1% |
| DOT8=3 (per-16) | 1015.98 | 83.45 | ~984 / 82.56 (54 anchor) | +3.3% / +1.1% |
| KSPLIT=1 (DOT8 unset) | 951.87 | 81.71 | 948.6 / 82.3 | +0.3% / -0.7% |
| DOT8=1 + KSPLIT=1 | 1243.28 | 82.49 | 1249.75 / 82.3 | -0.5% / +0.2% |

All within +/-3% of the reference table (DOT8=3 vs the 54 per-16 anchor +3.3%, same ballpark).
No crashes. KSPLIT=1 inert at pp512 as expected (fires only at 16<=ne11<=64). DOT8=1+KSPLIT=1 ==
gate1 within noise: the Q2 guard provably keeps K-split off the u4act path (52a: the combo was
known-negative). Engagement markers: llama-bench silences ggml logs (known, phaseB precedent);
in llama-perplexity logs the DOT8 gate marker prints ("CUDA w4a4: GGML_CUDA_W4A4_DOT8=1 gate
ACTIVE (per-32 u4 activations)") and the q8_repack witness line proves admission. NIT: the
REPACK_Q4_0 admission WARN itself does not surface in llama-perplexity stderr (code verified
present + correct in the .so; witness line "q8_repack: repacking q4_0 tensors" is the effective
engagement proof).

### 55b. Q5_1 regression (B1 stride-union; Q5_1 quantized locally from Qwen_Qwen3.5-9B-bf16.gguf
-> rig/q5_1-regression/Qwen3.5-9B-Q5_1.gguf)
Stock (all gates unset): loads + 5-token gen clean, coherent; PPL ch2 7.3137 +/- 0.89207; zero
gate markers = gates provably inert. REPACK_Q4_0=1 (Q5_1 hits the FIXED union-stride path - the
predicate is a union now): loads + 5-token gen clean, no OOB/fault, PPL ch2 7.3137 +/- 0.89207
(identical), witness line "q8_repack: repacking q5_1 tensors on device 2" present. PASS.

### 55c. gate1 PPL parity + KLD re-verify (fixed build)
gate1 shallow PPL (-c 512 -b 2048 -ub 512 --chunks 256): 8.7630 +/- 0.08938 - EXACTLY
reproduces the 54 gate1 shallow anchor (8.7630). (The brief's "~11.55-11.65" band matches no
recorded gate1 anchor in BENCH.md; 8.7630 is the 54 reference.)
KLD vs f16-c512-ch128 base (-c 512 -b 2048 -ub 512 --chunks 128):
| arm | KLD | expected | verdict |
|---|---|---|---|
| gate1 (REPACK_Q4_0=1 DOT8=1) | 0.122150 +/- 0.002028 | 0.12215 +/- 0.002 | EXACT |
| SKIP-early (L4-18) | 0.097917 +/- 0.001819 | 0.097917 +/- 0.0018 | EXACT |

BRANCH VALIDATED: fix/review-2026-10-10 reproduces every reference number within
band; Q5_1 union-stride path clean. Raw logs: rig/ppmax/logs/.


## 56. Phase-3 GPU ladder: Q4_1 u4 (gate 4) validated end-to-end (2026-10-12, branch q41-u4-probe , ROCm2)

FULL build rig/wt-q41/build-q41 + ISOLATION build rig/wt-q41-iso (, Sigma_a staging
forced SA=0.0f). Serialized on rig/gpu2.lock, VRAM baseline 10.87 MB/GPU before/after every leg.
VERDICT: GO on every leg. Full report: rig/ppmax/q41q4k-probe/phase3-gpu-report.md.

LEG 0 (exact-anchor regression, upstream 9B Q4_0): gate1 PPL ch256 8.7630 EXACT; gate1 KLD 0.122150 EXACT;
SKIP-early KLD 0.097917 EXACT; DOT8=3 PPL ch32 7.2774 EXACT (the 54 per-16 ref). The U4Q41 wrapper-threading
+ KSPLIT macro-arg shift perturbed gates 1/3 by zero ULP.

LEG 0b (isolation A/B): on the pure Q4_0 file the gate1 kernel counters are BIT-IDENTICAL FULL vs ISO
(VALU 68,111,311/call, LDS 6,271,669.7, WAVES 1440, n=992) - shipped path provably untouched. Kernel-level
isolation on the map file (U4Q41 actually runs): FULL 181,390,336 VALU/call vs ISO 180,114,432 = Sigma_a
staging +0.71% VALU, 0.00% LDS; vs stock Q4_1 repack on the same tensors (SKIP arm): +2.50% VALU, -26.8%
LDS. Far below the +22.9% U4X16 NO-GO bar. Disassembly (llvm-objdump on extracted hsaco,
mmq_gemm_repacked<...,Q4_1,...,U4Q41=true>): FULL 2192 vs ISO 2132 instrs; the entire delta is the forced-out
staging chain (+32 v_dot8_i32_i4, +8 cvt, +8 fma, +8 add, +6 s_waitcnt); 71 opcodes identical counts;
vgpr/sgpr identical (128/44) -> inner-loop ISA identity PROVEN, not argued. gfx906 exposes no dynamic
occupancy counters (SQ_LEVEL_WAVES/SQ_WAVE_CYCLES/SQ_BUSY_CYCLES unavailable); static metadata + SQ_WAVES
identical used instead. Bench-level FULL-vs-ISO on the map file is confounded ( predates the
superset redesign: ISO at gate4 leaves Q4_0 stock) - counters are the clean isolation.

LEG 1-2 (gate 4 full gates): (a) SUPERSET property on pure Q4_0 file: gate4 PPL ch256 8.7630 bit-identical
to gate1 (dPPL 0.000), pp512 within noise. (b) map-built Q4_1 file (bf16 -> Q4_0 + explicit
blk.{4..7}.ffn_down=q4_1 full-name overrides on the ffmap2 map; histogram Q4_0x169/Q4_1x4): pp512 g0 951.5 /
g1 1195.0 / g4 1197.2 -> gate4 gain +25.8% = 81% of the pure-Q4_0 gate1 gain (bar >=50%) PASS; tg128
+3.1% no-regress (6 reps, 6s drain); KLD internal anchors on the same file g0 0.053048 / g1 0.091156 /
g4 0.092895 - gate4 adds only +0.0017 over gate1 (4.5% of the Q4_0-u4 penalty), inside the u4 envelope;
greedy primes 2..29 EXACT; SKIP-under-gate4 runs Q4_1 tensors on the repacked STOCK kernel (rocprofv3
witness, admission gate-aware); VRAM g0 vs g4 resident sample -4.75 MB (noise) = canonical 5 bpw neutral.
Depth PPL c2048 ch64: g0 7.2473 -> g4 7.4194 (the Q4_0-u4 penalty dominates; superset dPPL on Q4_0 files 0.000).

## 57. SKIP KLD transferred to the 27B: f16 3-GPU base, skip-layer recovery does NOT transfer (2026-10-12, build-merge , rig/kld27b/)

sec 54's skip-layer frontier (GGML_CUDA_W4A4_SKIP) was measured on the 9B with a single-GPU f16
KLD base; the fleet A/B could only re-check the 27B by PPL (weaker instrument). This leg builds
the missing KLD base: ggml-org Qwen3.8-27B-BF16.gguf (53.8 GB) -> llama-quantize BF16->F16
passthrough (rc=0, 16.00 BPW) -> llama-perplexity --save-all-logits on 3 GPUs
(-dev ROCm0,ROCm1,ROCm2 -sm tensor, c512 b2048 ub512 ch128, wiki.test.raw); base PPL
6.8457 +- 0.09353, logits 16.2 GB. 54 GB weights + KV fit 3x32 GB tensor-split with no split
tuning. KLD legs on Qwen3.8-27B-Q4_0 (2 GPUs, same chunk protocol, gate markers verified):

| arm | Mean KLD vs f16 base | % gap closed (gate1->stock) | 9B sec 54 |
|---|---|---|---|
| gate1 (DOT8=1) | 0.062311 +- 0.000777 | anchor | 0.12215 |
| gate1+skipE blk.4-18 | 0.058537 +- 0.000707 | **12.6%** | 45.1% |
| gate1+skipL blk.19-31 | 0.057454 +- 0.000750 | **16.3%** | 34.9% |
| stock repack floor | 0.032452 +- 0.000374 | anchor | 0.0684 |

gate1->stock gap = 0.029859 on 27B vs 0.05375 on 9B: the gate costs ~44% less KLD on the 27B.
Bonus sec 55-style: gate1 PPL ch256 fresh 6.9896 +- 0.06770 (ch128 refs: gate1 7.0916, stock
6.9514) - depth closes ~73% of the ch128 gate1->stock PPL gap.

FINDING: skip-layer recovery does NOT transfer at fixed layer ranges. Closure collapses
45.1->12.6 (early) and 34.9->16.3 (late) and the early/late ordering INVERTS (9B early>late,
27B late>early). The blk.4-18/19-31 ranges are tuned to the 9B's 48-layer outlier band; on
65 layers they cover 23%/20% instead of 31%/27%, and the 27B's gate-damage mass sits later.
PPL-based closure from the fleet A/B (skipE 15.5%, skipL 29.0%) agrees: low-teens, not 45%.
DISPOSITION: do not ship skipE/skipL on the 27B on the strength of sec 54; re-derive any 27B skip
list via per-layer KLD attribution on this f16 base. Raw: rig/kld27b/.


## 59. Adopt-validation: stock+ksplit A/B (27B + community-FT MoE) + first MTP x W4A4 interaction measurement (2026-10-13, build-merge b11507, rig/ppmax/adopt/)

PART A - ksplit adoption gate on the 27B (Qwen3.8-27B-Q4_0, ROCm0, -t 128, r=5): stock vs
GGML_RP_KSPLIT=1: pp16 +10.6% (81.19->89.78), pp32 +7.2%, pp64 +1.2%, pp512/pp2048/tg128 parity
(<=0.7%). PPL c512 ch256 BIT-IDENTICAL (6.8534 +- 0.06639 both arms, per-chunk arrays byte-equal;
ch128 prefix 6.9514 matches sec-57 stock ref). per16 (DOT8=3) arm: +4..+6% pp, and per16+ksplit ==
per16 within noise - the ksplit gate (q8_repack/mul-mat.cu: !u4act && !u4x16 && Q4_0 && 16<=ne11<=64)
is self-guarding, so GGML_RP_KSPLIT=1 is inert under every DOT8 gate and on non-Q4_0 quants.
community-MoE-35B-A3B Q4_0 MoE (2-GPU tensor, r=3): ksplit parity (-0.6..+0.4%); per16 +2.3..2.8% pp,
tg spread +-15 swamps the -11% point est. VERDICT: ADOPT-READY. llama-swap per-model env diff
drafted (26 models get GGML_RP_KSPLIT=1, the 2 DOT8 models untouched) at rig/ppmax/adopt/
config.yaml.ksplit-draft; validated by a live llama-swap instance on :19191 (28 models listed,
zero errors). Prod config NOT touched (it runs --watch-config).

PART B - MTP x W4A4 on the same 27B (llama-server -c 4096, /completion temp 0, ~775-tok wikitext
prompt, tg128 x3 reps, draft-mtp n_max=4 p_min=0.75). ARCH FACT: both the main gguf and the
mtp-Qwen3.8-27B draft module are general.architecture=qwen35 (NOT qwen4exp) - the hc_norm fix
 is qwen4exp-specific and irrelevant; zero build_hc_mix SIGABRTs, qwen35 MTP loads clean.
The "mtp" gguf is a draft-only module (18 tensors, blk.64.nextn.* + embd/output); the main Q4_0
gguf carries a NATIVE nextn head (qwen35.nextn_predict_layers=1) which works with no -md.

tg128 t/s: stock-noMTP 32.4 | stock+MTP 48.7 (+50%, mean acc len 4.13) | gate1+MTP 29.8 |
gate1+ksplit+MTP 29.6 | per16+MTP 30.6 | per16+ksplit+MTP 30.8 | native-MTP 51.1 (best).
no-MTP gate baselines: gate1 33.2, per16 33.2. Accept-rate per pos is NOT hurt by the gates
(gate1 4.28, per16 4.39 mean acc len vs stock 4.13; ksplit adds zero accept-rate delta).
FINDING (the inversion): MTP's win COLLAPSES under the gates - gate1+MTP is -10% vs gate1 no-MTP,
per16+MTP -8%: MTP verify runs at ne11 ~= n_max+1 = 5 rows, below the gates' winning window and
below ksplit's 16-row floor (ksplit is inert under DOT8 by design anyway), where u4act/u4x16 are
slower than stock q8_1. ksplit x MTP coexistence is safe but inert (c4 ~= c3, c6 ~= c5 within 0.7%).
pp unaffected by MTP. VRAM: 5.74 GB no-MTP, +0.98 GB draft module. No TP AR warnings (single GPU).
DISPOSITION: ship GGML_RP_KSPLIT=1 globally (env-only); NEVER combine DOT8=1/3 with MTP on
qwen35-family models; for Qwen3.8-27B Q4_0 the best decode config is native MTP, no -md, stock.

## 60. Per-block least-squares Q4_0 scale: encode-side headroom probe - NEGATIVE, imatrix path is already LS (2026-10-13, branch exp/q40-ls-scale off , rig/bench-results/q40ls/)

R3 accuracy-headroom item 1: replace the Q4_0 per-block scale choice with iterated least-squares
(seed max|x|, then d = sum(w?x?l)/sum(w?l?l) with l = clamp(round(x/d)), 4 sweeps), env-gated in
ggml-quants.c (GGML_Q40_LS=sweeps; GGML_Q40_LS_MODE=1 weighted by the imatrix weight vector
w=qw?sqrt(sigma2+x?), 2 = unweighted, the brief's literal sum(scale?q - w)?). CPU-only quantizer
(rig/wt-q40ls/build-cpu); files re-encoded from the same bf16 source + map-ff4 + train-split
imatrix as ffmap4, size identical (6,288,666,624 B). Control arm GGML_Q40_LS=0 reproduced ffmap4
**bit-exactly** (md5 49b7aed8...), so every delta below is the LS scale alone. KLD c512 ch128 vs
f16-c512-ch128 base, wiki.test, build-merge, ROCm2:

| arm | gate0 (weight floor) | vs 0.041157 | gate1 (W4A4) | vs 0.076382 |
|---|---|---|---|---|
| ffmap4 anchor (known) | 0.041157 | - | 0.076382 | - |
| LS ctrl (LS=0, bit-exact) | 0.041157 +- 0.001281 | 0.0% | 0.076382 +- 0.001605 | 0.0% |
| LS mode1 s4 (imatrix-weighted) | 0.050086 +- 0.001615 | **+21.7%** | 0.086219 +- 0.001744 | **+12.9%** |
| LS mode2 s4 (unweighted) | 0.051336 +- 0.001534 | **+24.7%** | 0.087302 +- 0.001774 | **+14.3%** |

Weight-MSE cross-check (blk.10.ffn_gate dequant vs bf16, gguf-py dequantize): ffmap4 9.5175e-07,
m1s4 +5.0%, m2s4 +3.8% - LS also loses on plain MSE while it *shrinks* max|err| (3.13e-2 ->
1.05e-2): the max|x| seed + imatrix pattern search is better than iterated LS even on the
unweighted block-MSE proxy the LS objective itself optimizes.

VERDICT: **NO - LS scale regresses, do not pursue.** Success bar was >=2% KLD *reduction*; measured
is a 12.9-24.7% *increase* (5-7x the per-arm KLD error bar, and the control arm reproduces the
anchor bit-exactly so the delta is systematic, not drift). Mechanism: the imatrix Q4_0 path is
already least-squares - make_qx_quants(rmse_type=1) computes scale = sumlx/suml2 over the
imatrix-derived weight vector, then searches discrete iscale variants (-nmax-0.1i)/max, i?[-9,9],
picking the best **clamping-pattern** by sumlx?/suml2. Iterated LS from the max|x| seed converges
to the continuous stationary point of its own objective but drops that discrete pattern search,
and the pattern search is what wins: LS lands in a worse local optimum of the *rounding* problem
(the clamping pattern at the LS scale is not the one the search finds). The brief's literal
unweighted LS is worse still: unweighted block MSE is the wrong proxy for KLD - the imatrix
weighting is load-bearing. R3's "possible negative near saturation" prediction confirmed: the
imatrix encode is saturated with respect to scale-selection changes; headroom, if any, is in the
weight *values* (SmoothQuant-style rescale, see rig/ppmax/smoothquant-feasibility.md) or the
activation side, not the Q4_0 scale rule. Raw: rig/bench-results/q40ls/; code: branch
exp/q40-ls-scale (env-gated, inert by default).

## 61. llama-bench -dev delimiter fix (comma = device list, slash = cross-product) + first proper sharded-model bench (2026-10-13, branch fix/llama-bench-dev-comma)

Root cause of the "llama-bench cannot load sharded multi-GPU models" mystery: llama-bench parsed `-dev` with **inverted delimiters** vs every
other tool. common/arg.cpp `parse_device_list` (llama-server, llama-perplexity) treats COMMA
as the device list for one model instance; llama-bench split the value on COMMA into separate
benchmark combos and each combo on SLASH into a device list. So `-dev ROCm0,ROCm1,ROCm2
-sm tensor` created three single-device combos, each trying to load the full 64.5 GiB shard set
onto one 32 GB GPU - silent load failure (error only under --verbose). The slash form
`ROCm0/ROCm1/ROCm2` worked but was backwards vs the codebase convention.

Fix (one file, tools/llama-bench/llama-bench.cpp, +9/-3): swap the delimiters -
`parse_devices_arg` splits on comma (device list per instance, matches common parser); the
`-dev` arg handler splits on slash (cross-product of device-set combos); help text updated.
Verified: `-dev ROCm0,ROCm1 -sm tensor` -> one tensor-split instance; `-dev ROCm0/ROCm1` ->
two single-device combos (slash semantics unchanged for existing users). Build RC=0.
Branch pushed to gitea + github fork; upstream PR to be opened by Brad.

The fix unblocked llama-bench on sharded multi-GPU models. First such bench (dense
production model Q4_0, 64.5 GiB, 2 shards, 3x MI50, `-dev ROCm0/ROCm1/ROCm2 -sm tensor
-ngl 999 -fa on -lm dio -p 813 -n 128 -r 3 -t 128`): gate1 (DOT8=1) +34.4% pp813 vs
stock, decode within noise (per-model raw numbers withheld from the public record).

Cross-check vs server-based prod numbers: bench agrees within 1.9% (stock prefill),
7.9% (gate1 prefill), <=3.7% (decode) - bench fidelity confirmed, r=3 noise <=0.4 t/s. Gate1 prefill delta reads larger in bench (+34.4%) than server
(+22.2%): bench pp813 runs the whole prompt as one batch and misses the server's mixed
chunked-prefill/decode overlap that partially hides the stock path's prefill cost. Direction
and magnitude class (large prefill win, decode within noise) both confirmed.

## 62. W4A4 width floor GGML_CUDA_W4A4_MIN_TOKENS: gate+MTP coexistence via per-GEMM auto-fallback (2026-10-14, master , rig/ppmax/mtp-floor-validation.md, rig/bench-results/{gate-crossover-threads,mtp-floor}/)

PHASE A: (1) the two llama-bench columns after ngl are threads (-t) and fa (flash_attn), NOT
ubatch - the sec 61 crossover ran -t 128 -fa on. (2) Crossover is thread-count-INDEPENDENT: spot
widths 8/32/128 at t=32 vs t=128 give ratios 0.357/0.351, 0.846/0.840, 1.277/1.280 (r=3) - the
sign flip and magnitudes are identical, so a width floor is not a -t artifact.

PHASE B: GGML_CUDA_W4A4_MIN_TOKENS (default unset = OFF = byte-identical shipped behavior).
The u4act predicate in q8_repack/mul-mat.cu gains (w4a4_min_tokens <= 0 || ne11 >=
w4a4_min_tokens) at the same per-GEMM dispatch site where ksplit checks 16<=ne11<=64. Below the
floor a call runs the REPACKED STOCK q8_1 activation path (SKIP-tensor semantics). Dense path
only; MoE (mul-mat-id.cu) untouched. Build RC=0.

PHASE C (27B, llama-server, tg128 x3, draft-mtp n_max=4 p_min=0.75, external draft module):
stock+MTP 49.05+-0.27 | gate1 NO-floor +MTP 29.24+-0.02 (-40%, the sec 59 verify hit reproduced) |
gate1 +floor128 +MTP 47.29+-0.28 (-3.5%: VERIFY HIT GONE; residual is prefill-numerics accept-
sequence jitter - verify/decode run bit-stock under the floor, mean acc len 4.07 vs 4.13).
no-MTP baselines: floor 31.60 vs stock 31.50 (parity). pp sweep gate1+floor128 (r=3): pp4 68.04,
pp8 85.60, pp16 82.40, pp32 156.94, pp64 187.99 = STOCK PARITY at every narrow width (the sec 61
pp16-64 losses are gone); pp128 308.78 (+27.9%), pp512 405.82 (+32.9%) = FULL gate win.
pp512+tg128 combo: gate1+floor pp 403.96 (+31.4% vs stock 307.53), tg 31.15 (parity).

PHASE D (flagship): native-MTP + gate1 + floor128 = 49.89+-0.37 t/s (acc len 4.32) vs native-MTP
stock 51.85 (-3.8%) WITH +31% pp512 - gate prefill win and MTP decode win coexist in one server
for the first time. VERDICT: GO. Recommended MTP+gate config: DOT8=1 MIN_TOKENS=128. Caveats:
chunked prefill below the floor runs those chunks stock (correct, no win); MoE path not floored.

## 63. MTP output-consistency under the width floor: temp-0 sequence identity + accept-length distributions (2026-10-14, master , rig/mtp-consistency/, rig/ppmax/mtp-floor-validation.md)

Closes the last unmeasured corner of sec 62: does gate1+floor128 change WHAT the model outputs with
MTP, not just how fast? Harness: cons-cell.sh/cons-run.sh (flock gpu1.lock, PORT GUARD pre/post +
orphan pgrep, per-arm gen-count check). 27B + external MTP draft, sec 62 server flags. 5 prompts
(code/prose/math/short/long-form, 21-61 tok) x 20 reps, /completion temp 0 seed 42 n=256
logprobs+top5; plus a >floor leg (775-tok wikitext x 20) where prefill actually runs u4act.

DETERMINISM: temp-0 MTP is fully deterministic - 300 gens, all 20 reps per prompt token-identical
within every arm (stock+MTP, gate1+floor+MTP, stock no-MTP). No logprob fallback needed.

| pair | token-identical | divergence |
|---|---|---|
| stock+MTP vs gate1+floor+MTP (short prefill < floor) | **100/100 = 100%** | none - bit-stock path both arms |
| stock+MTP vs gate1+floor+MTP (775-tok prefill > floor) | 0/20 | ONE flip at pos 2 in all 20 reps: " Ben"(-1.182) vs " Robert"(-1.293), 0.11-nat near-tie; post-flip coherent, both hit cap |
| stock+MTP vs stock no-MTP (greedy continuity ref) | 20/100 | first flip pos 9/31/62/205; mean logprob gap at flip 0.018 nats (median 0.013, max 0.047) - near-tie flips |

Accept-length distributions (per-gen mean acc len, n=100/arm): stock+MTP 3.448 +- 0.584 vs
gate1+floor+MTP 3.448 +- 0.584 - identical (identical sequences). Per-prompt values identical
arm-to-arm {math 4.14, code 3.97, short 3.60, prose 2.77, long 2.76}. Wikitext leg: 4.480 vs
4.460 (sd 0 within arm; ?0.02 sequence jitter, same shape as sec 62). Output-length distributions
identical per prompt across all arms. Degenerate outputs (empty / 3x-repeated token window):
**0/300+ gens in every arm.**

PRIME SEAL with floor (9B Q4_0, no MTP, llama-cli greedy temp 0 top-k 1 seed 42 n 700, ROCm0):
stock vs gate1+floor128 output BYTE-EXACT (raw diff = progress spinner + t/s footer only), primes
2..29 all correct - sec 56 non-MTP greedy identity survives the floor.

VERDICT: **GO on output consistency.** Below the floor the gate+floor config is output-identical
to stock+MTP at temp 0 (100%); above the floor the only effect is single near-tie logit flips at
the prefill-numerics boundary (0.11 nats), with unchanged accept-length distribution shape and
zero degeneration or compounding degradation. Flagship config (gate1 + floor128 + MTP) is safe for
greedy/temp-0 serving workloads.

## 64. Long-context validation of the flagship config (gate1+floor128) on Qwen3.5-9B: PPL ladder flat to 131k, perf +14..+52% pp / decode flat, MTP coexists at depth -> GO (2026-10-15, master build-merge, rig/ppmax/longctx*)

Model: Qwen3.5-9B Q4_0 (qwen35, trained ctx 262144 - 150k is INSIDE trained range, no
NIAH-degradation confound). Hybrid attention (full-attn every 4th layer) -> f16 KV ~18 KB/tok,
~2.7 GiB at 150k: c=150000 fits ONE MI50 with FA on. Arms: stock vs DOT8=1+MIN_TOKENS=128
(floor-ACTIVE log line confirmed in every gated run). ROCm0 PPL / ROCm1 bench / ROCm2 MTP,
per-GPU flock. Full report: rig/ppmax/longctx-validation.md.

LEG A - PPL vs context (llama-perplexity default mode, -b/-ub 2048 n_seq=1 per 47e, -fa on,
corpus = first 330k tok of ppl-test.txt, same evaluated slice every rung, vary -c):

| rung | stock | gate1+floor128 | dPPL | ratio |
|---|---:|---:|---:|---:|
| 4k | 9.1384+-0.058 | 9.6658+-0.062 | +0.5274 | 1.0577 |
| 32k | 9.3318+-0.060 | 9.8689+-0.064 | +0.5371 | 1.0576 |
| 131k | 8.3746+-0.058 | 8.8944+-0.062 | +0.5198 | 1.0621 |

Ratio FLAT 4k->131k (absolute dPPL smallest at 131k). NO explosion, no upward trend.
Wiki.test.raw @8k continuity: stock 7.7413 (47c: 7.7476), gate 7.9680 (47c: 7.9541) - harness
reproduces 47c. 150k PPL rung NOT ATTEMPTED: perplexity.cpp reserves n_ctx*n_vocab floats
(80 GB @131k, 91 GB @150k) > 40 GB host RAM; 131k legs completed via swap (~50 min).

LEG B - perf vs context (llama-bench ROCm1 r=3, pp131k r=1): pp4096 820.68+-100 (noisy run:
samples 741/934/787) vs 1248.92 (+52%) | pp8192 955.42 vs 1188.28 (+24.4%) | pp16384 867.46
vs 1082.61 (+24.8%) | pp32768 766.77 vs 912.40 (+19.0%) | pp65536 644.06 vs 780.85 (+21.2%) |
pp131072 477.33 vs 544.62 (+14.1%). Decode at depth: tg64@32k 74.74 vs 74.52 (-0.3%),
tg64@131k 55.68 vs 55.85 (+0.3%) - bit-flat as the floor promises. Gated >= stock EVERYWHERE.
Prefill win persists at depth (+14-21% at 32-131k), consistent with 35/41 and the README bound
(NOT +30% flat).

LEG C - MTP x long context: the 9B GGUF HAS the native MTP head (blk.32.nextn.*,
nextn_predict_layers=1) - no 27B substitution needed. llama-server -c 150000 draft-mtp n-max 6,
greedy 256-tok decode after long prefill:

| depth | arm | pp t/s | tg t/s | mean acc len |
|---|---|---:|---:|---:|
| 32k | stock+MTP | 724.27 | 59.28 | 3.18 |
| 32k | gate+floor+MTP | 851.59 (+17.6%) | 78.56 (+32.5%) | 4.21 |
| 65k | stock+MTP | 508.01 | 58.01 | 3.42 |
| 65k | gate+floor+MTP | 563.91 (+11.0%) | 75.21 (+29.6%) | 4.49 |
| 131k | stock+MTP | 342.51 | 31.29 | 2.98 |
| 131k | gate+floor+MTP | 366.16 (+6.9%) | 39.41 (+25.9%) | 3.87 |

Caveat: arms diverge at the prefill-numerics boundary so tg/acc are not prefix-matched; the
gate arm's 131k continuation entered a repetition loop which inflates its accept-len - an
ALTERNATE 131k cut probe showed the STOCK arm loop there instead (gate stayed coherent): the
loop is prompt-position base-model behavior at depth, not a gate defect. No empty/degenerate
outputs (all gn=256). Accept-length healthy both arms at every depth; prefill keeps the gate
win under MTP at depth; decode never below stock.

VERDICT: GO on the flagship config at long context. Accuracy ratio flat to 131k, perf never
below stock, MTP coexists (floor keeps verify/decode bit-stock at 150k context too).
Harness bugs fixed en route: (1) prompt files cut mid-UTF-8 -> U+FFFD -> instant EOS (empty
gens) - cut at a valid UTF-8 boundary first; (2) llama-perplexity needs >=2c tokens in file
and reserves n_ctx*n_vocab logits (the 150k-PPL wall); (3) llama-bench has no -c flag.

## 65. Merge-validation: upstream sync (354 commits) - anchors EXACT, MoE gain unchanged (2026-10-15, master , rig/merge-ladder-71a0624fb/)

Upstream base = mxxm-t master tip at merge time, 354 commits / 896 files.
Merge commit on master (parents + ). Gate-off AND gate-on
builds RC=0 (build-merge in the master worktree).

CONFLICTS: 11 files. Resolutions (full detail in the merge commit message):
ggml-backend-meta.cpp (upstream #29266 view-inheritance supersedes our early multi-buffer
abort - duplicate dropped), common.cuh (fast_bf16_hardware_available dedupe - upstream now
carries our gfx906 fix verbatim), ggml-cuda.cu (upstream fusion block = shared-expert rotate
+ our weighted-reduction loop, now upstream; brace repaired), mmq.cu (upstream wholesale -
zero fork delta vs base), common/speculative.cpp (upstream common_batch process_decode API),
llama-context.cpp + llama-graph.cpp (upstream; GLM5_NEXT joins the hc_mult arch list),
qwen35.cpp (upstream build_qkv refactor + fork Kcur/Vcur redeclaration removal; fork
KV-only-prefill block intact), qwen4exp.cpp (upstream wholesale - upstream natively carries
the MTP head AND our hc_norm {n_embd,hc} fix), llama-bench.cpp (upstream -nr/--no-repack),
windows-setup-cuda action.yml (upstream hardened downloads). No nc_t case collisions this
cycle; q8_repack/mul-mat.cu + repack-kernels.cuh byte-identical to pre-merge master
(U4AFFINE machinery, MIN_TOKENS floor, RP_KSPLIT, u4q41/u4q4k all present; buffer.cu picked
up upstream designated-initializer buffer-type interface only).

ANCHOR LADDER (9B Q4_0, ROCm0, same flags as sec 54/sec 56): ALL EXACT -
gate1 PPL ch256 8.7630 +- 0.08938 EXACT; gate1 KLD ch128 0.122150 +- 0.002028 EXACT;
SKIP-early KLD 0.097917 +- 0.001819 EXACT; per16 PPL ch32 7.2774 +- 0.20253 EXACT;
gate4 PPL ch256 8.7630 +- 0.08938 bit-identical to gate1 (superset property intact).
MTP arm (27B gate1+floor128+MTP, sec 62 protocol, port guard + rep-count 3/3):
48.36/48.66/48.63 t/s (mean 48.55), mean acc len 4.07 EXACT vs sec 62's 4.07, t/s +2.6%
vs sec 62's 47.29 +- 0.28 (upstream MoE/MMQ workspace wins, sanity band).

MoE CHECK (community-MoE-35B Q4_0, 2-GPU tensor, r=3, stock vs gate1): pp512 1891.80 -> 2046.31
= +8.17% (vs sec 41-era +8.4-8.6% ref - unchanged within noise; upstream #27828 MMVQ-for-
MUL_MAT_ID + #29887 MoE cache did NOT shift the MoE gate gain); tg128 105.60 -> 98.73
(-6.5%, the known MoE decode regression - gate stays OFF for MoE entries).

FOLLOW-UPS: fix/llama-bench-dev-comma rebased onto post-merge master ( -> ,
clean, no conflict with upstream -fitc/-nr changes); int4 branch merged master (,
parents + ) - README front door kept (sole conflict, --ours).
VERDICT: merge does not perturb the gates. is the new floor.

## 66. MoE pp16 c2 regression cell (224.3 vs stock 272.0): forensics - the 2026-10-18 binary predates the MoE floor (2026-10-21, branch perf/int4-next-20261021, rig/sumw-seal-ab/RESULTS.md, rig/wt-sumw)

The 2026-10-18 five-config matrix (INT4-FINDINGS-2026-10-18.md) left one
unexplained cell: MoE qwen3-30b-a3b pp16 with the full production config c2
(DOT8=1 + MIN_TOKENS=128 + KSPLIT=1) measured 221.9-222.7 t/s vs stock 272.0
(-17.6%), a width where the floor should have forced the stock path.

FORENSICS (this section, verified 2026-10-21):
- The c2 MoE rows in rig/sumw-seal-ab/RESULTS.md (m-bench-c2-g1,
 m-bench-c2-g2, timestamps 2026-10-18T20:43Z) record build_commit
 - the perf/sumw-plane staging binary.
- (2026-10-15 15:41:13 -0700, tip of wt-sumw on mi50) does NOT
 contain the MoE floor: `git merge-base --is-ancestor HEAD` is
 false in rig/wt-sumw, and mul-mat-id.cu in that tree carries NO
 min_tokens clause (grep empty). The MoE floor commit
 (2026-10-15 15:11:49 -0700, "apply the width floor to the MoE u4
 predicate") lives on the int4 lineage only.
- Therefore hypothesis (a) of the next-work brief is CONFIRMED: the floor
 could not have engaged in that run; per-32 u4act ran ungated at pp16 on
 MoE and paid the documented narrow-width loss (cf. section 61
 crossover: u4 below stock below ne11 ~128).

RESULT (2026-10-21, build, ROCm0, -t 128, 3 interleaved rounds,
sigma < 0.4%): attribution CONFIRMED with a refinement. Three arms:
stock / c2+floor (REPACK=1 DOT8=1 MIN_TOKENS=128 KSPLIT=1) / c2-nofloor.
pp16: 303.5 / 222.1 / 198.0. The floor recovers +12.2% of the narrow
prefill loss (198.0 -> 222.1) - the 2026-10-18 cell ran UNGATED, so the
regression was floor-absence as hypothesized. REFINEMENT: the "expected
~stock" prediction was wrong in this regime - the residual -26.8% pp16 vs
stock is the REPACK narrow-width characteristic (section 68: dense
repack-only arm reads the same -26%), which is weight-side and floor-
immune. Wide width: +11.3% pp512 both configs (repack win, floor-
invariant). tg128: c2+floor 127.1 (+9.3%, repacked-stock decode win,
mirrors dense +10.3%); c2-nofloor 116.8 (+0.4% = stock parity - u4act MoE
decode is stock-class on this build). PROGRAM COROLLARY: REPACK_Q4_0
admission carries a ~-26% narrow-prefill (pp16-32) cost in the single-GPU
-t128 regime that NO gate combination removes - narrow-prefill-sensitive
deployments on repacked Q4_0 files trade that for the +9-11% tg/pp512
wins. Regime caveat: all single-GPU; the 2026-10-18 3-GPU layer-split
matrix absolutes are not comparable (deltas within-session only). NOTE (dates, not ancestry, per rule): postdates the dense
floor 4abc9d735/1434fda3b (2026-10-14) by one day - the deployed
build-merge binary carries the dense floor only per section 65-era
validation; any MoE floor claim must come from a build at/after
or its scrubbed descendant 2992b4275.

## 67. NVFP4 repack port, Phase 0: identity sealed exact on CPU; NO-GO for Phase 1 (2026-10-21, branch perf/int4-next-20261021, seals/seal_nvfp4_identity.cc, /home/brad/workspace/nvfp4-phase0/)

Phase 0 (CPU-only, per plan; full note + per-regime tables in the workspace
dir): the E2M1 x UE4M3 integer identity for an NVFP4 repack path is derived,
sealed, and falsifiable:
- Decode: k = kvalues_fp4[code] in int8, |k| <= 12 (= 2 x E2M1 value),
 re-derived from the bit layout (mag = (e==0)? m : (2+m)<<(e-1); sign-mag
 FP, no bias - the Q4_0 XOR 0x88 trick does NOT apply).
- D = sum(a_i k_i) exact in i32 (|D|max 24384 int8-act / 1536 int4-act;
 88069x / 1.4Mx margin to 2^31).
- Epilogue exact in fp32 (NOT fp16): D needs 15 bits and UE4M3 mantissa 4,
 19 <= 24 fp32 bits; fp16 fold rejected (15 > 11 bits; 1536*224 > 65504).
 CORRECTION to the plan hint: UE4M3 scales are NOT powers of two (3-bit
 mantissa; d=0x0B -> 11*2^-10); exactness survives via the M <= 15 bound.
- Undefined scale bytes d >= 0x7F ALIAS (bit7 shifted out; 0xFF -> 240 but
 0x7F -> 0 via the NaN check): a port must reject/normalize them at
 repack admission.
- Seal: 1296 trials (1000 random + adversarial regimes + sweeps), EXACT
 equality vs an independent double reference, 0 failures; independently
 reproduced by a separate Python implementation; falsified 3 ways
 (decode-table off-by-one, dropped /2 fold, dropped sign) - all FAIL
 loudly. Worst per-regime vacuous-trial fraction 0.9961
 (all-subnormal-nibbles, adversarial-only by design); every regime has
 >= 5 realized hits.

UPSTREAM STATE (verified in-tree, this branch): GGML_TYPE_NVFP4 = 40 and
kvalues_fp4 are native in the fork's master via the sync;
upstream PRs #29173 (merged 2026-10-01) and #29857 (merged 2026-10-05)
are already absorbed. The port would carry no local format patch - but
section 45's generic-MMQ NVFP4 baseline (972.5 pp8192) is STALE: upstream
optimized mmq accumulation after it was measured, so any future perf A/B
must re-baseline on a post-sync build.

VERDICT: NO-GO for proposing Phase 1. The identity work stands as
decision-final kill evidence on the BUILDABILITY axis (the missing repack
was unbuilt, not impossible), but the adoption bar (+22.9% repack-class
prefill) is a measured perf bar CPU Phase 0 cannot clear, and Phase 2
measured a KLD dead tie vs Q4_0+map with NVFP4 forfeiting the u4
activation gate - the repack would have to beat the Q4_0 repack path
itself, not generic MMQ. Count-based readings justify a kill, never a
build. If the perf question is ever reopened, the defensible step is a
measured micro-GEMM A/B on a post-sync build (probe, not commitment).
The identity retains value for Blackwell-side FP4-MMA work.

## 68. MIN_TOKENS floor extension to u4x16/u4q41: gates PASS, floor-invariance exact on two files, pp16 anomaly attributed to repack (2026-10-21, branch perf/int4-next-20261021, build rig/floor-u4x16/build (11758), rig/floor-u4x16/{ledger2,ledger3}.txt)

CHANGE : the width floor (GGML_CUDA_W4A4_MIN_TOKENS) now gates ALL
u4 predicates in the dense dispatch (mul-mat.cu): u4act (per-32), u4x16
(per-16 GEMM) and u4q41 (Q4_1 dp8a) each gain `ne11 >= min_tokens`;
constant-true when unset (floor==0 byte-identical by construction). MoE site
unchanged (u4act-only, already floored). Marker text + README tuning row
updated to "covers every u4 path".

GATES (all on , ROCm0, -fa on -lm dio, -t 128, single GPU):
- Build gate: ggml+llama rc=0.
- Marker gate: llama-perplexity DOT8=3+MIN_TOKENS=128 prints
 "GGML_CUDA_W4A4_MIN_TOKENS=128 floor ACTIVE [explicit] (every u4 path
 (u4act/u4x16/u4q41) falls back to stock q8_1 below this batch width)".
- Greedy seals (temp 0 top-k 1, re.findall number-sequence compare):
 SEAL-1 short prompt, DOT8=3+floor128+REPACK vs stock: IDENTICAL
 [1,1,2,3,4,5] - the floor forces the stock path below width 128.
 SEAL-2 long prompt (>=300 tok), floor128 vs floor-unset: IDENTICAL - wide
 per-16 path untouched. SEAL-3 negative control (short, floor-unset vs
 stock): DIVERGES (extra trailing token) - proves the u4 decode path was
 actually engaged in the comparison (non-vacuous harness).
- Engagement witness: 'q8_repack: repacking q4_0 tensors' present on every
 gated leg, absent on stock legs. NOTE: GGML_CUDA_REPACK_Q4_0=1 is REQUIRED
 for Q4_0 u4 on this build (opt-in admission; the 2026-10-18 matrix builds
 admitted Q4_0 natively at the gate - a first perf spot without REPACK was
 VACUOUS, all arms stock, caught by the ~0% deltas + missing witness).
- KLD (c512 -b 2048 -ub 512 --chunks 128, f16 base):
 floor-invariance EXACT on two files: upstream pure-Q4_0 floor128 vs
 floor-unset = 0.090147 +- 0.001853 BOTH (PPL(Q) 8.559974 both); bartowski
 mixed file 0.066060 +- 0.001572 BOTH. Anchor class reproduced: 0.090147
 vs historical 0.090161 - residual (1.4e-5) attributed to the REGENERATED
 f16 base (see incident note; base PPL 8.020648 vs 8.020672 historical).
 The bartowski file is a MIXED quant (q4_0 + q6_K attn + q8_0 ssm_out
 repacked per its log) - NOT pure Q4_0; its lower KLD vs f16 is a file
 property, not a config effect. Public claims stay on the upstream file.
- Perf spot (3 interleaved rounds x 3 arms, means): stock pp16 305.3 /
 pp512 941.0 / tg128 71.6; per16 (DOT8=3, no floor) 240.9 / 994.8 / 80.6;
 floor128 226.5 / 993.1 / 79.0.

PP16 ANOMALY ATTRIBUTED (4-arm probe, ledger3): repackonly (REPACK_Q4_0=1,
no gate) pp16 = 225.7 +- 0.2 - statistically identical to floor-per32
(225.4) and floor-per16 (225.2), vs stock 304.8. The -26% narrow-prefill
deficit is a PRE-EXISTING REPACK characteristic (q8_1 cache misses at
ne11=16 per counters), NOT the floor: the floor adds zero pp16 cost beyond
repack itself, and routes exactly as designed. Corollary (new open item):
repack-only narrow prefill (pp16-32) pays ~-26% vs canonical stock in this
single-GPU/-t128 regime - pp-parity claims for repack were wide-width only.
Repack wins are floor-independent: pp64 +5.5%, tg128 +9.5-10.5%.

INCIDENT + REPAIR (no silent regression): the first anchor leg omitted the
--kl-divergence boolean; --kl-divergence-base is an alias of
--save-all-logits (common/arg.cpp:2515), so the leg ran in SAVE mode and
OVERWROTE the retained f16 logits base with per-16 Q4_0 logits. Base was
regenerated from the bf16 upstream model same-protocol (repair-save rc=0);
post-repair PPL(base) 8.020648 vs 8.020672 historical (regenerated base is
not bit-identical; all KLD numbers above are internally consistent on the
new base). Trap documented in the skill bench-gotchas reference with a
pre-flight check. k1/k2 results predate the clobber and remain valid.

## 69. W4A8 ASM census: NO-GO on the production GEMM - the i8-activation form already exists as the stock q8_1 kernel and is the heavier one (2026-10-21, branch perf/int4-next-20261021, docs/w4a8-census.md, mi50:/tmp/w4a8-census/)

Census of all 10 Q4_0 repacked kernels (code objects 48/49 of 144) in the
 build's libggml-hip.so.0, per the program census discipline
(roc-obj-ls -> dd per-object ELFs -> llvm-objdump -d --mcpu=gfx906; YAML
.note spill counters cross-checked against scratch ops). Full table:
docs/w4a8-census.md.

- The W4A8 shape (i8 activations, v_dot4_i32_i8) ALREADY EXISTS in the
 binary: the stock q8_1 GEMM is exactly that kernel. Production w64 GEMM
 hot loop: dot4 form = 256 v_dot4 in 653 instrs (39.1% dot density,
 67 loads, 128 VGPR + 20 VGPR SPILLS, 6 scratch ops in-loop) vs
 dot8/u4act form = 128 v_dot8 in 647 (19.8%, 42 loads, 127 VGPR,
 ZERO spills). Same MACs, strictly more instructions and bytes.
- The go criterion ("u4 hot loops bandwidth-dominated -> doubling dots is
 nearly free") FAILS on the production GEMM: it is dot-dense at the spill
 cliff. The only kernels passing the criterion are the mat-vecs (4-15%
 density, zero spills) - not where prefill wall time lives.
- The trade has in fact been MEASURED all along: every gate-vs-stock bench
 (e.g. section 61: gate +27..+32% pp at ne11 >= 128) is dot8-form vs
 dot4-form on the same shapes. W4A8's Phase-1 kernel would land on the
 measured-slower side.
- Toolchain: v_dot4_i32_i8 EMITS (probe + --save-temps, bogus-symbol
 control failed-compile as required). Operand pairing VERIFIED EMPIRICALLY
 on hardware (ROCm0 probe, flock'd): little-endian byte order, exact
 (hw=192 vs LE prediction 192, reversed-order 48 rejected).
- VERDICT: NO-GO for a Phase-1 W4A8 kernel proposal on the throughput
 premise. What remains is a PURE ACCURACY trade: an 8-bit activation path
 would halve the ~45% activation-error class (accuracy brief R3) at the
 measured cost of the stock-q8_1 GEMM shape (dot-dense + spilled, and
 ~22-27% slower prefill than the u4 form at wide widths per the gate
 benches). That is the owner's Pareto call, not a census question.

## 70. Dense SUMW interleaved cell: documented null-to-negative - tg flat, pp512 consistent -2.15% (0/15 positive); NOT a dense default-on candidate (2026-10-21, branch perf/int4-next-20261021, rig/floor-u4x16/{ledger4.txt,dense-sumw/})

Design: 15 interleaved paired rounds on ONE card (ROCm0; card-split regime
- the original 3-card x 15 design adapted), arms identical except
GGML_CUDA_W4A4_SUMW 0/1 on the c2 base (REPACK=1 DOT8=1 MIN_TOKENS=128
KSPLIT=1), 2 warmup legs discarded (DVFS class), 30 timed legs, all
witness-verified (SUMW gate ACTIVE iff SUMW=1; 0 invalid, 0 retries).

RESULT (medians of per-round deltas): tg128 -0.20% (5 pos / 9 neg / 1
zero - no consistent decode win); pp16 -0.004% (flat); pp512 -2.15%
CONSISTENT (0/15 positive, 14 rounds tight at -2.05..-2.22%, one -5.17%
outlier with 10x stddev = transient host perturbation, median robust).
No pp512 reading above +2%: the prior cell's +11.5% is formally
attributed as wrong-sign contamination (the SUMW precedent).

MECHANISM: the SUMW plane adds +2 B/block (18 -> 20 B, +11% weight bytes)
which costs bandwidth-bound dense prefill. VERDICT: dense SUMW is NOT a
free win and NOT a default-on candidate; SUMW's value remains MoE-
decode-specific (+2.22% tg128 MoE, section 59-era measurement, stands -
MoE decode amortizes the recompute differently). Shipped default (off)
is correct for dense; the `speed` profile presets SUMW=1 - that preset
is justified by the MoE decode win only, worth a README note if dense
prefill matters to a deployment.

## 71. MTP_SAFE=2: not implemented (mode 2 is a log line on top of mode 1), and untestable on one card - BLOCKED on two parts (2026-10-21, branch perf/int4-next-20261021, tp-allreduce.cu:1100-1128)

Step-zero inspection (before any GPU leg; no benchmark was run, nothing
fabricated): GGML_TP_AR_MTP_SAFE=2 IS parsed (tp-allreduce.cu:1101-1105)
but is behaviorally IDENTICAL to mode=1. On the first width transition,
tp_custom_ar_mtp_safe_transition sets the one-way sticky flag
ctx->mtp_safe_locked = true ("custom is off for good", :1107/:1122) and
mode 2's ONLY difference is a single fprintf of the transition (:1123-1126).
Confirmed across 5 worktrees. The A->B->A revisit semantics the brief
describes do not exist anywhere; "drafted" turns out to mean a log line.

WHY IT IS NOT A SMALL PATCH (source comment, :1110-1120): the custom
kernels' staging layout, parity double-buffering and SHARED PER-BLOCK
SIGNAL COUNTERS are only proven safe for a single steady width per
context; alternating widths across asynchronously-replayed captured graphs
corrupts the reduction (violation repro 2026-09-25). A true revisit design
must ISOLATE SIGNAL STATE PER WIDTH (per-width staging alone was already
tried and documented insufficient) - that is a new subsystem, not a flag.

SECOND, INDEPENDENT BLOCKER: the custom AR lives in the multi-device TP
comm path (ggml_backend_cuda_comm_allreduce_custom). On ONE card there is
no tensor split, the custom AR never engages, and arms b-d are UNTESTABLE
on the ROCm0-only card split regardless of implementation status.

STATUS: parked. To unpark needs (1) owner go-ahead to design the per-width
signal-isolated revisit state machine (multi-day, new subsystem, not the
brief's "half a day"), and (2) a >=2-card TP window for the losslessness
gauntlet (causal-flip classification + CPU-sampler-line grep mandatory).
Related untested branch noted: fix/mtp-sampler-guard-t1 targets
the set_sampler CPU-sampler hard-guard root cause; needs its own >=2-card
window and the same gauntlet.

## 72. Layer-selective activation granularity - CHEAP PRIOR: FAILS the 50% bar (43.5% best), benefit is proportional to tensor fraction, not concentrated -> NULL, no mechanism built (2026-10-21, branch perf/int4-next-20261021, rig/floor-u4x16/ledger6.txt)

Four KLD cells, dense Qwen3.5-9B upstream pure Q4_0, c512 -b 2048 -ub 512
--chunks 128 vs the repaired f16 base, REPACK=1 KSPLIT=1 floor128, all
gate-marker witnessed. Per-16-on-attn vs per-16-on-ffn produced with
GGML_W4A4_ONLY=attn|ffn under DOT8=3 (gate-3 semantics: the filter selects
which tensors get the per-16 GEMM; non-matching tensors keep per-32 u4act
- exactly the mixed cells needed).

 cell Mean KLD recovery of full per-16 gain
 per-32 (DOT8=1, full) 0.122150 -
 per-16 (DOT8=3, full) 0.090147 100% (gap 0.032003)
 per-16 attn + per-32 ffn 0.111959 31.8%
 per-32 attn + per-16 ffn 0.108239 43.5%

ANCHOR CROSS-CHECK: per-32 reproduces 0.122150 EXACTLY (6 decimals, the
historical anchor). per-16 reads 0.090147 vs the historical 0.090161 -
5th-decimal offset, consistent with the per-16 path's build delta (this
branch's scrub2 restorations) and/or the regenerated base; the relative
gain (0.0320) stands. State this pairing whenever both numbers are quoted.

VERDICT: NULL under the brief's phase gate (either mixed cell >= ~50% of
the full gain was the bar; best is 43.5%). The two halves sum to 75.3% of
the whole, i.e. SUB-additive: activation-granularity benefit is roughly
proportional to the tensor fraction it is applied to, not concentrated in
a quarter of layers the way the weight-side SKIP divergence was. A
per-tensor allowlist would therefore buy a linear accuracy/prefill trade,
not a free frontier point - the mechanism is NOT built (no
GGML_CUDA_W4A4_DOT16_ONLY code, no sweep needed). Owner may tune the 50%
threshold; at 31.8/43.5 the frontier point is poor either way: per-16 on
ffn alone pays per-16 prefill on half the tensors for 43.5% of the gain.

## 73. Stock-form w64 GEMM spill fix: THREE approaches, all bit-exact-or-rejected; NO-GO on adoption - the spills are latency, not throughput (2026-10-22, branch perf/int4-next-20261021, scratch trees rig/floor-u4x16/{src-spillfix,build-spillfix}, ledger7.txt)

Target: the stock-form q8_1-activation `mmq_gemm_repacked` (the kernel behind every floored narrow-width fallback, the MTP verify pass and SKIP tensors; 128 VGPR / 20 vgpr spills / 52 B private segment / 6 in-loop scratch ops per the section 69 census). Success criterion: spills materially down with dot density unchanged AND bit-exact math. Seal gate: greedy temp-0 top-k-1 seed-42, floor env (REPACK_Q4_0=1 DOT8=1 MIN_TOKENS=128 KSPLIT=1), short + ~300-token prompts.

**A1 - launch_bounds occupancy hint: `__launch_bounds__(64 * NRL, 2)` -> `(64 * NRL, 1)`** (one line, wrapper-wide). Reported census: spills 20 -> 0, private seg 52 -> 0 B, in-loop scratch 6 -> 0, dot4 density unchanged; VGPR 128 -> 184 (2 -> 1 blocks/CU). Seal: BIT-EXACT, re-verified in the parent window (only the `[ Prompt: X t/s ]` footer differs; generated text byte-identical on both prompts). Bench, 5 interleaved paired rounds, ROCm0, floor env, `-p 16,32,64 -n 128 -fa on -lm dio -t 128` - medians of per-round deltas: pp16 +0.97 (3/5 pos), pp32 +1.65 (3/5 pos), pp64 +0.06 (3/5 pos), tg128 +0.20 (4/5 pos) t/s. Verdict: NOT ADOPTED. 0 spills buys nothing measurable at these widths, and the hint halves occupancy for EVERY `mmq_gemm_repacked` instantiation (Q8_0 / Q4_K / u4x16 siblings share the wrapper) - a blast radius this bench did not cover. Consistent with the section 69 reading: the 20 spills are latency the scheduler already hides at 2 blocks/CU, not a throughput ceiling.

**A2 - reduce unrolling (`#pragma unroll 1` on the stock g loop): FAILED, reverted.** Rolling the loop makes the `acc[r][n]` indices dynamic and destroys register allocation of the 32-float accumulator tile: spills 20 -> 335, private seg 460 B, 237 in-loop scratch ops. GENERALIZES: any partial unroll of the g/rr/n loops is a dead end for the same reason - do not retry it.

**A3 - order-preserving restructure: drop the double-buffered `wlo`/`whi` LDS prefetch, load `sW_lo`/`sW_hi` per row at point of use: REJECTED on bench.** Spills 20 -> 5 at 128 VGPR (occupancy preserved), in-loop scratch 6 -> 4, FP op counts provably identical (512 dot4 / 64 fma / 64 add / 160 mul - a scheduling-only diff), BIT-EXACT seal. But medians vs pre: pp16 -1.42 (1/5 pos), pp32 -7.19 (0/5 pos), pp64 -2.39 (1/5 pos), tg128 +0.08 (5/5 pos). The prefetch is worth more than the spills it costs - the fallback path is memory-latency-bound on LDS issue, not spill-bound.

PROGRAM CONCLUSION: the stock-form GEMM's 20 spills are NOT a throughput lever. Close the queue item opened by the section 69 external review. If the fallback path ever needs to be faster, the lever is the LDS/prefetch schedule (A3 shows it dominates), not register pressure. Reproduction: the A1 diff is one line (`__launch_bounds__` second arg 2 -> 1 at repack-kernels.cuh:1887) and the A3 diff is the two `wlo`/`whi` prefetch blocks; both scratch trees are on mi50.

## 74. MoE activation-granularity mandate, RESOLVED: accuracy is monotone in granularity (per-8 < per-16 < per-32) but no profile dominates; per-32 stays default, per-8 becomes the quality profile (2026-10-22, branch perf/int4-next-20261021, rig/floor-u4x16/{moe-base-kld.sh,moe-kld-self.sh,moe-mode-bench.sh}, ledgers ledger-moe-kld*.txt + ledger-moe-mode-bench.txt)

Model: Qwen3-30B-A3B BASE (qwen3moe, 128 experts / 8 active, 48 blocks). Base = bf16 converted from the HF repo (Qwen/Qwen3-30B-A3B), llama-perplexity ch64 c512 wikitext, PPL(base) = 8.8240 +/- 0.20984, logits 4.96 GB with sha256 recorded (moe30b-bf16-c512-ch64.logits).

**LANDMINE FOUND AND DEFUSED - the pre-existing MoE Q4_0 in multiarch-q40 is NOT a quantization of this base.** qwen3-30b-a3b-Q4_0.gguf shares architecture, tokenizer (151936) and all declared tensor shapes with the HF base but is a DIFFERENT weight set: blk.0.attn_k dequantized relative RMS difference 13.27% (Q4_0 quantization error is ~1-2%), blk.0.attn_norm mean relative difference 204%, and PPL 6.34 vs base 8.82. The first KLD run against it produced KLD ~0.32 for all three arms - that number was model mismatch, not quantization error, and is DISCARDED. Any KLD run pairing that file with an HF-derived base is invalid. Fix: self-made Q4_0 from the same bf16 with llama-quantize (80 s, 17.30 GB).

**KLD (same-source pair, ch64 c512, gate witness ACTIVE in all three legs):**
| arm | DOT8 | form | Mean KLD | PPL(Q) | PPL(Q)/PPL(base) | Same top p |
|---|---|---|---|---|---|---|
| c1 | 1 | per-32 u4 activations | 0.077100 +/- 0.001625 | 9.1765 | 1.045015 | 88.87% |
| c2 | 2 | per-32 decode + per-8 u4p GEMM | **0.057376 +/- 0.001283** | 9.1400 | 1.040861 | 90.58% |
| c3 | 3 | per-16 u4x16 GEMM | 0.063966 +/- 0.001343 | 9.1906 | 1.046617 | 89.92% |

Accuracy is MONOTONE in activation granularity - finer is better, and the ordering reproduces the dense result (per-16 beats per-32 there too): per-32 -> per-16 is -17% KLD, per-16 -> per-8 is a further -10%, per-32 -> per-8 is -25.6%. Every pairwise difference is > 4 sigma (errors 0.0013-0.0016).

**Speed (self-made Q4_0, ROCm0, -fa on -lm dio -t 128, 3 interleaved rounds, medians, repack+floor env, r=1 per round):**
| arm | pp16 | pp32 | pp512 | tg128 |
|---|---|---|---|---|
| stock | 282.8 | 357.3 | 1334.1 | 117.5 |
| m1 per-32 | 208.8 (-26.1%) | 346.6 (-3.0%) | **1491.5 (+11.8%)** | 128.2 (+9.1%) |
| m2 per-8 | 185.1 (-34.5%) | 314.3 (-12.0%) | 1272.0 (-4.7%) | 127.7 (+8.7%) |
| m3 per-16 | 207.6 (-26.6%) | 344.0 (-3.7%) | 1427.4 (+7.0%) | 128.5 (+9.4%) |

No profile dominates: m2 wins accuracy and loses wide prefill; m1 wins wide prefill and is the worst on accuracy; decode is flat across all three (+8.7 to +9.4%). MoE is still deliberately unfloored (section 62), so the narrow-width repack cost (-26 to -34% at pp16) remains visible in every arm.

VERDICT - NO MANDATE, and m3 is dominated: keep DOT8=1 (per-32 u4) as the MoE default for prompt-heavy throughput, publish DOT8=2 (per-8 u4p) as the quality-first MoE profile (-25.6% KLD for -4.7% pp512, decode unchanged), and drop per-16 from MoE consideration - it is strictly beaten by per-8 on accuracy and gives back half the wide-prefill win. The per-16-vs-per-32 question that section 72 left open for MoE is answered by granularity monotonicity: the choice is a per-deployment point on a single quality/speed curve, not a mandate.

## 75. TP2 serving UNBLOCKED (the hang was a build config, not the code); spec-decode MTP gauntlet: +65-99% but NOT reliably lossless (2026-10-22, branch perf/int4-next-20261021, build-rccl on mi50, logs rig/t6, rig/t6s)

**PART A - the TP2 hang, root-caused to the BUILD.** Every `-sm tensor` run on rig/floor-u4x16/build hung, spun at 339% CPU, ignored TERM, leaked VRAM proportional to the resident model (14.4 GB/card for a 27B Q8, 3.4 GB/card for a 9B Q4_0, no kfd holder) and wedged the SMC (rocminfo 0 agents, dmesg `Failed to send message 0x2a, response 0xfc`), needing a host POWER CYCLE each time (5 wedges). Cause: that build has `GGML_HIP_RCCL:BOOL=OFF` and links NO librccl (`ldd libggml-hip.so | grep -c rccl` = 0), so any TP collective outside the custom-AR fast path waits forever. The previously validated TP2 build (rig/mx-merge-src/build-merge, the one that measured +13.9% tg / +37% pp) has `GGML_HIP_RCCL=ON`. llama-bench masked it (its shapes all ride custom AR); llama-cli samples per token and hit the missing fallback. Ruled out along the way: shexp-split (LLAMA_SHEXP_SPLIT=0 engaged, hang persisted), custom AR (ON BY DEFAULT per ggml-cuda.cu:1102, so the "clean" leg already had it), the 27B MTP vehicle (9B cli wedged too), and HIP_VISIBLE_DEVICES as sole cause (it independently breaks TP - always pass devices via `-dev`).
FIX, verified end to end: build with `-DGGML_HIP_RCCL=ON` (build-rccl) AND run with TheRock nightly libs first in LD_LIBRARY_PATH, per the rig's RCCL rule: with the apt 7.2.3 stack the same binary SIGSEGVs inside libamdhip64 during `ggml_backend_cuda_comm_init` (gdb backtrace, rig/gdb.log). With `LD_LIBRARY_PATH=rig/therock-venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib:_rocm_sdk_core/lib` the 9B runs clean on TP2: gen 85.3 t/s, prompt 32.5 t/s, rc=0, 3 agents and 10.8 MB VRAM after - no leak, no wedge.
STANDING RULE: any TP binary here must be RCCL-built AND run against the nightly libs; verify with `ldd | grep -E "rccl|amdhip64"` before the first run. A missing/wrong RCCL does not fail loudly - it deadlocks or segfaults and takes the host with it.

**PART B - Task 6 spec-decode gauntlet on TP2 (27B MTP internal vehicle, Q8_0, build-rccl + nightly, LLAMA_SHEXP_SPLIT=0, greedy temp 0 top-k 1, -n 64 / -n 128, single prompt then a 3-prompt suite).**
Speed (t/s generation, TP2, ROCm0+ROCm1): control 30.4 / 30.5 / 30.7 x3 = 30.4-30.7; MTP unset 52.7; MTP_SAFE=1 52.6; MTP_SAFE=2 52.3; suite MTP 50.6 / 57.7 / 61.2. So MTP is +65% to +99% over control, and MTP_SAFE=1 costs nothing measurable here (52.6 vs 52.7) - the recorded -10.4% lock-off penalty did NOT reproduce on this build. MTP_SAFE=2 reads identical to unset, consistent with section 71 (mode 2 is a log line, not an implementation).
CPU-sampler fallback: grep for "CPU sampler|using CPU" = 0 across all 11 legs, so the historically cited disqualifier is NOT what is happening on this build.
Losslessness (the binding gate): control determinism verified identical (1172 chars, two runs). Single prompt: MTP, MTP_SAFE=1 and MTP_SAFE=2 are ALL byte-identical to control. Three-prompt suite at -n 128: prompt3 IDENTICAL; prompt1 and prompt2 DIVERGE late (first difference at char 1075/1299 and 1122/1304, i.e. roughly 85% through, on near-tie wording such as "mechanics/engineering)" vs "mechanics/engineering basics)"). No first-token flips observed.
VERDICT: MTP on TP2 is a large real speed win and its divergences are late near-tie flips, not early corruption - but the lossless contract is NOT met at temp 0 as measured (1/3 prompts identical). Per the gauntlet's own classification rule these must be classified BENIGN vs FUNDAMENTAL from per-token top-20 logprobs, which needs the llama-server path; that classification, not more cli comparisons, is the next step. This also matches the commit note: the residual speculative divergence on this rig is the verify-kernel batch-invariance class (upstream #25618), not the sampler guard. Adopting MTP for production is therefore still an owner decision on a quantified near-tie rate, not a settled win.

## 76. Spec-decode classification (server logprobs path): MTP on TP2 shows ZERO fundamental flips - both divergences are near-ties the base itself could not call (2026-10-22, build-rccl + TheRock nightly, rig/t6srv, classifier rig/t6-classify.py)

Method (the gauntlet's distributional gate, finally run on the server path): llama-server on build-rccl, TP2 via `-dev ROCm0,ROCm1 -sm tensor`, TheRock nightly libs first in LD_LIBRARY_PATH, LLAMA_SHEXP_SPLIT=0, port 8099, one server start per leg (TERM after capture, never KILL). Three prompts x {control, --spec-type draft-mtp} = 6 legs, greedy (`temperature 0`, `top_k 1`, `seed 42`, `n_predict 128`), each requesting `logprobs: true, top_logprobs: 20`. All 6 legs rc=0, HTTP 200, 18-21 s per leg, 3 gfx906 agents before and after every leg, VRAM back to 10.8 MB/card, no wedge. Classification rule: first token divergence only (causal flip); BENIGN if base margin <= 0.5 nats AND the spec pick is in the base top-3; FUNDAMENTAL if margin > 1.0 OR the pick is outside top-20; else MIXED. Post-flip positions are trajectory drift, counted separately and never as flips.

RESULTS
| prompt | aligned | first diff | margin (nats) | spec pick rank | verdict |
|---|---|---|---|---|---|
| p1 (piston engine) | 128 | tok 70 | 0.06 | 2 | BENIGN |
| p2 (merge two sorted lists) | 128 | tok 86 | 0.04 | 2 | BENIGN |
| p3 (train catch-up) | 128 | none | - | - | LOSSLESS |

Totals: LOSSLESS=1, BENIGN=2, MIXED=0, FUNDAMENTAL=0; post-flip drift tokens 98 (NOT flips).
First-divergence detail: p1 control top-3 `) -1.265 | ' basics' -1.324 | ' but' -1.452`, MTP picked `' basics'` at -1.283. p2 control top-3 `' (' -0.673 | '.' -0.714`, MTP picked `'.'` at -0.687. Both are true coin flips - the base's own top-2 are 0.04-0.06 nats apart, i.e. indistinguishable, and the spec pick is the base's #2.

VERDICT: the spec-decode losslessness gap on this rig is NOT corruption - it is the near-tie flip class, exactly what the protocol's BENIGN category exists to name. Combined with section 75's speed (+65% to +99% over no-spec) and the absence of any CPU-sampler fallback, MTP on TP2 is now characterized as: large real win, zero fundamental divergence in 3/3 prompts, 2 benign near-ties. Adoption still rests on a near-tie rate the owner accepts (2/3 prompts diverged somewhere in 128 tokens; neither changed meaning). CAVEATS: n=3 prompts, 128 tokens, one seed, one model (27B MTP internal vehicle); scaling the suite is the cheap way to tighten the rate. The classifier is deterministic and self-tested (5 synthetic cases + a drift-not-a-flip guard, ALL PASS) and re-run by the parent before use.

HARNESS NOTES (for whoever reruns this): `rocminfo | grep -c gfx906` returns 6, not 3 - each card contributes a `Name: gfx906` line AND an ISA line, so the correct gate is `rocminfo | awk '$1=="Name:" && $2=="gfx906"' | wc -l` = 3. A 3-card gate written the naive way aborts the whole run at leg 1 (cost: two no-op runs). The classifier takes `--dir`, not a positional path.

## 77. MoE K-split factorial half (Task 2 closed): K-split HELPS MoE narrow prefill - +7.4% pp16 / +6.0% pp32, opposite direction from the dense side (2026-10-22, build-rccl, ROCm0, rig/moe-ksplit/)

The last unanswered cell of the 2026-10-18 knob matrix: the MoE half of the {KSPLIT 0/1} x {per-32} factorial never ran (the dense side showed K-split SUBSTITUTES for per-16 - both attack narrow-width scale-plane bandwidth - and "NULL at per-16"). Open question per the brief: MoE narrow prefill contains K-split-covered DENSE linears (q/k/v/o) plus expert GEMMs the K-split guard cannot touch - what share is dense-tensor work?

SETUP: model = moe30b-base-Q4_0-selfmade.gguf (the same-source 30B-A3B Q4_0 from section 74, NOT the mismatched multiarch-q40 file), binary build-rccl/bin/llama-bench, single card ROCm0, `-fa on -lm dio -t 128 -r 1`, 6 legs interleaved KS0,KS1,KS0,KS1,KS0,KS1. Arms: REPACK_Q4_0=1 + DOT8=1 + MIN_TOKENS=128, GGML_RP_KSPLIT 0 vs 1. Gate witness confirmed engaged on every leg (KS0 = 3 `gate ACTIVE` lines, KS1 = 4 - the K-split line is the extra one). All 6 legs rc=0. Run by a delegated child from a fixed recipe; ledger verified verbatim on the box by the parent.

RESULTS (t/s, 3 interleaved rounds, medians):
| metric | KSPLIT=0 | KSPLIT=1 | delta |
|---|---|---|---|
| pp16 | 193.11 | 207.43 | **+7.4%** (3/3 rounds positive: +10.1/+7.1/+7.4) |
| pp32 | 327.61 | 347.25 | **+6.0%** (3/3 positive: +6.4/+5.7/+6.0) |
| pp512 | 1493.18 | 1493.82 | +0.0% (flat, as expected - ksplit is a narrow-width lever) |
| tg128 | 126.72 | 126.04 | -0.5% (flat; KS1_R3 read 113.23, an isolated outlier ~10% below its own arm's other rounds - discarded from the verdict, noted for the record) |

CROSS-CHECK: KS1 today (207.32-208.58 pp16) reproduces section 74's independently measured m1 arm (208.85) within 0.7% across sessions - the harness and the model are stable.

VERDICT: MoE narrow prefill DOES contain enough K-split-covered dense-tensor work for the split to pay: +7.4% pp16 / +6.0% pp32 at per-32, wide widths and decode untouched. This is the OPPOSITE direction from the dense finding (where K-split merely substituted for per-16 and was null at per-16) - on MoE there was no per-16 alternative in play, so the dense-linears share of narrow prefill is directly visible. PRACTICAL: keep GGML_RP_KSPLIT=1 in every MoE per-32 configuration (the `speed` profile already does); there is no regime where it loses. Task 2 is now fully closed: part 1 = the pp16 mystery was the missing MoE floor (section 66), part 2 = this factorial.

## 78. Outside-the-brief batch: TP3 measured (flat at 9B), near-tie rate tightened to n=10 (zero fundamental flips in 2560 tokens) (2026-10-22, build-rccl + nightly libs)

**A. TP3 vs TP2 (9B Q4_0, `-sm tensor`, 3 interleaved rounds, 6 legs all rc=0).** pp512: TP2 942.21 vs TP3 941.51 (-0.07%). tg128: 70.90 vs 70.86 (-0.06%). TP2_R1 was a cold-start outlier (695/55.8) correctly dropped by medians. VERDICT: the third card adds nothing at 9B Q4 - the model is too small for a 3-way split to amortize; TP2 remains the TP recommendation at this size. OPEN: TP3 on the 27B/30B class (fits 3x32 GB) is the interesting cell and is unmeasured. Rig note: llama-swap auto-loaded the production 27B onto ROCm0/1 ~29 s after the bench exited (observed 15 GB/card, pid traced to llama-swap) - check for that before diagnosing "orphaned VRAM" after a bench; it unloaded again later.

**B. Near-tie rate, n=10 prompts x 256 tokens (up from n=3 x 128).** 20 server legs, all rc=0 / HTTP 200 / 3 agents, ~3 min wall. Classifier: **LOSSLESS=3, BENIGN=7, MIXED=0, FUNDAMENTAL=0**; every one of the 7 divergences is a rank-2 pick with base margin 0.01-0.06 nats (examples: p4 `invest` -0.814 vs `buy` -0.833; p8 `\n` -0.671 vs `\n\n` -0.716; p10 `each` -0.693 vs `a` -0.699); 1081 post-flip drift tokens excluded as non-flips. QUOTABLE RATE: zero fundamental divergence across 2560 generated tokens; ~70% of prompts contain exactly one near-tie flip where the base's own top-2 are <= 0.06 nats apart. Combined with section 75 speed (+65-99%), MTP on TP2 is now fully characterized for an adoption decision. CAVEAT: one vehicle (27B MTP internal), one seed; rerun the suite per model before generalizing.

HARNESS NOTES: the classifier's PROMPT_IDS was hardcoded (1,2,3) - now auto-detects prompt ids from the directory (set-deduped; a first-run double-count bug from matching ctrl+mtp files per id was fixed). The naive `rocminfo | grep -c gfx906` agent count returns 6 and even name-line greps must anchor the value (ISA lines match loosely); the reliable count is `awk '$1=="Name:" && $2=="gfx906"'`. The TP3 ledger's per-leg `agents=` column recorded 2 on TP2 legs (device-count of that leg, not the rig) - a script quirk, not a rig state.

## 79. THE STACKED NUMBER: all shipped levers at once vs stock on the dense 9B - +17.7% decode, and the composition is super-additive (2026-10-22, build-rccl, ROCm0, rig/stacked/)

The run the brief never contained: one binary (build-rccl), one model (dense 9B Q4_0), 4 arms x 3 interleaved rounds, 12 legs all rc=0. Arms: stock (no env); mtp (--spec-type draft-mtp, native head); speed (REPACK_Q4_0=1 + DOT8=1 + MIN_TOKENS=128 + KSPLIT=1 + SUMW=1 - the full `speed` profile); speed_mtp (speed + draft-mtp). Greedy, single ~50-token technical prompt, -n 128. Gate engagement VERIFIED post-hoc: the sweep's .err files were empty because this build's llama-cli FILTERS GGML_LOG_WARN without -v (same verbosity-filter class as llama-bench; upstream fixed it for ERROR only, #28229 lineage) - a -v rerun with the identical env shows all markers (`GGML_CUDA_REPACK_Q4_0 gate ACTIVE`, `W4A4_SUMW gate ACTIVE`, repacking q4_0/q6_K/q8_0). STANDING RULE EXTENDED: gate witnesses require -v on llama-cli too, not just llama-bench.

RESULTS (medians of 3; round variance <2% per arm):
| arm | prompt t/s | gen t/s | gen vs stock |
|---|---|---|---|
| stock | 93.5 | 71.6 | - |
| mtp | 81.8 | 73.4 | +2.5% |
| speed | 96.6 | 78.3 | +9.4% |
| speed_mtp | 81.2 | **84.3** | **+17.7%** |

READINGS:
- The stack COMPOSES, and better than additively: speed alone +9.4%, mtp alone +2.5%, naive additive expectation +12.2%, measured +17.7%. (Mechanism: the repacked path reads less weight per token, so each spec round-trip is cheaper; the two levers amplify.)
- MTP alone on stock is nearly free (+2.5%) because acceptance is high and the draft is cheap; its measurable cost is PROMPT throughput (-13%, the draft context's share) - visible in every mtp arm, consistent across rounds.
- The speed profile also WINS prefill (+3.3% prompt) - the repack narrow-width cost documented in section 68 does not appear at this prompt length (~50 tokens) once the floor is on; do not extrapolate the pp16 -26% repack cost to short-prompt serving.
- Execution notes: 12/12 legs rc=0; the timing line prints to STDOUT in this build's llama-cli (the child's first ledger parse read stderr and got NA - rebuilt from the .txt files); gate_witness was recorded 0 everywhere and is a measurement artifact of the WARN filter, corrected by the -v probe above.

CONTEXT AND HONEST SCOPE: this is the dense 9B at short-prompt/128-token decode - the cheapest clean A/B, not the production config. The production-relevant numbers that already exist: MoE 30B stacked (per-32+floor+ksplit) vs stock = +11.8% pp512 / +9.1% tg / -26% pp16 (section 74); MTP on TP2 27B = +65-99% decode with benign-only flips (sections 75/76/78); K-split adds +7.4% pp16 on MoE (section 77). What is still NOT measured stacked: MTP on the MoE (no native MTP head on the base 30B - would need a draft), and any stacked config on TP2/TP3.

## 80. MoE + MTP stacked sweep on the CLEAN Q4_0 35B: MTP ALONE is the best MoE config - the gate stack SUBTRACTS 4.3% from MTP, inverting the dense composition rule (2026-10-22, build-rccl, ROCm0, benchmarks/qwen35-35b-a3b/moe-baseline-RESULTS.md)

The capstone cell: dense 9B showed the gate stack and MTP composing SUPER-additively (+17.7%, section 79). This section ran the identical 4-arm protocol on the MoE that can actually run every lever: CLEAN Qwen3.6-35B-A3B Q4_0 (bartowski, 87.6% of weight bytes in Q4_0, native nextn MTP head) + its official MTP sidecar draft (1.19 GB). 12 legs, 3 interleaved rounds, all rc=0, gate witness 5/5 on speed arms and 0 on stock/mtp as designed, box clean after.

RESULTS (GENERATION t/s, medians of 3, single card ROCm0, -n 128):
| arm | gen t/s | vs stock | vs mtp |
|---|---|---|---|
| stock | 84.9 | - | -25.2% |
| mtp (draft sidecar) | **106.3** | **+25.2%** | - |
| speed (gate stack) | 89.2 | +5.1% | -16.1% |
| speed_mtp | 101.7 | +19.8% | **-4.3%** |

HEADLINE: on MoE, MTP ALONE beats the full gate stack WITH MTP. The gate stack that composes super-additively on dense SUBTRACTS 4.3% from MTP here. Round spreads are tight (<= 0.9 t/s within every arm); prompt t/s is flat-to-down in the MTP arms (draft ctx cost, same as dense).

MECHANISM (consistent with section 74): the MoE decode penalty is activation-ENCODER LAUNCH COUNT. MTP turns every decode step into an n-token verify batch, multiplying encoder launches; the repacked path's per-verify encoder cost exceeds its weight-read savings on MoE. Dense has no expert encoders, so there repack savings compound with MTP instead. The composition rule is therefore ARCHITECTURE-DEPENDENT: dense = stack everything; MoE = MTP alone.

PRACTICAL CONFIG MATRIX (generation, vs stock, this build):
- dense 9B: gate stack + MTP = +17.7%
- MoE 35B (Q4_0, gate-capable): MTP alone = +25.2% (gate OFF; add the floor only if narrow-prefill serving matters, and K-split stays +7.4% pp16 per section 77 for prefill-heavy serving)
- MoE 35B (Q4_K_M-class): NOTHING engages; MTP is -6.6% (section 75 K_M result) - quant choice is a first-order variable.
RECIPE NOTE: bartowski splits the MTP head into a sidecar GGUF - the spec arms need `--spec-type draft-mtp --model-draft <sidecar>`; the main file alone carries only the nextn norms/proj.

SCOPE/CAVEATS: single vehicle (clean base 35B - good provenance), one prompt, 3 rounds, one seed, 225 W caps (host was cycled; within-session paired deltas valid, absolutes not comparable to 175 W sections). The 4.3% gate penalty on MTP is one measurement on one MoE; rerun per model before generalizing - but the DIRECTION is mechanistically explained and matches section 74's encoder accounting.

## 81. MTP-overhead suite PORTED from CV - and the A/B verdict is a clean NULL: zero gain on our fork, output byte-identical everywhere (2026-10-22, branch perf/cv-mtp-overhead, gitea, rig/tp-cvab/)

Items 1 (LLAMA_GRAPH_CACHE2, +84/-1 in llama-context.{h,cpp}) and 3 (LLAMA_CKPT_LAST_VERIFY, server-side) ported to branch perf/cv-mtp-overhead; item 2 (LLAMA_SPEC_MTP_MERGE) correctly STOPPED - our common/speculative.cpp draft_mtp diverged structurally from CV's (no process_impl/deferred_batch/set_dft_full_ckpt; we use deferred_prefill/prefill_chunks), so porting meant rewriting the feature. Build rc=0 including llama-server. Default-off seal: token-identical. Pushed to gitea.

A/B (27B MTP internal vehicle Q8_0, TP2 ROCm0+ROCm1, nightly libs, greedy, -n 128, 3 interleaved rounds, 9 legs all rc=0): PRE (pre-port build-rccl) gen **30.8** | OFF (suite compiled, LLAMA_GRAPH_CACHE2=0) **30.8** | ON (suite default) **30.8**. Prompt t/s 145.9 / 146.3 / 144.7 - all flat within 1%.

**VERDICT: the ported MTP-overhead suite is a NO-OP on our fork at this config.** The CV-engine +20.7% tg (v5/v6, OURS-measured on their binary) does NOT transfer. Output byte-identity confirmed across all 9 legs after stripping the build banner (first "divergence" was `build: b11758` vs `b11785` metadata - a comparison artifact, not a behavior change).

WHY ZERO (hypothesis, ranked): (1) our fork is mxxm-lineage - the same lineage CV's meta-backend diverged from - and already carries the two-context (`ctx_other`) architecture that GRAPH_CACHE2's trigger condition keys on; the drafter graph-alternation cost CV's suite removes may simply not exist in our scheduler path; (2) shape dependence: the 1.6 ms/step re-alloc cost was characterized on CV's 2+2 MI50 MTP config, not our 27B TP2; (3) the suite's other members (MTP_MERGE - not portable, CKPT - server-only, untested here) may carry CV's gain, not GRAPH_CACHE2.

STATUS/DECISION: branch perf/cv-mtp-overhead stays on gitea (harmless: default-on for MTP contexts measured zero-delta and byte-identical; =0 available). NOT recommended for merge on performance grounds - there is nothing to gain at the measured config. Item 2 remains a design question (would need a speculative.cpp rework, not a port). The +49%-vs-+19% MTP gap between CV and our fork (v8) is therefore NOT overhead-suite territory: our MTP gain is limited by something else (acceptance dynamics or verify economics), not host overhead - re-examine if that question reopens.

MEASUREMENT NOTES: the timing line in this build's llama-cli prints to stdout and the first comparison pass flagged a false DIFFERS from the build banner (`b11758` vs `b11785`) at char 264 - always strip the banner block before byte-comparing runs from different binaries. Absolute gen level this session (30.8 t/s) is well below section 75's 52.7 t/s for the same model/config - same-session deltas are valid, but the level gap is an open question (power caps read 225 W = uncapped after the host cycle; thermal or state differences suspected, unresolved).

## 81a. CORRECTION to section 81: the first A/B accidentally ran WITHOUT spec decode; with MTP active the ported suite's default-on path CRASHES deterministically under TP2 (2026-10-22, rig/tp-cvab/)

The 9-leg A/B in section 81 was INVALID: the launcher omitted `--spec-type draft-mtp`, so PRE/OFF/ON were three identical plain-decode arms (~30.8 t/s, "speculative": false in the diagnostic) - that is also why everything read byte-identical and why the gen level was 30.8 instead of the ~47-52 expected under MTP. Caught by a verbose diagnostic; invalid legs archived at rig/tp-cvab/nospec/.

RE-RUN with MTP active (27B internal vehicle Q8_0, TP2 ROCm0+ROCm1, nightly libs, greedy, -n 128, 3 interleaved rounds, 9 legs):
- PRE (pre-port build-rccl): 47.6 t/s gen median (47.5-47.8), prompt 89.2-93.2 - consistent with section 75.
- OFF (build-cv, LLAMA_GRAPH_CACHE2=0): 47.6 t/s, generated text byte-identical to PRE (sha256-verified) - **default-off is safe and performance-neutral. PASS.**
- ON (build-cv, suite default-on): **CRASH 3/3, deterministic** - rc=134 SIGABRT, `ggml-backend-meta.cpp:3519: GGML_ASSERT(bcj.nodes[i]) failed`, zero generation. The alt-slot path trips the meta-backend's copy/join node-array assert under TP2 + draft-mtp.

REVISED VERDICT (supersedes section 81's "harmless no-op"): the port is INCOMPLETE for our serving path. CV's graph_cache2 works against CV's scheduler/meta configuration; against OUR meta-backend under TP2 + draft-mtp, activating the second slot hits an unpopulated backend-copy/join node array. Status: **default-off is the only safe state; default-on must not ship until the meta-backend graph-cache2 path is fixed** (the assert site is the debugging entry point - the MTP draft graph's backend-copy/join node array is not populated when the alt slot serves it). The -6.6%-vs-+73% MTP-on-K_M quant finding (section 75 addendum) and all other section 75 numbers are unaffected (they ran on the pre-port binary or with the suite off).
Also corrected: the section 81 "absolute level gap" open question (30.8 vs 52.7 t/s) is RESOLVED - 30.8 was plain decode, 47.6-52.7 is MTP; no anomaly.

## 82. Measurement batch: TP3 flat on 27B, MTP_SAFE=1 penalty is ZERO (old -10.4% stale), HSA env tuning does not reproduce (2026-10-22, build-rccl + nightly libs, rig/tp3-27b/, rig/t6-safe/, rig/hsa-ab/)

Three owner-approved measurement closes, all same-session on the post-RCCL-fix stack, all children-run with parent-verified ledgers.

**A. TP3 vs TP2 on the 27B MTP vehicle (Q8_0, -sm tensor, no spec): TP3 is flat-to-slightly-worse.** pp512: TP2 335.67 vs TP3 332.86 (-0.7 to -0.9% in all 3 rounds). tg128: TP2 22.04 vs TP3 22.13 (+0.3 to +0.5%, all rounds). VERDICT: TP2 remains the production cell at 27B; the third card adds nothing here either. With the 9B also flat (section 78A), TP scaling beyond 2 cards shows no benefit on this rig for any measured model. Rig note: llama-swap held the resident 27B (~14.4 GB/card on GPU0/1) BEFORE this run - not auto-loaded by it; the bench ran alongside it without contention. The ledger's agents=0 field is a rocm-smi parser artifact (this version emits no `compute` token in the KFD table).

**B. MTP_SAFE=1 penalty: ZERO - the recorded -10.4% is STALE.** 4 arms x 3 interleaved rounds, 27B vehicle, TP2, greedy -n 128: PLAIN 30.7 / MTP 47.6 (+55% over plain) / SAFE1 47.3 (-0.6% vs MTP) / SAFE0OFF 47.4 (-0.4%). All 9 spec arms byte-identical (sha ); PLAIN differs as expected (batched-verify numerics under TP). cpu_fallback=0 on every leg. One SAFE0OFF_r2 transient dip (41.7 gen / 53.1 prompt) absorbed by the median. CONCLUSION: the AR lock-off is free on this build; the old -10.4% figure (earlier campaign, different build/env) is stale - do not cite it. Practical: MTP ships with SAFE=1 ON at zero cost, which closes the SAFE=2 design question permanently (there is nothing for a revisit state machine to recover). Also: all spec arms byte-identical to each other while differing from PLAIN confirms the losslessness boundary is the batched-verify numerics class (section 76), quantified elsewhere.

**C. HSA env tuning (CV claim +4-7% pp): DOES NOT REPRODUCE single-card.** 3 arms (NONE / HSA_KERNARG_POOL_SIZE=64MiB / +GPU_MAX_HW_QUEUES=8) x 3 rounds x 2 models, 18 valid legs (3 stale NA rows from an aborted first launch excluded): dense 9B all deltas <=0.5%; MoE 30B pp512/tg within 1%, pp16 KAQ +2.6% is inside the 268-291 cross-arm jitter band of a 16-token prompt. VERDICT: no measurable gain on single-card ROCm0; CV's claim likely comes from multi-GPU/tensor-split paths or companion env vars (HSA_FORCE_FINE_GRAIN_PCIE et al). Do not adopt; re-test only if a TP-scale prefill question reopens.

BENCH-file note: section 74's m1 arm and section 77's KS1 arm already measured today's KS1 config on this model; this batch's MoE arms in C used the same self-made Q4_0 with no gate env - NONE-arm pp512 1336 matches section 74's stock 1334 within 0.2% (harness consistency check PASSED).

## 83. CORRECTION: sections 78A and 82A were NOT tensor-parallel runs - this build's -dev delimiter is inverted; corrected TP1/TP2/TP3 matrix (2026-10-23, build-rccl + nightly libs, rig/tp3-30b/, rig/tp-matrix-fix/, rig/tp-clean/)

**A. THE DEFECT.** build-rccl was built at , which PREDATES pr4-llama-bench-dev-comma . On that build the -dev delimiters are INVERTED relative to post-PR4 master:
- `-dev ROCm0,ROCm1` (comma) = one test PER DEVICE, i.e. two independent single-card runs. This is NOT tensor parallel.
- `-dev ROCm0/ROCm1` (slash) = the device SET for ONE split instance. This IS tensor parallel.
`-sm tensor` on a comma list does not error - it silently produces solo rows. Every TP number in sections 78A and 82A was produced with the comma form (`rig/tp3-bench/run.sh`, `rig/tp3-27b/run.sh`), so those two arms compared solo-card rows to solo-card rows. "TP3 is flat" was structurally guaranteed by the harness, not measured.

**B. PROOF (three independent lines).**
1. Row count: `-dev ROCm0,ROCm1` yields 2 pp512 rows; `-dev ROCm0,ROCm1,ROCm2` yields 3 rows. A split run yields exactly 1 row whose dev column is the set.
2. Dev column, archived runs: section 82's `rig/tp3-27b/r1_tp2.log` prints `dev=ROCm0` and `dev=ROCm1` as separate rows; `rig/tp3-bench/tp2_r1.out` (section 78A) the same. No row anywhere reads `ROCm0/ROCm1`.
3. Direct paired measurement, 9B Q4_0, same session: `-dev ROCm0` = 944.84 pp512 / 70.10 tg128; `-dev ROCm0/ROCm1` = 1615.87 / 89.95. That is +71.0% pp / +28.3% tg from one character, and it matches the historical TP2 win direction (references/tp-tensor-split-perf.md), not the flat reading of section 82A.

**C. CORRECTED MATRIX** (slash form; -p 512 -n 128 -fa on -lm dio -sm tensor -t 128 -r 1; medians of clean rounds; same build, same session, 175 W/card, llama-swap stopped, GPU2 co-tenant guard active):

| model | arm | dev | pp512 t/s | tg128 t/s | rounds |
|---|---|---|---|---|---|
| 9B Q4_0 (5.34 GiB) | TP1 | ROCm0 | 941.53 | 70.91 | 6 |
| 9B Q4_0 | TP2 | ROCm0/ROCm1 | 1611.40 | 90.00 | 6 |
| 9B Q4_0 | TP3 | ROCm0/ROCm1/ROCm2 | 1952.21 | 93.54 | 3 clean |
| 30B-A3B Q4_0 (16.11 GiB) | TP1 | ROCm0 | 1328.19 | 112.26 | 6 |
| 30B-A3B Q4_0 | TP2 | ROCm0/ROCm1 | 1785.66 | 99.99 | 3 clean |
| 30B-A3B Q4_0 | TP3 | ROCm0/ROCm1/ROCm2 | 2181.18 | 94.96 | 3 clean |
| 27B Q8_0 (MTP vehicle) | TP2 | ROCm0/ROCm1 | 580.86 | 31.03 | 3 clean |
| 27B Q8_0 | TP3 | ROCm0/ROCm1/ROCm2 | 741.99 | 35.87 | 3 clean |

Deltas:
- 9B: TP2 vs TP1 +71.1% pp / +26.9% tg; TP3 vs TP1 +107.3% pp / +31.9% tg; **TP3 vs TP2 +21.1% pp / +3.9% tg**.
- 30B MoE: TP2 vs TP1 +34.4% pp / -10.9% tg; TP3 vs TP1 +64.2% pp / -15.4% tg; **TP3 vs TP2 +22.1% pp / -5.0% tg**.
- 27B Q8 (single card does not hold the model, no TP1 arm): **TP3 vs TP2 +27.7% pp / +15.6% tg**.

TP2 repeatability is excellent (9B 1602.6-1616.0 pp / 89.85-90.10 tg across six rounds; 30B 1771.9-1795.9 / 97.59-100.32). TP3 prefill is consistent across clean rounds (9B 1948-1954; 30B 2175-2194; 27B 741.5-742.6).

**D. VERDICT - TP scaling is real and workload/architecture dependent.** The third card is NOT dead: it buys +21 to +28% prefill everywhere measured. Decode splits by model: 9B and 27B gain (+3.9%, +15.6%), the 30B-A3B MoE LOSES decode at every TP step (TP1 112.26 is the best decode cell on the rig; TP2 -10.9%, TP3 -15.4%). Production rule, replacing "TP2 is the production cell":
- dense 9B: TP3 is best on both axes.
- 27B Q8: TP3 is best on both axes (and TP1 cannot hold it).
- 30B-A3B MoE: choose by workload - TP3 for prefill-heavy (+64% pp vs TP1), TP1 for decode-heavy (+11 to +15% tg vs TP2/TP3). TP2 is the middle compromise, not the optimum.
Mechanism for the MoE decode loss is untested here; the standing MoE decode story (activation-encoder launches, section 74) is the first hypothesis, and TP multiplies collective cost per token on a model whose decode is already encoder-launch-bound.

**E. CO-TENANT CONTAMINATION INCIDENT (why three rounds were discarded).** Mid-sweep, a memory-provider `llama-server` (30B-A3B MoE, Q4_0) appeared pinned to `-dev ROCm2`, holding 18.7 GB (rocm-smi KFD table, PID 515639 / proc 42783, started 21:20). Its effect on TP3 arms was severe and BIMODAL, not a uniform slowdown: 9B tg 93.2-93.6 clean vs 64.1-77.7 contended; 30B tg 94.5-95.3 vs 61.8-67.1; 27B tg 35.9 vs 25.3. Note the contended/clean ratio clustered at 0.65-0.70, which initially read as a clock-stall signature - it is not; telemetry showed GPU2 at 1850 MHz / 166-192 W while GPU0/1 idled at ~20 W, i.e. the tenant owned the third card. First suspects checked and cleared: PCIe (all three cards at 16 GT/s x16; the x2/2.5 GT/s entry is the BMC VGA at 05:00.0), NUMA (all node 0), thermal (30-47 C idle). After the tenant exited, re-measured legs returned the clean mode exactly. RULE: before and during any TP3 leg, check `rocm-smi --showpids` for a KFD holder on the third card and per-card power; a contended TP3 arm manufactures a false "TP3 is unstable/flat" verdict in the same way a vacuous arm manufactures a null.

**F. RETRACTIONS.** Section 82A ("TP3 flat on 27B, TP2 remains the production cell") and section 78A's 9B TP3 leg are STRUCK as TP measurements - they measured solo-device rows. Section 82A's MTP_SAFE=1 arms are NOT affected (single-model spec comparison on the same TP2 setup; direction and byte-identity conclusions stand). Section 76/75 spec work is unaffected. The artifact evidence is preserved: contaminated copy at rig/tp-matrix-fix (r1-r6, includes both modes), clean copy at rig/tp-clean, first (comma, invalid) 30B sweep kept at rig/tp3-30b for audit.

**G. HARNESS NOTES.** (1) With `-r 1` llama-bench prints `+- 0.00`; repeatability comes from interleaved rounds, never from the printed sigma. (2) Parse the table by the `dev` COLUMN, not by row count: the split arm is the one whose dev field contains a slash. (3) `-sm tensor` with a single device is legal and is the correct TP1 anchor. (4) A TP3 arm must be re-verified per session - a co-tenant can appear between rounds.

## 84. MTP ADOPTION DECISION (owner, 2026-10-23) - config-only; on for inference, off by default in benchmarks

Scope first, because it was ambiguous in the transfer brief: NOTHING in this program is a pending MTP code change. MTP_SAFE=2 was found to be a log line (BENCH 71); the sampler-guard fix is unreachable on the serving path and stays unmerged (BENCH 78); the CV MTP-overhead port is default-off, byte-identical and a perf null (BENCH 81/81a). The open item was always a deployment decision.

DECISION: MTP ON for real inference where the model qualifies; OFF by default in any benchmark arm that is not measuring MTP.

PRODUCTION CONFIG (from BENCH 79/80):
- dense Q4_0-class: gate stack + MTP = +17.7% decode; MTP costs ~13% prompt t/s.
- MoE Q4_0-class: MTP ALONE, gate OFF = +25.2%; the gate stack SUBTRACTS 4.3% from MTP on MoE. K-split stays on for prefill.
- Q4_K_M-class MoE: do NOT enable MTP (-6.6%); quant family is first-order for spec decode, check the tensor histogram first.
- Accepted risk, stated once: 7/10 prompts carry one benign rank-2 flip at 0.01-0.06 nats, zero fundamental flips in 2560 tokens, on ONE vehicle and ONE seed. Rerun the losslessness suite per model before promising a new one.

BENCHMARKING RULE: two independent reasons MTP stays off in bench arms. (a) The ~13% prompt-t/s cost contaminates any pp comparison against a plain arm. (b) Batched verify changes decode numerics - spec arms are byte-identical to each other while differing from PLAIN - so a plain decode A/B with MTP on measures a different computational path. When MTP IS the variable, every arm must witness `"speculative": true`; section 81 is the cost of skipping that check.

UNMEASURED, recorded not assumed: MTP x TP3. TP2+MTP is characterized (+65-99% on the 27B vehicle); TP3+MTP is not, and TP3 decode is already -5.0% vs TP2 on the 30B MoE without MTP (BENCH 83). Validate per (model, TP cell) before shipping MTP in a mixed prefill+decode workload.

## 85. CV PREFILL LEAD DECOMPOSED with CV's own knobs: transport is half the gap, CV's kernels are the other half (2026-10-23, benchmarks/qwen38-27b/cv-h2h-RESULTS.md)

Motivation: v9b confirmed CV +22.6% pp over our fork at TP2 with decode tied. Before porting anything, decompose the lead using CV's OWN env knobs (names verified by strings on their libggml-hip.so): GGML_CUDA_AR_P2P (BAR peer-write transport) and GGML_CUDA_AR_BF16_THRESHOLD (bf16 wire). Arms: fork (session anchor), cv-def, cv-nobar (P2P=0 + BF16TH=0 = host-staging F32 ~= our effective transport), cv-barf32 (P2P=1 + BF16TH=0 = BAR transport, F32 wire). TP2 ROCm1+ROCm2, 27B Q4_0, plain, 2 rounds, rotated order, /proc/environ witness captured per leg.

Medians (pp / tg t/s):
- fork 433.0 / 45.76
- cv-nobar 485.4 / 44.22
- cv-barf32 509.4 / 45.20
- cv-def 534.6 / 45.20

DECOMPOSITION of the +23.5% prefill lead:
- +12.1%: NOT the collective. CV with our effective transport still beats our fork by 12% - that residual lives in CV's compute/scheduler side (their GCN GEMM family incl. split-K, edge-balanced split, host-side pipelining).
- +4.9%: BAR peer-write transport (uncached peer-VRAM inbox, per-block flags, ONE barrier vs our two, no host bounce).
- +5.2%: bf16 wire (half the PCIe bytes).

Decode: every arm 44.2-45.9, fork highest - the transport difference is invisible at decode, consistent with a bytes-bound large-message prefill effect and with v9b's decode tie.

PORT SCOPING (owner directive: apply CV's method to our build, stacked on W4A4): the collective port (BAR peer-write + bf16 wire into tp-allreduce.cu) is worth ~+10% TP2 prefill. Risk class: parity/deadlock - CV proves the protocol on this exact silicon, but our NRANKS-template structure differs. Staged plan: (1) bf16 wire first (smaller change: wire dtype + threshold on our existing 2-rank broadcast path), CPU-seal + TG-anchor + KLD-neutral check, measure; (2) single-barrier peer-write transport second, behind its own env gate, greedy seal + parity harness; (3) only then consider the kernel-side half (their GEMM family - separate project, much larger).

WITNESS RULE EXTENSION: for CV knob A/Bs, /proc/PID/environ of the server process is the engagement witness (their knobs emit no log lines).

## 86. bf16 wire port to our AR: implemented, witnessed, and NEUTRAL at TP2 (2026-10-23, rig/bf16-patch, build-bf16, snapshot rig/bf16-bin, A/B rig/bf16-srv)

Goal: capture the +5.2% of CV's prefill lead that the v10 decomposition attributed to their bf16 wire, stacked on our W4A4 config.

IMPLEMENTATION (tp-allreduce.cu/.cuh, env gate GGML_TP_AR_BF16_THRESHOLD, default 0 = byte-identical stock):
- new kernel k_broadcast_reduce_bf16<NRANKS>: phase 1 narrows our input to bf16 (bit-op round-to-nearest-even, no bf16 intrinsics) and peer-writes 4 elements per 8-byte store; phase 2 widens peers back to F32 and accumulates in F32, self contribution read from `input` in full F32. Parity/xbar support retained.
- host: threshold + 4-element alignment select the kernel on the broadcast path only (twoshot/one-shot stay F32); staging per-slot size halves when bf16 is selected; plan carries a bf16 flag; a one-shot witness line "TP custom AR: bf16 wire ACTIVE (n_elements=... >= threshold=...)" marks engagement.

MEASUREMENT (TP2 ROCm1+2, 27B Q4_0, server /completion, 840-token pp + 10x128 tg, W4A4 speed profile, 2 interleaved rounds 2; witness checked per leg):
- f32 (build-rccl control): pp 451.1 / 454.2, tg 46.03 / 46.09
- bf16 (build-bf16): pp 448.3 / 453.5, tg 46.40 / 46.36
- delta: pp -0.38%, tg +0.69% = NEUTRAL.

WHY (mechanism, not hand-waving): at n_embd=5120 the per-AR payload is 20 KB F32 vs 10 KB bf16, but our AR cost is dominated by barrier WAIT (39-49% of AR ticks, the SKIP_XBAR record) - latency, not bytes. Halving the wire removes bandwidth pressure we were not paying for. CV's bf16 arm gained +5.2% inside THEIR transport (BAR peer-write into an uncached peer-VRAM inbox, one barrier, no host bounce), so the bytes saving pays there and not here.

VERDICT: bf16 wire is NOT a perf feature for our build. Kept env-gated default-off and documented neutral (harmless, useful only if a future transport makes us byte-bound). The transferable part of CV's lead is their transport design as a whole and their compute/scheduler side (+12.1%), not the wire width.

HARNESS NOTES (both cost a run): (1) the custom AR engages ONLY with HSA_FORCE_FINE_GRAIN_PCIE=1 (w4a4-env.sh); an A/B that drops that env silently measures the generic NCCL-style path on both arms and reads identical - the witness line is what caught it. (2) The incremental-rebuild trap bit again: building the feature, restoring pristine sources, then building other targets recompiled tp-allreduce.cu from restored source and silently removed the feature; build all targets in ONE cmake invocation and snapshot `bin/` before restoring.

## 87. TENSOR SHARDING ACTIVATED (one-line fix) + CV GCN K-quant GEMM port + AR A/B closure: peer-write loses to the RCCL ring in sharded topology too (2026-10-23/24, rig/floor-u4x16, build-tp2, 27B UD-Q4_K_XL)

### 87.1 The sharding divergence: one line

Our TP2 was a layer-split pipeline with FULL-SIZE weights duplicated on both cards
(~17.6 GB/card). CV true-shards: ~8.8 GB/card, half-size GEMMs, 2 ARs/layer
(Megatron-style: attn o-proj axis-0 -> partial-sum AR; ffn_down axis-0 -> AR; qkv/up
axis-1, no comm). Root cause of the divergence was NOT missing machinery - our
split-axis declarations and split-state plumbing are CV-equivalent. It was usage
ordering: ggml_backend_meta_buffer_type_alloc_buffer_n ran init_tensor_impl before the
buffer usage was stamped, and llama stamps GGML_BACKEND_BUFFER_USAGE_WEIGHTS only AFTER
the allocator returns. The split-state leaf test (usage != COMPUTE) therefore never saw
weights as weights and fell through to the silent MIRRORED default.

FIX (ggml-backend-meta.cpp, one line): meta_buf->usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS
stamped directly inside alloc_buffer_n before the init loop (set_usage would deref
not-yet-allocated lane bufs - do not use it here).

RESULT (27B UD-Q4_K_XL, TP2 warm server protocol, GCN off):
- VRAM: 12.21 GB/card (was ~18 duplicated) = 8.8 GB sharded weights + KV.
- Correctness: 17x23 = 391 (note: reasoning-template models put the answer in
 reasoning_content; an earlier "empty output" was a max_tokens truncation artifact).
- Throughput: 408 tok/s pp steady vs 392 duplicated = +4%.
- Pipeline-fill control: 4200-token prompt (4.1 ubatches) runs at the same 401 tok/s -
 no pipelining gain from longer prompts; under layer-split the serial layer chain caps
 scaling at ~1.75x, which is why sharding was the correct fix.

### 87.2 GCN K-quant GEMM port (CV's kernel family)

CV's gcn-q8-gemm.cu/.cuh (2300 lines: 64x64x128 tile, 256 threads 3 blocks/CU, XOR-swizzled
LDS, alignbyte/perm gfx906 asm) ported into our tree, hooked at the top of
ggml_cuda_mul_mat_q (env GGML_CUDA_GCN_KQ, default on), gated to dense K-quant GEMM.

MEASUREMENT (16-request warm sweeps, TP2 and 1-GPU):
- TP2 sharded: ~409 vs ~408 GCN off = FLAT (+0.3%).
- 1-GPU: +4.6% (235 vs 225).
- CV's own stack: +32% (443 vs 587... their gain with the kernel ON).
- The dip-and-recover pattern seen in earlier short runs was a shared periodic stall
 (GCN=0 shows it too), not GCN decay; streak-medians protocol handles it.

WHY FLAT HERE, +32% THERE: under our (former) layer-split the GEMMs were not the critical
path - the serial layer chain was. Under sharding the GEMMs ARE half-size and on the
critical path, yet still no gain. The remaining difference vs CV is below the kernel gate:
CV's +32% includes their scheduler/split-K/edge-balanced work. GCN stays default-on
(harmless, worth +4.6% single-GPU).

### 87.3 AR A/B in sharded topology: chunked peer-write loses to RCCL ring

The one remaining transport hypothesis (CV's +4.9% BAR peer-write) re-tested in the
topology that matters - sharded, 64 per-layer ARs on the critical path. CV-protocol
rework of the chunked path (k_p2p_piece: volatile arrival flags, monotonic token,
s_sleep backoff, uncached inbox, NO end barrier - piece parity instead), uncommitted
rework discarded after the verdict (the earlier chunked WIP stays committed as
).

MEASUREMENT (TP2 sharded, 27B UD-Q4_K_XL, warm server, 12 requests, GCN off):
- RCCL ring (default): 408 steady.
- chunked piece=131072: 397.8 median = -2.5%.
- chunked piece=1048576 (tuned): 398.5 = -2.4%. Piece-size tuning does not recover it.

NOT a vacuous arm: GGML_TP_AR_KTIMING shows 640 custom-AR calls/rank with bar_end=0%
(the no-end-barrier kernel engaged on the 64 large ARs/request; the ar-debug gate lines
did not print - instrumentation gap, KTIMING is the reliable witness).

MECHANISM (measured): ktime phase breakdown - bar_start (volatile-flag spin) 33k
ticks/call vs 11.7k write ticks. Waiting on the peer's arrival flag over PCIe BAR costs
~3x the data movement. Same conclusion as section 85's +4.9% attribution: peer-write
pays off inside CV's whole transport design (one barrier, no host bounce), not bolted
onto ours. RCCL ring is the right AR for prefill-sized tensors on this rig; transport
is CLOSED.

Also landed here: tg_warm_needed 4->2 - alternating-shape A/B (840/420 tok x 12) shows
NO post-switch penalty at steady state (407/406/405.6 immediately after switches); the
7-13% shape-change penalty is gone. Debug instrumentation (GCN-KQ pointer dump) stripped
before commit.

### 87.4 Scoreboard after this phase

| config (TP2, 27B UD-Q4_K_XL, warm pp) | tok/s |
|---|---|
| our fork, duplicated weights, MMQ | 392 |
| our fork, SHARDED, MMQ | 408 |
| our fork, sharded + GCN | 409 |
| CV, MMQ equivalent | 443 |
| CV, + their GCN | 587 |

Remaining gap 408 vs 443 (-8%) is inside CV's compute/scheduler side. Next candidates:
their split-K / edge-balanced GEMM scheduling, host-side pipelining. NOT transport
(closed twice), NOT the GCN kernels themselves (flat on our stack).

## 88. Wide-J MMQ tile port: neutral on our stack - the gap is in the kernel bodies, not the tile table (2026-10-24, build-tp2, 27B UD-Q4_K_XL sharded TP2)

Mechanism #1 from the sched-gap analysis: CV's mmq-config-gcn.cuh carries wide token-tile
CASEs (512 threads, I=128, J=80/96/112/128) for Q4_K/Q5_K/Q6_K/IQ4_XS that our table
lacked (ours capped K-quants at J=64). Ported all 8 insertion points verbatim (fb +
non-fb variants), added MMQ_J_TRACE (per-GEMM J selection log) and MMQ_J_MAX (J cap for
A/B, default 128) env knobs.

QUANT-MIX FINDING (MMQ_J_TRACE): our copy of UD-Q4_K_XL is dominated by IQ4_XS
(majority of gate/up + all KV/QKV/O shapes), with IQ3_S down-projections and a minority
of Q3_K gate/up rows. CV's in-code comment claims "mostly q5_K/iq4_xs/q4_K/q6_K" - a
different mix than what we measure on the same GGUF file. Neither fork has wide-J cases
for Q3_K or IQ3_S, so those shapes run J=64 on both stacks regardless.

SAME-WINDOW A/B (12 requests per arm, same build, GCN off, warm protocol):
- MMQ_J_MAX=64: steady ~394.8 (3 dips: 288/380/385 - the shared periodic stall).
- MMQ_J_MAX=128: steady ~393.0 (1 mild dip 389.0), J=128 engaged on 1680 calls
 (IQ4_XS majority), J=64 on 96 (Q3_K/IQ3_S).
- VERDICT: wide-J = -0.5%, NEUTRAL on our stack. The sched-gap report's +5-8% estimate
 (anchored on CV's in-code tuning comment, its own caveat #1) does not reproduce.

IMPLICATION: at MATCHED tile configurations (J=128 IQ4_XS / J=64 IQ3_S+Q3_K on both
stacks), CV still leads ~13% (443 vs 393). Combined with the kq4 GCN port also being
flat on our stack, the gap is pinned to the MMQ KERNEL BODIES: our mul_mat_q kernel
implementation differs from CV's gfx906-tuned kernels even at identical tile shapes.
Their fork is gfx906-only and every kernel is hand-tuned for it; ours is upstream-derived
mmq. Next leg: kernel-body diff of the K-quant MMQ path (mmq.cu kernel templates for
Q8_1/Q6_K/Q8_0 SRAM layouts at 512 threads), CV vs ours.

DECISION: keep the wide-J CASEs (CV parity, harmless, may pay once our kernel bodies
are tuned), keep MMQ_J_MAX + MMQ_J_TRACE as permanent A/B/diagnostic knobs.

## 89. CV P2P transport engaged on our fork: +3.8% - the gap was two dispatcher conditions, not the transport (2026-10-24, build-tp2, 27B UD-Q4_K_XL sharded TP2)

Ported CV's allreduce.cu/.cuh verbatim (1327 lines: p2p push kernel with per-block
volatile tokens in the peer's VRAM, uncached inbox/flags, 256-slot event ring, BF16
wire with 128K-element auto-threshold) replacing our 977-line stock file. First
drop-in changed NOTHING (~394): the transport was compiled in but unreachable. The
instrument chain (4 fprintf probes: pipeline init entry, P2P status, ar_allreduce
entry, allreduce_tensor dispatch) found TWO dispatcher conditions:

1. ggml-cuda.cu: ggml_cuda_has_internal_allreduce was a compile-time false for
 GGML_USE_HIP. True for the stock allreduce.cu (which stubs HIP), but CV's file
 has a full HIP implementation (only MUSA is stubbed). The dispatcher refused the
 internal path before init_internal ran. Now MUSA-only.
2. ggml-cuda.cu: the custom-AR size gate was wrapped in
 'if (s_gate_on && s_have_nccl)'. With GGML_HIP_RCCL=OFF, s_have_nccl=false meant
 the gate never applied, so the custom path claimed EVERY tensor (dispatch trace:
 custom=1 on ne=4280320 prefill ARs) and try_allreduce never fired. Gate now
 applies unconditionally.

MEASUREMENT (same-window, warm, n=12): 402 -> 417.4 median pp (415.4-418.8 tight).
Correctness 17x23=391. Witness: internal FIRED=254, ar_allreduce engages the p2p
kernel. CV's own decomposition: their all-off 403 + AR_P2P = 439-440 (+9% on their
stack); ours gains +3.8% - the delta is calling-context integration, not concept.

METHOD NOTE (the confound that hid this for hours): single-knob tests on CV left
their other custom families active, so each test measured one feature inside a
stack that still had the rest. The decisive experiment: ALL custom families off
simultaneously (= 403 = our level), then re-enable one at a time (AR_P2P +9%,
FA +0.7%, GCN_GEMM ~0%). Attribution rule: decompose on an all-off base.

Also closed: edge-balanced split (LLAMA_SPLIT_EDGE_BALANCE) is inert at -sm tensor
(the gate needs multi-device layer split; tensor mode exposes one meta device on
both forks); HSA env knobs neutral on the sharded path; ROCm runtime eliminated
(CV's binary on our nightly libs reproduces its numbers exactly).

DECOMPOSITION CORRECTION (review finding): against CV's FULL stack (587) we closed
13% (392->417.4 of 392->587); the dominant residual is the GCN K-quant GEMM family
(+32% on CV, flat +0.3% on ours, kernel bodies byte-identical - unexplained).
Against CV's non-GCN p2p level (439) we closed half. Scoreboard: ours+RCCL ~394 /
ours+customAR ~402 / ours+CV-P2P ~417 / CV all-off 403 / CV+P2P 439 / CV full 587.

Build state: build-tp2 = GGML_HIP_RCCL=OFF + CV allreduce.cu + the two fixes +
GGML_CUDA_ALLREDUCE=internal required (Linux default is nccl). The committed tree
is diagnostic-free; the box source mirror still carries diagnostic fprintf blocks
(AR-DISP/AR-CALL/META-AR/AR-P2P/AR-INIT) - strip or leave as scratch.
## 90. MoE W4A4 gate re-VALIDATED on Qwen3.5-35B-A3B: AMEND the gate-OFF verdict to config-conditional gate-ON (2026-10-24, build-tp2, benchmarks/qwen35-35b-a3b/moe-levers-sec90.md / -sec91.md)

Section 80's "MoE prod = MTP alone, gate OFF" was measured on Qwen3.6-35B-A3B
Q4_0 in the MTP-ON regime (the -4.3% is a gate x MTP interaction: batched
verify multiplies activation-encoder launches). It does not transfer to
MTP-OFF serving on the new model class. Re-validation on Qwen3.5-35B-A3B
bartowski Q4_0, TP2 (prod cell), same-window same-binary env arms, server
protocol, MTP off, 3 discarded warmups, medians of 3:

| arm | pp512 | tg128 | probe |
|---|---|---|---|
| canon (REPACK_Q4_0=0) | 1561.7 | 95.31 | PASS |
| u4 (REPACK_Q4_0=1 + PROFILE=speed) | 1620.2 (+3.7%) | 100.43 (+5.4%) | PASS |

Quality battery (temp-0 500-tok gens x 4 prompts, canon/rep/u4 arms): arms are
NOT byte-identical - repack changes prefill accumulation, the speed profile
changes decode numerics (KSPLIT/SUMW) even below the MIN_TOKENS floor; all
outputs coherent. Byte-compare is therefore not the gate for this path (same
finding as prior kernel ports). llama-perplexity single-engine sanity (wikitext
32 chunks, same binary, env arms): canon 5.4564 vs u4 5.5343 = +1.43% paired -
the known activation-quant cost the dense prod fleet already pays, far below
the +5.5% red line that flagged the IQ2_S decode bug.

AMENDED PROD GUIDANCE (MoE Q4_0-class, gfx906):
- MTP OFF serving: gate stack ON (REPACK_Q4_0=1 + W4A4_PROFILE=speed).
- MTP ON serving: gate stays OFF until the gate x MTP interaction is re-measured
 per model (section 80 vehicle: -4.3%; Qwen3.5-35B-A3B x MTP: UNMEASURED).
- Treat the speed profile as numerics-changing: evidence bar = PPL + coherence,
 not byte-equivalence.

Also this session: moe30b-base-Q4_0-selfmade cleared as quant-healthy (the
"391" repetition loop is a prompt-specific base-model attractor: 3 substantive
temp-0 generations fully coherent, 8-gram repetition scan clean; Task-4 KLD
PPL ratio 1.04 / KLD 0.057-0.077 stands). Keep in baselines.

## 91. LEVER 2: gcn_kq_moe MoE launcher wired into mul_mat_id + Q3_K admission: +26-30% MoE prefill, decode untouched (2026-10-24, build-tp2, benchmarks/qwen35-35b-a3b/moe-levers-sec90.md / -sec91.md)

The expert-grouped v4 MoE GEMM launcher (gcn-q8-gemm.cu, byte-equivalent to
CV's) was dead code in our tree - CV hooks it in ggml_cuda_mul_mat_id before
the MMQ branch and fuses gate/up+GLU at their ggml-cuda.cu:8039. Ported the
dispatch hook (after the MMVQ decode branch, before MMQ, CV pattern) + Q3_K
admission (UD routed gate/up are q3_K in 39/40 blocks; the v4 template
already supported it - dense path admits it). Env gates:
GGML_CUDA_GCN_KQ_MOE (default 1), GGML_CUDA_GCN_KQ_MOE_Q3K (default 1),
routing witness GGML_CUDA_GCN_KQ_MOE_TRACE=1 (first 12 launches, off by
default - traces OFF in measurement legs per standing policy). CV gate
predicates inherited verbatim: MIN_AVG=12 tokens/expert (decode stays on
MMVQ), BN=32 tiles when avg<=64 else 64, K%QK_K==0, alignment checks.
NOT ported (follow-up): CV's gate/up+SwiGLU graph fusion (launch-count cut).

MEASUREMENT (same-window same-binary env arms, server protocol, MTP off,
3 discarded warmups, medians of 3, UD-Q4_K_XL = Qwen3.5-35B-A3B):
| leg | pp512 OFF | pp512 ON | delta | tg128 OFF | tg128 ON |
|---|---|---|---|---|---|
| TP2 | 1315.4 | 1708.4 | +29.9% | 90.06 | 90.03 (neutral) |
| 1 GPU | 1040.5 | 1310.8 | +26.0% | 73.52 | 73.44 (neutral) |
Decode neutral as designed (MMVQ keeps decode; the gate only takes prefill).
UD TP2 pp512 1708 now EXCEEDS the Q4_0-u4 arm from section 90's vehicle class
(1620) on the K-quant file - the K-quant gap vs our own Q4_0 path is now
mostly closed on prefill.

CORRECTNESS GATES:
- 17x23 probe PASS on every leg (Q4_0 canon/u4, UD off/on, 1G + TP2).
- Q4_0 (hook must NOT fire): new binary byte-identical to pre-patch binary on
 4 temp-0 500-tok prompts, 0 hook launches witnessed. PASS.
- UD greedy: prompts with prefill < MIN_AVG gate byte-identical off vs on
 (gate correctly declines); 512-token prefill (avg 16/expert) diverges as
 expected for a numerics-changing kernel port; all outputs coherent.
- PPL spot check (single engine, same shape arms): llama-perplexity does NOT
 exercise this path - MoE MUL_MAT_ID in perplexity never passes the MIN_AVG
 gate at any tested shape (c512/ub512 and c2048/ub2048, n_seq 2/4: identical
 PPL off vs on, 0 MoE launches traced while the dense kq4 witness fires in
 the same process). Numerics evidence = greedy divergence + coherence +
 substantive long-prompt A/B + probe (documented, same bar as prior ports).
- Routing witness: 12 launches witnessed server-side: q3_K gate/up (K=2048
 M=512) + q4_K down (K=512 M=2048), experts=256, tokens=508, avg=16, bn=32.
Build note: box tree carried a pre-existing uncommitted lane2 hunk (dense
iq2_s mask, inert for all models tested); committed source excludes it.


## 2026-10-25: gcn_kq_mv matvec gate default flipped ON (GGML_CUDA_GCN_KQ_MV)

Dense 27B K-quant TP2 (UD-Q4_K_XL, FA_DENSE_GCN on). Quality evidence, same-window
paired arms, build-tp2:
- Decode-shaped PPL (llama-perplexity -b 1 -ub 1; matvec engagement witnessed per
 leg): OFF 5.8273 vs ON 5.8320 -> ratio 1.0008 (Task-4 DOT8 precedent 1.04).
- Native KLD, same tokens, decode-shaped: mean 0.001805 +/- 0.000147, median
 0.000573, max 0.0891, 99pct 0.0371 (Task-4 precedent 0.057-0.077).
- Paired temp-0 gen, 1000 tok, top-5 logprobs: identical argmax first 172 tokens,
 mean |logprob delta| shared top-k 0.0074 pre-divergence; fluent post-divergence.
- Default-ON build: 17x23=391 PASS; greedy battery byte-identical to explicit-ON.
Baseline perf on record (item1 A/B): 548.55 pp840 / 38.71 tg128 gate-off; +5.76%
tg with gate on. Set GGML_CUDA_GCN_KQ_MV=0 to restore stock dispatch.


## 2026-10-31: gate-truth verification - true binary defaults vs the production env bundle (dual seal)

Source-verified gate audit plus a dual seal run on the seal build (same binary as the
prior seal witness):

- `GGML_CUDA_REPACK_Q4_0` defaults **OFF** in source (opt-in admission check). With true
  binary defaults, Q4_0 tensors are never admitted to the repack path, so the tiled
  `gcn_u4_gemm` (its own gate is default-ON but sits downstream of admission) and the
  W4A4 u4 GEMMs do not engage for Q4_0. The "default-ON tiled/W4A4" phrasing describes
  the env bundle, not the binary default.
- Env-profile seal (production bundle: `REPACK_Q4_0=1 TP_AR_SKIP_XBAR=1 REPACK_KQUANT=0
  MOE_VEC=1 MOE_FUSE=1 GDN_COLTHREAD=1`): all 4 cells GREEN on case hashes - all 24 case
  hashes bit-match the prior seal run and the promoted baselines, zero drift. The
  baselines were recorded WITHOUT `REPACK_Q4_0`, so this is the zero-divergence
  evidence: tiled/W4A4 repack ON is bit-identical to the default path (validated
  widths; batch 2-15 unsealed).
- True-defaults seal (zero `GGML_*` vars in the server environ): diverges from the
  baselines on the long-prefill cases (plus one tensor-def cell) - exactly the cases
  sensitive to the bundle gates (`MOE_VEC`/`MOE_FUSE`/`GDN_COLTHREAD` off, `REPACK_KQUANT`
  back to default ON). Empirical proof that the published results describe the env
  bundle, not bare defaults.
- Engagement witnesses: `[attn-qk-fused] decode q/k chain fusion ACTIVE` printed in all
  8 cells of both profiles (fusion is structurally default-ON, a genuine engagement
  witness at the call site). The repack engagement witness is the PAIR `REPACK_Q4_0 gate
  ACTIVE` + `REPACK_Q4_0_TILED gate ACTIVE`; the TILED banner alone prints whenever the
  gate is read and is not an engagement witness on its own. The seal harness fails any
  cell whose required banners are missing.

Full gate-default table (per-gate source defaults vs bundle values) and both seal
matrices: `benchmarks/defaults-vs-env.md`.
