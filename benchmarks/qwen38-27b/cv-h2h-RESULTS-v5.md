# CV vs stock vs fork -- v5 head-to-head (Qwen3.8-27B, Q4_0 + Q8_0, TP2)

**2026-10-19, box mi50.** Three engines, each at its best-case config, server-based A/B,
`-sm tensor` on ROCm0+ROCm1, 6 arms x 2 rounds with reversed order in r2.

## Engines (build-verified before the run)

| engine | commit | verify |
|---|---|---|
| Castagna Veloce | (0.4.1-dev) | `--version` + own libggml-hip.so |
| upstream stock (mxxm-t) | (0.6.0-dev b11756) | `--version` |
| our fork (int4-profile) | (0.6.0-dev b11757) | `--version` + W4A4_PROFILE x9 / MIN_TOKENS x18 in libggml-hip |

## Protocol

- Server `/completion` A/B (llama-bench unusable: TP+repack pool bug, tp-tensor-split-perf.md).
- Shared flags: `-fa on -sm tensor -c 16384 -b 4096 -ub 1024 -t 16 -np 1`; 175W caps; flock-gpu serialized.
- pp = 840-token prompt, n=1; tg = 10 prompts x 128 greedy tokens, cache_prompt=false.
- Engine envs (best-case each):
 - CV: system ROCm, `HIP_VISIBLE_DEVICES=0,1`, `LLAMA_CKPT_LAST_VERIFY=1`, **VPG left at CV default (auto-ON)**, `-ngl 999`, no `-lm dio` -- their docs/models/qwen3.8-27b.md serve env (MTP flags omitted: this matrix is plain decode/pp).
 - stock/fork: `rig/w4a4-env.sh` (TheRock runtime first, CUSTOM_AR=1, FINE_GRAIN_PCIE, HW_QUEUES=8, SKIP_XBAR), `--device ROCm0,ROCm1 -lm dio`, `-ngl 99`, `GGML_CUDA_REPACK_Q4_0=1`.
 - fork adds `GGML_CUDA_W4A4_PROFILE=speed` (= DOT8=1 + MIN_TOKENS=128 + KSPLIT=1 + SUMW=1; greedy-seal-identical to the explicit set).
- Models: models/Qwen3.8-27B-{Q4_0,Q8_0}.gguf (16.06 / 29.05 GB, both pre-existing).

## Results (t/s; r1/r2 spread <= 0.8% on every leg)

| arm | pp | tg |
|---|---|---|
| cv-q4 | 530.8 [532.0/529.5] | 45.07 [45.08/45.06] |
| stock-q4 | 366.8 [362.3/371.3] | 42.84 [42.97/42.71] |
| fork-q4 | 442.8 [444.5/441.1] | **45.64** [45.60/45.69] |
| cv-q8 | **564.6** [565.5/563.7] | 35.71 [35.69/35.73] |
| stock-q8 | 512.8 [512.6/513.0] | **36.69** [36.64/36.73] |
| fork-q8 | 507.5 [507.4/507.6] | 36.46 [36.50/36.43] |

### Paired deltas (same-round arm/stock, both rounds shown)


Q4_0: cv vs stock: pp +46.9%/+42.6% tg +4.9%/+5.5%
 fork vs stock: pp +22.7%/+18.8% tg +6.1%/+7.0%
 fork vs cv: pp -16.4%/-16.7% tg +1.1%/+1.4%
Q8_0: cv vs stock: pp +10.3%/+9.9% tg -2.6%/-2.7%
 fork vs stock: pp -1.0%/-1.1% tg -0.4%/-0.8%
 fork vs cv: pp -10.3%/-10.0% tg +2.3%/+1.9%


## Findings

1. **CV's headline is real but prefill-only and Q4_0-weighted.** +45% pp over stock on Q4_0 -- but on
 Q8_0 it collapses to +10% pp and their tg is *negative* vs stock (-2.7%). Their decode engine is
 not faster; their TG story is entirely MTP overhead engineering (their own serve guide ships MTP).
2. **Our fork wins tg on every pairing it can influence**: fork tg beats CV on Q4_0 (+1.3%) AND Q8_0
 (+2.1%), and beats stock on Q4_0 (+6.5%). Best plain-decode arm in the whole matrix on Q4_0.
3. **Q4_0 pp: fork closes half of CV's lead** (+22.7% vs stock vs CV's +45%); the residual -16.5% is
 the TP-comms/meta-scheduler residual -- the BAR peer-write AR port target, consistent with the
 Oct-9 analysis (CV-FORK-ANALYSIS 3.1b, 5.1).
4. **Q8_0 parity where we have no path**: fork ~= stock (-1% pp, -0.6% tg) as expected -- repack/W4A4
 are Q4_0-keyed; profile knobs are inert on a Q8_0 file. CV's Q8 pp edge (+10%) is their
 gcn-q8-gemm; they still lose tg to stock there.
5. **VPG null result (closes an open question from Oct-9)**: CV ran VPG auto-ON today (their default,
 2x gfx906 qualifies) vs VPG=1 (off) in October -- pp 530.8 today vs 529.0 then: no measurable
 effect on 27B/TP2 at these settings. Their virtual-device pp claim was measured on 4-card
 layer-split; it does not transfer to this topology.
6. **Consistency cross-check**: the Q8 tg prompt-1 early EOS (82 tokens) reproduces EXACTLY on all
 three engines (pred_n 1234 in every Q8 arm) -- greedy determinism holds across engines; the
 truncation is prompt-side, not engine-side.

## Watch items

- CV q8 server segfaulted AT TEARDOWN (after "cleaning up before exit", post-data) in r1 -- the
 driver reported it; server logs end cleanly and both rounds' JSONs are complete. Teardown-only,
 data unaffected; same class as the Oct-9 v3 queue-hang watch item.
- This matrix is PLAIN decode/pp (no MTP), per the 6-test spec. CV's published serve config is
 MTP-on; the natural follow-up is a v6 with MTP legs using today's fork (floor makes gate+MTP safe).

Raw artifacts: mi50:rig/cv-h2h-v5/ (h2h-v5.log, *-r[12]-{pp,tg}.json, *-server.log, BUILDS.md, PREP.md).
Driver: rig/cv-h2h-v5/h2h-driver-v5.sh. Aggregator: aggregate-v5.py.
