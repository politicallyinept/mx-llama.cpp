# v8 -- full 4-engine x MTP matrix: stock mx vs W4A4 fork vs Castagna Veloce vs Llaminar

**2026-10-21/22, box mi50.** Qwen3.8-27B, TP2 on ROCm1+ROCm2 (ROCm0 reserved -- external
agent), 175 W caps, flock-serialized, 2 rounds/arm, engine order rotated. Q4_0 file and
Q8_0-f32gdn file identical across all engines (llaminar requires f32 GDN controls).
MTP = each engine's own proven flags; shared Q4_0 draft for the llama.cpp family.

## Engines
| engine | commit | note |
|---|---|---|
| stock mx | (upstream master) | + REPACK_Q4_0 (stock-available), w4a4-env, -lm dio |
| W4A4 fork | (int4-profile) | PROFILE=speed (= stock config + W4A4 gates) |
| Castagna Veloce | (0.4.1-dev) | CKPT_LAST_VERIFY=1, VPG default, no -lm dio |
| Llaminar | (develop) | own patched RCCL (built from source this session), FP32 activations |

## Results -- Q4_0 (pp / tg t/s, acceptance %)

| engine | plain | MTP | accept |
|---|---|---|---|
| stock | 367.0 / 42.87 | 306.7 / 53.81 | 73.4 |
| CV | **534.0** / 43.39 | 374.8 / **64.75** | 72.8 |
| fork | 445.4 / 45.70 | 350.8 / 54.43 | 71.9 |
| llaminar | 415.0 / **46.53** | 416.0 / 59.93 | 70.5 |

## Results -- Q8_0-f32gdn

| engine | plain | MTP | accept |
|---|---|---|---|
| stock | 409.5 / 35.97 | 326.3 / 50.64 | 71.4 |
| CV | **571.9** / 35.58 | **511.5** / **58.06** | 72.4 |
| fork | 406.7 / 35.82 | 325.7 / 50.46 | 71.4 |
| llaminar | BLOCKED (GDN family + nextn) | | |

## MTP gains per engine (vs own plain)

| engine | Q4 pp | Q4 tg | Q8 pp | Q8 tg |
|---|---|---|---|---|
| stock | -16.4% | +25.5% | -20.3% | +40.8% |
| CV | **-29.8%** | **+49.2%** | -10.6% | **+63.2%** |
| fork | -21.2% | +19.1% | -19.9% | +40.9% |
| llaminar | **+0.2%** | +28.8% | -- | -- |

## Verdicts

1. **Plain prefill ranking (Q4): CV 534 > fork 445 > llaminar 415 > stock 367.**
 CV's pp lead over our fork is +20% at TP2 (was +43.6% vs stock-lineage in the
 Oct-9 h2h; our W4A4 stack closed half of it). The residual is CV's BAR
 peer-write allreduce + scheduler work -- still the top infra port target.
2. **llaminar prefill sits BETWEEN stock and fork** (+13% over stock, -6.7% vs
 fork) despite FP32 activations -- respectable for an alpha-stage engine with
 its own kernels, and its planner/TP works on 2x MI50 with its patched RCCL.
3. **MTP decode ranking: CV 64.75 > llaminar 59.93 > fork 54.43 > stock 53.81.**
 CV's MTP suite delivers +49% decode over its own plain -- by far the largest
 MTP gain of the four. Our fork gains only +19%: the gap is per-step overhead
 (graph reuse, catch-up merge, ckpt-last-verify), re-confirming the port
 priority at TP2 with same-session data.
4. **MTP prefill-tax structure differs fundamentally**: llaminar pays ~0%
 (device-owned execution), CV pays the most (-30%) yet still wins MTP prefill
 outright at Q8 (+56.8% over stock-MTP). Different architectures, different
 trade-off curves -- not a single-knob comparison.
5. **Q8_0-f32gdn**: CV dominates every axis; fork ~ stock (W4A4 is Q4-keyed);
 llaminar cannot load it (GDN weight-family + missing nextn in re-quantable
 sources -- surgical file rewrite queued if Q8-on-llaminar ever matters).
6. **Q8 MTP decode gains are the largest of the session** (+41% stock, +63% CV,
 +41% fork over own plain): speculative decode scales with the heavier Q8
 weights -- verify cost dominates more, drafts save more.

## Caveats
- Cross-engine protocol boundary: llaminar measured via its own benchmark harness
 (their warmup + 3 iterations, built-in prompt semantics); llama.cpp family via
 server /completion A/B. Prompt bytes: shared prompt file for all legs.
- v5/v6 numbers used the blanket Q8_0 file; from v7 on, Q8 = f32-GDN file
 (llaminar requirement). Cross-version Q8 deltas are directional only.
- llaminar built with spill-guard pass-through + -O1 heals on 3 gensrc TUs
 (uncertified kernels; perf impact on those host stubs negligible).
- ROCm0 excluded (external agent); card pair = ROCm1+ROCm2 for all legs.

Raw: mi50:rig/cv-h2h-v7/ (h2h-v8.log, 56 JSONs this run, v7d llaminar JSONs,
server logs). Drivers: h2h-driver-v8.sh (+v7d). Aggregator: aggregate-v8.py.
Baselines: RESULTS-v5/v6/v7.md in workspace/cv-headtohead/.
