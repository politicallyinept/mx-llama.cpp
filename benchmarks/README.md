# benchmarks/ - in-repo benchmark result summaries

Summaries of benchmark campaigns measured on the project's 3x MI50 (gfx906)
measurement rig, copied out of the rig's campaign record directories. Only
campaigns run on the project's benchmark model set are included here:

- **Qwen3.8-27B** (dense; Q4_0 / Q8_0 / UD-Q4_K_XL quants)
- **Qwen3.5-35B-A3B** (MoE; Q4_0 / Q8_0 / UD-Q4_K_XL quants)

Campaigns run on other models (including the large dense production model
referenced throughout the older parts of `docs/BENCH.md`) are excluded from
this folder; their raw logs remain on the measurement rig. Raw per-run logs,
server logs, and multi-GB artifacts are likewise not mirrored here - these are
the small text/json/csv summaries the docs cite.

Local build paths and internal revision ids present in the source records were
stripped or replaced with `(internal ref)` during the copy.

## Index

### qwen38-27b/ - Qwen3.8-27B dense

| file / dir | campaign | contents |
|---|---|---|
| `full-bench-20261026/` | full fork-vs-stock bench, 2026-10-26 | per-cell perf json + audit txt for stock / fork / fork+W4A4 / CV-comparison legs, Q4_0 and Q8_0 (pp840/pp512/tg256, cold/warm) |
| `dot84-seal-SUMMARY.md` | DOT8=4 seal attempt | greedy-seal + KLD + PPL summary; DOT8=4 REJECTED, DOT8=3 kept |
| `cv-h2h-RESULTS*.md` | CV head-to-head, v1-v8 | engine-vs-engine prefill/decode head-to-head on the 27B across successive build revisions |

### qwen35-35b-a3b/ - Qwen3.5-35B-A3B MoE

| file / dir | campaign | contents |
|---|---|---|
| `moe-baseline-RESULTS.md`, `moe-baseline-AGGREGATE.txt` | MoE 3-build baseline, 2026-10-25 | stock / fork / CV-comparison baseline table, TP2, Q4_0/Q8_0/UD |
| `moe-h2h-results.jsonl` | MoE cross-engine head-to-head | per-cell medians (pp840/pp512/tg128), Q4_0 + UD |
| `moe-levers-sec90.md`, `moe-levers-sec91.md` | MoE lever validation (BENCH sections 90/91) | W4A4 gate re-validation on the 35B; gcn_kq_moe launcher + Q3_K admission |
| `tier3-decodev2-report.md` | tier-3 decode v2 | tiled-repack decode ladder report |
| `ladder-report.md` | correctness ladder | KL/PPL ladder across gate configs |
| `decode-diag-report.md` | decode diagnostics | decode-path regression diagnosis (Q8_0 repack mmv finding) |
| `q81cache-report.md` | Q8_1 cache analysis | Q8_0/Q8_1 KV-cache behaviour report |
| `she-fusion-tp2-report.md` | SHE-fusion adoption evidence | fused shared-expert decode path TP2 confirmation + seal suite |
| `seals/seal-q40-tp{1,2}.json`, `seals/seal-ud-tp{1,2}.json` | greedy-seal baselines | sealed greedy-decode output hashes (TP1/TP2, Q4_0/UD) used by the adoption gates |
| `convfix-freeform/` | conv-fix ladder freeform arms | per-arm generation json (gate on/off) |

### shared/

| file | campaign | contents |
|---|---|---|
| `p03-h2h-summary.md` | P03 head-to-head | cross-engine summary covering both benchmark models |

### top level

| file | campaign | contents |
|---|---|---|
| `defaults-vs-env.md` | gate-truth verification, 2026-10-31 | per-gate source defaults vs the production env bundle; dual seal matrices (env-profile vs true-defaults) proving the published results describe the env bundle and the tiled repack path is bit-identical to the default path |
