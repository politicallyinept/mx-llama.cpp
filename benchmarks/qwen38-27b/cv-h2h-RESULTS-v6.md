# v6 -- the same matrix with MTP enabled (Qwen3.8-27B, TP2, 2026-10-19)

Same builds as v5 (CV / stock / fork marker-verified), same protocol,
same 6 arms x 2 rounds -- every arm now runs MTP: shared draft mtp-Qwen3.8-27B-Q4_0.gguf (the only
draft on disk, used by all engines incl. CV's own guide), `--spec-type draft-mtp --spec-draft-n-max 2`.
CV additionally gets `--spec-draft-ngl 999` + `LLAMA_CKPT_LAST_VERIFY=1` (their guide's spec set);
mx arms use the Oct-9-proven flag set.

## Results (t/s; means; r1/r2 spread <=0.9% except where noted)

| arm | pp | tg | accept % |
|---|---|---|---|
| cv-mtp-q4 | 473.8 (clean r2/r3/r4; r1 pp leg discarded -- known CV transient, 180/480 bimodal, re-run v6b reproduced 476.8/464.3) | **66.09** | 72.6-73.0 |
| stock-mtp-q4 | 306.0 | 55.20 | 73.4 |
| fork-mtp-q4 | **355.3** | 54.56 | 71.9 |
| cv-mtp-q8 | **509.8** | **60.06** | 72.5-73.0 |
| stock-mtp-q8 | 400.1 | 52.31 | 73.1 |
| fork-mtp-q8 | 397.9 | 52.06 | 73.1 |

## MTP gain vs v5 plain (same engine + quant)

| arm | pp | tg |
|---|---|---|
| cv-mtp-q4 | (contaminated r1; clean legs ~-10%) | **+43.2%** vs 45.07 |
| cv-mtp-q8 | -9.7% | **+68.2%** vs 35.71 |
| stock-mtp-q4 | -16.6% | +28.9% |
| stock-mtp-q8 | -22.0% | +42.6% |
| fork-mtp-q4 | -19.8% | +19.6% |
| fork-mtp-q8 | -21.6% | +42.8% |

## Findings

1. **CV's MTP decode lead replicates on fresh builds: +21.1% tg (q4) / +15.4% (q8) over our fork, at
 EQUAL acceptance (71.9-73.4% everywhere).** Equal acceptance + tg gap = pure per-step overhead
 delta -- the Oct-9 attribution, now confirmed against today's tree. Cross-check: Oct-9 measured
 cv-mtp 65.3 vs fk-mtp 54.2; today 66.09 vs 54.56. Reproduction solid.
2. **The MTP prefill tax gap also replicates, bigger than Oct-9**: CV pays -9.7% pp for the resident
 drafter (q8) vs our -21.6/-22.0% (stock/fork). Their graph_cache2 + ckpt-last-verify + catch-up-merge
 suite is doing exactly what the porting map said. On q4: CV ~-10% vs fork -19.8%.
3. **Under MTP the fork's decode advantage disappears BY DESIGN**: the MIN_TOKENS=128 floor sends
 verify+decode down the bit-stock path, so fork-mtp ~ stock-mtp on tg (54.56 vs 55.20 = -1.2%,
 within accept jitter). The gate's value survives only in the real prompt prefill: fork-mtp-q4 pp
 355.3 vs stock-mtp 306.0 = **+16.1%**, and fork-mtp-q8 pp parity as expected (gate inert on Q8_0).
 The 49.9 t/s "flagship" from 63 is a no-regret composition, not an MTP gain -- v6 quantifies that
 honestly: gate+MTP = stock+MTP decode + gated prefill.
4. **Fork acceptance on q4 is 71.9 vs stock 73.4** -- consistent with the 63 near-tie-flip mechanism
 (gated prefill shifts logits slightly -> different draft accept pattern downstream of flips); on q8
 (gate inert) acceptance is identical (73.1 both). Small but real; worth remembering when comparing
 accept-normalized numbers.
5. **Port priority confirmed with today's numbers**: the MTP overhead suite is worth ~+21% tg on q4 /
 +15% on q8 to whichever engine ships it, engine-agnostic, and it would stack on the fork's prefill
 win (fork has the best MTP pp of the mx pair AND the best plain tg). It remains the single
 highest-value un-ported lever. BAR peer-write AR remains the pp-lever for the remaining CV prefill
 lead (v5: -16.5% q4 residual).

## Watch items

- cv-mtp-q4 r1 transient recurred (pp bimodal 180/480, tg slightly high) -- same signature as Oct-9 v4
 Part-A. Part-only re-run (v6b) produced clean legs. Pattern: CV + MTP + first-round-after-launch.
 If a future session needs CV MTP numbers: warm up one throwaway round first.
- No segfaults in any v6/v6b server log (the v5 CV-q8 teardown segfault did not recur).

Raw: mi50:rig/cv-h2h-v6/ (h2h-v6.log, h2h-v6b.log, 24+4 JSONs, server logs). Drivers:
h2h-driver-v6.sh / h2h-driver-v6b.sh. Aggregator: aggregate-v6.py. Plain-decode baseline: RESULTS-v5.md.
