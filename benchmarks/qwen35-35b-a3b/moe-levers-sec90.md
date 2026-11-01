## 90. MoE W4A4 gate re-VALIDATED on Qwen3.5-35B-A3B: AMEND the gate-OFF verdict to config-conditional gate-ON (2026-10-24, build-tp2 @ , rig/moe-levers/)

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

