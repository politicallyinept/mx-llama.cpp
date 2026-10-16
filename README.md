# W4A4: Weight-4/Activation-4 GEMM with dp8a dot for gfx906-class (Vega 20, 32 GB HBM2) AMD accelerators

> ## **AI-GENERATED PROJECT NOTICE**
>
> **This README and all coding was done by several AI agents. I looked over all the code but to be frank I have no idea what I'm looking at. This was a project I did for fun — hopefully someone finds it helpful.**

This branch contains the complete W4A4 weight-quantized GEMM path developed for gfx906-class
(32 GB HBM2) accelerators, including the activation-side u4 path, K-split scheduling, the
correctness-first validation lineage, and the full measured performance/correctness record.

**Everything ships env-gated and default-off.** No model files, no quantizer changes to
downstream tools, no behavior change without the operator opting in per model or globally.

## What this is

Three coordinated pieces, layered so each is independently useful:

0. **Repack** (default-on, no gate): Q4_0 weights are repacked into a GEMM-friendly
   layout as the base optimization layer — its own measured win, roughly comparable in
   magnitude to the gated path's, credited separately below so readers don't fold it
   into the opt-in gate's number.
1. **W4A4 dot-product GEMM** (`GGML_CUDA_W4A4_DOT8=1`): builds on the repack and adds
   a dp8a-friendly weight layout (32 nibbles per uint4) computing the dot product of 4-bit weights
   against 8-bit activations in exact integer arithmetic, with a sealed algebraic epilogue
   converting the integer dot into the fp32 accumulator.
2. **Per-16 variant** (`GGML_CUDA_W4A4_DOT8=3`): same scheme at 16-value granularity.
3. **K-split** (`GGML_RP_KSPLIT`): splits the K dimension of the repacked w32 GEMM
   to reduce latency at narrow batch widths, on the stock path only
   (self-guarding: excluded from the gate-1 dp8a path by static_assert — gate-1 and ksplit cannot combine; gate-3 + ksplit IS a supported pairing).

## Turning it on

One switch enables the recommended configuration; everything else is optional tuning.

```bash
# The switch: enable the W4A4 fast path (repack + gated dot-GEMM + auto width floor)
export GGML_CUDA_W4A4_DOT8=1
```

That single variable turns on: Q4_0 weight repacking admission, the dp8a dot GEMM, and —
when `GGML_CUDA_W4A4_MIN_TOKENS` is set — automatic fallback to the stock path below the
floor width. A complete server invocation:

```bash
GGML_CUDA_W4A4_DOT8=1 GGML_CUDA_W4A4_MIN_TOKENS=128 \
  llama-server -m model-Q4_0.gguf -ngl 99 -fa on -c 8192
```

This one config is correct for **both** batch-prefill serving and MTP/speculative-decode
workloads: prefill runs the fast path (widths ≥ 128), MTP verify passes and decode run
the stock path automatically (widths below the floor). No per-model branching needed.

### Recommended configs

| workload | env |
|---|---|
| **Default (recommended)** | `GGML_CUDA_W4A4_DOT8=1` + `GGML_CUDA_W4A4_MIN_TOKENS=128` |
| Stock-path narrow-width boost (no W4A4) | `GGML_RP_KSPLIT=1` |
| Conservative / testing | no env at all — everything is default-off, byte-identical to stock |

### Optional tuning variables

| variable | default | effect | when to touch it |
|---|---|---|---|
| `GGML_CUDA_W4A4_MIN_TOKENS` | unset (floor off) | gate falls back to stock below this batch width (dense AND MoE dispatch) | set `128` for MTP coexistence or if narrow-batch/MoE-decode throughput dips |
| `GGML_RP_KSPLIT` | unset | K-splits the repacked w32 GEMM (1 = auto slice count, 2/4/8 = explicit) | only useful on the stock path at 16 ≤ width ≤ 64; inert under the gate |
| `GGML_CUDA_W4A4_DOT8=3` | — | per-16 activation variant | alternative accuracy/width profile; no narrow-width cliff, smaller prefill win |
| `GGML_CUDA_REPACK_Q4_0` | unset | forces repack admission for Q4_0 regardless of dot8 mode | diagnostic — normally leave unset |
| `GGML_RP_GENERIC_MMV` | unset | forces the generic (non-repacked) MMV path | troubleshooting — compare against to isolate the repack path |
| `GGML_W4A4_ONLY` | unset | restricts execution to W4A4 ops only (skips everything else) | debugging graph coverage, not for serving |

Gate values are **modes, not additive flags**: `DOT8=1` (per-32), `=3` (per-16), `=4`
(= 1 + Q4_1 u4), `=5` (= 4 + Q4_K u4, NO-GO quality — keep off). Set exactly one.

### Troubleshooting

- **"Did it even turn on?"** — every gate prints a one-line `WARN` marker to the log when
  active (e.g. `CUDA w4a4: GGML_CUDA_W4A4_DOT8=1 gate ACTIVE (...)`, the floor's
  `... floor ACTIVE (u4act falls back to stock below this batch width)`). No marker in
  the log = the path did not engage; check the variable spelling and that the model file
  is Q4_0 (the gate is Q4_0-family; Q4_K files stay stock — see the Q4_K note above).
- **Output differs from stock**: expected and bounded — the gated prefill shifts logits
  slightly (measured KLD envelope above; decode/verify run bit-stock under the floor).
  If outputs look *degraded* (loops, nonsense) rather than merely different: unset
  `GGML_CUDA_W4A4_DOT8` to confirm the gate is the cause, then report it — that is a bug
  (the envelope says it shouldn't happen).
- **Throughput regression at some width**: check the floor — if the width in question is
  below `MIN_TOKENS` you're seeing stock (correct); if above, try `DOT8=3` (the per-16
  variant has no narrow-width cliff).
- **Multi-GPU + `llama-bench`**: this fork's bench `-dev` takes comma = device list
  (slash = benchmark cross-product) — the opposite of upstream; see the note in
  Reproduce below.
- **RAM at very long context**: `llama-perplexity` needs ≥ 2× context tokens of corpus
  and builds n_ctx×n_vocab logits in host RAM (a 150k rung needs ~91 GB) — a tooling
  limit, not the model's.

## Design philosophy

- **Correctness-first**: a negative result is a result. Every claimed number below comes
  from a run recorded in `docs/BENCH.md` with reproducible commands.
- **Env-gated, never hard-defaulted**: operators opt in per model or fleet-wide via
  environment variables; gate-off behavior is byte-identical to stock.
- **Sealed algebra**: the epilogue's arithmetic is verified against f64 ground truth
  (bit-exact at double precision) with negative controls that must fail.

## What we measured (performance)

All numbers on ROCm MI50 (gfx906), llama-bench / llama-perplexity, base-model Q4_0 files,
r≥3 unless noted, interleaved arms, VRAM returned to baseline (10.87 MB/GPU) between legs.
Full tables and repro commands: `docs/BENCH.md` §40–§61.

### Prefill (prompt processing)

| model | stock | W4A4 (gate1) | Δ | PPL cost |
|---|---:|---:|---:|---:|
| Qwen3.5-9B (Q4_0) | 1131 | 1462 | **+29.3%** | +2.0% |
| Qwen3.8-27B (Q4_0) | 284.9 | 373.6 | **+31.1%** | +2.0% |
| Qwen3-30B-A3B (MoE, Q4_0) | 1269 | 1378 | +8.6% | +3.2% |
| Llama-3.1-8B (Q4_0) | — | +30.9% (pp512) | — | +20.3% (pp2048 PPL) |
| gemma-4-12B (Q4_0) | — | +39.5% (pp512) | — | +15% PPL — gemma's error is NOT mid-band-FFN-localized; outlier, do not generalize from it |

The MoE model measured gains ~a third of dense on its expert-heavy file (+8.2–8.6%);
this scales with expert-to-attention GEMM ratio and should not be generalized from one point.
The width floor covers MoE too (same `MIN_TOKENS` knob): bare gate on a small-expert MoE
measured a −13.9% decode penalty that the floor eliminates (Qwen3-30B-A3B:
gate+floor = pp +5.0% / tg −0.7%). In contrast, the 27B's merged-path bench reading was +34.4% (bench) — batch/prefill overlap accounting differs between bench and server; see the decode section.
per-16 trades prefill width for accuracy and narrow-width stability (§54, §61).

### Narrow widths — where it loses

`W4A4=1` inverts across batch width: at pp16–64, gate1 is **10–16% slower than stock**
(r=5, 27B: pp16 69.3 vs 78.5, pp32 131.7 vs 151.9; the width crossover sweep measured
−14.9%/−16.3% on its run). This matters for speculative-decode
verify passes, which run in exactly this window — but the fix is built in: a **width
floor** (`GGML_CUDA_W4A4_MIN_TOKENS`, e.g. 128) makes the gate auto-fall-back to the
stock path below that batch width, so MTP verify passes and decode run bit-stock while
prefill keeps the full gate win. Measured on 27B with MTP: gate1+floor+MTP 47.3 t/s vs
stock+MTP 49.05 (−3.5%, accept-jitter residual; accept-length distributions identical),
native-MTP+gate+floor 49.9 t/s **with** +31% prefill — the previously impossible
combination now works. Output consistency verified: temp-0 MTP deterministic (20/20
reps token-identical), below-floor outputs 100/100 token-identical to stock, above-floor
divergences are single near-tie token flips (0.11 nats), zero degenerate outputs in
300+ generations, greedy primes byte-exact with the floor on. Un-floored reference:
stock+MTP 48.7 vs gate1+MTP 29.8 t/s — don't run the gate bare with MTP.

### K-split — narrow-width win on the stock path

`GGML_RP_KSPLIT=1` on the **stock** path (no W4A4 gate): +10.6% pp16 / +7.2% pp32 / +1.2%
pp64 on 27B, wide-batch parity, **PPL bit-identical** (per-chunk arrays byte-equal, 6.8534).
Proven inert under DOT8 gates (byte-identical), so it is safe to enable globally:
only models without the gate see it.

### Decode (token generation)

tg128 within noise on dense models (stock-path ksplit measured +1–4% at model level in early sections); the gate-1+ksplit combination is NOT buildable (static_assert, see above) and its early +3.7% reading was refuted in the program's own records. Decode is not where this path pays.

## What we measured (correctness)

- **Seals**: epilogue algebra bit-exact vs f64 ground truth (0.0 max rel err, 20k trials,
  adversarial patterns); staged fp32 form ≤ 9.7e-6; isolation form ≤ 1.05e-5. Negative
  controls (sign-flipped-M 15813/15813, swapped sc/m) fail as required. Harness validated
  by per-regime declared-spec input checking + validator self-test (corrupt-everything
  control; deliberate degeneracies declared).
- **Greedy seals**: greedy prime sequences byte-identical vs stock across primes 2–29.
- **KLD envelope** (9B Q4_0, wikitext-2-raw, c512 ch128, vs f16 base):
  stock recipe (no imatrix) 0.0684; +imatrix + map 0.0412; W4A4 gate1 0.1222 (0.0764 with
  calibration on our map); SKIP-band variants 0.0979. The u4 activation penalty is
  ≈ one quant step: calibrated, it sits ~33% of the way across the Q4→Q3 gap; uncalibrated
  ~68%. Seals prove the kernel is bit-exact — all accuracy cost is encode-channel
  (quantizer design), not implementation.
- **Anchors EXACT across refactors**: gates 1/2/3 reproduce to zero ULP on every
  re-baseline (PPL 8.7630 / KLD 0.122150 / 0.097917 / 7.2774, gate4 PPL bit-identical).

## What we tried and rejected (so you don't have to)

- **Q4_K u4** (`GGML_CUDA_W4A4_DOT8=5`): prefill +12.5% (below the +25% adopt bar),
  KLD +0.0173 (over the 0.01 relative envelope), tg128 −60% (per-call epilogue cost
  doesn't amortize at single-token). NO-GO measured end-to-end; code exists on a branch,
  default-off, inert.
- **Per-block least-squares scale**: KLD +13–25% vs the shipped imatrix+map pipeline
  (which is already weighted-LS + discrete clamping-pattern search). Negative; branch
  `exp/q40-ls-scale` is env-gated/inert.
- **Hadamard rotation (H32)**: simulated gains didn't survive the live path — recorded
  as the program's cautionary precedent for activation-side levers.
- **SmoothQuant-style offline channel fold**: *positive*, but size-dependent — recovers
  67% of the u4 activation penalty at 9B, ~32% at 27B (weight-side gain was a 9B-file
  artifact; +1.07% at 27B, nothing). Filed as a small-model lever and the u4 gate's
  accuracy partner; held for adoption (adds a fold step to the encode recipe).
- **SKIP layer bands**: layer selection does not transfer across models (measured closure
  collapses ~3× on a different depth, ordering inverts). Per-model sweep required; band
  is not shipped as portable.

## Cost/benefit summary

**Benefits** (for batch-shaped inference workloads):
- +29–40% prefill on dense Q4_0 models at SHORT-to-MEDIUM context; at long context
  (131k) the gain narrows to ~+8–10% (KV-bound). The headline is a short-context number,
  not a universal one — bound claims to the regime they were measured in.
- +10.6%/+7.2% at pp16/pp32 from K-split on the stock path, bit-identical output
- Zero VRAM overhead in steady state; per-model env-gated adoption; greedy-identical

**Costs**:
- 10–13% prefill loss at pp16/32 under the W4A4 gate (pp64's leg has 22% σ — only pp16/32
  support the claim; the spec-decode window — the reason gate and MTP are mutually
  exclusive per model)
- ~2% PPL (dense Q4_0 calibrated), ~3% (MoE); gemma-4-12B measured +15% and is the
  outlier — its error is not mid-band-localized, do not generalize from it
- The u4 activation path costs ≈ one quant step of additional divergence — ~42% of the
  measured Q4→Q3 gap calibrated, ~65% uncalibrated (denominator: the 9B's own
  8→4→3 pure-type step 0.083232) — this is encode-channel, not the GEMM

**Who should use this**: batch-heavy serving on gfx906-class cards where prefill
throughput dominates and PPL tolerance is ≥2%. **Who should not**: MTP/speculative-decode
workloads (use stock + MTP; measured best), MoE-heavy decode workloads (gains are ~⅓),
or deployments where the narrow-width cliff matters.

## Provenance

All benchmarks used **base models** (upstream or bartowski-style uncalibrated
Q4_0/quantized-from-bf16), never fine-tuned for content: Qwen3.5-9B-Base,
Qwen3.8-27B-Base (Unsloth), Qwen3-30B-A3B-Base, Llama-3.1-8B-Base (Meta),
gemma-4-12B-Base. One tooling-validation arm used Mistral-7B-Instruct-v0.3 (Q4_K_M,
single-GPU single-request arm only) — flagged here as an instruct (finetune) so it is
never cited for accuracy claims. GGUF metadata carries the base-model proof
(`general.base_model` fields). Calibration uses the files' own imatrix statistics where
stated; uncalibrated arms are labeled as such.

## Branch structure

- `master` — validated lineage, gates OFF (this branch)
- `q41-u4-probe` — Q4_1/Q4_K u4 work (Q4_1 GO'd and merged; Q4_K NO-GO'd, kept for the record)
- `pr0-w4a4-correctness`, `pr1-q40-repack`, `pr2-ksplit` — per-PR slices of the above
- `fix/llama-bench-dev-comma` — tooling fix (bench `-dev` delimiter semantics)
- `exp/q40-ls-scale` — negative result, kept env-gated for reproducibility

## Reproduce

Every number cites a BENCH section with the exact command; the harness is
`llama-bench` + `llama-perplexity --kl-divergence` against f16 logits bases built on the
same corpus/chunk protocol. Note: this fork's llama-bench `-dev` fix changes delimiter
semantics (comma = device list for split, slash = benchmark cross-product) vs upstream's
inverted convention — commands in these docs use the fork convention. Raw logs retained under `/root/bench-results/` and
`/root/ab-gates/` on the measurement box.
