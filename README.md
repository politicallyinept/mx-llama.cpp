# W4A4: Weight-4/Activation-4 GEMM with dp8a dot for gfx906-class (Vega 20, 32 GB HBM2) AMD accelerators

> ## **AI-GENERATED PROJECT NOTICE**
>
> **This README and all coding was done by several AI agents. I looked over all the code but to be frank I have no idea what I'm looking at. This was a project I did for fun - hopefully someone finds it helpful.**

This branch contains the complete W4A4 weight-quantized GEMM path developed for gfx906-class
(32 GB HBM2) accelerators, including the activation-side u4 path, K-split scheduling, the
correctness-first validation lineage, and the full measured performance/correctness record.

**Default behavior:** with `GGML_CUDA_W4A4_PROFILE` unset the int4-path knobs resolve to
the sealed `speed` preset (DOT8=3 per-16, MIN_TOKENS=128, KSPLIT=1, SUMW=1). The
tiled/W4A4 repack paths engage only for tensors admitted through the repack admission
gate (`GGML_CUDA_REPACK_Q4_0`, default OFF in source); the production profile sets these
envs explicitly. Set `GGML_CUDA_W4A4_PROFILE=off` (or `none`), or the kill-switch
`GGML_CUDA_W4A4_OFF=1`, for byte-identical stock behavior. The debug gates remain
opt-in. Three small non-int4 items ride along: an opt-in
`MXFP4` quantize ftype, a TP-3 entry in the custom-AllReduce auto-select list (only
relevant when the fork's custom AllReduce is engaged), and a DSV4 KV-cache flag that
reports checkpoint restore as unsupported (conservative: the compressed state cannot be
restored losslessly).

## Attribution / Provenance

- **Base project:** this tree is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp)
  (ggml-org).
- **Lineage:** this branch descends from the mx performance fork
  ([mxxm-t/mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp), itself a fork of
  ggml-org/llama.cpp); the fork's own records reference that lineage (`docs/BENCH.md`
  cites "mxxm-t master" as the upstream base).
- **CV fork (external source):** part of the kernel work on this branch derives from
  Castagna Veloce, an external llama.cpp fork by benpeterson40
  ([benpeterson40/castagna-veloce](https://github.com/benpeterson40/castagna-veloce)).
  Ported kernels are marked in the source comments (`CV (castagna-veloce) port`, pinned
  to CV commit `3483d46`); the Provenance column of the gate table below notes which
  gates are CV ports.
- **CV-port / in-house additions:** on top of that lineage this branch carries GPU kernel
  work for gfx906-class (MI50) accelerators, ported from the CV fork and developed
  in-house. Headline items:
  - tiled Q4_0 repack GEMM (gfx906 `gcn_u4_gemm`, 128x64 tiles, LDS-staged), engaged for
    Q4_0 tensors admitted to the repack path (tier3-adopt), sealed bit-identical to the
    generic MMQ loop at the validated widths (batch sizes 2-15 are not covered by the
    seal);
  - `gcn_u4_gemm` activation nibble-pairing fix (canonical Q4_0 order);
  - `attn_qk_fused` call-site restore: the fused decode attention q/k chain is re-engaged
    in the CUDA fusion matcher.

## Environment gates / feature flags

Env gates active in this tree (each verified in source; provenance noted). The W4A4
int4-path knobs (`GGML_CUDA_W4A4_PROFILE`, `GGML_CUDA_W4A4_DOT8`, `GGML_CUDA_REPACK_Q4_0`,
`GGML_RP_KSPLIT`, ...) are documented under "Turning it on" below and are not repeated here.

| Env var | Default | Effect | Provenance |
|---|---|---|---|
| `GGML_CUDA_REPACK_Q4_0_TILED` | ON (`=0` off) | repacked Q4_0 GEMMs run the tiled gfx906 `gcn_u4_gemm` (128x64 tiles, LDS-staged); engages only for Q4_0 admitted via `GGML_CUDA_REPACK_Q4_0` (default OFF); sealed bit-identical to the generic MMQ loop at validated widths (batch 2-15 not sealed) | tier3-adopt |
| `GGML_CUDA_ATTN_QK_FUSED` | ON (`=0` off) | fuses the decode attention q/k chain (norm + rope + K/V cache rows) into one kernel | CV attn_qk_fused port; call site restored |
| `GGML_CUDA_GDN_STATE_ROWS_OK` | ON (`=0` off) | GDN decode reads its recurrent-state row directly via the GET_ROWS ids (gather node skipped) | CV gdn_state_rows port |
| `GGML_CUDA_GDN_AB_DEC` | OFF (`=1` on) | fuses the GDN decode gate chain (alpha softplus path + beta sigmoid) into one kernel | CV hc-dec port (default ON in CV) |
| `GGML_CUDA_DISABLE_FUSION` | OFF | kill switch: disables the whole CUDA fusion-matcher pass | upstream llama.cpp |
| `GGML_CUDA_REPACK_KQUANT` | ON (`=0` off) | repacks Q4_K/Q5_K/Q6_K into the GEMM-friendly layout; `=0` keeps canonical layout (stock path) for A/B | gcn-repro A/B lane |
| `GGML_CUDA_REPACK_KQUANT_MOE_FUSION` | ON (`=0` off) | admits K-quant/IQ4_NL expert tensors to the repacked MoE up/gate mat-vec fusion | repack MoE-fusion lane |
| `GGML_CUDA_REPACK_Q8_0` | MoE-conditional (`=1` force on, `=0` force off) | admits Q8_0 weights to the gfx906 repack path. Default follows the model shape: ON only when the loaded model has 3D expert tensors (MoE: +1.3..1.5% decode, bit-identical to stock), OFF for dense models (dense repack measured -1.6% decode / -8.4% prefill at +24 GB VRAM). `=1` forces admission on for dense models too; `=0` forces it off even for MoE | q8_0-repack E4 lane |
| `GGML_CUDA_REPACK_Q8_0_RESTYLE` | ON (`=0` off) | repacked Q8_0 mat-vec/GEMM reproduce the stock accumulation order and granularity on the repacked layout (bit-identical greedy output at a decode win); `=0` restores the pre-RESTYLE kernels | q8_0-repack E4 lane |
| `GGML_CUDA_REPACK_Q8_0_YIELD` | ON (`=0` off) | dense Q8_0 prefill GEMMs admitted by the gcn_kq gate yield to the gcn_kq v4 kernel via the canonical copy kept next to the repacked record; inert unless Q8_0 is admitted | q8_0-repack E4 lane |
| `GGML_CUDA_REPACK_Q8_0_STOCK_ACC` | OFF (`=1` on) | numerics-neutral kill switch: pins Q8_0 matmuls to stock accumulation order (Q8_0 stays canonical, repack suppressed) even when admitted; the A/B fallback from before the RESTYLE kernels | anchor3 consolidation |
| `GGML_Q8_YIELD_TRACE` | OFF (`=1` on) | debug trace of the yield branch decisions on stderr (capped); never affects output | q8_0-repack E4 lane |
| `GGML_CUDA_DFLASH_STAGE_FUSE` | ON (`=0` off) | dflash width-8 GDN ring staging fusion: conv-state ring scatter and recurrent-state snapshot copies fold into the direct-scatter kernels on spec-verify graphs only; byte-identical output, +5.1% verify-width decode | dflash fix lane |
| `GGML_CUDA_MOE_VEC` | OFF (`=1` on) | row-lane MoE decode kernel family (decode-shape admission) | CV port |
| `GGML_CUDA_MOE_FUSE` | OFF (`=1` on) | MoE gate/up + SwiGLU prefill fusion on the expert-grouped GCN v4 GEMM | CV-port prefill fusion |
| `GGML_TP_AR_MTP_SAFE` | OFF (`=1`/`=2`) | disables the custom TP all-reduce once a tensor-width change is seen (MTP/spec-decode safety) | fork TP work |
| `GGML_GDN_COLTHREAD` | OFF (`=1` on) | column-thread GDN kernel path; lanes/column via `GGML_GDN_LPC` (8 on GCN, 4 elsewhere) | CV port |
| `GGML_CUDA_GCN_HC_UP_MIX` | ON (`=0` off) | fuses prefill HC up-projection -> reshape -> gated DSV4_HC_PRE on GCN | CV port |
| `GGML_CUDA_GCN_GEMM` | ON (`=0` off) | GCN GEMM path admission switch | CV port |
| `GGML_CUDA_SHE_FUSE` | ON (`=0` off) | fused shared-expert / dense-f32 decode path | CV port (default flipped ON at adoption) |
| `GGML_CUDA_FA_DENSE_GCN` | ON (`=0` off) | GCN prefill attention with D = 256, f32 accumulation | CV port |
| `GGML_CUDA_GCN_KQ_MV` | ON (`=0` off) | GCN q4_K/q5_K/q6_K/iq4_xs mat-vec for 1-4 token decode | CV port (default flipped ON after KLD/PPL audit) |
| `GGML_CUDA_CONCAT_SMALL` | OFF (`=1` on) | flat-grid short-row concat kernel | CV port |
| `GGML_CUDA_ASYNC_CKPT` | ON (`=0` off) | async recurrent-state checkpoint capture in the server (sync deferred to next synchronize) | fork server work (staging robustness under further investigation) |
| `GGML_CUDA_GDN_CONV_SCATTER` | ON (`=0` off) | GDN conv-state ring written by direct kernel-side scatter instead of a gather+copy | mtp-pipe (default ON) |
| `LLAMA_MTP_FAST_SAMPLE` | ON (`=0` off) | MTP draft skips backend top-k chain construction and samples directly | mtp-pipe (default ON) |

### Speculative-decode serving guidance (draft depth)

`--spec-draft-n-max` should follow concurrency: at `np <= 2` serve `n_max 2`; at
`np >= 3` disable speculative decode or drop to `n_max 1`; for mixed-concurrency
endpoints serve `n_max 1`. Wider drafts only pay while per-request GPU slack
exists - at higher batch counts the extra verify width costs more than the
accepted tokens return.

## What this is

Three coordinated pieces, layered so each is independently useful:

0. **Repack** (`GGML_CUDA_REPACK_Q4_0=1`, default-off): Q4_0 weights are repacked into a GEMM-friendly
   layout as the base optimization layer - its own measured win, roughly comparable in
   magnitude to the gated path's, credited separately below so readers don't fold it
   into the opt-in gate's number.
1. **W4A4 dot-product GEMM** (`GGML_CUDA_W4A4_DOT8=1`): builds on the repack and adds
   a dp8a-friendly weight layout (32 nibbles per uint4) computing the dot product of 4-bit weights
   against 8-bit activations in exact integer arithmetic, with a sealed algebraic epilogue
   converting the integer dot into the fp32 accumulator.
2. **Per-16 variant** (`GGML_CUDA_W4A4_DOT8=3`): same scheme at 16-value granularity.
3. **K-split** (`GGML_RP_KSPLIT`): splits the K dimension of the repacked w32 GEMM
   to reduce latency at narrow batch widths, on the stock path only
   (self-guarding: excluded from the gate-1 dp8a path by static_assert - gate-1 and ksplit cannot combine; gate-3 + ksplit IS a supported pairing).

## Turning it on

The **profile** preset resolves the four int4-path knobs; an unset profile resolves to
`speed` (the P5 default flip). Note that the tiled/W4A4 repack paths engage only for
tensors admitted through the repack admission gate (`GGML_CUDA_REPACK_Q4_0`, default OFF
in source); the production profile sets these envs explicitly. Everything else is
optional per-knob tuning.

```bash
# Optional: override the default preset
export GGML_CUDA_W4A4_PROFILE=balanced  # or: speed (the default when unset) | off | none
```

| profile | DOT8 | MIN_TOKENS | KSPLIT | SUMW | character |
|---|---|---|---|---|---|
| `speed` (default when unset) | 3 (per-16 u4x16) | 128 | 1 | 1 | the sealed F config: prefill win (see What we measured for per-gate numbers), K-split boost at 16-64, precomputed SUMW plane |
| `balanced` | 3 (per-16 u4x16) | 128 | unset | unset | lower KLD (0.090 vs 0.122 on 9B), no narrow-width cliff, smaller prefill win |
| `off` / `none` | - injects nothing - | | | | every flag behaves exactly as before the profile system; byte-identical to stock when no knobs are set |

An **explicit per-knob env always beats the profile** (e.g. `PROFILE=speed` +
`GGML_CUDA_W4A4_DOT8=3` -> per-16 wins; `PROFILE=speed` + `GGML_RP_KSPLIT=0` -> ksplit
stays off). The profile only injects defaults for the four int4-path knobs above -
`GGML_CUDA_REPACK_Q4_0`, `GGML_W4A4_ONLY`, `GGML_RP_GENERIC_MMV`, `GGML_CUDA_W4A4_SKIP`
and the debug gates are never touched by it, in either direction. Unknown values
(e.g. `PROFILE=fast`) print a WARN and behave as `off`. A complete server invocation:

```bash
GGML_CUDA_W4A4_PROFILE=speed \
  llama-server -m model-Q4_0.gguf -ngl 99 -fa on -c 8192
```

This config is correct for **both** batch-prefill serving and MTP/speculative-decode
workloads: prefill runs the fast path (widths >= 128), MTP verify passes and decode run
the stock path automatically (widths below the floor). No per-model branching needed.
(The equivalent explicit set - `DOT8=3 MIN_TOKENS=128 KSPLIT=1 SUMW=1` - is
greedy-seal bit-identical to `PROFILE=speed`, and likewise for `balanced`. `DOT8=4` was
rejected by the seal - do not raise the preset.)

### Recommended configs

| workload | env |
|---|---|
| **Default (recommended)** | *(unset - resolves to `speed`)* or explicit `GGML_CUDA_W4A4_PROFILE=speed` |
| Accuracy-first / narrow-width stability | `GGML_CUDA_W4A4_PROFILE=balanced` |
| Stock-path narrow-width boost (no W4A4) | `GGML_CUDA_W4A4_PROFILE=off` + `GGML_RP_KSPLIT=1` |
| Conservative / testing (stock-identical) | `GGML_CUDA_W4A4_PROFILE=off` (or kill-switch `GGML_CUDA_W4A4_OFF=1`) |

### Production profile (what the published numbers assume)

Every benchmark and seal result published in this README and `docs/BENCH.md` was
measured under a fixed env bundle (the "w4a4-env" production profile), not under
bare binary defaults:

```bash
export GGML_CUDA_REPACK_Q4_0=1     # repack admission for Q4_0 (default OFF in source)
export GGML_TP_AR_SKIP_XBAR=1      # TP all-reduce: skip the xbar leg
export GGML_CUDA_REPACK_KQUANT=0   # K-quant repack OFF (binary default is ON)
export GGML_CUDA_MOE_VEC=1         # row-lane MoE decode kernels (default OFF)
export GGML_CUDA_MOE_FUSE=1        # MoE gate/up + SwiGLU prefill fusion (default OFF)
export GGML_GDN_COLTHREAD=1        # column-thread GDN kernel path (default OFF)
```

plus the rig-level runtime knobs already listed at the top of `docs/BENCH.md`
(`GGML_ENABLE_CUSTOM_AR=1`, `HSA_XNACK=0`, `HSA_FORCE_FINE_GRAIN_PCIE=1`,
`GPU_MAX_HW_QUEUES=8`). Because `GGML_CUDA_REPACK_Q4_0` defaults OFF in source, a
run without this bundle does not engage the tiled/W4A4 repack paths at all.

Gate-truth verification (2026-10-31, dual seal on the seal build): the
env-profile seal bit-matches the promoted baselines in all cells -- and those
baselines were recorded with repack OFF -- which is precisely the zero-divergence
evidence that the tiled repack path is **bit-identical to the default path where
engaged** (seal-verified at validated widths; batch sizes 2-15 are not covered by
the seal). A true-defaults seal run diverges from the baselines on the
long-prefill cases, confirming empirically that the published numbers describe
the bundle above. Full gate-default table and seal matrices:
`benchmarks/defaults-vs-env.md`.

### Optional tuning variables

Explicit values of these knobs **override the profile preset**; unset values fall
through to the profile, then to the built-in default.

| variable | default | effect | when to touch it |
|---|---|---|---|
| `GGML_CUDA_W4A4_PROFILE` | unset -> `speed` | preset over DOT8/MIN_TOKENS/KSPLIT/SUMW (speed \| balanced \| off \| none) | the one switch - see table above |
| `GGML_CUDA_W4A4_MIN_TOKENS` | unset (floor off; speed preset sets 128) | gate falls back to stock below this batch width (dense AND MoE dispatch; floors every u4 path: per-32 u4act, per-16 u4x16 GEMM and Q4_1 u4q41 alike) | set `128` for MTP coexistence or if narrow-batch/MoE-decode throughput dips |
| `GGML_RP_KSPLIT` | unset (speed preset sets 1) | K-splits the repacked w32 GEMM (1 = auto slice count, 2/4/8 = explicit) | only useful on the stock path at 16 <= width <= 64; inert under the gate |
| `GGML_CUDA_W4A4_DOT8=3` | - | per-16 activation variant | the shipped preset value (both profiles); no narrow-width cliff, smaller prefill win than the historical per-32 gate-1 number |
| `GGML_CUDA_W4A4_SUMW` | unset | precomputed SUMW plane for the Q4_0 u4 epilogue (+2 B/block) | `speed` presets it on; trade VRAM for MoE/narrow decode - NOTE: dense prefill pays ~-2% pp512 (BENCH section 70), so keep it off when dense prefill throughput matters |
| `GGML_CUDA_REPACK_Q4_0` | unset (repack admission OFF) | admits Q4_0 to the repack path regardless of dot8 mode | the production profile sets `=1` (see Production profile); unset = stock path |
| `GGML_RP_GENERIC_MMV` | unset | forces the generic (non-repacked) MMV path | troubleshooting - compare against to isolate the repack path |
| `GGML_W4A4_ONLY` | unset | restricts execution to W4A4 ops only (skips everything else) | debugging graph coverage, not for serving |

Gate values are **modes, not additive flags**: `DOT8=1` (per-32), `=3` (per-16), `=4`
(= 1 + Q4_1 u4), `=5` (= 4 + Q4_K u4, NO-GO quality - keep off). Set exactly one.

### Troubleshooting

- **"Did it even turn on?"** - every gate prints a one-line `WARN` marker to the log when
  active (e.g. `CUDA w4a4: GGML_CUDA_W4A4_DOT8=1 gate ACTIVE (...)`, the floor's
  `... floor ACTIVE (u4act falls back to stock below this batch width)`). No marker in
  the log = the path did not engage; check the variable spelling and that the model file
  is Q4_0 (the gate is Q4_0-family; Q4_K files stay stock - see the Q4_K note above).
- **Output differs from stock**: expected and bounded - the gated prefill shifts logits
  slightly (measured KLD envelope above; decode/verify run bit-stock under the floor).
  If outputs look *degraded* (loops, nonsense) rather than merely different: unset
  `GGML_CUDA_W4A4_DOT8` to confirm the gate is the cause, then report it - that is a bug
  (the envelope says it shouldn't happen).
- **Throughput regression at some width**: check the floor - if the width in question is
  below `MIN_TOKENS` you're seeing stock (correct); if above, try `DOT8=3` (the per-16
  variant has no narrow-width cliff).
- **Multi-GPU + `llama-bench`**: this fork's bench `-dev` takes comma = device list
  (slash = benchmark cross-product) - the opposite of upstream; see the note in
  Reproduce below.
- **RAM at very long context**: `llama-perplexity` needs >= 2x context tokens of corpus
  and builds n_ctxxn_vocab logits in host RAM (a 150k rung needs ~91 GB) - a tooling
  limit, not the model's.

## Design philosophy

- **Correctness-first**: a negative result is a result. Every claimed number below comes
  from a run recorded in `docs/BENCH.md` with reproducible commands.
- **Env-gated with a sealed default**: since the P5 default flip, an unset profile
  resolves to the sealed `speed` preset; `PROFILE=off`/`none` or the
  `GGML_CUDA_W4A4_OFF=1` kill-switch restores byte-identical stock behavior, and
  explicit per-knob envs override the preset per model or fleet-wide.
- **Sealed algebra**: the epilogue's arithmetic is verified against f64 ground truth
  (bit-exact at double precision) with negative controls that must fail.

## What we measured (performance)

All numbers on ROCm MI50 (gfx906), llama-bench / llama-perplexity, base-model Q4_0 files,
r>=3 unless noted, interleaved arms, VRAM returned to baseline (10.87 MB/GPU) between legs.
Unless a cited section says otherwise, runs used the production env profile above (see
Production profile), since `GGML_CUDA_REPACK_Q4_0` defaults OFF in source.
Full tables and repro commands: `docs/BENCH.md` (campaign log); summary records for
the benchmark-model runs are mirrored under `benchmarks/`.

### Prefill (prompt processing)

| model | stock | W4A4 (gate1) | ? | PPL cost |
|---|---:|---:|---:|---:|
| Qwen3.5-9B (Q4_0) | 1131 | 1462 | **+29.3%** | +2.0% |
| Qwen3.8-27B (Q4_0) | 284.9 | 373.6 | **+31.1%** | +2.0% |
| Qwen3-30B-A3B (MoE, Q4_0) | 1269 | 1378 | +8.6% | +3.2% |
| Llama-3.1-8B (Q4_0) | - | +30.9% (pp512) | - | +20.3% (pp2048 PPL) |
| gemma-4-12B (Q4_0) | - | +39.5% (pp512) | - | +15% PPL - gemma's error is NOT mid-band-FFN-localized; outlier, do not generalize from it |

The MoE model measured gains ~a third of dense on its expert-heavy file (+8.2-8.6%);
this scales with expert-to-attention GEMM ratio and should not be generalized from one point.
The width floor covers MoE too (same `MIN_TOKENS` knob): bare gate on a small-expert MoE
measured a -13.9% decode penalty that the floor eliminates (Qwen3-30B-A3B:
gate+floor = pp +5.0% / tg -0.7%). In contrast, the 27B's merged-path bench reading was +34.4% (bench) - batch/prefill overlap accounting differs between bench and server; see the decode section.
per-16 trades prefill width for accuracy and narrow-width stability (sec 54, sec 61).

### Narrow widths - where it loses

`W4A4=1` inverts across batch width: at pp16-64, gate1 is **10-16% slower than stock**
(r=5, 27B: pp16 69.3 vs 78.5, pp32 131.7 vs 151.9; the width crossover sweep measured
-14.9%/-16.3% on its run). This matters for speculative-decode
verify passes, which run in exactly this window - but the fix is built in: a **width
floor** (`GGML_CUDA_W4A4_MIN_TOKENS`, e.g. 128) makes the gate auto-fall-back to the
stock path below that batch width, so MTP verify passes and decode run bit-stock while
prefill keeps the full gate win. Measured on 27B with MTP: gate1+floor+MTP 47.3 t/s vs
stock+MTP 49.05 (-3.5%, accept-jitter residual; accept-length distributions identical),
native-MTP+gate+floor 49.9 t/s **with** +31% prefill - the previously impossible
combination now works. Output consistency verified: temp-0 MTP deterministic (20/20
reps token-identical), below-floor outputs 100/100 token-identical to stock, above-floor
divergences are single near-tie token flips (0.11 nats), zero degenerate outputs in
300+ generations, greedy primes byte-exact with the floor on. Un-floored reference:
stock+MTP 48.7 vs gate1+MTP 29.8 t/s - don't run the gate bare with MTP.

### K-split - narrow-width win on the stock path

`GGML_RP_KSPLIT=1` on the **stock** path (no W4A4 gate): +10.6% pp16 / +7.2% pp32 / +1.2%
pp64 on 27B, wide-batch parity, **PPL bit-identical** (per-chunk arrays byte-equal, 6.8534).
Proven inert under DOT8 gates (byte-identical), so it is safe to enable globally:
only models without the gate see it.

### Decode (token generation)

tg128 within noise on dense models (stock-path ksplit measured +1-4% at model level in early sections); the gate-1+ksplit combination is NOT buildable (static_assert, see above) and its early +3.7% reading was refuted in the program's own records. Decode is not where this path pays.

## What we measured (correctness)

- **Seals**: epilogue algebra bit-exact vs f64 ground truth (0.0 max rel err, 20k trials,
  adversarial patterns); staged fp32 form <= 9.7e-6; isolation form <= 1.05e-5. Negative
  controls (sign-flipped-M 15813/15813, swapped sc/m) fail as required. Harness validated
  by per-regime declared-spec input checking + validator self-test (corrupt-everything
  control; deliberate degeneracies declared).
- **Greedy seals**: greedy prime sequences byte-identical vs stock across primes 2-29.
- **KLD envelope** (9B Q4_0, wikitext-2-raw, c512 ch128, vs f16 base):
  stock recipe (no imatrix) 0.0684; +imatrix + map 0.0412; W4A4 gate1 0.1222 (0.0764 with
  calibration on our map); SKIP-band variants 0.0979. The u4 activation penalty is
  ~ one quant step: calibrated, it sits ~33% of the way across the Q4->Q3 gap; uncalibrated
  ~68%. Seals prove the kernel is bit-exact - all accuracy cost is encode-channel
  (quantizer design), not implementation.
- **Anchors EXACT across refactors**: gates 1/2/3 reproduce to zero ULP on every
  re-baseline (PPL 8.7630 / KLD 0.122150 / 0.097917 / 7.2774, gate4 PPL bit-identical).

## What we tried and rejected (so you don't have to)

- **Q4_K u4** (`GGML_CUDA_W4A4_DOT8=5`): prefill +12.5% (below the +25% adopt bar),
  KLD +0.0173 (over the 0.01 relative envelope), tg128 -60% (per-call epilogue cost
  doesn't amortize at single-token). NO-GO measured end-to-end; code exists on a branch,
  default-off, inert.
- **Per-block least-squares scale**: KLD +13-25% vs the shipped imatrix+map pipeline
  (which is already weighted-LS + discrete clamping-pattern search). Negative; branch
  `exp/q40-ls-scale` is env-gated/inert.
- **Hadamard rotation (H32)**: simulated gains didn't survive the live path - recorded
  as the program's cautionary precedent for activation-side levers.
- **SmoothQuant-style offline channel fold**: *positive*, but size-dependent - recovers
  67% of the u4 activation penalty at 9B, ~32% at 27B (weight-side gain was a 9B-file
  artifact; +1.07% at 27B, nothing). Filed as a small-model lever and the u4 gate's
  accuracy partner; held for adoption (adds a fold step to the encode recipe).
- **SKIP layer bands**: layer selection does not transfer across models (measured closure
  collapses ~3x on a different depth, ordering inverts). Per-model sweep required; band
  is not shipped as portable.

## Cost/benefit summary

**Benefits** (for batch-shaped inference workloads):
- +29-40% prefill on dense Q4_0 models at SHORT-to-MEDIUM context; at long context
  (131k) the gain narrows to ~+8-10% (KV-bound). The headline is a short-context number,
  not a universal one - bound claims to the regime they were measured in.
- +10.6%/+7.2% at pp16/pp32 from K-split on the stock path, bit-identical output
- Zero VRAM overhead in steady state; per-model env-gated adoption; greedy-identical

**Costs**:
- 10-13% prefill loss at pp16/32 under the W4A4 gate (pp64's leg has 22% ? - only pp16/32
  support the claim; the spec-decode window - the reason gate and MTP are mutually
  exclusive per model)
- ~2% PPL (dense Q4_0 calibrated), ~3% (MoE); gemma-4-12B measured +15% and is the
  outlier - its error is not mid-band-localized, do not generalize from it
- The u4 activation path costs ~ one quant step of additional divergence - ~42% of the
  measured Q4->Q3 gap calibrated, ~65% uncalibrated (denominator: the 9B's own
  8->4->3 pure-type step 0.083232) - this is encode-channel, not the GEMM

**Who should use this**: batch-heavy serving on gfx906-class cards where prefill
throughput dominates and PPL tolerance is >=2%. **Who should not**: MTP/speculative-decode
workloads (use stock + MTP; measured best), MoE-heavy decode workloads (gains are ~?),
or deployments where the narrow-width cliff matters.

## Provenance

All benchmarks used **base models** (upstream or bartowski-style uncalibrated
Q4_0/quantized-from-bf16), never fine-tuned for content: Qwen3.5-9B-Base,
Qwen3.8-27B-Base (Unsloth), Qwen3-30B-A3B-Base, Llama-3.1-8B-Base (Meta),
gemma-4-12B-Base. One tooling-validation arm used Mistral-7B-Instruct-v0.3 (Q4_K_M,
single-GPU single-request arm only) - flagged here as an instruct (finetune) so it is
never cited for accuracy claims. GGUF metadata carries the base-model proof
(`general.base_model` fields). Calibration uses the files' own imatrix statistics where
stated; uncalibrated arms are labeled as such.

## Branch structure

- `CV-port`: this branch - the CV-port lineage (tiled Q4_0 repack GEMM, CV kernel ports
  and fusion gates) on top of the int4 shipping tree
- `int4-profile`: the int4 shipping tree (single squashed commit on exact upstream)
- `master`: exact upstream copy
- `q41-u4-probe` - Q4_1/Q4_K u4 work (Q4_1 GO'd and merged; Q4_K NO-GO'd, kept for the record)
- `pr0-w4a4-correctness`, `pr1-q40-repack`, `pr2-ksplit` - per-PR slices of the above
- `fix/llama-bench-dev-comma` - tooling fix (bench `-dev` delimiter semantics)
- `exp/q40-ls-scale` - negative result, kept env-gated for reproducibility

## Reproduce

Every number cites a BENCH section with the exact command; the harness is
`llama-bench` + `llama-perplexity --kl-divergence` against f16 logits bases built on the
same corpus/chunk protocol. Note: this fork's llama-bench `-dev` fix changes delimiter
semantics (comma = device list for split, slash = benchmark cross-product) vs upstream's
inverted convention - commands in these docs use the fork convention. Raw logs are
retained on the measurement rig; the summary records for the benchmark-model campaigns
are mirrored under `benchmarks/` in this repo.
