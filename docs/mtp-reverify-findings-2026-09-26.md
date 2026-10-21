# MTP lossless re-verification + fix (2026-09-25/26)

**Slot A brief** (`docs/mtp-reverify-brief.md`). Box mi50 (3x MI50, gfx906),
binary `/root/build-w4a4` (build 10956, commit 8473f62c3), tree
`perf/w4a4-dot8-gfx906` @ 2570c9a5a (post-M4-fix 2e8499bb6). Model
`<27B community MTP-preserved finetune>.gguf` (29 GB),
port 8094, `-c 32768 -b 512 -ub 512 -fa on -sm tensor -tps 3`, env
`/root/w4a4-env.sh` (CUSTOM_AR=1, SKIP_XBAR=1).

## VERDICT: violation REPRODUCES on the current binary → H3 (AR-state) CONFIRMED

The M4-era lossless violation is **not** transient/binary-dependent. Re-running
the original failing comparison on the current binary reproduces the exact
divergence (same "Structure: List the primes..." vs "First ten primes:..."
flip at line 22 of the reasoning stream).

### STEP 1 re-verification table (greedy, temp 0 top-k 1, n 400)

| Comparison | Result |
|---|---|
| noMTP run1 vs run2 (determinism control) | **IDENTICAL** (8e110515…) |
| MTP (SKIP_XBAR=1) leg1 vs noMTP | **DIFFERS** (8cbba228…) |
| MTP (SKIP_XBAR=1) leg2 vs noMTP | **DIFFERS** (8cbba228…) |
| MTP-vs-MTP determinism (SKIP_XBAR=1) | **IDENTICAL** (8cbba228…) |
| MTP (SKIP_XBAR=0) vs noMTP | **DIFFERS** (8cbba228…) |
| MTP (CUSTOM_AR=0) vs noMTP | **IDENTICAL** (8e110515…) |

MTP legs are deterministic with themselves but consistently diverge from the
no-MTP control under every custom-AR knob combination. The ONLY arm that makes
MTP byte-identical to no-MTP is disabling custom AR entirely. This is the
H3 signature: the bug is in the custom-AR cross-call state under MTP batch
shapes, not in the sampler or the draft head.

### Server-log warning evidence

`spec common_specu: backend offload failed for seq_id=0..3; using CPU sampler`
appears **4x per MTP leg** (all arms), **0x on no-MTP legs** — same signature
as the M4-era failing test and the Ornith MTPv2 violation. The warning is a
necessary-but-not-sufficient marker: it fires on every MTP leg regardless of
whether the AR knobs make the output lossless, so it tracks the draft-sampler
CPU fallback, not the divergence itself. The divergence is AR-side.

## Root cause (H3 confirmed)

Speculative decoding alternates AllReduce batch widths every token: the draft
head runs on a narrow batch, the verify pass on a wider batch. The custom AR
kernels (`tp-allreduce.cu`) carry cross-call state that is only proven safe
for a **single steady width per context**:

- the broadcast staging layout is `[nranks][n_elements]` — the slot offsets
  scale with `n_elements`, so two widths alias each other's slots;
- `GGML_TP_AR_SKIP_XBAR` parity double-buffering derives the set from the
  per-block `_flag` counter, which assumes a constant cadence;
- the lazy single-slot staging realloc in `tp_custom_ar_prepare` frees /
  re-points buffers that peer kernels from a previously-captured graph may
  still be peer-writing into.

Under graph capture + async replay the two widths interleave on this shared
state and the reduction picks up stale/garbage partials → a near-tie flips →
cascades over the greedy stream. `-sm layer` mode (no cross-GPU AR) and
`CUSTOM_AR=0` (regular AllReduce path) are both immune, which is exactly what
H3 predicts.

### Per-width staging pool: tried and FALSIFIED

A per-width staging pool (one staging + one device-side RankData per distinct
`(n_elements, slots)`, so no live buffer is ever freed or re-pointed) was
implemented and tested. It does **NOT** restore losslessness: the shared
per-block signal/flag state (`Signal.start/end/_flag`) still pairs
mismatched-width calls across the barrier, so the pool alone is insufficient.
This is why the shipped guard is a sticky lock rather than a pool.

## The fix: env-gated sticky-lock guard (GGML_TP_AR_MTP_SAFE=1/2)

`tp_custom_ar_mtp_safe_transition(ctx, n_elements)` in `tp-allreduce.cu`
tracks the last AR width. On the first width transition it sets a sticky
`mtp_safe_locked` flag and returns true; the call site in
`ggml_backend_cuda_comm_allreduce_custom_prepare` (ggml-cuda.cu) then routes
the whole context to the regular AllReduce path. The staging is deliberately
NOT freed — earlier-captured graphs still reference it and as the sole
remaining custom user (constant width) it stays safe. `=2` logs the
transition for evidence.

**Fix-test (final binary, MTP_SAFE=2):**
- `MTP-vs-noMTP (MTP_SAFE=2): IDENTICAL` (both 8e110515…) — **LOSSLESS PASS**
- guard fired once: `width transition 10240 -> 102400; custom AR locked off`
  (10240 = decode-width AR, 102400 = prefill-width AR)

## Throughput (creative ~200 tok, temp 0.1, median of 3, final binary)

| Leg | tg median | note |
|---|---:|---|
| base, custom AR on (noguard) | **45.74** | fastest baseline |
| MTP, custom AR on (noguard) | 45.75 | MTP gain ~0% (violates lossless) |
| MTP, guard locked (MTP_SAFE) | **40.99** | lossless MTP |
| base, guard locked (MTP_SAFE) | 32.60 | custom AR off baseline |

Honest deployment read: the correct comparison is **noguard base (45.74) vs
lossless-guarded MTP (40.99) = −10.4%**. The guard disables custom AR for the
whole context on any width change, and custom AR is a large decode win on this
box (45.74 vs 32.60). So lossless MTP is **slower than plain no-MTP with
custom AR** — MTP remains **undeployable** on this stack, consistent with the
M4-era +7.5% < +10% threshold conclusion. (Relative to the locked-off
baseline, MTP is +25.7%, but that is not the deployment-relevant baseline.)

## Guard coarseness (known limitation + follow-up)

The shipped guard locks on the FIRST width transition, which includes the
ordinary prefill->decode width change (observed: 10240 -> 102400 during
warmup+prefill). So a guarded context loses custom AR for decode even without
speculative decoding — that is why base_safe is 32.6 not 45.7. A refinement
(lock only on the A->B->A width-REVISIT pattern, which is the true
draft/verify alternation signature and never occurs in one-way prefill->decode
chunking) was drafted but is UNTESTED for losslessness and deliberately not
shipped. If MTP is ever revisited, test the pattern guard first: it would keep
custom AR for non-speculative decode while still locking spec-decode contexts.

## Decision

- MTP is **NOT cleared** on Qwen3.6 Native-MTP at 3rk MI50. The violation
  reproduces on the current binary; H3 (custom-AR cross-call state under MTP
  batch shapes) is confirmed.
- The env-gated guard (`GGML_TP_AR_MTP_SAFE=1`) makes MTP lossless at the
  cost of disabling custom AR for the context → net −10.4% vs the fast
  no-MTP baseline. No deployment win.
- Do not deploy `--spec-type draft-mtp` on this stack. The guard exists for
  correctness-critical runs that must use MTP; it is off by default.

## Files

- `ggml/src/ggml-cuda/tp-allreduce.cu` — `tp_custom_ar_mtp_safe_transition`
- `ggml/src/ggml-cuda/tp-allreduce.cuh` — guard state + declaration
- `ggml/src/ggml-cuda/ggml-cuda.cu` — guard call site in custom_prepare
- `scripts/mtp_reverify.sh`, `scripts/mtp_fixtest.sh` — test drivers
- Raw: `mi50:/root/mtp-reverify.txt`, `mi50:/root/mtp-reverify-logs/`,
  `mi50:/root/mtp-fixtest.txt`, `mi50:/root/mtp-fixtest-logs/`
