# defaults-vs-env: true binary defaults vs the production env bundle - gate truth + dual seal

Date: 2026-10-31 (PDT). Build: the seal build (identical libggml-hip binary to the
prior seal witness). The seal harness runs each cell in two modes: `env` (production
bundle) and `defaults` (no `GGML_*` vars in the server environ). Trimmed copy of the
internal gate-truth record; local build paths and script names omitted.

## 1. Gate-default truth (source-verified)

| Gate | Binary default (source) | Production bundle | Non-default in the bundle? |
|---|---|---|---|
| `GGML_CUDA_REPACK_Q4_0` | **OFF** - `e != nullptr && atoi(e)!=0` (repack-common.cu:48-49) | `=1` | **YES** |
| `GGML_CUDA_REPACK_Q4_0_TILED` | **ON** - `e == nullptr \|\| atoi(e)!=0` (gcn-u4-gemm.cu:40-43) | unset | no (default ON) |
| `GGML_CUDA_W4A4_PROFILE` | unset -> resolves to **speed** (P5, w4a4-profile.h:94-96): DOT8=3, MIN_TOKENS=128, KSPLIT=1, SUMW=1 | `=speed` | no (same resolution) |
| `GGML_ENABLE_CUSTOM_AR` | **ON** - unset/empty/!='0' enables (ggml-cuda.cu:1757-1759) | `=1` | no (= default) |
| `GGML_TP_AR_SKIP_XBAR` | **OFF** (tp-allreduce.cu:1201-1203) | `=1` | **YES** |
| `GGML_TP_AR_TWOSHOT` | auto (tp-allreduce.cu:1154-1157) | unset | no |
| `GGML_CUDA_REPACK_KQUANT` | **ON** (off only on explicit `=0`, repack-common.cu:83-84) | `=0` | **YES** (bundle flips OFF) |
| `GGML_CUDA_REPACK_Q8_0` | OFF, opt-in (repack-common.cu:109-110) | unset | no |
| `GGML_CUDA_REPACK_Q8_0_STOCK_ACC` | ON (repack-common.cu:155-156) | unset | no |
| `GGML_GDN_COLTHREAD` | **OFF** (gated_delta_net.cu:445) | `=1` | **YES** |
| `GGML_CUDA_GCN_KQ_MOE` | ON, default 1 (gcn-q8-gemm.cu:2291) | `=1` | no (= default) |
| `GGML_CUDA_MOE_VEC` | **OFF** (ggml-cuda.cu:2949-2952) | `=1` | **YES** |
| `GGML_CUDA_MOE_FUSE` | **OFF** (ggml-cuda.cu:5714) | `=1` | **YES** |
| `GGML_CUDA_MOE_DEC_MAX_TOKENS` | 8 (ggml-cuda.cu:2965-2966) | `=8` | no (= default) |
| `HSA_XNACK=0`, `HSA_FORCE_FINE_GRAIN_PCIE=1`, `GPU_MAX_HW_QUEUES=8` | HSA runtime knobs, not GGML gates | set | kept in both modes (hardware stability) |

**Headline:** `GGML_CUDA_REPACK_Q4_0` defaults **OFF**. With true binary defaults, Q4_0
tensors are never admitted to the repack path, so the tiled `gcn_u4_gemm` never
processes them - even though the `REPACK_Q4_0_TILED` gate itself is default-ON, it sits
downstream of admission. Likewise the W4A4 u4 GEMMs act only on repacked layouts, so
under true defaults they do not engage for Q4_0 either (the W4A4 *profile* resolves to
speed by default and its banners print, but that is gate-resolution noise, not
engagement). **The "default-ON tiled/W4A4" phrasing describes the env-bundle profile,
not the binary default.**

**Banner semantics (false-pass nuance):** the `[attn-qk-fused] decode q/k chain fusion
ACTIVE` banner prints at the fused-kernel call site (rope.cu:1259, static init at
dispatch) - a true engagement witness, and it appeared in all 8 cells of both profiles
(fusion is structurally default-ON, no env gate). The `REPACK_Q4_0_TILED gate ACTIVE`
banner prints whenever the gate is *read* (it appeared even in defaults mode where no
repacked tensor exists) - so it is a gate witness, not an engagement witness on its own;
the engagement witness is the **pair** `REPACK_Q4_0 gate ACTIVE` +
`REPACK_Q4_0_TILED gate ACTIVE`.

## 2. Harness enforcement (script-level, no source changes)

The seal harness greps each cell's server log after generation and **fails the cell**
(cell returns 1 regardless of generation exit code) if a required banner is absent:
- all cells, both profiles: `[attn-qk-fused] decode q/k chain fusion ACTIVE`
- `env` mode + q40 cells additionally: `GGML_CUDA_REPACK_Q4_0 gate ACTIVE` and
  `GGML_CUDA_REPACK_Q4_0_TILED gate ACTIVE`

A broken/fallback kernel can no longer score a passing hash silently.

## 3. Seal run (a): production env profile

| cell | verdict | primes8 | tensor-def | secs-conv | long-pp-r01 | long-pp-r02 | long-pp-r03 |
|---|---|---|---|---|---|---|---|
| ud-tp1 | GREEN | 7f8fcbbddf1f | df2de750f0ad | d52628b1adc0 | 687dbf064255 | a3a6ce42f71c | f2d63e2937a4 |
| ud-tp2 | GREEN | 7f8fcbbddf1f | df2de750f0ad | d52628b1adc0 | 687dbf064255 | a3a6ce42f71c | 0e109abdf4e1 |
| q40-tp1 | YELLOW^1 | 7f8fcbbddf1f | 4b5bc3be1e56 | d52628b1adc0 | bd179188648f | e5bf6924d256 | f704aaf2a131 |
| q40-tp2 | YELLOW^1 | 7f8fcbbddf1f | 4b5bc3be1e56 | d52628b1adc0 | bd179188648f | 2ef8738adc2b | 0818dd4f4b59 |

^1 YELLOW solely from `fingerprint-mismatch: gates` (the profile declares
REPACK_Q4_0=1/W4A4_PROFILE=speed; the baseline fingerprint lacks them). All 24 case
hashes **bit-match the prior seal run (4/4 cells) and the promoted baselines (4/4
cells)** - zero drift, zero divergence. Banners: all present, none missing.

## 4. Seal run (b): TRUE defaults

Witness confirms zero `GGML_*` vars in the server environ (ROCm runtime + HSA knobs
only). Honest default-profile fingerprints:

| cell | verdict vs baseline | primes8 | tensor-def | secs-conv | long-pp-r01 | long-pp-r02 | long-pp-r03 |
|---|---|---|---|---|---|---|---|
| ud-tp1 | RED | 7f8fcbbddf1f | df2de750f0ad | d52628b1adc0 | 687dbf064255 | a3a6ce42f71c | **492cca331df9** (!= f2d63e2937a4) |
| ud-tp2 | RED | 7f8fcbbddf1f | **b249291e9084** (!= df2de750f0ad) | d52628b1adc0 | 687dbf064255 | a3a6ce42f71c | **648b76ec108a** (!= 0e109abdf4e1) |
| q40-tp1 | RED | 7f8fcbbddf1f | 4b5bc3be1e56 | d52628b1adc0 | bd179188648f | **a3a6ce42f71c** (!= e5bf6924d256) | **17c7d67a7e3e** (!= f704aaf2a131) |
| q40-tp2 | RED | 7f8fcbbddf1f | 4b5bc3be1e56 | d52628b1adc0 | bd179188648f | **ae639bad32c0** (!= 2ef8738adc2b) | **17c7d67a7e3e** (!= 0818dd4f4b59) |

Divergences are confined to the long-prefill cases (plus ud-tp2 tensor-def) - exactly
the cases sensitive to the seal gate bundle (`MOE_VEC`/`MOE_FUSE`/`GDN_COLTHREAD` OFF,
`REPACK_KQUANT` back to default ON). Short cases are gate-neutral. Banners:
`attn-qk-fused` present in all cells (fusion engages under true defaults); repack
banner correctly absent; TILED gate banner prints but is non-engaging (see section 1).

## 5. Which profile do the promoted results describe?

- The **promoted baselines** record gates `GCN_KQ_MOE=1; MOE_DEC_MAX_TOKENS=8;
  MOE_FUSE=1; MOE_VEC=1; REPACK_KQUANT=0; GDN_COLTHREAD=1` - i.e. the **seal gate-bundle
  profile**, NOT true binary defaults (MOE_VEC/MOE_FUSE/GDN_COLTHREAD default OFF,
  REPACK_KQUANT defaults ON). Run (b) proves this empirically: true defaults diverge
  from the baselines on the long-pp cases.
- The baselines were recorded **without** `REPACK_Q4_0` - so with respect to the
  repack/tiled gate specifically, the promoted results describe the repack-OFF
  (true-default) path. The env-profile seal (run a) matching those hashes bit-for-bit
  is precisely the zero-divergence evidence: **tiled/W4A4 repack ON == repack OFF
  output, bit-identical** - the adoption claim stands, but it must be phrased as
  "bit-identical to the default path, engagement requires `GGML_CUDA_REPACK_Q4_0=1`",
  not "default-ON".
- The prior tier3-adopt 4-cell seal was an **env-bundle run** (witnesses show
  `REPACK_Q4_0=1` etc. in the server environ); run (a) reproduces it exactly (no drift).
