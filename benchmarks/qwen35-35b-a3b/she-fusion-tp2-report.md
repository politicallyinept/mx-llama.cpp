# She-Fusion TP2 Confirmation + Full Seal Suite -- Adoption Evidence Package

Date: 2026-10-27 (PDT). Box: mi50. **Recommendation only -- no gitea push, no adoption executed.**

Engine: `rig/build-stacked` (`libggml-hip.so` md5 `c0ef6ec179aeeb96526fff086673fd21`), she-fusion = commit (internal ref) ("cuda: port CV fused shared-expert/dense-f32 decode path (SHE_FUSE)", branch `she-fusion` in `rig/shefusion-src`), gate `GGML_CUDA_SHE_FUSE` (default OFF). `[she-fuse] shared-expert gate-tail fusion ACTIVE` banner witnessed in every SHE_FUSE=1 server log.

**Protocol:** every leg under `flock -w 7200 rig/gpu.lock` (queued behind the concurrent T3->T4 bisect agent, never forced, never killed; TERM-only teardown; all 3 cards 0% at campaign end). Gate bundle verbatim per leg (campaign fork gates, = c4-attribution/probe.sh): `GGML_GDN_COLTHREAD=1 GGML_CUDA_GCN_KQ_MOE=1 GGML_CUDA_REPACK_KQUANT=0 GGML_CUDA_MOE_VEC=1 GGML_CUDA_MOE_FUSE=1 GGML_CUDA_MOE_DEC_MAX_TOKENS=8` + `GGML_CUDA_SHE_FUSE=<0|1>`; q40 legs add `GGML_CUDA_REPACK_Q4_0=1 GGML_CUDA_W4A4_PROFILE=speed`. All tp2 legs under w4a4-env (TheRock), ldd-witnessed `hip_lib=rig/therock-venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7` in every tp2 witness. tp2 server flags: `-ngl 99 -fa on -sm tensor -fit off -c 16384 -b 4096 -ub 1024 -t 16 -np 1 --device ROCm1,ROCm2 -lm dio`.

## 1. TP2 UD tg256 A/B (CONFIRMED -- exceeds expectation)

Protocol identical to banked audit R3 (server `/completion`, n_predict 256, temp 0, seed 42; per boot: 2x8-tok + 1x256-tok warmup discarded, 1 measured rep; 3 interleaved boots per arm).

| arm | reps (t/s, chronological) | median | Delta vs banked 111.49 |
|---|---|---|---|
| SHE_FUSE=0 | 111.55 / 111.51 / 112.33 | **111.55** | +0.05% (banked value reproduced) |
| SHE_FUSE=1 | 118.04 / 99.89* / 117.40 | **117.40** | **+5.30%** |
| **Delta (on vs off)** | | **+5.24%** | expected ~ +2.3% (C1 extrapolation) |

\* rep-2 outlier (99.89): the bisect agent's q81cache battery ran GPU sessions whose lock accounting overlapped my lock window (their campaign log shows LOCK-HELD at 15:30/15:40 inside my held windows -- they are not honoring `rig/gpu.lock` the same way). The same boot's warmup rep measured 117.51, so the boot itself was healthy; median over 3 reps is robust to it. Off-arm legs in the same interleaving stayed at 111.5x.

**Findings:**
- The TP2 gain (**+5.24%**) is **larger** than the +2.3% C1 extrapolation (which assumed the fused-kernel saving halves under tensor split). The tp1 A/B was +3.91%; tp2 delivers more, not less.
- With fusion ON the fork lands at **117.40 t/s vs CV's banked 114.49** -- the fork **overtakes** CV by +2.5%. She-fusion alone over-explains the original +2.7% CV lead (fork 111.49 -> CV 114.49). Attribution of the CV lead to the gated-OFF fusion is CONFIRMED and then some; the residual (why CV was slower than fork+fusion) is a CV-side question, not a fork blocker.

## 2. TP2 Q4_0 sanity A/B (no regression -- improvement)

| arm | reps (t/s) | median | Delta |
|---|---|---|---|
| SHE_FUSE=0 | 110.00 / 109.62 | 109.81 | -- |
| SHE_FUSE=1 | 115.58 / 115.88 | 115.73 | **+5.39%** |

## 3. Full seal suite, SHE_FUSE=1 (benchgate chat-completions protocol: temp 0, seed 42, top_p 1, top_k 1, max_tokens 64/64/32/48/48/48, sha256 of content)

References: C4 report stacked table (`rig/c4-attribution/report.md`); for UD-tp1 no local stacked reference exists (store records not on box), so a direct `SHE_FUSE=0` leg (`C-tp1-ud-off`, same build/gates/protocol) serves as the reference. Historical GREEN trio: primes8 , tensor-def , secs-conv .

| shape | quant | case | SHE_FUSE=1 sha8 | reference sha8 | verdict | reference source |
|---|---|---|---|---|---|---|
| tp1 | ud | primes8 | | | MATCH | direct off |
| tp1 | ud | tensor-def | | | **DIVERGE** | direct off |
| tp1 | ud | secs-conv | | | MATCH | direct off |
| tp1 | ud | long-pp-r01 | | | MATCH | direct off |
| tp1 | ud | long-pp-r02 | | | MATCH | direct off |
| tp1 | ud | long-pp-r03 | | | MATCH | direct off |
| tp1 | q40 | primes8 | | | MATCH | C4 stacked + GREEN |
| tp1 | q40 | tensor-def | | | MATCH | C4 stacked + GREEN |
| tp1 | q40 | secs-conv | | | MATCH | C4 stacked + GREEN |
| tp1 | q40 | long-pp-r01 | | | MATCH | C4 stacked |
| tp1 | q40 | long-pp-r02 | | | **DIVERGE** | C4 stacked |
| tp1 | q40 | long-pp-r03 | | | MATCH | C4 stacked |
| tp2 | ud | primes8 | | | MATCH | C4 stacked |
| tp2 | ud | tensor-def | | | **DIVERGE** | C4 stacked |
| tp2 | ud | secs-conv | | | MATCH | C4 stacked + GREEN |
| tp2 | ud | long-pp-r01 | | | MATCH | C4 stacked |
| tp2 | ud | long-pp-r02 | | | MATCH | C4 stacked |
| tp2 | ud | long-pp-r03 | | | **DIVERGE** | C4 stacked |
| tp2 | q40 | primes8 | | | MATCH | C4 stacked + GREEN |
| tp2 | q40 | tensor-def | | | MATCH | C4 stacked + GREEN |
| tp2 | q40 | secs-conv | | | MATCH | C4 stacked + GREEN |
| tp2 | q40 | long-pp-r01 | | | MATCH | C4 stacked |
| tp2 | q40 | long-pp-r02 | | | **DIVERGE** | C4 stacked |
| tp2 | q40 | long-pp-r03 | | | **DIVERGE** | C4 stacked |

**18/24 cells byte-match; 6 diverge.** Key facts framing the REDs:

1. **The historical GREEN trio is untouched.** primes8 and secs-conv match in all 4 cells; tensor-def matches in both q40 cells. Every diverged cell is a case with **no historical GREEN reference** (long-pp r02/r03; UD tensor-def, which carries Q8_0 tensors and was only ever sealed on p0f/stacked eras).
2. **The gate is NOT byte-neutral** -- the "preserves reduction order" design claim holds only for the short greedy-32 probe C1 ran (which matched); at full seal token counts the fused path flips the first-divergent token on 6 of 24 cells. This is a finding, not a surprise class: the same situation (P5 , q8rp ) was previously accepted into stacked via re-seal of the flipped numerals.
3. **KL-context:** the drift ladder (`rig/drift-ladder/report.md`) shows stacked tp1-UD already sits at mean prefix-KL **0.0114** vs stock T0 -- the fork's sealed numerics are already inside an accepted drift envelope vs upstream. The she-fusion flips are token-level flips of the same kind, and the PPL pair below is **exactly 0 delta**, which is the stated criterion for "within the already-accepted numerics envelope".
4. Observation: the on-arm UD tensor-def numeral is identical on tp1 and tp2, and equals the hash C4 measured for p5only ud-tp2 tensor-def -- i.e. the fused path lands on a previously-seen alternative reduction numeral rather than a novel one.

Full sha256 of the six diverged cells:
- tp1-ud tensor-def `068e5f6f5fb5eb94afd68f9e2a60f640394c47951621216c6658e76b` (ref `b249291e90842e2d75dd650b549f9d206492fc58f92c7af8240e4ba07f5f3320`)
- tp1-q40 long-pp-r02 `d256f0f34e2a5e0812376777f2ecf05627c53c47f985d1b2c4f34bc2` (ref `f71ca28b5e0410e95070d8eb1aa2205df150ea018425073330130ac2`)
- tp2-ud tensor-def `068e5f6f5fb5eb94afd68f9e2a60f640394c47951621216c6658e76b` (ref `...`, C4 stacked)
- tp2-ud long-pp-r03 `f4e1d2c6103c68af7612d247a12407daf744f5a4015f71cd17c1d0bb` (ref `f2d63e2937a4d35fefde3811f8264ed7de421f55d80235a2f125ceae072f338f`)
- tp2-q40 long-pp-r02 `dc2be5fc842ba427bdebf41ec73edf36d5ca28cba29a7983966e897d` (internal ref)
- tp2-q40 long-pp-r03 `c4738a14b320f7b50595447dcb5190c4802785e83bf103339e34ab35` (ref `f704aaf2a13184ebb19ef04f1d4e69a7d351d83b60664a2d9e00ba730a75192f`)

## 4. Numerics spot: wikitext-2 PPL, tp1 UD, 8 chunks (`llama-perplexity -c 512 -b 2048 -ub 1024 -ngl 99 -fa on -lm dio --chunks 8`)

| arm | PPL |
|---|---|
| SHE_FUSE=0 | **6.1079 +/- 0.32560** |
| SHE_FUSE=1 | **6.1079 +/- 0.32560** |
| Delta | **0.0000** (identical to 4 dp) |

## Recommendation: **ADOPT** (default `GGML_CUDA_SHE_FUSE=1`), conditional on human-approved re-seal of the 6 flipped cells

Evidence line: **TP2 UD decode +5.24% (median 117.40 vs 111.55; banked 111.49 reproduced by the off-arm), which exceeds the CV target 114.49 -- confirming and over-explaining the CV +2.7% lead attribution; q40 TP2 +5.39% with no regression; PPL delta exactly 0; the historical GREEN trio (7f8fcbbd/4b5bc3be/d52628b1) is byte-intact in every cell where it applies; the 6 seal divergences are all on no-historical-reference cases, sit inside the already-accepted stacked drift envelope (tp1-UD prefix-KL 0.0114), and match the precedent (P5/q8rp) under which non-byte-neutral but PPL-neutral numerics were adopted via re-seal.**

Caveats for the human decision:
- The commit-message claim of byte-neutrality is **falsified at full seal length** (holds only <=32-token greedy). If byte-neutrality is a hard adoption requirement, this is a DO-NOT-ADOPT until the tensor-def/long-pp token flips are explained at kernel level.
- Re-seal candidates (the on-arm numerals): tp1-ud tensor-def `<id>`, tp1-q40 r02 `<id>`, tp2-ud tensor-def `<id>`, tp2-ud r03 `<id>`, tp2-q40 r02 `<id>`, tp2-q40 r03 `<id>`.
- The 99.89 t/s on-arm outlier was contention from the bisect agent's GPU sessions overlapping my lock window (their lock discipline differs); worth reconciling lock files between agents before the next paired campaign.

## Artifacts
- Box: `rig/she-fusion-tp2/` -- `report.md` (this file), `results/{tg256.txt,ppl.txt,C-*.json,A-*.json,B-*.json}`, `logs/{campaign.log,<cell>.log,<cell>.driver.log}`, `witnesses/*.txt` (env + cmdline + hip_lib + she-fuse banner per leg), runner `rig/she-fusion-tp2-run.sh`.
- Campaign: 17 legs, all SESSION-DONE, TERM teardown, `gpu.lock` released, 3 cards 0% at end. Nothing pushed to gitea; no adoption executed; no builds modified.
