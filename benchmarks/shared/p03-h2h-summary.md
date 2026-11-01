# P0.3 sanctioned cross-engine h2h -- fork (P0 combined) vs CV, Qwen models, llama-server protocol

Date: 2026-10-26 (PDT). Box: mi50, GPU1 only (`HIP_VISIBLE_DEVICES=1`), no code changes.
Engines: **fork** = `rig/build-p0f/bin/llama-server` (combined build, commit ; pp840 tp1 llama-bench sanity 1575.71+/-25.33 -- today's server-protocol fork MoE-Q4_0 pp of 1572.4 matches it). **CV** = `rig/cv-h2h-v5/build-cv/bin/llama-server` (commit , verified via banner; cv-h2h-v10/v9b contain artifacts only, no binaries).
Protocol: llama-server `/completion` only (llama-bench excluded per standing rule). Prompt: 827-token pp (paragraph x7, closest to 840 with whole repeats -- identical bytes both engines), tg prompt 15 tokens / n_predict 128. Per leg: 3 discarded warmups + 3 measured per metric. All numbers below are **tp1 single-card** -- the recorded references were TP2 warm-to-warm; do not mix protocols.

## Results (mean +/- sd of 3 measured)

| model x engine | pp t/s | tg t/s | protocol | witness |
|---|---|---|---|---|
| Dense 27B Q4_0 -- fork | 302.72 +/- 0.23 | 29.94 +/- 0.06 | no `-lm dio` (see deviations) | OK |
| Dense 27B Q4_0 -- CV | 304.44 +/- 0.19 | 30.08 +/- 0.32 | one-shot (see bug) | OK |
| Dense 27B UD -- fork | 339.57 +/- 0.18 | 24.40 +/- 0.03 | no `-lm dio`, `GGML_CUDA_GCN_KQ=8191` | OK |
| Dense 27B UD -- CV | 335.46 +/- 0.55 | 26.61 +/- 0.01 | one-shot (see bug) | OK |
| MoE 35B-A3B Q4_0 -- fork | 1572.36 +/- 1.91 | 92.31 +/- 0.08 | full flags incl `-lm dio` | OK |
| MoE 35B-A3B Q4_0 -- CV | 632.31 +/- 0.27 | 109.29 +/- 0.15 | full flags incl `-lm dio` | OK |
| MoE 35B-A3B UD -- fork | 1870.92 +/- 2.12 | 90.01 +/- 0.37 | full flags incl `-lm dio` | OK |
| MoE 35B-A3B UD -- CV | 1917.05 +/- 2.32 | 110.88 +/- 0.04 | full flags incl `-lm dio` | OK |

## Gap statement vs recorded references (references were TP2 warm-to-warm; today is tp1 -- different protocols, stated explicitly)

- **MoE Q4_0** (ref: fork led pp 587 vs CV 538, TP2): today tp1 fork leads pp decisively, 1572 vs 632 (+149%). CV still leads decode, 109.3 vs 92.3 (+18%).
- **MoE UD** (ref: CV led pp 588 vs fork 396 pre-REPACK_KQUANT=0, then 515 vs 587 after): today tp1 the pp gap is essentially closed -- fork 1871 vs CV 1917 (fork at 97.6% of CV, vs 88% in the post-REPACK reference). The combined build closed most of the UD pp gap as expected. Decode still favors CV: 110.9 vs 90.0 (+23%).
- **Dense 27B** (no TP2 reference quoted): pp is a wash -- Q4_0 CV +0.6%, UD fork +1.2%. Decode: CV +0.5% (Q4_0), CV +9.1% (UD).

## Findings / deviations (all recorded in leg artifacts)

1. **CV bug -- dense 27B tp1 aborts on the 2nd prefill** (`ROCm error: illegal memory access`, castagna-veloce ggml-cuda.cu:123, in `ggml_backend_sched_compute_splits`). Deterministic on request #2 regardless of flags: tested +/-`-lm dio`, +/-`-b 4096`, `-fa on/off`, +/-`GGML_CUDA_MOE_VEC`, `-ngl 99/999`, identical vs varied prompts (7 probes, `probe*-cv-*.log`). First prefill always OK and very stable (~305 t/s). CV's own guide (`docs/models/qwen3.8-27b.md`) only supports this model on TP2. MoE models are unaffected (6 identical prefills fine).
 -> **CV dense cells measured via one-shot protocol**: 1 request per fresh server (3 warmup + 3 measured, fresh load each). Samples extremely stable (sd <= 0.55) and consistent with fork warm numbers; raw samples in `cvdense-*-samples.jsonl`.
2. **`-lm dio` dropped from the dense pair (both engines)**: CV aborts on dense with dio too; dio is a no-op on a single GPU. Fork dense with-dio vs without-dio numbers differ <0.3% (302.38->302.72 pp; archived `*-dio-crash*` artifacts). MoE legs kept `-lm dio` on both engines per protocol.
3. CV env = launch2.sh cv case (`LD_LIBRARY_PATH` cv-bin+rocm-7.2.3, `LLAMA_CKPT_LAST_VERIFY=1`, `GGML_CUDA_MOE_VEC=1`), with `HIP_VISIBLE_DEVICES=1` per the GPU constraint. No fork gate vars on CV. Fork env = `w4a4-env.sh` + the 8 prescribed gates; `GGML_CUDA_GCN_KQ=8191` added on the dense-UD leg (witnessed).
4. Discipline held: flock rig/gpu.lock per leg, foreign-proc + free-g checks, llama-swap stopped for the sweep and restarted (active), VRAM back to ~10 MB/card on all 3 GPUs after every leg and at end. Dense pair fully re-run (both arms) after the CV dio crash, per re-run-the-pair rule.
5. Config witnesses: `/proc/PID/environ` snapshot + server log per leg (`*-environ.txt`, `*-witness.txt`, `*-server.log`). CV dense witnesses are from the one-shot measured servers (`cv-dense-*-pp-s4-server-environ.txt`, copied to `*-environ.txt`).

## Artifacts (mi50:rig/p03-h2h/)
`driver.sh` (main sweep), `cvdense-oneshot.sh` (CV dense workaround), `bench.py`, `calib.py`, `summarize.py`, prompts (`pp-prompt.txt` 827 tok, `tg-prompt.txt` 15 tok), per-leg `*-results.json` / `*-server.log` / `*-environ.txt` / `*-witness.txt`, `cvdense-*-samples.jsonl`, probe logs, `driver.log`, `rerun-dense.log`, `summary-table.md`, `summary.json`.
