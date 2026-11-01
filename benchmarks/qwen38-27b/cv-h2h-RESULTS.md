# CV vs mx-fork head-to-head -- Qwen3.8-27B Q4_0, TP2 (GPUs 0,1), 2026-10-09

Castagna Veloce (benpeterson40) vs our mx-llama.cpp fork (build-rccl, prod binary).
Server-based A/B: `-c 16384 -b 4096 -ub 1024 -fa on -sm tensor -np 1`, pp ~ 1k-token prompt
(n_predict 1), tg = 10 distinct prompts x 128 greedy tokens, 2 interleaved rounds (cross-engine
order flipped), fork arms with standard env (CUSTOM_AR, TheRock runtime, -lm dio), CV arms on
system ROCm with GGML_CUDA_VIRTUAL_PER_GPU=1. MTP arms: -md mtp-Qwen3.8-27B-Q4_0.gguf,
--spec-type draft-mtp --spec-draft-n-max 2. Power caps 225 W (both engines equal).
Raw artifacts: mi50:rig/cv-headtohead/*.json + h2h-v2.log; driver h2h-driver-v2.sh.

| arm | pp t/s | tg t/s | accept |
|----------|------------------|--------|--------|
| cv | 529.0 [530/528] | 44.4 | - |
| cv-mtp | 491.9 [492/492] | 65.3 | 73.9% |
| fk-stock | 368.3 [363/374] | 42.7 | - |
| fk-repack| 383.9 [386/382] | 46.1 | - |
| fk-mtp | 327.0 [321/333] | 54.2 | 74.0% |

Deltas: cv vs fk-stock pp **+43.6%** / tg +3.9% - fk-repack vs fk-stock pp +4.2% / tg **+7.9%** -
cv-mtp vs cv pp -7.0% / tg **+47.1%** - fk-mtp vs fk-repack pp -14.8% / tg +17.5% -
cv-mtp vs fk-mtp pp +50.5% / tg **+20.7%** (acceptance EQUAL 73.9% vs 74.0%).

## Findings
1. CV pp advantage on our hardware is +43.6% (not their 70-75%: our fork baseline is already
 ~+25-30% over the upstream stock they compared against -- the gains partially stack).
 With virtual devices disabled and dense 27B (no MoE/i-quant kernels in play), the residual
 is their TP comms (BAR peer-write allreduce, pair layout) + meta-buffer/scheduler work.
2. CV raw decode is only +3.9% over our fork -- their engines' tg comes from MTP overhead
 engineering, not faster decode kernels. Equal acceptance (74%) + +20.7% decode under MTP
 = pure per-step overhead delta (draft graph reuse, catch-up merge, verify-shaped chunk).
3. fk repack tg +7.9% on 27B replicates the 9B +7.2% (BENCH 52) -- PR1 tg-side claim now has
 two model sizes. pp +4.2% positive at ub1024 (27B) vs parity (9B) -- "pp no worse" holds.
4. MTP tax: fork pays -14.8% pp for a resident drafter; CV pays -7.0% (their graph-reuse +
 ckpt-last-verify). Port targets: LLAMA_CKPT_LAST_VERIFY, MTP drafter graph reuse,
 catch-up-in-draft-step, LLAMA_MTP_DRAFT_VOCAB (FR-Spec; = our Strata draft-vocab subset item).

## W4A4 addendum (same rig/protocol, 2026-10-09 late)

| arm | pp t/s | tg t/s | accept |
|---|---|---|---|
| fk-w4a4 (gate1, GGML_CUDA_W4A4_DOT8=1) | **414-416** [416/...] | **46.5** | - |

- gate1 vs fk-repack: pp **+8.6%**, tg parity (+0.8%). vs CV: gate1 still -21.7% pp, tg +4.7%.
- 27B/TP2 dilutes gate1's pp gain hard (9B single-GPU +31.8%, 9B TP +11.3%, 27B TP2 +8.6%) --
 consistent with 53's trend; CV's pp lead survives W4A4. Their decode lead does not: our
 gate1 tg 46.5 ~ CV plain 44.4.
- gate1+MTP at 27B: NOT measured (see transient below) -- open item.
- TRANSIENT (watch item, 1 occurrence): first v3 attempt hung the server completion queue on
 the first tg request after a 1k-token pp (request never reached slot processing; TERM wedged
 briefly, cascaded readiness failures into the next arms). NOT reproducible in 3 probes
 including the identical pp->tg sequence (probe2/probe3 all pass, 34 s). If it recurs:
 rig/cv-headtohead/{probe,probe2,probe3}.log + h2h-v3.log are the evidence trail.

## v4 full matrix (2026-10-10): Part A perf re-run + Part B accuracy (PPL/KLD, 9B ladder)

### Part A -- 27B TP2 perf, 7 arms x 2 rounds
| arm | pp t/s | tg t/s | accept |
|---|---|---|---|
| cv | 528.3 [529/527] | 43.7 | - |
| cv-mtp | 355.8 [223/488 OUTLIER r1] | 63.6 | 73.9% |
| fk-stock | 352.9 [350/356] | 41.9 | - |
| fk-repack | 348.1 [359/337] | 45.4 | - |
| fk-w4a4 | 367.8 [415/321] | 45.8 | - |
| fk-mtp | 317.7 [312/323] | 47.8 | 70.3% |
| **fk-w4a4-mtp** | **381.6 [378/385]** | **51.9** | **75.3%** |

Noise caveats: v4 ran ~04:00 after a full day of benching at 225 W; cv-mtp r1, fk-w4a4 r1/r2,
fk-repack show outlier rounds (consistent with mid-run interference). Anchor legs reproduce
v2/v3 within ~1% (cv 529/527; fk-mtp 312/323). Robust reads only:
- **gate1+MTP COMPOSES**: tg 51.9 (+13.3% over gate1-plain), acceptance 75.3% -- HIGHER than
 repack+MTP's 70.3% on the same draft. W4A4 does not degrade acceptance here (contrast 9B
 finding that gate1 degrades it). Clean 378/385 spread.
- CV pp lead holds (~+44-50% vs fork stock; +44% vs gate1 on clean rounds).
- repack tg +8.5% / gate1 tg parity replicate v2.

### Part B -- 9B accuracy ladder (wikitext-2, c512/b2048/ub512/ch128, KLD vs fork F16 base)
| leg | PPL | KLD |
|---|---|---|
| cv-bf16 | 8.0239 +/- 0.115 | **0.000000 +/- 0.000000** |
| cv-q8 | 8.0365 +/- 0.115 | 0.001626 +/- 0.000370 |
| cv-q4 | 8.4570 +/- 0.122 | 0.069180 +/- 0.001729 |
| fk-bf16 | 8.0239 (KLD leg) | - (is the base) |
| fk-q8 | 8.0358 +/- 0.115 | 0.001189 +/- 0.000058 |
| fk-q4-stock | 8.4514 +/- 0.121 | 0.068689 +/- 0.001702 |
| fk-q4-repack | 8.4505 +/- 0.121 | 0.068399 +/- 0.001726 |
| fk-q4-gate1 (TEST bin ) | 8.8760 +/- 0.128 | **0.122150 +/- 0.002028** |

Findings:
- **CV bf16 is EXACTLY the fork bf16** (KLD 0.000000; PPL identical to 4 decimals). Cross-engine
 numerics identity -- CV's kernels do not corrupt the model.
- CV Q8/Q4 parity with fork within noise (KLD same class, PPL deltas << CI).
- Fork ladder reproduces the record: stock 0.0687 (46: 0.0683-0.0687), repack identity
 0.0684 (46b), **gate1 KLD 0.122150 vs 52a's 0.122170 -- 2e-5 apart across different binaries**.
- gate1 c512-slice PPL penalty tonight: +0.42 (+5.0%) vs stock on the matched slice.
- bf16 PPL 8.0239 vs 46's 7.9504 is a CHUNK-SLICE difference (46 PPL legs used --chunks 256,
 tonight 128; both engines identical at 128) -- matched-chunks rule confirmed, not drift.
- Prod binary (build-rccl) has NO multiseq fix -> gate1 KLD harness (n_seq=4) faults there
 (rc=134, mmq_gemm_repacked<U4ACT=true> illegal access, exactly 47d). Gate1 KLD/PPL legs
 must run on rig/build-w4a4 (, has the fix). The fix rides PR1 as commit 2.


