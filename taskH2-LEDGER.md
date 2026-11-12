# taskH2 LEDGER - (A) M3 i8narrow corruption re-run on the stream-fixed build; (B) width-8 nc verify warp A/B

Date: 2026-11-12 (PST). Box: mi50. Branch: INT4-noquant worktree /root/int4-noquant-wt.
No pushes; local commits only. Serving legs under flock -w 600 /root/gpu.lock, TERM-only
teardown, TheRock runtime, per-arm files. Siblings H1/H3 concurrent - all GPU touches flocked.

## A. M3 CORRUPTION RE-RUN - RESOLVED: the stream bug was the root cause

Build: /root/int4-noquant-wt/build @ 8054bbd770-era (i8narrow caller-stream fix present;
tree later committed tip 5a918c5576 is ledger-only). libggml-hip.so md5 22397e8cf509d6028b34888ef5e079cc
(M3's corrupt build was 496dde34...). md5 bracketed before/after the run: stable.
Model: bartowski Qwen_Qwen3.5-35B-A3B-Q4_0.gguf (20836243264 B), GPU2 tp1, production-env +
GGML_TP_AR_MTP_SAFE=1 - the exact taskM3 protocol (m3_i8narrow_ab.sh structure reused).
Driver /root/taskH2/m3_rerun.sh, artifacts /root/taskH2/m3-rerun/.

### Verdict

Corruption GONE. All gate-ON arms emit coherent temp-0 text ("...in a world full of wonders
and mysteries, there was a young adventurer named L...") with the [I8NARROW] banner present
(4/4 smoke arms coherent: on-1, on-2, on-nosumw; gate-OFF controls coherent too). The default-
stream launch (8054bbd770) was the root cause: the narrow-case kernels ran on the legacy default
stream while the activation quantizer and all consumers ran on the context stream - an
unsynchronized read of freshly-quantized hi/lo/ad planes.

Suspect 2 (SUMW plane offsets) is CLOSED, code + empirically:
- Code: rp_traits<Q4_0>::geom computes dplane_off = ne1*rs UNCONDITIONALLY; the SUMW plane is
  appended AFTER the d plane (swplane_off = ne1*rs + ne1*n_sub*2). The i8narrow kernel never
  reads the SUMW plane at all - it recomputes the signed weight sum inline from the raw qs
  uint4 (i8n_sumw_signed). The M3 gate-site comment "[qs|d] planes" is accurate; the external
  reviewer was right that the offsets are independent of the SUMW gate.
- Empirical: gate-ON with GGML_CUDA_W4A4_SUMW=0 (arm on-nosumw) is equally coherent.

### Re-measured perf (M3's numbers were measured on a corrupted model - invalid)

pp4096 x3 + tg256 x3 per arm (mtp_bench.py, temp-0 seed-42), arms interleaved on/off/on/off:

| arm   | pp4096 med t/s | tg256 med t/s | smoke     |
|-------|---------------:|--------------:|-----------|
| off-1 | 1119.1         | 108.40        | coherent  |
| on-1  | 1114.7         | 59.17         | coherent  |
| off-2 | 1118.5         | 107.80        | coherent  |
| on-2  | 1114.3         | 59.36         | coherent  |

Medians: off pp 1118.6 / tg 108.02 (matches M3's off baseline 1119.056/107.563 - build-era
drift <0.5%); on pp 1114.5 / tg 59.21.
Delta gate-ON vs OFF: **pp -0.37%, tg -45.19%**.

M3's apparent "+19.5% pp / +15.6% tg" was entirely the collapsed-routing artifact its own
finding #5 predicted: the corrupted hidden states starved the router so nearly all 256
experts were skipped. With the stream fix the model computes real MoE work and the true
i8narrow cost is visible: pp neutral (the narrow window is decode-only, ne11<=8), tg -45%.
Mechanism (inferred, not profiled): the kernel is latency-bound at decode widths - one lane
per row, KSPLIT=1, so each lane walks all K blocks in a serial dependent dp8a chain, while
the stock repacked mmvq path splits K across warps. M3's "do NOT enable GGML_CUDA_I8NARROW
on Q4_0 MoE serving" verdict STANDS, now on clean numbers - and the gate stays default OFF.

## B. WIDTH-8 NC VERIFY WARP A/B (taskG5 fix candidate #3)

### Setup

Build: /root/int4-noquant-wt/build rebuilt with commit 13482c3353 - new env gate
GGML_CUDA_RP_NC68_WARPS (default 1 = task E2 stock-matching table; =2 forces the 2-warp
geometry at nc widths 6-8). NOTE provenance: the build was made from a dirty tree whose
diff is exactly commit 13482c3353. Phase B binary md5s: llama-server 691394cc..., libggml-hip.so
08c41b4a...; bracketed stable across all legs. Gate default-OFF: 1-warp arm dispatch is
byte-identical to the pre-gate build.

Config: taskG5 DFlash2 stack - Qwen_Qwen3.6-35B-A3B-Q8_0.gguf target + DFlash2-Q8_0 drafter,
TP2 GPU1+2, -sm tensor, --spec-type draft-dflash --spec-draft-n-max 7 (verify width 8),
w4a4-env + ALLREDUCE=internal REPACK_KQUANT=0 GDN_COLTHREAD=1, PLUS the G2-adopted un-gating
env pair GGML_CUDA_REPACK_Q8_0=1 GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0 so the repacked nc family
actually serves the width-8 verify (witnessed: "REPACK_Q8_0 gate ACTIVE (mode 1)" +
"repacking q8_0 tensors" banners; the G5 legs ran mode 0 = canonical, so the nc kernel was
NOT on the G5 verify path - it is here).
Arms: w1 (gate unset) vs w2 (GGML_CUDA_RP_NC68_WARPS=2, banner witnessed in every w2 server
log, absent in every w1 log). Clean arms interleaved w1,w2,w1,w2 then instrumented w1-p,w2-p
(GGML_DFLASH_PROF=1 LLAMA_SERVER_TIMING=1). Driver /root/taskH2/w8warp_ab.sh (+ w2p_retry.sh:
first w2-p attempt hit LOCKFAIL - sibling held gpu.lock >600 s - retried at flock -w 1800).
Artifacts /root/taskH2/w8warp/.

### Results

Clean arms (bench.py natural suite, tg256 median of 8 prompts x 3 reps, temp-0):

| arm  | tg256 t/s | accept | mean len |
|------|----------:|-------:|---------:|
| w1-1 | 60.19     | 0.2362 | 2.6535   |
| w2-1 | 58.85     | 0.2361 | 2.6530   |
| w1-2 | 58.92     | 0.2362 | 2.6535   |
| w2-2 | 59.30     | 0.2361 | 2.6530   |

tg medians: w1 59.55, w2 59.08 -> **-0.80% (no gain; within inter-leg noise)**.
pp medians 1888.7 vs 1888.9 (prefill never enters the nc window - sanity).

Verify ms at width 8 (instrumented legs, tgt-decode width=8 bucket):

| arm  | n     | mean ms | p50 ms | p90 ms |
|------|------:|--------:|-------:|-------:|
| w1-p | 3129  | 29.265  | 27.398 | 30.810 |
| w2-p | 3141  | 28.687  | 27.201 | 30.582 |

Delta: mean -0.58 ms (-2.0%), p50 -0.20 ms (-0.7%). G5's d7-p reference (canonical path,
different build) was 28.89/27.35 - same band. Per G5's exchange rate (1 ms at width 8 =
+2.2% tg at d7) the best-case reading of this delta is ~+1.3% tg; the clean-arm tg says
-0.8%. The verify delta is at the noise floor.

### Numeric divergence (documented, per the policy-exception requirement)

- Fixed greedy probe (temp-0 seed-42, 64 tok, raw /completion): byte-IDENTICAL across arms
  (w1==w2, rep-stable both ways). A 64-token probe does not certify byte-neutrality at seal
  length - see next lines.
- Pooled acceptance over the natural suite differs DETERMINISTICALLY: 0.2362/2.6535 (w1, both
  reps) vs 0.2361/2.6530 (w2, both reps) - the 2-warp reduction order (blocks_per_iter 32 vs
  16, cross-warp shared partial) perturbs verify logits at the ulp level and flips borderline
  accepts.
- Per-prompt acceptance diverges where trajectory amplification kicks in: math-reasoning mean
  len 3.183 -> 2.575 (-17.3% t/s), dialogue 1.814 -> 1.989 (+11.2%). Single ulp flips early
  in a chain re-roll the whole accept trajectory.
=> 2-warp at width 8 DOES break bit-match-to-stock, exactly as d67cd96f0f predicted.

### Verdict: 1-warp is already optimal at width 8 - NO policy exception needed

2-warp buys no measurable verify time (<=0.6 ms/step, below noise; clean tg -0.8%) and costs
bit-identity. The stock-matching 1-warp geometry stays canonical. Mechanism (inferred, not
profiled): the nc family launches one block per output row (grid = ne01), so CU occupancy is
already saturated by blocks; widening the block to 2 warps only adds a cross-warp reduce
without adding useful in-flight work. The A/B gate GGML_CUDA_RP_NC68_WARPS stays in-tree
(default OFF, byte-identical when unset) for cheap re-A/B if the kernel or shapes change.
This closes taskG5 fix candidate #3 as REFUTED (no recovery available at width 8 via warp count);
candidate #1 (adaptive depth) remains the live lever for the d7 regression.

## Rule compliance

All serving legs under flock -w 600 /root/gpu.lock (retry leg -w 1800 after a sibling LOCKFAIL);
GPU2 tp1 for Phase A (M3 protocol parity), GPU1+2 TP2 for Phase B; TERM-only teardown, curl
/health polling, per-arm files; TheRock runtime witnessed via /proc/maps; RAM guard >=12 GB;
binary md5 brackets stable within each phase; no llama-swap touch; commits ASCII, no pushes.

## Artifacts

/root/taskH2/m3-rerun/ (Phase A: server logs, smoke jsons, witnesses, environ proofs,
h2-*.jsonl benches, bin-md5 brackets, arm-status.txt); driver /root/taskH2/m3_rerun.sh.
/root/taskH2/w8warp/ (Phase B: per-leg logs/witnesses/env-witnesses/greedy jsons/reps agg,
w{1,2}-p.agg-prof.json instrumented phase summaries); drivers w8warp_ab.sh, w2p_retry.sh.
Commit 13482c3353 (GGML_CUDA_RP_NC68_WARPS gate, default OFF).
