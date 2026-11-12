# Task G3 ledger: floor-0 forced-W4A4 review closure + AR-wire A/B on the rq production entry

Date: 2026-11-12 (PST). Branch INT4-noquant-ksplit (fork docs tree /root/wt-ksplit).
All GPU legs under flock -w 600 /root/gpu.lock, RAM gate 12 GB, free-GPU window
checks, TERM-only teardown with PID poll (no kill -9, ever). Drivers:
/root/taskG3-driver.sh (A legs), taskG3-driver2.sh + taskG3-driver3.sh (B legs),
taskG3-driver4.sh (B4 perf re-runs). Raw artifacts: /root/taskG3/.

## A. floor-0 forced-W4A4 (GGML_CUDA_W4A4_MIN_TOKENS=0, env-only) -- REVIEW EVIDENCE

Pre-existing evidence (taskF; docs/KLD-LADDER-w4a4-forced.md; RESULTS-microbench-addendum.md
F5 section; parked as REVISIT in commit d74c9e900f):
- Perf (microbench, np1 d4, 8-prompt suite): tg256 median 45.4 t/s vs F4q 36.6 (+24%)
  and F1/F2 37.8-38.0 (+20%); acceptance -3%; pp flat. The win is target-side:
  verify-batch time -25% at narrow width; drafter unchanged.
- Wide-width anchor protocol (16 chunks, ctx 2048, TP2 vs q8_0-16.kld): floor-0 KLD
  0.065577 == floor-ON 0.065577, byte-identical; the wide ladder is structurally
  blind to F5 (2048-token chunks sit above the 128 floor in both arms).
- Endpoint ladder vs Q8 ref (narrow width): prefill KL 0.684 (floor-ON) -> 1.357
  (floor-0); stays BELOW the sibling rq engine's own 1.455; greedy first-decode
  token changes (S2 direct floor-ON vs floor-0 prefill KL 0.864).
- Provenance correction: the 0.0499 anchor is rq-engine plain Q4_0, NOT a
  fork-sealed number; fork sealed repack = 0.0656 (L1n control 0.0490).

New legs this session (fork engine /root/int4-noquant-wt build, Q4_0 + DFlash2-Q4_0
drafter d4, TP2, w4a4-env + ALLREDUCE=internal REPACK_KQUANT=0 GDN_COLTHREAD=1):

A1 perf A/B (natural-suite harness /root/p2-prefill/bench.py, draft_n_max 4,
tg256 x3 x 8 prompts + pp4096 x1):
  floor-ON  (MIN_TOKENS=128 default, witnessed): tg median 40.44 t/s,
            accept_rate 0.3489, mean accept len 2.396, pp median 550.5
  floor-0   (MIN_TOKENS=0 explicit, witnessed): tg median 45.89 t/s (+13.5%),
            accept_rate 0.3320 (-4.8%), mean accept len 2.328, pp median 544.9 (-1.0%)
  per-prompt tg delta: code +21.4%, list +36.8%, dialogue +18.0%, tech +13.3%,
  mystery +9.8%, scifi +7.7%, econ -2.8%, math -1.5%.

A2 narrow-width teacher-forced KLD (llama-perplexity, 16 chunks, b 2048, -ub 64
i.e. below the 128 floor; base = floor-ON logits capture, candidate = floor-0):
  Mean KLD 0.040019 +/- 0.003138 nats (rc=0 both legs, banners witnessed).
  This is the served-config delta of floor-0 vs the sealed floor-ON config at the
  operating point the floor protects. It sits BELOW the 0.0653 KLD band production
  already accepts for the W4A4 class, and below the fork sealed-vs-q8 distance.

Verdict (floor-0): PASS on the distributional gauntlet.
- KLD gate: KL(floor-ON -> floor-0) = 0.0400 at narrow width < 0.0653 accepted
  band; wide-width anchor byte-identical (0.0656).
- Acceptance gate: -4.8% acceptance (0.3489 -> 0.3320) with net tg +13.5% --
  the spec-decode throughput is measured end-to-end and is positive.
- Caveat on record: endpoint greedy seal breaks (first-decode token changes;
  KL-vs-Q8 doubles at narrow width) -- the gauntlet is distributional, not
  byte-equality, so this does not block, but it is the owner's throughput/quality
  call. Config is env-only and reversible per fleet entry.
Recommendation: adopt as per-entry env toggle (GGML_CUDA_W4A4_MIN_TOKENS=0) on
dense Q4_0 entries with DFlash2; keep floor 128 default elsewhere. Unpark from
REVISIT.

## B. AR wire on the rq engine (GGML_NCCL_AR_WIRE: q4_K default vs f16)

Context: the ruquant engine moves prefill-sized all-reduces as q4_K blocks by
default since 2026-11-04 (ruquant-repo REPORT-2026-11-02); GGML_NCCL_AR_WIRE=f16
restores the 16-bit ring. Production benchmark entry = rq-mi50-tp2-q38-27b-w4a4
(port 8612; Qwen3.8-27B-W4A4-OST-GPTQ-wg64-rb4 + DFlash2-W4A4H-drafter-rb4,
draft 7; 94.2 t/s class: /root/ruquant-bench/redo/legR-r1.log "all accept len
4.25  94.2 t/s  45.1 ms/step"). The pending item was: this entry was never
benchmarked wire-on vs wire-f16 on the current box.

B2 KLD vs q8_0-16 anchor (llama-perplexity 16 chunks TP2; perplexity quantizes
every token = worst case for the wire):
  q4_K wire (default): Mean KLD 0.064212 +/- 0.003325 (rc=0)
  f16 wire:            Mean KLD 0.065291 +/- 0.004052 (rc=0; bit-exact
                       reproduction of the 0.0653 anchor)
  -> the quantized wire shows NO measurable KLD cost at the anchor protocol even
     in the every-token worst case (delta within noise).

B4 perf A/B on the production entry (natural-suite bench.py draft 7, interleaved
q4K,f16,q4K,f16, clean 2-GPU windows 04:19-04:46, no co-tenants):
  arm            tg median t/s   pp median t/s   accept_rate  mean accept len
  q4K (default)  47.32 / 50.35   645.0 / 647.0   0.2033       2.423
  f16            55.77 / 55.83   635.7 / 634.3   0.2505       2.754
  (pair 1 / pair 2 shown; acceptance is deterministic temp0/seed42 and identical
  within arm across pairs; the q4K per-prompt acceptances reproduce legR-r1
  exactly, e.g. econ 0.2477, code 0.8158.)
  Engagement witness: exit line "ggml_cuda: 20320 quantizable all-reduces" on both
  q4K legs, "0 quantizable all-reduces" on both f16 legs.
  Reading: the wire's prefill noise lands in the KV and LOWERS DFlash2 acceptance
  0.2505 -> 0.2033 (-18.8% rel), costing ~14% of spec tg (55.8 -> 48.8 mean of
  medians), while the pp gain on THIS box is only +1.7% (646.0 vs 635.0).
  KLD is a wash (0.0642 vs 0.0653): the gauntlet fails on ACCEPTANCE, not KLD.

Verdict (AR wire): REJECT the quantized prefill wire on this entry/box.
Set GGML_NCCL_AR_WIRE=f16 on the rq production entry (port 8612 class): +14% spec
tg, +23% acceptance for -1.7% pp. The engine default (q4_K since 2026-11-04) was
justified by box-1 numbers (pp +15-16%, "DFlash2 unchanged +-3%") that do NOT
reproduce on mi50 with the current entry; the historical pair below is kept for
provenance only.

Historical same-session pair (REPORT-2026-11-02, old 27B, box 1, TP2):
  pp512 793 (q4K) vs 689 (f16) = +15%; pp2048 834 vs 721 = +16%;
  tg64 and DFlash2 unchanged within +-3%; DFlash2 acceptance 4.28 vs 4.27;
  KLD prompt-only 0.0491 vs 0.0481; every-token 0.0537 vs 0.0481.

Ops notes:
- Engagement witness: server-exit line "ggml_cuda: N quantizable all-reduces"
  (N=2032 on the q4K-wire leg, 0 on the f16-wire leg).
- driver3 B1 legs (03:35-03:38) were killed by an external SIGINT ~40-60 s after
  server start (cause unidentified; second TERM forced teardown of a stuck
  cleanup); B4 legs add a BENCH-DONE gate + one retry + stable-window check.
- B2-f16 first attempt OOM'd on co-residency (FTQ38 held GPU1 mid-load); clean-window
  retry succeeded.
- KLD legs tolerate co-residency (deterministic); perf legs wait for a stable
  2-free-GPU window (two consecutive checks 60 s apart) before taking the lock.
