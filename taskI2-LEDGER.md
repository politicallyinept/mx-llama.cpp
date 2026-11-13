# taskI2 LEDGER - DFlash-OFF plain-decode parity: fork vs ruquant engine, real-vs-real

Date: 2026-11-12 (PST). Box: mi50 (3x MI50 gfx906, TheRock runtime). No pushes; local commit only.
Parent correction applied: the "123B TP4 8.4 t/s" anchor was a parent error and is NOT used
anywhere; all numbers below are this task's own runs (plus cited same-box taskG4/taskH3/redo
artifacts for context).

## 1. Question

With DFlash2 disabled, does our fork match ruquant's raw decode? Published anchor: their
DFlash2-on 27B table (50.8 t/s TP2 at equal acceptance vs our 37.9). Open question: is any of
that gap plain-decode engine speed, or is it all the speculative path?

## 2. Comparison models and engines

Primary: Qwen3.8-27B Q4_0 (unsloth) TP2 -- the target of the DFlash2 27B stack.
  sha256 5eb1de39e9b42aa7f99f1a4fe149d1952cc8103c698cf380b15a43d47e72b7c7
  (their card's "Q4_0" comparison row is this same quant; cross-engine same-model is only
  possible on Q4_0 -- their W4A4 dtypes do not load on the fork, GGML_TYPE_COUNT=43.)
Secondary: Qwen_Qwen3.6-35B-A3B Q8_0 MoE TP2 (rq engine supports qwen35moe; verified load).
  sha256 bc57de096ec30e45773aa86103f35a8a32cea01f2193a53a3c636d618e4434e9
Drafter (on-arm context only): Qwen3.8-27B-DFlash2-Q8_0.gguf, sha256 c18e800daedc59ca68fd13b6a856d795746af6d399a9279ac6a277d1d422f87e.

Engines:
- fork: /root/int4-noquant-wt/build/bin/llama-server, "0.6.0-dev (build 11899, stamp 5a918c5576)",
  tree HEAD 80389528db (tip 2208af6a04 + taskI1 commit), binary mtime 2026-11-12 14:43.
  llama-bench rebuilt fresh 2026-11-12 18:18 (the Nov 11 binary was 1 build stale; r1 bench legs
  used the stale one, r2/r3 legs the fresh one).
- ruquant: /root/ruquant-src/build-82c73b5/bin/llama-server, "0.6.0-dev (build 10, commit 82c73b5)"
  (taskG4 build; llama-bench target built this session).

Protocol per arm: server leg (natural suite via /root/p2-prefill/bench.py: 8 prompts x 3 reps
tg256, temp0 seed42 ignore_eos, cache_prompt=false, warmup x3 + probe_pp) + llama-bench leg
(-p 512,8 -n 128 -r 3, the published protocol). Fork env: source /root/production-env.sh +
HSA_KERNARG_POOL_SIZE=67108864. Rq env: their published recipe (GGML_CUDA_DISABLE_GRAPHS=1
HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_META_TOKEN_GRAPH=0 GGML_META_SPIN=2000000) + TheRock
LD_LIBRARY_PATH. All legs: flock /root/gpu.lock -w 1800 (sibling I1 contention observed and
absorbed), TERM-only teardown, TheRock runtime witness (no apt libamdhip64 mapped), sha256
witnesses, port/quiet checks. Drivers+logs: /root/taskI2/{drive.sh,drive35.sh,rerun.sh}.

## 3. Results: 27B Q4_0 TP2, DFlash OFF (real-vs-real, same box, same model, interleaved)

Natural suite tg256 median (primary metric; tight reps):
| leg  | fork  | rq    |
| r1   | 46.84 | 43.34 |
| r2   | 47.21 | 43.38 |
| r3   | 46.99 | -     |
clean-pair median: fork 46.99 vs rq 43.36 -> fork +8.4%

llama-bench (pp512 / tg128 t/s):
| leg            | fork pp512 | fork tg128   | rq pp512 | rq tg128     |
| r1             | 540.72     | 44.79 +-2.29 | 519.89   | 38.71 +-5.01 |
| r2             | 582.71     | 47.10 +-0.39 | 519.64   | 44.18 +-0.20 |
| r3 / on-arm    | 573.83     | 43.20 +-3.92 | -        | -            |
| (fork-on-r2)   | 581.85     | 47.23 +-0.47 | -        | -            |
pp512: fork 573.8-582.7 vs rq 519.8 -> fork +10-12%. tg128 noisier both engines; means
fork ~43-47 vs rq ~38.7-44.2 -> fork +4-8%. No metric shows rq ahead.

VERDICT (27B, DFlash off): MATCH -- no fork deficit. Fork AHEAD: +8.4% natural tg,
+10.4% pp512, +4-8% tg128.

## 4. Cross-check vs published (their card, local copy /root/bench-quants/ruquant/README.md;
live HF card is gated 401 without token)

Published Q4_0 TP2 plain decode (llama-bench tg128, their box 4x MI50 ROCm 5.7): 39.7 t/s.
Our local run of THEIR engine on this box: 38.71 / 44.18 (mean 41.4) -> +4.4% vs published,
inside the 4-6% day/box drift band (taskG4 replicated their DFlash claim at -1.7%).
Their engine reproduces on this box; our fork then beats that engine same-box same-model.

## 5. Fork DFlash-on vs DFlash-off, same model (context)

Natural suite tg256 median, 27B Q4_0 TP2, drafter Q8_0, n_max 4 (fork's best):
- off: 46.99 (this task, clean)
- on:  38.10 (fork-on-r2, clean; contaminated r1 36.56; redo legF history 37.85/38.00)
- DELTA: DFlash-on is -18.9% vs plain decode on our fork. Acceptance bit-identical across
  all on legs (draft_n 10185, accepted 3546, rate 0.3482, mean accept len 2.39).
Even crediting taskH3's drafter-pinning win (+14.2% at n_max 4), on would be ~43.5 -- still
below plain decode. On our fork, DFlash2 on 27B Q4_0 TP2 is net-negative.

Contrast their card: Q4_0 plain 39.7 -> Q4_0 + DFlash2 60.5 (TP2) = +52% from spec decode on
their engine. The published on-gap (50.8 vs our 37.9) is therefore ENTIRELY the speculative
path (verify-step cost + acceptance), not raw decode: our plain decode is already +8.4% over
their engine same-box.

## 6. Secondary: 35B Q8_0 MoE TP2, DFlash OFF

Natural suite tg256 median:
| leg  | fork   | rq   |
| r1   | 106.81 | 91.29 |
| r2   | 88.57* | 91.52 |
| r3   | 106.83 | -    |
* r2 invalid: env leak (see 8); clean pair 106.81/106.83.
fork 106.8 vs rq 91.4 -> fork +16.9%. VERDICT (35B, DFlash off): MATCH, fork ahead.
pp4096 (bench.py): fork 2641 vs rq 1910-1914 -> fork +38%.
35B llama-bench legs failed (BENCH-RC=1): bench runs while server holds VRAM; 35GB Q8_0 x2
does not fit 2x32GB. Natural-suite numbers unaffected; no 35B llama-bench numbers reported.
Context: fork 35B DFlash-on (taskH3) rep-d4 69.18 / pin-d4 79.54 -> on is -35%/-26% vs plain.

## 7. Findings

1. DFlash-OFF parity: our fork MATCHES ruquant's raw decode on both program models and is
   ahead on every metric measured (27B: +8.4% natural tg, +10.4% pp512; 35B: +16.9% natural
   tg, +38% pp4096). There is no plain-decode gap to close.
2. The published DFlash2-on gap is 100% speculative-path: their DFlash2 adds +52% over their
   plain decode; our DFlash2 SUBTRACTS ~19% (27B) / 26-35% (35B) from ours. The fork's
   verify/draft per-step cost is the entire deficit; engine kernels for plain decode are not.
3. Their published plain-decode number replicates on this box with their engine (39.7
   published vs 41.4 local mean, +4.4%, within drift).
4. Serving implication: with the current fork DFlash2 stack, plain decode is the faster arm
   on both models; DFlash2-on should stay off on 27B Q4_0 TP2 until verify-step cost improves.

## 8. Issues / caveats

- Driver env leak (found mid-campaign): leg_rq exports leaked into later fork legs in the
  same shell (GGML_CUDA_DISABLE_GRAPHS=1 et al. -- the fork DOES read this var). Impact:
  27B fork-off-r2 and fork-on-r1 slightly affected (27B dense decode nearly immune: r3 clean
  46.99 ~= r2 47.21); 35B fork-off-r2 badly affected (88.6 vs 106.8 clean -- HIP graphs matter
  for MoE decode). Fixed with subshell isolation + clean reruns (fork-off-r3, fork-on-r2,
  fork35-off-r3); contaminated legs kept on disk, marked.
- Rq legs also inherited fork production env exports, but the only shared getenv site
  (GGML_CUDA_REPACK_KQUANT) is inert for Q4_0/Q8_0 models; rq reps reproduced each other to
  0.1% and replicated the published number, so rq legs are treated as valid.
- rq-off-r1 llama-bench tg128 noisy (38.71 +-5.01); r2 clean (44.18 +-0.20). Natural suite
  used as the primary metric for exactly this reason.
- fork-off-r1 llama-bench used the stale (Nov 11) bench binary; superseded by r2/r3.
- Live HF card gated (HTTP 401); local card snapshot /root/bench-quants/ruquant/README.md
  (shipped with the model files, same one taskG4 used) is the published reference.
- Ops: all GPU legs under flock /root/gpu.lock (waits up to ~4 min behind sibling I1),
  TERM-only teardown, pgrep -x llama-server empty after every leg, TheRock runtime witnessed
  on every leg, llama-swap untouched.
