# taskI1 ledger — q4_0-KV cache with H-rotation (INT4-noquant fork)

Date: 2026-11-12 (PST). Branch INT4-noquant, base tip 2208af6a04, gate commit 80389528db.
Build: /root/int4-noquant-wt/build-i1 (mi50-build.sh, TheRock amdclang++, src_sha 80389528db,
dirty_files=0, llama_server_md5 4a7edbf80f19b8daf7094610c2490243). All arms run on this ONE binary.

## 1. Prior-art recovery (what actually exists vs what the brief claimed)

- The brief pointed at "/root/ruquant-repo (upstream 82c73b5, local 7a08316)". WRONG PATH:
  /root/ruquant-repo has neither commit. The real tree is /root/ruquant-src, branch ruquant
  (HEAD 82c73b5 "Q8_0 narrow mat-vec on gfx906"), where BOTH cited commits exist:
  * 7a08316 "ruquant: W4A4 (int4 x int4) Qwen3.8-27B on gfx906, TP and DFlash2 speed work"
  * e509cee "q4_0 KV attention with v_dot8 for decode and verify on gfx906" - the int4-KV
    scaffolding: a gfx906 FA kernel computing directly on the q4_0 KV cache via
    v_dot8_i32_i4 (Q quantized per-32 to int4, K nibbles fed after one xor; env RQ_FA_DOT8).
    Its commit message states it computes "with mainline's Hadamard rotation of Q / K / V" -
    i.e. the ruquant int4-KV work is BUILT ON the upstream Hadamard KV rotation, confirming
    that rotation as the canonical "H-rotation" of the recorded design.
  * Recorded numbers from e509cee (one MI50, 64k depth): decode 44.7 -> 38.5 ms vs f16 cache
    (stock q4_0 132.4); verify 64.4 -> 59.6 ms; KLD vs UD-Q8_K_XL: q4_0+rotation
    0.0423-0.0434 vs f16 0.0416 vs stock q4_0 0.0442 - prior expectation: rotation closes
    roughly half the q4_0->f16 KLD gap.
  * The "TP4 306->271 us" figure is a gfx906 attention MICROBENCH label from that exploration
    (parent-confirmed), not a served-model config. No W4A4 123B dense model exists in this
    program (parent correction); validation stack is the 35B Qwen3.6 MoE Q8_0 TP2.
- The recorded exploration in THIS repo is README.md "What we tried and rejected":
  "Hadamard rotation (H32): simulated gains didn't survive the live path" - that was an
  ACTIVATION-side lever for the W4A4 GEMM, not KV. Different problem, does not condemn this one.
- THE MECHANISM IN OUR FORK: upstream llama.cpp's quantized-KV Hadamard rotation is already in
  the INT4-noquant baseline (src/llama-kv-cache.cpp): ggml_gen_hadamard() builds orthonormal
  Walsh-Hadamard matrices (H*H^T = I, res^2 == I); llama_mul_mat_hadamard() (llama-impl.h)
  applies them via ggml_mul_mat with GGML_HINT_SRC0_IS_HADAMARD; llama-graph.cpp build_attn()
  rotates q_cur/k_cur (and v_cur) BEFORE the cache write (build_cpy_k) and reads back the
  rotated, quantized K for the dot product - i.e. exactly the recorded design: rotate K at
  cache-write, Q-side rotation makes the dot product exact (q'*k' = q*H^T*H*k = q*k since H
  is orthonormal and applied to both sides). V rotation + exact output de-rotation also exist.
  The k_shift path (rolling-window requant) de-rotates, applies RoPE, re-rotates, requantizes.
- Upstream AUTO-ENABLES this whenever type_k/type_v are quantized and head_dim % 64 == 0,
  with only a kill switch (LLAMA_ATTN_ROT_DISABLE). That violates the fork convention that
  numerics-changing levers are env-gated and default-OFF.

## 2. Implementation (commit 80389528db)

src/llama-kv-cache.cpp ctor: rotation now requires explicit opt-in:
- LLAMA_KV_KROT=1   -> rotate K (the taskI1 headline lever)
- LLAMA_KV_KROT_V=1 -> rotate V (+ exact output de-rotation)
Default OFF for both: quantized KV stores raw q4_0 (upstream pre-rotation behavior).
LLAMA_ATTN_ROT_DISABLE still hard-kills everything. The DeepSeek lightning-indexer override
that forces rotation for DSA/GLM-DSA/DOTS3NOTE indexer caches is intentionally left ungated
(those caches require it; behavior unchanged for those archs).
Engagement witness: server/perplexity log line "llama_kv_cache: n_rot_k = N" (N=0 gate off,
N=256 for the 35B target head_dim 256, N=128 for the DFlash2 drafter head_dim 128).

Scope notes:
- RoPE ordering: qwen35moe applies IMRoPE to Q/K in the layer graph BEFORE build_attn, so the
  cache stores H*rope(k) and the dot product uses H*rope(q) - rotation is applied to the
  post-RoPE values that actually land in the cache. Correct by construction.
- DeltaNet/GDN recurrent layers (build_layer_attn_linear) have no KV cache at all; the
  rotation lives in the standard-attention build_attn path only. Structurally excluded.
- FA path: ggml-cuda fattn has Q4_0 K and Q4_0 V template instantiations (fattn.cu
  FATTN_VEC_CASES_ALL_D(Q4_0, Q4_0)); rotation is graph-level, FA just sees rotated q/k.
- TP2: rotation is per-head along head_dim, device-local; no cross-rank interaction.

## 3. Protocol

Stack: Qwen_Qwen3.6-35B-A3B-Q8_0.gguf TP2 (GPU1+GPU2, -sm tensor), DFlash2-Q8_0 drafter
d4 for the spec-decode legs. Driver /root/taskI1-drive.sh, outputs /root/taskI1/.
Arms: f16 (baseline, default KV) | q40 (q4_0 K+V, gate off) | q40krot (q4_0 K+V + LLAMA_KV_KROT=1).
- KLD: llama-perplexity wikitext-2-raw, -c 2048 -b 2048 --chunks 16 (taskF/taskV1 anchor
  protocol), base logits dumped from the f16 arm on the SAME binary (--save-all-logits),
  scored with --kl-divergence-base/--kl-divergence.
- Greedy probe: llama-cli temp0 seed42 n128 fixed prompt, per arm; sha + divergence vs f16.
- Perf: llama-bench -p 512 -n 256 -r 2 (tg256/pp512) per arm; DFlash2 tg via
  /root/p2-prefill/bench.py natural suite (tg256 x3 + acceptance) per arm on server legs
  launched via mi50-run.sh (production-env, TheRock runtime witnessed per leg).
- Ops: flock -w 3600 /root/gpu.lock per leg, TERM-only teardown, pgrep -x llama-server empty
  after each leg, refuse-if-exists per-arm files.

## 4. Results

### Interim (2026-11-12 ~18:35)

Protocol-determinism control (happy accident): the first base logits dump was mislabeled and
made with RAW q4_0 KV, so the first q40 scoring leg was q4_0-vs-itself:
- raw q4_0 vs raw q4_0: Mean KLD -0.000000 +/- 0.000000, Max KLD 0.000063 - the harness floor
  (uint16 base-logprob quantization, matches taskV1's +/-6e-5 control). Protocol is deterministic.
- rotated q4_0 vs raw q4_0 (same base): Mean KLD 0.011661 +/- 0.000269, Max 1.98;
  PPL(Q) 5.912044 vs 5.925301 raw q4_0 -> rotation IMPROVES PPL by 0.22% over raw q4_0
  (quantization error spread across channels, no anomaly-channel outlier). Saved as
  /root/taskI1/kld-q40krot-vs-q40raw.log.
True f16-base KLD legs + greedy + llama-bench all completed (driver2/driver4).

### DFlash2 tg baseline (f16 KV, DFlash2-Q8_0 drafter d4, natural suite, 24 tg256 reps)
- tg256 median 80.54 t/s; pp4096 median 1962.1 t/s; accept_rate 0.3581; mean_accept_len 2.4324.
  (/root/taskI1/f16.reps.jsonl.agg.json)

### Plain llama-bench pp512/tg256 (same binary, -r 2 aggregated, type_k/type_v witnessed in jsonl)
| arm | pp512 t/s | tg256 t/s | tg256 vs f16 |
|---|---|---|---|
| f16     | 2255.7 | 110.74 | -     |
| q40     | 2268.4 | 104.52 | -5.6% |
| q40krot | 2176.7 | 100.16 | -9.6% |

### DFlash2 tg arms (same suite, same binary, interleaved by phase order)
| arm | tg256 t/s | vs f16 | pp4096 t/s | accept_rate | mean_accept_len |
|---|---|---|---|---|---|
| f16     | 80.54 | -      | 1962.1 | 0.3581 | 2.4324 |
| q40     | 76.50 | -5.0%  | 1952.7 | 0.3508 | 2.4031 |
| q40krot | 74.97 | -6.9%  | 1934.8 | 0.3393 | 2.3573 |
q4_0 KV is SLOWER than f16 at this short-context stack (dequant cost dominates; the bandwidth
win needs long context - ruquant's dot8 kernel got its win at 64k depth and our fork lacks
that kernel). Rotation adds ~2% on top (two extra mul_mats per token per layer) and acceptance
drops a further ~3% relative. KV quantization here buys VRAM (4x smaller cache), not speed,
on pp512/tg256 shapes.

## 5. KLD verdict

Numerics gate vs f16-KV baseline (same binary, wikitext-2-raw 16ch ctx2048 anchor protocol):

| arm | Mean KLD vs f16 | Max KLD | PPL(Q)-PPL(f16) | greedy probe (temp0 seed42 n128) |
|---|---|---|---|---|
| q40 (raw q4_0 K+V)     | 0.011223 +/- 0.000237 | 1.35 | +0.0375 (+0.64%) | 107/109 tok agree, first divergence token 78 |
| q40krot (q4_0 + LLAMA_KV_KROT=1) | 0.009821 +/- 0.000204 | 1.29 | +0.0243 (+0.41%) | 107/109 tok agree, first divergence token 90 |
| control: raw q4_0 vs raw q4_0 | -0.000000 (max 0.000063) | - | - | harness floor = uint16 logprob quantization |

VERDICT: PASS. q4_0 KV sits at ~0.011 KLD for a 4x cache compression (well inside the
program's quantization-step envelope; the W4A4 weight-side gate alone measured 0.1222 on the
same harness family). The H-rotation strictly improves numerics: KLD -12.5% (0.0112 -> 0.0098),
PPL penalty -35% (+0.64% -> +0.41%), greedy probe diverges later (token 90 vs 78). Direction
and rough magnitude match the recovered ruquant prior (0.0442 raw -> 0.0423-0.0434 rotated,
f16 0.0416). Rotation-vs-raw-q4_0 KLD measured directly: 0.011661 +/- 0.000269.

Engagement witnesses: llama-cli --verbose legs show "llama_kv_cache: n_rot_k = 256" with
LLAMA_KV_KROT=1 (full 256-wide Hadamard on the target head_dim) and n_rot_k = 0 with the gate
off; n_rot_v = 0 in both (V rotation is a separate gate, not exercised). DFlash2 acceptance
(same suite): f16 0.3581 -> q40 0.3508 -> q40krot 0.3393 (mean accept len 2.4324 -> 2.4031 ->
2.3573): the drafter pays a small extra acceptance cost for the rotated cache even though the
target's distribution is closer to f16.

Perf summary (tables above): q4_0 KV is NOT a throughput lever at pp512/tg256 shapes
(plain llama-bench tg256 -5.6% q40 / -9.6% q40krot, pp512 neutral-to-3.5%-worse; DFlash2 tg
-5.0%/-6.9%, pp4096 -0.5%/-0.5%; dequant dominates at short context). The ruquant prior got its win
(44.7 -> 38.5 ms decode) only at 64k depth AND with a dedicated v_dot8 q4_0 FA kernel
(ruquant-src e509cee) that this fork does not carry. What q4_0-KV buys today on this fork is
~4x KV VRAM compression at a modest numerics cost that the rotation gate measurably reduces.
Recommendation: ship the gate default-OFF as implemented; adopt LLAMA_KV_KROT=1 + q4_0 KV only
for VRAM-bound long-context cells; a port of the ruquant dot8 q4_0 FA kernel is the prerequisite
for any perf claim and is out of scope here.

## 6. Ops notes / deviations

- All GPU legs serialized under flock /root/gpu.lock (waited behind a concurrent 27B lane
  multiple times; TERM-only teardown; pgrep -x llama-server verified empty after each leg).
- llama-perplexity/llama-server at default log level do not print the kv-cache n_rot lines in
  this fork; engagement was witnessed via a --verbose llama-cli leg per arm instead.
- The llama-cli greedy phase output is spinner/chat-template polluted (fork's cli runs the
  chat stream with reasoning mode); the authoritative greedy probe is the raw /completion
  temp-0 probe captured on the DFlash2 legs (spec decode is lossless vs target greedy).
- llama-bench rejects -c/--fit in this vintage; bench legs use its native context sizing.
- Driver: /root/taskI1-drive.sh + drive2.sh; raw artifacts /root/taskI1/.
- f16-KV base logits kept for re-scoring: /root/taskI1/i1-base-f16.logits (8.1 GB,
  --save-all-logits, same binary 80389528db). Delete when review closes.
