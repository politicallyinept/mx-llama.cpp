# taskL ledger - Q4_0 drafter A/B vs Q8_0 drafter (net-tg verdict), 27B DFlash2

Date: 2026-11-13 (PST). Box: mi50. Branch int4-l-draftq4 from INT4-noquant @ 15196ec20f.
Build under test: /root/build-jmerge (BUILD-PROVENANCE src_sha=a63c468d60; `git diff
a63c468d60 15196ec20f` is EMPTY - the tip re-merge is content-identical, so the binary
matches the branch tip). Gates witnessed present in libggml-hip.so: RP_Q8_NC_LDS,
FA_DOT8, W4A4_MIN_TOKENS.

## 1. Question

Does a native Q4_0 drafter beat the Q8_0 drafter on net tg? The Q8_0 drafter costs
~19 ms/step draft-fwd at d7 (taskJ2: 18.9-19.2 ms, unchanged by the LDS gate - the gate
targets the target-model verify path, not the M=1..8 draft path). Halving drafter weight
bytes should cut draft-fwd on the bandwidth-bound draft forward - but only if acceptance
survives the requant. ACCEPTANCE IS THE GATE: a drafter requant that loses more acceptance
than its draft-time gain is a FAIL even if draft-fwd drops.

## 2. Drafter artifacts (no new requant needed)

- Q8_0 control: /mnt/old/home/brad/models/Qwen3.8-27B-DFlash2-Q8_0.gguf (2,056,414,816 B,
  identical to incoai/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-Q8_0.gguf).
- Q4_0 arm: /mnt/old/home/brad/models/Qwen3.8-27B-DFlash2-Q4_0.gguf (1,094,346,336 B).
  Already on-box (2026-11-11, used by taskG3 A1 on the Q4_0-main stack). HF publishes only
  BF16/Q8_0/Q4_K_M for this drafter - no upstream Q4_0 - so this local file IS the standard
  Q4_0 requant. VERIFIED this session: 81 tensors, names+order identical to the Q8_0 file;
  all 49 matmul tensors Q8_0->Q4_0; all 32 norm tensors stay F32. Dequant spot-check vs the
  Q8_0 file (numpy, gguf-py): rms_rel 0.0867-0.0875 across fc/attn_q/ffn_down/selector
  (mean 0.0870) - faithful per-block requant, not garbage.
- Head handling: the DFlash2 27B drafter GGUF carries NO output/token_embd tensors - it is
  a borrowed-head drafter (borrows the target's head via ctx_other; taskH3). The "leave the
  output head at higher precision" arm is therefore moot: the head is the target's Q8_0
  head in every arm. Only the drafter body changed quant.

## 3. Protocol

Driver /root/taskL-drive.sh + /root/taskL-chain.sh (supervised retry loop), outputs
/root/taskL/. Config = LDS-alone (taskJ2 lds arm): production-env.sh + GGML_CUDA_REPACK_Q8_0=1
+ GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0 + GGML_CUDA_RP_Q8_NC_LDS=8, main model
unsloth Qwen3.8-27B-Q8_0 TP2 (-sm tensor, -ngl 99 -fa on --fit off -c 8192 -ub 1024 -t 16
-np 1 -lm dio), GGML_DFLASH_PROF=1. Sibling K's q4_0-KV x LDS combine did not land before
this run started (K queued behind a foreign ship-build server holding GPU1+2), so the
fallback config per brief is LDS-alone.
Arms: q8d = Q8_0 drafter (control); q4d = Q4_0 drafter; q4w0 = Q4_0 drafter +
GGML_CUDA_W4A4_MIN_TOKENS=0 (see s4). Depths 4 and 7 (q4w0 at d7), 2 interleaved reps.
Measure per leg: natural suite bench.py (8 prompts x 3 reps tg256 + pp4096), acceptance
from timings.draft_n/_accepted (bench.py agg), DFLASH-PROF draft-fwd us/step medians,
greedy 64-token temp0 seed42 probe. Doctrine: flock /root/gpu.lock -w 900, load/RAM/VRAM
gates, TERM-only teardown + pgrep -x llama-server empty check, production env + TheRock
runtime witnessed per leg (proc environ + maps), sha256 of main+drafter per leg,
refuse-if-exists.

## 4. The forced-W4A4 question (F5 floor-0 scoped to the drafter)

F5 env = GGML_CUDA_W4A4_MIN_TOKENS (q8_repack/w4a4-profile.h:109). On this build the
W4A4 profile default (unset -> speed) resolves DOT8=3 + MIN_TOKENS=128: the u4 GEMM/act
paths (u4act/u4x16/u4q41) are compiled and admitted but AUTO-DISABLED per GEMM call below
batch width 128 (q8_repack/mul-mat.cu floor comment: "MTP speculative verify runs at
ne11 ~= n_max+1 = 5 rows, squarely in the losing zone"). F5 = floor-0 (MIN_TOKENS=0
explicit) forces the u4 path at narrow widths.

Can it be scoped to the drafter forward? YES, by dtype composition, no code change:
the floor knob itself is process-global, but the u4 predicates only fire on REPACKED Q4_0
tensors (weight-side admission via GGML_CUDA_REPACK_Q4_0=1, which production-env already
exports). With a Q8_0 main model + Q4_0 drafter, the ONLY q4_0 tensors in the process are
the drafter's - the main model's Q8_0 tensors go through the q8 repack/LDS family and the
KV is f16. So MIN_TOKENS=0 changes exactly the drafter's narrow-width matmuls (draft-fwd
runs n_tokens=8 at d7) and nothing else. The q4w0 arm measures it; the gate witness is
the "MIN_TOKENS=0 [explicit]" + "floor ACTIVE" banner pair in the server log plus the
proc-environ witness.

## 5. Results

Natural suite tg256 medians (bench.py 8 prompts x 3 reps = 24 legs/leg-run), 2 interleaved
reps per cell, LDS-alone config, build-jmerge. Aggregator /root/taskL-agg.py; per-leg
records /root/taskL/*.reps.jsonl + .agg.json. Acceptance counts were BYTE-IDENTICAL across
reps within every arm (greedy verify is deterministic), so rep-to-rep tg spread is pure box
drift.

| leg        | tg256 r1/r2 | tg med | accept (acc/draft)     | draft-fwd med |
|------------|-------------|--------|------------------------|---------------|
| q8d-d4     | 46.35/48.09 | 47.22  | 0.3605 (3597/9978)     | 14.67 ms      |
| q4d-d4     | 50.54/50.33 | 50.44  | 0.3544 (3573/10083)    | 12.08 ms      |
| q8d-d7     | 38.72/39.54 | 39.13  | 0.2327 (3777/16233)    | 19.11 ms      |
| q4d-d7     | 40.39/40.83 | 40.61  | 0.2349 (3789/16128)    | 16.07 ms      |
| q4w0-d7    | 38.84/40.93 | 39.88  | 0.2349 (3789/16128)    | 16.13 ms      |

NET-TG VERDICT (median over reps):
* d4: q4d vs q8d  tg +6.81%  accept -0.0061  draft-fwd -17.6%
* d7: q4d vs q8d  tg +3.78%  accept +0.0022  draft-fwd -15.9%
* d7: q4w0 vs q4d tg -1.78%  accept +0.0000  draft-fwd +0.4%  (floor-0 = no-op, see 5a)

Controls: q8d-d7 accept 3777/16233 is DIGIT-IDENTICAL to taskJ2 lds-r1 (3777/16233) -
same drafter, same suite, deterministic greedy verify. Absolute tg sits ~3% under taskJ2
(q8d-d7 39.13 vs 39.71; sibling K's same-config ldsctrl-r1 read 38.06 in the same window)
- session drift, so only the paired within-session deltas above are quotable. Greedy
64-token probes: ALL 10 legs byte-identical (sha16 77841cb6a251235d, same as taskJ2) -
the probe prompt is too predictable to discriminate drafter quant; acceptance is the
discriminating gate, as designed.

### 5a. Forced-W4A4 (F5 floor-0) scoped to the drafter: NO (inert on this stack)

Env witness on both q4w0 legs: "resolved: DOT8=3 MIN_TOKENS=0 (explicit)" + "MIN_TOKENS=0
[explicit]" floor banner - the env landed. But the u4 paths only fire on REPACKED q4_0
tensors, and the drafter's q4_0 tensors are NOT repacked on this stack: the repack
activation witness ("q8_repack: repacking q4_0 tensors on device N", buffer.cu:50, fires
from every repack entry once per type+device) is ABSENT in every q4d/q4w0 leg log while
the q8_0 witness fires normally for the main model. The drafter's tensors do reach the
admission code (log: "3D non-expert tensor blk.0.attn_conv_base kept canonical") but the
q4_0 dense tensors never enter the repack path, so the floor knob has nothing to act on.
Empirical confirmation, both reps: q4w0 vs q4d acceptance BYTE-IDENTICAL (3789/16128) and
draft-fwd within noise (16.13 vs 16.07 ms, +0.4%) - identical compute path; the tg delta
is drift.
Verdict on the forced-w4a4 question: the env is process-global (not scopeable by env), and
on a Q8_0-main + Q4_0-drafter stack it is INERT because drafter-context tensors are not
admitted to the repack/u4 path. Forcing W4A4 on the drafter forward would need a code
change (admit drafter tensors to repack admission) - out of scope for this train-free,
standard-format task. The Q4_0 drafter's speedup comes from the canonical stock q4_0 path
(half the weight bytes at draft widths), not from the u4 GEMM family.

## 6. Verdict: SHIP the Q4_0 drafter (both depths)

ACCEPTANCE GATE: PASS at both depths. d7 acceptance RISES slightly (+0.0022); d4 loses
0.0061 (24 fewer accepted tokens on ~10k drafts) - far smaller than the draft-time gain
(-17.6%), so net tg is positive by a wide margin at both depths. The requant loses nothing
that matters: the drafter only needs to guess tokens the target will confirm, and 4-bit
weight noise costs ~0.6pct relative acceptance at d4 while cutting draft-fwd ~18%.

SHIP RECOMMENDATION: adopt Qwen3.8-27B-DFlash2-Q4_0.gguf as the drafter for the LDS-alone
config at both d4 and d7: +6.8pct net tg at d4, +3.8pct at d7, acceptance neutral, VRAM
halved for the drafter (2.06 GB -> 1.09 GB). Do NOT bother with the floor-0 W4A4 arm
(witnessed no-op). Depth note for the parent: on this config d4 beats d7 with BOTH
drafters (50.4 vs 40.6 t/s) - d4 is the better depth on LDS-alone f16-KV; the q4_0-KV
combine (sibling K) may reorder that.

Caveats: single box, single session; tg absolutes carry ~3pct session drift (paired deltas
only). Acceptance is suite-specific (natural prose, temp0); the d4 acceptance delta
(-0.0061) is small but consistent (identical both reps). No KLD leg run: the drafter does
not touch the shipped output distribution (spec decode is lossless - the target verifies
every token), acceptance is the correct numerics readout per taskJ1 precedent.

## 7. Artifacts

* Branch int4-l-draftq4 (worktree /root/wt-l, from INT4-noquant @ 15196ec20f): this
  ledger + driver/chain/aggregator. NOT pushed.
* /root/taskL/: per-leg reps.jsonl + agg.json, server logs, witnesses (proc environ,
  sha256 of main+drafter, gate banners), greedy probes, chain.log.
* /root/taskL-drive.sh, /root/taskL-chain.sh, /root/taskL-agg.py.
