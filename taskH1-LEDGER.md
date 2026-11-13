# taskH1 ledger — DFlash2 depth fixes: block-width interference root cause + adaptive-depth prototype

Stack: 35B Qwen3.6-35B-A3B Q8_0 main + DFlash2 Q8_0 drafter, TP2 (GPU1+2), same serving
flags as taskG5 (flock gpu.lock, TheRock runtime, TERM-only teardown). Driver: /root/taskH1/drive.sh,
drive2.sh. Code: common/speculative.cpp @ 2effe9dd3a + 804484f348 (all new behavior env-gated, default OFF).

## 0. TOOLCHAIN FINDING (program-relevant, read first)

A fresh cmake build dir of this tree silently configures the HIP compiler to
/opt/rocm-7.2.3/lib/llvm/bin/clang++, while the canonical build/bin (cache from Nov 11,
used by G5 and every prior benchmark) was configured with the TheRock devel compiler
/root/therock-venv/.../_rocm_sdk_devel/bin/amdclang++ (+ therock AMDDeviceLibs, CMAKE_HIP_ARCHITECTURES=gfx906).
mi50-build.sh does NOT pin CMAKE_HIP_COMPILER, so a fresh build dir changes kernel codegen and
therefore numerics. The two toolchains are each internally deterministic but produce
DIFFERENT acceptance on this workload (same model sha, same flags, same env):

  leg (build-h1 = /opt/rocm clang++) vs G5 (build/bin = therock amdclang++):
    d4 per-pos acc: (0.658,0.445,0.328,0.270) vs (0.647,0.394,0.298,0.248)
    econ-reasoning mean len 4.38 vs 2.26; math 2.82 vs 3.98; suite tg median 77.2 vs 70.7

This initially masqueraded as an adaptive-controller artifact (C-d4-ctrl reproduced the
adaptive arm's per-prompt numbers exactly). PROOF: build-h1c (same source as build-h1,
reconfigured with the therock compiler) leg C-d4-ctrlc reproduces G5 d4 per-pos acceptance
BIT-EXACTLY: (0.647, 0.394, 0.298, 0.248) = G5 d4-r1/r2; per-prompt t/s within contention
noise (econ 70.1 vs 71, math 115.6 vs 115, dialogue 49.8 vs 49.9; tg med-all 70.15 vs 70.7).
The compiler is the cause. ACTION for the program: pin CMAKE_HIP_COMPILER (+AMDDeviceLibs_DIR,
CMAKE_HIP_ARCHITECTURES) in mi50-build.sh, or never create a fresh build dir for benchmark
legs. (Not changed here - shared script, siblings mid-flight.)

## 1. Block-width interference: root cause

Same-binary width sweep (build-h1, fixed depth, natural suite, per-pos draft acceptance):

  width 5 (d4):  0.658 0.445 0.328 0.270                    tg med-all 77.2
  width 6 (d5):  0.652 0.399 0.279 0.224 0.195              tg med-all 66.6
  width 7 (d6):  0.650 0.385 0.259 0.204 0.177 0.157        tg med-all 62.2
  width 8 (d7):  0.594 0.310 0.204 0.165 0.142 0.126 0.113  tg med-all 49.1 (F-d7-ctrl)

Positions 1-4 acceptance degrades MONOTONICALLY with block width - there is no peak at the
trained width; the trained width is the worst early-position config. The degradation is in the
DRAFTER, not the verify: GGML_DFLASH_LATTICE dumps at the only guaranteed-identical context
(first block, pos0=492, code-discussion) show the selector walk at i=1..4 picks the same tokens
at w5 and w8, but the lattice's per-position candidate sets and rankings shift with width
(i=3 top-1 flips 12->18 and token 12 leaves the top-16 at w8; i=4 argmax conf 0.930->0.856).

Mechanism (code read + probes):
- DFlash2 never reads drafter logits: the lattice head emits, per block slot, a top-k candidate
  row plus a k x k score matrix indexed by the PREDECESSOR rank (common/speculative.cpp
  is_dflash2 walk). Position i's proposals are computed from the slot-i hidden state, which
  mixes the ENTIRE block through the trained bidirectional block-internal attention.
- Extra mask slots therefore rewrite early-slot candidate sets even at identical committed
  context. On natural prose the tail slots are confidently wrong (selector conf 0.80-0.87 at
  positions 5-7 while acceptance there is 0.09-0.15), and that wrong block state feeds back
  into positions 1-4 through the bidirectional mixing. The predecessor-rank conditioning then
  compounds the shifted proposals down the chain.
- LLAMA_DFLASH_CAUSAL=1 (block-internal attention forced causal after init) REFUTES the
  trailing-mask-visibility fix. Same-binary width-8 comparison: causal per-pos =
  (0.607, 0.311, 0.195, 0.152, 0.137, 0.127, 0.117) vs bidirectional (F-d7-ctrl)
  (0.594, 0.310, 0.204, 0.165, ...) - causal does NOT restore early positions toward width-5
  (0.658/0.445/0.328/0.270); pos2 stays at 0.31 vs 0.445. The bidirectional block context is
  trained-in and load-bearing; the interference is not a masking artifact serving can mask
  away. It is intrinsic to the trained block mixing: width is a drafter-side quality dial on
  OOD text, and shorter blocks are strictly better for early positions.
- Per-prompt mean accept len vs width (build-h1): econ 4.38@w5 -> 2.00@w8 (t/s 127.7 -> 49.8),
  scifi 2.20 -> 1.83, mystery 1.82 -> 1.93; math 2.82 -> 5.10 (t/s 86 -> 115 - math is the one
  prose-class prompt whose tail slots are good); tech 4.47 -> 6.22, code 4.81 -> 7.50.
  The w8/w5 mean-len ratio >= ~1.35 cleanly separates deep-winning prompts (tech 1.39,
  code 1.56, math 1.81) from shallow-winning ones (all prose <= 1.06) on this binary.

## 2. Adaptive-depth prototype

LLAMA_DFLASH_ADAPTIVE=1 (DFlash2 only): sticky per-request n_max. shallow=LLAMA_DFLASH_SHALLOW
(default 4), deep = flag --spec-draft-n-max (7). Depth switches only at begin() (request start)
- block width is baked into the draft graph, rebuild ~40 ms on MI50. Sizing: keep the flag at the
DEEP bound so output/graph sizing covers the widest block (flag < deep aborts GGML_ASSERT in
output_reserve at the first shallow->deep switch - witnessed, documented).
Signals: (a) prompt-class classifier (>=2 hits of code/tech markers -> deep),
(b) saturation: a prompt whose prior shallow run reached mean accept len >= LLAMA_DFLASH_SAT
(default 4.4) gets deep next time. LLAMA_DFLASH_CLASS=0 disables (a).

build-h1 A/B (tg256, median-of-3 per prompt; mean = mean of per-prompt medians):
  C-d4-ctrl  fixed d4:  mean 89.42, tg med-all 77.21
  A-adapt    adaptive:  mean 93.94 / 92.85 (+5.0%), tg med-all 77.05 / 75.93
  F-d7-ctrl  fixed d7:  mean 79.11, tg med-all 49.08 (-11.5% vs d4 mean, -36% med-all)

Matched-toolchain triangle (build-h1c = therock, the canonical compiler; the headline numbers):
  C-d4-ctrlc fixed d4:  mean 84.11, tg med-all 70.15   (= G5 d4 reproduced)
  A-adapt-r1c adaptive: mean 88.40 (+5.1%), tg med-all 70.12
  F-d7-ctrlc fixed d7:  (fill)
Per-prompt adaptive vs fixed-d4 (build-h1c): code 156.9 vs 123.7 (+26.8%), tech 126.6 vs 123.8
(+2.3%), all six prose prompts unchanged within noise (controller keeps them shallow).
Per-prompt (adaptive vs fixed-d4): code-discussion 156.5 vs 133.5 (+17%), tech-explain 133.6 vs
123.4 (+8%); all six prose prompts unchanged (controller keeps them shallow; identical acceptance
and t/s within noise). Depth log: classifier drove every deep decision (class_hits=4 on tech/code);
sat never fired on build-h1 (max shallow mean len 4.38, econ, just under the 4.4 threshold -
threshold is toolchain-sensitive, see sec 0). On build-h1c the sat signal DID populate
(math sat=3.98 < 4.4 -> correctly stayed shallow; G5 showed d7 hurts math on that binary).

## 3. Matched-toolchain legs (build-h1c = therock amdclang++, same source as build-h1)

  C-d4-ctrlc    fixed d4:  per-pos (0.647,0.394,0.298,0.248)  mean 84.11  med-all 70.15  == G5 d4
  F-d7-ctrlc    fixed d7:  per-pos (0.635,0.344,0.237,0.189,0.153,0.125,0.107) mean 73.31 med-all 59.74  == G5 d7
  A-adapt-r1c   adaptive:  mean 88.40 (+5.1% vs d4, +20.6% vs d7)  med-all 70.12
  CA-d7-causalc causal w8: per-pos (0.638,0.350,0.231,0.174,0.145,0.120,0.109) mean 76.75 med-all 53.18

Causal-vs-bidirectional at width 8 on the canonical binary: +0.003/+0.006/-0.006/-0.015 at
positions 1-4 - no recovery of the width-5 gap (0.647/0.394/0.298/0.248). Trailing-mask
visibility is decisively not the interference channel.

## 4. Combined recommendation

1. SHIP adaptive depth (env-gated, default OFF): LLAMA_DFLASH_ADAPTIVE=1 with the flag at the
   trained deep bound (7) and shallow=4. Canonical-toolchain A/B: +5.1% mean-of-prompts over
   fixed d4, +20.6% over fixed d7, prose prompts untouched (sticky shallow), tg median equal
   to d4 (gains concentrate in structured traffic: code +26.8%, tech +2.3%).
2. The interference is drafter-side and NOT fixable at serving (causal override refuted);
   depth selection IS the interference mitigation - shallow blocks for prose are the fix.
3. v2 controller upgrade (not built here): explore-then-stick per prompt hash - run request 1
   shallow, request 2 deep, then stick with the depth whose mean-len ratio >= ~1.35 (the ratio
   cleanly separates deep-winning prompts on each binary and is toolchain-adaptive by
   construction, unlike the fixed 4.4 sat threshold and the keyword classifier, which missed
   math on build-h1 where d7 wins +34% on that binary).
4. PROGRAM ACTION: pin CMAKE_HIP_COMPILER/AMDDeviceLibs_DIR/CMAKE_HIP_ARCHITECTURES in
   mi50-build.sh (sec 0) - a fresh build dir silently changes acceptance numerics.

Caveats: n=8-prompt suite; classifier is a keyword heuristic (class_hits>=2); t/s carries
sibling-contention noise (+-7% on code-discussion across legs), acceptance is exact.
