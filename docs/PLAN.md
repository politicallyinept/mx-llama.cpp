# mx-llama.cpp → Mistral-Large-123B fine-tune-v2 q4_0 on 3× MI50 (gfx906, ROCm 7.14)

Status: PLANNING (awaiting fork-analysis + bench-matrix subagent outputs, then SSH key)

## Goal
Tune the mx-llama.cpp fork for Mistral-Large-123B fine-tune-v2 (Mistral-Large-Instruct-2407 arch) q4_0
on 3× MI50 32GB, ROCm/HIP backend, comparing `-sm layer` vs `-sm tensor`. All work lands
in local Gitea (root/mx-llama.cpp) with benchmark evidence.

## Fixed facts
- Hardware: 3× MI50 32GB (gfx906), ROCm 7.14, PCIe. 96GB total VRAM.
- Model: Mistral-Large-123B fine-tune-v2 q4_0 = 69.32GB (split GGUF dir from bartowski).
  Arch: 88 layers, 96 attn heads, 8 KV heads (GQA), head_dim 128, hidden 12288,
  intermediate 28672, vocab 32768, rope 1M, max_pos 131072. Dense Mistral.
- VRAM math @ full offload: 69.32/3 = 23.1GB weights + KV + compute per GPU.
  KV f16 @ 32k ctx ≈ 2.0GB total (8 KV heads × 128 × 88 layers × 2(K+V) × 2B × 32768
  = 2.147e9 B ≈ 2.0GiB) → tiny thanks to GQA-8. Headroom is comfortable (~7-8GB/GPU).
- Fork runtime requirements (FEATURES.md): `-lm dio` (mmap hangs),
  GGML_ENABLE_CUSTOM_AR=1, HSA_FORCE_FINE_GRAIN_PCIE=1, GPU_MAX_HW_QUEUES=8,
  HSA_OVERRIDE_GFX_VERSION=9.0.6 if the ROCm 7.14 runtime is trimmed.
- Build: cmake -DGGML_HIP=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_RCCL=ON
  -DAMDGPU_TARGETS=gfx906 -DLLAMA_OPENSSL=ON -DHIP_COMPILER=clang
  -DCMAKE_CXX_FLAGS="-O3 -Wno-unused-command-line-argument"

## Repo state
- Gitea root/mx-llama.cpp master = 1f4cdc5 (our Aug-22 merge of upstream 3f545be + brace fix)
- Fork upstream has since: merged llama.cpp b10760 itself (0fdda7f1d, Sep 2) and added
  gfx906 work through Sep 7 (50d5fdeb HEAD), incl.:
  - 2e9d29fe7 q8_repack gfx906 path for IQ4_NL/Q6_K/Q4_K/Q5_K (prefill) — NOT q4_0
  - b8baef09f fused K-quant + IQ4_NL MoE up/gate mat-vec on gfx906 (+6-8% decode)
  - 50d5fdeba fused MoE witness safety under parallel meta dispatch
  (MoE-focused commits are inert for dense Mistral; K-quant mat-vec fusion is relevant
  background for our q4_0 work.)
- DECISION (pending): re-point Gitea master to fork 50d5fdeb (their upstream merge is
  authoritative), keep 1f4cdc5 on an archive branch. Avoids duplicate-merge conflicts.

## Tuning opportunity (hypothesis)
Fork's kernel work is q8_0/K-quant-centric. Mistral-Large fine-tune q4_0 decode (batch 1 mat-vec) and
prefill (mmq) on gfx906 have no fork-specific tuning yet. Levers:
1. mmq tile config for q4_0 at our N,K shapes (row length 12288 / 28672 / 32768 vocab).
2. q8_repack-style path for q4_0 (fork did Q4_K etc. — q4_0 structurally simpler).
3. -sm tensor viability: 96 heads / 3 GPUs = 32 per GPU ✓ but 8 KV heads / 3 ✗ —
   behavior TBD by fork-analysis agent (replicate? error?).
4. AllReduce: hidden 12288 × f32 = 49KB/msg per layer ×88 — custom AR should cover.

## Phases
- P0 sync Gitea master to fork 50d5fdeb; push branch gfx906-w4a4 off it.
- P1 SSH onto MI50 box: env audit (ROCm 7.14 gfx906 detection, disk space for 70GB),
  build baseline binaries (fork @50d5fdeb + upstream llama.cpp control).
- P2 download bartowski Q4_0 split dir (69.32GB) — hf CLI or aria2c.
- P3 run bench matrix (subagent-designed harness) — establishes layer vs tensor, fa,
  KV quant, batch, env toggles, ctx scaling baselines.
- P4 quality gate: determinism + greedy-output parity between split modes.
- P5 tuning iterations on q4_0 kernels; every change re-benched against P3 baseline.
- P6 land tuned branch in Gitea with BENCH.md results doc.

## Constraints
- Upstream llama.cpp AGENTS.md: our private-Gitea commits are exempt; any upstream PR
  must be human-authored by Brad (we only prepare patches, ASCII style, no auto-PR).
- Every perf claim needs a measured baseline vs control (Brad: root-cause/verify first).
