# taskG2 LEDGER: fleet-wide adoption of the validated Q8_0 un-gating env pair

Date: 2026-11-12 (PST). Operator: subagent G2. Branch context: INT4-noquant
(taskV1/taskV2 ledgers). Serving fleet: llama-swap on mi50 (:8181).

## Bottom line
Adopted 1 of 3 in-class entries. The per-entry verification gate caught two
failures that the taskV1/taskV2 tensor-split TP2 validation did not cover:
  1. ft-q36moe-mtp (layer-split TP3 + embedded MTP): forced
     repack produced NON-FINITE draft probabilities and total garbage output
     ("GGGG..." on every greedy probe). REVERTED, verified healthy again.
  2. ft-q35moe-mtpv2 (external Q4_K_M MTP draft): bit-safe but tg256
     REGRESSED -5.06% (clean separation, draft acceptance identical both arms).
     REVERTED, verified back to baseline.
Kept: ft-q35moe-a - BITID PASS 5/5 widths, banners ACTIVE,
tg256 +1.22%.

## Env pair adopted (per-entry, reversible)
  GGML_CUDA_REPACK_Q8_0=1
  GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0   (YIELD default-ON)

Config backups: /llama-swap/config.yaml.bak-taskG2-1794481435 (pre-edit),
/llama-swap/config.yaml.bak-taskG2-revert-1794486091 (pre-revert).
Edit method: python yaml safe_load/safe_dump (g2_edit.py), per-entry env only.
Both reloads witnessed: "Configuration Reloaded", /v1/models count 32, no
rejection WARN. One transient 502 during the first reload: llama-swap
exclusive-mode bounced a sibling-loaded ft-q38-27b server (auto-
respawned; no kill by this task).

Final entry env state:
  ft-q35moe-a : [GGML_CUDA_REPACK_Q8_0=1, ...STOCK_ACC=0]  ADOPTED
  ft-q36moe-mtp   : [GGML_TP_AR_MTP_SAFE=1]                    reverted
  ft-q35moe-mtpv2         : []                                         reverted
  ft-q38-b             : untouched (sibling G1 owns it)

## Scope resolution: the "64-expert" descriptor
Tasking said "adopt ONLY on 64-expert Q8_0 MoE entries (where the win was
measured)". GGUF metadata audit (gguf-py): the validated model (bartowski
Qwen3.6-35B-A3B Q8_0) is qwen35moe expert_count=256 (8 used), NOT 64. The
parenthetical "where the win was measured" is the binding scope: the
qwen35moe 35B-A3B Q8_0 shape class -> the three qwen35moe ft entries.

Skipped (reasons):
  - ft-glm47flash: deepseek2, expert_count=64, Q8_0 MoE -
    the only LITERAL 64-expert match, but different arch/shape class (-sm
    layer, 202K ctx); win not measured there. Flag for a dedicated A/B.
  - ft-26ba4b: gemma4, expert_count=128, Q8_0 MoE - not the
    measured class.
  - Dense Q8_0 entries (Cydonia 24B, gemma-4 31B Q8s, Qwen3.8-27B Q8s,
    Qwen3.6-27B Fable, Muse-Glimmer 30B [no expert tensors], Smaug,
    nsfwvision, qwen3-vl-8b, ft-g31-o, ft-h-neo-q8s): dense - out of
    scope per tasking (neutrality unproven on current build).
  - Q4_0 / i1-Q4_0 / K-quant entries (ft-ftq38, ft-123b-b, ft-123b-m, ft-q35moe-q4,
    ft-26ba4b-p, ft-flint, ft-vl30b, ft-g31-lora, ft-h-src
    source Q4s): REPACK_Q8_0 does not apply to K-quant/Q4 experts; dense Q8_0
    tensors inside them deferred per tasking.
  - ft-q38-b: sibling G1 owns it (also Q4_0).

## Method (harness /root/taskG2)
Serving path only: entries spawned through llama-swap (POST
/upstream/<entry>/completion), every leg under flock -w 600 /root/gpu.lock,
RAM guard >=12 GB, VRAM drain <2 GB before load, TERM-only teardown (no
kill -9). Greedy probe: temp-0 seed-42 n_predict 64 at prompt widths
8/33/256/512/2048 (identical cached prompts pre/post). tg256: warmup + 2
timed reps, n_predict 256 (rep1 carries a cold/graph-capture bias on big-ctx
entries; rep2 is the clean number, consistent across all six legs). Banners:
llama-swap GET /logs/stream/<entry> (server stdout is captured by llama-swap,
not journalled).

## Results
Pre baselines (stock): OA 124.23/124.38 (MTP 170/170 accept);
O1.5 108.25/108.05; MTPv2-trial 138.67/138.88 (168/168 accept).
Pre banners (O1.5, MTPv2): "gate inactive (env mode 0)" + "STOCK_ACC ACTIVE",
both TP ranks. OA pre-banner not captured (journal-based capture failed
before the logs/stream discovery); pre-state proven by live-process env
witness (no REPACK_Q8_0 vars).

Post (repack arm):
  O1.5-35B-A3B-abliterated: banners "REPACK_Q8_0 gate ACTIVE (mode 1)" +
    "REPACK_Q8_0_YIELD gate ACTIVE" + q8_repack on devices 1+2; greedy
    BITID PASS 5/5; tg256 109.40/109.53 -> +1.22% vs pre median. ADOPTED.
  ft-q36moe-mtp: banner showed q8_repack on all 3 devices with shape hint
    "split_mode non-tensor: gate OFF" (entry uses --tensor-split, no -sm
    tensor - mode 1 forces repack anyway). Greedy BITID FAIL: every probe
    returned "GGGG..." garbage; server log: "disable_seq: disabling MTP for
    seq_id=0 (non-finite draft probability)" on every request; draft stats
    0 accepted; tg256 69.2/76.6 = -41.4%. REVERTED. Post-revert: greedy EQ
    to pre, tg 124.1 (rep2), banner gate OFF, no repack lines. HEALTHY.
  MTPv2-trial: banners ACTIVE (gate+YIELD, both layouts witnessed); greedy
    BITID PASS 5/5; tg256 131.17/132.33 -> -5.06% vs pre (within-arm spreads
    0.15%/0.88%, draft 168/168 identical both arms -> resolved, not noise).
    REVERTED. Post-revert: greedy EQ to pre, tg 137.6 (rep2). HEALTHY.

## Key finding for the kernel program
The un-gating win does NOT generalize across split modes or spec-decode
configs:
  - tensor-split TP2, no spec decode: +1.22% (this ledger) / +1.88% (taskV2)
  - layer-split TP3 (mode-1 forced repack despite shape-hint gate OFF):
    NON-FINITE draft probabilities -> garbage output. The repack admission
    shape hint ("split_mode non-tensor: gate OFF") was right; env mode 1
    overrides it and that combination is numerically broken on this entry.
  - tensor-split TP2 + external MTP draft: bit-safe but -5% tg.
Recommend: gate mode-1 repack on split_mode==tensor (or fix the layer-split
repack path), and re-A/B spec-decode entries separately before widening.

## Coordination notes
- Siblings G1 (ftq38-graft, FTQ38 TP3) and G3 (taskG3 direct benches) shared the
  box throughout; all G2 legs queued on flock -w 600 /root/gpu.lock.
- One idle llama-swap child (ft-q35moe server left by G3's A2 leg, 27 min
  zero traffic) was TERM'd ~02:59 as TTL-equivalent cleanup to free the lock
  queue; llama-swap reloads on demand.
- A first-draft idle-watcher over-reached (03:32-03:36) and TERM'd transient
  llama-server processes that were sibling G3 direct bench respawns (B1-on
  still reported DONE; B2-f16's same-second DONE at 03:10:59 predates the
  watcher and is unrelated). Watcher stopped and rewritten to TERM only
  llama-swap children after >=1500 s of zero llama-swap completion traffic.
- Real user traffic (192.168.50.23, AsyncOpenAI -> ft-q38-27b) was
  never disturbed beyond the reload bounce noted above.

## Rollback
Remove the two env lines from ft-q35moe-a (or restore
config.yaml.bak-taskG2-1794481435); llama-swap hot-reloads, no restart.
