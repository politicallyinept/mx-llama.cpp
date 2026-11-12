# taskM3-LEDGER: GGML_CUDA_I8NARROW on the bartowski Qwen3.5-35B-A3B Q4_0 quant (mi50)

Date: 2026-11-11 (PST). Agent: M3 subagent. Box: mi50, GPU2 only.
Build: /root/int4-noquant-wt/build/bin/llama-server @ INT4-noquant (version stamp
build 11881, commit 7129719aa5), md5 stable across ALL arms:
llama-server 895219cb739b4867411a383acf096eae, libggml-hip.so 496dde34b8da7f4c7c51b80d855d69c9.
No rebuild. Model: /root/bench-quants/bartowski/Qwen_Qwen3.5-35B-A3B-GGUF/
Qwen_Qwen3.5-35B-A3B-Q4_0.gguf (20836243264 B).

## Verdict

HURTS. Unlike UD-Q4_K_XL (taskM2: gate statically unreachable), on this quant the gate
DOES fire - [I8NARROW] banner + repacked Q4_0 dispatches at admissible widths - but it
CORRUPTS OUTPUT: every gate-ON arm (4/4) emits degenerate "33333..." from the first
token at temp 0, while gate-OFF arms (3/3) are coherent. The apparent tg +15.6% /
pp +19.5% is an artifact of measuring a numerically broken model (see Findings #5).
Do NOT enable GGML_CUDA_I8NARROW on Q4_0 MoE serving.

## GGUF tensor-type histogram (gguf-py GGUFReader, tensor-info block only)

arch qwen35moe, 41 blocks, 256 experts, 753 tensors:

    Q4_0  208   <-- gate ADMISSIBLE (production-env exports GGML_CUDA_REPACK_Q4_0=1
    F32   368       via w4a4-env.sh; load banners confirm "Q4_0 admitted to the gfx906
    Q8_0  105       repack path" + "repacking q4_0 tensors on device 2")
    Q5_0   60
    Q6_K    5   (output.weight + 4x attn_q)
    Q4_1    5   (5x ffn_down_exps - NOT admitted, DOT8=4-only)
    BF16    2   (blk.40 router graft tensors)

MoE expert tensors: ffn_gate_exps Q4_0 x40 + Q8_0 x1; ffn_up_exps Q4_0 x40 + Q8_0 x1;
ffn_down_exps Q4_0 x35 + Q4_1 x5 + Q8_0 x1 (3D, [512,2048,256]-class); shared experts
ffn_*_shexp Q5_0 x20 + Q8_0 x21. Dense: attn_qkv/attn_gate/attn_output/ssm_out mostly
Q4_0; token_embd Q4_0. nextn head present (blk.40.nextn.*, eh_proj Q8_0) but unused
(no spec flags, mirroring the taskM1 ab-ud-off control config).

## Serving A/B (GPU2 --device ROCm2, flock -w 600 /root/gpu.lock, production-env +
## GGML_TP_AR_MTP_SAFE=1; -ngl 99 -fa on -c 16384 -b 4096 -ub 4096 -t 16 -np 1
## -lm dio -fit off; mtp_bench.py pp4096 x3 + tg256 x3, temp0 seed42, per-arm jsonl)

Arms interleaved off/on/off, then a repro pair (on/off) on a second flock window:

| arm        | pp4096 t/s | tg256 t/s | smoke (temp0) |
|------------|-----------:|----------:|---------------|
| gate-off-1 |   1119.056 |   107.099 | coherent      |
| gate-on-1  |   1336.960 |   124.668 | '33333...'    |
| gate-off-2 |   1117.916 |   107.864 | coherent      |
| gate-on-2  |   1337.578 |   124.017 | '33333...'    |
| gate-off-3 |   1119.365 |   107.563 | coherent      |

off medians: pp 1119.056, tg 107.563. Base-to-base spread (n=3): pp 0.129%, tg 0.710%.
on medians:  pp 1337.269 (+19.50%), tg 124.342 (+15.60%) - deltas are 20-150x the base
spread and track the env var exactly, but are measured on CORRUPTED output (see below).

## Witness evidence

1. [I8NARROW] banner: PRESENT in all 4 gate-ON arms, absent in all 3 gate-OFF arms.
   Emitted at the first narrow-window dispatch (same millisecond as the first
   [rp-dispatch] line), i.e. the dispatch site is reached and the gate env is live.
2. [rp-dispatch] type histogram (GGML_CUDA_RP_DISPATCH_TRACE=1): 60/60 lines type=2
   (Q4_0), ALL at ne11=2, in every arm including gate-OFF (the trace is gate-agnostic).
   The 60-line per-process cap is consumed by the graph-warmup pass (ne11=2 dummy
   batch) before any request traffic. Tensor names in the 60 lines (blk.0..blk.25):
   attn_qkv 20, attn_gate 19, ssm_out 11, attn_output 6, attn_q 4 - DENSE Q4_0 only;
   no ffn_*_exps appears between adjacent blocks' attn entries, so the 3D expert
   grouped matmuls do NOT route through the dense repacked narrow window (they run in
   the MOE_VEC family under the production bundle).
3. Width coverage caveats: trace floor is ne11>=2, so ne11=1 decode dispatches (the
   dominant serving width, and IN the gate set) are admitted but invisible to the
   trace; width 16 is UNREACHABLE - the gate sits inside the ne11<=rp_narrow_max
   window and rp_narrow_max is clamped to MMQ_RP_Q8_MMV_MAX_TOKENS=8. A width-burst
   sampler (m3_burst.py) was written but the warmup cap made it moot (and it hit a
   /tokenize-returns-strings API change; left broken, harmless).
4. Corruption is NOT graph-replay staleness: gate-ON with LLAMA_GRAPH_REUSE_DISABLE=1
   (graphs reused = 0 confirmed) STILL emits '33333...'. The i8narrow kernel verified
   standalone in the B2 microbench (maxrel <= 0.0084), so the breakage is in the
   serving-context integration. Prime suspect: the production bundle runs with
   GGML_CUDA_W4A4_SUMW ACTIVE (PROFILE=speed default, "+2 B/block SUMW plane" appended
   to the repacked Q4_0 record) while the gate-site comment asserts the record is the
   plain [qs|d] two-plane layout the kernel reads; a stride/offset mismatch there would
   read wrong d-plane values -> garbage. Flag for the branch owner.
5. The perf "win" is a collapsed-routing artifact: GEMM time is data-independent, yet
   pp4096 (which never enters the narrow window) is +19.5% faster with the gate ON.
   The corrupted hidden states collapse the router (degenerate single-token output
   from step 1), so nearly all 256 experts receive zero tokens and the grouped-MoE
   path skips their GEMMs. The speedup measures a model doing almost no expert work,
   not a faster kernel. (pp delta direction reproduced 2/2; mechanism inferred, not
   profiled.)
6. Incidental: gate-on-2 aborted during post-bench teardown ("terminate called without
   an active exception", after BENCH-DONE) - same teardown-path fragility M2 flagged;
   bench data unaffected.

## Rule compliance

All GPU touches under flock -w 600 /root/gpu.lock; GPU2 only (--device ROCm2, the
M2-proven single-GPU pin); TERM-only teardown, curl /health polling; RAM guard >=12GB;
per-arm jsonl (mtp_bench.py opens 'w'); GGUF read via gguf-py tensor-info block only;
binary md5 bracketed before/after every run; no rebuild; no pushes.

## Artifacts

/root/moe-nextn/m3-i8narrow-q40/ - server logs, witness.txt, environ proofs,
m3-gate-*.jsonl, smoke jsons, bin-md5 brackets, doctor.txt.
/root/moe-nextn/m3_i8narrow_ab.sh (5-arm driver), m3_repro.sh, m3_nograph.sh
(graph-reuse-disabled probe), m3_burst.py (broken, see finding 3), m3_final.py (stats).
