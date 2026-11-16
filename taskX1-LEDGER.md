# taskX1 ledger - head-coverage lane (requant path): Q6_K head -> Q8_0, Q4_1 ffn_down -> Q4_0

Date: 2026-11-15 (PST). Box: mi50. Branch `int4-x1-headreq` off INT4-noquant tip
204739d62d (worktree /root/wt-x1; NOT pushed). Build /root/build-uwave, llama-server
md5 b956a6858d8976624271378f97d825cb (re-verified vs taskW3 record; git log -1 = tip).
GPU0 QUARANTINED per brief; every leg ran HIP_VISIBLE_DEVICES=1,2, device 0 never
touched. The :9400 exporter read 0/0 correctable/uncorrectable on all four cards at
session start (counters appear re-armed since the 919k era); quarantine kept regardless.
Canonical greedy probe sha16 confirmed 77841cb6a251235d (full sha; brief string OK).

## 1. Question

taskQ coverage audit: the Q6_K lm_head (output.weight, 5120x248320, 6.7% of file
bytes, the largest single verify matvec) and the 8 Q4_1 ffn_down tensors sit OUTSIDE
all LDS/repack coverage. Requantize head -> Q8_0 (the class the ruquant precedent
chose "for speed and accuracy"; q8_0 is repack/LDS-admissible via GGML_CUDA_REPACK_Q8_0)
and ffn_down -> Q4_0 (repack-admitted by default) in a GGUF copy, then measure at the
champion config (f128q4kv-p-d4 = 58.20 t/s, taskW3) whether the larger matvec enters
coverage fast enough to pay.

## 2. Surgery (CPU-only, no GPU, no training)

Tooling: /root/taskX1/{x1_selftest.py,x1_requant.py,x1_verify.py}. Raw byte surgery
per the taskR graft_ownhead.py pattern (streamed copies, empirically-verified dtype
table, RSS-capped requant chunks, never stream-edits the source). Requant math =
plain gguf-py `quants.dequantize -> quants.quantize` (no custom kernels; gguf-py has
no Q6_K/Q4_1 quantizer and none is needed - we only dequantize FROM those types).
Selftest on real source blocks: Q6_K->Q8_0 requant rms_rel 0.0054-0.0074,
Q4_1->Q4_0 0.083-0.084, both deterministic.

Output: /root/taskX1/Qwen3.8-27B-Q4_0-headq8-ffndq40.gguf
  sha256 cb36f5e6c53bea3f6a3c201f10a178046c5652180209fd66697bc841af76c266
  16,612,120,288 B (source 16,348,767,968 B; disk gate 34.9 GB free before write)
  * output.weight            Q6_K 1,042,944,000 B -> Q8_0 1,350,860,800 B (+308 MB)
  * blk.{0..7}.ffn_down.weight Q4_1 55,705,600 B -> Q4_0 50,135,040 B each (-44.6 MB)
  * the 8 attn_q Q6_K tensors left untouched (out of scope: brief = head + ffn_down)
Conversion wall time 50 s.

Verify (x1_verify.py, VERIFY PASS):
  * 866 tensors; names+order+dims identical; dtype diffs EXACTLY the expected 9
  * KV+header bytes identical; output gap cross-check PASS
  * ALL 857 unmodified tensors md5 byte-identical (14.85 GB streamed compare)
  * modified-tensor sample dequant vs source dequant: head rms-rel 0.0054/0.0074
    (head/tail), ffn_down ~0.083 - matches selftest bounds

## 3. KLD gate - GREEN

taskQ narrow-window protocol: llama-perplexity -b 8 -c 512, 64KB wikitext sub
(/root/taskW3/wiki-sub.txt), f128q4kv arm env (-ctk q4_0 -ctv q4_0, no W4A4 levers),
TP2 pair 1,2, GGML_TP_AR_MTP_SAFE=1, production-env via the driver.
Base logits: /root/taskW3/w3-f128q4kv-tp2.logits = ORIGINAL model, SAME f128q4kv
config, SAME binary, captured 2026-11-15 13:54 (reuse per the taskV1 kld27 precedent).

    Mean KLD:  0.007962 +/- 0.000685    Maximum KLD: 4.389359
    Mean PPL(Q): 5.693266    PPL ratio: 1.001194 (+0.12%)

Inside the accepted q4_0-KV class (taskU q4KV 0.006855 - same order), 7x under the
worst accepted floor-0 x q4KV 0.055 AMBER. The requant numerics themselves are clean.

## 4. Perf A/B at the champion config (f128q4kv-p-d4, TP2 pair 1,2, solo legs)

| leg (modified GGUF)     | tg256 med | pp4096 | accept | mean len | probe sha16        |
|-------------------------|----------:|-------:|-------:|---------:|--------------------|
| f128q4kv-p-d4-r1        | 45.66     | 546.5  | 0.3681 | 2.4723   | 77841cb6a251235d |
| f128q4kv-p-d4-r2        | 45.62     | 547.2  | 0.3681 | 2.4723   | 77841cb6a251235d |
| baseline (taskW3)       | 58.20     | 545    | 0.4377 | 2.7508   | canonical          |

* tg256: **-21.6% DECISION-GRADE** (reps within 0.04 t/s of each other; acceptance
  deterministic to 4 decimals). pp4096: +0.3% flat.
* Greedy probe: BYTE-IDENTICAL to the canonical sealed bytes on both legs. The brief
  anticipated a legitimate sha divergence (head weights changed); the fixed 64-token
  greedy trajectory survived the requant, so byte-identity held and the numerics
  evidence is instead the KLD (0.008) + acceptance delta above.
* Repack engagement (server-log gate banners, witnessed per leg):
  - `model shape hint for the Q8_0 repack admission: ... dense Q8_0 2.01 GiB
    (min 2.00 GiB) ... gate OFF` - the Q8_0 head pushed the model OVER the shape-hint
    floor (0.75 -> 2.01 GiB): the head is now QUALIFYING, but
  - `GGML_CUDA_REPACK_Q8_0 gate inactive (env mode 0)` - the champion arm sets no
    env, so the head is NOT admitted; it is served by the canonical Q8_0 stock path.
  - The 8 new Q4_0 ffn_down tensors DID enter the repack lane (`REPACK_Q4_0 gate
    ACTIVE`, `repacking q4_0 tensors` on both devices).
* Per-prompt attribution: code-discussion (acceptance unchanged 0.8884) tg -0.9%;
  the loss concentrates where acceptance collapsed: dialogue 0.6421->0.3949,
  econ 0.6607->0.4663, narrative 0.17-0.23. Mechanism: the canonical Q8_0 head byte
  cost (+308 MB on the largest matvec) is near-neutral at decode width (-1% on the
  acceptance-neutral prompt), but the requant's logit drift (KLD 0.008) degrades
  DFlash2 drafter agreement with the target: acceptance -15.9%, mean len -10.1%,
  and that IS the -21.6%.

## 5. Exploratory counterfactual (Q8_0 repack env ON)

f128q4kvq8-p-d4-r1 = champion + GGML_CUDA_REPACK_Q8_0=1 + STOCK_ACC=0 + RP_Q8_NC_LDS=8
(1 rep, exploratory - NOT an adoption arm; the repack accumulation is explicitly
"not bit-identical to stock" per its own banner and was not separately KLD-scored).
Engagement witnessed: `REPACK_Q8_0 gate ACTIVE (env mode 1)` + `repacking q8_0
tensors on device 0/1` + `RP_Q8_NC_LDS=8 ACTIVE` - the head DID enter the repack/LDS
lane here. Result: tg 45.56, pp 546.3, accept 0.3853/2.5412, probe still canonical.
Admitting the head to repack buys NOTHING over the canonical Q8_0 path (45.56 vs
45.64, within noise) - the acceptance deficit dominates tg regardless of the head's
kernel lane.

## 6. Verdict

**REJECT the requant path as-is.** Gates: KLD GREEN (0.007962, inside the q4_0-KV
class), PPL +0.12% GREEN, perf RED decision-grade (tg -21.6% at the champion config,
deterministic reps).

Answers to the lane's questions:
1. Did the larger matvec enter repack/LDS coverage at the champion config? **No.**
   The Q8_0 head pushed the per-model shape hint over its floor (dense Q8_0
   0.75 -> 2.01 GiB vs min 2.00 GiB - the head is now QUALIFYING) but the champion
   arm runs REPACK_Q8_0 at env mode 0 (gate inactive), so the head was served by the
   canonical Q8_0 stock path. The 8 Q4_1->Q4_0 ffn_down tensors DID enter the repack
   lane (REPACK_Q4_0 is default-ACTIVE).
2. Did it pay? **No, and it would not have even if admitted.** The canonical Q8_0
   head's +308 MB byte cost is near-neutral at decode width (-0.9% on the
   acceptance-neutral code-discussion prompt), and the forced-repack counterfactual
   (sec 5) is indistinguishable from canonical. The -21.6% is the DRAFT ACCEPTANCE
   collapse: the requant's logit drift (KLD 0.008) puts the target out of agreement
   with the unchanged DFlash2 drafter (0.4377 -> 0.3681, mean len 2.75 -> 2.47),
   concentrated on the low-predictability prose prompts.
3. Byte-identity: the greedy probe stayed byte-identical to the canonical
   77841cb6a251235d on all legs (the brief anticipated a legitimate divergence;
   identity held, so no substitution of acceptance/PPL evidence was needed - they
   are reported anyway and are what actually moved).

What a follow-up lane would need before revisiting: (a) a drafter retrain or head
re-graft against the requantized target - the acceptance mismatch is the binding
constraint, not the kernel lane; (b) if the head is ever repack-admitted by default,
mode 2 already qualifies this file (2.01 GiB hint) - but per sec 5 that alone buys
~0, so the gate change is not worth shipping for this model. The ffn_down Q4_1->Q4_0
migration is the only piece with a (small, pp-flat) positive story; on its own it
does not justify the head requant.

## 7. Ops + artifacts

* flock /root/gpu.lock -w 2700 per leg; zero LOCKFAIL (one sibling lane active,
  pair-free gate honored). TERM-only teardown; `pgrep -x llama-server` empty after
  every leg (final sweep 0). llama-swap untouched. Production env asserted by
  mi50-run.sh + production-env.sh, env+maps witnessed per leg, zero RUNTIME-FAIL.
* /root/taskX1/Qwen3.8-27B-Q4_0-headq8-ffndq40.gguf - the modified GGUF (deliverable).
* /root/taskX1/{x1_selftest.py,x1_requant.py,x1_verify.py,convert.log,verify.log,
  chain.log}; /root/taskX1-drive.sh, /root/taskX1-chain.sh; data in /root/taskX1/data/
  ({f128q4kv-p-d4-r1,f128q4kv-p-d4-r2}.{server.log,reps.jsonl,reps.jsonl.agg.json,
  bench.log,probe-*.json,witness,proc.witness,gate.witness,prof-count,run.out,.done},
  kld-f128q4kv-mod-vs-orig.log(+.done)).
* Disk after: ~16 GB free on / (was 33). The modified GGUF is the lane's deliverable;
  reclaim it if a later lane needs the space.
