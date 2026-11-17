# taskL4 LEDGER - llama-swap serving-config update (fleet binary ab115415bf)

Date: 2026-11-16 (PST). Branch: int4-l4-fleetcfg from INT4-noquant tip ab115415bf.
Operator: hermes subagent. Owner authorized the full parked serving-config update.
No repo changes; ledger-only commit; NOT pushed.

## 1. Build (step 1)

- Fresh build of tip ab115415bf via /root/mi50-build.sh into /mnt/old/build-fleet.
- /mnt/old/build-fleet/BUILD-PROVENANCE.txt: src_sha=ab115415bfa0a2c2e849ba69e9793480b89755a9,
  dirty_files=0, built_at=2026-11-16T20:00:23-08:00, TheRock-pinned (HIP 7.2.53211).
- Recorded md5s verified against files on disk:
  llama_server_md5=9f04613379a41f54a32346c61fc5ff31 (MATCH)
  libggml_hip_md5=23e8a3a2001bbd74aabbb451db073501 (MATCH)
- Gate strings present in libggml-hip.so (pipefail-safe strings|grep, per taskD0 lesson):
  GGML_CUDA_RP_Q6K_LDS, GGML_CUDA_RP_Q4_NC_LDS, GGML_CUDA_RP_Q8_NC_LDS,
  GGML_CUDA_REPACK_Q8_0, GGML_CUDA_REPACK_Q8_0_STOCK_ACC, GGML_CUDA_W4A4_MIN_TOKENS,
  GGML_CUDA_FA_DOT8, LLAMA_DFLASH_ADAPTIVE, LLAMA_DFLASH_SHALLOW.
  (GGML_META_TG_INNER_PER_UID present but NOT adopted; stays OFF per H.)
- Drafter file: /mnt/old/home/brad/models/DFlash2-Q4_0-ownhead-q4.gguf,
  sha256 7920500c27e09edb99f46b02659c2074ad42c9619415323435f911b476f64578
  (byte-identical to /root/taskR/draft/ source; copied for path stability).

## 2. Config audit (step 2)

32 entries. DFlash2 spec entries: Qwen3.8-27B-Q4_0 (d4 champion class), Qwen3.8-27B
(Q8_0 chat class). Embedded-MTP spec entries (adaptive N/A): RVN-Flash-Next-MTP,
Ornith-Agents-A1-3.6-35B-A3B, Qwen3.6-27B-Fable-Fusion-711-MTP, Huihui-Qwen3.8-27B-
abliterated, Ornith-1.5-MTPv2-trial, Ornith-1.5-35B-A3B-abliterated-Q4, Qwen3.8-27B-
Halloween-Terror-source-Q4_0. 35B MoE plain: Ornith-1.5-35B-A3B-abliterated (owner
already had the E pair). Non-27B/35B entries untouched per dispatch.
Backups: /llama-swap/config.yaml.bak-l4-1794888911 (pre-L4),
config.yaml.bak-l4final-1794890426 (pre-final-edit).

## 3. Changes applied (final state) and justification

macros.llama-server: /root/build-qmerge/bin/llama-server -> /mnt/old/build-fleet/bin/llama-server
  Justification: tip ab115415bf carries all adopted+shipped gates; qmerge build too old.

Qwen3.8-27B-Q4_0  (class A champion; class D applied then REVERTED - see 5.1)
  env: [MTP_SAFE, REPACK_Q8_0 pair, RP_Q8_NC_LDS=8, RP_Q4_NC_LDS=8] -> [MTP_SAFE]
       (class A: floor-ON = no GGML_CUDA_W4A4_MIN_TOKENS; no RP_*_NC_LDS pair)
  cmd: -md ${model_path}/Qwen3.8-27B-DFlash2-Q4_0.gguf -> -md ${model_path}/DFlash2-Q4_0-ownhead-q4.gguf
       + -devd ROCm1 (pinned drafter, first card of entry device list ROCm1,ROCm0)
       + -ctk q4_0 -ctv q4_0
  Justification: f128q4kv-p-d4 champion 58.20 t/s, +3.72%, acceptance bit-equal
  0.4377/2.7508 (taskW3 f128q4kv-p-d4-r1/r2: tg 58.33/58.06, accept 0.4377, len 2.7508).

Qwen3.8-27B (Q8_0 chat)  (class C attempted -> REVERTED to pre-L4 state - see 5.2)
  Attempted: --spec-draft-n-max 4 -> 7 + LLAMA_DFLASH_ADAPTIVE=2 + LLAMA_DFLASH_SHALLOW=4
  (taskT verdict: multi-turn +9.7% vs fixed-d7; flag n-max must stay at deep bound 7).
  Reverted: entry fails to load on the new binary at fleet shape EVEN AT ORIGINAL CONFIG
  (see 4.3/4.5). Pre-L4 state restored exactly.

Ornith-Agents-A1-3.6-35B-A3B  (class E, MoE spec)
  env: + GGML_CUDA_REPACK_Q8_0=1 + GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0
  Justification: taskV2-moe +1.82% plain-decode sealed bit-identical; taskW2: MoE SPEC
  entries same pair sealed PASS (=1 superset of =2).

Ornith-1.5-MTPv2-trial  (class E, MoE spec)
  env: [] -> + same pair. Same justification (taskW2).

Ornith-1.5-35B-A3B-abliterated  (class E, MoE plain)
  NO CHANGE NEEDED: owner already had the pair. Smoke-validated on new binary (4.2).

Class B (27B Q4_0 d7-class f0ldsq4kv + RP_Q6K_LDS): NO MATCHING FLEET ENTRY - the only
27B Q4_0 DFlash2 entry is the d4 champion (class A). RP_Q6K_LDS therefore absent.
Class F (R7 combine), G (QWIRE), H (PER_UID): OFF everywhere; verified absent from config.
Owner's pre-existing GGML_CUDA_W4A4_DOT8=1 + W4A4_MIN_TOKENS=128 on the out-of-scope
123B entries (Behemoth, Monstral) left untouched.

## 4. Smoke results (step 5; all under flock /root/gpu.lock -w 2700)

Witnesses: /root/taskL4-smoke/ (final), /root/taskL4-smoke-leg1/ (leg1 backup),
/root/taskL4-drive.log, /root/taskL4-drive2.log. Greedy probe = canonical prompt,
n_predict 64 temp 0 seed 42; sha16 of content. Mini-acceptance = 1 tg256 rep per
mtp-natural prompt (acceptance deterministic per prompt at temp0 seed42; matches
bench.py protocol temp0/seed42/ignore_eos/cache_prompt-false).

4.1 Q4_0 d4 pinned, FINAL config (no FA_DOT8) - PASS
  health OK; cmdline/env witnesses match config (ownhead -md, -devd ROCm1, q4_0 KV).
  probe sha16 = 77841cb6a251235d on both reps (canonical probe preserved).
  mini: accept=0.4377 meanlen=2.7508 -> DIGIT-MATCH vs taskW3 ledger ref (0.4377/2.7508).
  tg med 57.72 (ref 58.20/58.33/58.06, within run-to-run); sanity tg 53.89.
  Clean unload, VRAM drained to ~21 MB/card.

4.2 Ornith-1.5-35B-A3B-abliterated (35B MoE plain, class E pair) - PASS
  probe sha16 = 77841cb6a251235d both reps (bit-identical class E).
  Banner witness: "CUDA repack: GGML_CUDA_REPACK_Q8_0 gate ACTIVE", "REPACK = 1",
  "Meta(ROCm1_Repacked,ROCm0_Repacked) model buffer size = 1432.08 MiB".
  tg med 113.37, sanity 112.10 (taskV2-moe pair-on ref 111.46). Clean unload.

4.3 Q8_0 chat with class C (n-max 7 + adaptive v2) - FAIL (load OOM)
  Server exit status 1 at ~13 s. Captured trace (/tmp/q8log.txt): main model loads
  (repack mode 1 forces repack: Meta(..._Repacked) = 24696 MiB), then draft load
  "allocating 1033.21 MiB on device 1: cudaMalloc failed: out of memory" ->
  "failed to load draft model Qwen3.8-27B-DFlash2-Q4_0.gguf".

4.4 Q4_0 d4 pinned WITH FA_DOT8=1 (class D as dispatched) - FAIL vs acceptance ref
  probe sha still 77841cb6a251235d (short-KV bytes unchanged), but mini:
  accept=0.3652 meanlen=2.4607 (ref 0.4377/2.7508), tg med 44.86 (-22%).
  Per-prompt: list-structured/mystery match ref closely; tech-explain/dialogue/econ
  drop hard. FA_DOT8 costs -1.5% acceptance at 64k KV per taskS; at ~4k KV on this
  d4+q4_0-KV stack the cost is -16.6% relative.

4.5 AB-A: Q8_0 entry at ORIGINAL config (n-max 4, no adaptive) on new binary - FAIL
  Identical OOM signature at draft load (capture: q8-orig-config.server.log).
  => The Q8_0 load failure is NOT caused by the class C edit; the entry cannot load
  ab115415bf at -c 65536 TP2 + DFlash2 draft regardless of our parked-change edits.
  Journal history since Nov 4 shows no successful <Qwen3.8-27B> load at all, so this
  is likely pre-existing shape breakage, not a regression introduced by the repoint.

4.6 AB-B: Q4_0 entry with FA_DOT8 removed = 4.1 (digit-match recovered exactly).

## 5. Deviations from the dispatch mapping

5.1 Class D (FA_DOT8 on ctx>=32k entries) REVERTED on Qwen3.8-27B-Q4_0.
  Evidence 4.4 vs 4.1/4.6: at the entry's real traffic KV length (~4k) FA_DOT8 costs
  -16.6% relative acceptance and -22% tg; the taskS win lives at 32k+ KV. The entry's
  max ctx (65536) is not its typical KV length. If the owner wants FA_DOT8 for a
  dedicated long-context entry, it should be a separate entry with a measured shape.
5.2 Class C (adaptive v2) NOT APPLIED: entry OOMs at fleet shape even at original
  config (4.5). Also note taskT measured adaptive on the Q4_0 stack, so Q8_0
  application was extrapolation anyway. Entry restored to pre-L4 state.

## 6. Final validation

- YAML parses; llama-swap journal "Configuration Reloaded" with no WARN; /v1/models = 32.
- Forbidden gates absent: QWIRE, PER_UID, RP_Q6K_LDS (whole file); MIN_TOKENS absent
  from every entry this task touched.
- Macro verified live via three successful swap loads (4.1, 4.2, 4.4/4.6).
- GPU left idle: no llama-server processes, VRAM ~21 MB/card, gpu.lock free.

## 7. Open items for owner

1. Qwen3.8-27B (Q8_0 DFlash2 chat entry) cannot load on ab115415bf at -c 65536
   (draft-model OOM on ROCm1, 1033 MiB, reproducible at original config). Needs a
   shape decision: lower -c, q4_0 KV (-ctk/-ctv q4_0), or drafter placement change.
   Until then class C (adaptive v2) cannot ship on this entry.
2. FA_DOT8: keep OFF for short-KV serving shapes (quantified regression 4.4);
   consider only on a dedicated >=32k-KV entry per taskS.
3. Class B / RP_Q6K_LDS: no fleet entry serves the 27B Q4_0 d7-class; the gate is
   shipped but unserved. Add an entry or drop the arm from the next sweep.
