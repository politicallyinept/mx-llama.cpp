# v7 -- llaminar (Llaminar/llaminar develop) vs our fork, Qwen3.8-27B, TP2 on ROCm1+ROCm2

**2026-10-21.** Three engines planned; CV dropped for scope, matrix = llaminar vs fork.
Same-model-bytes discipline: fork runs the SAME files llaminar runs (Q4_0 original;
Q8_0-f32gdn for the Q8 comparison attempt). Cards: ROCm1+ROCm2 (ROCm0 reserved --
external agent), 175 W caps, flock-serialized, 2 rounds where applicable.

## Engines (build-verified)
- llaminar (develop, 2026-10-18), built from source on-box: HIP gfx906,
 AVX2 host (5950X = Zen3, no AVX512 -- their cmake default SIGILLs here), OneDNN
 from-source, **patched RCCL built from source** (rocm-7.2.4 + their 4 patches,
 ONLY_FUNCS scoped, spill-guard pass-through'd + opt-record flags stripped).
- fork (int4-profile), PROFILE=speed.

## Results -- Q4_0, TP2, same cards, same session

| engine | pp | tg | notes |
|---|---|---|---|
| llaminar plain | 415.0 | 46.53 | their harness, warmup+3 iters |
| llaminar MTP (dynamic) | 416.0 | **59.93** | accept 70.5%, **pp tax ~ 0%** |
| fork (PROFILE=speed) | **443.0** [440.8/445.1] | 45.70 [45.68/45.71] | server A/B, 2 rounds |

### Q4_0 verdicts (TP2)
1. **Fork wins prefill: +6.7%** over llaminar (443.0 vs 415.0). llaminar's famed pp
 lead does NOT replicate against our fork at TP2 -- their +70-75% claims are vs
 old stock on their rig; vs our W4A4 fork they are behind.
2. **Plain tg: statistical tie** (46.53 vs 45.70, +1.8% llaminar).
3. **llaminar MTP is the standout: +28.8% tg over its own plain (59.93) with ZERO
 prefill tax (416.0 vs 415.0)** and 70.5% acceptance. Their per-step overhead
 engineering (graph capture, device-owned MTP verify) is real and best-in-class.
 Cross-ref v6: their MTP suite beat our fork-MTP by +21% on cards 0+1.
4. Our unanswered lever remains the **MTP overhead suite port** (est. fork-MTP
 TP2 tg ~54-58 -> competitive) -- unchanged priority, now with TP2 evidence.

## Q8_0: BLOCKED for llaminar -- three documented failures
1. Original blanket Q8_0 (single card, first probe): GDN family rejection --
 "measured prefill producer disagrees with prepared weight family".
2. Q8_0-f32gdn (re-quant from f16, ssm_alpha/beta forced f32, 2 cards): SAME
 GDN family rejection -> alpha/beta precision was not the discriminator.
3. Q8_0-f32gdn + MTP: "no MTP/nextn metadata or tensors found".
**Root cause**: our f16 quantize source is an older conversion (851 tensors, no
nextn block); the working Q4_0 file is a newer conversion (866 tensors, 4 nextn
tensors, +11 aux tensors). llaminar's qwen35 graph expects the newer tensor set.
A compatible Q8 artifact requires a surgical rewrite of the original Q8_0 file
(dequant ssm_alpha/beta q8_0->f32 in place, preserving nextn + tensor set) --
queued as follow-up, not attempted tonight.
**Fork Q8_0-GDN reference (TP2, fresh)**: pp 414.1 [416.0/412.2], tg 35.92
[35.97/35.86] -- matches v7c; fork Q8 pp is ~19% below its Q4_0 pp (f32 GDN
projections quadruple that traffic at prefill -- quantizer-policy insight for us).

## Build recipe findings (for any future llaminar build on this box)
- cmake gates: `-DHAVE_CUDA=OFF -DLLAMINAR_BUILD_TESTS=OFF -DROCM_PATH=/opt/rocm
 -DCMAKE_PREFIX_PATH=/opt/rocm -DLLAMINAR_CPU_ISA=AVX2` (5950X = Zen3: their
 AVX512 default SIGILLs) + gcc-14 (`<print>` needs it) + libssl-dev + ccache
 (their HIP launcher chain breaks without it) + OneDNN from source (required
 by CpuExpertSlotPool) -- OneDNN/-j1 ~ 18 min, 8 GB peak.
- RCCL: stock gfx906 rccl lacks their `llaminarNccl*DeviceRows` symbols -> TP2
 requires their patched build: clone rocm-7.2.4 + their 4 patches +
 `RCCL_ONLY_FUNCS` scoping (full-set device link needs >40 GB, swap-thrashes
 for hours) + spill-guard opt-record flags stripped (single gensrc TU >34 GB
 with them at O3). Final link still needs the 38 GB ceiling. Recipe in
 build-rccl-llaminar.sh + build-rccl-resume.sh (self-healing -O1 per-TU).
- llaminar MTP requires embedded nextn tensors in the GGUF (llama.cpp's
 separate-draft workflow does not apply).

Raw artifacts: mi50:rig/cv-h2h-v7/ (h2h-v7d.log, JSONs, server logs, rccl-build.log,
BUILDS-llaminar.md). Drivers: h2h-driver-v7d.sh, build-rccl-resume.sh, build-llaminar-v2.sh.
Baselines: RESULTS-v5.md (TP2, cards 0+1), RESULTS-v6.md (MTP matrix).
