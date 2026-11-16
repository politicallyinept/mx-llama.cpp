# taskL1 -- CV fork fused decode kernel: port-status audit

Date: 2026-11-16 (PST). Lane: int4-l1-cvport @ e5c91714e3 (INT4-noquant tip).
Auditor: L1 subagent. No GPU legs run; /root/gpu.lock untouched; probe
77841cb6a251235d untouched; GPU0 burn-in and suspect GPU2 untouched.

## 1. Verdict -- the answer to Brad

**YES, the CV fork's fused decode kernel is already ported.** The standing
belief ("NO -- it lives only in /root/cvport-new-src") is wrong.

- `attn_qk_fused` (the CV fused decode attention q/k chain: q/k norm + mul +
  rope + k/v cache rows in one kernel) is in our tree at
  `ggml/src/ggml-cuda/rope.cu`, **byte-identical** to the CV lineage
  (`diff` of the two rope.cu files: 0 lines).
- Gate `GGML_CUDA_ATTN_QK_FUSED` is **default ON** in our tree (rope.cu:1103-
  1105, "anchor-final flip, byte-clean 0/28, Brad-approved"); dispatcher
  wiring at ggml-cuda.cu:5771-5777.
- It is **live in the current champion legs**: the witness line
  `[attn-qk-fused] decode q/k chain fusion ACTIVE` appears in
  /root/taskW1/w1-meas-p-d4.server.log (the 58.20 f128q4kv-p-d4 measurement).
- The entire CV fused-decode commit series exists in our lineage as
  cherry-picks (subject-identical, hash-mapped in sec.4).

Consequence: deliverable-B steps (port, build, microbench, e2e) are **not
applicable** -- there is nothing to port. No build was made, no microbench, no
e2e leg. The 144 t/s MTP-on reference is NOT the fused decode kernel (sec.5).

## 2. Port-status table

| Piece (CV lineage) | Where it lives in CV | INT4-noquant status | Serving-relevant |
|---|---|---|---|
| Fused decode attn q/k chain `attn_qk_fused` (norm+mul+rope+k/v rows, one kernel) | rope.cu (CV-port @7a18553e4) | **PORTED**, byte-identical, gate `GGML_CUDA_ATTN_QK_FUSED` default ON | YES -- active in 58.20 champion |
| Fused shared-expert / dense-f32 decode path (SHE_FUSE) | she-fuse.cu | **PORTED** (ours 69d45ca141), gate `GGML_CUDA_SHE_FUSE` default ON | YES |
| GDN decode gate-chain fusion (gdn_ab_dec, CV hc-dec.cu port) | ggml-cuda.cu fusion dispatcher | **PORTED**; gate `GGML_CUDA_GDN_AB_DEC` default OFF (CV default ON -- deliberate divergence, documented in comment) | Marginal (GDN models only) |
| GDN state-rows gather elimination (gdn_state_rows) | ggml-cuda.cu | **PORTED**, default ON | YES |
| GDN verify-batch state gather split (state-rows fusion admits K>1) | CV 94bad4262 | **PORTED** (ours 189441b3da), env-gated | YES (width>1 verify) |
| KV-only MTP prefill replay (qwen35moe) | CV 214d5f3ca | **PORTED** (ours 2babc21729) | MTP path |
| mtp-replay per-subgraph graph replay on token-graph failure | CV d2a5766b1 | **PORTED** (ours 55473f5f05) | MTP path |
| csar split-uid stamp (aperture-violation fix) | CV 6e1e10261 | **PORTED** (ours 5116802839) | YES |
| gcn_kq lane system (gcn-kq-matvec, gcn-q8-gemm) | ggml/src/ggml-cuda/ | **PRESENT**, byte-identical (not in the differ list) | YES |
| W4A4 / DOT8 / u4x16 machinery (mmq_gemm_repacked, w4a4-i8narrow) | q8_repack/ | **PRESENT**, ours strictly ahead (repack-kernels.cuh: 13 CV-only vs 485 ours-only lines) | YES |
| Repack admission ladder | q8_repack/ | **PRESENT** (agreed) | YES |
| IQ-class fast lanes (IQ4_XS paths in mmvq/mmq/moe-vec/convert) | same file set | **PRESENT**, identical file set, byte-identical (not in differ list) | YES |
| draft-mtp (native nextn spec type) | common/speculative.cpp | **PRESENT** (`--spec-type draft-mtp`; ours +422 lines ahead) | YES |

## 3. Tree relationship (cvport-new-src vs INT4-noquant)

- `ggml/src/ggml-cuda` file sets: **zero CV-only files**. Our tree adds
  ar-quant.cu/.cuh and rq-fattn.cu/.cuh (the qwire port, X2).
- 10 differing files, all strictly ahead on our side (CV-only / ours-only
  lines): ggml-cuda.cu 14/410, q8_repack/repack-kernels.cuh 13/485,
  q8_repack/mul-mat.cu 8/355, fattn-common.cuh 7/129, fattn.cu 1/24,
  gdn-conv-scatter.cu 0/38. The CV-only lines are superseded legacy (old
  BF16-NCCL allreduce replaced by ar-quant; old template signatures;
  truncated comments our tree expanded).
- common/speculative.cpp 5/422, common/common.cpp 0/73, src/llama-model.cpp
  14/178 -- same pattern.
- The CV lineage is vendored in our repo: 7a18553e4 (CV-port tip in
  cvport-new-src) is reachable from our refs cv-port-20261105c,
  gitea/b6-concat-wide, gitea/b7-tensile-router.
- **fb18241213 is a dangling commit in OUR repo** (on no branch) -- it is the
  PROF-PLAN-dflash-step.md line-cite base for our worktrees, not a CV base.
  The task premise "CV lineage (fb18241213 base)" inverts this.
- cvport-new-src working tree clean (no uncommitted CV work, no stash).

## 4. Cherry-pick map (CV-port -> INT4-noquant, subject-identical)

| CV hash | Our hash | Subject |
|---|---|---|
| d8be4aacd | a4fe5060d2 | cuda: fuse decode attention q/k chain (norm+mul+rope+k/v cache rows) behind gate |
| 3acf0a0b8 | a99cbb8bb0 | cuda: byte-match fused attn q/k chain norm reduction per side |
| a08a9f58d | 31ef7df86f | anchor-final: attn_qk_fused default ON -- byte-clean 0/28, Brad-approved |
| 80a94e141 | 42eb9b3999 | ggml-cuda: restore attn_qk_fused call site (tier3-adopt part B fold-in) |
| b4ab4de5b | 69d45ca141 | cuda: port CV fused shared-expert/dense-f32 decode path (SHE_FUSE) |
| 94bad4262 | 189441b3da | llama: split GDN verify-batch state gather so state-rows fusion admits K>1 |
| 214d5f3ca | 2babc21729 | p2: KV-only MTP prefill replay for qwen35moe |
| d2a5766b1 | 55473f5f05 | mtp-replay: restore per-subgraph graph replay on token-graph failure |
| 6e1e10261 | 5116802839 | csar: stamp split uids post-allocation |

## 5. Where the 144 t/s MTP-on number actually ran

Source: `/root/dflash-eval/report.md` (overnight sweep 2026-11-06/07).

- Build: `/root/build-ship`, tree `/root/mtp-pipe-src` @ cd57c8c19d --
  7a18553e4 (CV-port tip) IS an ancestor of that HEAD, i.e. the CV lineage.
- Cell: Target A = Qwen3.6-35B-A3B UD-Q4_K_XL TP2 (production MoE),
  **MTP-on n_max 2 = native nextn-head spec decode, verify width 3**:
  tg256 144.0 t/s, accept 0.596, accept-len 2.19. (Target B dense: 45.3.)
- The 144 is therefore the **native MTP (nextn) path**, not the fused decode
  kernel. The fused decode q/k chain is a T=1 decode fusion that is ON in
  both trees and orthogonal to the gap.
- The 2.6x gap vs DFlash2 (55 -> our 58.20 champion) is verify-width cost:
  width-8 verify ~ 3x a T=1 step on our T=1-optimized kernels (report sec.4),
  plus MTP's structural advantage -- the draft rides the verify's weight
  reads at ~0 marginal cost (taskW1 sec.7.3, taskX3 sec.5). That is step
  machinery (verify pass ~33-42 ms + draft 9.26 ms), consistent with
  PROF-PLAN-dflash-step; it is not a missing-kernel gap.
- Our tree supports the same path (`--spec-type draft-mtp`), so the MTP lane
  is reproducible on INT4-noquant if the profiling lane wants a same-tree
  baseline.

## 6. Unported diffs found (non-fused-decode; flagged, not blocking)

1. `src/llama-kv-cache-dsv4.cpp`: CV adds a `get_can_checkpoint()` override
   returning false ("compressed attention state cannot be restored
   losslessly from a partial sequence checkpoint"). Our tree has only the
   base virtual (`llama-memory.h:109`, returns true) -- but our lineage
   carries b4cab9458e "server: align DSV4 prompt checkpointing with
   upstream", which may supersede it. **Action for whoever owns the dsv4
   checkpoint path: confirm b4cab9458e subsumes the safety; if not, port the
   6-line override.** Not on the DFlash2 champion serving path.
2. `GGML_CUDA_GDN_AB_DEC` default: CV ON, ours OFF -- documented as
   deliberate; revisit only if a GDN model lane wants it.

## 7. Evidence index

- rope.cu byte-identity: `diff /root/cvport-new-src/ggml/src/ggml-cuda/rope.cu /root/int4-noquant-wt/ggml/src/ggml-cuda/rope.cu` -> 0 lines.
- File-set diff: `diff -rq` of ggml/src/ggml-cuda -> 10 differing files, 0 CV-only files.
- Cherry-pick map: `git log --oneline -i --grep=fused` on CV-port (cvport-new-src) vs INT4-noquant (ours).
- Runtime witness: `[attn-qk-fused] decode q/k chain fusion ACTIVE` in /root/taskW1/w1-meas-p-d4.server.log and /root/taskW1/rocprof-draft/server.log.
- Gate + wiring: rope.cu:1101-1261, ggml-cuda.cu:5771-5777 (ours).
- 144 t/s provenance: /root/dflash-eval/report.md L4 (build), L30 (number); `git merge-base --is-ancestor 7a18553e4 HEAD` in /root/mtp-pipe-src -> YES.
- Ancestry: 7a18553e4 reachable from our refs (git for-each-ref --contains); fb18241213 on no ref in our repo.
