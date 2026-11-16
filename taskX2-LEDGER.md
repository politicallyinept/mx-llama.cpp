# taskX2 — AR qwire prefill bench (first measurement ever)

Date: 2026-11-15 (PST). Lane: X2. Branch: `int4-x2-qwire` @ b0230dc386 (tip 204739d62d + qwire port).
Build: `/mnt/old/home/brad/builds/build-x2-qwire` (llama-server md5 f198f9896595376c2b7c29bcddead719).
Model: Qwen3.8-27B-Q4_0 + borrowed DFlash2 d4 (taskW3 stack). TP2 pair 1,2 (GPU0 quarantined).

## 1. Where the code lived (first deliverable)

The qwire code was NOT lost and was never merged to the working lineage — it lives on a
**branch of the CV-port clone**, which is why every search of the int4 repo came up empty:

* **Code**: `/root/cvport-new-src` (a separate clone whose `origin` is the GitHub mirror
  `politicallyinept/mx-llama.cpp`), branch **`perf/ar-qwire`**, tip
  `ae20eda909309bb3b1f9ad6822ba35645febca00` — "ar-qwire: quantized prefill all-reduce
  wire on the RCCL fallback path (GGML_TP_AR_QWIRE, opt-in)", 2026-11-05, +505 lines in
  `ggml/src/ggml-cuda/{ar-quant.cu,ar-quant.cuh,ggml-cuda.cu}`. Its parent chain is the
  p2-prefill lineage (`7a18553` p2 flips), i.e. the CV-port lineage — which has **no
  common git ancestor** with the gitea-lineage `INT4-noquant` branch (sha-different
  mirrored histories), so `git log --all -S QWIRE` in `/root/int4-noquant-wt` finds
  nothing and `git grep qwire` at tip is empty. The brief's guess "CV-port base
  fb18241213 lineage" was close but not it: fb18241213 exists only in the gitea-lineage
  repo (a repack-semantics commit) and is not qwire's base.
* **Prior validation state** (from `brief-response-int4-headroom.md` §7.2): "P1 ar-quant
  wire: PORTED (branch perf/ar-qwire), gate-off byte-identical x2, gate-on engages
  (wire=q4_K). pp4096 A/B pending — queued as next GPU leg." So the code was built and
  smoke-verified on the CV lineage (`/root/cvport-new-src/build`, `GGML_HIP_RCCL=ON`,
  `ar-quant.cu.o` present), but **never benchmarked** — consistent with the brief.
* **Harness/profiles**: the standing benchgate endpoint-harness + taskV1/taskW3-style
  driver legs (bench.py natural suite: pp4096 x1 + tg256 x3 per prompt x 8 prompts,
  greedy 64-tok temp0 seed42 probe). No qwire-specific harness existed; this task reuses
  the taskW3 driver recipe verbatim (see §3).

**Mechanism** (read from the port): `GGML_TP_AR_QWIRE=1` hooks
`ggml_backend_cuda_comm_allreduce_nccl` (inside `#ifdef GGML_USE_NCCL`). The custom-AR
size gate routes TP2 messages with `ne >= 32768` to the NCCL/RCCL fallback; qwire
replaces the BF16-ring payload with quantized blocks (default `GGML_NCCL_AR_WIRE=q4_K`,
algo `ag` at N=2: all-gather of quantized partials, own chunk exact). `GGML_NCCL_AR_QMIN`
(default 262144 elements) keeps decode/verify-sized ARs (ne < 32768 ride custom AR
anyway) off the wire. Production builds link **no** RCCL (`GGML_HIP_RCCL=OFF`), so the
qwire code is compiled out there and large AR uses the internal allreduce.cu pipeline —
measuring qwire therefore requires an RCCL-enabled build, which changes the large-AR
routing by itself. Hence the three-arm design in §3.

## 2. Port + build

* `git worktree add -b int4-x2-qwire /root/wt-x2-qwire 204739d62d`; `git am` of
  `format-patch` of ae20eda — **applied clean, zero edits** (the two lineages'
  ggml-cuda.cu context matched). Commit b0230dc386.
* Build: `/root/mi50-build-x2.sh` = verbatim copy of `mi50-build.sh` + three cmake flags
  (`-DGGML_HIP_RCCL=ON -DGGML_CUDA_NCCL=ON -Drccl_DIR=$TR/lib/cmake/rccl`, mirroring the
  CV-port build's cache) + disk guard fixed to measure the build-dir filesystem instead
  of `/` unconditionally (/ is at 98%; /mnt/old has 149G). Shared `mi50-build.sh`
  untouched. Fresh dir `/mnt/old/home/brad/builds/build-x2-qwire`, TheRock amdclang++
  pinned (verified in CMakeCache), `BUILD-PROVENANCE.txt`: src_sha=b0230dc386, dirty=0.
  `librccl.so.1` linked; `strings libggml-hip.so | grep -c "quantized AR wire ACTIVE"` = 1.

## 3. A/B design

Arms (all legs: flock /root/gpu.lock -w 2700, TERM-only teardown, production-env via
mi50-run.sh + HSA_KERNARG_POOL_SIZE=67108864, HIP_VISIBLE_DEVICES=1,2, env/maps/banner
witnessed per leg from /proc/PID):

| arm | build | GGML_TP_AR_QWIRE | large-AR path (ne>=32768) |
|---|---|---|---|
| P | /root/build-uwave (tip-equivalent, code delta to tip is ledger-only) | n/a (not compiled) | internal allreduce.cu |
| R | build-x2-qwire | unset | RCCL BF16 ring |
| Q | build-x2-qwire | 1 | RCCL + q4_K wire (algo ag) |

R-vs-Q is the brief's "qwire 1 vs off" pair; P contextualizes against production and the
sealed 553 pp4096 cell. Legs: interleaved paired reps P,R,Q x 3 reps; per leg: bench.py
natural suite (pp4096 + tg256 medians), pp2048 (token-trimmed suite text, 3 cold reps),
greedy 64-tok probe (GPROMPT, temp0 seed42, sha256[:16] of content). Driver
`/root/taskX2-drive.sh`, data `/root/taskX2/`, aggregator `/root/taskX2-agg.py`.

## 4. Gates — greedy probe byte-identity (canonical 77841cb6a251235d)

(TBD after matrix)

## 5. Perf results

(TBD after matrix)

## 6. Verdict

(TBD)
