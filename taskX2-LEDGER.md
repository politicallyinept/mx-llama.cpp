# taskX2 — AR qwire prefill bench (first measurement ever)

Date: 2026-11-15 (PST). Lane: X2. Branch: `int4-x2-qwire` @ b0230dc386 (tip 204739d62d + qwire port).
Build: `/mnt/old/home/brad/builds/build-x2-qwire` (llama-server md5 f198f9896595376c2b7c29bcddead719).
Model: Qwen3.8-27B-Q4_0 + borrowed DFlash2 d4 (taskW3 stack). TP2 pair 1,2 (GPU0 quarantined).
Data: `/root/taskX2/` (per-leg jsonl + witnesses), driver `/root/taskX2-drive.sh`, aggregator `/root/taskX2-agg.py`.

**VERDICT: REJECT as a gap-closer (adopt: no).** The wire works and is byte-neutral at probe
scale, but delivers +1.9% pp4096 vs its own RCCL-BF16 baseline and only +1.1% vs production —
~6% of the ~18% prefill gap. The gap is not AR-bound (consistent with the pp-diag precedent:
the fork-vs-CV prefill gap survives AR removal). Shelving the branch; see §6.

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
  `librccl.so.1` linked (resolves from TheRock `_rocm_sdk_libraries` at runtime —
  witnessed from /proc/PID/maps per leg); `strings libggml-hip.so | grep -c "quantized
  AR wire ACTIVE"` = 1.
* First-ever boot of an RCCL-linked build on this lineage: smoke legs (R then Q) booted
  clean on pair 1,2, no wedge, TERM teardown clean. Q smoke banner:
  `ggml_cuda: quantized AR wire ACTIVE (wire=q4_K algo=ag ne=5242880` (= 1024-token ub
  chunk x 5120 hidden; hidden=5120 so pp-chunk AR payloads are 5.24M elements, 20x qmin).

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
sealed 553 pp4096 cell. Legs: interleaved paired reps P,R,Q x 3 reps (17:38-19:05,
sibling taskX1 legs interleaved between ours — paired design absorbs the drift); per
leg: bench.py natural suite (pp4096 + tg256 medians), pp2048 (token-trimmed suite text,
3 cold reps), greedy 64-tok probe (GPROMPT, temp0 seed42, sha256[:16] of content).
QWIRE env witnessed in server /proc/environ on Q legs only; wire banner count 1 on every
Q leg, 0 on every P/R leg.

## 4. Gates — greedy probe byte-identity: PASS in all states

All 9 perf legs + both smoke legs: probe sha16 = **77841cb6a251235d** (canonical; the
brief's "77841cb6a235d" is the known truncation typo per taskW3-LEDGER). P MATCH,
R MATCH, Q MATCH. The wire is byte-neutral at probe scale by construction: probe prefill
(13 tok x 5120 = 66,560) is under qmin 262,144 and decode/verify AR (5,120-25,600) is
under the 32,768 custom-AR gate, so neither rides the wire. No KLD seal was needed for
the gate itself; see §6 for the acceptance-drift caveat that a shipping adoption would
still owe.

## 5. Perf results (medians of 3 interleaved reps; within-arm CV = noise floor)

| leg | pp4096 | pp2048 | tg256 | accept | probe | qbanner |
|---|---|---|---|---|---|---|
| P (production) | 546.6 (cv 0.14%) | 533.8 (cv 0.07%) | 40.4 | 0.3489 | MATCH | 0 |
| R (RCCL BF16)  | 543.0 (cv 0.12%) | 525.4 (cv 0.11%) | 46.2 | 0.3581 | MATCH | 0 |
| Q (q4_K wire)  | 552.2 (cv 0.16%) | 535.9 (cv 0.18%) | 41.7 | 0.3309 | MATCH | 1 |

Paired per-rep deltas (all three pairs same sign in every rep):

| pair | pp4096 | pp2048 | tg256 |
|---|---|---|---|
| Q vs R (the A/B) | **+1.88%** (+1.67/+1.93/+2.02) | **+2.06%** (+1.98/+2.45/+1.76) | -9.86% |
| Q vs P (production) | +1.09% (+1.02/+0.78/+1.46) | +0.37% | +3.95% |
| R vs P | -0.77% | -1.66% | +15.32% |

P reproduces the sealed cell (546.6 vs 553.5 sealed, -1.2%, different session/build era
— within the known cross-era band). Noise floor 0.07-0.18% on pp: the +1.9% delta is
~10-20x the floor.

**tg256 is not a decode-path A/B.** Decode/verify AR (ne < 32,768) rides custom AR
identically in all three arms; the tg medians move because the three arms' *prefill* AR
numerics differ (internal F32 / RCCL BF16 / q4_K), which changes the KV the drafter
scores against: per-prompt tg is deterministic per arm (rep CV < 0.3%) but reshuffled
across arms (e.g. tech-explainer P 44.4 -> R 56.4 -> Q 45.0; math-reasoning P 71.6 ->
R 62.6 -> Q 49.7; narrative-2 P 36.5 -> R 33.6 -> Q 44.9), with pooled acceptance
0.3489 / 0.3581 / 0.3309. So the wire neither speeds nor slows the decode kernels; it
shifts spec-decode acceptance -5.2% rel vs production, which nets to -9.9% suite-median
tg vs R. That is a numerics effect, not a kernel regression.

## 6. Verdict

**REJECT (shelve).** Numbers: closing the 553.5 -> 653.1 ruquant gap needs +18.0%;
qwire delivers +1.1% over production pp4096 (+1.9% over its own RCCL-BF16 baseline) —
about 6% of the gap. The RCCL fallback itself is a *net loss* vs production (-0.8% pp),
so the wire's +1.9% partly just claws back the routing change it lives on. This matches
the pp-diag precedent (fork-vs-CV prefill gap survives `GGML_CUDA_AR_SKIP=1`; the gap is
busy_noar + inter-batch host path, not AR transport) — now confirmed on the int4 lineage
with the actual quantized-wire arm.

Additional adoption costs even for the ~1%: (a) production would have to link RCCL
(a routing + numerics change for every large AR, not just prefill); (b) the q4_K wire
reshuffles spec-decode acceptance (0.3489 -> 0.3309 pooled) — shipping would owe a full
KLD seal on the natural suite before adoption-grade claims; the 64-tok probe MATCH does
not certify 512+-token generations (drift-ladder rule). Prefill-only scoping does not
rescue it: the wire already only engages prefill-sized ARs, and the prefill win is ~1%.

Keep branch `int4-x2-qwire` (port + this ledger) as the reference for the measurement;
do not merge. The prefill-gap hunt should target the busy_noar/inter-batch host path
(pp-diag §, context-checkpoint cost), not AR transport. The old "CUSTOM_AR=1 decode
-25%" memory note was not exercised here (all legs ran the production custom-AR bundle).

## 7. Ops notes

* Sibling lane taskX1 shared /root/gpu.lock throughout; flock -w 2700 + paired
  interleaving held the noise floor at <= 0.2% on pp despite interleaved sibling legs.
* Witness bug fixed mid-run: /proc/PID/maps must be read from the real llama-server pid
  (`resolve_server`), not the `timeout` wrapper mi50-run.sh returns.
* `/root/mi50-build-x2.sh` left on the box (RCCL variant of the sanctioned recipe);
  shared `mi50-build.sh` untouched. Build dir on /mnt/old (root fs at 98%).
