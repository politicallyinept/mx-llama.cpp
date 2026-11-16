# taskD0 LEDGER - determinism chase (E5), q6lds d7 e2e leg pair (E4), K1/K2 ledger corrections (E1)

Date: 2026-11-16 (PST). Box: mi50. Branch `int4-d0-determ` off `int4-fix4` @ e3dac6d835
(NOT pushed; parent merges). GPU0 under L3b burn-in - never touched; GPU2 (recovered from
SMC wedge) never touched; all GPU work on GPU1 (mb) and pair 1,3 (e2e) under
`flock /root/gpu.lock -w 2700` per leg incl. retries; TERM-only teardown.
Canonical probe sha16 = 77841cb6a251235d (16 hex, sha256(content)[:16] of the greedy
64-tok temp0 seed42 GPROMPT probe - method re-verified against /root/taskV1 probe files:
all five d7/d4 cells hash to 77841cb6a251235d, 327 chars).

## 1. E1 - taskK1/taskK2 ledger corrections (DONE, commit e3dac6d835 on int4-fix4)

The shipped ownhead graft drafter head facts (re-verified live by taskD0 via GGUFReader on
`/root/taskR/draft/DFlash2-Q4_0-ownhead-q4.gguf`): `output.weight` = **Q6_K (type 14),
[5120, 248320], 1,042,944,000 B = 0.971 GiB** - not the Q8_0 [2048, 248320] 0.50 GiB
stand-in taskK1 benched. Real head kernel time from the taskW1 `rocprof-draft` trace
(`mul_mat_vec_q<(ggml_type)14, 5>`, grid_x=7946240 = 248320 rows at rpb=2/64-thr):
n=46 launches, **min 3.488 ms / median 3.601 ms** => **~3.5 ms of the 9.26 ms draft slice
(~38%)**, not 1.43 ms (~15%). Corrected split ceiling ~-1.6..-1.7 ms/step (~3.6-3.8% of
the 44.8 ms step) - above the 2.9% tg256 noise floor, so K1's "ceiling below noise"
argument does not survive; **the REJECT verdict stands on its own e2e** (gate-ON measured
the real model: +2.4 ms/step draft phase, tg256 -5.1%; meta overhead ~4.1 ms ~ 2.4x the
corrected ceiling). Added CORRECTIONS block + inline markers to taskK1-LEDGER.md sec
1/3/7; appended a correction note to taskK2-LEDGER.md pointing at the fix4 header fix
(468f0bbd84: "bit-exact only at width 5").

## 2. E5 - T=1 generic-path determinism chase (K2 sweep footnote)

CLAIM under test (taskK2-LEDGER Notes 1): at T=1, Q6_K 5120x248320, the GENERIC path
(stock mul_mat_vec_q; the K2 gate provably does not route at ncols_dst=1 - the hook
requires ncols_dst>=2, verified in mmvq.cu) produced two different dst hashes across
gate-off-equivalent fresh-process runs (base != W0=W4=dd088cf64c417abe; W8=626c9fd66467559f).

PROTOCOL: N=20 fresh-process runs of the exact mb shape, `/root/mb_k2_head graph 1 248320
15 5120` (mb md5 3201eb6925637b442b1b3ccf0dc58692 against libggml-hip md5 e56c6617c6b47f1a
= build-k2, identical to the K2 sweep binary per its ledger note 3), GPU1
(HIP_VISIBLE_DEVICES=1), gate env vars explicitly unset, TheRock LD_LIBRARY_PATH as in the
K2 sweep driver, each run under `flock -w 2700 /root/gpu.lock` with up to 8 retries (the
L3b burn-in held the lock until ~13:00). Driver /root/taskD0/supervisor.sh, data
/root/taskD0/e5-hashes.tsv.

RESULTS (20 runs, 13:02-13:45 box time, ~2 min/run on an idle CPU):

| runs 1-20 | hash             | within-run nondet | rc |
|-----------|------------------|-------------------|----|
| all 20    | dd088cf64c417abe | none              | 0  |

20/20 fresh processes produced the SAME dst hash, `dd088cf64c417abe` - which is exactly
the hash K2's sweep recorded for its W0 and W4 T=1 rows. Neither K2's W8 hash
(`626c9fd66467559f`) nor K2's uncaptured base hash (recorded as != dd088...) appeared in
any of the 20 isolated runs. No NONDETERMINISTIC lines (within-run determinism held in
all 300 graph computes).

SUPPLEMENT (gate-env variants, 3x W=8 + 3x W=4, same shape, env
`GGML_DFLASH_HEAD_KERNEL=1 GGML_DFLASH_HEAD_W={8,4}`): K2's differing hashes came from runs
that had the gate ENV set (W4/W8 rows) even though the gate provably cannot route at
ncols_dst=1 (the hook in mmvq.cu requires `ncols_dst >= 2 && ncols_dst <= 5 && ne00==5120
&& ne01==248320` - re-read in the fix4 tree). The supplement tests whether the env itself
perturbs the T=1 generic hash:

| arm | runs | hash             | within-run nondet |
|-----|------|------------------|-------------------|
| W=8 | 1-3  | dd088cf64c417abe | none              |
| W=4 | 1-3  | dd088cf64c417abe | none              |

All six gate-env runs hash identically to the 20 gate-off runs: the gate env does not
perturb the T=1 generic path, and K2's W8 hash `626c9fd66467559f` did not reproduce even
under the exact env that produced it.

VERDICT: **does not reproduce in 26 isolated fresh processes** (20 gate-off + 6 gate-env).
Conditions recorded: GPU1 (HIP_VISIBLE_DEVICES=1), gate env vars unset, TheRock
LD_LIBRARY_PATH prefix identical to the K2 sweep driver, IT=15, each run a fresh
`mb_k2_head` subprocess under `flock -w 2700 /root/gpu.lock` (run 1 needed 2 lock attempts
behind the L3b burn-in; runs 2-20 acquired on attempt 1). Differences vs K2's in-sweep
context: (a) K2's T=1 base run was the FIRST GPU run after the box reboot (cold VRAM),
mine ran after burn-in + prior mb traffic had cycled the card; (b) K2's runs executed
back-to-back inside one sweep session (sequential subprocesses, prior run's VRAM state
inherited), mine likewise sequential but spread over a longer wall window; (c) K2's W4/W8
rows carried the gate env - covered by the supplement. If the phenomenon is an
uninitialized-buffer read whose content depends on process-specific VRAM allocation
state, isolation in fresh processes reproduces it only when the allocator hands out
differing dirty memory - 20 same-binary consecutive runs on the same card plausibly
recycle the same block with the same residue, which would mask it. This is consistent
with every observation K2 recorded (within-run determinism held; cross-process variance;
T>=2 unaffected - different geometry). No bisect was triggered (nothing to bisect: zero
variance observed). compute-sanitizer is NOT installed on this box (only rocgdb); the
prepared bisect path (mb_k2_head_d0: PRE-hash + memset(dst,0xAA) + dst dump + byte-diff)
is staged at /root/taskD0/ and unused.

STANDING RULE recommended for future sweeps: **every hash-claiming mb run must be a fresh
process AND must record its dst hash in the run record** (K2's base-run hash was lost,
which is why the footnote is only half-evidenced); a hash claimed from a run whose
predecessor's VRAM state it inherits is not a reproducibility claim. If a cross-run hash
difference is ever seen again, rerun the EXACT arm env (including gate vars) in >=5 fresh
processes before escalating - the W8-vs-W0 difference at T=1 here did not survive
isolation.

## 3. E4 - q6lds (fix4 best arm) e2e on f0ldsq4kv-d7 - BLOCKED (llama-swap occupancy)

Prepared and staged, legs NOT run. Setup complete and verified:
- Build: /mnt/old/build-fix4 (mi50-build.sh, source line `int4-fix4 @ e3dac6d835, 0 dirty`,
  TheRock amdclang++ pinned, BUILD-RC=0, all 162 HIP objects fresh, no compile errors).
  Gate string present: `strings bin/libggml-hip.so.0.26.0 | grep -c RP_Q6K_LDS` = 2
  (the getenv lives in libggml-hip, NOT in the thin 17 KB llama-server main - a
  zero-count on llama-server is expected and was a false alarm during the chase).
  An initial "gate string missing" ABORT was a bug in THIS lane's preflight
  (`strings | grep -q` under `set -o pipefail`: grep -q SIGPIPEs strings -> rc 141 ->
  false abort); fixed by dumping strings to a file first. No rebuild was needed.
- Retarget verified in-tree: `rp_q6k_nc_lds<T_, W_, 2, false, 2>` (STEP=2, act-only) =
  exactly the mb best-arm template args; `GGML_CUDA_RP_Q6K_LDS=4` -> W4 S2 act = the
  measured-best arm (+0.87 ms/head/card at T=8, TP2 serving shape).
- Leg driver: /root/taskD0/supervisor2.sh `e4` - f0ldsq4kv-d7 arm env byte-identical to
  taskU/taskV1 (`GGML_CUDA_W4A4_MIN_TOKENS=0 GGML_CUDA_REPACK_Q8_0=1
  GGML_CUDA_REPACK_Q8_0_STOCK_ACC=0 GGML_CUDA_RP_Q8_NC_LDS=8` + `-ctk q4_0 -ctv q4_0`,
  dflash drafter DFlash2-Q4_0, `--spec-draft-n-max 7`, taskV1 measure protocol: probe_pp
  warmup, 3x tg256 warm curls, bench.py natural suite, greedy probe, acceptance from
  server log), pair PINNED 1,3, legs off-r1/on-r1/off-r2/on-r2 (paired-interleaved x2),
  ports 8471-8474, flock -w 2700, TERM-only teardown, refuse-if-exists .done markers.
  Gates to be applied by d0-agg.py: probe sha16 == 77841cb6a251235d on ALL four legs
  (kernel is bit-exact per K3 mb 96/96, so ON arms must not move it); acceptance
  digit-match vs the 52.09-class reference 0.3301/3.3105 (bench.py AGG convention);
  tg256 ON-vs-OFF mean delta vs noise floor = |off-r1 - off-r2| / mean(off).

BLOCKER: llama-swap (PID 237, user-facing daemon on :8181) loaded
`Wicked-Tutu-26B-A4B-Q8_0-hb16.gguf` (server PID 34409, `--device ROCm1,ROCm2,ROCm3`) at
11:50 and holds ~11-12 GB on GPU1/2/3 indefinitely (no TTL on that entry; GPU util 0% -
idle but squatting the benchmark pair). The lane's VRAM gate correctly refuses to launch
(`GPU-BUSY-13 retry ... foreign=34409`); doctrine forbids killing llama-swap servers and
forbids treating "lock free" as "GPU free". E4 needs ~1 h of quiet pair 1,3.
UNBLOCK: unload Wicked-Tutu via llama-swap (or user OKs co-schedule, not recommended for
a ~2% question), then rerun `/root/taskD0/supervisor2.sh` (idempotent: E5 + supp + any
completed legs skip via .done markers; ground truth of leg completion = the four
/root/taskD0/e4/<tag>.done files, NOT e4.done which the driver touches even on gate
failures). Decision rule on results: delta inside noise floor -> archive the q6lds kernel
with the negative recorded; delta >= +1% with probe sha holding on both ON legs ->
recommend `GGML_CUDA_RP_Q6K_LDS=4` for the d7 entry.

## 4. Files

- /root/taskD0/: supervisor.sh + supervisor2.sh (fixed preflight; idempotent E5->E4
  driver, rerun to resume E4), esupp.sh/esupp-now.sh (gate-env supplement),
  supervisor.log (+.out), e5-hashes.tsv (20 rows), e5-supp.tsv (6 rows), e5-run-*.out/.err,
  mb_k2_head_d0{.cu,} (bisect variant: PRE-hash + optional memset + dst dump; staged,
  unused), d0-agg.py (E4 aggregator: probe sha / acceptance digit-match / tg medians /
  noise floor), d0-diff.py (dst-dump byte diff; staged, unused), e4/ (empty - legs blocked)
- /mnt/old/build-fix4: fix4 build (BUILD-PROVENANCE src_sha=e3dac6d835 dirty=0,
  llama-server md5 36f9fca3b22a86c27b0a639c83b2ba43, gate string present x2 in
  libggml-hip.so.0.26.0)
- Branch int4-d0-determ: this ledger. Branch int4-fix4 += e3dac6d835 (E1 ledger fixes).
  Neither pushed.
