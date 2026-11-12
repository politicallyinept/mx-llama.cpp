# taskG4 LEDGER - ruquant engine update + DFlash2 8-token-verify replication vs fork head-to-head

Date: 2026-11-12 (PST). Box: mi50 (3x MI50 gfx906, LXC on PVE). No pushes; local commit only.

## 1. Engine update
- /root/ruquant-src: was 7a08316 (9 behind), tree clean, git pull --ff-only ->
  82c73b56263a5f644fb47ae34dfd21831396d025 ("Q8_0 narrow mat-vec on gfx906: activation blocks staged in LDS per workgroup").
- Claim under test (commit message): DFlash2 91 -> 93.3 t/s (one GPU), 105.3 -> 107.4 (TP2), 119.7 -> 120.5 (TP4),
  8-token verify (n_max 7), MI50.
- Published methodology (ruquant-repo RESULTS.md + rig/final_mi50.sh): acc_bench.py on acc_prompts.json,
  draft 7, greedy, drafter pinned -devd ROCm0, env GGML_CUDA_DISABLE_GRAPHS=1 HSA_FORCE_FINE_GRAIN_PCIE=1
  GGML_META_TOKEN_GRAPH=0 GGML_META_SPIN=2000000. Their box: "box 1" 4x MI50 32GB, ROCm 5.7.

## 2. Build (fresh dir per stale-object rule)
- Dir: /root/ruquant-src/build-82c73b5 (old build/ kept intact as same-day control; binary = commit 7a08316).
- Configure line (flags replicated from old build/CMakeCache.txt; only delta on cache diff was LLAMA_OPENSSL,
  forced OFF to match):
  cmake -S /root/ruquant-src -B /root/ruquant-src/build-82c73b5 -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
    -DGPU_TARGETS=gfx906 -DAMDGPU_TARGETS=gfx906 -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_HIP_COMPILER=/root/therock-venv/lib/python3.12/site-packages/_rocm_sdk_devel/bin/amdclang++ \
    -DGGML_HIP_RCCL=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_OPENSSL=OFF
- Build: nice -n 19 cmake --build build-82c73b5 -j 8 --target llama-server; RC=0.
  Logs: /root/ruquant-bench/taskG4/{configure.log,build.log}
- Version witness: new = "0.6.0-dev (build 10, commit 82c73b5)"; old control = "(build 1, commit 7a08316)";
  fork = "(build 11887, commit d67cd96f0f)" = "task E2: repacked nc verify widths 6-8 at 1 warp" (nc fix present).

## 3. Legs (drivers /root/ruquant-bench/taskG4/{drive.sh,drive_old.sh}, artifacts in same dir)
All legs: flock -w 600 /root/gpu.lock, TERM-only teardown, TheRock runtime witness (no apt libamdhip64 mapped),
model sha256 witnesses, quiet-check logged. TP2 = two free cards via HIP_VISIBLE_DEVICES, -sm tensor -devd ROCm0,
-c 8192 -np 1 --spec-type draft-dflash --spec-draft-n-max 7 -bs --samplers temperature.
Harnesses per ruquant leg: /root/p2-prefill/bench.py natural suite (tg256 median over 8 prompts x 3 reps)
+ /root/ruquant-repo/rig/acc_bench.py (their harness; "all" row = the published metric).
Fork leg: same n_max 7 + bench.py; fork model stack (see findings).

## 4. Results (TP2, n_max=7 = 8-token verify, same day 2026-11-12)
| leg | engine | natural tg256 median | acc_bench all t/s | ms/step | mean acc len (acc/natural) |
|---|---|---|---|---|---|
| rqnew-r1 | 82c73b5 | 67.31 | 106.2 | 40.0 | 4.25 / 2.62 |
| rqnew-r2 | 82c73b5 | 67.33 | 105.0 | 40.5 | 4.25 / 2.62 |
| rqold-r1 | 7a08316 | 50.34 | 92.1  | 46.2 | 4.25 / 2.42 |
| rqold-r2 | 7a08316 | 50.61 | 90.9  | 46.8 | 4.25 / 2.42 |
| fork7-r1 | d67cd96f0f | 36.50 | - | - | - / 2.83 |
| fork7-r2 | d67cd96f0f | 36.32 | - | - | - / 2.79 |

Published claim: 107.4 t/s (TP2, post-fix). Our updated engine: median 105.6 (106.2 / 105.0) = -1.7% -> REPLICATED
within rig noise (day-drift band 4-6%; arithmetic check: 4.25 / 105.6 * 1000 = 40.2 ms/step matches the 40.0-40.5 witness).
Same-day old-engine control: 91.5 median (matches Nov 11 redo 94.2 within drift) -> full 9-commit update delta:
acc-all +15.4%, natural +33.4%. The published pre-fix 105.3 was evidently measured one commit behind (with the
other 8 fixes present), not at 7a08316; our stale baseline sat 14% below it.
Fork at the identical 8-token config: 36.41 median (36.50 / 36.32). Fork n_max=4 history was 37.85 (redo legF):
the fork LOSES ~4% going to 8-token verify; ruquant engine GAINS (its whole design point).

## 5. Findings
- REPLICATED: updated ruquant engine reproduces the published DFlash2 TP2 8-token-verify number on this box:
  105.6 median vs 107.4 published (-1.7%), same harness (rig/acc_bench.py), same model files (sha256 witnessed),
  same env recipe. Not an exact match: different box (ours 3x MI50 TheRock runtime vs their box 1 4x MI50 ROCm 5.7).
- Head-to-head at the identical 8-token config (natural suite): ruquant-new 67.32 vs fork 36.41 = 1.85x.
  Decomposition: acceptance is NOT the lever (fork mean accept len 2.81 >= ruquant 2.62); the entire gap is
  per-step speed, attributable to (a) quant scheme: W4A4-GPTQ+OST+online-Hadamard 4.5bpw vs stock Q4_0
  (their own RESULTS.md: W4A4 vs Q4_0 at TP2 = 89.5 vs 60.5 = 1.48x), (b) drafter placement: ruquant pins the
  drafter to one GPU (-devd ROCm0), the fork replicates its Q8 drafter on both cards (their docs price this at 19%),
  (c) engine kernels: the 9-commit ruquant update (LDS-staged Q8_0 narrow mat-vec, GDN fusions, q4_0 attention)
  is absent from the fork; the fork's nc 6-8 fix is included but does not close the class of gap.
- Cross-engine "same model" is impossible: dtype 44/45 (W4A4/W4A4H) exist only in the ruquant tree; the fork's
  GGML_TYPE_COUNT=43 refuses the files. Fork arm ran its own Q4_0 + DFlash2-Q8_0 stack (same axis as redo legF).
- The 8-token verify is a regression for the fork (-4% vs its n_max=4 number) and a win for the updated ruquant
  engine; serving implication: do not raise the fork's draft length to 7.
- Old-engine same-day control re-anchors the redo numbers (natural 50.34/50.61 vs redo 50.77; acc-all 91.5 vs 94.2):
  box is consistent day-to-day within ~3%.
- Driver bug note: first-pass rqold legs passed build/ instead of build/bin (instant SERVERDIED, no GPU work,
  lock released cleanly); fixed in drive_old.sh. No wedge, no kill -9, all teardowns clean.
