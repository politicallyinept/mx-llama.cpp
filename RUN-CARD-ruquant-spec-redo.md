# RUN-CARD: ruquant speculative-decode redo (Task A)

Status: GPU-QUEUED. Authored 2026-11-10 (PST) from a full non-GPU audit; every path,
flag, hash, and gate below was verified on mi50 on 2026-11-10. No GPU work was run
while authoring this card. When a GPU window opens, execute legs 5-7 verbatim: zero
further design work is required.

Canonical program refs:
- /home/brad/.hermes/skills/gfx906-w4a4-program/references/ruquant-h2h-2026-11-05.md
- /home/brad/.hermes/skills/gfx906-w4a4-program/references/foreign-fork-h2h.md
- /home/brad/workspace/ruquant-bench/RESULTS-h2h.md (mi50 copy: /root/ruquant-bench/RUN-PLAN.md)

## 0. Why this redo exists

Owner run (2026-11-05 p2 harness, cell rq-spec): W4A4 + DFlash2 draft 7, TP2 =
742.8 pp4096 / 49.5 tg256, harness-reported acceptance 0.230 / 1.46 tokens-per-step.
ruquant published (RESULTS.md, box 1, TP2): 89.5 t/s, acceptance 4.27 tokens/step,
47.7 ms/step. The brief called the run "unverified; setup problem likeliest" and
listed causes (a) runtime mismatch, (b) drafter not pinned, (c) unconverted drafter,
(d) missing sampling flags. Audit verdicts in section 2: (c) is REFUTED, (b) and (d)
are CONFIRMED contributing factors, (a) is RESOLVED-no-issue, and a fifth cause (e),
a harness acceptance-math artifact, explains the 0.230/1.46 figure outright.

## 1. Asset audit (all verified on mi50, 2026-11-10)

### 1.1 Models (exists + path + size + sha256)

| asset | path | bytes | sha256 |
|---|---|---|---|
| rq W4A4 target (OST+GPTQ) | /root/bench-quants/ruquant/Qwen3.8-27B-W4A4-OST-GPTQ-wg64-rb4.gguf | 17190149440 | b1015b212068a2838f23a68985065352e337d9e4e0f40abab3832e610a6a5ef8 |
| rq DFlash2 drafter, CONVERTED (W4A4H) | /root/bench-quants/ruquant/Qwen3.8-27B-DFlash2-W4A4H-drafter-rb4.gguf | 2654431456 | b174a526c134a5b2a68b13a239bae961e6c14b81b5ebf43670c86ed90adc6ad2 |
| stock Q4_0 target (unsloth) | /root/bench-quants/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf | 16348767968 | 5eb1de39e9b42aa7f99f1a4fe149d1952cc8103c698cf380b15a43d47e72b7c7 |
| stock DFlash2 drafter Q8_0 (z-lab/incoai original) | /mnt/old/home/brad/models/Qwen3.8-27B-DFlash2-Q8_0.gguf | 2056414816 | c18e800daedc59ca68fd13b6a856d795746af6d399a9279ac6a277d1d422f87e |
| DFlash2 drafter Q4_K_M (unconverted, NOT used) | /root/bench-quants/dflash/Qwen3.8-27B-DFlash2-Q4_K_M.gguf | 1143006560 | ebd3a84f4fd056af9870401ef3b7ae8cd78020ffb78393fb0c3acd2273725d12 |

Both ruquant GGUFs match /root/bench-quants/ruquant/SHA256SUMS exactly (published
hashes). The converted drafter carries rotation metadata; the unconverted ones do
not. Witness (GGUF header KV dump, CPU, reads header only):

```
$ python3 <header-kv-dump> Qwen3.8-27B-DFlash2-W4A4H-drafter-rb4.gguf
  ruquant.drafter = fc inputs x R1 (OSTQuant); token_embd, output unrotated from ...
  general.quantized_by = ruquant
  general.organization = incoai ; general.source.url = https://huggingface.co/incoai/Qwen3.8-27B-DFlash2
$ ... Qwen3.8-27B-W4A4-OST-GPTQ-wg64-rb4.gguf
  general.quantized_by = ruquant
  ruquant.w4a4 = R1 = dense from models/ost-s44/r1.f32, OSTQuant smoothing; online H (4 block(s) per row) ...
$ ... Qwen3.8-27B-DFlash2-Q4_K_M.gguf   (unconverted: NO ruquant.* keys)
  general.name = Qwen Draft ; (no ruquant.* keys)
```

The header-KV dumper is inline in section 4 (W2).

/mnt/old/home/brad/models/rq-probe/ also holds conv-rtn.gguf (16516541600 B, RTN-only
convert, NOT the OST+GPTQ file) and moe-w4a4-test.gguf (20836243232 B). Neither is
used by this redo.

### 1.2 Engines

| engine | binary | tree | state |
|---|---|---|---|
| ruquant patched llama.cpp | /root/ruquant-src/build/bin/llama-server, llama-bench | ruquant-src @ 7a08316 ("ruquant: W4A4 ... TP and DFlash2 speed work") | BUILT, present |
| ruquant repo (Rust CLI + rig scripts) | /root/ruquant-repo @ 4959bd7 | rig/final_mi50.sh, rig/acc_bench.py, rig/acc_prompts.json, README.md, RESULTS.md, REPORT-2026-11-02.md, gfx906/ | present |
| our fork (INT4-noquant worktree) | /root/int4-noquant-wt/build/bin/llama-server | branch INT4-noquant @ fb18241213 | **BUILD ABSENT - GAP G1, see section 9** |
| our fork reference build (CV-port @ 7a18553) | /root/cvport-new-src/build/bin/llama-server, llama-bench | cvport-new-src | BUILT (h2h engine; NOT the redo arm) |

Fork dflash support verified in-tree: common/speculative.cpp:78 maps "draft-dflash";
src/models/dflash.cpp present; -devd flag present at common/arg.cpp:4271.

### 1.3 Harness + scripts (all present)

- /root/p2-prefill/bench.py (canonical client: temp 0, seed 42, ignore_eos,
  cache_prompt false, per-prompt warmup x2, pp4096 x1 + tg256 x3, suite medians)
- /root/p2-prefill/probe_pp.py (pp4096 warmup probe, usage: probe_pp.py <port> [n])
- /root/mtp-natural/prompts.json (8-prompt suite, 158238 B)
- /root/ruquant-bench/{start-rq.sh,stop-rq.sh,start-fork.sh,cell_p2.sh}
- /root/ruquant-repo/rig/{final_mi50.sh,acc_bench.py,acc_prompts.json}
- /root/gpu.lock (flock file, present)
- gates: rocm-smi + flock present; 32 cores; /opt/rocm -> /opt/rocm-7.2.3

### 1.4 Their exact recipe (extracted from docs/source, not guesses)

Server (RESULTS.md:126-138 "Server, TP4 with speculative decoding", TP2 = two GPUs
visible, drop nothing else; and /root/ruquant-bench/start-rq.sh which is our
validated TP2 driver of it):

```
llama-server -m <W4A4> -md <W4A4H-drafter> \
  -ngl 99 -ngld <default auto> -fa 1 -lm dio -c 8192 -np 1 \
  -sm tensor -devd ROCm0 \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  -bs --samplers temperature
```

Env (RESULTS.md:120-122 + start-rq.sh):
```
HSA_FORCE_FINE_GRAIN_PCIE=1   GGML_META_TOKEN_GRAPH=0   GGML_META_SPIN=2000000
GGML_CUDA_DISABLE_GRAPHS=1    LD_LIBRARY_PATH=<TheRock libs first>:/opt/rocm-7.2.3/lib
```

Flag semantics (their docs):
- `-devd ROCm0` pins the drafter to one GPU. Replicating (mx default, no -devd) is
  19% slower for their converted drafter; splitting is 24% slower (RESULTS.md:134).
  -devd default "follows --device" (ruquant-src common/arg.cpp:4271-4273).
- `-bs --samplers temperature` = greedy with backend sampling; under -sm tensor
  their argmax fast path does the greedy pick on CPU (RESULTS.md:138).
- draft 7 is their best for the W4A4 arm (RESULTS.md:136).
- Their acceptance metric (rig/acc_bench.py): accepted length = completion tokens /
  verify steps, steps = draft_n / n_max, read from server /completion timings plus
  "acc per pos = (...)" lines emitted at -lv 4.
- Their stock baseline line (RESULTS.md:171-173, verbatim):
  `NMAX=4 SCORE=0 DEVD= rig/final_mi50.sh Qwen3.8-27B-Q4_0.gguf Qwen3.8-27B-DFlash2-Q8_0.gguf`
  i.e. draft 4, drafter REPLICATED ("its drafter can't be pinned under TP" - the
  z-lab drafter borrows the target lm_head which -sm tensor splits).
- llama-bench cells in final_mi50.sh: `-p 512,8 -n 128 -r 3` (pp8 = the 8-token
  verify forward, printed as ms = 1000*8/avg_ts), `-p 2048 -n 0 -b 2048 -ub 2048`.

### 1.5 Published targets (RESULTS.md, box 1 = 4x MI50 32GB, PCIe 3.0, TP2 column)

| cell | TP2 t/s | acceptance | ms/step |
|---|---|---|---|
| W4A4 + DFlash2 d7 | 89.5 | 4.27 tok/step | 47.7 |
| Q4_0 + DFlash2 d4 (stock baseline) | 60.5 | 3.68 tok/step | 60.8 |
| 8-token verify forward | W4A4 35.1 ms / Q4_0 62.8 ms | | |
| pp2048 | W4A4 833.1 / Q4_0 529.0 | | |
| tg128 plain | W4A4 41.6 / Q4_0 39.7 | | |

NOTE: the combined brief cites "62.6 t/s two cards" for the stock baseline; that
number does NOT appear anywhere in their repo (grep 62.6 over all .md = no hits).
The published TP2 stock baseline is 60.5 t/s (RESULTS.md table). Use 60.5.

## 2. Root-cause findings on the 49.5 t/s run (evidence, file:line)

(a) RUNTIME MISMATCH - RESOLVED, not an issue. TheRock 7.15.0a20260702 is installed
    (/root/therock-venv/lib/python3.12/site-packages/rocm_sdk_*-7.15.0a20260702.dist-info)
    and provides the same sonames the binary needs (libamdhip64.so.7, librccl.so.1,
    librocblas.so.5, libhipblaslt.so.1). Bare `ldd` resolves to /opt/rocm-7.2.3, but
    both launchers (start-rq.sh:19, cell_p2.sh) prepend TheRock to LD_LIBRARY_PATH,
    so runtime resolves TheRock first. Witness W4 proves it per-leg.

(b) DRAFTER NOT PINNED - CONFIRMED contributing factor. The 49.5 run was launched by
    /root/ruquant-bench/cell_p2.sh, whose rq spec args are exactly
    `(-md $DRAFT --spec-type draft-dflash --spec-draft-n-max 7)` - NO -devd. Default
    follows --device => drafter replicated on both GPUs => documented 19% penalty
    (RESULTS.md:134). The redo pins -devd ROCm0.

(c) UNCONVERTED DRAFTER - REFUTED. /root/ruquant-bench/p2logs/rq-spec.server.log:24:
    `loading draft model '/root/bench-quants/ruquant/Qwen3.8-27B-DFlash2-W4A4H-drafter-rb4.gguf'`
    - the CONVERTED W4A4H drafter, sha256 matching their published SHA256SUMS, with
    `ruquant.drafter` rotation metadata present. The `unknown type w4a4h` warning at
    rq-spec.server.log:25 is cosmetic: it comes from the file-type GUESS path
    (ruquant-src/src/llama-model-loader.cpp:786, default arm of the ftype table),
    not from tensor loading. Do not treat it as a failure witness.

(d) MISSING SAMPLING FLAGS - CONFIRMED contributing factor. cell_p2.sh server line
    has no `-bs --samplers temperature`; their recipe (RESULTS.md:130) does. The redo
    includes them.

(e) NEW, PRIMARY for the acceptance figure: the 0.230/1.46 acceptance is a HARNESS
    ARTIFACT, not model behavior. cell_p2.sh invokes
    `bench.py 8630 rq-spec $LOG.reps.jsonl 2` - draft_n_max=2 against a draft-7
    server. bench.py computes accept_rate = draft_n_accepted/draft_n and
    mean_accept_len = 1 + da/(dn/2) (bench.py:40-44), which is wrong for DFlash2
    block drafting. The SERVER's own diagnostics from that same run show healthy
    acceptance: rq-spec.server.log `draft acceptance = 0.63609 (208 accepted / 327
    generated), mean len = 5.43` (and 0.816/6.71 on later tasks); the earlier
    start-rq.sh session log shows 0.307/3.11 with per-pos (0.744, 0.500, 0.317,
    0.220, 0.134, 0.110, 0.085). Authoritative acceptance = server log lines
    (witness W5), never the bench.py derived field.

Residual t/s gap (49.5 vs 89.5) is attributed to: drafter replication (-19%),
missing backend-sampling fast path, ctx/batch shape differences (cell_p2 used
-c 16384 -b 4096 -ub 4096 vs their -c 8192), and box differences (their box 1:
PCIe 3.0 through CPU, 12 cores, ROCm 5.7.1; ours: TheRock 7.15.0a). The redo
reproduces THEIR flags exactly so the remaining delta is box-only and reportable.

## 3. Preconditions and gates (run before EVERY leg; any failure = do not start)

```bash
# G1 RAM: MemAvailable >= 12 GB
awk '/MemAvailable/{exit !($2>=12582912)}' /proc/meminfo || { echo "GATE-RAM-FAIL $(awk '/MemAvailable/{print $2/1024}' /proc/meminfo)MB"; exit 99; }
# G2 load: 1-min loadavg < 16 (32 cores)
awk '{exit !($1<16)}' /proc/loadavg || { echo "GATE-LOAD-FAIL"; exit 97; }
# G3 GPUs: pick TWO free GPUs (VRAM used < 2 GB). NEVER touch the llama-swap fleet
#    (pgrep -af llama-swap; port 8181) - its GPUs show >2GB used and are excluded.
DEVS=$(python3 -c '
import subprocess,re
o=subprocess.run(["rocm-smi","--showmeminfo","vram"],capture_output=True,text=True).stdout
free=[m.group(1) for l in o.splitlines() for m in [re.match(r"GPU\[(\d+)\].*Used Memory \(B\):\s*(\d+)",l)] if int(m.group(2))<2*1024**3]
print(",".join(free[:2]) if len(free)>=2 else "")')
[ -n "$DEVS" ] || { echo "GATE-GPU-FAIL free=$DEVS"; exit 96; }
# G4 ports free (rq=8612, fork=8197; fleet uses 8181)
ss -ltn | grep -qE ":8612 " && { echo "GATE-PORT-FAIL"; exit 95; }
# G5 box-wide single model load: flock held for the WHOLE leg (server + client + teardown)
exec 9>/root/gpu.lock; flock -w 1800 9 || { echo "LOCKFAIL"; exit 1; }
```

State at authoring time (2026-11-10 12:38 local): MemAvailable ~40 GB (parent's
earlier 37/40-used reading was stale at this hour), loadavg 0.28, llama-swap fleet
UP (PID 316605, port 8181 LISTEN) => GPU1/GPU2 busy => legs wait on G3 until two
GPUs free. The fleet is OFF LIMITS: do not kill/restart/reconfigure it or /llama-swap/.

## 4. Witness protocol (record all into the leg log)

W1 pre-launch hash echo (must equal section 1.1 table):
```
sha256sum <model> <drafter> | tee -a $LOG.witness
```
W2 rotation-metadata echo (proves converted target/drafter; header-only read):
```
python3 /root/int4-noquant-wt/scripts/rq_gguf_kv.py <model> ruquant general.quantized_by
python3 /root/int4-noquant-wt/scripts/rq_gguf_kv.py <drafter> ruquant general.organization
```
(rq_gguf_kv.py = the header KV dumper used in this audit; committed alongside this
card under scripts/. Expect `ruquant.w4a4` on the target, `ruquant.drafter` on the
W4A4H drafter, and NO ruquant.* keys on unconverted files.)
W3 engine/knob engagement (server log greps, exact strings):
```
loading model '<path>'                      # engine + target
loading draft model '<path>'                # drafter file actually engaged
common_speculative_impl_draft_dflash: adding speculative implementation 'draft-dflash'
common_speculative_impl_draft_dflash: - n_max=<7|4>, n_min=0, p_min=0.00
common_speculative_impl_draft_dflash: - block_size=8, mask_token_id=248070, n_extract=5, sample_from_anchor=true
```
The n_max line is the draft-length witness; block_size line proves DFlash2 engaged.
W4 runtime witness (which ROCm the PID actually maps):
```
tr '\0' '\n' < /proc/$SRV/environ | grep -E '^(LD_LIBRARY_PATH|GGML_|HSA_)' > $LOG.env-witness
grep -oE '/[^ ]*lib(amdhip64|rccl|rocblas)[^ ]*' /proc/$SRV/maps | sort -u | tee -a $LOG.witness
```
Expect TheRock paths (rocm_sdk 7.15.0a20260702). /opt/rocm-7.2.3 here = runtime bug, stop the leg.
W5 acceptance witness (AUTHORITATIVE, server-side; bench.py derived fields are
diagnostic only - see section 2e):
```
grep -E 'draft acceptance = ' $LOG.server.log | tail            # per-task accepted/generated + mean len
grep -E 'statistics +draft-dflash' $LOG.server.log | tail -1    # #mean acc len, #acc rate/pos, dur ms
```
Record tokens-accepted/step = mean acc len from the statistics line; ms/step from
timings (predicted_ms/predicted steps) or the dur fields.
W6 per-GPU VRAM placement (drafter pin witness): after load, `rocm-smi
--showmeminfo vram`; with -devd ROCm0 the drafter (~2.6 GB W4A4H / ~2.0 GB Q8_0)
lands on the FIRST visible GPU only (used_0 - used_1 ~ drafter size). Replicated =
both equal.
W7 command-line echo: `tr '\0' ' ' < /proc/$SRV/cmdline >> $LOG.witness` before any
client traffic.

## 5. Leg R - ruquant arm, THEIR RECIPE UNCHANGED (TP2, draft 7)

Server (identical to /root/ruquant-bench/start-rq.sh tp2 graphs-off spec, plus
--no-webui -lv 4 taken from their own rig/acc_bench.py to enable the per-pos acc
trace; port 8612 as in start-rq.sh):

```bash
BIN=/root/ruquant-src/build/bin
M=/root/bench-quants/ruquant/Qwen3.8-27B-W4A4-OST-GPTQ-wg64-rb4.gguf
D=/root/bench-quants/ruquant/Qwen3.8-27B-DFlash2-W4A4H-drafter-rb4.gguf
TR=/root/therock-venv/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$TR/_rocm_sdk_libraries/lib:$TR/_rocm_sdk_core/lib:$TR/_rocm_sdk_devel/lib:/opt/rocm-7.2.3/lib
export HSA_FORCE_FINE_GRAIN_PCIE=1 GGML_META_TOKEN_GRAPH=0 GGML_META_SPIN=2000000 GGML_CUDA_DISABLE_GRAPHS=1
setsid nohup env HIP_VISIBLE_DEVICES=$DEVS $BIN/llama-server \
  -m "$M" -md "$D" -ngl 99 -fa 1 -lm dio -c 8192 -np 1 -sm tensor -devd ROCm0 \
  --spec-type draft-dflash --spec-draft-n-max 7 -bs --samplers temperature \
  --host 0.0.0.0 --port 8612 --no-webui -lv 4 > $LOG.server.log 2>&1 < /dev/null &
SRV=$!
```

Health: poll `curl -s http://127.0.0.1:8612/health` up to 90x10s; if $SRV dies
(`kill -0`), stop, keep log tail, do not retry blindly. Then W1-W4, W6.

Warmups (doctrine: >=3 discarded same-shape warmups before any measured rep):
```bash
python3 /root/p2-prefill/probe_pp.py 8612 6 >/dev/null 2>&1          # 6x pp4096
for i in 1 2 3; do curl -s -m 600 http://127.0.0.1:8612/completion -d \
 '{"prompt":"Warmup pass for the decode shape.","n_predict":256,"temperature":0,"cache_prompt":false,"ignore_eos":true}' >/dev/null; done
```

Measured client (canonical p2 harness; NOTE the trailing 7 = correct draft_n_max,
the 2 that produced the artifact in cell_p2.sh must NOT be repeated):
```bash
python3 /root/p2-prefill/bench.py 8612 rq-spec-redo-r$REP $LOG.reps.jsonl 7
```

Cross-check with THEIR acceptance harness (their metric, ~4.27 expected):
```bash
BIN=$BIN GPUS=$DEVS ROCM_LIB=$TR/_rocm_sdk_libraries/lib:$TR/_rocm_sdk_core/lib \
  EXTRA="-sm tensor -devd ROCm0" \
  python3 /root/ruquant-repo/rig/acc_bench.py "$M" "$D" 7
```
(ROCM_LIB override is required: acc_bench.py defaults to /opt/rocm/lib = apt 7.2.3.)

Then W5 + teardown (section 8). Exit criteria for leg R: acceptance >= 3.5 tok/step
(acc_bench) and tg256 suite median >= 75 t/s. If acceptance is healthy but t/s < 75,
the residual is box-level (PCIe/topology) - report, do not chase. If acceptance < 3.0,
STOP: re-check W3 n_max line and W6 pin witness before any rerun.

## 6. Leg F - our fork arm (INT4-noquant, stock Q4_0 + standard DFlash2, draft 4)

Fills the missing cell against their published stock baseline (60.5 t/s TP2).

F0 BUILD (queued; CPU-only but forbidden during the freeze - run in the window,
before any leg, once, with the same RAM gate):
```bash
cd /root/int4-noquant-wt
TR=/root/therock-venv/lib/python3.12/site-packages
cmake -B build -G Ninja -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGPU_TARGETS=gfx906 \
  -DCMAKE_HIP_COMPILER=$TR/_rocm_sdk_devel/bin/amdclang++ \
  -DCMAKE_PREFIX_PATH="$TR/_rocm_sdk_core;$TR/_rocm_sdk_libraries;$TR/_rocm_sdk_devel" \
  -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON \
  -DGGML_HIP_RCCL=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_OPENSSL=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```
(Flags = cvport-new-src CMakeCache values, verified: GGML_HIP=ON, HIP_GRAPHS=ON,
HIP_NO_VMM=ON, HIP_RCCL=ON, BACKEND_DL=ON, CPU_ALL_VARIANTS=ON, TheRock amdclang++,
Release. -j8 to stay under the load gate.)

Server (start-fork.sh serving config + DFlash2 drafter at draft 4, drafter
REPLICATED - no -devd - matching their baseline line `DEVD=` and their note that
the z-lab drafter borrows the target lm_head and cannot be pinned under TP):
```bash
BIN=/root/int4-noquant-wt/build/bin
M=/root/bench-quants/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf
D=/mnt/old/home/brad/models/Qwen3.8-27B-DFlash2-Q8_0.gguf
source /root/w4a4-env.sh
export GGML_CUDA_ALLREDUCE=internal GGML_CUDA_REPACK_KQUANT=0 GGML_GDN_COLTHREAD=1
export HIP_VISIBLE_DEVICES=$DEVS HSA_KERNARG_POOL_SIZE=67108864 GPU_MAX_HW_QUEUES=8
setsid nohup $BIN/llama-server -m "$M" -md "$D" \
  --host 0.0.0.0 --port 8197 -ngl 99 -fa on -sm tensor --fit off \
  -c 65536 -ub 1024 -t 16 -np 1 -lm dio \
  --spec-type draft-dflash --spec-draft-n-max 4 > $LOG.server.log 2>&1 < /dev/null &
SRV=$!
```
Notes: GGML_TP_AR_MTP_SAFE=1 (set by start-fork.sh for MTP) is deliberately NOT set
for the dflash arm; if the leg aborts at first decode, set it and rerun once,
recording the change. Warmups as in section 5 (port 8197). Measured client:
```bash
python3 /root/p2-prefill/bench.py 8197 fork-q40-dflash-redo-r$REP $LOG.reps.jsonl 4
```
Witnesses W1-W5 apply (W2: expect NO ruquant.* keys on both files - this arm is
stock Q4_0 + unrotated drafter; the witness proves exactly that). No exit-criterion
gate on this arm: it fills the missing cell; report whatever it measures.

## 7. Leg T - 8-token forward timing (BOTH engines; DIAGNOSTIC-ONLY)

Program doctrine: llama-bench is NEVER a reported number; authoritative spec-decode
numbers go through llama-server + the /completion client above. These cells exist
only to compare verify-step forward cost across engines at widths 1/6/8 (1 = plain
decode, 6 = draft5+1, 8 = draft7+1). ms per forward = 1000 * width / avg_ts.

```bash
# rq engine (same env as leg R, same $DEVS)
HIP_VISIBLE_DEVICES=$DEVS /root/ruquant-src/build/bin/llama-bench \
  -m /root/bench-quants/ruquant/Qwen3.8-27B-W4A4-OST-GPTQ-wg64-rb4.gguf \
  -ngl 99 -fa 1 -lm dio -sm tensor -p 1,6,8 -n 0 -r 3 -o jsonl
# fork engine (same env as leg F)
HIP_VISIBLE_DEVICES=$DEVS /root/int4-noquant-wt/build/bin/llama-bench \
  -m /root/bench-quants/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf \
  -ngl 99 -fa on -lm dio -sm tensor --fit off -p 1,6,8 -n 0 -r 3 -o jsonl
```
Published TP2 reference (8-token forward): W4A4 35.1 ms, Q4_0 62.8 ms.
Optional context cells (their final_mi50.sh shapes): `-p 512,8 -n 128 -r 3` and
`-p 2048 -n 0 -b 2048 -ub 2048 -r 3`.

## 8. Teardown after EVERY leg (TERM-only, PID-verified)

The wrapper exiting is NOT evidence the server died. The launcher's flock fd is
inherited by the server, so gpu.lock stays held while the server lives.

```bash
kill -TERM $SRV 2>/dev/null
for i in $(seq 1 24); do kill -0 $SRV 2>/dev/null || break; sleep 5; done
kill -0 $SRV 2>/dev/null && { echo "TEARDOWN-STUCK pid=$SRV - STOP, human review, do NOT kill -9"; exit 90; }
pgrep -f "port 8612|port 8197" | grep -v $$ && echo "TEARDOWN-ORPHAN - investigate"
# VRAM release witness:
sleep 5; rocm-smi --showmeminfo vram | grep -E "Used"
```
NEVER kill -9 a loading server (program rule; stop-rq.sh's KILL fallback is NOT to
be used in this redo). A leg that cannot tear down cleanly ends the session: record,
stop, report.

## 9. Session schedule (same-session interleaved) + recording rules

One session, one GPU window, two free GPUs ($DEVS from G3), flock per leg:
  1. F0 build (if not already done)
  2. T-rq, T-fork            (llama-bench diagnostics, no server)
  3. R rep1  ->  F rep1      (server legs alternate arms to cancel session drift)
  4. R rep2  ->  F rep2
Report per cell: median of rep1/rep2; acceptance from W5 (server lines); ms/step
from timings. Quote numbers only from the $LOG.reps.jsonl + agg files, never from
chat (foreign-fork-h2h rule). Keep every $LOG.* under /root/ruquant-bench/redo/.
Cross-engine parity rule: each engine first at its OWN documented best (rq: their
flags above; fork: start-fork.sh config above); harmonize only after both reproduce.

## 10. GAPS / what Brad must supply

- G1 (BLOCKER for leg F): /root/int4-noquant-wt has NO build. F0 cmake+ninja step
  required first (~25-45 min CPU at -j8, RAM gate applies). Everything else for leg
  F is ready (model + drafter present and hashed).
- G2: brief's "62.6 t/s" stock baseline does not exist in their docs; published
  TP2 figure is 60.5 t/s (RESULTS.md). Card uses 60.5. If 62.6 came from another
  source, supply it.
- G3: ost-s44 OST checkpoint absent on box (only needed to REGENERATE the W4A4
  target; not needed for this redo - the converted GGUF is present and hash-matched).
- G4: converted drafter: PRESENT (no gap) - hypothesis (c) refuted; no converter run
  needed. (Their converter is `ruquant drafter` per README.md:103-104 if ever needed.)
- G5: GPU window: fleet on GPU1/GPU2 (llama-swap PID 316605, port 8181) is off
  limits; legs wait on gate G3 for two GPUs with <2GB VRAM used.

## 11. Risks

- Co-residency with the llama-swap fleet: G3 excludes busy GPUs; a fleet load
  starting mid-leg slows that leg (record rocm-smi snapshots in the log; do not
  touch the fleet).
- `unknown type w4a4h` warning at drafter load is COSMETIC (ftype-guess path,
  ruquant-src/src/llama-model-loader.cpp:786) - not a failure witness.
- bench.py derived acceptance fields are wrong for DFlash2 (section 2e); W5 server
  lines are the only acceptance numbers to report.
- -ngld default is "auto" (ruquant-src common/arg.cpp:4278-4282); start-rq.sh omits
  -ngld and their acc_bench passes -ngld 99. Arm kept unchanged (auto); W6 pin
  witness catches a draft-off-GPU surprise.
- Box drift: their box drifts +/-3% between sessions (RESULTS.md); ours unmeasured -
  hence interleaved reps + medians.
- Ports: 8612 (rq), 8197 (fork) chosen clear of fleet 8181; G4 gate checks 8612,
  repeat for 8197 on the fork leg.

## Appendix A. Live verification outputs (pasted, mi50, 2026-11-10)

```
$ git -C /root/int4-noquant-wt rev-parse --abbrev-ref HEAD
INT4-noquant
$ free -m | head -2
Mem: 40960 total, 737 used, 40222 available
$ uptime
load average: 0.28, 0.39, 0.51
$ sha256sum (section 1.1 table) - both ruquant files MATCH /root/bench-quants/ruquant/SHA256SUMS:
b1015b212068a2838f23a68985065352e337d9e4e0f40abab3832e610a6a5ef8  Qwen3.8-27B-W4A4-OST-GPTQ-wg64-rb4.gguf
b174a526c134a5b2a68b13a239bae961e6c14b81b5ebf43670c86ed90adc6ad2  Qwen3.8-27B-DFlash2-W4A4H-drafter-rb4.gguf
$ git -C /root/ruquant-src log --oneline -1
7a08316 ruquant: W4A4 (int4 x int4) Qwen3.8-27B on gfx906, TP and DFlash2 speed work
$ ls /root/ruquant-src/build/bin/llama-server /root/ruquant-src/build/bin/llama-bench
both present
$ ls /root/int4-noquant-wt/build/bin
No such file or directory   (GAP G1)
$ llama-server --help (rq build), flag spellings verified:
  --spec-draft-device, -devd, --device-draft <dev1,dev2,..>
  --spec-draft-ngl, -ngld, --gpu-layers-draft, --n-gpu-layers-draft N   (default: auto)
  --spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,...
  --spec-draft-n-max N   (default: 3)
  -bs, --backend-sampling
  --samplers SAMPLERS
$ ss -ltn | grep 8181
LISTEN 0 4096 *:8181 (llama-swap fleet, PID 316605 - OFF LIMITS)
$ rq-spec.server.log (the 49.5 run):
  line 24: loading draft model '/root/bench-quants/ruquant/Qwen3.8-27B-DFlash2-W4A4H-drafter-rb4.gguf'
  line 28: common_speculative_impl_draft_dflash: - n_max=7, n_min=0, p_min=0.00
  line 29: - block_size=8, mask_token_id=248070, n_extract=5, sample_from_anchor=true
  slot print_timing: draft acceptance = 0.63609 (208 accepted / 327 generated), mean len = 5.43
$ cell_p2.sh (the flawed launcher): EXTRA=(-md "$DRAFT" --spec-type draft-dflash --spec-draft-n-max 7)
  [no -devd, no -bs --samplers temperature]; client call: bench.py 8630 "$CELL" "$LOG.reps.jsonl 2"
```
