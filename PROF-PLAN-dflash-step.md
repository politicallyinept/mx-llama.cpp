# PROF-PLAN-dflash-step

GPU leg plan for the DFlash2 spec-decode step-cost question: where does the
2.6x gap vs MTP-on (55 vs 144 t/s at 6.6-token acceptance) actually go -
drafter forward, target verify forward, state snapshots/copies, or graph
rebuild churn? Consumes the GGML_DFLASH_PROF instrumentation patch.

Line cites below are at fb18241213 (pre-instrumentation); the patch only
inserts lines, so functions keep their names as anchors.

## 0. Path map (file:line at fb18241213)

### Drafter (DFlash2 GGUF, separate ctx_dft)
- Model/graph: src/models/dflash.cpp (arch LLM_ARCH_DFLASH; selector tensors
  157-159, conv tensors 260-263). TP tensor-split path: src/llama.cpp:399.
- Impl registry: common/speculative.cpp:78 ("draft-dflash"), 3117-3130
  (arch detect; Markov head distinguishes dspark from dflash),
  1013-1064 (impl struct), is_dflash2 = selector_top_k > 0 at 1100-1101.
- Drafter forward per step: common/speculative.cpp:1583 draft(); noise block
  built at 1616-1625 (id_last + n_max mask tokens => block width n_max+1);
  the forward is llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch)
  at ~1633; DFlash2 reads the selector lattice via
  llama_get_embeddings_nextn at ~1654 (no logits).
- Target-feature path per step: process() ~1440-1580: llama_synchronize
  (tgtsync) ~1520-1530, hidden-state gather memcpys ~1537-1550, inject
  batch add_embd ~1560-1568, fused encode+inject
  llama_process(ctx_dft, ..., batch_inject) ~1569-1572.
- Pre-existing cumulative timers: common/speculative.cpp:209-224
  (t_tgtsync/t_gather/t_encode/t_readnx/t_inject + pp variants), printed in
  the spec perf line at 3808-3817. Always measured (gen_perf=true, 196).

### Verify (target forward with draft tokens, width up to 8)
- Verify batch build: tools/server/server-context.cpp:604-640
  handle_last_sampled_token(): sampled token + spec_draft appended
  (width = n_draft+1); spec_i_batch rows at 627-631.
- Target forward: server-context.cpp:4308 decode() ->
  llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view) at 4348.
  Pre-existing per-width cumulative timers t_dec_k[1..8] at 4357-4359,
  printed by LLAMA_SERVER_TIMING=1 (server_loop_timing, 47-75).
- Draft generation call site: server-context.cpp:3521-3528 (t_draft bucket).

### State snapshot / rollback
- Accept + rollback: server-context.cpp:4580-4700:
  sample_and_accept_n 4614-4621; n_rollback 4627; checkpoint restore path
  4636-4656 (load_tgt/load_dft + mem.seq_rm); common_speculative_accept 4665.
- Checkpoint buffers: common/common.h:1278 (common_prompt_checkpoint:
  data_tgt/data_dft host vectors); common/common.cpp:2424-2492
  update_tgt/update_dft/load_tgt/load_dft wrap
  llama_state_seq_get_data_ext/set_data_ext (src/llama-context.cpp:5439,
  5446 -> KV or recurrent memory copy).
- Server call sites: update_pos/update_dft 3487-3496 (new draft cycle),
  update_dft 3574, load_tgt/load_dft 4650-4653, prompt_save 369-389.

### HIP graph capture / rebuild
- Per-subgraph caches: ggml/src/ggml-cuda/ggml-cuda.cu:6884
  ggml_backend_cuda_graph_compute(); graph key mixes nodes[0], n_nodes,
  nodes[0]->ne[1] and the batch-type hint at 3911-3925 (draft/verify shapes
  deliberately separated); warmup/reset/unstable-disable 6915-6960; capture
  begin 6977-6984; evaluate_and_capture 6556; instantiate 3990; counters
  via GGML_CUDA_GRAPH_STATS=1 at 6862-6882 (periodic graph_stats lines).
- Whole-token graph: ggml/src/ggml-backend-meta.cpp:2743-2792 (state,
  tg_warm_needed=2 => ~2 warm + capture, plus the per-shape permanent
  disable on failure), 3268 (GGML_META_TOKEN_GRAPH), state machine
  5410-5760: inner_graphs_reenable 5427-5440, eligibility 5600-5617,
  capture 5620-5735, replay 5490-5537, scratch-gen invalidation 5475-5487.
  Capture failures log only at debug level unless GGML_META_TG_DEBUG=1.
- nc (narrow) kernel family implicated at verify widths 6-8:
  ggml/src/ggml-cuda/norm.cu:444-533 (rms_norm_narrow + trace).

### Draft-side offload through llama-server
- common/arg.cpp: --spec-draft-model/-md 4299, --spec-type 4307,
  --spec-draft-n-max 4183, --spec-draft-device/-devd 4270,
  --spec-draft-ngl 4278, --spec-draft-tensor-parallel-size 2855,
  --cpu-moe-draft 4165, --spec-draft-type-k/v 4133/4146.

### What the new patch adds (banner [DFLASH-PROF], env GGML_DFLASH_PROF=1)
- draft-fwd: per-call drafter forward us + n_tokens (speculative.cpp draft()).
- draft-inject: per-call fused encode+inject us + n_tokens (process()).
- draft-gather: per-call gather us + bytes.
- tgt-decode: per-call target forward width (= ne[1] of the verify batch)
  + us + inner_us (server-context.cpp decode()).
- cuda-graph: one line per graph_compute per device: call#, ne11 (first
  mul_mat src1 ne[1] = verify width), mode replay|capture|outer|owner-off|
  direct, nodes, us. cuda-graph-unstable on per-shape disable.
- tg-capture (ok=1/0 + us), tg-replay (throttled), tg-not-eligible,
  tg-invalidate: token-graph events per uid with timing.
- ckpt-update-tgt/dft, ckpt-load-tgt/dft: snapshot us + size + ptr +
  distinct-pointer counter (pointer churn).

## 1. Preconditions (all must hold before the GPU leg)
- RAM gate: `free -m | awk '/MemAvailable/{print $2}'` >= 12288.
- GPU lock: run the whole leg under `flock /root/gpu.lock -c '...'`.
- One model load in flight: `pgrep -f 'llama[-]server'` shows only your PID;
  do not touch the llama-swap fleet on 8181; pin your run to the free GPU
  with ROCR_VISIBLE_DEVICES (GPU0 if free; verify with
  `rocm-smi --showuse` that the chosen device is idle before load).
- No other build in flight (`pgrep -f '[c]make|[m]ake'` empty).

## 2. Build (GPU-QUEUED item 1; NOT run: RAM gate + explicit no-build pause)
```
cmake -S /root/int4-noquant-wt -B /root/int4-noquant-wt/build-dprof \
  -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DLLAMA_CURL=OFF
cmake --build /root/int4-noquant-wt/build-dprof --target llama-server -j4
```
Align the configure with the fleet's standard gfx906 HIP flags before
running (this worktree has no build dir; flags above are the llama.cpp HIP
defaults). Exit criteria: llama-server links; `git diff fb18241213 --stat`
shows only the 5 instrumented files.

## 3. Server cells
Prompt classes (>=3 requests each, same op traced across requests):
- structured: code/JSON prompts (the 6.6-acceptance class)
- natural prose (acceptance ~0.52-0.6)

Cells (same target Q4_0 + dflash2 drafter):
- A: -np 1 --spec-draft-n-max 7   (verify width 8 = the nc-kernel edge)
- B: -np 1 --spec-draft-n-max 3   (width 4; separates width cost from step cost)
- C: -np 4 --spec-draft-n-max 7   (wide batch; expect nc-kernel divergence)
- D: MTP-on baseline, same prompts (reference for the 144 t/s number)

```
GGML_DFLASH_PROF=1 GGML_META_PROFILE=1 GGML_META_TG_DEBUG=1 \
GGML_META_LANE_PROF=1 GGML_CUDA_GRAPH_STATS=1 LLAMA_SERVER_TIMING=1 \
ROCR_VISIBLE_DEVICES=<free> flock /root/gpu.lock -c \
  'build-dprof/bin/llama-server -m <target-Q4_0.gguf> -ngl 99 \
   --spec-draft-model <dflash2-drafter.gguf> --spec-type draft-dflash \
   --spec-draft-n-max 7 -np 1 -c 4096 --port <free-port> \
   > dprof-A.log 2>&1 & echo $! > dprof-A.pid'
```
Teardown: `kill -TERM $(cat dprof-A.pid)`, poll `kill -0` up to 60 s,
verify exit; never kill -9.

## 4. What to grep per phase
| phase | grep |
|---|---|
| drafter forward | `[DFLASH-PROF] draft-fwd` |
| fused encode+inject | `[DFLASH-PROF] draft-inject` |
| hidden-state gather | `[DFLASH-PROF] draft-gather` |
| target verify forward | `[DFLASH-PROF] tgt-decode` (width col = ne11 witness) |
| graph mode + width per device | `[DFLASH-PROF] cuda-graph` (mode=capture lines are the rebuild events; us on those lines is capture cost) |
| per-shape graph disable | `[DFLASH-PROF] cuda-graph-unstable` |
| token-graph events | `[DFLASH-PROF] tg-capture|tg-replay|tg-not-eligible|tg-invalidate` |
| snapshot/restore + churn | `[DFLASH-PROF] ckpt-update-tgt|ckpt-update-dft|ckpt-load-tgt|ckpt-load-dft` |

Reading rules:
- Conclusions need >=3 requests tracing the SAME op; compare ptr= values of
  the same op across requests (distinct= counter is the running tally).
- cuda-graph call numbers are PER DEVICE (TP lanes interleave, offsetting by
  one call per device); group per dev= before comparing.
- Per-step cost = tgt-decode(us at width w) + draft-fwd + draft-inject +
  draft-gather + ckpt-* us, summed per decode iteration; cross-check the
  sum against the LLAMA_SERVER_TIMING per-iteration line.

## 5. Cross-checks with existing instruments
- tgt-decode inner_us vs `[meta-prof]` per-call compute/sync
  (GGML_META_PROFILE): host-bound vs GPU-bound split of the verify step.
- cuda-graph mode counts vs periodic `graph_stats dev` lines
  (GGML_CUDA_GRAPH_STATS): replay fraction must match.
- tg-capture/tg-not-eligible vs `[tg-debug] uid` lines
  (GGML_META_TG_DEBUG): every permanent disable must have a reason line.
- tgt-decode width histogram vs `server loop timing: target decode+sync of
  N tokens` (LLAMA_SERVER_TIMING=1): same widths, same means.
- draft-fwd + draft-inject sums vs the `draft` bucket in the per-iteration
  line; residual = spec bookkeeping + sampler.
- Expected first finding to confirm/refute: on a NEW token count the token
  graph burns ~5 requests per shape with zero graph acceleration (warm +
  capture), and per-shape churn between draft shapes (width n_max+1) and
  verify shapes (width 1..n_max+1) keeps re-triggering it; the witness is
  tg-capture/tg-invalidate density + cuda-graph mode=capture density in the
  first N requests of each cell vs steady state.

## 6. Exit criteria
A per-step table per cell: drafter fwd us, inject us, gather us, verify fwd
us by width, ckpt us, graph capture count + cost, replay fraction; plus a
verdict on which phase explains the 55 vs 144 t/s gap.
