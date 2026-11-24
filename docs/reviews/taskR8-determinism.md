================================================================================
taskR8 LEDGER - pinned-drafter acceptance nondeterminism on MI50 W4A4 spec-decode
================================================================================
Date         : 2026-11-23 (runs 21:00-21:23 PST)
Host         : mi50  (3x AMD MI50 32GB gfx906, ROCm LXC; ROCm0/1/2 live)
Binary       : /root/build-g4p2b/bin/llama-server  (17888 B, 2026-11-17 13:16)
Target       : /root/bench-quants/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_0.gguf
Drafter      : /mnt/old/home/brad/models/DFlash2-Q4_0-ownhead-q4.gguf (STOCK, not pruned-v2)
Env          : /root/w4a4-env.sh (TheRock libs first on LD_LIBRARY_PATH)
Champion     : --spec-type draft-dflash --spec-draft-n-max 4 -ctk q4_0 -ctv q4_0
               -ngl 99 -fa on --fit off -c 8192 -ub 1024 -t 16 -np 1 -lm dio

VERDICT: DETERMINISTIC-BUT-SEED-SENSITIVE
--------------------------------------------------------------------------------
The claim is REAL but MISATTRIBUTED. Fixed-seed runs reproduce bit-exactly on
BOTH the single-card control and the cross-card arm. The nondeterminism the
prior lane saw was produced by an UNPINNED seed: llama's seed sentinel
0xFFFFFFFF ("use a random seed") draws from std::random_device once per
request. With that sentinel, EVERY run differs - on the single card too, where
no cross-device path exists. Cross-device placement is therefore exonerated as
the source.

--------------------------------------------------------------------------------
1. DIVERGENCE TABLE  (arm x run x first-divergent-step)
--------------------------------------------------------------------------------
"first-divergent-step" is the 1-based index into the per-step "accepted N/ 4"
sequence at which a run first deviates from that arm's run 0. IDENTICAL = no
deviating step for the whole sequence.

SINGLE-CARD arm  (--device ROCm2 -devd ROCm2 -sm none; drafter+target same card)
 run  seed        steps  accepted-file sha256[:16]  first-divergent-step
 r0   42          105    7ec57750c4d61bd4           IDENTICAL (reference)
 r1   42          105    7ec57750c4d61bd4           IDENTICAL
 r2   42          105    7ec57750c4d61bd4           IDENTICAL
 r3   42          105    7ec57750c4d61bd4           IDENTICAL
 r4   42          105    7ec57750c4d61bd4           IDENTICAL
 r5   42          105    7ec57750c4d61bd4           IDENTICAL
 r6   42          105    7ec57750c4d61bd4           IDENTICAL
 r7   42          105    7ec57750c4d61bd4           IDENTICAL
 r8   42          105    7ec57750c4d61bd4           IDENTICAL
 r9   42          105    7ec57750c4d61bd4           IDENTICAL
 r10  42 (extra)  105    7ec57750c4d61bd4           IDENTICAL
 r11  42 (extra)  105    7ec57750c4d61bd4           IDENTICAL
 r12  42 (extra)  105    7ec57750c4d61bd4           IDENTICAL
                                                  => 13/13 BIT-IDENTICAL
                                                     acceptance 0.35800 (150/419), mean len 2.43

CROSS-CARD arm  (--device ROCm1,ROCm2 -devd ROCm1 -sm tensor; drafter on ROCm1,
                 target row-split across ROCm1+ROCm2)
 run  seed        steps  accepted-file sha256[:16]  first-divergent-step
 r0   42          98     d38af7fa0bc26cba           IDENTICAL (reference)
 r1   42          98     d38af7fa0bc26cba           IDENTICAL
 r2   42          98     d38af7fa0bc26cba           IDENTICAL
 r3   42          98     d38af7fa0bc26cba           IDENTICAL
 r4   42          98     d38af7fa0bc26cba           IDENTICAL
 r5   42          98     d38af7fa0bc26cba           IDENTICAL
 r6   42          98     d38af7fa0bc26cba           IDENTICAL
 r7   42          98     d38af7fa0bc26cba           IDENTICAL
 r8   42          98     d38af7fa0bc26cba           IDENTICAL
 r9   42          98     d38af7fa0bc26cba           IDENTICAL
 r10  42 (extra)  98     d38af7fa0bc26cba           IDENTICAL
 r11  42 (extra)  98     d38af7fa0bc26cba           IDENTICAL
 r12  42 (extra)  98     d38af7fa0bc26cba           IDENTICAL
                                                  => 13/13 BIT-IDENTICAL
                                                     acceptance 0.40464, mean len ~2.5x

NO ARM DIVERGED. There is no first-divergent-step to report for either arm
under a fixed seed. Both arms are internally bit-identical across all 13
repetitions (10 required + 3 extra per arm), verified with `cmp` on the full
per-step accept files, not just the summary line.

PLACEMENT PROVEN (so the cross arm is genuinely cross-device):
  single r0 log: q8_repack on "device 2" only  -> single-card, -sm none
  cross  r0 log: q8_repack on "device 1" AND "device 2" -> real tensor split

SEED IS THE ONLY MOVING PART (the confound that produced the original claim):
 every one of these differs from every other, in BOTH arms:
 single s123  100 steps  66e3f961fe0da609  acceptance 0.39043
 single s777   95 steps  d419d0e519134fda  acceptance 0.42513
 single s2026  95 steps  6623e1c11dec515c  acceptance 0.42667
 cross  s123   91 steps  9ba3e97097bd8de8  acceptance 0.45429
 cross  s777   93 steps  33e9bec9d2114e5a  acceptance 0.43631
 cross  s2026  98 steps  93dac168ce847462  acceptance 0.40153
 => cross-arm vs single-arm acceptance differs (0.4046 vs 0.3580) because the
    two arms are different CONFIGS (split mode changes prefill numerics); each
    config is itself reproducible. This is not nondeterminism.

RANDOM-SEED REPRODUCTION OF THE ORIGINAL SYMPTOM (seed = 0xFFFFFFFF sentinel):
 SINGLE-CARD: r20 97 steps/28e417cda48c3694/0.40568
              r21 97 steps/10ebc11f881d20e3/0.41146
              r22 99 steps/e25fc03d027ece84/0.39594   -> 3/3 DIFFERENT
 CROSS-CARD : r20 107 steps/08080f92b64f88b2/0.34579
              r21  93 steps/ec3cfe3b9093977c/0.43631
              r22  92 steps/8607f97fdb23d494/0.44262  -> 3/3 DIFFERENT
 Every run differs in step count, per-step sequence AND acceptance rate - the
 exact fingerprint the prior lane reported. It appears on the SINGLE CARD, so it
 cannot be a multi-device race.

--------------------------------------------------------------------------------
2. PER-STEP LOGS - LOCATION
--------------------------------------------------------------------------------
/root/r8b/logs/<tag>.log   full llama-server log (contains every
                           "accepted N/ 4 draft tokens" line)
/root/r8b/logs/<tag>.acc   parsed per-step accept counts, one "N M" per step,
                           in order (N = accepted, M = drafted)
/root/r8b/logs/<tag>.resp.json  full completion response
/root/r8b/logs/<tag>.vram  per-5s VRAM samples during the run
/root/r8b/sweep.log        arm-level timeline, rc per run, drain-gate lines
tag = r8b_<arm>_s<seed>_t<temp>_r<runindex>,  arm in {single,cross}

Harness: /root/r8b_run2.sh (per-run), /root/r8b_sweep.sh (loop),
         /root/r8b_batch.sh (lock-held batch runner)
40 .acc files present. Parsed-step guard held on every run (steps>0 asserted;
zero parsed steps would have aborted the arm). Parse pattern uses the ` *` after
the slash that the server's "%2zu/%2zu" format requires.

--------------------------------------------------------------------------------
3. CODE INSPECTION - target-hidden-state write -> drafter read dependency
--------------------------------------------------------------------------------
Repo: /root/int4-noquant-wt (no commits made, tree untouched)

THE DATA PATH. The dflash drafter does not read the target's KV or logits; it
reads the target's per-layer INPUT features ("taps"), copied out to a host
buffer, then re-packed into the drafter's inject batch.

  write side (target graph)
    src/llama-graph.cpp:1553-1577
      taps selected by cparams.embeddings_layer_inp are expanded at graph end;
      if a persistent device tap exists (params.layer_inp_dev), a ggml_cpy node
      copies the tap into that persistent device tensor.

  tap buffer placement (single device, NOT per-drafter-device)
    src/llama-context.cpp:769-799   layer_inp_dev[] allocated from
      ggml_backend_dev_buffer_type(model.dev_output())   <-- ONE device
                                    (llama-context.cpp:795)
    src/llama-model.cpp:2251        pimpl->dev_output = get_layer_buft_list(n_layer_all)
      i.e. the OUTPUT-layer device. In the cross arm that is ROCm2, while the
      drafter lives on ROCm1.

  readback (async D2H, then host memcpy)
    src/llama-context.cpp:2937,2949  ggml_backend_tensor_get_async(backend, t, dst, ...)
                                     writes embd_layer_inp[il] / layer_inp_accum
    src/llama-context.cpp:2985-2994  ggml_backend_event_record per participating
                                     backend (readiness epoch published only
                                     after EVERY tapped layer queued its copy)
    src/llama-context.cpp:3022-3030  tap_readback_waits saved and handed to the
                                     NEXT graph_compute as a device-side dependency
    src/llama-context.cpp:3380-3386  at each graph_compute:
                                       for (wait : tap_readback_waits)
                                           ggml_backend_event_wait(wait.first, wait.second);
                                     i.e. "this graph's copies into the persistent
                                     tap would overwrite rows the previous
                                     ubatch's D2H may still be reading" - handled
                                     as an event dependency, not a stall.

  drafter read side
    common/speculative.cpp:1798   llama_get_embeddings_layer_inp(ctx_tgt, layer_id)
    src/llama-context.cpp:5161-5167
        float * llama_get_embeddings_layer_inp(llama_context * ctx, uint32_t lid) {
            ctx->synchronize();                 <-- FULL ctx_tgt SYNC
            return ctx->get_embeddings_layer_inp(lid);
        }
    src/llama-context.cpp:917-924  llama_context::synchronize() ->
        ggml_backend_sched_synchronize(sched) - drains ALL backends of the
        target scheduler, so every queued D2H on every target device has landed
        before the host memcpy at speculative.cpp:1805.

  explicit barrier before the gather
    common/speculative.cpp:1771-1784
        // "Must be a full synchronize. The gather below reads embd_layer_inp,
        //  the per-batch buffer, while the accum readiness event only covers
        //  accum rows below p_end ..."
        llama_synchronize(ctx_tgt);
    This is an UNCONDITIONAL, deliberate full synchronize issued on every
    non-deferred (decode) chunk, before the feature gather.

ANSWER TO THE KEY QUESTION - is there an explicit event/barrier between the
target's write and the drafter's read when they are on DIFFERENT devices?
  YES. Three independent ordering guarantees, none of them implicit-only:
   (a) speculative.cpp:1783  llama_synchronize(ctx_tgt)  - unconditional full
       target-side barrier immediately before the gather;
   (b) llama-context.cpp:5162 ctx->synchronize() inside the exact API the
       drafter calls - a second full barrier, so a caller cannot skip it;
   (c) llama-context.cpp:3383-3386 ggml_backend_event_wait on every tap-readback
       event before the next graph overwrites the single-ubatch-deep tap buffer.
  The cross-device hop itself (tap tensor on dev_output -> host) is a plain
  async D2H on the target's own scheduler, not a peer copy, so there is no
  cross-device visibility window at all: (a)/(b) drain the producing device
  before the host touches the bytes. The one true peer-copy path,
  ggml_backend_cuda_cpy_tensor_async (ggml/src/ggml-cuda/ggml-cuda.cu:3906-3928:
  cudaMemcpyPeerAsync at 3910, copy_event record on the src stream at 3915-3919,
  cudaStreamWaitEvent on the dst stream at ~3924)
  also records an event on the src stream and cudaStreamWaitEvent's the dst
  stream; and the scheduler's own split copies use the BLOCKING
  ggml_backend_tensor_copy (ggml/src/ggml-backend.cpp:2795,2978; cross-device
  fallback ggml-backend.cpp:676-681 synchronizes BOTH backends before a blocking
  copy). No unprotected write->read window was found.

WHY THE SYMPTOM WAS ONCE ATTRIBUTED TO THIS PATH: src/llama-context.cpp:3037-3045
carries a comment describing exactly the hazard the old report suspected -
"ubatch i's readback is still draining ... That is why output varies run to run
once a drafter is attached while the same model without one is reproducible."
That comment documents a hazard that was ALREADY FIXED by the event dependency at
3383-3386; it is a historical note, not a live defect. Its presence plausibly
seeded the race hypothesis.

--------------------------------------------------------------------------------
4. METHOD / CONTROLS
--------------------------------------------------------------------------------
- Every remote command wrapped in `timeout`; every GPU leg under
  `flock -w 2700 /root/gpu.lock`, released between batches (4 runs/acquisition).
- Box was clean at dispatch and at teardown: 10 MiB used on GPU0/1/2, no
  llama-server, no sweep process, lock free, 38GB RAM available, temps reporting
  (no SMC wedge).
- VRAM parser self-tested before use (prints integer MiB; `awk -F': *'` avoided).
- Accept parser self-tested against a known-good prior log: 105 steps recovered.
- steps>0 asserted per run; zero would have aborted the arm loudly. No run
  produced zero steps in this campaign.
- Seed passed BOTH as request "seed": N and server --seed N; generation_settings
  in the response confirms "seed": 42 took effect.
- Sampling: temperature 1.0, top_k 20, top_p 0.95 (server defaults), seeded
  `dist` sampler - deterministic given a fixed seed, deliberately NOT greedy, so
  the test exercises the real sampling path rather than only argmax.
- 3 extra repeats per arm beyond the required 10 (r10/r11/r12 each arm).
- No git operations, /llama-swap untouched, no model/quant rebuild.

--------------------------------------------------------------------------------
5. WHAT THE PRIOR LANE ACTUALLY HAD
--------------------------------------------------------------------------------
/root/r8/ contains only FOUR completed runs: single_s42_r0..r3, ALL with the
same accepted-file sha (7ec57750c4d61bd4, 105 steps, acceptance 0.35800) - i.e.
the prior lane's single-card data also showed NO divergence. It never completed
a cross-card run (sweep.log stops at "START single s42 r4"), and
/root/r8/divergence.json was never written. The reported "steps=0" was the
documented parser bug (`accepted +[0-9]+/[0-9]+` missing the space after the
slash), now fixed and verified.
Corroborating prior record: commit 719671e773 (taskW2) already states the pinned
d4 drafter sealed "greedy 10/10 BITID, acceptance 0.373/2.4919 bit-equal,
placement numerics-invariant (replicated-vs-pinned cross-check BITID)".

--------------------------------------------------------------------------------
6. VERDICT
--------------------------------------------------------------------------------
DETERMINISTIC-BUT-SEED-SENSITIVE.

- Fixed seed 42: single-card 13/13 bit-identical, cross-card 13/13 bit-identical.
  Zero first-divergent steps in either arm. The multi-device path is
  numerically deterministic here.
- The acceptance nondeterminism reproduces exactly - and equally on ONE card -
  when the seed is llama's 0xFFFFFFFF sentinel, because
  src/llama-sampler.cpp:340-351 resolves that sentinel to std::random_device.
- Seed is the only variable that moves the numbers. Any claim of a cross-device
  race must first fix a real seed; none of the evidence supports a race, and the
  code has three explicit barriers on the suspected dependency.
- Residual unknown (stated plainly): this campaign fixes ONE prompt, ONE
  n_predict (256), ONE n_max (4). It does not prove determinism for concurrent
  multi-slot serving, adaptive depth (LLAMA_DFLASH_ADAPTIVE), or the deferred
  prefill path with prompts >~16k where the lag-one flush is active. Those are
  untested here, not exonerated.
================================================================================
