# taskX3 LEDGER - draft/verify overlap: feasibility-first refutation of the overlap proposal

Date: 2026-11-15 (PST). Box: mi50. Branch `int4-x3-overlap` off INT4-noquant tip 204739d62d
(NOT pushed). Analysis-only task: no code change, no GPU leg, no probe/KLD leg required
(nothing cleared the prototype bar, see section 5). Canonical probe sha16 77841cb6a251235d
(the brief's "77841cb6a235d" is the known corrupted copy, same convention as taskW1).

## 0. Verdict (headline)

**W1's death certificate CONFIRMS. The external review's overlap proposal (draft k+1 on the
pinned card while verify k runs on the TP pair) is refuted twice over:**

1. DATA DEPENDENCY (primary): draft k+1's launch gate is the accept walk's final output
   (id_last = last accepted/bonus token), which is the LAST thing verify k produces. There
   is zero slack inside the verify slice for a sound draft k+1 launch. Traced at file:line
   level in section 2.
2. BOTTLENECK INVARIANCE (secondary, kills the proposal even if the dependency vanished):
   the pinned draft card (-devd ROCm0 = visible GPU1) is rank 0 of the verify TP pair.
   Serial and "overlapped" both push the same 9.2 GiB through GPU1 per step (7.25 verify +
   1.95 draft), dominated by the same mul_mat_vec_q kernel family (80% of draft device
   time, taskW1 4c) => no complementary-bottleneck gain; step time is invariant to the
   interleaving. The proposal saves ~0 ms even with an oracle that breaks the dependency.

No relaxation reaches "strictly sound AND >=1.5 ms/step" (best surviving narrow version:
speculative draft-ahead on a DISJOINT card, EV ~0.3 ms/step, numerics-changing at
temp>0, and the disjoint card does not exist in production). No prototype built, per the
task's own conditional. Section 5 has the full EV math.

## 1. What draft k+1 consumes from verify k (file:line trace, tip 204739d62d)

Steady-state single-slot decode step, DFlash2 path (common_speculative_impl_draft_dflash,
is_dflash2=true). Call sites in tools/server/server-context.cpp (srv) and
common/speculative.cpp (spec):

| # | artifact produced by verify k | where consumed | when available |
|---|-------------------------------|----------------|----------------|
| D1 | target layer-input taps (hidden states at target_layer_ids) for every row of the verify-k batch (5 rows: anchor + 4 draft) | spec:1792-1814 gather (host memcpy from llama_get_embeddings_layer_inp after spec:1783 FULL llama_synchronize(ctx_tgt)); injected into draft KV by the fused encode+inject decode spec:1821-1837 (0.36 ms) | end of verify-k FORWARD (device), before logits/sampling |
| D2 | id_last = last accepted token (bonus token if all 4 accepted, else target's sampled token at the rejection point) | srv:3502-3512 draft params (.id_last = slot.sampled, .pos0 = pos_next); spec:1912 batch row 0 = dp.id_last (rest are mask tokens) | end of the ACCEPT WALK - the last output of verify k |
| D3 | pos0 = position after the accepted prefix | spec:1899,1912 (row positions n+i) | end of accept walk (host, same instant as D2) |
| D4 | draft-KV rollback: positions past the acceptance point must not persist | srv:3546 llama_memory_seq_rm(ctx_dft, ...); re-injection at the same absolute positions overwrites (position-indexed KV) | host, sub-ms |

Accept walk semantics (common/sampling.cpp:689 common_sampler_sample_and_accept_n):
per-position target sample vs draft token, first mismatch breaks, bonus sampled last.
DFlash2 fills no draft distribution (spec:1336 "reads its selector lattice from h_nextn
and never consumes raw logits"), so production temp>0 uses match-acceptance, not proper
rejection sampling - relevant to relaxation (a)'s numerics claim below.

Draft forward itself: spec:1931 llama_process(ctx_dft, DECODE, batch) = ONE monolithic
690-node HIP graph replay (taskW1: replay 96%, host launch 8-84 us, device 9.26 ms on
GPU1 while GPU2 idles); selector lattice read at spec:1960 (this sync carries the 9.26 ms
device wait), host walk spec:1971-2000 (~1 us).

## 2. Step timeline (means; taskW1 phase table + srv loop-timing lines)

```
 t=0.0   verify k launch: target decode width-5, TP2 GPU1+GPU2 (host submit ~0.6-2.6)
 t=34.4  verify-k forward DONE on device  <-- D1 (taps) available here
 t=34.4  spec process(): tgtsync 0.02 + gather 0.03 + inject 0.36 (draft KV now has D1)
 t=34.8  logits readback + sample + accept walk 0.4-0.7  <-- D2/D3 available here (LAST)
 t=35.5  srv pre_decode: draft params set (id_last, pos0); ckpt/seq_rm (D4)
 t=35.6  draft() launch: 690-node graph replay on GPU1, device 9.26 ms (GPU2 idle)
 t=44.9  selector walk ~0.001; gap draft->decode ~0.9 (batch build, sched)
 t=44.8  verify k+1 launches (consumes draft block k+1)
```

The draft launch gate is D2 = the accept walk's last output. The apparent slack between
D1 (t=34.4) and D2 (t=35.5) is ~1.1-1.5 ms of host work (inject + sample + accept +
batch build) and is already scheduled optimally: process() runs before sampling
(srv:4433 precedes post_decode sampling), i.e. everything that depends only on D1 already
runs as early as physically possible. The monolithic graph replay cannot be split to
start on D1 and wait mid-graph for D2.

## 3. Relaxations, quantified

Measured acceptance (taskW1 w1-meas-p-d4 server log, n=3421 drafts): per-position accept
P = (0.639, 0.431, 0.320, 0.233); conditional P(pos_i | pos_i-1) = (0.639, 0.674, 0.742,
0.728); mean accepted len 2.62; P(4/4 accepted) = 0.233.

### (a) draft-ahead-on-unverified (speculate past the last verified token)

Mechanics: at end of verify-k forward (D1 ready), launch draft k+1 with a GUESSED id_last
(the draft block k's own next-token proposal - free to obtain: extend the block to width
6, taskV3 showed draft device time is width-insensitive). Valid iff all 4 draft tokens
accepted AND bonus == guess; at greedy that is exactly P(5th draft token accepted)
~ 0.233 x 0.72 ~= 0.17.

Rollback cost math:
- Same pinned card (the review's proposal): the guessed draft's 1.95 GiB streams through
  GPU1 regardless of validity; when valid the step is bottleneck-invariant (~44.8 ms
  either way, section 0.2); when invalid (83%) a serial re-draft adds 9.26 ms.
  EV ~= 44.8 + 0.83 x 9.26 ~= 52.5 ms = 7.7 ms/step WORSE. Dead on arrival.
- Disjoint card (GPU3, disjoint bandwidth): hiding is capped by the D2->launch window
  (~1.5 ms), not by the 9.26 ms slice, because the draft still cannot start before verify
  k's forward completes (D1 gate). EV saving ~= 0.17 x 1.5 ~= 0.3 ms/step. And GPU3 is
  not deployable: production fleet runs llama-swap on ROCm1,2,3 (taskR remap); the
  benchmark pair 1,2 has no fourth card inside the serving shape.
- HIP graph survivability: mechanically survives (same replay, guessed row-0 token fed
  through the same staging buffer); rollback = full serial re-replay (9.26 ms device).
- Soundness: FAILS. At production temp>0 the accept walk compares target samples against
  the draft; a different draft changes accept boundaries and bonus draws -> the output
  stream changes (greedy temp-0 byte-identity would survive via verify's determinism, but
  production numerics change => KLD territory, not "strictly sound").

### (b) two-deep draft buffering (draft k+1 while verify k is still sampling)

The premise "if the only dependency is the final sampled token vs the full verify output"
does not hold: the dependency IS the final sampled token (D2) plus the accepted-row taps
(D1). The sampled token is the last thing the verify step produces; sampling is the tail
of the step, not a window to hide behind. Everything D1-only (gather+inject, 0.39 ms) is
already hoisted ahead of sampling (srv:4433). Nothing left to buffer; gain ~= 0 ms.
Graph structure untouched (no change).

### (c) verify-side overlap (prefill next batch while current verify finishes)

Single-stream tg (the champion metric, f128q4kv-p-d4 58.20) has no next batch: gain 0.
In-server, chunked prefill already interleaves pp rows into the same decode batch, and
the DFlash prefill path already ships the only real target/draft overlap that exists:
the lag-one deferred flush (spec:1714-1760) replays draft encodes for chunks the target
has ALREADY passed while it computes the newest chunk. That is the decode-analog's
proof-by-existence: overlap works where a ready-work queue exists; in the decode step
there is none (section 2). Gain to the 44.8 ms decode step: 0.

## 4. Why the GPU2-idle-during-draft window is not a free lunch either

GPU2 idles 9.26 ms/step while the draft runs on GPU1. TP-splitting the drafter across
both cards IS the borrowed path: measured worse (11.5 vs 9.26 ms, allreduce storm over
690 tiny kernels, taskR H3 / taskW1 5). Splitting only the draft HEAD (0.97 GiB, ~4.5 ms,
pure row-parallel matvec, no allreduce until the selector scores) onto GPU2 is the one
unexplored sliver - it is a device-parallelism change inside the draft slice (not
draft/verify overlap), numerics-neutral by construction (row-slice matvec), and belongs
to the kernel/pinning program, not this lane; flagging it for the parent as the only
overlap-shaped idea that survives with upside >1.5 ms (up to ~4.5 ms if GPU2's 226 GB/s
adds to GPU1's for the head matvec). Untested here: it changes -devd semantics and the
selector-lattice readback path; out of remit.

## 5. Feasibility verdict

| relaxation | strictly sound? | rollback bounded? | worth >=1.5 ms/step? | verdict |
|------------|-----------------|-------------------|----------------------|---------|
| (a) draft-ahead, pinned card | no (temp>0 numerics) | yes (1 re-draft) | EV -7.7 ms (negative) | REJECT |
| (a') draft-ahead, disjoint card | no (temp>0 numerics) | yes | EV ~0.3 ms; card unavailable in prod | REJECT |
| (b) two-deep buffering | n/a | n/a | ~0 ms (dependency is the step's last output) | REJECT |
| (c) verify-side prefill overlap | n/a | n/a | 0 ms on the decode metric | REJECT |

No relaxation is strictly sound AND worth >=1.5 ms/step => per the task's conditional,
NO prototype; refutation writeup delivered instead. No GPU legs were run; /root/gpu.lock
untouched; llama-swap untouched. The structural overlap that actually pays is MTP-style
fusion of the drafter into the target forward (draft rides the verify's weight reads at
~0 marginal cost, taskW1 7.3) - kernel-program territory, consistent with the 144 t/s
reference architecture.

## 6. Evidence index

- srv = tools/server/server-context.cpp @204739d62d: 3502-3512 (draft params: id_last=
  slot.sampled, pos0=pos_next), 3525 (draft() call), 3546 (draft KV seq_rm), 4433
  (process() before sampling), 4629 (sample_and_accept_n), 4637-4669 (tgt rollback path).
- spec = common/speculative.cpp: 1783 (full tgt sync), 1792-1837 (gather+inject, D1),
  1912 (row0=id_last), 1931 (monolithic draft decode), 1960-2000 (lattice read + walk),
  1714-1760 (prefill lag-one flush - the existing overlap).
- common/sampling.cpp:689 (accept walk; bonus last).
- Acceptance stats + loop timing: /root/taskW1/w1-meas-p-d4.server.log (per-position
  accept rates; "per iteration ... draft 9.64 + target decode+sync 34.65 + verify/accept
  0.68"; "decode -> accept 0.40, draft -> decode 0.90").
- Device mapping: /root/w1-leg.sh (HIP_VISIBLE_DEVICES pair + -devd ROCm0 => pinned
  draft card IS verify TP rank 0).
- Bandwidth floor + kernel census: taskW1 sections 4/4c (226 GB/s/GPU both phases;
  mul_mat_vec_q 80% of draft device time).

## 7. Status

COMPLETE (analysis). Ledger committed on int4-x3-overlap off 204739d62d, ASCII, NOT
pushed. No builds, no probes, no GPU legs, no numerics gates triggered (no code change).
