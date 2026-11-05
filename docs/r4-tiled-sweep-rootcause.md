# R4 tiled-repack sweep — corrected root cause (READ BEFORE DEBUGGING THE W=2 CRASH)

This note corrects an earlier misdiagnosis. The prior agent reported a
"jump-to-NULL during the first repacked mul_mat dispatch" in
`tests/test-tiled-kl-sweep.cpp`. That was wrong.

## What actually happened

The crash was in the **harness's own output readback**, not in any matmul:

```
#0  0x0000000000000000 in ?? ()
#1  run_dump(...) — return addr = call ggml_backend_tensor_get@plt + 2
```

Chain:
1. `ggml/src/ggml-cuda/q8_repack/buffer.cu:723-732` — the repack buffer
   **deliberately nulls `get_tensor`** in its interface. Weights are write-once;
   the inherited CUDA `get_tensor` would misread the two-plane layout as
   canonical.
2. `ggml_backend_tensor_get()` (ggml/src/ggml-backend.cpp:504) has **no null
   guard** on `buft->iface.get_tensor`.
3. Harness v3 allocated its OUTPUT tensors on the repack buft; the first
   `ggml_backend_tensor_get` on one of those → NULL call → SIGSEGV (rip=0x0).

The compute itself completed — dispatch banners print before the fault.

## Evidence

- `GGML_CUDA_RP_NARROW_MAX_TOKENS=1` (routes narrow batches away from the nc
  mat-vec family) did **NOT** stop the crash → the defect was never in the
  mat-vec family.
- Harness v4 (`tests/test-tiled-kl-sweep.cpp` @ 9a1ea83ed, on CV-port as
  8a81dea82) allocates weights on the repack buft and activations/outputs on
  the default buft — exactly like production. All 29 shapes × 3 seeds then
  PASS; zero untestable shapes; full sweep verdict PASS.
- Production exposure: **none**. Production weights are write-once and
  `ggml_backend_tensor_get` is never called on a repack-buft tensor
  (outputs/logits live on the default buft). The w=2 repacked dense matvec
  that MTP verify batches do exercise ran fine in the sweep.

## Two real follow-ups

1. **Latent bug (unfixed):** `ggml_backend_tensor_get()` calls
   `buft->iface.get_tensor` without a null check. Any future caller reading a
   repack-buft tensor gets a SIGSEGV instead of a clean error. Fix = null
   guard: log once and return without copying (or GGML_ABORT with a clear
   message).
2. **Diagnostic lesson:** tail-call frames made gdb attribute the fault to
   `run_dump` rather than `ggml_backend_tensor_get`. When a fault frame is
   `0x0 in ?? ()`, check the *return address* of the calling frame before
   naming a kernel.

Reference: full report at `/root/r-closeout/report-r4.md` (box-local).
