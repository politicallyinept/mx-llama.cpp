# task B2 ledger -- i8narrow dispatch wiring + stock head-to-head

Branch: INT4-noquant-dispatch (worktree /root/wt-dispatch), base 058e420c76.
Code commit: 04a57ca2be709c4dbed64fab56ae5465418dfefe.
Date: 2026-11-11. Toolchain: TheRock amdclang++
(/root/therock-venv/lib/python3.12/site-packages/_rocm_sdk_devel/bin/amdclang++),
`-x hip --offload-arch=gfx906 -O3 -std=gnu++17 -DGGML_USE_HIP
-D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -D_GNU_SOURCE -D_XOPEN_SOURCE=600`
+ worktree includes. No cmake build, no model loads; GPU use = microbench only
under `flock -w 600 /root/gpu.lock`.

## 1. What was wired (mul-mat.cu)

- `rp_i8narrow_on()`: env gate `GGML_CUDA_I8NARROW` (unset/0 = OFF), witness
  `GGML_LOG_WARN "[I8NARROW] gate ACTIVE ..."`.
- New branch at the TOP of the narrow-batch window (`ne11 >= 1 && ne11 <=
  rp_narrow_max`, ~line 343), before the existing activation staging:
  gate && `src0->type == GGML_TYPE_Q4_0` &&
  `ggml_cuda_repack_mul_mat_should_fire(src0)` && `ne11 in {1,2,4,8,16}`
  -> quantize src1 via `quantize_mmq_i8split_cuda` into three pool planes
  (hi 16 B, lo 16 B, d f16 2 B per (token, 32-block); count
  ne13*ne12*ne11*x_stride blocks -- same count as the u4 narrow path), then
  per (i3,i2) slice call `mul_mat_i8narrow_launch(T=ne11, w, hi, lo, d,
  dst_d, ne00, ne01, xs=x_stride, ys=dst_s1, stream)` and return.
- Admission: the window is only reached when should_fire held (caller gate);
  the branch re-checks it explicitly. The repacked Q4_0 record IS the
  i8narrow-compatible [qs|d] layout (rp_traits<Q4_0>: raw nibble plane +
  f16 d plane at ne1*qs_stride; kernel sums nibbles inline -- no SUMW plane).
  Widths 3/5/6/7 are not instantiated in the launcher switch and fall
  through to the existing path (no padding: the kernel writes all T columns).
- Quantizer invoked with `ne0 = ne10_padded` (token stride = x_stride), the
  same convention as the u4 narrow-path call; per-slice plane offsets taken
  in each plane's own element size.

## 2. Device quantizer (w4a4-i8narrow.cu, replaced the GGML_ABORT stub)

`quantize_mmq_i8split` + launcher `quantize_mmq_i8split_cuda(..., ne2, ne3,
stream)` (ne3 added to the .cuh decl -- nothing else called the stub).
Mirror of `quantize_mmq_q8_1_u4` (quantize.cu:579-672): same grid/block
(128 threads, 4 values each, z = ne2*ne3), same de-interleaved thread map
(byte j = {val j low, val j+16 high}), same `__shfl_xor_sync` fold style;
q8_1 SYMMETRIC semantics per quantize_row_q8_1_ref: amax over the 32-block,
d = amax/127 (f32, stored f16), id = 1/d, a = roundf(x*id) clamped
[-127,127], then hi = (a>>4)&0xF, lo = (a&0xF)^8. Token-major
ib = (z*ne1 + x)*nkb + k_block. Padding blocks get the zero-input pattern
(never read by the kernel).
VERIFIED: microbench `[devq]` check -- device planes bit-match the host
packer quantize_row_i8split on ALL planes, both shapes
(hi 40960/40960, lo 40960/40960, d 2560/2560; 139264/139264, 8704/8704).

## 3. Byte-identical-off argument

With GGML_CUDA_I8NARROW unset, `rp_i8narrow_on()` is false: the new branch
is a constant-false dispatch-site if evaluated before untouched code -- no
allocations, no kernel launches, no numeric change; the serving path is the
same code as the base commit. The quantizer replaces a stub nothing else
called. The .cuh signature change only affects the stub's replacement. The
harness TU is excluded from the ggml build (CMake `*_test.cu` filter).

## 4. Single-TU compile evidence (TheRock amdclang++, gfx906)

- w4a4-i8narrow.cu: clean, 2.8 s.
- mul-mat.cu: clean, 15.4 s -- only the two PRE-EXISTING
  `mmq_gemm_repacked ... amdgpu-waves-per-eu` occupancy warnings.
- mb_i8narrow_test.cu: links to a runnable binary (own main + ggml_abort
  stub, as before).

## 5. Stock head-to-head (mb_i8narrow_test.cu, flock /root/gpu.lock)

Stock arm = the kernels the narrow window runs TODAY for repacked Q4_0 with
q8_1 activations (DOT8 gate off): T=1 `mul_mat_vec_rp<Q4_0,16,16,false>`
(grid (N+15)/16 x 1024); T=2 lanes16 `<32,8,2,1,16>` x 512 (both shapes
satisfy the dispatch's work_ok + grid>=2*nsm conditions); T=4/8
`<2,1,T,2,64>` x 64; T=16 `<2,1,16,2,64>` = nc EXTRAPOLATION (today width
16 goes to the GEMM path). Activations: host q8_1 packer (bit-equal codes
to the i8split packer). Correctness: double reference of the stock affine
epilogue incl. its f16 s-fold -- all widths verify ok (maxrel <= 0.0084).
NOTE: stock and i8narrow differ legitimately on near-zero outputs (stock
folds -8*sum via the f16 s field, i8narrow folds it exactly on the weight
side), so the stock arm verifies against its own reference, not vs i8out.

Medians (ms), 35 reps skip-5, same shapes as the i8narrow arm:

| T  | i8 (5120x17408) | stock (5120x17408) | stock/i8 (smaller = i8 slower) | vs 95% bar | i8 (17408x5120) | stock (17408x5120) | stock/i8 (smaller = i8 slower) | vs 95% bar |
|----|-----------------|--------------------|----------|------------|-----------------|--------------------|----------|------------|
| 1  | 0.328           | 0.104              | 0.317    | FAIL       | 0.449           | 0.093              | 0.207    | FAIL       |
| 2  | 0.358           | 0.137              | 0.383    | FAIL       | 0.489           | 0.134              | 0.275    | FAIL       |
| 4  | 0.337           | 0.157              | 0.465    | FAIL       | 0.677           | 0.147              | 0.217    | FAIL       |
| 8  | 0.391           | 0.299              | 0.765    | FAIL       | 0.899           | 0.266              | 0.296    | FAIL       |
| 16 | 0.510           | 0.645 (extrap)     | 1.265    | PASS       | 1.567           | 0.586 (extrap)     | 0.374    | FAIL       |

VERDICT: the adoption bar (>= 95% of stock nc at widths 1-8) is NOT met:
i8narrow runs at 32-77% of stock on K=5120 N=17408 and 21-30% on
K=17408 N=5120. It only beats the (extrapolated) nc shape at T=16 on the
wide-N shape. The dispatch gate therefore stays DEFAULT OFF; wiring is
in place for kernel-side optimization work to A/B against stock without
touching the serving path.

## 6. Pre-existing issues observed (NOT introduced here)

- i8narrow T=16 on K=5120 N=17408: 87040/278528 bad (NaN tail, maxrel
  1.7e9) -- reproduced IDENTICALLY with the pristine harness binary; owned
  by the sibling /root/wt-t16fix worktree.

  SUPERSEDED (taskP, 2026-11-13): taskA2-LEDGER + commit b392f48d proved this
  T=16 corruption was NOT a kernel bug. The mb harness hipMemcpy'd the f16
  activation scale (d) plane with the ELEMENT count of a vector<uint16_t>
  instead of its byte count, under-copying the plane so the kernel read
  uninitialized device memory for token planes c>=8. That harness root-cause
  supersedes the "real bug" framing in this section and in the sibling
  /root/wt-t16fix record. The width-16 dispatch admission was subsequently
  removed (taskP) pending a determinism re-proof on the fixed harness.
- i8narrow K=17408 N=5120 widths >= 4: 1 element out of 20480/40960 at
  maxrel 0.021 (tolerance 1e-2) -- reproduced with the pristine binary;
  borderline fp32-contraction class.
- i8narrow K=17408 N=5120 T8/T1 = 1.99x (bar 1.5x) -- pre-existing kernel
  characteristic on the tall-K shape (task C validated the bar on shape 1).

## 7. Files touched

- ggml/src/ggml-cuda/q8_repack/mul-mat.cu (+65: include, gate, branch)
- ggml/src/ggml-cuda/q8_repack/w4a4-i8narrow.cu (+~110: quantizer)
- ggml/src/ggml-cuda/q8_repack/w4a4-i8narrow.cuh (ne3 in decl)
- ggml/src/ggml-cuda/q8_repack/mb_i8narrow_test.cu (+186: stock arm,
  host q8_1 packer, cpu_ref_q81, devq bit-match check)
- taskB2-LEDGER.md (this file)
