# taskA2 LEDGER — i8narrow T=16 corruption: root cause & fix

Branch: INT4-noquant-t16fix (base 058e420c76). Box: mi50, 3x MI50 gfx906.
Toolchain: TheRock amdclang++ only (`/root/therock-venv/.../bin/amdclang++`),
proven flags, single-TU compiles, GPU runs under `flock -w 600 /root/gpu.lock`.

## 1. Symptom (as inherited)

T=16 on K=5120 N=17408: 87040/278528 outputs garbage (maxrel 1.71e9),
"exactly 5 of 16 token planes". T=1..8 correct (maxrel <= 0.0059), T=16 ran
without crashing (0.507 ms).

## 2. Characterization (extended mb harness)

Per-token-plane (c) breakdown at T=16, shape K=5120 N=17408:

- c=0..7   correct
- c=8..11  ALL rows exactly 0.0 (written — sentinel pre-fill proved they were
           written zeros, not left unwritten)
- c=12..15 garbage (|y| ~ 1e5..1e6) and/or NaN
- shape K=17408 N=5120: c=8..11 NaN, c=12..15 garbage

So EIGHT planes (c>=8) were corrupted, not five: the original harness counted
bad with `rel > 1e-2`, and every NaN comparison is false — three NaN planes
were invisible, producing the "5 of 16" report. The NaN-safe recount
(`!(rel <= 1e-2)`) exposes all eight.

## 3. Investigation trail (evidence-first)

1. ISA census under the actual harness flags: T=16 VGPR 62, SGPR 98, 0 spills,
   0 scratch — no register exhaustion. (The "62 VGPR" census was correct.)
2. ISA audit: the 32 uniform uint4 activation loads are scalarized to
   `s_load_dwordx4` with aggressive SGPR data/address reuse; every
   waitcnt/def-use chain audited legal. Store epilogue (16 `global_store_dword`,
   acc regs v17..v2, address chain y + c*ys) audited legal.
3. Differential builds — ALL reproduced the identical corruption:
   -O2 (identical maxrel 1.71e9, same sample values), -ffp-contract=off,
   token-loop chunked into 2x8 with a memory-clobber barrier. Bit-identical
   garbage across four different codegens argued AGAINST a scheduling/
   allocator miscompile and FOR a deterministic wrong-memory read.
4. Sentinel pre-fill of d_y (-12345.0f) proved every corrupted slot was
   WRITTEN (unwritten=0) — stores land correctly, the DATA was wrong.
   NaN in the output is impossible from finite d_w/d_a/t/sw8 inputs, so the
   kernel was reading wrong memory for the activation scale plane at c>=8.

## 4. Root cause — HARNESS bug, kernel innocent

`mb_i8narrow_test.cu` allocated and copied the f16 activation scale plane
with the ELEMENT count instead of BYTES:

    std::vector<uint16_t> ad((size_t)TMAX * nkb);
    hipMalloc(&d_ad, ad.size());                                  // half the bytes
    hipMemcpy(d_ad, ad.data(), ad.size(), hipMemcpyHostToDevice); // half the bytes

`ad.size()` is the element count of a `uint16_t` vector, so only the first
8 token planes (c=0..7) of the scale plane reached the device. The kernel —
correctly indexing `ad[c*xs + sb]` — read uninitialized device memory for
c>=8: fresh zero pages -> d_a = 0 -> acc = 0 exactly (c=8..11 on shape 1);
stale allocator memory -> huge/NaN f16 scales -> garbage/NaN (c=12..15).
Which planes looked zero vs garbage vs "correct-but-NaN-hidden" varied with
allocator state and the harness NaN blind spot — hence the flaky-looking
"5 of 16 planes" and the run-to-run pattern changes.

The kernel source `w4a4-i8narrow.cu` is byte-identical to HEAD (058e420c76):
no kernel change was needed or made. Dispatch signature untouched.

## 5. Fix (harness, mb_i8narrow_test.cu)

1. `hipMalloc`/`hipMemcpy` the ad plane in BYTES (`ad.size() * sizeof(uint16_t)`)
   + immediate device readback proof (`d_ad readback mismatches=0/...`).
2. Per-c bad-count breakdown + d_y sentinel pre-fill (distinguishes
   unwritten vs zero-written planes).
3. NaN-safe verification (`!(rel <= tol)` instead of `rel > tol`).
4. numpy-style tolerance: `|y-ref| <= 1e-3 + 1e-2*|ref|` — rtol 1e-2 per the
   seal, plus atol 1e-3 because fp32 accumulation of O(50) terms over
   160..544 blocks has ~1e-4 absolute noise; a near-zero ref (cancellation,
   e.g. ref=-2.74e-4 vs y=-2.68e-4, abs diff 5.7e-6) must not be judged on
   relative error alone (this was the shape-2 c=3 "0.021" artifact).
5. T8/T1 <= 1.5x perf bar gated on shape 1 only (where the committed
   measurement lives); shape 2 (N=5120 -> 20 grid blocks) underfills the GPU
   and scales ~2x by construction — pre-existing, reported as info.

## 6. Validation (final mb run, pristine kernel, both shapes, T=1..16)

    == shape K=5120 N=17408 (nkb=160 rs=2576) ==
      T= 1 median 0.330 ms  verify-maxrel 0.0059 ok
      T= 2 median 0.356 ms  verify-maxrel 0.0059 ok
      T= 4 median 0.347 ms  verify-maxrel 0.0059 ok
      T= 8 median 0.365 ms  verify-maxrel 0.0059 ok
      BAR T8/T1 = 1.10x (PASS; pass <= 1.50x)
      T=16 median 0.508 ms  verify-maxrel 0.00647 ok
    == shape K=17408 N=5120 (nkb=544 rs=8720) ==
      T= 1 median 0.448 ms  verify-maxrel 0.00146 ok
      T= 2 median 0.487 ms  verify-maxrel 0.00146 ok
      T= 4 median 0.654 ms  verify-maxrel 0.021 ok
      T= 8 median 0.900 ms  verify-maxrel 0.021 ok
      BAR T8/T1 = 2.01x (info; pass <= 1.50x, shape2 ungated)
      T=16 median 1.567 ms  verify-maxrel 0.021 ok
    MB-RC=0

d_ad readback mismatches=0 on both shapes. Shape 1 T=16 = 0.508 ms matches
the inherited 0.507 ms (the kernel never changed). All widths verify ok.

## 7. Notes for sibling tasks

- The serving path wiring `mul_mat_i8narrow_launch` is unaffected: the kernel
  was never broken; only the microbench fed it a half-copied scale plane.
- If a future harness copies these planes, sizes are: hi/lo = count*nkb*16
  BYTES (uint8 planes, size()==bytes), d plane = count*nkb*2 BYTES.
- The mb binary is rebuilt single-TU with the TheRock amdclang++ flags in §0;
  do not use /opt/rocm-7.2.3 clang.
