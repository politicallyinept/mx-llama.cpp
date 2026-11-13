# taskP cleanup ledger -- external-review response

Branch: int4-p-cleanup (created from INT4-noquant tip 3d122c72b5).
Date: 2026-11-13. Scope: CPU/docs/code edits only; no GPU work, no build run
(parent builds after merge). One commit per logical group; ASCII messages;
NOT pushed (parent merges + pushes).

## Review items addressed (9)

### 1. Committed binaries removed -- commit 5c99a02bfd
Review item: build artifacts committed under scripts/isa-census/.
- `git rm` scripts/isa-census/{carved.dis, carved.elf, gfx906.hsaco,
  mb_i8narrow_test, mb_merged, rock_carved.elf, rock_hsaco, symbols.txt}
  (8 files, ~350 KB incl. 4 binaries).
- .gitignore: added `scripts/**/*.elf` and `scripts/**/*.hsaco` (the `**`
  form so scripts/isa-census/ subdirs match; `*` alone does not cross `/`).
- scripts/isa_census.sh kept (the committed source that regenerates dumps).
- Done-when verified: `git ls-files scripts/isa-census` -> 0 files.

### 2. dctl.seq bounds guards -- commit d9296db99d
Review item: dctl.seq[seq_id] indexed without the bounds check accept() has.
- common/speculative.cpp begin(): `if (dctl.on)` ->
  `if (dctl.on && seq_id >= 0 && seq_id < (llama_seq_id) dctl.seq.size())`
  (out-of-range bails the depth-decision block; begin()'s own n_seq guard
  above is untouched).
- common/speculative.cpp draft path (n_draft selection): the ternary now
  requires `seq_id < dctl.seq.size()` before reading `.depth`; out-of-range
  falls back to params.n_max.
- Gate-off byte-identical: both changes sit inside dctl.on evaluation; with
  the gate off both expressions reduce to the originals.

### 3. taskB2-LEDGER.md record fixes -- commit 104e823a07
Review item (a): section 5 ratio column mislabeled. The column labeled
"i8/stock" contains stock_time/i8_time (e.g. T=1: 0.104/0.328 = 0.317).
Relabeled both occurrences to "stock/i8 (smaller = i8 slower)". Verdict
text unchanged (it already reads the values as speed ratios: "32-77% of
stock").
Review item (b): section 6 T=16 bullet annotated SUPERSEDED: taskA2-LEDGER +
commit b392f48d proved the T=16 corruption was the mb harness d-plane
under-copy (hipMemcpy with the element count of a vector<uint16_t> instead
of bytes -> token planes c>=8 read uninitialized device memory), NOT a
kernel bug. The note supersedes taskB2's "real bug" framing and the sibling
/root/wt-t16fix record.

### 4. T=16 dispatch removal -- commit 943c86fbba
Review item: T=16 admitted despite the unresolved determinism question.
- ggml/src/ggml-cuda/q8_repack/mul-mat.cu: i8narrow admission set
  {1,2,4,8,16} -> {1,2,4,8}; banner text and the admission comment updated
  to match; comment references the pending GPU determinism re-proof on the
  fixed harness (taskA2/b392f48d root-cause) as the reason for the
  conservative removal.
- Gate GGML_CUDA_I8NARROW is default OFF -> default serving unaffected.
- The kernel-side width-16 instantiation (w4a4-i8narrow.cu) and the mb
  harness width list (mb_i8narrow_test.cu) are left in place: the harness is
  the tool the future re-proof lane needs.

### 5. 0.0499 KLD label sweep -- commit f9bbb6d077
Review item: 0.0499 mislabeled as fork-sealed. Per
docs/KLD-LADDER-w4a4-forced.md the correct labels are: 0.0499 = plain Q4
weights scored via the sibling (ruquant) engine; fork repack-OFF = 0.0490;
fork sealed repack = 0.0656; ruquant W4A4 = 0.0653.
- grep -rn '0\.0499' across the repo (root ledgers, docs/, README) found
  occurrences ONLY in: taskF-LEDGER.md (the provenance finding itself --
  correct), taskN-FLOOR0-LEDGER.md (3x, cites the number without the
  sealed label -- correct usage, untouched), taskM1-LEDGER.md (A0 control
  reference -- correct), docs/KLD-LADDER-w4a4-forced.md (the reference doc
  -- correct), README.md (no occurrences), and RESULTS-microbench-addendum.md
  (THE mislabel).
- CHANGED: RESULTS-microbench-addendum.md -- "era-1 runtime KLD 0.0756 vs
  sealed 0.0499" -> "vs plain-Q4 anchor 0.0499", plus a LABEL FIX note
  stating the four correct labels with the ladder-doc pointer.

### 6. README ruquant attribution -- commit f71ecae3cf
Review item: README credits castagna-veloce but not the ruquant work.
- README.md Attribution/Provenance: new bullet naming
  skyne98/mx-llama.cpp branch `ruquant` + skyne98/ruquant as the source of
  the rq-fattn kernel (587 lines, MIT) and the LDS-staged narrow matvec
  design; port sources pinned to commits 047d127, b9bf750, 82c73b5 (the
  source comments already cite e.g. "ruquant 82c73b5 port"). Existing
  credits untouched.

### 7. KROT witness banner -- commit 5bc7ec66a7
Review item: LLAMA_KV_KROT/LLAMA_KV_KROT_V read via plain getenv with no
activation witness.
- src/llama-kv-cache.cpp (constructor, after the two getenv reads): added
  the once-per-process static-lambda WARN banner convention the other gates
  use (pattern: speculative.cpp fast_banner / rq-fattn.cu GGML_CUDA_FA_DOT8
  static gate). Prints once when either gate is on; env values are
  process-invariant so the first construction decides. Gate-off: no output,
  no behavior change.

### 8. FA_DOT8 device-consistency guard -- commit 707a7bf58c
Review item: alloc path consults ggml_cuda_rq_fattn_q4_applies with the
BUFFER's device while dispatch uses the current device.
- ggml/src/ggml-cuda/fattn.cu dispatch (ggml_cuda_flash_attn_ext):
  `ggml_cuda_get_device()` -> `ctx.device` (identical value -- the function
  opens with ggml_cuda_set_device(ctx.device) -- but the explicit pairing
  removes the implicit ambient-device dependency and matches the alloc-size
  callback's parameter).
- ggml/src/ggml-cuda/fattn.cu alloc path
  (ggml_cuda_flash_attn_ext_get_alloc_size): comment documenting the
  same-device assumption: applies() uses `device` only for the gfx906 CC
  check; the CUDA backend computes each graph on the device that owns its
  buffers (one context per device), so the two consults coincide; the
  comment names the invariant to enforce if mixed-device graphs ever appear.
  Minimal diff: no behavior change.

### 9. GGML_DFLASH_PROF counters atomic -- commit 93a2a7601f
Review item: plain static counters where ggml-cuda.cu already uses atomics.
- common/speculative.cpp: dprof_n_gather, dprof_n_inj, dprof_n_fwd ->
  `static std::atomic<unsigned long long>` (matches
  `dprof_calls[GGML_CUDA_MAX_DEVICES]` in ggml-cuda.cu); added
  `#include <atomic>`. `++counter` on std::atomic returns the value
  (seq_cst RMW), so the fprintf %llu args are unchanged.

## Files changed (whole branch)

- .gitignore
- scripts/isa-census/ (8 files deleted from tracking)
- common/speculative.cpp (guards x2, atomics x3, include)
- taskB2-LEDGER.md
- ggml/src/ggml-cuda/q8_repack/mul-mat.cu
- RESULTS-microbench-addendum.md
- README.md
- src/llama-kv-cache.cpp
- ggml/src/ggml-cuda/fattn.cu
- taskP-CLEANUP-LEDGER.md (this file)

## Verification

- No build run (no build dir in this lane; parent builds after merge).
  Syntax sanity: every edited hunk re-read from `git diff` -- braces
  balanced, ternary split valid, lambda captures by value legal, ctx in
  scope at the fattn.cu site, atomic prefix-++ returns unsigned long long.
- Gate-off byte-identical argument holds for items 2 (dctl.on-gated), 4
  (gate default OFF), 7 (banner only prints when gate on).
- `git ls-files scripts/isa-census` -> empty (item 1 done-when).
- 0.0499 sweep re-grepped after the fix: no remaining fork-sealed mislabels.
