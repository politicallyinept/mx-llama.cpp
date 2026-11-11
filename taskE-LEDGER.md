# Task E: DFlash2 drafter Q8_0 -> Q4_0 requant (CPU-only prep)

Date: 2026-11-11 (PST). Performed with the already-built binary in this worktree;
no builds, no GPU work, no other files touched.

## Source
- Path: /mnt/old/home/brad/models/Qwen3.8-27B-DFlash2-Q8_0.gguf
- Size: 2,056,414,816 bytes
- sha256: c18e800daedc59ca68fd13b6a856d795746af6d399a9279ac6a277d1d422f87e
- Arch KV: general.architecture = dflash (LLM_ARCH_DFLASH), 51 KV pairs, 81 tensors
- Source dtype histogram: Q8_0 x49, F32 x32

## Command (exact)
    /root/int4-noquant-wt/build/bin/llama-quantize --allow-requantize \
      /mnt/old/home/brad/models/Qwen3.8-27B-DFlash2-Q8_0.gguf \
      /mnt/old/home/brad/models/Qwen3.8-27B-DFlash2-Q4_0.gguf Q4_0

Exit 0, no warnings/errors. Full stdout: taskE-logs/requant.log (untracked).
llama-quantize reported: model size 1950.71 MiB (8.50 BPW) -> quant size 1033.21 MiB (4.50 BPW), 2.2 s.
No fallback flags were needed (--tensor-type-file etc. not used).

## Output
- Path: /mnt/old/home/brad/models/Qwen3.8-27B-DFlash2-Q4_0.gguf
- Size: 1,094,346,336 bytes
- sha256: 92d2e2d5755e8a7fcaff21a3152f67d2333f781200e190c0727db28c09f42677
- Verified via gguf-py GGUFReader (header/metadata only, no tensor-data reads):
  - general.architecture = dflash (preserved); all 51 KVs preserved, none added/dropped
  - Tensor count: 81 in / 81 out; tensor name sets identical
  - Output dtype histogram: Q4_0 x49, F32 x32 (every Q8_0 tensor converted; none left behind)

## dtype histogram detail
- -> Q4_0 (49): fc.weight; selector_hidden.weight; selector_predecessor.weight;
  selector_successor.weight; per-block (x5): attn_{q,k,v,output}.weight,
  attn_conv_proj.weight, ffn_conv_proj.weight, ffn_{gate,up,down}.weight
- Stayed F32 (32): all attn_q_norm/attn_k_norm/attn_norm/ffn_norm weights,
  attn_conv_base and ffn_conv_base tensors, enc.output_norm.weight, output_norm.weight
- No f16 and no Q8_0 tensors remain; no token_embd/output.weight tensor exists in
  this drafter (output head comes from the target model), so --leave-output-tensor
  and --token-embedding-type were not applicable.

## Load-safety note for the parent (GPU leg)
- File is structurally intact for --spec-type draft-dflash: dflash.* KVs
  (block_count=5, conv_kernel_size=2, conv_group_size=16, selector_rank=256,
  selector_top_k=16, target_layers=[5,20,34,48,62], ...) all present.
- RISK: selector_predecessor.weight and selector_successor.weight
  (248320 x 256 vocab-side matrices) and selector_hidden.weight were REQUANTIZED
  to Q4_0. If the selector/lattice readout consumes these directly, 4-bit
  rounding can shift top-k candidate selection and change acceptance rate
  independently of the drafter speedup being measured. If acceptance drops
  unexpectedly, re-run with a --tensor-type-file keeping selector_*.weight at
  f16/q8_0 (they are only ~130 MB combined at Q8_0).
- conv_base tensors stayed F32 (conv state path unaffected); conv_proj weights
  are Q4_0.
