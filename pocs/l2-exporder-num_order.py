#!/usr/bin/env python3
# taskL2 numerics harness: fp32 summation-order delta between the CURRENT cross-expert
# accumulation order (left fold over top-k slots in argsort order, as built by
# llama-graph.cpp build_ffn: ggml_mul(weights) then ggml_add chain) and a GROUPED
# order (fold in ascending expert-id order, what a grouped/fused epilogue would do),
# on representative 35B MoE shapes: n_embd=2048, k=8, verify width W in {1,5,8}.
# Also quantifies the FMA-vs-mul-add delta a fused kernel would introduce if it does
# not use explicit __fmul_rn/__fadd_rn (the real numerics question for a fused sum).
import numpy as np

rng = np.random.default_rng(42)
NE, K = 2048, 8

def orders(s, w, perm):
    # s: [K, W, NE] fp32 expert down outputs (pre-weight); w: [K, W, 1] weights
    t = (w * s).astype(np.float32)          # current graph: separate rounded mul
    # A: current left fold, slot order 0..K-1
    accA = t[0].copy()
    for i in range(1, K):
        accA = (accA + t[i]).astype(np.float32)
    # B: grouped by expert id (perm: [K, W] slot index order per token)
    accB = np.zeros_like(t[0])
    for j in range(s.shape[1]):  # token
        a = t[perm[0, j], j, :].copy()
        for i in range(1, K):
            a = (a + t[perm[i, j], j, :]).astype(np.float32)
        accB[j] = a
    # C: pairwise tree (reference alternative)
    ts = [t[i].copy() for i in range(K)]
    while len(ts) > 1:
        ts = [ (ts[i] + ts[i+1]).astype(np.float32) if i+1 < len(ts) else ts[i]
               for i in range(0, len(ts), 2) ]
    accC = ts[0]
    # D: fp64 ground truth
    accD = (w.astype(np.float64) * s.astype(np.float64)).sum(axis=0)
    # E: fused-kernel FMA variant: acc = fma(w_i, s_i, acc) (single rounding)
    accE = np.zeros_like(t[0])
    for i in range(K):
        accE = (w[i, :, 0][:, None].astype(np.float64) * s[i].astype(np.float64) + accE.astype(np.float64)).astype(np.float32)
    return accA, accB, accC, accD, accE

print(f"{'sigma':>8} {'W':>3} {'max|A-B|':>10} {'max|A-C|':>10} {'max|A-D|':>10} {'max|A-E|':>10} {'max|A|':>8}")
for sigma in [0.0316, 0.316, 3.16, 31.6]:
    for W in [1, 5, 8]:
        s = (rng.standard_normal((K, W, NE)) * sigma).astype(np.float32)
        logits = rng.standard_normal((K, W)).astype(np.float32)
        e = np.exp(logits - logits.max(axis=0))
        w = (e / e.sum(axis=0)).astype(np.float32)[:, :, None]
        # grouped order: sort slots by expert id (random expert assignment per token)
        exp_ids = np.stack([rng.permutation(256)[:K] for _ in range(W)]).T  # [K, W]
        perm = np.argsort(exp_ids, axis=0).astype(np.int64)
        A, B, C, D, E = orders(s, w, perm)
        print(f"{sigma:8.4f} {W:3d} {np.abs(A-B).max():10.3e} {np.abs(A-C).max():10.3e} "
              f"{np.abs(A.astype(np.float64)-D).max():10.3e} {np.abs(A-E).max():10.3e} {np.abs(A).max():8.2f}")
