#!/usr/bin/env python3
"""w4a4-outlier-analysis.py — offline analysis of GGML_W4A4_DUMP activation dumps.

Reads the record file written by the debug hook in quantize_mmq_q8_1_u4_cuda
(perf/w4a4-attrib): magic 'U4DP' u32 | name_len u32 | name | ne00 i64 | ne1 i64
| float32 [ne1][ne00] (row = token).

Produces, per dumped tensor:
  1. u4 per-32 quantization error stats (the shipped scheme):
     d4 histogram, fraction of blocks whose range exceeds k x median range.
  2. Fix-candidate headroom, measured as MSE reduction of the activation
     quantization (the quantity that drives the KLD gap):
       (a) per-16 sub-block scales: requantize with block=16.
       (b) clipping: per-32 asymmetric with clamp at c*sigma of the block
           (c in {3,4,6}) — accept outlier error, recover resolution.
       (c) SmoothQuant-style per-channel equalization: s_j = max|x_j|^alpha
           folded into the preceding norm/weight; measure residual block
           range concentration + top-1% channel range share (s-vector sparsity).
Usage: python3 w4a4-outlier-analysis.py DUMPFILE [OUTFILE]
"""
import sys, struct, json
import numpy as np

def read_records(path):
    recs = []
    with open(path, "rb") as f:
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            magic, = struct.unpack("<I", hdr)
            assert magic == 0x50443455, f"bad magic {magic:#x}"
            nlen, = struct.unpack("<I", f.read(4))
            name = f.read(nlen).decode()
            ne00, ne1 = struct.unpack("<qq", f.read(16))
            x = np.frombuffer(f.read(int(ne00) * int(ne1) * 4), dtype=np.float32)
            recs.append((name, x.reshape(int(ne1), int(ne00)).copy()))
    return recs

def u4_quant_err(x, block=32, clip_sigma=None):
    """MSE of per-`block` asymmetric uint4 quant (d=(mx-mn)/15), fp16-rounded
    scale like the kernel. If clip_sigma, clamp block to c*std around mean."""
    n, k = x.shape
    xb = x.reshape(n, k // block, block).astype(np.float64)
    mn = xb.min(axis=-1, keepdims=True)
    mx = xb.max(axis=-1, keepdims=True)
    if clip_sigma is not None:
        mu = xb.mean(axis=-1, keepdims=True)
        sd = xb.std(axis=-1, keepdims=True)
        mn = np.maximum(mn, mu - clip_sigma * sd)
        mx = np.minimum(mx, mu + clip_sigma * sd)
    d4 = (mx - mn) / 15.0
    d4_16 = d4.astype(np.float16).astype(np.float64)  # kernel stores fp16 scale
    m_16 = mn.astype(np.float16).astype(np.float64)
    id4 = np.where(d4_16 != 0, 1.0 / np.where(d4_16 != 0, d4_16, 1.0), 0.0)
    q = np.clip(np.rint((xb - m_16) * id4), 0, 15)
    xq = q * d4_16 + m_16
    err = ((xq - xb) ** 2).mean()
    return err, d4.squeeze(-1)

def analyze(name, x):
    out = {"tensor": name, "shape": list(x.shape)}
    e32, d4 = u4_quant_err(x, 32)
    e16, _ = u4_quant_err(x, 16)
    out["mse_per32"] = float(e32)
    out["mse_per16"] = float(e16)
    out["per16_mse_reduction_pct"] = float(100.0 * (1 - e16 / e32))
    # block-range concentration
    xb = x.reshape(-1, 32).astype(np.float64)
    rng = xb.max(-1) - xb.min(-1)
    med = np.median(rng)
    for k in (4, 8, 16):
        out[f"frac_blocks_range_gt_{k}x_median"] = float((rng > k * med).mean())
    out["d4_p50"] = float(np.median(d4)); out["d4_p99"] = float(np.percentile(d4, 99))
    out["d4_max"] = float(d4.max())
    # clipping variants
    for c in (3.0, 4.0, 6.0):
        ec, _ = u4_quant_err(x, 32, clip_sigma=c)
        out[f"mse_clip{int(c)}sigma"] = float(ec)
        out[f"clip{int(c)}sigma_mse_reduction_pct"] = float(100.0 * (1 - ec / e32))
    # per-channel stats for SmoothQuant
    ch_absmax = np.abs(x).max(axis=0)
    ch_range = x.max(axis=0) - x.min(axis=0)
    srt = np.sort(ch_range)[::-1]
    top1 = max(1, len(srt) // 100)
    out["top1pct_channels_range_share"] = float(srt[:top1].sum() / srt.sum())
    out["ch_range_max_over_median"] = float(ch_range.max() / np.median(ch_range))
    # SmoothQuant alpha sweep: equalize per-channel by s_j = absmax_j^alpha,
    # requantize x/s (the fold moves s into the preceding weight/norm).
    best = None
    for a in (0.25, 0.5, 0.75, 1.0):
        s = (ch_absmax + 1e-12) ** a
        xs = x / s
        es, _ = u4_quant_err(xs, 32)
        # error in ORIGINAL space: (x/s quant err)*s per element, weighted 1:1
        # because the matmul sees (x/s)·(s·W): elementwise error scales by s too.
        # Measure the s-weighted MSE properly:
        n, k = xs.shape
        xbs = xs.reshape(n, k // 32, 32).astype(np.float64)
        mn = xbs.min(-1, keepdims=True); mx = xbs.max(-1, keepdims=True)
        d4s = ((mx - mn) / 15.0).astype(np.float16).astype(np.float64)
        ms = mn.astype(np.float16).astype(np.float64)
        ids = np.where(d4s != 0, 1.0 / np.where(d4s != 0, d4s, 1.0), 0.0)
        q = np.clip(np.rint((xbs - ms) * ids), 0, 15)
        err_sq = ((q * d4s + ms - xbs) ** 2) * (s.reshape(1, k // 32, 32) ** 2)
        w_err = float(err_sq.mean())
        out[f"smoothquant_a{a}_weighted_mse"] = w_err
        out[f"smoothquant_a{a}_mse_reduction_pct"] = float(100.0 * (1 - w_err / e32))
        if best is None or w_err < best[1]:
            best = (a, w_err)
    out["smoothquant_best_alpha"] = best[0]
    # worst blocks: (token, block) by range, top-5
    idx = np.argsort(rng)[::-1][:5]
    out["worst_blocks"] = [[int(i // 32), int((i % 32) * 32), float(rng[i])] for i in idx]
    # worst channels top-5
    ci = np.argsort(ch_range)[::-1][:5]
    out["worst_channels"] = [[int(i), float(ch_range[i])] for i in ci]
    return out

def main():
    path = sys.argv[1]
    recs = read_records(path)
    print(f"records: {len(recs)}")
    results = [analyze(n, x) for n, x in recs]
    txt = json.dumps(results, indent=1)
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w") as f:
            f.write(txt)
    print(txt)

if __name__ == "__main__":
    main()
