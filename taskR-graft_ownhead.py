#!/usr/bin/env python3
"""taskR: graft the target's token_embd.weight + output.weight into a copy of the
DFlash2 drafter GGUF, so the drafter references no target tensors and can be pinned
to a single device with -devd under a tensor-split target
(see src/models/dflash.cpp: a draft with its own embeddings + head references no
target tensors and can run on devices the target does not use).

Adapted from /root/taskH3/graft_ownhead.py (35B own-head graft). Differences:
  * parameterized: DRAFT_IN TARGET OUT via argv
  * dtype table extended to this fork's on-disk enum (empirically verified against
    gguf-py n_bytes on the 27B files): 0=F32, 1=F16, 2=Q4_0, 3=Q4_1, 4=Q5_0,
    5=Q5_1, 8=Q8_0, 9=Q8_1, 10=Q2_K, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K
  * gap-method cross-check: every tensor's dtype-derived size must equal the
    offset-gap to the next tensor (belt-and-braces on the dtype table)

Raw byte surgery per the gguf-nextn-tensor-graft procedure: streamed copies, no
numpy round-trip, dims in on-disk order, verify after write. Norms are untouched
(F32 dtype-0 preserved).
"""
import struct, sys, os

DRAFT_IN = sys.argv[1]
TARGET   = sys.argv[2]
OUT      = sys.argv[3]
GRAFT    = ["token_embd.weight", "output.weight"]

# on-disk dtype -> (block_size, block_bytes); None for plain types handled separately
BLOCK = {2: (32, 18), 3: (32, 20), 4: (32, 22), 5: (32, 24),
         8: (32, 34), 9: (32, 40),
         10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210)}
PLAIN = {0: 4, 1: 2, 6: 1, 7: 2}   # F32, F16, (unused slots kept conservative)

def u64(f): return struct.unpack("<Q", f.read(8))[0]
def u32(f): return struct.unpack("<I", f.read(4))[0]

def skip_string(f):
    n = u64(f); f.seek(n, 1)

def skip_value(f, t):
    if t in (0, 1, 7): f.seek(1, 1)   # u8/i8/bool = 1 byte
    elif t in (2, 3): f.seek(2, 1)
    elif t in (4, 5, 6): f.seek(4, 1)
    elif t == 8: skip_string(f)
    elif t == 9:
        et = u32(f); n = u64(f)
        if et in (0, 1, 7): f.seek(n, 1)
        elif et in (2, 3): f.seek(2 * n, 1)
        elif et in (4, 5, 6): f.seek(4 * n, 1)
        elif et in (10, 11, 12): f.seek(8 * n, 1)
        elif et == 8:
            for _ in range(n): skip_string(f)
        elif et == 9:
            for _ in range(n): skip_value(f, 9)
        else: raise ValueError("array elem type %d" % et)
    elif t in (10, 11, 12): f.seek(8, 1)
    else: raise ValueError("type %d" % t)

def parse_header(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    ver = u32(f); assert ver == 3, ver
    n_tensors = u64(f); n_kv = u64(f)
    kv_start = f.tell()
    align = 32
    for _ in range(n_kv):
        nl = u64(f); name = f.read(nl).decode("utf-8")
        t = u32(f)
        if name == "general.alignment" and t == 4:
            pos = f.tell(); align = u32(f); f.seek(pos)
        skip_value(f, t)
    tensors_start = f.tell()
    tensors = []
    for _ in range(n_tensors):
        nl = u64(f); name = f.read(nl).decode("utf-8")
        nd = u32(f); dims = [u64(f) for _ in range(nd)]
        dt = u32(f); off = u64(f)
        tensors.append((name, dims, dt, off))
    infos_end = f.tell()
    data_start = (infos_end + align - 1) // align * align
    fsize = os.path.getsize(path)
    f.close()
    return dict(n_tensors=n_tensors, n_kv=n_kv, kv_start=kv_start,
                tensors_start=tensors_start, infos_end=infos_end,
                data_start=data_start, align=align, tensors=tensors, fsize=fsize)

def nbytes(dt, dims):
    n = 1
    for d in dims: n *= d
    if dt in PLAIN: return n * PLAIN[dt]
    if dt in BLOCK:
        bs, bb = BLOCK[dt]
        assert n % bs == 0, "dtype %d needs block multiple" % dt
        return n // bs * bb
    raise ValueError("unsupported dtype %d" % dt)

def gap_sizes(hd):
    """dtype-agnostic sizes from offset gaps (data is contiguous, file ends at data end)."""
    ts = sorted(hd["tensors"], key=lambda x: x[3])
    out = {}
    for i, (n, d, dt, o) in enumerate(ts):
        if i + 1 < len(ts):
            nxt = ts[i + 1][3]
        else:
            nxt = hd["fsize"] - hd["data_start"]
        out[n] = nxt - o   # includes alignment padding; >= true size
    return out

def stream_copy(src, dst, abs_off, nb, chunk=1 << 26):
    src.seek(abs_off); rem = nb
    while rem > 0:
        b = src.read(min(chunk, rem))
        assert len(b) > 0
        dst.write(b); rem -= len(b)

def main():
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    hd = parse_header(DRAFT_IN)
    ht = parse_header(TARGET)
    td = {n: (d, dt, o) for (n, d, dt, o) in ht["tensors"]}
    dd = {n: (d, dt, o) for (n, d, dt, o) in hd["tensors"]}
    for n in GRAFT:
        assert n in td, "target missing %s" % n
        assert n not in dd, "draft already has %s" % n

    # cross-check dtype table against gap sizes on BOTH inputs (grafted tensors exact)
    gt, gd = gap_sizes(ht), gap_sizes(hd)
    for n in GRAFT:
        d, dt, o = td[n]
        assert nbytes(dt, d) == gt[n], "target %s: dtype-size %d != gap %d" % (n, nbytes(dt, d), gt[n])
    for (n, d, dt, o) in ht["tensors"]:
        assert nbytes(dt, d) <= gt[n], "target %s dtype-size exceeds gap (bad table)" % n
    for (n, d, dt, o) in hd["tensors"]:
        assert nbytes(dt, d) <= gd[n], "draft %s dtype-size exceeds gap (bad table)" % n
    print("dtype-table cross-check vs offset gaps: PASS")

    align = hd["align"]
    data_size = max(o + nbytes(dt, d) for (n, d, dt, o) in hd["tensors"])
    assert hd["fsize"] - hd["data_start"] == data_size, \
        "draft data_size %d != fsize-data_start %d" % (data_size, hd["fsize"] - hd["data_start"])
    offs = {}
    cur = (data_size + align - 1) // align * align
    for n in GRAFT:
        d, dt, o = td[n]
        offs[n] = (cur, ht["data_start"] + o, nbytes(dt, d), d, dt)
        cur = (cur + nbytes(dt, d) + align - 1) // align * align
    total_new = cur

    new_infos = b""
    for n in GRAFT:
        off, abs_off, nb, d, dt = offs[n]
        nb_name = n.encode("utf-8")
        new_infos += struct.pack("<Q", len(nb_name)) + nb_name
        new_infos += struct.pack("<I", len(d))
        for x in d: new_infos += struct.pack("<Q", x)
        new_infos += struct.pack("<I", dt) + struct.pack("<Q", off)

    new_data_start = (hd["infos_end"] + len(new_infos) + align - 1) // align * align
    with open(DRAFT_IN, "rb") as r, open(TARGET, "rb") as tr, open(OUT, "wb") as w:
        w.write(b"GGUF"); w.write(struct.pack("<I", 3))
        w.write(struct.pack("<Q", hd["n_tensors"] + 2))
        w.write(struct.pack("<Q", hd["n_kv"]))
        r.seek(24)
        w.write(r.read(hd["infos_end"] - 24))   # raw KV + raw tensor infos
        w.write(new_infos)
        w.write(b"\0" * (new_data_start - (hd["infos_end"] + len(new_infos))))
        stream_copy(r, w, hd["data_start"], data_size)
        pos = new_data_start + data_size
        for n in GRAFT:
            off, abs_off, nb, d, dt = offs[n]
            target_pos = new_data_start + off
            w.write(b"\0" * (target_pos - pos))
            stream_copy(tr, w, abs_off, nb)
            pos = target_pos + nb
    print("WROTE", OUT, os.path.getsize(OUT), "expected data end", new_data_start + total_new)
    assert os.path.getsize(OUT) == new_data_start + total_new, "file size mismatch"

    # ---- verify ----
    hv = parse_header(OUT)
    assert hv["n_tensors"] == hd["n_tensors"] + 2
    vd = {n: (d, dt, o) for (n, d, dt, o) in hv["tensors"]}
    for n in GRAFT:
        assert n in vd, "verify: %s missing" % n
        assert vd[n][:2] == td[n][:2], "verify: %s dims/dtype" % n
    for (n, d, dt, o) in hd["tensors"]:
        assert vd[n] == (d, dt, o), "verify: original %s moved" % n
    # byte-compare grafted tensors against target source (full)
    with open(OUT, "rb") as a, open(TARGET, "rb") as b:
        for n in GRAFT:
            off, abs_off, nb, d, dt = offs[n]
            a.seek(hv["data_start"] + off); b.seek(abs_off)
            rem = nb; bad = 0
            while rem > 0:
                x = a.read(min(1 << 24, rem)); y = b.read(len(x))
                if x != y: bad += 1
                rem -= len(x)
            assert bad == 0, "verify: %s byte mismatch" % n
            print("OK full byte-compare", n, nb, "bytes")
    # spot-check original tensors head/tail 64KB vs draft source
    with open(OUT, "rb") as a, open(DRAFT_IN, "rb") as r:
        for (n, d, dt, o) in hd["tensors"][:3] + hd["tensors"][-3:]:
            nb = nbytes(dt, d)
            for base in (0, max(0, nb - 65536)):
                a.seek(hv["data_start"] + o + base); r.seek(hd["data_start"] + o + base)
                ln = min(65536, nb - base)
                assert a.read(ln) == r.read(ln), "verify spot %s" % n
            print("OK spot", n)
    print("GRAFT-VERIFY-PASS")

if __name__ == "__main__":
    main()
