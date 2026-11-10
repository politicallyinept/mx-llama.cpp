#!/usr/bin/env python3
"""GGUF header KV dumper (read-only, header only - safe on huge files, no GPU).
Usage: rq_gguf_kv.py FILE [substr ...]   # prints KVs whose key contains any substr (case-insensitive); no substr = all.
Exit 0. Used as the rotation-metadata witness (ruquant.w4a4 / ruquant.drafter keys)."""
import struct
import sys

def kvs(path):
    out = []
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit("not a GGUF file: " + path)
        f.read(4)  # version
        f.read(8)  # n_tensors
        nkv = struct.unpack("<Q", f.read(8))[0]

        def rs():
            l = struct.unpack("<Q", f.read(8))[0]
            return f.read(l).decode("utf-8", "replace")

        def rv(t):
            if t == 0: return struct.unpack("<B", f.read(1))[0]
            if t == 1: return struct.unpack("<b", f.read(1))[0]
            if t == 2: return struct.unpack("<H", f.read(2))[0]
            if t == 3: return struct.unpack("<h", f.read(2))[0]
            if t == 4: return struct.unpack("<I", f.read(4))[0]
            if t == 5: return struct.unpack("<i", f.read(4))[0]
            if t == 6: return struct.unpack("<f", f.read(4))[0]
            if t == 7: return struct.unpack("<?", f.read(1))[0]
            if t == 8: return rs()
            if t == 9:
                et = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                vals = [rv(et) for _ in range(n)]
                return vals if n <= 8 else vals[:8] + ["...%d total" % n]
            raise SystemExit("unknown GGUF KV type %d" % t)

        for _ in range(nkv):
            k = rs()
            t = struct.unpack("<I", f.read(4))[0]
            out.append((k, rv(t)))
    return out

def main():
    path = sys.argv[1]
    subs = [s.lower() for s in sys.argv[2:]]
    for k, v in kvs(path):
        if not subs or any(s in k.lower() for s in subs):
            s = str(v)
            print("%s = %s" % (k, s if len(s) <= 200 else s[:200] + "..."))

main()
