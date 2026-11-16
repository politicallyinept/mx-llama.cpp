#!/usr/bin/env python3
"""k2-mb-sweep.py — taskK2 microbench sweep driver (runs on mi50).

For each T in TLIST and each arm (gate-off generic, gate-on W=4, gate-on W=8):
run the same mb_k2_head binary as a subprocess under flock on one clean card,
parse the GRAPH line, assert bit-identity of the hash vs the gate-off arm,
append a TSV row. usage: k2-mb-sweep.py <mb-binary> [Tlist]
env: N K IT CARD LOCK
"""
import os, re, subprocess, sys

mb = sys.argv[1]
tlist = (sys.argv[2].split() if len(sys.argv) > 2 else ["1", "2", "3", "4", "5"])
N = os.environ.get("N", "248320")
K = os.environ.get("K", "5120")
IT = os.environ.get("IT", "15")
CARD = os.environ.get("CARD", "1")
LOCK = os.environ.get("LOCK", "/root/gpu.lock")
OUT = os.environ.get("OUT", "/root/taskK2-mb.tsv")

env0 = dict(os.environ)
trlib = "/root/therock-venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib"
env0["LD_LIBRARY_PATH"] = trlib + ":" + env0.get("LD_LIBRARY_PATH", "")

pat = re.compile(r"hash=([0-9a-f]+) min_us=([\d.]+) med_us=([\d.]+) mean_us=([\d.]+) wGB/s=([\d.]+)")

def run(T, w):
    e = dict(env0)
    e["HIP_VISIBLE_DEVICES"] = CARD
    if w:
        e["GGML_DFLASH_HEAD_KERNEL"] = "1"
        e["GGML_DFLASH_HEAD_W"] = str(w)
    else:
        e.pop("GGML_DFLASH_HEAD_KERNEL", None)
        e.pop("GGML_DFLASH_HEAD_W", None)
    cmd = ["flock", "-w", "2700", LOCK, mb, "graph", T, N, IT, K]
    r = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=3000)
    m = pat.search(r.stdout)
    banner = "DFLASH-HEAD-KERNEL" in r.stderr
    if not m:
        print(f"  T={T} W={w} FAILED rc={r.returncode}\n{r.stdout[-400:]}\n{r.stderr[-400:]}", flush=True)
        return None
    return m.groups() + ("banner" if banner else "nobanner",)

with open(OUT, "w") as f:
    f.write("T\tW\thash\tmin_us\tmed_us\tmean_us\twGBps\tbitexact_vs_off\tbanner\n")
    for T in tlist:
        base = run(T, 0)
        for w in (0, 4, 8):
            g = run(T, w)
            if not g or not base:
                continue
            exact = "yes" if g[0] == base[0] else "NO"
            f.write(f"{T}\t{w}\t{g[0]}\t{g[1]}\t{g[2]}\t{g[3]}\t{g[4]}\t{exact}\t{g[5]}\n")
            f.flush()
            print(f"T={T} W={w} hash={g[0]} med={g[2]}us {g[4]}GB/s exact={exact} {g[5]}", flush=True)
print("done ->", OUT)
