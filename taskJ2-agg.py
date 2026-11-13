#!/usr/bin/env python3
"""taskJ2 aggregator: natural tg256 medians, DFLASH-PROF step phases, greedy probes, KLD."""
import json, os, re, sys, hashlib, statistics

B = "/root/taskJ2"
TAGS = [t for t in ("lds-r1", "base-r1", "base-r2", "lds-r2", "rq-r1", "rp-r1", "oldrp-r1")
        if os.path.exists(f"{B}/{t}.reps.jsonl")]

print("== natural suite tg256 median (bench.py, draft 7) ==")
med = {}
for t in TAGS:
    ts = [json.loads(l)["t_s"] for l in open(f"{B}/{t}.reps.jsonl")
          if '"tg256"' in l]
    if ts:
        med[t] = statistics.median(ts)
        acc = open(f"{B}/{t}.bench.log", errors="ignore").read() if os.path.exists(f"{B}/{t}.bench.log") else ""
        m = re.findall(r"draft acceptance = .*", acc)
        print(f"{t:10s} median {med[t]:7.2f} t/s  (n={len(ts)})  {m[-1] if m else ''}")

if "base-r1" in med and "lds-r1" in med:
    b = statistics.median([med["base-r1"], med.get("base-r2", med["base-r1"])])
    l = statistics.median([med["lds-r1"], med.get("lds-r2", med["lds-r1"])])
    print(f"DELTA lds vs base: {100*(l-b)/b:+.2f}%")
if "rq-r1" in med and "base-r1" in med:
    print(f"rq vs fork-base: {100*(med['rq-r1']/med['base-r1']-1):+.2f}%")

print("\n== DFLASH-PROF step phases (us/call) ==")
for t in TAGS:
    log = f"{B}/{t}.server.log"
    if not os.path.exists(log):
        continue
    fwd, tgt = [], {}
    for l in open(log, errors="ignore"):
        m = re.search(r"draft-fwd call=\d+ n_tokens=(\d+) us=(\d+)", l)
        if m:
            fwd.append(int(m.group(2)))
            continue
        m = re.search(r"tgt-decode call=\d+ width=(\d+) us=(\d+)", l)
        if m:
            tgt.setdefault(int(m.group(1)), []).append(int(m.group(2)))
    fmed = statistics.median(fwd) if fwd else 0
    line = f"{t:10s} draft-fwd med {fmed:7.0f}us (n={len(fwd)})"
    for w in sorted(tgt):
        if w <= 8:
            line += f" | tgt w{w} med {statistics.median(tgt[w]):6.0f}us n={len(tgt[w])}"
    print(line)

print("\n== greedy probes (64 tok, temp0 seed42, drafter d7) ==")
probes = {}
for t in TAGS:
    f = f"{B}/probe-{t}.json"
    if not os.path.exists(f):
        continue
    try:
        txt = json.load(open(f)).get("content", "")
    except Exception:
        txt = "PARSE-FAIL"
    probes[t] = txt
    print(f"{t:10s} sha16={hashlib.sha256(txt.encode()).hexdigest()[:16]} head={txt[:60]!r}")
def cmp2(a, b):
    if a in probes and b in probes:
        x, y = probes[a], probes[b]
        if x == y:
            print(f"probe {a} == {b}: BYTE-IDENTICAL ({len(x)} chars)")
        else:
            i = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
            print(f"probe {a} vs {b}: diverge at char {i}: {x[max(0,i-20):i+20]!r} vs {y[max(0,i-20):i+20]!r}")
cmp2("rp-r1", "oldrp-r1")   # gate-off byte-id proof (new vs old binary, repack pair)
cmp2("lds-r1", "base-r1")   # gate-on divergence (expected)
cmp2("lds-r1", "lds-r2")    # determinism across reps

print("\n== KLD ==")
for arm in ("base", "lds"):
    log = f"{B}/kld-{arm}.log"
    if os.path.exists(log):
        for l in open(log, errors="ignore"):
            if re.search(r"Mean\s+KLD|Maximum KLD|Mean PPL\(Q\)|Mean\s+PPL\b|^rc=", l):
                print(arm, "|", l.strip())

print("\n== plain llama-bench pp512/tg256 ==")
for arm in ("base", "lds"):
    f = f"{B}/plain-{arm}-r1.jsonl"
    if os.path.exists(f):
        for l in open(f):
            r = json.loads(l)
            kind = f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}"
            print(f"plain-{arm}: {kind} = {r['avg_ts']:.2f} t/s")
