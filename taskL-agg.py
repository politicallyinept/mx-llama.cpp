#!/usr/bin/env python3
"""taskL aggregator: Q8_0 vs Q4_0 drafter net-tg table, acceptance + draft-fwd deltas."""
import json, os, re, hashlib, statistics
from collections import defaultdict

B = "/root/taskL"
LEGS = ["q8d-d4-r1","q4d-d4-r1","q8d-d7-r1","q4d-d7-r1","q4w0-d7-r1",
        "q8d-d4-r2","q4d-d4-r2","q8d-d7-r2","q4d-d7-r2","q4w0-d7-r2"]

def tg_med(tag):
    f = f"{B}/{tag}.reps.jsonl"
    if not os.path.exists(f): return None, 0
    ts = [json.loads(l)["t_s"] for l in open(f) if '"tg256"' in l]
    return (statistics.median(ts) if ts else None), len(ts)

def accept(tag):
    f = f"{B}/{tag}.reps.jsonl.agg.json"
    if not os.path.exists(f): return None, None, None
    a = json.load(open(f))
    return a.get("accept_rate"), a.get("draft_n_total"), a.get("draft_n_accepted_total")

def draft_fwd(tag):
    log = f"{B}/{tag}.server.log"
    if not os.path.exists(log): return None, 0
    fwd = []
    for l in open(log, errors="ignore"):
        m = re.search(r"draft-fwd call=\d+ n_tokens=(\d+) us=(\d+)", l)
        if m: fwd.append(int(m.group(2)))
    return (statistics.median(fwd) if fwd else None), len(fwd)

def tgt_w(tag, w):
    log = f"{B}/{tag}.server.log"
    if not os.path.exists(log): return None
    xs = []
    for l in open(log, errors="ignore"):
        m = re.search(rf"tgt-decode call=\d+ width={w} us=(\d+)", l)
        if m: xs.append(int(m.group(1)))
    return statistics.median(xs) if xs else None

rows = defaultdict(list)
for t in LEGS:
    arm = re.sub(r"-d\d-r\d$", "", t)
    d = re.search(r"-d(\d)-", t).group(1)
    rows[(arm, d)].append(t)

print("== per-leg: tg256 median | accept (acc/draft) | draft-fwd med us ==")
for (arm, d), tags in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    for t in tags:
        m, n = tg_med(t); a, dn, da = accept(t); f, nf = draft_fwd(t)
        print(f"{t:12s} tg={m if m is None else round(m,2)} (n={n})  accept={a} ({da}/{dn})  draft-fwd={f if f is None else round(f)}us (n={nf})")

def med2(arm, d, fn):
    vals = []
    for t in rows.get((arm, d), []):
        v = fn(t)[0] if fn not in (accept,) else fn(t)[0]
        if v is not None: vals.append(v)
    return statistics.median(vals) if vals else None

print("\n== net-tg table (median over reps) ==")
print(f"{'depth':5s} {'arm':6s} {'tg256':>8s} {'accept':>8s} {'draft-fwd ms':>12s}")
for d in ("4", "7"):
    for arm in ("q8d", "q4d", "q4w0"):
        if (arm, d) not in rows: continue
        tg = med2(arm, d, tg_med); ac = med2(arm, d, accept); df = med2(arm, d, draft_fwd)
        s_tg = "-" if tg is None else f"{tg:.2f}"
        s_ac = "-" if ac is None else f"{ac:.4f}"
        s_df = "-" if df is None else f"{df/1000:.2f}"
        print(f"d{d:4s} {arm:6s} {s_tg:>8} {s_ac:>8} {s_df:>12}")
    # deltas
    if ("q8d", d) in rows and ("q4d", d) in rows:
        t8, t4 = med2("q8d", d, tg_med), med2("q4d", d, tg_med)
        a8, a4 = med2("q8d", d, accept), med2("q4d", d, accept)
        f8, f4 = med2("q8d", d, draft_fwd), med2("q4d", d, draft_fwd)
        if t8 and t4:
            sa = "-" if (a4 is None or a8 is None) else f"{a4-a8:+.4f}"
            sf = "-" if (f4 is None or f8 is None) else f"{100*(f4-f8)/f8:+.1f}%"
            print(f"d{d}: q4d vs q8d  tg {100*(t4-t8)/t8:+.2f}%  accept {sa}  draft-fwd {sf}")
        if ("q4w0", d) in rows and t4:
            t0 = med2("q4w0", d, tg_med); a0 = med2("q4w0", d, accept); f0 = med2("q4w0", d, draft_fwd)
            if t0:
                sa0 = "-" if (a0 is None or a4 is None) else f"{a0-a4:+.4f}"
                sf0 = "-" if (f0 is None or f4 is None) else f"{100*(f0-f4)/f4:+.1f}%"
                print(f"d{d}: q4w0 vs q4d tg {100*(t0-t4)/t4:+.2f}%  accept {sa0}  draft-fwd {sf0}")

print("\n== greedy probes (64 tok temp0 seed42) ==")
probes = {}
for t in LEGS:
    f = f"{B}/probe-{t}.json"
    if not os.path.exists(f): continue
    try: txt = json.load(open(f)).get("content", "")
    except Exception: txt = "PARSE-FAIL"
    probes[t] = txt
    print(f"{t:12s} sha16={hashlib.sha256(txt.encode()).hexdigest()[:16]} head={txt[:50]!r}")
