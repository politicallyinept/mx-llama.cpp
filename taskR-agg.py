#!/usr/bin/env python3
"""taskR pindraft aggregator: net-tg table + mandatory gates.
Per leg: tg256 median (bench agg), acceptance (bench agg), greedy probe sha16,
DFLASH-PROF draft-fwd us/step median. Per (arm,depth): b vs p medians, delta,
acceptance digit-match, probe byte-identity."""
import json, glob, os, re, statistics, hashlib

BASE = "/root/taskR"

def probe_sha(path):
    try:
        d = json.load(open(path))
        txt = d.get("content") or d.get("content_details", {}).get("content", "")
        return hashlib.sha256(txt.encode()).hexdigest()[:16]
    except Exception as e:
        return "ERR:%s" % e

def draft_fwd_median(log):
    us = []
    try:
        for l in open(log, errors="ignore"):
            m = re.match(r"\[DFLASH-PROF\] draft-fwd call=\d+ n_tokens=\d+ us=(\d+)", l)
            if m: us.append(int(m.group(1)))
    except FileNotFoundError:
        return None, 0
    if not us: return None, 0
    return statistics.median(us), len(us)

rows = {}
for f in sorted(glob.glob(BASE + "/*.reps.jsonl.agg.json")):
    tag = os.path.basename(f).replace(".reps.jsonl.agg.json", "")
    a = json.load(open(f))
    m = re.match(r"(f0q4|f128q4w4|q8lds)-(b|p)-d(\d)-r(\d)", tag)
    if not m: continue
    arm, head, depth, rep = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    fwd, nfwd = draft_fwd_median(BASE + "/%s.server.log" % tag)
    rows.setdefault((arm, depth), {}).setdefault(head, {})[rep] = dict(
        tg=a.get("tg_t_s_median"), pp=a.get("pp_t_s_median"),
        accept=a.get("accept_rate"), length=a.get("mean_accept_len"),
        dn=a.get("draft_n_total"), da=a.get("draft_n_accepted_total"),
        probe=probe_sha(BASE + "/probe-%s.json" % tag), fwd=fwd, nfwd=nfwd)

print("%-10s %-4s %-5s %8s %8s %8s %9s %18s" %
      ("arm", "dep", "head", "tg", "accept", "len", "fwd-us", "probe16"))
for (arm, depth) in sorted(rows):
    for head in ("b", "p"):
        for rep in sorted(rows[(arm, depth)].get(head, {})):
            r = rows[(arm, depth)][head][rep]
            print("%-10s d%-3d %-5s %8s %8s %8s %9s %18s" %
                  (arm, depth, head, r["tg"], r["accept"], r["length"],
                   r["fwd"], r["probe"]))

print("\n== A/B summary (median over reps; gates) ==")
verdicts = []
for (arm, depth) in sorted(rows):
    b, p = rows[(arm, depth)].get("b", {}), rows[(arm, depth)].get("p", {})
    if not b or not p:
        print("%s d%d: INCOMPLETE b=%s p=%s" % (arm, depth, sorted(b), sorted(p))); continue
    btg = statistics.median([b[r]["tg"] for r in b])
    ptg = statistics.median([p[r]["tg"] for r in p])
    delta = 100.0 * (ptg - btg) / btg
    accs = {(b[r]["accept"], b[r]["length"]) for r in b} | {(p[r]["accept"], p[r]["length"]) for r in p}
    probes = {b[r]["probe"] for r in b} | {p[r]["probe"] for r in p}
    bf = statistics.median([b[r]["fwd"] for r in b if b[r]["fwd"]]) if any(b[r]["fwd"] for r in b) else None
    pf = statistics.median([p[r]["fwd"] for r in p if p[r]["fwd"]]) if any(p[r]["fwd"] for r in p) else None
    ok_acc = len(accs) == 1
    ok_pr = len(probes) == 1
    print("%-10s d%d: b %.2f -> p %.2f tg (%+.2f%%) | accept %s %s | probe %s | fwd-us %s -> %s" %
          (arm, depth, btg, ptg, delta,
           sorted(accs)[0] if ok_acc else "MISMATCH %s" % sorted(accs),
           "DIGIT-MATCH" if ok_acc else "FAIL",
           "BYTE-IDENTICAL" if ok_pr else "MISMATCH %s" % sorted(probes),
           bf, pf))
    verdicts.append((arm, depth, btg, ptg, delta, ok_acc, ok_pr))
allok = all(v[5] and v[6] for v in verdicts) and len(verdicts) == 6
print("\nGATES:", "ALL-PASS" if allok else "SEE ABOVE (%d/6 cells complete)" % len(verdicts))
