#!/usr/bin/env python3
"""taskS aggregator: width-8 KLD table, short d7 acceptance, long-context plain + DFlash2 d4."""
import glob, hashlib, json, os, re, statistics

B = "/root/taskS"

for prefix, label in (("kld8-", "width-8 KLD gate (taskJ2 -b 8 protocol, 27B Q4_0, wiki-sub 64KB)"),
                      ("kld8q8-", "width-8 KLD gate (taskJ2 -b 8 protocol, 27B Q8_0 stack, wiki-sub 64KB)")):
    print(f"== {label} ==")
    names = ("q40off", "q40kld", "d1", "d2", "d1vsoff", "d2vsoff") if prefix == "kld8-" else \
            ("q40off", "q40kld", "d1", "d2", "d1vsbase", "d2vsbase")
    for name in names:
        f = f"{B}/{prefix}{name}.log"
        if not os.path.exists(f):
            continue
        txt = open(f, errors="ignore").read()
        mk = re.search(r"Mean    KLD:\s+(\S+)", txt)
        xk = re.search(r"Maximum KLD:\s+(\S+)", txt)
        mq = re.search(r"Mean PPL\(Q\):\s+(\S+)", txt)
        mp = re.search(r"Mean    PPL:\s+(\S+)", txt)
        hits = open(f + ".dot8-hits").read().strip() if os.path.exists(f + ".dot8-hits") else "?"
        rc = re.search(r"rc=(\d+)", txt)
        print(f"{name:9s} meanKLD={mk.group(1) if mk else '-':>10s} maxKLD={xk.group(1) if xk else '-':>10s} "
              f"PPL(Q)={mq.group(1) if mq else '-':>8s} PPL={mp.group(1) if mp else '-':>8s} "
              f"FA-DOT8-banner-hits={hits} rc={rc.group(1) if rc else '?'}")

print("\n== short d7 legs (verify width 8, natural suite, acceptance + tg) ==")
for f in sorted(glob.glob(f"{B}/s-*.reps.jsonl")):
    tag = os.path.basename(f).replace(".reps.jsonl", "")
    ts = [json.loads(l)["t_s"] for l in open(f) if '"tg256"' in l]
    blog = open(f"{B}/{tag}.bench.log", errors="ignore").read()
    agg = re.findall(r"AGG (\{.*\})", blog)
    acc = mal = "-"
    if agg:
        d = json.loads(agg[-1]); acc = f"{d['accept_rate']:.4f}"; mal = f"{d['mean_accept_len']:.3f}"
    med = statistics.median(ts) if ts else 0
    print(f"{tag:8s} tg256 median {med:6.2f} t/s (n={len(ts)})  accept={acc} mal={mal}")
    pf = f"{B}/probe-{tag}.json"
    if os.path.exists(pf):
        try:
            txt = json.load(open(pf)).get("content", "")
            print(f"         greedy sha16={hashlib.sha256(txt.encode()).hexdigest()[:16]} head={txt[:50]!r}")
        except Exception as e:
            print(f"         probe parse fail {e}")

print("\n== plain long-context decode (llama-bench, tg256 at depth) ==")
for f in sorted(glob.glob(f"{B}/plain-*.jsonl")):
    tag = os.path.basename(f).replace(".jsonl", "")
    for l in open(f):
        d = json.loads(l)
        if d.get("test_mode") == "tg" or "tg" in str(d.get("test", "")):
            print(f"{tag:22s} {d['test']:8s} {d['avg']:.2f} t/s  ({d.get('model','')[-20:]})")

print("\n== DFlash2 d4 long-context legs ==")
for f in sorted(glob.glob(f"{B}/lc-*.q.jsonl")):
    tag = os.path.basename(f).replace(".q.jsonl", "")
    lines = [json.loads(l) for l in open(f)]
    prime = next((l for l in lines if l["phase"] == "prime"), None)
    qs = [l for l in lines if l["phase"].startswith("q")]
    tgs = [q["tg_t_s"] for q in qs if q.get("tg_t_s")]
    dn = sum(q.get("draft_n") or 0 for q in qs)
    da = sum(q.get("draft_n_accepted") or 0 for q in qs)
    pn = prime.get("prompt_n") if prime else None
    pp = f"{prime['pp_t_s']:.1f} t/s" if prime and prime.get("pp_t_s") else "-"
    line = f"{tag:16s} prime_pn={pn} pp={pp} tg_med={statistics.median(tgs):6.2f} t/s" if tgs else f"{tag:16s} no-q-data"
    if dn:
        line += f"  accept={da/dn:.4f} (dn={dn} da={da})"
    print(line)
    prof = f"{B}/{tag}.prof"
    if os.path.exists(prof):
        w5 = [int(m.group(2)) for l in open(prof)
              if (m := re.search(r"tgt-decode call=\d+ width=(\d+) us=(\d+)", l)) and int(m.group(1)) == 5]
        if w5:
            print(f"                 tgt-decode w5 median {statistics.median(w5)/1000:.1f} ms (n={len(w5)})")
