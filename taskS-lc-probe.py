#!/usr/bin/env python3
"""taskS long-context probe: prime the KV with a long wikitext doc, then ask short
questions (ruquant card protocol). Each request is doc + question with cache_prompt
so only the question tokens are newly prefilled. Emits one JSON line per phase:
prime + q1..q3 with prompt_n, tg t/s, draft acceptance counters."""
import hashlib, json, sys, urllib.request

port = int(sys.argv[1]); doc = open(sys.argv[2]).read(); out = sys.argv[3]
BASE = "http://127.0.0.1:%d" % port

def post(obj, timeout=3600):
    req = urllib.request.Request(BASE + "/completion", data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def spec(t):
    dn = t.get("draft_n", t.get("spec_draft_n"))
    da = t.get("draft_n_accepted", t.get("spec_draft_n_accepted"))
    return dn, da

QS = [
    "In two sentences, summarize the main topic of the passage above.",
    "List three specific facts stated in the passage.",
    "What is the most surprising detail in the passage? Explain briefly.",
]

lines = []
# prime: doc alone, 1 token, populates the prefix cache
r = post({"prompt": doc, "n_predict": 1, "temperature": 0, "seed": 42,
          "ignore_eos": True, "cache_prompt": True})
t = r.get("timings", {})
lines.append({"phase": "prime", "prompt_n": t.get("prompt_n"),
              "prompt_ms": t.get("prompt_ms"),
              "pp_t_s": (t.get("prompt_n", 0) / (t.get("prompt_ms", 1) / 1000.0)) if t.get("prompt_ms") else None})
for i, q in enumerate(QS):
    r = post({"prompt": doc + "\n\nQuestion: " + q + "\nAnswer:", "n_predict": 128,
              "temperature": 0, "seed": 42, "ignore_eos": True, "cache_prompt": True})
    t = r.get("timings", {})
    dn, da = spec(t)
    pn = t.get("prompt_n"); pm = t.get("prompt_ms")
    lines.append({"phase": "q%d" % (i + 1), "prompt_n": pn, "prompt_ms": pm,
                  "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
                  "tg_t_s": (t.get("predicted_n", 0) / (t.get("predicted_ms", 1) / 1000.0)) if t.get("predicted_ms") else None,
                  "draft_n": dn, "draft_n_accepted": da,
                  "text_sha16": hashlib.sha256(r.get("content", "").encode()).hexdigest()[:16]})
with open(out, "w") as f:
    for l in lines:
        f.write(json.dumps(l) + "\n")
print("probe done:", " | ".join("%s pn=%s tg=%s" % (l["phase"], l.get("prompt_n"),
      round(l["tg_t_s"], 2) if l.get("tg_t_s") else "-") for l in lines))
