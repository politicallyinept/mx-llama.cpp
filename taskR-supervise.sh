#!/bin/bash
# taskR supervisor: after the in-flight matrix pass, re-run matrix.sh to retry any
# legs skipped by LOCKFAIL (refuse-if-exists skips completed ones) until all 24 done.
NEED="f0q4-b-d4-r1 f0q4-p-d4-r1 f0q4-b-d7-r1 f0q4-p-d7-r1 f0q4-b-d4-r2 f0q4-p-d4-r2 f0q4-b-d7-r2 f0q4-p-d7-r2 f128q4w4-b-d4-r1 f128q4w4-p-d4-r1 f128q4w4-b-d7-r1 f128q4w4-p-d7-r1 f128q4w4-b-d4-r2 f128q4w4-p-d4-r2 f128q4w4-b-d7-r2 f128q4w4-p-d7-r2 q8lds-b-d4-r1 q8lds-p-d4-r1 q8lds-b-d7-r1 q8lds-p-d7-r1 q8lds-b-d4-r2 q8lds-p-d4-r2 q8lds-b-d7-r2 q8lds-p-d7-r2"
missing() { local m=""; for t in $NEED; do [ -e /root/taskR/$t.done ] || m="$m $t"; done; echo $m; }
while pgrep -f "bash matrix.sh" >/dev/null 2>&1; do sleep 60; done
for i in 1 2 3 4 5 6; do
  M=$(missing); [ -z "$M" ] && break
  echo "SUPERVISOR pass $i missing:$M $(date -Is)"
  bash /root/taskR/matrix.sh
done
M=$(missing)
if [ -z "$M" ]; then echo "SUPERVISOR-ALL-DONE $(date -Is)"; else echo "SUPERVISOR-INCOMPLETE missing:$M $(date -Is)"; fi
