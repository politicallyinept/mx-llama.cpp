#!/bin/bash
# taskL chain: Q8_0 vs Q4_0 drafter, interleaved reps, d4 + d7, LDS-alone config.
# Re-runnable: legs skip on .done. Supervisor loops until ALL-DONE.
set -u
D=/root/taskL-drive.sh
L=/root/taskL/chain.log
mkdir -p /root/taskL
run() { echo "CHAIN $* $(date -Is)" >> "$L"; bash "$D" leg "$@" >> "$L" 2>&1; echo "CHAIN-RC=$? $* $(date -Is)" >> "$L"; }
# rep 1
run q8d  4 1 8541
run q4d  4 1 8542
run q8d  7 1 8543
run q4d  7 1 8544
run q4w0 7 1 8545
# rep 2
run q8d  4 2 8546
run q4d  4 2 8547
run q8d  7 2 8548
run q4d  7 2 8549
run q4w0 7 2 8550
# completion check
miss=0
for t in q8d-d4-r1 q4d-d4-r1 q8d-d7-r1 q4d-d7-r1 q4w0-d7-r1 q8d-d4-r2 q4d-d4-r2 q8d-d7-r2 q4d-d7-r2 q4w0-d7-r2; do
  [ -e "/root/taskL/$t.done" ] || { echo "MISSING $t" >> "$L"; miss=1; }
done
[ "$miss" = 0 ] && echo "CHAIN-ALL-DONE $(date -Is)" >> "$L"
exit "$miss"
