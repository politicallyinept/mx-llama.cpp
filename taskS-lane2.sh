#!/bin/bash
# taskS lane-2 chain: Q8_0 width-8 KLD legs, then the decisive DFlash2 d4
# long-context legs (kernel arm first; 64k first). Retries absorb gpu.lock
# contention with the sibling lanes.
set -u
DRV=/root/wt-s-fadot8/taskS-drive.sh
bash /root/wt-s-fadot8/taskS-q8kld.sh
for L in 64k 32k; do
  for A in q40d1 q40 f16; do
    P=8734; [ "$L" = 32k ] && P=8731
    for t in 1 2 3; do
      bash "$DRV" lcd4leg "$A" "$L" "$P" && break
      echo "LANE2 RETRY-$t lcd4leg $A $L $(date -Is)"; sleep 60
    done
  done
done
echo "LANE2-CHAIN-END $(date -Is)"
