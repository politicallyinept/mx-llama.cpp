#!/bin/bash
# GPU wedge diagnostics (mi50-hardened-operations vet: mgmt health != compute health)
echo "== my leg sessions =="
pgrep -af "leg_t.s[h]" | head -3 || echo none
echo "== rocminfo gfx906 agent count (want 3) =="
rocminfo 2>/dev/null | awk '$1=="Name:" && $2=="gfx906"' | wc -l
echo "== /dev/kfd =="
ls -la /dev/kfd 2>&1 | head -1
echo "== D-state process count (phantom load check) =="
ps -eo stat | grep -c '^D' || true
ps -eo pid,stat,wchan:24,comm | awk '$2 ~ /^D/' | head -8
echo "== COMPUTE VET (fires HIP init; telemetry cannot prove this) =="
timeout 90 /root/int4-noquant-wt/build/bin/llama-bench --list-devices 2>&1 | tail -10
echo "VET-RC=$?"
echo "== kmsg tail (SMC signatures; may be unreadable in LXC) =="
dmesg 2>/dev/null | grep -iE "smc|flush_work|amdgpu" | tail -5 || echo "dmesg unreadable from container"
