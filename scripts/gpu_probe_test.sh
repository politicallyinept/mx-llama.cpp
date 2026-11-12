#!/bin/bash
# Standalone G3 GPU-probe test: prints exactly what the leg drivers' gate computes.
DEVS=$(python3 -c '
import subprocess,re
o=subprocess.run(["rocm-smi","--showmeminfo","vram"],capture_output=True,text=True).stdout
free=[]
for l in o.splitlines():
    m=re.match(r"GPU\[(\d+)\].*Used Memory \(B\):\s*(\d+)",l)
    if m and int(m.group(2))<2*1024**3: free.append(m.group(1))
print(",".join(free[:2]) if len(free)>=2 else "")')
echo "PROBE-RESULT: DEVS=[$DEVS]"
[ -n "$DEVS" ] && echo PROBE-PASS || echo PROBE-FAIL
