#!/bin/bash
# taskR matrix: interleaved borrowed(b) vs pinned(p) legs, 2 reps, per stack/env/depth.
# Refuse-if-exists is enforced inside taskR-drive.sh (SKIP-EXISTS on .done).
D=/root/taskR/taskR-drive.sh
P=8810
run() { bash "$D" leg "$1" "$2" "$3" "$4" "$P"; P=$((P+1)); }

# ---- Q4_0 stack, f0q4 (best d4 56.04 / d7 44.30) ----
run f0q4 b 4 1; run f0q4 p 4 1; run f0q4 b 7 1; run f0q4 p 7 1
run f0q4 b 4 2; run f0q4 p 4 2; run f0q4 b 7 2; run f0q4 p 7 2
# ---- Q4_0 stack, f128q4w4 (quality-neutral d4 54.30 / d7 33.96) ----
run f128q4w4 b 4 1; run f128q4w4 p 4 1; run f128q4w4 b 7 1; run f128q4w4 p 7 1
run f128q4w4 b 4 2; run f128q4w4 p 4 2; run f128q4w4 b 7 2; run f128q4w4 p 7 2
# ---- Q8_0 stack, q8lds (taskL q4d d4 50.44 / d7 40.61) ----
run q8lds b 4 1; run q8lds p 4 1; run q8lds b 7 1; run q8lds p 7 1
run q8lds b 4 2; run q8lds p 4 2; run q8lds b 7 2; run q8lds p 7 2
echo "MATRIX-ALL-DONE $(date -Is)"
