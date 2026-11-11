#!/bin/bash
# Task C ISA census: mul_mat_i8narrow_q40<T,NWAVES,KSPLIT,INTERLEAVED> instantiations
# in libggml-hip.so. Bars (Task C GPU-QUEUED): 0 VGPR spills, <=64 VGPR at T<=8,
# dot8 count per hot loop == 8*T expectation.
set -u
SO=/root/int4-noquant-wt/build/bin/libggml-hip.so
OBJ=/opt/rocm/llvm/bin
OUT=/root/int4-noquant-wt/scripts/isa-census
mkdir -p "$OUT"
echo "== symbols =="
$OBJ/llvm-nm -S --defined-only "$SO" 2>/dev/null | grep -i i8narrow | tee "$OUT/symbols.txt" | head -12
echo "== per-instantiation disassembly counts =="
while read -r addr size sym; do
  case "$sym" in *i8narrow*) ;; *) continue ;; esac
  S=$((16#$addr)); E=$((S + 16#$size))
  F="$OUT/$(echo "$sym" | md5sum | cut -c1-8).dis"
  timeout 300 $OBJ/llvm-objdump -d --start-address=$S --stop-address=$E "$SO" > "$F" 2>/dev/null
  D8=$(grep -c "v_dot8" "$F" || true)
  D4=$(grep -c "v_dot4" "$F" || true)
  SP=$(grep -cE "scratch_load|scratch_store" "$F" || true)
  INS=$(grep -cE "^\s+[0-9a-f]+:" "$F" || true)
  echo "$sym | insns=$INS dot8=$D8 dot4=$D4 scratch(spills)=$SP | $F" | tee -a "$OUT/census.txt"
done < "$OUT/symbols.txt"
echo "== done: $(ls "$OUT"/*.dis 2>/dev/null | wc -l) .dis files =="
