#!/bin/bash
# Expand "4-18" style layer ranges into a GGML_CUDA_W4A4_SKIP comma prefix list.
# Each layer N becomes the prefix "blk.N." (matches EVERY tensor class in that
# layer: engine matching rule is "tensor name starts with prefix"). Multiple
# ranges may be comma- or space-separated, e.g. "0-2,20-23".
#
# Usage:
#   w4a4_skip_expand.sh 4-18            -> print prefix list to stdout
#   w4a4_skip_expand.sh 4-18 out.txt    -> also write to out.txt
#   w4a4_skip_expand.sh                 -> write the three standard lists:
#        /root/w4a4-skip-early.txt   (L4-18,  the 48b outlier band)
#        /root/w4a4-skip-late.txt    (L19-31, the phase-3 tail hypothesis)
#        /root/w4a4-skip-narrow.txt  (the 27 exact tensor names from
#                                      /root/h32-allowlist-sel.txt, used as
#                                      prefixes — exact names trivially
#                                      prefix-match themselves)
set -euo pipefail

expand_range() {
    local range="$1" out=""
    if [[ "$range" == *-* ]]; then
        local a="${range%%-*}" b="${range##*-}"
        for ((i=a; i<=b; i++)); do
            [ -n "$out" ] && out+=","
            out+="blk.${i}."
        done
    else
        out="blk.${range}."
    fi
    printf '%s' "$out"
}

if [ $# -ge 1 ]; then
    list=""
    for r in ${1//,/ }; do
        [ -n "$list" ] && list+=","
        list+="$(expand_range "$r")"
    done
    printf '%s\n' "$list"
    [ $# -ge 2 ] && printf '%s\n' "$list" > "$2"
    exit 0
fi

# default lists
list=""
for ((i=4; i<=18; i++)); do [ -n "$list" ] && list+=","; list+="blk.${i}."; done
printf '%s\n' "$list" > /root/w4a4-skip-early.txt

list=""
for ((i=19; i<=31; i++)); do [ -n "$list" ] && list+=","; list+="blk.${i}."; done
printf '%s\n' "$list" > /root/w4a4-skip-late.txt

# narrow: the 27 exact tensor names (comma-separated file) reformatted as a
# comma-separated prefix list — exact full names, trivial prefix-match.
tr -d ' \n' < /root/h32-allowlist-sel.txt > /root/w4a4-skip-narrow.txt
echo "wrote /root/w4a4-skip-early.txt ($(wc -c < /root/w4a4-skip-early.txt) B), /root/w4a4-skip-late.txt ($(wc -c < /root/w4a4-skip-late.txt) B), /root/w4a4-skip-narrow.txt ($(wc -c < /root/w4a4-skip-narrow.txt) B)"
