// GGML_CUDA_W4A4_SKIP: env-gated per-TENSOR-PREFIX exclusion from the u4act
// (W4A4) activation path — Lever 2 (layer-selective W4A8). Excluded GEMMs run
// the STOCK q8_1 activation kernel: bitwise-stock for those GEMMs, zero new
// numerics (this is dispatch SELECTION, not a new quantizer).
//
// Value: comma-separated tensor-NAME prefixes, e.g. "blk.4.,blk.5.,...,blk.18."
// (all tensor classes in layers 4-18) or exact full tensor names (a full name
// trivially prefix-matches itself). Matching rule: name starts with prefix.
// Unset/empty = EXACTLY today's dispatch (empty vector => check is false).
// Composes with GGML_W4A4_ONLY: ONLY applies first, then SKIP subtracts.
//
// Static-once parse (b9476ae9f pattern); ONE startup log line listing the
// active exclusion count = auditable gate-active marker.
#pragma once

#include "common.cuh"

#include <cstring>
#include <string>
#include <vector>

inline const std::vector<std::string> & ggml_cuda_w4a4_skip_prefixes() {
    static const std::vector<std::string> prefixes = [] {
        std::vector<std::string> v;
        const char * e = getenv("GGML_CUDA_W4A4_SKIP");
        if (e != nullptr) {
            const char * p = e;
            while (*p) {
                const char * q = strchr(p, ',');
                if (q == nullptr) q = p + strlen(p);
                if (q > p) v.emplace_back(p, q);
                p = (*q == '\0') ? q : q + 1;
            }
        }
        if (!v.empty()) {
            GGML_LOG_WARN("CUDA w4a4: GGML_CUDA_W4A4_SKIP gate ACTIVE: %zu tensor-name prefix(es) excluded from u4act (stock q8_1 activations)\n",
                          v.size());
        }
        return v;
    }();
    return prefixes;
}

inline bool ggml_cuda_w4a4_skip(const char * name) {
    const std::vector<std::string> & prefixes = ggml_cuda_w4a4_skip_prefixes();
    if (prefixes.empty() || name == nullptr) {
        return false;
    }
    for (const std::string & p : prefixes) {
        if (strncmp(name, p.c_str(), p.size()) == 0) {
            return true;
        }
    }
    return false;
}
