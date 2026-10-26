// GGML_CUDA_W4A4_PROFILE: one top-line switch over the int4-path knobs.
//
//   speed    = DOT8=1, MIN_TOKENS=128, KSPLIT=1, SUMW=1
//   balanced = DOT8=3, MIN_TOKENS=128, KSPLIT unset, SUMW unset
//   off      = inject NOTHING (default): every knob behaves exactly as before
//
// RESOLUTION ORDER per knob: explicit per-knob env > profile default > built-in
// default (0 = today's off). PROFILE=speed + DOT8=3 -> per-16 wins (explicit
// dial beats the preset); PROFILE=off + DOT8=1 -> gate on (off disables
// nothing). Unknown values WARN and behave as off — never guess.
//
// SCOPE-LIMITED: the profile injects defaults for the int4-path knobs ONLY
// (DOT8, MIN_TOKENS, KSPLIT, SUMW). REPACK_Q4_0, W4A4_ONLY, RP_GENERIC_MMV,
// W4A4_SKIP and every debug gate are NEVER touched by the profile — no coupling
// in either direction.
//
// DEFAULT-OFF PROPERTY: no profile + no explicit knobs -> every resolved value
// is its built-in default (0) and NO warning is printed: byte-identical to the
// pre-profile behavior at every gate site.
//
// Resolved ONCE per process: inline function with a function-local static —
// one instance across TUs (mul-mat.cu, mul-mat-id.cu, repack-common.cu all see
// the same values), so the profile summary WARN fires exactly once.
//
// Deliberately free of ggml/HIP includes so the CPU seal (seals/build_seal.sh)
// can include it after its own GGML_LOG_WARN macro definition.
#pragma once

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <strings.h>
#ifdef _MSC_VER
#define strcasecmp _stricmp
#endif

#ifndef GGML_LOG_WARN
#define GGML_LOG_WARN(...) fprintf(stderr, __VA_ARGS__)
#endif

struct w4a4_profile_knobs {
    long long dot8;        // GGML_CUDA_W4A4_DOT8
    long long min_tokens;  // GGML_CUDA_W4A4_MIN_TOKENS
    long long ksplit;      // GGML_RP_KSPLIT
    long long sumw;        // GGML_CUDA_W4A4_SUMW (0/1)
    const char * dot8_src;        // "explicit" | "PROFILE=speed" | "default"
    const char * min_tokens_src;
    const char * ksplit_src;
    const char * sumw_src;
    const char * name;     // "speed" | "balanced" | "off"
};

inline const w4a4_profile_knobs & w4a4_profile() {
    static const w4a4_profile_knobs k = [] {
        w4a4_profile_knobs r{};
        r.name = "off";
        bool speed = false, balanced = false;
        const char * p = getenv("GGML_CUDA_W4A4_PROFILE");
        if (p != nullptr) {
            if (strcasecmp(p, "speed") == 0)         { r.name = "speed";    speed = true; }
            else if (strcasecmp(p, "balanced") == 0) { r.name = "balanced"; balanced = true; }
            else if (strcasecmp(p, "off") == 0)      { r.name = "off"; }
            else {
                GGML_LOG_WARN("CUDA w4a4: GGML_CUDA_W4A4_PROFILE='%s' is not one of speed|balanced|off: treating as off\n", p);
            }
        }
        // explicit env > profile preset > built-in default (0). has_preset is
        // false for the knobs balanced leaves unset (KSPLIT, SUMW) and for off.
        auto resolve = [&](const char * env, long long preset_val, bool has_preset,
                           long long * out, const char ** src) {
            const char * e = getenv(env);
            if (e != nullptr)      { *out = atoll(e);          *src = "explicit"; }
            else if (has_preset)   { *out = preset_val;        *src = speed ? "PROFILE=speed" : "PROFILE=balanced"; }
            else                   { *out = 0;                 *src = "default"; }
        };
        resolve("GGML_CUDA_W4A4_DOT8", speed ? 1 : 3, speed || balanced, &r.dot8, &r.dot8_src);
        resolve("GGML_CUDA_W4A4_MIN_TOKENS", 128, speed || balanced, &r.min_tokens,  &r.min_tokens_src);
        resolve("GGML_RP_KSPLIT",           1,   speed,             &r.ksplit,       &r.ksplit_src);
        resolve("GGML_CUDA_W4A4_SUMW",      1,   speed,             &r.sumw,         &r.sumw_src);
        if (speed || balanced) {
            GGML_LOG_WARN("CUDA w4a4: GGML_CUDA_W4A4_PROFILE=%s resolved: DOT8=%d (%s) MIN_TOKENS=%lld (%s) KSPLIT=%d (%s) SUMW=%d (%s)\n",
                          r.name,
                          (int) r.dot8, r.dot8_src, (long long) r.min_tokens, r.min_tokens_src,
                          (int) r.ksplit, r.ksplit_src, (int) r.sumw, r.sumw_src);
        }
        return r;
    }();
    return k;
}

// Eager process-start resolution: forces the summary/unknown WARN to fire
// exactly once at library load, even when no gate site ever runs (unknown ->
// off injects nothing, so a lazy read would print nothing). Function-local
// static => the resolution happens exactly once no matter which TU initializes
// first; the other TUs' initializers see it done.
namespace w4a4_profile_detail {
struct w4a4_profile_init {
    w4a4_profile_init() { w4a4_profile(); }
};
} // namespace w4a4_profile_detail
