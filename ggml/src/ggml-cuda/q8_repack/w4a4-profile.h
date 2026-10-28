// GGML_CUDA_W4A4_PROFILE: one top-line switch over the int4-path knobs.
//
//   speed    = DOT8=3, MIN_TOKENS=128, KSPLIT=1, SUMW=1   (the audited F config)
//   balanced = DOT8=3, MIN_TOKENS=128, KSPLIT unset, SUMW unset
//   off|none = inject NOTHING: every knob behaves exactly as the pre-profile build
//
// P5 DEFAULT FLIP: when GGML_CUDA_W4A4_PROFILE is UNSET the resolution is
// PROFILE=speed (the F config sealed on 2026-10-26: DOT8=3 per-16, MIN_TOKENS=128,
// KSPLIT=1, SUMW=1). DOT8 stays at 3: DOT8=4 was REJECTED by the seal (greedy
// 24.6% identical, KLD decode RED, PPL +0.92%) — do not adopt it. DOT8=1 remains
// available as an explicit per-knob override. Set GGML_CUDA_W4A4_PROFILE=off (or
// none) to restore the old default-off behavior explicitly.
//
// KILL-SWITCH: GGML_CUDA_W4A4_OFF=1 disables the WHOLE W4A4 gate stack
// (DOT8, MIN_TOKENS, KSPLIT, SUMW all forced to 0) regardless of every other
// W4A4 env var, including explicit per-knob settings and PROFILE=. The repack
// gates (GGML_CUDA_REPACK_Q4_0 etc.) are NOT W4A4 knobs and stay as set — with
// the kill-switch on the process behaves exactly like the stock-repack build.
//
// RESOLUTION ORDER per knob: explicit per-knob env > profile default > built-in
// default (0 = today's off). PROFILE=speed + DOT8=1 -> per-32 wins (explicit
// dial beats the preset); PROFILE=off + DOT8=1 -> gate on (off disables
// nothing). Unknown values WARN and behave as off — never guess.
//
// SCOPE-LIMITED: the profile injects defaults for the int4-path knobs ONLY
// (DOT8, MIN_TOKENS, KSPLIT, SUMW). REPACK_Q4_0, W4A4_ONLY, RP_GENERIC_MMV,
// W4A4_SKIP and every debug gate are NEVER touched by the profile — no coupling
// in either direction.
//
// KILL-SWITCH-OFF PROPERTY: GGML_CUDA_W4A4_OFF=1 -> every resolved value is 0
// and the u4 paths are silent at every gate site, byte-identical to a
// pre-W4A4 stock-repack run.
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
    const char * dot8_src;        // "explicit" | "PROFILE=speed" | "PROFILE=speed default" | "default" | "kill-switch"
    const char * min_tokens_src;
    const char * ksplit_src;
    const char * sumw_src;
    const char * name;     // "speed" | "balanced" | "off"
    bool         defaulted; // true when PROFILE was UNSET and speed was applied
    bool         off;       // true when the GGML_CUDA_W4A4_OFF kill-switch fired
};

inline const w4a4_profile_knobs & w4a4_profile() {
    static const w4a4_profile_knobs k = [] {
        w4a4_profile_knobs r{};
        r.name = "off";
        // KILL-SWITCH first: it outranks every other W4A4 env var, explicit
        // knobs included. All knobs stay zero-initialized -> every gate site
        // sees the pre-W4A4 off state.
        const char * kill = getenv("GGML_CUDA_W4A4_OFF");
        if (kill != nullptr && atoi(kill) != 0) {
            r.off = true;
            r.dot8_src = r.min_tokens_src = r.ksplit_src = r.sumw_src = "kill-switch";
            GGML_LOG_WARN("CUDA w4a4: GGML_CUDA_W4A4_OFF='%s' KILL-SWITCH ACTIVE: whole W4A4 gate stack forced off (DOT8=0 MIN_TOKENS=0 KSPLIT=0 SUMW=0) regardless of every other W4A4 env var\n", kill);
            return r;
        }
        bool speed = false, balanced = false;
        const char * p = getenv("GGML_CUDA_W4A4_PROFILE");
        if (p != nullptr) {
            if (strcasecmp(p, "speed") == 0)         { r.name = "speed";    speed = true; }
            else if (strcasecmp(p, "balanced") == 0) { r.name = "balanced"; balanced = true; }
            else if (strcasecmp(p, "off") == 0 || strcasecmp(p, "none") == 0) { r.name = "off"; }
            else {
                GGML_LOG_WARN("CUDA w4a4: GGML_CUDA_W4A4_PROFILE='%s' is not one of speed|balanced|off|none: treating as off\n", p);
            }
        } else {
            // P5: unset resolves to the audited F config (speed).
            r.name = "speed"; speed = true; r.defaulted = true;
        }
        // explicit env > profile preset > built-in default (0). has_preset is
        // false for the knobs balanced leaves unset (KSPLIT, SUMW) and for off.
        const char * preset_src = r.defaulted ? "PROFILE=speed default"
                                              : (speed ? "PROFILE=speed" : "PROFILE=balanced");
        auto resolve = [&](const char * env, long long preset_val, bool has_preset,
                           long long * out, const char ** src) {
            const char * e = getenv(env);
            if (e != nullptr)      { *out = atoll(e);          *src = "explicit"; }
            else if (has_preset)   { *out = preset_val;        *src = preset_src; }
            else                   { *out = 0;                 *src = "default"; }
        };
        // DOT8 preset is 3 for BOTH profiles (F config; DOT8=4 REJECTED by the
        // 2026-10-26 seal — do not raise the preset).
        resolve("GGML_CUDA_W4A4_DOT8", 3, speed || balanced, &r.dot8, &r.dot8_src);
        resolve("GGML_CUDA_W4A4_MIN_TOKENS", 128, speed || balanced, &r.min_tokens,  &r.min_tokens_src);
        resolve("GGML_RP_KSPLIT",           1,   speed,             &r.ksplit,       &r.ksplit_src);
        resolve("GGML_CUDA_W4A4_SUMW",      1,   speed,             &r.sumw,         &r.sumw_src);
        if (speed || balanced) {
            GGML_LOG_WARN("CUDA w4a4: GGML_CUDA_W4A4_PROFILE=%s%s resolved: DOT8=%d (%s) MIN_TOKENS=%lld (%s) KSPLIT=%d (%s) SUMW=%d (%s)\n",
                          r.name, r.defaulted ? " (unset -> default)" : "",
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
