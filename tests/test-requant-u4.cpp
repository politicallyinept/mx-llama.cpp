// test-requant-u4: verify the GGML_LLAMA_REQUANT_U4 load-time converter
// (llama_requant_u4_convert) is bit-identical to a direct dequantize_row_<type>()
// + quantize_row_q4_0_ref() pass over the whole tensor. CPU-only: no model files,
// no GPU required.

#include "llama-model-loader.h"

#include "ggml.h"
#include "ggml-quants.h"

#include <cstdio>
#include <cstring>
#include <vector>

static uint32_t rng_state = 123456789;

static float frand(void) {
    // xorshift32, mapped to [-0.5, 0.5)
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return (float) (rng_state % 1000000u) / 1000000.0f - 0.5f;
}

static int run_case(ggml_type type, int64_t k, int64_t n) {
    const int64_t ne = k * n;

    std::vector<float> f32(ne);
    for (auto & x : f32) {
        x = frand();
    }

    std::vector<uint8_t> src(ggml_row_size(type, k) * n, 0);
    ggml_quantize_init(type);  // codebook tables for the quantize/dequantize references
    switch (type) {
        case GGML_TYPE_IQ3_S:  quantize_iq3_s (f32.data(), src.data(), n, k, nullptr); break;
        case GGML_TYPE_Q2_K:   quantize_q2_K  (f32.data(), src.data(), n, k, nullptr); break;
        case GGML_TYPE_Q4_K:   quantize_q4_K  (f32.data(), src.data(), n, k, nullptr); break;
        case GGML_TYPE_IQ4_XS: quantize_iq4_xs(f32.data(), src.data(), n, k, nullptr); break;
        default:
            printf("FAIL: no quantizer wired for type %s\n", ggml_type_name(type));
            return 1;
    }

    const size_t dst_size = ggml_row_size(GGML_TYPE_Q4_0, ne);
    std::vector<uint8_t> out(dst_size, 0xAA);
    llama_requant_u4_convert(type, src.data(), out.data(), ne, k);

    // reference: whole-tensor dequantize_row_<type> + quantize_row_q4_0_ref
    std::vector<float> ref_f32(ne);
    switch (type) {
        case GGML_TYPE_IQ3_S:  dequantize_row_iq3_s ((const block_iq3_s  *) src.data(), ref_f32.data(), ne); break;
        case GGML_TYPE_Q2_K:   dequantize_row_q2_K  ((const block_q2_K   *) src.data(), ref_f32.data(), ne); break;
        case GGML_TYPE_Q4_K:   dequantize_row_q4_K  ((const block_q4_K   *) src.data(), ref_f32.data(), ne); break;
        case GGML_TYPE_IQ4_XS: dequantize_row_iq4_xs((const block_iq4_xs *) src.data(), ref_f32.data(), ne); break;
        default: GGML_ASSERT(false);
    }
    std::vector<uint8_t> ref(dst_size, 0);
    quantize_row_q4_0_ref(ref_f32.data(), (block_q4_0 *) ref.data(), ne);

    if (memcmp(out.data(), ref.data(), dst_size) != 0) {
        printf("FAIL: %s -> Q4_0 (k=%zu, n=%zu): output differs from dequantize -> quantize_row_q4_0_ref\n",
                ggml_type_name(type), (size_t) k, (size_t) n);
        return 1;
    }

    printf("ok: %s -> Q4_0 bit-identical (k=%zu, n=%zu, %zu -> %zu bytes)\n",
            ggml_type_name(type), (size_t) k, (size_t) n, src.size(), dst_size);
    return 0;
}

int main(void) {
    int rc = 0;

    rc += run_case(GGML_TYPE_IQ3_S,  512, 8);
    rc += run_case(GGML_TYPE_Q2_K,   512, 8);
    rc += run_case(GGML_TYPE_Q4_K,   512, 8);
    rc += run_case(GGML_TYPE_IQ4_XS, 512, 8);
    rc += run_case(GGML_TYPE_IQ3_S,  256, 33);  // k == 256 boundary, odd row count

    if (rc == 0) {
        printf("test-requant-u4: PASS\n");
    } else {
        printf("test-requant-u4: FAIL (%d cases)\n", rc);
    }
    return rc;
}
