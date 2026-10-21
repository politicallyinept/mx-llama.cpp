// G3 debug harness: run mmq_gemm_repacked<false,2,4,Q4_0,true> on a small
// problem and compare against a CPU reference.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <hip/hip_runtime.h>

#include "repack-kernels.cuh"

// host quantizer fp32 -> block_q8_1_u4 (mirror of quantize_row_q8_1_u4)
static void host_quant_u4(const float * x, block_q8_1_u4 * y, int64_t k) {
    for (int64_t b = 0; b < k/32; b++) {
        const float * xb = x + b*32;
        float mn = xb[0], mx = xb[0];
        for (int i = 1; i < 32; i++) { mn = fminf(mn, xb[i]); mx = fmaxf(mx, xb[i]); }
        float d4 = (mx - mn) / 15.0f;
        float id = d4 != 0 ? 1.0f/d4 : 0.0f;
        // de-interleaved: byte 2t = {val 2t low, val 2t+16 high}, byte 2t+1 = {val 2t+1, val 2t+17}
        for (int t = 0; t < 8; t++) {
            int q_l0 = (int)roundf((xb[2*t]   - mn)*id); q_l0 = q_l0<0?0:(q_l0>15?15:q_l0);
            int q_l1 = (int)roundf((xb[2*t+1] - mn)*id); q_l1 = q_l1<0?0:(q_l1>15?15:q_l1);
            int q_h0 = (int)roundf((xb[2*t+16] - mn)*id); q_h0 = q_h0<0?0:(q_h0>15?15:q_h0);
            int q_h1 = (int)roundf((xb[2*t+17] - mn)*id); q_h1 = q_h1<0?0:(q_h1>15?15:q_h1);
            uint32_t packed = ((q_l0-8)&0xF) | (((q_h0-8)&0xF)<<4) | (((q_l1-8)&0xF)<<8) | (((q_h1-8)&0xF)<<12);
            memcpy(y[b].qs + 2*t, &packed, 2);
        }
        y[b].d4 = (ggml_half)d4;
        y[b].m  = (ggml_half)mn;
    }
}

// host repack Q4_0 weights into the repacked plane layout:
// qs rows de-aliased: row r, byte j = canonical qs byte j of block j (16B per 32 vals)
// d plane: f16 per 32-block after the qs plane.
// canonical Q4_0 block: {f16 d; uint8 qs[16]} with nibble pairs (v0,v16),(v1,v17)...
static void host_repack_q40(const float * w, int ne0, int ne1, uint8_t * out) {
    const int nsub = ne0/32;
    size_t rs = (size_t)ne0/2; if (rs % 128 == 0) rs += 16;  // repack_qs_stride
    for (int r = 0; r < ne1; r++) {
        for (int sb = 0; sb < nsub; sb++) {
            const float * wb = w + (size_t)r*ne0 + sb*32;
            float mn = wb[0], mx = wb[0];
            for (int i = 1; i < 32; i++) { mn = fminf(mn, wb[i]); mx = fmaxf(mx, wb[i]); }
            float d = (mx - mn) / 15.0f;
            float id = d != 0 ? 1.0f/d : 0.0f;
            // canonical Q4_0 nibble order: byte j = {val j low, val j+16 high}
            uint8_t * qrow = out + (size_t)r*rs + sb*16;
            for (int j = 0; j < 16; j++) {
                int lo = (int)roundf((wb[j]    - mn)*id); lo = lo<0?0:(lo>15?15:lo);
                int hi = (int)roundf((wb[j+16] - mn)*id); hi = hi<0?0:(hi>15?15:hi);
                qrow[j] = (uint8_t)(lo | (hi << 4));
            }
            uint16_t dh = __half_as_ushort((ggml_half)d);
            memcpy(out + (size_t)ne1*rs + ((size_t)r*nsub + sb)*2, &dh, 2);
        }
    }
}

int main(int argc, char ** argv) {
    int ne00 = 128, ne01 = 64, ntok = 64;
    if (argc >= 4) { ne00 = atoi(argv[1]); ne01 = atoi(argv[2]); ntok = atoi(argv[3]); }
    const int nkb = ne00/32;

    float * w = (float*)malloc((size_t)ne01*ne00*4);
    float * x = (float*)malloc((size_t)ntok*ne00*4);
    srand(42);
    for (size_t i = 0; i < (size_t)ne01*ne00; i++) w[i] = (rand()%2000-1000)/500.0f;
    for (size_t i = 0; i < (size_t)ntok*ne00; i++) x[i] = (rand()%2000-1000)/500.0f;

    // repacked weights
    size_t rs = ne00/2; if (rs % 128 == 0) rs += 16;  // repack_qs_stride padding
    uint8_t * wq = (uint8_t*)malloc((size_t)ne01*rs + (size_t)ne01*nkb*2);
    host_repack_q40(w, ne00, ne01, wq);

    // u4 activations, token-major: pool[tok*nkb + sb]
    block_q8_1_u4 * xq = (block_q8_1_u4*)malloc((size_t)ntok*nkb*sizeof(block_q8_1_u4));
    for (int t = 0; t < ntok; t++) host_quant_u4(x + (size_t)t*ne00, xq + (size_t)t*nkb, ne00);

    // CPU reference using the QUANTIZED values (what the kernel should compute)
    // Variants to identify which term the kernel gets wrong:
    //   ref  = d*(d4*(idot + 8*sw) + m*sw)          (sealed formula)
    //   vA   = d*(d4*idot + m*sw)                   (missing 8*d4*sw)
    //   vB   = d*(d4*idot + 8*d4*sw + m*sw)         (same as ref, assoc diff)
    //   vC   = d*d4*idot                             (no affine terms)
    //   vD   = d*(d4*idot + m*sw + 8*sw)            (8*sw not scaled by d4)
    float * ref = (float*)malloc((size_t)ntok*ne01*4);
    float * vA = (float*)malloc((size_t)ntok*ne01*4);
    float * vC = (float*)malloc((size_t)ntok*ne01*4);
    float * vD = (float*)malloc((size_t)ntok*ne01*4);
    for (int t = 0; t < ntok; t++) {
        for (int r = 0; r < ne01; r++) {
            double acc = 0, aA = 0, aC = 0, aD = 0;
            for (int sb = 0; sb < nkb; sb++) {
                const block_q8_1_u4 * xb = &xq[t*nkb + sb];
                float d4 = __half2float(xb->d4), m = __half2float(xb->m);
                const uint8_t * qrow = wq + (size_t)r*rs + sb*16;
                uint16_t dh; memcpy(&dh, wq + (size_t)ne01*rs + ((size_t)r*nkb + sb)*2, 2);
                float d = __half2float(*(const ggml_half*)&dh);
                for (int j = 0; j < 16; j++) {
                    uint8_t ab = xb->qs[j];
                    int a0 = (int)(signed char)((ab & 0xF) << 4) >> 4;
                    int a1 = (int)(signed char)(ab & 0xF0) >> 4;
                    int v0 = (qrow[j] & 0xF) - 8;
                    int v1 = ((qrow[j] >> 4) & 0xF) - 8;
                    double idot = (double)(v0*a0 + v1*a1);
                    double sw = (double)(v0 + v1);
                    acc += (double)d * ((double)d4 * (idot + 8.0*sw) + (double)m * sw);
                    aA  += (double)d * ((double)d4 * idot + (double)m * sw);
                    aC  += (double)d * ((double)d4 * idot);
                    aD  += (double)d * ((double)d4 * idot + (double)m * sw + 8.0*sw);
                }
            }
            ref[t*ne01 + r] = (float)acc;
            vA[t*ne01 + r] = (float)aA;
            vC[t*ne01 + r] = (float)aC;
            vD[t*ne01 + r] = (float)aD;
        }
    }

    // device
    uint8_t * dw; block_q8_1_u4 * dxq; float * dy;
    hipMalloc(&dw, (size_t)ne01*rs + (size_t)ne01*nkb*2);
    hipMalloc(&dxq, (size_t)ntok*nkb*sizeof(block_q8_1_u4));
    hipMalloc(&dy, (size_t)ntok*ne01*4);
    hipMemcpy(dw, wq, (size_t)ne01*rs + (size_t)ne01*nkb*2, hipMemcpyHostToDevice);
    hipMemcpy(dxq, xq, (size_t)ntok*nkb*sizeof(block_q8_1_u4), hipMemcpyHostToDevice);
    hipMemset(dy, 0xFF, (size_t)ntok*ne01*4);

    const int nrl = MMQ_RP_Q8_NROW_LANES;
    const int mmq_bm = MMQ_RP_Q8_BM;
    const int bn = 64 * MMQ_RP_Q8_TN;
    const dim3 grid((ne01 + mmq_bm - 1) / mmq_bm, (ntok + bn - 1) / bn, 1);
    if (argc >= 5 && atoi(argv[4]) == 32) {
        const int bn = 32;
        const dim3 g((ne01 + mmq_bm - 1) / mmq_bm, (ntok + bn - 1) / bn, 1);
        mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_Q4_0, true>
            <<<g, dim3(32, nrl*2), 0, 0>>>(
                dw, (const block_q8_1*)dxq, dy, (uint32_t)ne00, (uint32_t)ne01, (uint32_t)ntok,
                nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t)ne01);
    } else {
        mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_Q4_0, true>
            <<<grid, dim3(64, nrl), 0, 0>>>(
                dw, (const block_q8_1*)dxq, dy, (uint32_t)ne00, (uint32_t)ne01, (uint32_t)ntok,
                nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t)ne01);
    }
    hipError_t e = hipDeviceSynchronize();
    printf("kernel rc=%d (%s)\n", (int)e, hipGetErrorString(e));

    float * out = (float*)malloc((size_t)ntok*ne01*4);
    hipMemcpy(out, dy, (size_t)ntok*ne01*4, hipMemcpyDeviceToHost);

    int bad = 0; double maxrel = 0;
    int matchA = 0, matchC = 0, matchD = 0;
    for (int t = 0; t < ntok; t++) for (int r = 0; r < ne01; r++) {
        float got = out[t*ne01 + r], want = ref[t*ne01 + r];
        double rel = fabs(got - want) / fmax(1e-3, fabs(want));
        if (rel > 0.02) { bad++; if (bad < 6) printf("MISMATCH t=%d r=%d got=%.4f want=%.4f vA=%.4f vC=%.4f vD=%.4f\n", t, r, got, want, vA[t*ne01+r], vC[t*ne01+r], vD[t*ne01+r]); }
        if (rel > maxrel) maxrel = rel;
        if (fabs(got - vA[t*ne01+r]) / fmax(1e-3,fabs(vA[t*ne01+r])) < 0.02) matchA++;
        if (fabs(got - vC[t*ne01+r]) / fmax(1e-3,fabs(vC[t*ne01+r])) < 0.02) matchC++;
        if (fabs(got - vD[t*ne01+r]) / fmax(1e-3,fabs(vD[t*ne01+r])) < 0.02) matchD++;
    }
    printf("BAD=%d / %d  maxrel=%.4f  matchA(missing 8d4sw)=%d matchC(no affine)=%d matchD(8sw unscaled)=%d\n", bad, ntok*ne01, maxrel, matchA, matchC, matchD);
    return bad ? 1 : 0;
}
