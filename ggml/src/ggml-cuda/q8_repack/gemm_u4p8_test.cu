// C1/B8 M4 debug harness: run mmq_gemm_repacked<false,2,4,Q4_0,false,true>
// (per-8 u4p GEMM) on a small problem and compare against a CPU reference
// built from the QUANTIZED values with the sealed form-1 per-8 algebra.
// Mirrors gemm_u4_test.cu (G3) with the u4p host quantizer matching
// quantize_mmq_q8_1_u4p in quantize.cu byte-for-byte.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <hip/hip_runtime.h>

#include "repack-kernels.cuh"

// host quantizer fp32 -> block_q8_1_u4p (mirror of quantize_mmq_q8_1_u4p):
// per-8 sub-block k = t>>1 over threads t=0..7; thread t covers values
// {2t, 2t+1, 2t+16, 2t+17} of the 32-value block; the sub-block spans the
// 8 values of threads 2k and 2k+1. Canonical byte order:
//   byte 2t   = {val 2t low,  val 2t+16 high}
//   byte 2t+1 = {val 2t+1 low, val 2t+17 high}
// SIGNED nibbles (q-8)&0xF. d4[k] = (max-min)/15, m[k] = min.
static void host_quant_u4p(const float * x, block_q8_1_u4p * y, int64_t k) {
    for (int64_t b = 0; b < k/32; b++) {
        const float * xb = x + b*32;
        float mn[4], mx[4];
        for (int kk = 0; kk < 4; kk++) {
            // sub-block kk = values {4kk..4kk+3} and {16+4kk..16+4kk+3}
            mn[kk] = mx[kk] = xb[4*kk];
            for (int i = 1; i < 4; i++) {
                mn[kk] = fminf(mn[kk], xb[4*kk+i]);
                mx[kk] = fmaxf(mx[kk], xb[4*kk+i]);
                mn[kk] = fminf(mn[kk], xb[16+4*kk+i]);
                mx[kk] = fmaxf(mx[kk], xb[16+4*kk+i]);
            }
        }
        for (int t = 0; t < 8; t++) {
            const int kk = t >> 1;
            const float d4 = (mx[kk] - mn[kk]) / 15.0f;
            const float id = d4 != 0 ? 1.0f/d4 : 0.0f;
            int q_l0 = (int)roundf((xb[2*t]   - mn[kk])*id); q_l0 = q_l0<0?0:(q_l0>15?15:q_l0);
            int q_l1 = (int)roundf((xb[2*t+1] - mn[kk])*id); q_l1 = q_l1<0?0:(q_l1>15?15:q_l1);
            int q_h0 = (int)roundf((xb[2*t+16] - mn[kk])*id); q_h0 = q_h0<0?0:(q_h0>15?15:q_h0);
            int q_h1 = (int)roundf((xb[2*t+17] - mn[kk])*id); q_h1 = q_h1<0?0:(q_h1>15?15:q_h1);
            uint32_t packed = ((q_l0-8)&0xF) | (((q_h0-8)&0xF)<<4) | (((q_l1-8)&0xF)<<8) | (((q_h1-8)&0xF)<<12);
            memcpy(y[b].qs + 2*t, &packed, 2);
            if ((t & 1) == 0) {
                y[b].d4[kk] = (ggml_half)d4;
                y[b].m[kk]  = (ggml_half)mn[kk];
            }
        }
    }
}

// host repack Q4_0 weights into the repacked plane layout (same as G3 harness):
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

    size_t rs = ne00/2; if (rs % 128 == 0) rs += 16;
    uint8_t * wq = (uint8_t*)malloc((size_t)ne01*rs + (size_t)ne01*nkb*2);
    host_repack_q40(w, ne00, ne01, wq);

    block_q8_1_u4p * xq = (block_q8_1_u4p*)malloc((size_t)ntok*nkb*sizeof(block_q8_1_u4p));
    for (int t = 0; t < ntok; t++) host_quant_u4p(x + (size_t)t*ne00, xq + (size_t)t*nkb, ne00);

    // CPU reference with the sealed per-8 form-1 algebra:
    //   acc += d * ( d4_k * (idot_k + 8*SW_k) + m_k * SW_k )
    // per word k of the 32-value block: SW_k = sum(raw weight codes in word k) - 64,
    // idot_k = sum over the 8 pairs of (w_tc_i * a_i), w_tc = code-8,
    // a_i = the activation's SIGNED nibble in the same byte position.
    float * ref = (float*)malloc((size_t)ntok*ne01*4);
    for (int t = 0; t < ntok; t++) {
        for (int r = 0; r < ne01; r++) {
            double acc = 0;
            for (int sb = 0; sb < nkb; sb++) {
                const block_q8_1_u4p * xb = &xq[t*nkb + sb];
                const uint8_t * qrow = wq + (size_t)r*rs + sb*16;
                uint16_t dh; memcpy(&dh, wq + (size_t)ne01*rs + ((size_t)r*nkb + sb)*2, 2);
                float d = __half2float(*(const ggml_half*)&dh);
                const uint32_t * w32 = (const uint32_t *) qrow;
                const uint32_t * a32 = (const uint32_t *) xb->qs;
                for (int kk = 0; kk < 4; kk++) {
                    const uint32_t wraw = w32[kk], a = a32[kk];
                    int sw = 0, idot = 0;
                    for (int j = 0; j < 8; j++) {
                        const int wcode = (wraw >> (4*j)) & 0xF;
                        const int wtc   = wcode - 8;
                        int aq = (a >> (4*j)) & 0xF;
                        const int asigned = (aq < 8) ? aq : aq - 16;  // two's-complement nibble
                        sw   += wcode;
                        idot += wtc * asigned;
                    }
                    const double SW = (double)(sw - 64);
                    const double d4 = __half2float(xb->d4[kk]);
                    const double m  = __half2float(xb->m[kk]);
                    acc += (double)d * (d4 * ((double)idot + 8.0*SW) + m * SW);
                }
            }
            ref[t*ne01 + r] = (float)acc;
        }
    }

    // device
    uint8_t * dw; block_q8_1_u4p * dxq; float * dy;
    hipMalloc(&dw, (size_t)ne01*rs + (size_t)ne01*nkb*2);
    hipMalloc(&dxq, (size_t)ntok*nkb*sizeof(block_q8_1_u4p));
    hipMalloc(&dy, (size_t)ntok*ne01*4);
    hipMemcpy(dw, wq, (size_t)ne01*rs + (size_t)ne01*nkb*2, hipMemcpyHostToDevice);
    hipMemcpy(dxq, xq, (size_t)ntok*nkb*sizeof(block_q8_1_u4p), hipMemcpyHostToDevice);
    hipMemset(dy, 0xFF, (size_t)ntok*ne01*4);

    const int nrl = MMQ_RP_Q8_NROW_LANES;
    const int mmq_bm = MMQ_RP_Q8_BM;
    if (argc >= 5 && atoi(argv[4]) == 32) {
        const int bn = 32;
        const dim3 g((ne01 + mmq_bm - 1) / mmq_bm, (ntok + bn - 1) / bn, 1);
        mmq_gemm_repacked_w32<false, 1, nrl*2, GGML_TYPE_Q4_0, false, true>
            <<<g, dim3(32, nrl*2), 0, 0>>>(
                dw, (const block_q8_1*)dxq, dy, (uint32_t)ne00, (uint32_t)ne01, (uint32_t)ntok,
                nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t)ne01);
    } else {
        const int bn = 64 * MMQ_RP_Q8_TN;
        const dim3 grid((ne01 + mmq_bm - 1) / mmq_bm, (ntok + bn - 1) / bn, 1);
        mmq_gemm_repacked<false, MMQ_RP_Q8_TN, nrl, GGML_TYPE_Q4_0, false, true>
            <<<grid, dim3(64, nrl), 0, 0>>>(
                dw, (const block_q8_1*)dxq, dy, (uint32_t)ne00, (uint32_t)ne01, (uint32_t)ntok,
                nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t)ne01);
    }
    hipError_t e = hipDeviceSynchronize();
    printf("kernel rc=%d (%s)\n", (int)e, hipGetErrorString(e));

    float * out = (float*)malloc((size_t)ntok*ne01*4);
    hipMemcpy(out, dy, (size_t)ntok*ne01*4, hipMemcpyDeviceToHost);

    int bad = 0; double maxrel = 0;
    for (int t = 0; t < ntok; t++) for (int r = 0; r < ne01; r++) {
        float got = out[t*ne01 + r], want = ref[t*ne01 + r];
        double rel = fabs(got - want) / fmax(1.0, fabs(want));
        if (rel > 0.02) { bad++; if (bad < 6) printf("MISMATCH t=%d r=%d got=%.4f want=%.4f\n", t, r, got, want); }
        if (rel > maxrel) maxrel = rel;
    }
    printf("cells=%d bad=%d maxrel=%.3e -> %s\n", ntok*ne01, bad, maxrel, bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
