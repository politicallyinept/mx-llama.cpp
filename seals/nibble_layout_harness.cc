// Settle the block_q8_1_u4 comment contradiction: run BOTH documented packers
// on the same input and compare pairing + signedness of the nibble stream.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>

// EXACT copies of the two in-tree packers (q0|q1<<4 form):
static void pack_host_ref(const float* xb, float mn, float id, uint8_t* q) {
    for (int i = 0; i < 16; i++) {
        int q0 = (int)roundf((xb[i] - mn)*id);     q0 = q0<0?0:(q0>15?15:q0);
        int q1 = (int)roundf((xb[i+16] - mn)*id);  q1 = q1<0?0:(q1>15?15:q1);
        q[i] = (uint8_t)(q0 | (q1 << 4));
    }
}
static void pack_device_mirror(const float* xb, float mn, float id, uint8_t* q) {
    for (int t = 0; t < 8; t++) {
        int a = (int)roundf((xb[2*t] - mn)*id),     b = (int)roundf((xb[2*t+1] - mn)*id);
        int c = (int)roundf((xb[2*t+16] - mn)*id),  d = (int)roundf((xb[2*t+17] - mn)*id);
        a=a<0?0:(a>15?15:a); b=b<0?0:(b>15?15:b); c=c<0?0:(c>15?15:c); d=d<0?0:(d>15?15:d);
        uint32_t p = ((a-8)&0xF) | (((c-8)&0xF)<<4) | (((b-8)&0xF)<<8) | (((d-8)&0xF)<<12);
        memcpy(q + 2*t, &p, 2);
    }
}
static int nib(const uint8_t* q, int byte, int hi) { return hi ? (q[byte]>>4)&0xF : q[byte]&0xF; }

int main() {
    float xb[32]; float mn = -1.7f, mx = 2.9f;   // distinct non-round values
    float id = 15.0f/(mx-mn);
    for (int i = 0; i < 32; i++) xb[i] = mn + (mx-mn)*(i*7 % 32)/31.0f;
    uint8_t ref[16], dev[16];
    pack_host_ref(xb, mn, id, ref);
    pack_device_mirror(xb, mn, id, dev);
    auto quant = [&](float v){ int q=(int)roundf((v-mn)*id); return q<0?0:(q>15?15:q); };
    int pairing_ok = 1, signed_ok = 1;
    for (int j = 0; j < 16; j++) {
        int e_lo = quant(xb[j]), e_hi = quant(xb[j+16]);        // expected: byte j = {q(j), q(j+16)}
        int r_lo = nib(ref,j,0), r_hi = nib(ref,j,1);
        if (r_lo != e_lo || r_hi != e_hi) pairing_ok = 0;       // host ref: raw q, {j, j+16}
        int d_lo = nib(dev,j,0), d_hi = nib(dev,j,1);
        int s_lo = (e_lo - 8) & 0xF, s_hi = (e_hi - 8) & 0xF;
        if (d_lo != s_lo || d_hi != s_hi) signed_ok = 0;        // device: signed form of ref
    }
    printf("host_ref pairing byte j = {j, j+16}: %s\n", pairing_ok ? "CONFIRMED" : "BROKEN");
    printf("device nibbles == (host_ref - 8) & 0xF per nibble: %s\n", signed_ok ? "CONFIRMED" : "BROKEN");
    printf("byte0: ref=0x%02x dev=0x%02x | byte1: ref=0x%02x dev=0x%02x\n", ref[0], dev[0], ref[1], dev[1]);
    return !(pairing_ok && signed_ok);
}
