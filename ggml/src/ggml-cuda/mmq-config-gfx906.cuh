// gfx906 (Vega20 / MI50, wave64) MMQ config.
//
// Everything but MXFP4 takes upstream's GCN table, which carries the same Q8_0 tuning this file used to hold (8 warps, tile widths to J=128) and is faster on Q4_0.
// The wide Q8_0 tiles are only SELECTED when they keep the CUs busy, see the gfx906 occupancy gate in the host tile-width loop (mmq.cu).
//
// MXFP4 keeps 8 warps over the rdna2 tile layout, the configuration measured on MI50.
// Perf-only knobs, results are bit-exact. Include after mmq-config-gcn.cuh and mmq-config-rdna2.cuh.
static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_gfx906(ggml_type type, int J, bool fallback) {
    if (type == GGML_TYPE_MXFP4) {
        const ggml_cuda_mmq_config rdna2 = ggml_cuda_mmq_get_config_rdna2(type, J, fallback);
        if (rdna2.type == GGML_TYPE_COUNT) {
            return rdna2;
        }
        return ggml_cuda_mmq_config(
            rdna2.type, 512, rdna2.occupancy, rdna2.I, rdna2.J, rdna2.sram_layout, rdna2.K_vram, rdna2.stream_k, rdna2.fallback);
    }
    return ggml_cuda_mmq_get_config_gcn(type, J, fallback);
}
