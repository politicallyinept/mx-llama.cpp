// E5 witness shim: LD_PRELOAD interposer on hipLaunchKernel that logs every device
// kernel launch (filtered to the interesting families) so a leg can PROVE which kernel
// served a matmul: mmq_* (stock MMQ), gcn_kq4_gemm* (gcn_kq v4), mmq_gemm_repacked*
// (repacked GEMM), mul_mat_vec_q8_0_repacked* (repacked mat-vec), mmvq (stock mat-vec),
// quantize_mmq_q8_1 / gcn_q8x_quantize (activation quantizers).
//
// Name resolution: in this HIP model hipLaunchKernel receives the address of a LOCAL
// per-kernel stub object in the launching library's .data/.text (STT_OBJECT/STT_FUNC
// symbol named _ZL<...>kernelname...), so the shim parses the ELF .symtab (library is
// not stripped) and maps func_addr - load_base -> symbol name. Verified: the launch
// pointer equals the stub symbol value exactly (delta 0).
//
// Build: hipcc -O2 -x c++ -D__HIP_PLATFORM_AMD__ -shared -fPIC
//   -I/opt/rocm-7.2.3/include kname_shim.c -o kname_shim.so -ldl
#define _GNU_SOURCE
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <hip/hip_runtime.h>

extern "C" {

typedef hipError_t (*launch_fn)(const void *, dim3, dim3, void **, size_t, hipStream_t);

struct ent { unsigned long val; unsigned idx; };  // sorted by val

struct symtab {
    const char *   file_map;   // mmap'd ELF
    const Elf64_Sym *syms;
    const char *   strs;
    struct ent *   ents;
    unsigned       n;
};

static int ent_cmp(const void * a, const void * b) {
    unsigned long x = ((const struct ent *) a)->val, y = ((const struct ent *) b)->val;
    return x < y ? -1 : x > y ? 1 : 0;
}

static int symtab_load(struct symtab * st, const char * path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat sb;
    if (fstat(fd, &sb) != 0 || sb.st_size < (off_t) sizeof(Elf64_Ehdr)) { close(fd); return -1; }
    void * m = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) return -1;
    const Elf64_Ehdr * eh = (const Elf64_Ehdr *) m;
    const Elf64_Shdr * sh = (const Elf64_Shdr *) ((const char *) m + eh->e_shoff);
    for (int i = 0; i < eh->e_shnum; ++i) {
        if (sh[i].sh_type == SHT_SYMTAB) {
            st->file_map = (const char *) m;
            st->syms     = (const Elf64_Sym *) ((const char *) m + sh[i].sh_offset);
            unsigned ns  = (unsigned) (sh[i].sh_size / sizeof(Elf64_Sym));
            st->strs     = (const char *) m + sh[sh[i].sh_link].sh_offset;
            st->ents     = (struct ent *) malloc(ns * sizeof(struct ent));
            st->n        = 0;
            for (unsigned s = 0; s < ns; ++s) {
                const unsigned char type = ELF64_ST_TYPE(st->syms[s].st_info);
                // kernel launch stubs are LOCAL OBJECTS in this HIP model; keep FUNC too
                if ((type == STT_OBJECT || type == STT_FUNC) && st->syms[s].st_value) {
                    st->ents[st->n].val = st->syms[s].st_value;
                    st->ents[st->n].idx = s;
                    st->n++;
                }
            }
            qsort(st->ents, st->n, sizeof(struct ent), ent_cmp);
            return 0;
        }
    }
    munmap(m, sb.st_size);
    return -1;
}

static const char * symtab_lookup(struct symtab * st, unsigned long off) {
    if (!st->n) return NULL;
    long lo = 0, hi = (long) st->n - 1, best = -1;
    while (lo <= hi) {
        long mid = (lo + hi) / 2;
        if (st->ents[mid].val <= off) { best = mid; lo = mid + 1; }
        else hi = mid - 1;
    }
    if (best < 0) return NULL;
    const Elf64_Sym * s = &st->syms[st->ents[best].idx];
    if (s->st_size && off >= s->st_value + s->st_size) return NULL;
    return st->strs + s->st_name;
}

static struct symtab g_st;
static int g_st_done = 0;

hipError_t hipLaunchKernel(const void *function_address, dim3 gridDim, dim3 blockDim,
                           void **paramPointer, size_t sharedMemBytes, hipStream_t stream) {
    static launch_fn real = NULL;
    if (!real) real = (launch_fn) dlsym(RTLD_NEXT, "hipLaunchKernel");
    Dl_info info;
    const char *name = NULL;
    if (dladdr(function_address, &info) && info.dli_fbase) {
        if (!g_st_done) {
            g_st_done = 1;
            if (info.dli_fname) symtab_load(&g_st, info.dli_fname);
        }
        unsigned long off = (unsigned long) ((const char *) function_address - (const char *) info.dli_fbase);
        name = symtab_lookup(&g_st, off);
        if (name && strncmp(name, "_ZL", 3) == 0) {
            name += 3;                       // drop the local-symbol mangling prefix
            if (strncmp(name, "17__device_stub__", 17) == 0) name += 17;
        }
    }
    if (!name) name = "?";
    if (getenv("E5_SHIM_DEBUG")) {
        fprintf(stderr, "[klaunch-any] %s\n", name);
    }
    if (strstr(name, "mmq") || strstr(name, "gcn") || strstr(name, "repack") ||
        strstr(name, "mmvq") || strstr(name, "mul_mat_vec") || strstr(name, "mul_mat_q") || strstr(name, "quantize") ||
        strstr(name, "dequantize") || strstr(name, "convert")) {
        fprintf(stderr, "[klaunch] %s grid=(%u,%u,%u) block=(%u,%u,%u) shmem=%zu\n",
                name, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
                sharedMemBytes);
    }
    return real(function_address, gridDim, blockDim, paramPointer, sharedMemBytes, stream);
}

}  // extern "C"
