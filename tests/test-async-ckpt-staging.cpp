// test-async-ckpt-staging: F2 robustness - a failed async-checkpoint staging allocation must
// degrade gracefully (log + synchronous fallback), never abort from the io destructor.
// CPU-only: no device required, no GPU touched.
#include "llama-context.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <memory>

static int failures = 0;

#define CHECK(cond, msg)                          \
    do {                                          \
        if (!(cond)) {                            \
            printf("FAIL: %s\n", (msg));          \
            ++failures;                           \
        } else {                                  \
            printf("ok:   %s\n", (msg));          \
        }                                         \
    } while (0)

int main(void) {
    // the CPU device registers no separate "host" buffer type (that is for pinned GPU staging),
    // so the plain CPU buffer type stands in for the staging buft here - the helper under test is
    // buft-agnostic
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    CHECK(dev != nullptr, "CPU device available");
    if (dev == nullptr) {
        return 1;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    CHECK(buft != nullptr, "cpu buffer type available");
    if (buft == nullptr) {
        return 1;
    }

    // 1. injected failure: the helper returns null and nothing aborts (this is the path that
    //    used to hit GGML_ASSERT inside ~llama_io_write_async)
    setenv("LLAMA_TEST_ASYNC_STAGING_FAIL", "1", 1);
    CHECK(llama_state_async_alloc_staging(buft, 64 * 1024) == nullptr,
          "injected staging-alloc failure returns null (no abort)");

    // 2. normal small allocation succeeds once the hook is off
    unsetenv("LLAMA_TEST_ASYNC_STAGING_FAIL");
    ggml_backend_buffer_t buf = llama_state_async_alloc_staging(buft, 64 * 1024);
    CHECK(buf != nullptr, "small staging allocation succeeds");
    if (buf != nullptr) {
        ggml_backend_buffer_free(buf);
    }

    // 3. natural failure: an absurd size must return null, not abort
    ggml_backend_buffer_t huge = llama_state_async_alloc_staging(buft, (size_t) 1 << 48);
    CHECK(huge == nullptr, "absurd-size (256 TiB) staging allocation fails gracefully");
    if (huge != nullptr) {
        printf("note: allocator overcommitted the 256 TiB request; freeing\n");
        ggml_backend_buffer_free(huge);
    }

    // 4. the slot destructor must be safe on a never-used slot (this is the object the io
    //    destructor used to leave with a null pinned buffer behind the removed GGML_ASSERT)
    {
        auto slot = std::make_unique<llama_context::state_async_slot>();
        slot->busy = true;  // left busy on purpose: the destructor must still be safe
        slot.reset();
        CHECK(true, "state_async_slot destructor on an unused slot does not abort");
    }

    if (failures == 0) {
        printf("test-async-ckpt-staging: PASS\n");
        return 0;
    }
    printf("test-async-ckpt-staging: FAIL (%d)\n", failures);
    return 1;
}
