// =============================================================================
// NexusRT — src/platform/cuda_driver_loader.cpp
// Tiny shim that ensures the CUDA backend is linked in even when no other
// translation unit in the build references it directly.
// =============================================================================
#include "platform/dispatch.hpp"

namespace nexusrt {
namespace platform {
// Pull the auto-registration symbol from cuda_driver_shim.cu.
extern void ensure_cuda_platform_linked();
namespace {
struct Linker {
    Linker() { ensure_cuda_platform_linked(); }
};
static Linker g_cuda_linker;
} // namespace
}
}
