// =============================================================================
// NexusRT — src/firmware/ilc_manager.hpp
// Inline Compression (ILC) manager — H100 only.
//
// ILC is a Hopper feature that compresses individual memory allocations
// during transactions. It does *not* reduce the application's memory
// footprint, but it does increase effective memory bandwidth: the TMA and
// SMs operate on compressed data, transferring fewer bits over the memory
// bus.
//
// We expose ILC transparently through the memory allocator: an allocation
// tagged with AllocHints::ilc=true goes through the ILC manager, which
// sets the CU_MEM_ALLOCATION_PROP_COMPRESSED bit on the underlying handle.
// =============================================================================
#pragma once

#include "firmware/types.hpp"
#include "firmware/context.hpp"
#include <atomic>
#include <mutex>
#include <unordered_set>

namespace nexusrt {
namespace firmware {

class IlcManager {
public:
    explicit IlcManager(FirmwareContext& ctx);
    ~IlcManager();

    Status enable();
    Status disable();

    // Mark an allocation as compressed. The handle is platform-specific
    // (CUmemGenericAllocationHandle on NVIDIA, id<MTLBuffer> on Apple — but
    // ILC is only available on Hopper, so Apple is a no-op).
    Status tag(void* alloc_handle, uint64_t bytes);

    // Untag (called automatically by the destructor of the allocation).
    Status untag(void* alloc_handle);

    bool   enabled() const { return enabled_.load(); }
    uint64_t total_tagged_bytes() const { return total_bytes_.load(); }
    uint64_t total_tagged_allocs() const { return total_allocs_.load(); }

private:
    FirmwareContext& ctx_;
    std::atomic<bool>    enabled_{false};
    std::atomic<uint64_t> total_bytes_{0};
    std::atomic<uint64_t> total_allocs_{0};
    std::mutex mtx_;
    std::unordered_set<void*> tagged_;
};

} // namespace firmware
} // namespace nexusrt
