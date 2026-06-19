// =============================================================================
// NexusRT — src/memory/manager.hpp
// High-level memory manager. Owns:
//   - the PageTableManager
//   - the unified-memory coalescer
//   - the GDS / GRDMA fetch router
//   - the ILC allocator (delegated to firmware::IlcManager)
//
// Exposes nexusrt_mem_alloc / nexusrt_mem_prefetch / nexusrt_mem_free via
// the unified C ABI.
// =============================================================================
#pragma once

#include "firmware/types.hpp"
#include "firmware/context.hpp"
#include "memory/page_table.hpp"

#include <memory>
#include <mutex>
#include <unordered_map>

namespace nexusrt {
namespace memory {

struct Allocation {
    uint64_t       vaddr      = 0;
    uint64_t       bytes      = 0;
    AllocHints     hints;
    void*          hbm_ptr    = nullptr;  // device pointer (== vaddr on CUDA)
    void*          host_ptr   = nullptr;  // for pinned/staging allocations
    bool           ilc_tagged = false;
};

class Coalescer;

class MemoryManager {
public:
    MemoryManager(FirmwareContext& ctx);
    ~MemoryManager();

    // Allocate a tensor in HBM. Honors AllocHints (ILC, read-mostly, etc.).
    Status alloc(uint64_t bytes, AllocHints const& hints, Allocation& out);

    // Free an allocation.
    Status free(Allocation const& a);

    // Prefetch a range into HBM (no-op if already resident).
    Status prefetch(void* hbm_ptr, uint64_t bytes);

    // Hint that a range is read-mostly (e.g., model weights).
    Status advise_read_mostly(void* hbm_ptr, uint64_t bytes);

    // Coalesce fragmented allocations. Triggered automatically when
    // fragmentation exceeds runtime.yaml:coalescer.merge_fragments_above_pct.
    Status coalesce(uint32_t* out_n_merged);

    // Stats
    uint64_t bytes_allocated() const;
    uint64_t bytes_resident_hbm() const;
    uint64_t bytes_spilled() const;
    uint32_t fragmentation_pct() const;

    // Access to the page table (used by scheduler / token_opt).
    PageTableManager& page_table() { return *pt_; }

private:
    FirmwareContext& ctx_;
    std::unique_ptr<PageTableManager> pt_;
    std::unique_ptr<Coalescer>        coalescer_;
    mutable std::mutex                mtx_;
    std::unordered_map<void*, Allocation> live_;
    uint64_t                          bytes_allocated_ = 0;
};

} // namespace memory
} // namespace nexusrt
