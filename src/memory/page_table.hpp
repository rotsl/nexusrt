// =============================================================================
// NexusRT — src/memory/page_table.hpp
// GPU-driven page table manager (DREAM-inspired).
//
// Standard CUDA Unified Memory uses the host OS to handle page faults. NexusRT
// instead maintains a page table *in HBM* and routes faults through a
// GPU-resident fault buffer (see firmware/fault_handler.hpp). This file
// implements the host-side mirror of that table — the authoritative copy
// lives in HBM and is updated by the firmware-equivalent poller.
// =============================================================================
#pragma once

#include "firmware/types.hpp"
#include "firmware/context.hpp"
#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace nexusrt {
namespace memory {

enum class PageResidency : uint8_t {
    Unmapped   = 0,
    Hbm        = 1,    // resident in HBM
    HostPinned = 2,    // in host pinned memory
    HostPageable = 3,
    RemoteGpu  = 4,    // on another GPU via GRDMA
    Spilled    = 5,    // backed by GDS / NVMe
};

enum class EvictionPolicy : uint8_t {
    Lru          = 0,
    LruRefcount  = 1,   // LRU but never evict pages with refcount > 0
    Arc          = 2,   // Adaptive Replacement Cache
};

// PageEntry — host-side mirror of the HBM-resident page-table entry.
// The HBM-resident entry has the same layout (32 bytes) for direct memcpy.
struct PageEntry {
    uint64_t       vaddr          = 0;
    uint64_t       phys_or_host   = 0;    // HBM pointer or host pointer
    uint32_t       page_size_kb   = 64;
    uint16_t       refcount       = 0;
    PageResidency  residency      = PageResidency::Unmapped;
    uint8_t        pad[5]         = {0};
    uint64_t       last_access_ns = 0;    // monotonic, for LRU
    uint64_t       fetch_count    = 0;    // how many times we've paged this in
};

// PageTableManager — owns the host-side mirror and pushes updates to HBM.
class PageTableManager {
public:
    PageTableManager(FirmwareContext& ctx, uint64_t va_base, uint64_t va_size);
    ~PageTableManager();

    // Map a virtual range to a backing store. The store is determined by
    // the AllocHints used when the allocation was created.
    Status map_range(uint64_t vaddr, uint64_t bytes, PageResidency res,
                     uint64_t phys_or_host);

    // Unmap and evict a range.
    Status unmap_range(uint64_t vaddr, uint64_t bytes);

    // Increment / decrement refcount. Pages with refcount > 0 are pinned
    // and cannot be evicted by the LRU policy.
    Status incref(uint64_t vaddr);
    Status decref(uint64_t vaddr);

    // Lookup — returns nullptr if the page is unmapped.
    PageEntry const* lookup(uint64_t vaddr) const;

    // Eviction — walks the table and evicts pages until HBM occupancy drops
    // below `low_watermark_pct`. Returns the number of pages evicted.
    uint32_t evict_until(uint32_t high_watermark_pct,
                         uint32_t low_watermark_pct,
                         EvictionPolicy pol);

    // Statistics
    uint64_t total_pages()        const;
    uint64_t resident_hbm_pages() const;
    uint64_t resident_host_pages()const;
    uint64_t spilled_pages()      const;

private:
    FirmwareContext& ctx_;
    uint64_t         va_base_;
    uint64_t         va_size_;
    uint32_t         default_page_kb_;
    mutable std::mutex mtx_;
    std::unordered_map<uint64_t, PageEntry> table_;   // keyed by page-aligned vaddr
};

} // namespace memory
} // namespace nexusrt
