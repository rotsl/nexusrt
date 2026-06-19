// =============================================================================
// NexusRT — src/memory/coalescer.hpp
// Unified-memory coalescer. Merges fragmented allocations and aligns to
// HBM warp boundaries so the TMA / async-copy engine can issue bulk
// transactions without straddling pages.
// =============================================================================
#pragma once

#include "firmware/types.hpp"
#include "firmware/context.hpp"
#include "memory/page_table.hpp"

namespace nexusrt {
namespace memory {

class Coalescer {
public:
    Coalescer(FirmwareContext& ctx, PageTableManager& pt)
        : ctx_(ctx), pt_(pt) {}

    // Walk the page table; merge adjacent HBM-resident allocations that
    // belong to the same allocation group. Returns the number of merges.
    Status run(uint32_t* out_n_merged);

    // Re-align an allocation to the warp boundary (128B on Ampere, 128B on
    // Hopper with 128B swizzle).
    static uint64_t align_to_warp(uint64_t v, Arch a) {
        uint64_t b = 128;
        (void)a;
        return (v + b - 1) & ~(b - 1);
    }

private:
    FirmwareContext&   ctx_;
    PageTableManager&  pt_;
};

} // namespace memory
} // namespace nexusrt
