// =============================================================================
// NexusRT — src/memory/page_table.cpp
// =============================================================================
#include "memory/page_table.hpp"
#include "firmware/fault_handler.hpp"
#include "platform/dispatch.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace nexusrt {
namespace memory {

namespace {
uint64_t now_ns() {
    return std::chrono::steady_clock::now().time_since_epoch().count();
}
} // namespace

PageTableManager::PageTableManager(FirmwareContext& ctx,
                                   uint64_t va_base, uint64_t va_size)
    : ctx_(ctx), va_base_(va_base), va_size_(va_size),
      default_page_kb_(ctx.device.arch == Arch::Hopper ? 128 : 64) {}

PageTableManager::~PageTableManager() {
    std::lock_guard<std::mutex> g(mtx_);
    table_.clear();
}

Status PageTableManager::map_range(uint64_t vaddr, uint64_t bytes,
                                   PageResidency res, uint64_t phys_or_host) {
    std::lock_guard<std::mutex> g(mtx_);
    uint64_t page_bytes = default_page_kb_ * 1024;
    uint64_t start = vaddr & ~(page_bytes - 1);
    uint64_t end   = (vaddr + bytes + page_bytes - 1) & ~(page_bytes - 1);
    for (uint64_t a = start; a < end; a += page_bytes) {
        PageEntry e;
        e.vaddr           = a;
        e.phys_or_host    = phys_or_host + (a - start);
        e.page_size_kb    = default_page_kb_;
        e.refcount        = 0;
        e.residency       = res;
        e.last_access_ns  = now_ns();
        e.fetch_count     = 1;
        table_[a] = e;
    }
    return Status::Ok;
}

Status PageTableManager::unmap_range(uint64_t vaddr, uint64_t bytes) {
    std::lock_guard<std::mutex> g(mtx_);
    uint64_t page_bytes = default_page_kb_ * 1024;
    uint64_t start = vaddr & ~(page_bytes - 1);
    uint64_t end   = (vaddr + bytes + page_bytes - 1) & ~(page_bytes - 1);
    for (uint64_t a = start; a < end; a += page_bytes) {
        auto it = table_.find(a);
        if (it == table_.end()) continue;
        if (it->second.refcount > 0) {
            // Refuse to evict a pinned page.
            return Status::ContractViolation;
        }
        table_.erase(it);
    }
    return Status::Ok;
}

Status PageTableManager::incref(uint64_t vaddr) {
    std::lock_guard<std::mutex> g(mtx_);
    uint64_t page_bytes = default_page_kb_ * 1024;
    uint64_t a = vaddr & ~(page_bytes - 1);
    auto it = table_.find(a);
    if (it == table_.end()) return Status::InvalidArgument;
    if (it->second.refcount == 0xFFFF) return Status::InvalidArgument;
    it->second.refcount++;
    it->second.last_access_ns = now_ns();
    return Status::Ok;
}

Status PageTableManager::decref(uint64_t vaddr) {
    std::lock_guard<std::mutex> g(mtx_);
    uint64_t page_bytes = default_page_kb_ * 1024;
    uint64_t a = vaddr & ~(page_bytes - 1);
    auto it = table_.find(a);
    if (it == table_.end()) return Status::InvalidArgument;
    if (it->second.refcount == 0) return Status::InvalidArgument;
    it->second.refcount--;
    it->second.last_access_ns = now_ns();
    return Status::Ok;
}

PageEntry const* PageTableManager::lookup(uint64_t vaddr) const {
    std::lock_guard<std::mutex> g(mtx_);
    uint64_t page_bytes = default_page_kb_ * 1024;
    uint64_t a = vaddr & ~(page_bytes - 1);
    auto it = table_.find(a);
    return it == table_.end() ? nullptr : &it->second;
}

uint32_t PageTableManager::evict_until(uint32_t high_pct,
                                       uint32_t low_pct,
                                       EvictionPolicy pol) {
    std::lock_guard<std::mutex> g(mtx_);
    // Count resident HBM pages.
    uint64_t page_bytes = default_page_kb_ * 1024;
    uint64_t total_hbm_pages = ctx_.hbm_pool_bytes / page_bytes;
    uint64_t resident = 0;
    for (auto const& [a, e] : table_) {
        if (e.residency == PageResidency::Hbm) ++resident;
    }
    uint64_t high = total_hbm_pages * high_pct / 100;
    uint64_t low  = total_hbm_pages * low_pct  / 100;
    if (resident <= high) return 0;

    // Build a sortable list of eviction candidates.
    struct Cand { uint64_t vaddr; uint64_t last_access; uint16_t refcount; };
    std::vector<Cand> cands;
    cands.reserve(table_.size());
    for (auto const& [a, e] : table_) {
        if (e.residency != PageResidency::Hbm) continue;
        if (e.refcount > 0) continue;
        cands.push_back({a, e.last_access_ns, e.refcount});
    }
    if (pol == EvictionPolicy::Lru || pol == EvictionPolicy::LruRefcount) {
        std::sort(cands.begin(), cands.end(),
                  [](auto const& x, auto const& y) {
                      return x.last_access < y.last_access;
                  });
    }

    uint32_t evicted = 0;
    for (auto const& c : cands) {
        if (resident <= low) break;
        // Mark as spilled (NVMe-backed) — the actual GDS write happens in
        // the dma_engine layer.
        auto it = table_.find(c.vaddr);
        if (it == table_.end()) continue;
        it->second.residency = PageResidency::Spilled;
        it->second.fetch_count++;
        --resident;
        ++evicted;
    }
    return evicted;
}

uint64_t PageTableManager::total_pages() const {
    std::lock_guard<std::mutex> g(mtx_);
    return table_.size();
}
uint64_t PageTableManager::resident_hbm_pages() const {
    std::lock_guard<std::mutex> g(mtx_);
    uint64_t n = 0;
    for (auto const& [a, e] : table_) if (e.residency == PageResidency::Hbm) ++n;
    return n;
}
uint64_t PageTableManager::resident_host_pages() const {
    std::lock_guard<std::mutex> g(mtx_);
    uint64_t n = 0;
    for (auto const& [a, e] : table_)
        if (e.residency == PageResidency::HostPinned ||
            e.residency == PageResidency::HostPageable) ++n;
    return n;
}
uint64_t PageTableManager::spilled_pages() const {
    std::lock_guard<std::mutex> g(mtx_);
    uint64_t n = 0;
    for (auto const& [a, e] : table_) if (e.residency == PageResidency::Spilled) ++n;
    return n;
}

} // namespace memory
} // namespace nexusrt
