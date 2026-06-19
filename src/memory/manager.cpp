// =============================================================================
// NexusRT — src/memory/manager.cpp
// =============================================================================
#include "memory/manager.hpp"
#include "memory/coalescer.hpp"
#include "firmware/ilc_manager.hpp"
#include "platform/dispatch.hpp"

#include <algorithm>

namespace nexusrt {
namespace memory {

MemoryManager::MemoryManager(FirmwareContext& ctx)
    : ctx_(ctx),
      pt_(std::make_unique<PageTableManager>(ctx,
          ctx.plat.vaddress_base, ctx.plat.vaddress_size)),
      coalescer_(std::make_unique<Coalescer>(ctx, *pt_)) {}

MemoryManager::~MemoryManager() {
    std::lock_guard<std::mutex> g(mtx_);
    for (auto& [p, a] : live_) {
        if (a.hbm_ptr) {
            platform::PlatformDispatch::instance().free_hbm(ctx_, a.hbm_ptr);
        }
        if (a.ilc_tagged && ctx_.ilc()) {
            ctx_.ilc()->untag(a.hbm_ptr);
        }
    }
    live_.clear();
}

Status MemoryManager::alloc(uint64_t bytes, AllocHints const& hints,
                            Allocation& out) {
    if (bytes == 0) return Status::InvalidArgument;

    void* hbm = nullptr;
    auto s = platform::PlatformDispatch::instance().alloc_hbm(ctx_, bytes, hbm);
    if (!ok(s)) return s;

    out.vaddr   = reinterpret_cast<uint64_t>(hbm);
    out.bytes   = bytes;
    out.hints   = hints;
    out.hbm_ptr = hbm;

    // Tag with ILC if requested and supported.
    if (hints.ilc && ctx_.ilc() && ctx_.ilc()->enabled()) {
        auto is = ctx_.ilc()->tag(hbm, bytes);
        out.ilc_tagged = ok(is);
    }

    // Register with the page table.
    pt_->map_range(out.vaddr, bytes, PageResidency::Hbm,
                   reinterpret_cast<uint64_t>(hbm));

    if (hints.read_mostly) {
        advise_read_mostly(hbm, bytes);
    }

    std::lock_guard<std::mutex> g(mtx_);
    live_[hbm] = out;
    bytes_allocated_ += bytes;
    return Status::Ok;
}

Status MemoryManager::free(Allocation const& a) {
    std::lock_guard<std::mutex> g(mtx_);
    auto it = live_.find(a.hbm_ptr);
    if (it == live_.end()) return Status::InvalidArgument;

    if (a.ilc_tagged && ctx_.ilc()) {
        ctx_.ilc()->untag(a.hbm_ptr);
    }
    pt_->unmap_range(a.vaddr, a.bytes);
    auto s = platform::PlatformDispatch::instance().free_hbm(ctx_, a.hbm_ptr);
    if (!ok(s)) return s;
    bytes_allocated_ -= a.bytes;
    live_.erase(it);
    return Status::Ok;
}

Status MemoryManager::prefetch(void* hbm_ptr, uint64_t bytes) {
    return platform::PlatformDispatch::instance()
        .prefetch_range(ctx_, reinterpret_cast<uint64_t>(hbm_ptr), bytes);
}

Status MemoryManager::advise_read_mostly(void* hbm_ptr, uint64_t bytes) {
    // On CUDA we would call cuMemAdvise(..., CU_MEM_ADVISE_SET_READ_MOSTLY, ...).
    // Here we just register it with the page table as a pinned range so the
    // eviction policy will not evict it.
    return pt_->map_range(reinterpret_cast<uint64_t>(hbm_ptr), bytes,
                          PageResidency::Hbm,
                          reinterpret_cast<uint64_t>(hbm_ptr));
}

Status MemoryManager::coalesce(uint32_t* out_n_merged) {
    if (coalescer_) return coalescer_->run(out_n_merged);
    if (out_n_merged) *out_n_merged = 0;
    return Status::Ok;
}

uint64_t MemoryManager::bytes_allocated() const {
    std::lock_guard<std::mutex> g(mtx_);
    return bytes_allocated_;
}
uint64_t MemoryManager::bytes_resident_hbm() const {
    return pt_->resident_hbm_pages() * (ctx_.device.arch == Arch::Hopper ? 128 : 64) * 1024;
}
uint64_t MemoryManager::bytes_spilled() const {
    return pt_->spilled_pages() * (ctx_.device.arch == Arch::Hopper ? 128 : 64) * 1024;
}
uint32_t MemoryManager::fragmentation_pct() const {
    // Approximate: fragmentation = 100 - (largest_free_block / total_free).
    // In production this would walk the buddy allocator.
    return 12;  // placeholder
}

} // namespace memory
} // namespace nexusrt

// =============================================================================
// C ABI wrappers
// =============================================================================
#include "platform/abi.h"
#include "firmware/boot.hpp"

extern "C" {

NEXUSRT_API int32_t nexusrt_mem_alloc(void* ctx, uint64_t bytes, uint32_t flags,
                          void** out_ptr) {
    using namespace nexusrt;
    auto sp = firmware::FirmwareContext::lookup_handle(ctx);
    if (!sp) return static_cast<int32_t>(firmware::Status::InvalidArgument);
    AllocHints h;
    h.ilc         = (flags & 0x01) != 0;
    h.read_mostly = (flags & 0x02) != 0;
    h.pinned_host = (flags & 0x04) != 0;
    h.gds_readable= (flags & 0x08) != 0;
    h.grdma_visible = (flags & 0x10) != 0;
    memory::Allocation a;
    auto s = sp->memory().alloc(bytes, h, a);
    if (!firmware::ok(s)) return static_cast<int32_t>(s);
    *out_ptr = a.hbm_ptr;
    return 0;
}

NEXUSRT_API int32_t nexusrt_mem_free(void* ctx, void* ptr) {
    using namespace nexusrt;
    auto sp = firmware::FirmwareContext::lookup_handle(ctx);
    if (!sp) return static_cast<int32_t>(firmware::Status::InvalidArgument);
    memory::Allocation a;
    a.hbm_ptr = ptr;
    auto s = sp->memory().free(a);
    return static_cast<int32_t>(s);
}

NEXUSRT_API int32_t nexusrt_mem_prefetch(void* ctx, void* ptr, uint64_t bytes) {
    using namespace nexusrt;
    auto sp = firmware::FirmwareContext::lookup_handle(ctx);
    if (!sp) return static_cast<int32_t>(firmware::Status::InvalidArgument);
    auto s = sp->memory().prefetch(ptr, bytes);
    return static_cast<int32_t>(s);
}

NEXUSRT_API int32_t nexusrt_mem_advise_read_mostly(void* ctx, void* ptr, uint64_t bytes) {
    using namespace nexusrt;
    auto sp = firmware::FirmwareContext::lookup_handle(ctx);
    if (!sp) return static_cast<int32_t>(firmware::Status::InvalidArgument);
    auto s = sp->memory().advise_read_mostly(ptr, bytes);
    return static_cast<int32_t>(s);
}

} // extern "C"
