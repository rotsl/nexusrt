// =============================================================================
// NexusRT — src/firmware/ilc_manager.cpp
// =============================================================================
#include "firmware/ilc_manager.hpp"
#include "platform/dispatch.hpp"

namespace nexusrt {
namespace firmware {

IlcManager::IlcManager(FirmwareContext& ctx) : ctx_(ctx) {}
IlcManager::~IlcManager() { disable(); }

Status IlcManager::enable() {
    if (!ctx_.device.features.ilc) return Status::NotImplemented;
    enabled_ = true;
    return Status::Ok;
}

Status IlcManager::disable() {
    if (!enabled_.load()) return Status::Ok;
    std::lock_guard<std::mutex> g(mtx_);
    for (auto h : tagged_) {
        platform::PlatformDispatch::instance().ilc_untag(ctx_, h);
    }
    tagged_.clear();
    enabled_ = false;
    return Status::Ok;
}

Status IlcManager::tag(void* alloc_handle, uint64_t bytes) {
    if (!enabled_.load()) return Status::NotImplemented;
    auto s = platform::PlatformDispatch::instance().ilc_tag(ctx_, alloc_handle);
    if (!ok(s)) return s;
    {
        std::lock_guard<std::mutex> g(mtx_);
        tagged_.insert(alloc_handle);
    }
    total_bytes_.fetch_add(bytes);
    total_allocs_.fetch_add(1);
    return Status::Ok;
}

Status IlcManager::untag(void* alloc_handle) {
    if (!enabled_.load()) return Status::Ok;
    std::lock_guard<std::mutex> g(mtx_);
    auto it = tagged_.find(alloc_handle);
    if (it == tagged_.end()) return Status::InvalidArgument;
    tagged_.erase(it);
    return platform::PlatformDispatch::instance().ilc_untag(ctx_, alloc_handle);
}

} // namespace firmware
} // namespace nexusrt
