// =============================================================================
// NexusRT — src/firmware/tma_engine.cpp
// =============================================================================
#include "firmware/tma_engine.hpp"
#include "platform/dispatch.hpp"

namespace nexusrt {
namespace firmware {

TmaEngine::TmaEngine(FirmwareContext& ctx) : ctx_(ctx) {}
TmaEngine::~TmaEngine() {
    std::lock_guard<std::mutex> g(mtx_);
    for (auto& [k, d] : cache_) {
        if (d->raw) {
            platform::PlatformDispatch::instance().tma_release(ctx_, d->raw);
        }
    }
}

Status TmaEngine::encode(const std::string& key,
                         void*              host_tensor_base,
                         uint32_t           elem_size,
                         TmaShape const&    shape,
                         TmaStride const&   stride,
                         TmaSwizzle         swizzle,
                         bool               multicast,
                         uint32_t           cluster_size,
                         TmaDescriptor&     out) {
    std::lock_guard<std::mutex> g(mtx_);
    auto it = cache_.find(key);
    if (it != cache_.end()) { out = *it->second; return Status::Ok; }

    auto d = std::make_shared<TmaDescriptor>();
    d->id          = next_id_++;
    d->host_ptr    = host_tensor_base;
    d->shape       = shape;
    d->stride      = stride;
    d->swizzle     = swizzle;
    d->elem_size   = elem_size;
    d->multicast   = multicast;
    d->cluster_size= cluster_size;

    auto s = platform::PlatformDispatch::instance().tma_encode(
        ctx_, *d);
    if (!ok(s)) return s;

    cache_[key] = d;
    out = *d;
    return Status::Ok;
}

TmaDescriptor const* TmaEngine::lookup(const std::string& key) const {
    std::lock_guard<std::mutex> g(mtx_);
    auto it = cache_.find(key);
    return it == cache_.end() ? nullptr : it->second.get();
}

Status TmaEngine::issue_copy(TmaDescriptor const& d,
                             void*                smem_dst,
                             uint64_t             tile_off_x,
                             uint64_t             tile_off_y,
                             void*                stream) {
    return platform::PlatformDispatch::instance()
        .tma_issue_copy(ctx_, d, smem_dst, tile_off_x, tile_off_y, stream);
}

void TmaEngine::invalidate(const std::string& key) {
    std::lock_guard<std::mutex> g(mtx_);
    auto it = cache_.find(key);
    if (it == cache_.end()) return;
    if (it->second->raw) {
        platform::PlatformDispatch::instance().tma_release(ctx_, it->second->raw);
    }
    cache_.erase(it);
}

} // namespace firmware
} // namespace nexusrt
