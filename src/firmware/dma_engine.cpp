// =============================================================================
// NexusRT — src/firmware/dma_engine.cpp
// =============================================================================
#include "firmware/dma_engine.hpp"
#include "firmware/microkernel.hpp"
#include "platform/dispatch.hpp"

namespace nexusrt {
namespace firmware {

DmaEngine::DmaEngine(FirmwareContext& ctx, DeviceDesc const& d)
    : ctx_(ctx), device_(d) {}

DmaEngine::~DmaEngine() { shutdown(); }

Status DmaEngine::enable_gds() {
    if (!device_.features.gds) return Status::NotImplemented;
    auto s = platform::PlatformDispatch::instance().gds_init(ctx_);
    if (ok(s)) gds_enabled_ = true;
    return s;
}

Status DmaEngine::enable_grdma() {
    if (!device_.features.grdma) return Status::NotImplemented;
    auto s = platform::PlatformDispatch::instance().grdma_init(ctx_);
    if (ok(s)) grdma_enabled_ = true;
    return s;
}

void DmaEngine::shutdown() {
    if (grdma_enabled_) {
        platform::PlatformDispatch::instance().grdma_fini(ctx_);
        grdma_enabled_ = false;
    }
    if (gds_enabled_) {
        platform::PlatformDispatch::instance().gds_fini(ctx_);
        gds_enabled_ = false;
    }
}

Status DmaEngine::submit(DmaRequest const& req, void** out_event) {
    return platform::PlatformDispatch::instance()
        .dma_submit(ctx_, req, out_event);
}

Status DmaEngine::wait(void* event, uint32_t timeout_ms) {
    return platform::PlatformDispatch::instance().dma_wait(ctx_, event, timeout_ms);
}

Status DmaEngine::gds_read(const std::string& path, uint64_t off,
                           void* dst, uint64_t bytes) {
    if (!gds_enabled_) {
        // Fall back to host-pinned bounce buffer + D2H→H2D.
        return platform::PlatformDispatch::instance()
            .gds_read_fallback(ctx_, path, off, dst, bytes);
    }
    DmaRequest r;
    r.kind = DmaRequest::Kind::GdsRead;
    r.dst  = dst;
    r.bytes = bytes;
    r.nvme_path = path;
    r.nvme_offset = off;
    void* ev = nullptr;
    auto s = submit(r, &ev);
    if (!ok(s)) return s;
    return wait(ev, 30000 /* 30s */);
}

Status DmaEngine::gds_write(const std::string& path, uint64_t off,
                            void* src, uint64_t bytes) {
    if (!gds_enabled_) {
        return platform::PlatformDispatch::instance()
            .gds_write_fallback(ctx_, path, off, src, bytes);
    }
    DmaRequest r;
    r.kind = DmaRequest::Kind::GdsWrite;
    r.src  = src;
    r.bytes = bytes;
    r.nvme_path = path;
    r.nvme_offset = off;
    void* ev = nullptr;
    auto s = submit(r, &ev);
    if (!ok(s)) return s;
    return wait(ev, 30000);
}

} // namespace firmware
} // namespace nexusrt
