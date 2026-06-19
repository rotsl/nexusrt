// =============================================================================
// NexusRT — src/firmware/dma_engine.hpp
// DMA engine abstraction — wraps GDS / GRDMA / TMA paths.
//
// On H100, the TMA is preferred for tensor copies (see tma_engine.hpp).
// On A100, this engine drives async-copy + warp-specialized producer queues.
// On M1 Pro, this engine drives Metal blit encoders.
// =============================================================================
#pragma once

#include "firmware/types.hpp"
#include "firmware/context.hpp"
#include <memory>
#include <string>

namespace nexusrt {
namespace firmware {

struct DmaRequest {
    enum class Kind : uint8_t { GdsRead, GdsWrite, GrdmaSend, GrdmaRecv, H2D, D2H, D2D };

    Kind      kind       = Kind::D2D;
    void*     dst        = nullptr;
    void*     src        = nullptr;
    uint64_t  bytes      = 0;
    int       dst_device = 0;       // for cross-device D2D / GRDMA
    int       src_device = 0;
    std::string nvme_path;          // for GDS
    uint64_t    nvme_offset = 0;
};

class DmaEngine {
public:
    DmaEngine(FirmwareContext& ctx, DeviceDesc const& d);
    ~DmaEngine();

    Status enable_gds();
    Status enable_grdma();
    void   shutdown();

    // Submit a DMA request on a Dma-class stream. Returns immediately; use
    // wait() to synchronize.
    Status submit(DmaRequest const& req, void** out_event);

    // Wait on a previously returned event.
    Status wait(void* event, uint32_t timeout_ms);

    // Synchronous convenience wrappers.
    Status gds_read (const std::string& nvme_path, uint64_t off,
                     void* dst_hbm, uint64_t bytes);
    Status gds_write(const std::string& nvme_path, uint64_t off,
                     void* src_hbm, uint64_t bytes);

    bool    gds_enabled()   const { return gds_enabled_; }
    bool    grdma_enabled() const { return grdma_enabled_; }

private:
    FirmwareContext& ctx_;
    DeviceDesc       device_;
    bool             gds_enabled_   = false;
    bool             grdma_enabled_ = false;
};

} // namespace firmware
} // namespace nexusrt
