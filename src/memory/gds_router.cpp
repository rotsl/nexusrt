// =============================================================================
// NexusRT — src/memory/gds_router.cpp
// GPUDirect Storage fetch router. Zero-copy NVMe → HBM.
//
// On systems with GDS installed we delegate to cuFile. On systems without
// GDS we fall back to host-pinned bounce buffers + cuMemcpyHtoDAsync. The
// fallback is exposed as gds_read_fallback in the platform layer.
// =============================================================================
#include "firmware/context.hpp"
#include "firmware/dma_engine.hpp"
#include "platform/dispatch.hpp"

#include <cstring>
#include <fstream>
#include <vector>

namespace nexusrt {
namespace memory {

// Public host-side entry. Routes a fetch through GDS if available, else
// falls back.
Status gds_fetch(FirmwareContext& ctx, const std::string& path,
                 uint64_t offset, void* dst_hbm, uint64_t bytes) {
    auto& dma = ctx.dma();
    return dma.gds_read(path, offset, dst_hbm, bytes);
}

// Bulk streaming read — chunks a large file into GDS-friendly sizes.
Status gds_stream(FirmwareContext& ctx, const std::string& path,
                  uint64_t offset, void* dst_hbm, uint64_t bytes,
                  uint64_t chunk_bytes /*=16MB*/) {
    if (chunk_bytes == 0) chunk_bytes = 16ull * 1024 * 1024;
    auto& dma = ctx.dma();
    uint8_t* p = reinterpret_cast<uint8_t*>(dst_hbm);
    for (uint64_t off = 0; off < bytes; off += chunk_bytes) {
        uint64_t n = std::min<uint64_t>(chunk_bytes, bytes - off);
        auto s = dma.gds_read(path, offset + off, p + off, n);
        if (!ok(s)) return s;
    }
    return Status::Ok;
}

} // namespace memory
} // namespace nexusrt
