// =============================================================================
// NexusRT — src/firmware/tma_engine.hpp
// Hopper TMA (Tensor Memory Accelerator) descriptor cache + dispatcher.
//
// Only instantiated on H100. On A100 this class is never constructed; the
// dma_engine provides the equivalent async-copy path.
//
// TMA is a hardware unit that issues async bulk copies between global and
// shared memory. The descriptor (CUtensorMap) encodes the source tensor
// layout, tile shape, swizzle, and fill mode. We cache descriptors in HBM
// because cuTensorMapEncodeTiled is a host-side call.
// =============================================================================
#pragma once

#include "firmware/types.hpp"
#include "firmware/context.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace nexusrt {
namespace firmware {

// 5-D is the maximum dimensionality supported by Hopper TMA.
using TmaShape   = std::array<uint64_t, 5>;
using TmaStride  = std::array<uint64_t, 5>;

enum class TmaSwizzle : uint8_t {
    None = 0,
    B32  = 1,
    B64  = 2,
    B128 = 3,    // most common for attention/MLP
};

enum class TmaFillMode : uint8_t {
    NoFill  = 0,
    Const   = 1,
};

struct TmaDescriptor {
    uint64_t    id        = 0;
    void*       raw       = nullptr;        // CUtensorMap
    void*       host_ptr  = nullptr;        // host tensor base
    uint64_t    host_size = 0;
    TmaShape    shape{};
    TmaStride   stride{};
    TmaSwizzle  swizzle   = TmaSwizzle::B128;
    uint32_t    elem_size = 2;              // bf16 = 2 bytes
    bool        multicast = false;
    uint32_t    cluster_size = 1;
};

class TmaEngine {
public:
    explicit TmaEngine(FirmwareContext& ctx);
    ~TmaEngine();

    // Encode a 2-D / 3-D tiled descriptor and cache it.
    Status encode(const std::string& key,
                  void*              host_tensor_base,
                  uint32_t           elem_size,
                  TmaShape const&    shape,
                  TmaStride const&   stride,
                  TmaSwizzle         swizzle,
                  bool               multicast,
                  uint32_t           cluster_size,
                  TmaDescriptor&     out);

    // Look up a previously encoded descriptor.
    TmaDescriptor const* lookup(const std::string& key) const;

    // Issue a TMA copy (used by the scheduler; rarely called directly).
    Status issue_copy(TmaDescriptor const& d,
                      void*                smem_dst,
                      uint64_t             tile_offset_x,
                      uint64_t             tile_offset_y,
                      void*                stream);

    void invalidate(const std::string& key);

private:
    FirmwareContext& ctx_;
    mutable std::mutex mtx_;
    std::unordered_map<std::string, std::shared_ptr<TmaDescriptor>> cache_;
    uint64_t next_id_ = 1;
};

} // namespace firmware
} // namespace nexusrt
