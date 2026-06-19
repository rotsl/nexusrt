// =============================================================================
// NexusRT — src/firmware/microkernel.hpp
// GPU-native micro-kernel abstraction.
//
// The micro-kernel is the central scheduler of the firmware-equivalent layer.
// It owns:
//   - the stream pool (compute / dma / fence)
//   - the warp-specialization queue table
//   - the loaded PTX/SASS module table
//
// On Hopper, the micro-kernel additionally owns the TMA descriptor cache,
// exposed via tma_engine.hpp.
// =============================================================================
#pragma once

#include "firmware/types.hpp"
#include "firmware/context.hpp"
#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace nexusrt {
namespace firmware {

// StreamHandle — opaque handle to a CUDA stream / Metal command buffer.
struct StreamHandle {
    uint64_t id = 0;
    void*    raw = nullptr;          // CUstream or id<MTLCommandBuffer>
    StreamClass klass = StreamClass::Compute;
    int      priority = 0;
};

// KernelModule — a loaded PTX / SASS / Metal library.
struct KernelModule {
    uint64_t    id = 0;
    std::string name;
    void*       raw_module = nullptr;        // CUmodule or id<MTLLibrary>
    std::map<std::string, void*> functions;  // name -> CUfunction or id<MTLFunction>
};

// WarpSpecializationQueue — declares which warps in a CTA perform which role.
struct WarpSpecQueue {
    uint32_t producer_warps = 2;   // TMA / async-copy issuers
    uint32_t consumer_warps = 14;  // compute warps
    uint32_t fence_warps    = 0;   // H100 only — mbarrier arrievement
    uint32_t total_warps() const { return producer_warps + consumer_warps + fence_warps; }
};

class MicroKernel {
public:
    MicroKernel(FirmwareContext& ctx, DeviceDesc const& d);
    ~MicroKernel();

    // ---- Lifecycle ---------------------------------------------------------
    Status spawn_stream_pool(uint32_t n_compute, uint32_t n_dma);
    Status drain_streams();        // sync + release
    Status load_module(const std::string& name,
                       const std::string& ptx_or_metal_src,
                       KernelModule&      out);

    // ---- Stream allocation -------------------------------------------------
    // Acquire a stream of the given class. Returns to the pool on release.
    StreamHandle acquire_stream(StreamClass klass, int priority = 0);
    void         release_stream(StreamHandle h);

    // ---- Warp specialization ----------------------------------------------
    WarpSpecQueue const& default_warp_spec() const { return default_ws_; }
    void set_default_warp_spec(WarpSpecQueue const& ws) { default_ws_ = ws; }

    // ---- Module table ------------------------------------------------------
    KernelModule const* find_module(const std::string& name) const;

    // ---- Submission --------------------------------------------------------
    // Submit a pre-loaded kernel + grid configuration on a stream.
    // This is the lowest-level "task" entry point. Higher levels (scheduler/)
    // build on top of this.
    Status launch(const std::string& module_name,
                  const std::string& function_name,
                  uint32_t           grid_x, uint32_t grid_y, uint32_t grid_z,
                  uint32_t           block_x, uint32_t block_y, uint32_t block_z,
                  uint32_t           shared_mem_bytes,
                  StreamHandle       stream,
                  void**             kernel_args,
                  size_t             n_args);

    // ---- Sync primitives ---------------------------------------------------
    Status sync_stream(StreamHandle s);
    Status sync_all();

private:
    FirmwareContext& ctx_;
    DeviceDesc       device_;
    WarpSpecQueue    default_ws_;

    mutable std::mutex                            mtx_;
    std::vector<StreamHandle>                     free_compute_;
    std::vector<StreamHandle>                     free_dma_;
    std::vector<StreamHandle>                     free_fence_;
    std::map<std::string, std::shared_ptr<KernelModule>> modules_;
    std::atomic<uint64_t>                         next_id_{1};
};

} // namespace firmware
} // namespace nexusrt
