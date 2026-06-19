// =============================================================================
// NexusRT — src/firmware/context.hpp
// FirmwareContext — single owner of all firmware-equivalent state.
//
// A FirmwareContext ties together:
//   - the discovered DeviceDesc
//   - the primary CUDA / Metal context
//   - the stream pool (compute / dma / fence)
//   - the HBM-resident fault buffer
//   - the TMA descriptor cache (H100 only)
//   - the ILC manager (H100 only)
//
// It is *thread-safe*: every member is either immutable after boot or
// guarded by an internal mutex. RAII is enforced — destruction releases all
// GPU resources even if nexusrt_firmware_shutdown() was not called.
// =============================================================================
#pragma once

#include "firmware/types.hpp"
#include "firmware/boot.hpp"
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace nexusrt {
namespace memory { class MemoryManager; }
namespace firmware {

class MicroKernel;
class DmaEngine;
class TmaEngine;
class FaultHandler;
class IlcManager;

// Opaque CUDA / Metal context handle. Implemented per-platform.
struct PlatformContextHandle {
    void* raw_primary_ctx = nullptr;   // CUcontext or id<MTLDevice>
    void* raw_command_queue = nullptr; // CUstream pool root or id<MTLCommandQueue>
    uint64_t vaddress_base = 0;        // base of reserved VA range
    uint64_t vaddress_size = 0;
};

// FirmwareContext — owns every firmware-equivalent resource.
class FirmwareContext : public std::enable_shared_from_this<FirmwareContext> {
public:
    explicit FirmwareContext(DeviceDesc d) : device(std::move(d)) {}
    ~FirmwareContext();

    // Move-only.
    FirmwareContext(const FirmwareContext&)            = delete;
    FirmwareContext& operator=(const FirmwareContext&) = delete;
    FirmwareContext(FirmwareContext&&) noexcept        = default;
    FirmwareContext& operator=(FirmwareContext&&) noexcept = delete;

    // ---- Subsystem accessors (thread-safe) ---------------------------------
    MicroKernel&    microkernel()  { return *mk_; }
    DmaEngine&      dma()          { return *dma_; }
    FaultHandler&   faults()       { return *faults_; }
    TmaEngine*      tma()          { return tma_.get(); }
    IlcManager*     ilc()          { return ilc_.get(); }
    ::nexusrt::memory::MemoryManager& memory();  // lazy-init

    PlatformContextHandle& platform_handle() { return plat; }

    // ---- Handle registry (for C ABI) ---------------------------------------
    static void register_handle(void* k, std::shared_ptr<FirmwareContext> sp);
    static std::shared_ptr<FirmwareContext> lookup_handle(void* k);
    static std::shared_ptr<FirmwareContext> take_handle(void* k);

public:
    const DeviceDesc  device;
    BootOptions       options;
    uint64_t          hbm_pool_bytes = 0;

    // Subsystems — populated by boot(). Order matters: destruction happens in
    // reverse declaration order, which mirrors the correct teardown sequence
    // (faults → mk → dma → ilc → plat).
    std::shared_ptr<MicroKernel>   mk_;
    std::shared_ptr<DmaEngine>     dma_;
    std::shared_ptr<TmaEngine>     tma_;     // nullptr on A100
    std::shared_ptr<FaultHandler>  faults_;
    std::shared_ptr<IlcManager>    ilc_;     // nullptr on A100
    std::shared_ptr<::nexusrt::memory::MemoryManager> mem_;  // lazy-init

    PlatformContextHandle plat;

private:
    static std::mutex& registry_mutex();
    static std::unordered_map<void*, std::shared_ptr<FirmwareContext>>& registry();
};

} // namespace firmware
} // namespace nexusrt
