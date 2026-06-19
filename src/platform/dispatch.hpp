// =============================================================================
// NexusRT — src/platform/dispatch.hpp
// PlatformDispatch — runtime hardware dispatch (NVIDIA CUDA vs Apple Metal).
//
// Each PlatformInterface implementation is registered at process start by a
// static initializer in the corresponding shim file (cuda_driver_shim.cu /
// metal_shim.mm). At boot time, PlatformDispatch::probe() tries each
// registered backend in priority order and selects the first one that
// responds.
//
// The dispatch layer is *the only* place where NexusRT chooses between
// NVIDIA and Apple paths. All higher layers (firmware, memory, scheduler,
// pipeline, token_opt) are platform-neutral and call into the dispatch
// table.
// =============================================================================
#pragma once

#include "firmware/types.hpp"
#include "firmware/context.hpp"
#include "firmware/microkernel.hpp"
#include "firmware/dma_engine.hpp"
#include "firmware/tma_engine.hpp"

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace nexusrt {
namespace platform {

using firmware::DmaRequest;
using firmware::KernelModule;
using firmware::StreamHandle;
using firmware::TmaDescriptor;

// ---------------------------------------------------------------------------
// PlatformInterface — pure virtual API implemented by each backend.
// ---------------------------------------------------------------------------
class PlatformInterface {
public:
    virtual ~PlatformInterface() = default;

    // ---- Detection --------------------------------------------------------
    virtual bool probe() = 0;
    virtual DeviceDesc describe_device(const std::string& profile) = 0;
    virtual Status lock_clocks(const DeviceDesc& d, FirmwareContext& ctx) = 0;

    // ---- Memory -----------------------------------------------------------
    virtual Status reserve_vaddress(FirmwareContext& ctx, uint64_t bytes) = 0;
    virtual Status map_hbm_pool(FirmwareContext& ctx) = 0;
    virtual Status release_vaddress(FirmwareContext& ctx) = 0;
    virtual Status alloc_hbm(FirmwareContext& ctx, uint64_t bytes, void*& out) = 0;
    virtual Status free_hbm(FirmwareContext& ctx, void* p) = 0;
    virtual Status prefetch_range(FirmwareContext& ctx,
                                  uint64_t vaddr, uint64_t bytes) = 0;

    // ---- Streams ----------------------------------------------------------
    virtual Status create_stream(FirmwareContext& ctx, StreamClass klass,
                                 int priority, void*& out) = 0;
    virtual Status destroy_stream(FirmwareContext& ctx, void* stream) = 0;
    virtual Status sync_stream(FirmwareContext& ctx, void* stream) = 0;

    // ---- Modules / kernels -----------------------------------------------
    virtual Status load_module(FirmwareContext& ctx, const std::string& name,
                               const std::string& src, KernelModule& out) = 0;
    virtual Status launch_kernel(FirmwareContext& ctx, void* fn,
                                 uint32_t gx, uint32_t gy, uint32_t gz,
                                 uint32_t bx, uint32_t by, uint32_t bz,
                                 uint32_t smem, void* stream,
                                 void** args, size_t n_args) = 0;

    // ---- GDS / GRDMA ------------------------------------------------------
    virtual Status gds_init(FirmwareContext& ctx) = 0;
    virtual Status gds_fini(FirmwareContext& ctx) = 0;
    virtual Status gds_read_fallback(FirmwareContext& ctx, const std::string& path,
                                     uint64_t off, void* dst, uint64_t bytes) = 0;
    virtual Status gds_write_fallback(FirmwareContext& ctx, const std::string& path,
                                      uint64_t off, void* src, uint64_t bytes) = 0;
    virtual Status grdma_init(FirmwareContext& ctx) = 0;
    virtual Status grdma_fini(FirmwareContext& ctx) = 0;

    // ---- Fault routing ----------------------------------------------------
    virtual Status install_fault_buffer(FirmwareContext& ctx, void* buf,
                                        uint32_t slots) = 0;
    virtual Status uninstall_fault_buffer(FirmwareContext& ctx, void* buf) = 0;
    virtual Status resolve_fault(FirmwareContext& ctx, uint64_t vaddr) = 0;

    // ---- TMA (H100) ------------------------------------------------------
    virtual Status tma_encode(FirmwareContext& ctx, TmaDescriptor& d) = 0;
    virtual Status tma_issue_copy(FirmwareContext& ctx, TmaDescriptor const& d,
                                  void* smem_dst, uint64_t off_x, uint64_t off_y,
                                  void* stream) = 0;
    virtual Status tma_release(FirmwareContext& ctx, void* raw) = 0;

    // ---- ILC (H100) ------------------------------------------------------
    virtual Status ilc_tag(FirmwareContext& ctx, void* alloc) = 0;
    virtual Status ilc_untag(FirmwareContext& ctx, void* alloc) = 0;

    // ---- DMA --------------------------------------------------------------
    virtual Status dma_submit(FirmwareContext& ctx, DmaRequest const& r,
                              void** out_event) = 0;
    virtual Status dma_wait(FirmwareContext& ctx, void* ev, uint32_t ms) = 0;
};

// ---------------------------------------------------------------------------
// PlatformDispatch — singleton front door.
// ---------------------------------------------------------------------------
class PlatformDispatch {
public:
    static PlatformDispatch& instance();

    // Registration — called by static initializers in each shim.
    void register_backend(Vendor v, std::shared_ptr<PlatformInterface> impl);

    // Probe all registered backends; cache the active one.
    bool probe();

    // Selected backend (post-probe).
    PlatformInterface& backend();

    // Convenience dispatch — forwards every call to the selected backend.
    bool probe_devices() { return probe(); }

    DeviceDesc describe_device(const std::string& profile) {
        return backend().describe_device(profile);
    }
    Status lock_clocks(const DeviceDesc& d, FirmwareContext& ctx) {
        return backend().lock_clocks(d, ctx);
    }
    Status reserve_vaddress(FirmwareContext& ctx, uint64_t bytes) {
        return backend().reserve_vaddress(ctx, bytes);
    }
    Status map_hbm_pool(FirmwareContext& ctx) {
        return backend().map_hbm_pool(ctx);
    }
    Status release_vaddress(FirmwareContext& ctx) {
        return backend().release_vaddress(ctx);
    }
    Status alloc_hbm(FirmwareContext& ctx, uint64_t bytes, void*& out) {
        return backend().alloc_hbm(ctx, bytes, out);
    }
    Status free_hbm(FirmwareContext& ctx, void* p) {
        return backend().free_hbm(ctx, p);
    }
    Status prefetch_range(FirmwareContext& ctx, uint64_t v, uint64_t b) {
        return backend().prefetch_range(ctx, v, b);
    }
    Status create_stream(FirmwareContext& ctx, StreamClass k, int p, void*& o) {
        return backend().create_stream(ctx, k, p, o);
    }
    Status destroy_stream(FirmwareContext& ctx, void* s) {
        return backend().destroy_stream(ctx, s);
    }
    Status sync_stream(FirmwareContext& ctx, void* s) {
        return backend().sync_stream(ctx, s);
    }
    Status load_module(FirmwareContext& ctx, const std::string& n,
                       const std::string& s, KernelModule& o) {
        return backend().load_module(ctx, n, s, o);
    }
    Status launch_kernel(FirmwareContext& ctx, void* fn,
                         uint32_t gx, uint32_t gy, uint32_t gz,
                         uint32_t bx, uint32_t by, uint32_t bz,
                         uint32_t smem, void* stream,
                         void** args, size_t n_args) {
        return backend().launch_kernel(ctx, fn, gx, gy, gz, bx, by, bz,
                                       smem, stream, args, n_args);
    }
    Status gds_init(FirmwareContext& ctx) { return backend().gds_init(ctx); }
    Status gds_fini(FirmwareContext& ctx) { return backend().gds_fini(ctx); }
    Status gds_read_fallback(FirmwareContext& ctx, const std::string& path,
                             uint64_t off, void* dst, uint64_t bytes) {
        return backend().gds_read_fallback(ctx, path, off, dst, bytes);
    }
    Status gds_write_fallback(FirmwareContext& ctx, const std::string& path,
                              uint64_t off, void* src, uint64_t bytes) {
        return backend().gds_write_fallback(ctx, path, off, src, bytes);
    }
    Status grdma_init(FirmwareContext& ctx) { return backend().grdma_init(ctx); }
    Status grdma_fini(FirmwareContext& ctx) { return backend().grdma_fini(ctx); }
    Status install_fault_buffer(FirmwareContext& ctx, void* buf, uint32_t slots) {
        return backend().install_fault_buffer(ctx, buf, slots);
    }
    Status uninstall_fault_buffer(FirmwareContext& ctx, void* buf) {
        return backend().uninstall_fault_buffer(ctx, buf);
    }
    Status resolve_fault(FirmwareContext& ctx, uint64_t vaddr) {
        return backend().resolve_fault(ctx, vaddr);
    }
    Status tma_encode(FirmwareContext& ctx, TmaDescriptor& d) {
        return backend().tma_encode(ctx, d);
    }
    Status tma_issue_copy(FirmwareContext& ctx, TmaDescriptor const& d,
                          void* smem, uint64_t ox, uint64_t oy, void* stream) {
        return backend().tma_issue_copy(ctx, d, smem, ox, oy, stream);
    }
    Status tma_release(FirmwareContext& ctx, void* raw) {
        return backend().tma_release(ctx, raw);
    }
    Status ilc_tag(FirmwareContext& ctx, void* a)   { return backend().ilc_tag(ctx, a); }
    Status ilc_untag(FirmwareContext& ctx, void* a) { return backend().ilc_untag(ctx, a); }
    Status dma_submit(FirmwareContext& ctx, DmaRequest const& r, void** ev) {
        return backend().dma_submit(ctx, r, ev);
    }
    Status dma_wait(FirmwareContext& ctx, void* ev, uint32_t ms) {
        return backend().dma_wait(ctx, ev, ms);
    }

private:
    PlatformDispatch() = default;
    std::mutex mtx_;
    std::map<Vendor, std::shared_ptr<PlatformInterface>> backends_;
    std::shared_ptr<PlatformInterface> active_;
};

} // namespace platform
} // namespace nexusrt
