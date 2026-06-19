// =============================================================================
// NexusRT — src/platform/dispatch.cpp
// =============================================================================
#include "platform/dispatch.hpp"
#include "firmware/types.hpp"

namespace nexusrt {
namespace platform {

PlatformDispatch& PlatformDispatch::instance() {
    static PlatformDispatch d;
    return d;
}

void PlatformDispatch::register_backend(Vendor v,
                                        std::shared_ptr<PlatformInterface> impl) {
    std::lock_guard<std::mutex> g(mtx_);
    backends_[v] = std::move(impl);
}

bool PlatformDispatch::probe() {
    std::lock_guard<std::mutex> g(mtx_);
    if (active_) return true;
    // Priority order: NVIDIA first (data-center target), then Apple.
    for (Vendor v : {Vendor::Nvidia, Vendor::Apple}) {
        auto it = backends_.find(v);
        if (it == backends_.end()) continue;
        if (it->second->probe()) {
            active_ = it->second;
            return true;
        }
    }
    return false;
}

PlatformInterface& PlatformDispatch::backend() {
    if (!active_) probe();
    if (!active_) {
        // Should never happen — boot() would have failed earlier.
        static std::shared_ptr<PlatformInterface> null_impl;
        if (!null_impl) {
            struct NullPlatform : PlatformInterface {
                bool probe() override { return false; }
                DeviceDesc describe_device(const std::string&) override { return {}; }
                Status lock_clocks(const DeviceDesc&, FirmwareContext&) override { return Status::DeviceNotFound; }
                Status reserve_vaddress(FirmwareContext&, uint64_t) override { return Status::DeviceNotFound; }
                Status map_hbm_pool(FirmwareContext&) override { return Status::DeviceNotFound; }
                Status release_vaddress(FirmwareContext&) override { return Status::Ok; }
                Status alloc_hbm(FirmwareContext&, uint64_t, void*&) override { return Status::OutOfMemory; }
                Status free_hbm(FirmwareContext&, void*) override { return Status::Ok; }
                Status prefetch_range(FirmwareContext&, uint64_t, uint64_t) override { return Status::Ok; }
                Status create_stream(FirmwareContext&, StreamClass, int, void*&) override { return Status::DeviceNotFound; }
                Status destroy_stream(FirmwareContext&, void*) override { return Status::Ok; }
                Status sync_stream(FirmwareContext&, void*) override { return Status::Ok; }
                Status load_module(FirmwareContext&, const std::string&, const std::string&, KernelModule&) override { return Status::DeviceNotFound; }
                Status launch_kernel(FirmwareContext&, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*, void**, size_t) override { return Status::DeviceNotFound; }
                Status gds_init(FirmwareContext&) override { return Status::NotImplemented; }
                Status gds_fini(FirmwareContext&) override { return Status::Ok; }
                Status gds_read_fallback(FirmwareContext&, const std::string&, uint64_t, void*, uint64_t) override { return Status::IoError; }
                Status gds_write_fallback(FirmwareContext&, const std::string&, uint64_t, void*, uint64_t) override { return Status::IoError; }
                Status grdma_init(FirmwareContext&) override { return Status::NotImplemented; }
                Status grdma_fini(FirmwareContext&) override { return Status::Ok; }
                Status install_fault_buffer(FirmwareContext&, void*, uint32_t) override { return Status::Ok; }
                Status uninstall_fault_buffer(FirmwareContext&, void*) override { return Status::Ok; }
                Status resolve_fault(FirmwareContext&, uint64_t) override { return Status::Ok; }
                Status tma_encode(FirmwareContext&, TmaDescriptor&) override { return Status::NotImplemented; }
                Status tma_issue_copy(FirmwareContext&, TmaDescriptor const&, void*, uint64_t, uint64_t, void*) override { return Status::NotImplemented; }
                Status tma_release(FirmwareContext&, void*) override { return Status::Ok; }
                Status ilc_tag(FirmwareContext&, void*) override { return Status::NotImplemented; }
                Status ilc_untag(FirmwareContext&, void*) override { return Status::Ok; }
                Status dma_submit(FirmwareContext&, DmaRequest const&, void**) override { return Status::NotImplemented; }
                Status dma_wait(FirmwareContext&, void*, uint32_t) override { return Status::Ok; }
            };
            null_impl = std::make_shared<NullPlatform>();
        }
        return *null_impl;
    }
    return *active_;
}

} // namespace platform
} // namespace nexusrt
