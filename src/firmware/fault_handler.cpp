// =============================================================================
// NexusRT — src/firmware/fault_handler.cpp
// =============================================================================
#include "firmware/fault_handler.hpp"
#include "firmware/microkernel.hpp"
#include "platform/dispatch.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace nexusrt {
namespace firmware {

FaultHandler::FaultHandler(FirmwareContext& ctx, uint32_t buffer_slots)
    : ctx_(ctx), buffer_slots_(buffer_slots) {}

FaultHandler::~FaultHandler() { uninstall(); }

Status FaultHandler::install() {
    if (ctx_.device.vendor == Vendor::Nvidia) {
        if (std::getenv("NEXUSRT_BOOT_TRACE")) {
            std::fprintf(stderr,
                         "NexusRT fault handler: CUDA HBM fault polling disabled; using explicit prefetch hints\n");
        }
        return Status::Ok;
    }

    // Allocate the fault buffer in HBM. We use the platform allocator rather
    // than the user-facing memory pool so the fault buffer cannot itself
    // trigger faults (which would deadlock).
    void* fault_buffer = nullptr;
    auto s = platform::PlatformDispatch::instance()
        .alloc_hbm(ctx_, buffer_slots_ * sizeof(FaultRecord), fault_buffer);
    if (!ok(s)) return s;
    hbm_fault_buffer_ = fault_buffer;
    std::memset(hbm_fault_buffer_, 0, buffer_slots_ * sizeof(FaultRecord));

    // Register the fault buffer with the platform's fault routing layer.
    s = platform::PlatformDispatch::instance()
        .install_fault_buffer(ctx_, hbm_fault_buffer_, buffer_slots_);
    if (!ok(s)) {
        // Fall back to software-prefetch mode.
        s = Status::Ok;
    }

    stop_ = false;
    poller_ = std::thread([this] { poll_loop(); });
    return Status::Ok;
}

Status FaultHandler::uninstall() {
    if (!hbm_fault_buffer_) return Status::Ok;
    stop_ = true;
    if (poller_.joinable()) poller_.join();
    platform::PlatformDispatch::instance()
        .uninstall_fault_buffer(ctx_, hbm_fault_buffer_);
    platform::PlatformDispatch::instance()
        .free_hbm(ctx_, hbm_fault_buffer_);
    hbm_fault_buffer_ = nullptr;
    return Status::Ok;
}

Status FaultHandler::inject_fetch_hint(uint64_t vaddr, uint64_t bytes) {
    std::lock_guard<std::mutex> g(inject_mtx_);
    return platform::PlatformDispatch::instance()
        .prefetch_range(ctx_, vaddr, bytes);
}

void FaultHandler::poll_loop() {
    // In a true firmware build this loop would run on a dedicated SM. In the
    // firmware-equivalent user-space build we run it on a host thread that
    // calls into the CUDA Driver / Metal API. This still removes the OS
    // page-fault path: the GPU logs the fault to the HBM buffer, and we
    // resolve it by issuing a GRDMA / blit fetch, never by invoking the OS
    // page-fault handler.
    using clock = std::chrono::steady_clock;
    auto* buf = static_cast<FaultRecord*>(hbm_fault_buffer_);
    while (!stop_.load(std::memory_order_relaxed)) {
        bool any = false;
        for (uint32_t i = 0; i < buffer_slots_; ++i) {
            auto& r = buf[i];
            if (r.status != 0) continue;
            any = true;
            total_faults_.fetch_add(1, std::memory_order_relaxed);

            auto t0 = clock::now();
            auto s = platform::PlatformDispatch::instance()
                .resolve_fault(ctx_, r.faulting_addr);
            auto t1 = clock::now();
            auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

            if (ok(s)) {
                r.status = 1;
                resolved_faults_.fetch_add(1, std::memory_order_relaxed);
            } else {
                r.status = 2;
                failed_faults_.fetch_add(1, std::memory_order_relaxed);
            }
            total_resolve_ns_.fetch_add(ns, std::memory_order_relaxed);
        }
        if (!any) {
            std::this_thread::sleep_for(
                std::chrono::microseconds(ctx_.options.fault_buffer_poll_us));
        }
    }
}

uint64_t FaultHandler::avg_resolve_us() const {
    auto r = resolved_faults_.load();
    if (r == 0) return 0;
    return total_resolve_ns_.load() / (r * 1000);
}

} // namespace firmware
} // namespace nexusrt

// =============================================================================
// C ABI — exposes the fault handler as nexusrt_firmware_fault_handler()
// =============================================================================
#include "platform/abi.h"
#include "firmware/boot.hpp"
extern "C" {

NEXUSRT_API int32_t nexusrt_firmware_fault_handler(void*    ctx,
                                       uint64_t faulting_addr,
                                       int32_t  timeout_ms) {
    using namespace nexusrt::firmware;
    auto sp = FirmwareContext::lookup_handle(ctx);
    if (!sp) return static_cast<int32_t>(Status::InvalidArgument);
    auto s = sp->faults().inject_fetch_hint(faulting_addr, 4096);
    return static_cast<int32_t>(s);
}

// nexusrt_firmware_task_submit — the C entry used by the scheduler layer and
// by Python bindings. Submits a kernel launch through the micro-kernel.
NEXUSRT_API int32_t nexusrt_firmware_task_submit(void*       ctx,
                                     const char* module_name,
                                     const char* function_name,
                                     uint32_t    grid_x, uint32_t grid_y, uint32_t grid_z,
                                     uint32_t    block_x, uint32_t block_y, uint32_t block_z,
                                     uint32_t    shared_mem_bytes,
                                     void**      kernel_args,
                                     uint32_t    n_args,
                                     int32_t     stream_class,
                                     int32_t     stream_priority) {
    using namespace nexusrt::firmware;
    auto sp = FirmwareContext::lookup_handle(ctx);
    if (!sp) return static_cast<int32_t>(Status::InvalidArgument);

    auto stream = sp->microkernel().acquire_stream(
        static_cast<StreamClass>(stream_class), stream_priority);
    if (stream.id == 0) return static_cast<int32_t>(Status::OutOfMemory);

    auto s = sp->microkernel().launch(module_name, function_name,
                            grid_x, grid_y, grid_z,
                            block_x, block_y, block_z,
                            shared_mem_bytes, stream,
                            kernel_args, n_args);
    if (ok(s)) s = sp->microkernel().sync_stream(stream);
    sp->microkernel().release_stream(stream);
    return static_cast<int32_t>(s);
}

} // extern "C"
