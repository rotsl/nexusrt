// =============================================================================
// NexusRT — src/firmware/boot.cpp
// Bare-metal-equivalent boot sequence implementation.
//
// The boot sequence here emulates vendor GPU firmware stages (clock config,
// HBM controller init, DMA engine registration, scheduler spawn, fault
// handler attach) using only public CUDA Driver / Metal APIs. We never
// modify vendor firmware — see docs/architecture.md §"Firmware Boundary".
// =============================================================================
#include "firmware/boot.hpp"
#include "firmware/microkernel.hpp"
#include "firmware/dma_engine.hpp"
#include "firmware/tma_engine.hpp"
#include "firmware/fault_handler.hpp"
#include "firmware/ilc_manager.hpp"
#include "firmware/context.hpp"
#include "platform/dispatch.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace nexusrt {
namespace firmware {

namespace {

void emit(BootOptions const& opts, BootPhase p, Status s, const char* detail) {
    if (std::getenv("NEXUSRT_BOOT_TRACE")) {
        std::fprintf(stderr,
                     "NexusRT boot phase=%d status=%d detail=%s\n",
                     static_cast<int>(p), static_cast<int>(s),
                     detail ? detail : "");
    }
    if (!opts.on_event) return;
    BootEvent e;
    e.phase  = p;
    e.ts     = std::chrono::steady_clock::now();
    e.status = s;
    e.detail = detail ? detail : "";
    opts.on_event(e);
}

// -- Phase 1: driver probe ---------------------------------------------------
Status probe_driver(const BootOptions& opts, DeviceDesc& out) {
    auto& plat = platform::PlatformDispatch::instance();
    if (!plat.probe()) {
        return Status::DeviceNotFound;
    }
    out = plat.describe_device(opts.profile);
    if (out.vendor == Vendor::Unknown) {
        return Status::DeviceNotFound;
    }
    return Status::Ok;
}

// -- Phase 2: clock config ---------------------------------------------------
// On real bare-metal, this would write the SM / memory clock PLLs. On NexusRT
// we lock the clocks via the driver (cuCtx... / MTLDevice) to prevent DVFS
// jitter during benchmarks.
Status lock_clocks(DeviceDesc const& d, FirmwareContext& ctx) {
    return platform::PlatformDispatch::instance().lock_clocks(d, ctx);
}

// -- Phase 3: HBM controller init -------------------------------------------
// Reserve a virtual address range and pre-map the HBM pool. On H100 we also
// install an ILC allocator.
Status init_hbm(DeviceDesc const& d, BootOptions const& opts, FirmwareContext& ctx) {
    ctx.hbm_pool_bytes = std::min<uint64_t>(
        d.hbm_capacity_bytes,
        d.hbm_capacity_bytes * 85 / 100);     // 85% watermark
    auto s = platform::PlatformDispatch::instance()
                 .reserve_vaddress(ctx, ctx.hbm_pool_bytes);
    if (!ok(s)) return s;
    return platform::PlatformDispatch::instance().map_hbm_pool(ctx);
}

// -- Phase 4: DMA engine registration ---------------------------------------
Status register_dma(DeviceDesc const& d, BootOptions const& opts, FirmwareContext& ctx) {
    ctx.dma_ = std::make_shared<DmaEngine>(ctx, d);
    if (d.features.gds && opts.enable_gds) {
        auto s = ctx.dma_->enable_gds();
        if (!ok(s)) return s;
    }
    if (d.features.grdma && opts.enable_grdma) {
        auto s = ctx.dma_->enable_grdma();
        if (!ok(s)) return s;
    }
    return Status::Ok;
}

// -- Phase 5: scheduler spawn ------------------------------------------------
Status spawn_scheduler(DeviceDesc const& d, BootOptions const& opts, FirmwareContext& ctx) {
    ctx.mk_ = std::make_shared<MicroKernel>(ctx, d);
    return ctx.mk_->spawn_stream_pool(
        opts.stream_pool_compute, opts.stream_pool_dma);
}

// -- Phase 6: fault handler attach ------------------------------------------
Status attach_fault_handler(DeviceDesc const& d,
                            BootOptions const& opts,
                            FirmwareContext& ctx) {
    ctx.faults_ = std::make_shared<FaultHandler>(ctx, opts.fault_buffer_slots);
    return ctx.faults_->install();
}

// -- Phase 7: ILC init (H100 only) ------------------------------------------
Status init_ilc(DeviceDesc const& d, BootOptions const& opts, FirmwareContext& ctx) {
    if (!d.features.ilc || !opts.enable_ilc) {
        ctx.ilc_ = nullptr;
        return Status::Ok;
    }
    ctx.ilc_ = std::make_shared<IlcManager>(ctx);
    return ctx.ilc_->enable();
}

} // namespace

// =============================================================================
// Public boot API
// =============================================================================

BootResult boot(const BootOptions& opts) {
    BootResult r;
    DeviceDesc d;

    emit(opts, BootPhase::PreInit, Status::Ok, "boot started");

    // 1. driver probe
    {
        auto s = probe_driver(opts, d);
        emit(opts, BootPhase::DriverProbe, s,
             ok(s) ? "driver found" : "no driver");
        if (!ok(s)) { r.status = s; return r; }
    }

    r.ctx = std::make_shared<FirmwareContext>(d);
    r.ctx->options = opts;

    // 2. clock config
    {
        auto s = lock_clocks(d, *r.ctx);
        emit(opts, BootPhase::ClockConfig, s,
             ok(s) ? "clocks locked" : "clock lock failed (continuing)");
        // Clock-lock failure is non-fatal — we proceed.
    }

    // 3. HBM controller init
    {
        auto s = init_hbm(d, opts, *r.ctx);
        emit(opts, BootPhase::HbmController, s,
             ok(s) ? "HBM pool mapped" : "HBM init failed");
        if (!ok(s)) { r.status = s; return r; }
    }

    // 4. DMA engine registration
    {
        auto s = register_dma(d, opts, *r.ctx);
        emit(opts, BootPhase::DmaEngine, s,
             ok(s) ? "DMA engines up" : "DMA init failed");
        if (!ok(s)) { r.status = s; return r; }
    }

    // 5. scheduler spawn
    {
        auto s = spawn_scheduler(d, opts, *r.ctx);
        emit(opts, BootPhase::SchedulerSpawn, s,
             ok(s) ? "stream pool ready" : "scheduler spawn failed");
        if (!ok(s)) { r.status = s; return r; }
    }

    // 6. fault handler attach
    {
        auto s = attach_fault_handler(d, opts, *r.ctx);
        emit(opts, BootPhase::FaultHandler, s,
             ok(s) ? "fault buffer installed" : "fault handler failed");
        if (!ok(s)) { r.status = s; return r; }
    }

    // 7. ILC init
    {
        auto s = init_ilc(d, opts, *r.ctx);
        emit(opts, BootPhase::IlcInit, s,
             ok(s) ? "ILC ready" : "ILC skipped");
        // ILC failure is non-fatal — we proceed without compression.
    }

    emit(opts, BootPhase::Ready, Status::Ok, "firmware-equivalent layer ready");
    r.status = Status::Ok;
    r.device = d;
    return r;
}

Status shutdown(FirmwareContext& ctx) {
    Status last = Status::Ok;
    if (ctx.faults_) last = ctx.faults_->uninstall();
    if (ctx.mk_)     ctx.mk_->drain_streams();
    if (ctx.dma_)    ctx.dma_->shutdown();
    if (ctx.ilc_)    ctx.ilc_->disable();
    auto s = platform::PlatformDispatch::instance().release_vaddress(ctx);
    if (!ok(s)) last = s;
    return last;
}

DeviceDesc probe_device(const std::string& profile_hint) {
    BootOptions o; o.profile = profile_hint;
    DeviceDesc d;
    auto s = probe_driver(o, d);
    if (!ok(s)) d.vendor = Vendor::Unknown;
    return d;
}

const char* status_string(Status s) {
    switch (s) {
        case Status::Ok:                  return "ok";
        case Status::InvalidArgument:     return "invalid argument";
        case Status::OutOfMemory:         return "out of memory";
        case Status::OutOfHbm:            return "out of HBM";
        case Status::DeviceNotFound:      return "device not found";
        case Status::DriverError:         return "driver error";
        case Status::NotImplemented:      return "not implemented";
        case Status::FaultBufferOverflow: return "fault buffer overflow";
        case Status::ContractViolation:   return "contract violation";
        case Status::Timeout:             return "timeout";
        case Status::IoError:             return "I/O error";
        case Status::Aborted:             return "aborted";
    }
    return "unknown";
}

} // namespace firmware
} // namespace nexusrt

// =============================================================================
// C ABI
// =============================================================================
#include "platform/abi.h"
extern "C" {

NEXUSRT_API int32_t nexusrt_firmware_init(const char* profile, void** out_ctx) {
    if (!out_ctx) return static_cast<int32_t>(nexusrt::firmware::Status::InvalidArgument);
    nexusrt::firmware::BootOptions opts;
    if (profile) opts.profile = profile;
    auto r = nexusrt::firmware::boot(opts);
    if (!nexusrt::firmware::ok(r.status)) {
        return static_cast<int32_t>(r.status);
    }
    // The opaque handle is the FirmwareContext raw pointer. Lifetime is owned
    // by the caller via nexusrt_firmware_shutdown().
    *out_ctx = r.ctx.get();
    // Stash the shared_ptr in a static registry so it survives the call.
    nexusrt::firmware::FirmwareContext::register_handle(*out_ctx, r.ctx);
    return 0;
}

NEXUSRT_API int32_t nexusrt_firmware_shutdown(void* ctx) {
    if (!ctx) return static_cast<int32_t>(nexusrt::firmware::Status::InvalidArgument);
    auto sp = nexusrt::firmware::FirmwareContext::take_handle(ctx);
    if (!sp) return static_cast<int32_t>(nexusrt::firmware::Status::InvalidArgument);
    auto s = nexusrt::firmware::shutdown(*sp);
    return static_cast<int32_t>(s);
}

NEXUSRT_API int32_t nexusrt_firmware_device_desc(void* ctx,
                                     int32_t*  out_vendor,
                                     int32_t*  out_arch,
                                     char*     out_name,
                                     size_t    name_cap,
                                     uint64_t* out_hbm_bytes,
                                     uint32_t* out_sm_count,
                                     uint32_t* out_features_bits) {
    auto sp = nexusrt::firmware::FirmwareContext::lookup_handle(ctx);
    if (!sp) return static_cast<int32_t>(nexusrt::firmware::Status::InvalidArgument);
    auto const& d = sp->device;
    if (out_vendor)        *out_vendor       = static_cast<int32_t>(d.vendor);
    if (out_arch)          *out_arch         = static_cast<int32_t>(d.arch);
    if (out_name && name_cap) {
        std::strncpy(out_name, d.name.c_str(), name_cap - 1);
        out_name[name_cap - 1] = '\0';
    }
    if (out_hbm_bytes)     *out_hbm_bytes    = d.hbm_capacity_bytes;
    if (out_sm_count)      *out_sm_count     = d.sm_count;
    if (out_features_bits) {
        uint32_t b = 0;
        if (d.features.tma)        b |= 1u << 0;
        if (d.features.ilc)        b |= 1u << 1;
        if (d.features.dsm)        b |= 1u << 2;
        if (d.features.gds)        b |= 1u << 3;
        if (d.features.grdma)      b |= 1u << 4;
        if (d.features.clusters)   b |= 1u << 5;
        if (d.features.unified_native) b |= 1u << 6;
        *out_features_bits = b;
    }
    return 0;
}

} // extern "C"
