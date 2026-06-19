// =============================================================================
// NexusRT — src/firmware/boot.hpp
// Bare-metal-equivalent boot sequence.
//
// The boot sequence emulates what vendor GPU firmware does at power-on:
//   clock config → HBM controller init → DMA engine registration →
//   scheduler spawn → fault handler attach.
//
// On NexusRT this is implemented *above* the CUDA Driver API; we never modify
// vendor firmware. See docs/architecture.md §"Firmware Boundary".
// =============================================================================
#pragma once

#include "firmware/types.hpp"
#include <memory>

namespace nexusrt {
namespace firmware {

// BootPhase — each phase is logged and timed for diagnostics.
enum class BootPhase : uint8_t {
    PreInit         = 0,
    DriverProbe     = 1,    // locate cuDriver / Metal device
    ClockConfig     = 2,    // query max clocks, lock via cuCtx
    HbmController   = 3,    // cuMemAddressReserve + cuMemMap setup
    DmaEngine       = 4,    // register GDS / GRDMA / TMA engines
    SchedulerSpawn  = 5,    // create stream pool + warp specialization queues
    FaultHandler    = 6,    // allocate HBM-resident fault buffer
    IlcInit         = 7,    // ILC enabled (H100 only)
    Ready           = 8,
};

// BootEvent — emitted for every phase transition; consumed by the metrics layer.
struct BootEvent {
    BootPhase              phase;
    std::chrono::steady_clock::time_point ts;
    Status                 status;
    std::string            detail;
};

// BootOptions — caller-supplied knobs for the boot sequence.
struct BootOptions {
    std::string  profile           = "auto";   // auto | a100 | h100 | m1pro
    bool         enforce_graph_capture = true;
    bool         enable_ilc        = true;     // auto-disabled if unsupported
    bool         enable_grdma      = true;
    bool         enable_gds        = true;
    bool         enable_tma        = true;     // auto-disabled if unsupported
    uint32_t     fault_buffer_slots = 4096;
    uint32_t     fault_buffer_poll_us = 100;
    uint32_t     stream_pool_compute = 8;
    uint32_t     stream_pool_dma     = 4;
    std::function<void(const BootEvent&)> on_event;
};

// BootResult — returned by nexusrt_firmware_init().
struct BootResult {
    Status                 status      = Status::Ok;
    DeviceDesc             device;
    std::vector<BootEvent> events;
    std::shared_ptr<FirmwareContext> ctx;     // opaque context, owned by caller
};

// Execute the bare-metal-equivalent boot sequence.
//
// This is the *only* entry point the rest of NexusRT calls to bring up the
// firmware-equivalent layer. After this returns successfully, the returned
// FirmwareContext owns:
//   - the primary CUDA context (or Metal device),
//   - the stream pool (compute / dma / fence),
//   - the HBM-resident fault buffer,
//   - the TMA descriptor cache (H100 only),
//   - the ILC manager (H100 only).
BootResult boot(const BootOptions& opts);

// Tear down — release all firmware-equivalent resources. Idempotent.
Status shutdown(FirmwareContext& ctx);

// Re-probe the device and refresh the DeviceDesc (used by tests).
DeviceDesc probe_device(const std::string& profile_hint);

} // namespace firmware
} // namespace nexusrt

// --- C ABI ------------------------------------------------------------------
extern "C" {
    // nexusrt_firmware_init — primary boot entry point.
    // Returns 0 on success, negative nexusrt::firmware::Status code on failure.
    // On success, *out_ctx is set to an opaque handle used by all other
    // firmware C ABI functions.
    int32_t nexusrt_firmware_init(const char* profile,
                                  void**     out_ctx);

    // nexusrt_firmware_shutdown — release the context.
    int32_t nexusrt_firmware_shutdown(void* ctx);

    // nexusrt_firmware_device_desc — copy out the discovered DeviceDesc.
    int32_t nexusrt_firmware_device_desc(void* ctx,
                                         int32_t*  out_vendor,
                                         int32_t*  out_arch,
                                         char*     out_name,       size_t name_cap,
                                         uint64_t* out_hbm_bytes,
                                         uint32_t* out_sm_count,
                                         uint32_t* out_features_bits);
}
