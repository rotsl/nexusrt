// =============================================================================
// NexusRT — src/firmware/types.hpp
// Core shared types for the firmware-equivalent layer.
//
// All public types live in `nexusrt::firmware` and are exposed through the
// unified C ABI in src/platform/abi.h.
// =============================================================================
#pragma once

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <chrono>
#include <mutex>
#include <condition_variable>

namespace nexusrt {
namespace firmware {

// ---------------------------------------------------------------------------
// Vendor / architecture identification
// ---------------------------------------------------------------------------
enum class Vendor : uint8_t {
    Unknown = 0,
    Nvidia  = 1,
    Apple   = 2,
};

enum class Arch : uint8_t {
    Unknown = 0,
    Ampere  = 1,    // A100
    Hopper  = 2,    // H100
    AppleSilicon = 3,
    Pascal  = 4,    // P100
    Turing  = 5,    // T4
};

// Feature flags discovered at init time. These drive conditional compilation
// of code paths (TMA, ILC, DSM, GDS, GRDMA).
struct FeatureFlags {
    bool tma        = false;   // Tensor Memory Accelerator (Hopper only)
    bool ilc        = false;   // Inline Compression (Hopper only)
    bool dsm        = false;   // Distributed Shared Memory via TB clusters
    bool gds        = false;   // GPUDirect Storage
    bool grdma      = false;   // GPUDirect RDMA
    bool clusters   = false;   // Thread Block Clusters
    bool unified_native = false; // Apple unified memory (no copy needed)
};

// ---------------------------------------------------------------------------
// Device descriptor — populated at boot by `nexusrt_firmware_init`.
// ---------------------------------------------------------------------------
struct DeviceDesc {
    Vendor         vendor          = Vendor::Unknown;
    Arch           arch            = Arch::Unknown;
    std::string    name;                       // "NVIDIA H100 SXM5", "Apple M1 Pro", ...
    int            compute_capability_major = 0;
    int            compute_capability_minor = 0;
    uint64_t       hbm_capacity_bytes = 0;
    uint64_t       hbm_bandwidth_bps  = 0;
    uint32_t       sm_count           = 0;
    uint32_t       smem_per_sm_bytes  = 0;
    uint32_t       nvlink_bandwidth_gbs = 0;
    FeatureFlags   features;
};

// ---------------------------------------------------------------------------
// Stream priorities — mirrors CUDA stream priorities.
// ---------------------------------------------------------------------------
enum class StreamClass : uint8_t {
    Compute = 0,    // SM-bound work
    Dma     = 1,    // GDS / GRDMA / TMA copy
    Fence   = 2,    // doorbell writers / mbarrier arrievement
    Control = 3,    // host-side orchestration
};

// ---------------------------------------------------------------------------
// Allocation hints — consumed by the memory layer.
// ---------------------------------------------------------------------------
struct AllocHints {
    bool        ilc           = false;  // request Inline Compression (H100)
    bool        read_mostly   = false;
    bool        pinned_host   = false;  // host-pinned, DMA-able
    bool        gds_readable  = false;  // mapped for GDS reads
    bool        grdma_visible = false;  // exposed for remote RDMA write
    std::string preferred_location = "device"; // device | host | numanode:N
};

// ---------------------------------------------------------------------------
// Result codes — non-zero is failure. Stable across ABI versions.
// ---------------------------------------------------------------------------
enum class Status : int32_t {
    Ok                  = 0,
    InvalidArgument     = -1,
    OutOfMemory         = -2,
    OutOfHbm            = -3,
    DeviceNotFound      = -4,
    DriverError         = -5,
    NotImplemented      = -6,
    FaultBufferOverflow = -7,
    ContractViolation   = -8,
    Timeout             = -9,
    IoError             = -10,
    Aborted             = -11,
};

inline bool ok(Status s) { return static_cast<int32_t>(s) >= 0; }
const char* status_string(Status s);

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
class MicroKernel;
class DmaEngine;
class TmaEngine;
class FaultHandler;
class IlcManager;
class FirmwareContext;

} // namespace firmware

using firmware::AllocHints;
using firmware::Arch;
using firmware::DeviceDesc;
using firmware::DmaEngine;
using firmware::FaultHandler;
using firmware::FirmwareContext;
using firmware::IlcManager;
using firmware::MicroKernel;
using firmware::Status;
using firmware::StreamClass;
using firmware::TmaEngine;
using firmware::Vendor;

} // namespace nexusrt
