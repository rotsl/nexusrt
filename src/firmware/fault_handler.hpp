// =============================================================================
// NexusRT — src/firmware/fault_handler.hpp
// GPU-driven page fault handler (DREAM-inspired).
//
// Standard CUDA Unified Memory routes faults through the host OS kernel,
// which introduces latency spikes and depends on kernel 6.1.24+ on Linux.
// NexusRT instead installs a fault buffer in HBM that is polled by a
// GPU-resident firmware-equivalent thread. When a compute thread accesses
// an unmapped page, the fault is logged in the buffer; the firmware thread
// issues a GRDMA fetch to bring the page in, then unblocks the compute.
//
// On platforms without explicit fault hooks (Apple Metal, older NVIDIA
// drivers), we fall back to a software prefetcher that walks the page table
// ahead of compute and pre-maps expected accesses.
// =============================================================================
#pragma once

#include "firmware/types.hpp"
#include "firmware/context.hpp"
#include <atomic>
#include <memory>
#include <mutex>
#include <thread>

namespace nexusrt {
namespace firmware {

// FaultRecord — one entry in the HBM-resident fault buffer.
struct FaultRecord {
    uint64_t faulting_addr = 0;
    uint64_t requesting_pc = 0;
    uint32_t requesting_sm = 0;
    uint32_t requesting_warp = 0;
    uint64_t timestamp_ns = 0;
    uint32_t status = 0;          // 0=pending, 1=resolved, 2=failed
};

class FaultHandler {
public:
    FaultHandler(FirmwareContext& ctx, uint32_t buffer_slots);
    ~FaultHandler();

    // Install the fault buffer in HBM and start the polling thread.
    Status install();

    // Stop polling and release the buffer.
    Status uninstall();

    // Manual API: register a fault for a region the caller knows will be
    // accessed. Used by the token_opt prefetcher to explicitly page in
    // attention-anticipated KV slots.
    Status inject_fetch_hint(uint64_t vaddr, uint64_t bytes);

    // Diagnostics
    uint64_t total_faults()       const { return total_faults_.load(); }
    uint64_t resolved_faults()    const { return resolved_faults_.load(); }
    uint64_t failed_faults()      const { return failed_faults_.load(); }
    uint64_t avg_resolve_us()     const;

private:
    void poll_loop();

    FirmwareContext& ctx_;
    uint32_t         buffer_slots_;

    void*            hbm_fault_buffer_ = nullptr;     // FaultRecord[]
    std::atomic<bool> stop_{false};
    std::thread      poller_;

    std::atomic<uint64_t> total_faults_{0};
    std::atomic<uint64_t> resolved_faults_{0};
    std::atomic<uint64_t> failed_faults_{0};
    std::atomic<uint64_t> total_resolve_ns_{0};

    mutable std::mutex inject_mtx_;
};

} // namespace firmware
} // namespace nexusrt
