// =============================================================================
// NexusRT — src/scheduler/warp_specialization.hpp
// Warp specialization policies for the scheduler.
//
// Warp specialization dedicates warps in a CTA to specific roles:
//   producer  — issues TMA / cp.async copies into shared memory
//   consumer  — computes on the data once it arrives
//   fence     — arrives at mbarriers to signal data readiness (H100 only)
//
// On Pascal (P100), Turing (T4), and Ampere (A100) we use a generic
// CUDA stream policy. A100 may use cp.async in specialized kernels.
// On Hopper (H100) we use 2 producer + 14 consumer warps with TMA bulk.
// On Apple Silicon there is no warp specialization; we map to threadgroup
// tiles instead.
// =============================================================================
#pragma once

#include "firmware/types.hpp"
#include "firmware/context.hpp"

namespace nexusrt {
namespace scheduler {

struct WarpSpecPolicy {
    uint32_t producer_warps = 2;
    uint32_t consumer_warps = 14;
    uint32_t fence_warps    = 0;       // H100 only
    uint32_t threads_per_warp = 32;
    bool     use_tma          = false; // H100 only
    bool     use_dsm          = false; // H100 only — Thread Block Clusters
    uint32_t cluster_size     = 1;     // valid when use_dsm

    static WarpSpecPolicy for_arch(Arch a) {
        WarpSpecPolicy p;
        switch (a) {
            case Arch::Hopper:
                p.producer_warps = 2;
                p.consumer_warps = 14;
                p.fence_warps    = 0;
                p.use_tma        = true;
                p.use_dsm        = true;
                p.cluster_size   = 8;
                break;
            case Arch::Ampere:
            case Arch::Pascal:
            case Arch::Turing:
                p.producer_warps = 2;
                p.consumer_warps = 14;
                p.fence_warps    = 0;
                p.use_tma        = false;
                p.use_dsm        = false;
                break;
            case Arch::AppleSilicon:
                p.producer_warps = 0;
                p.consumer_warps = 16;
                p.threads_per_warp = 32;
                break;
            default: break;
        }
        return p;
    }

    uint32_t total_warps() const {
        return producer_warps + consumer_warps + fence_warps;
    }
    uint32_t total_threads() const {
        return total_warps() * threads_per_warp;
    }
};

} // namespace scheduler
} // namespace nexusrt
