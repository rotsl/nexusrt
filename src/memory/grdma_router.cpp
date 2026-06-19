// =============================================================================
// NexusRT — src/memory/grdma_router.cpp
// GPUDirect RDMA fetch router. Direct GPU↔NIC communication for multi-node
// training and cross-GPU context sharing.
//
// On systems with IBV + GPUDirect RDMA we register CUDA IPC handles with
// the verbs context. On systems without, we fall back to NCCL collectives
// (configured at runtime.yaml:scheduler.grdma.nccl_fallback).
// =============================================================================
#include "firmware/context.hpp"
#include "firmware/dma_engine.hpp"
#include "platform/dispatch.hpp"

namespace nexusrt {
namespace memory {

// Issue a GRDMA fetch from a remote GPU's memory. The remote buffer must
// have been exported via cudaIpcGetMemHandle and the handle transmitted out
// of band (e.g., via NCCL's bootstrap).
Status grdma_fetch(FirmwareContext& ctx,
                   void* /*remote_ipc_handle*/,
                   uint64_t /*remote_offset*/,
                   void*  /*local_hbm*/,
                   uint64_t /*bytes*/) {
    // Real implementation:
    //   1. cudaIpcOpenMemHandle(&local_ptr, *handle, cudaIpcMemLazyEnablePeerAccess)
    //   2. cuMemcpyDtoDAsync(local_hbm, local_ptr + offset, bytes, dma_stream)
    //   3. cudaIpcCloseMemHandle(local_ptr)
    // We elide the verbs registration details here.
    auto& dma = ctx.dma();
    (void)dma;
    return Status::NotImplemented;
}

// Multi-node all-reduce via NCCL (fallback when IBV/GRDMA is unavailable).
// The actual NCCL communicator is owned by the scheduler layer, which
// establishes it once per pipeline.
Status grdma_allreduce_fallback(FirmwareContext& /*ctx*/,
                                void* /*sendbuf*/, void* /*recvbuf*/,
                                uint64_t /*count*/, int /*dtype*/) {
    return Status::NotImplemented;
}

} // namespace memory
} // namespace nexusrt
