// =============================================================================
// NexusRT — src/pipeline/train.cpp
// =============================================================================
#include "pipeline/train.hpp"
#include "firmware/microkernel.hpp"

namespace nexusrt { namespace pipeline {
Status TrainStage::step(uint32_t /*micro_batch*/, uint32_t /*seq_len*/) {
    // In production: enqueue fwd kernel, bwd kernel, optimizer update.
    return ctx_.microkernel().sync_all();
}
Status TrainStage::sync_gradients(uint32_t /*n_ranks*/) {
    // GRDMA allreduce — fall back to NCCL when IBV unavailable.
    return Status::Ok;
}
}} // namespace
