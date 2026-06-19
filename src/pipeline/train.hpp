// =============================================================================
// NexusRT — src/pipeline/train.hpp
// Training stage: forward + backward pass, gradient sync via NCCL/GRDMA.
// =============================================================================
#pragma once
#include "firmware/types.hpp"
#include "firmware/context.hpp"
namespace nexusrt { namespace pipeline {
class TrainStage {
public:
    explicit TrainStage(FirmwareContext& ctx) : ctx_(ctx) {}
    Status step(uint32_t micro_batch, uint32_t seq_len);
    Status sync_gradients(uint32_t n_ranks);
private:
    FirmwareContext& ctx_;
};
}} // namespace
