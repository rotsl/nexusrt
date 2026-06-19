// =============================================================================
// NexusRT — src/token_opt/prefetcher.hpp
// Attention-weighted prefetcher.
//
// Maintains a sliding window of recent attention rows and predicts which KV
// slots the next decode step will attend to. Issues prefetch hints to the
// fault handler so the slots are paged into HBM before the compute kernel
// touches them.
// =============================================================================
#pragma once

#include "firmware/types.hpp"
#include "firmware/context.hpp"
#include "token_opt/scope.hpp"

#include <deque>
#include <mutex>
#include <vector>

namespace nexusrt {
namespace token_opt {

class AttentionPrefetcher {
public:
    AttentionPrefetcher(FirmwareContext& ctx, ContextScope& scope)
        : ctx_(ctx), scope_(scope) {}

    // Record an attention row (probabilities over KV slots). The prefetcher
    // keeps a sliding window and uses it to predict the next step.
    void observe(const std::vector<float>& attention_row);

    // Predict the top-k slots for the next step and issue prefetch hints.
    Status prefetch_next(uint32_t k);

    // History size (window length). Larger = more stable prediction but
    // higher memory cost.
    void set_history_window(uint32_t n) { history_window_ = n; }

private:
    FirmwareContext& ctx_;
    ContextScope&    scope_;
    std::deque<std::vector<float>> history_;
    uint32_t         history_window_ = 16;
    mutable std::mutex mtx_;
};

} // namespace token_opt
} // namespace nexusrt
