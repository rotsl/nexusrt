// =============================================================================
// NexusRT — src/pipeline/infer.hpp
// Inference stage: autoregressive decoding with KV-cache residency mgmt.
// =============================================================================
#pragma once
#include "firmware/types.hpp"
#include "firmware/context.hpp"
#include "token_opt/scope.hpp"
#include <vector>
namespace nexusrt { namespace pipeline {
class InferStage {
public:
    explicit InferStage(FirmwareContext& ctx) : ctx_(ctx), scope_(ctx) {}
    Status prefill (const std::vector<int32_t>& prompt);
    Status decode  (uint32_t max_new_tokens, std::vector<int32_t>& out);
    token_opt::ContextScope& scope() { return scope_; }
private:
    FirmwareContext&   ctx_;
    token_opt::ContextScope scope_;
};
}} // namespace
