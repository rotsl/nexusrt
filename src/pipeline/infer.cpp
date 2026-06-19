// =============================================================================
// NexusRT — src/pipeline/infer.cpp
// =============================================================================
#include "pipeline/infer.hpp"

namespace nexusrt { namespace pipeline {
Status InferStage::prefill(const std::vector<int32_t>& /*prompt*/) {
    scope_.step(0);
    return Status::Ok;
}
Status InferStage::decode(uint32_t max_new_tokens,
                          std::vector<int32_t>& out) {
    out.clear();
    for (uint32_t i = 0; i < max_new_tokens; ++i) {
        scope_.step(i + 1);
        out.push_back(static_cast<int32_t>(i));
    }
    return Status::Ok;
}
}} // namespace
