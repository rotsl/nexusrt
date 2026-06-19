// =============================================================================
// NexusRT — src/pipeline/postprocess.hpp
// Detokenize + format; copy to host only on final answer.
// =============================================================================
#pragma once
#include "firmware/types.hpp"
#include "firmware/context.hpp"
#include <string>
#include <vector>
namespace nexusrt { namespace pipeline {
class PostprocessStage {
public:
    explicit PostprocessStage(FirmwareContext& ctx) : ctx_(ctx) {}
    Status run(const std::vector<int32_t>& tokens, std::string& out_text,
               bool copy_to_host = true);
private:
    FirmwareContext& ctx_;
};
}} // namespace
