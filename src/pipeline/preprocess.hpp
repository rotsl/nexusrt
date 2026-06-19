// =============================================================================
// NexusRT — src/pipeline/preprocess.hpp
// =============================================================================
#pragma once
#include "firmware/types.hpp"
#include "firmware/context.hpp"
#include <string>
namespace nexusrt { namespace pipeline {
class PreprocessStage {
public:
    explicit PreprocessStage(FirmwareContext& ctx);
    Status run(const std::string& corpus_path, uint64_t bytes_to_read,
               void* out_token_hbm, uint64_t out_capacity);
private:
    FirmwareContext& ctx_;
};
}} // namespace
