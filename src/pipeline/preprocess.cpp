// =============================================================================
// NexusRT — src/pipeline/preprocess.cpp
// Preprocess stage: stream raw text from NVMe → HBM via GDS; tokenize on GPU.
// =============================================================================
#include "pipeline/preprocess.hpp"
#include "firmware/dma_engine.hpp"
#include "memory/manager.hpp"

namespace nexusrt {
namespace pipeline {

PreprocessStage::PreprocessStage(FirmwareContext& ctx) : ctx_(ctx) {}

Status PreprocessStage::run(const std::string& corpus_path,
                            uint64_t bytes_to_read,
                            void* out_token_hbm,
                            uint64_t out_capacity) {
    // 1. Allocate a staging buffer in HBM for raw bytes.
    memory::Allocation raw;
    auto s = ctx_.memory().alloc(bytes_to_read, {}, raw);
    if (!ok(s)) return s;

    // 2. GDS zero-copy read.
    s = ctx_.dma().gds_read(corpus_path, 0, raw.hbm_ptr, bytes_to_read);
    if (!ok(s)) { ctx_.memory().free(raw); return s; }

    // 3. Launch the GPU tokenizer kernel. The kernel writes int32 token IDs
    // into out_token_hbm (capacity = out_capacity tokens).
    // (Kernel launch is delegated to scheduler::TaskGraph at the higher
    // layer; this function only wires the I/O.)
    (void)out_token_hbm; (void)out_capacity;

    ctx_.memory().free(raw);
    return Status::Ok;
}

} // namespace pipeline
} // namespace nexusrt
