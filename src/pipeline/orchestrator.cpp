// =============================================================================
// NexusRT — src/pipeline/orchestrator.cpp
// =============================================================================
#include "pipeline/orchestrator.hpp"
#include "firmware/dma_engine.hpp"
#include "firmware/microkernel.hpp"
#include "memory/manager.hpp"
#include "platform/dispatch.hpp"

#include <chrono>
#include <cstring>
#include <sstream>

namespace nexusrt {
namespace pipeline {

Pipeline::Pipeline(FirmwareContext& ctx, PipelineConfig const& cfg)
    : ctx_(ctx), cfg_(cfg), graph_(ctx), scope_(ctx) {}

Pipeline::~Pipeline() = default;

Status Pipeline::build() {
    // In production this would parse config/pipeline.yaml and add one
    // StageContract per entry. For the firmware-equivalent build we
    // synthesize four canonical stages with explicit dependencies.
    scheduler::StageContract preprocess;
    preprocess.name = "preprocess";
    preprocess.module = "nexusrt.kernels.tokenize";
    preprocess.function = "tokenize_bpe_u8";
    preprocess.token_budget = 8192;
    preprocess.sm_footprint_mb = 32;
    preprocess.mem_footprint_mb = 256;
    preprocess.grid[0] = 32; preprocess.block[0] = 256;
    auto s = graph_.add_stage(preprocess);
    if (!ok(s)) return s;

    scheduler::StageContract train;
    train.name = "train";
    train.module = "nexusrt.kernels.train";
    train.function = "transformer_fwd_bwd";
    train.token_budget = 8192;
    train.sm_footprint_mb = 128;
    train.mem_footprint_mb = 32768;
    train.depends_on = {"preprocess"};
    train.grid[0] = 64; train.block[0] = 256;
    s = graph_.add_stage(train);
    if (!ok(s)) return s;

    scheduler::StageContract infer;
    infer.name = "infer";
    infer.module = "nexusrt.kernels.infer";
    infer.function = "decode_step";
    infer.token_budget = 4096;
    infer.sm_footprint_mb = 64;
    infer.mem_footprint_mb = 16384;
    infer.depends_on = {"preprocess"};
    infer.grid[0] = 32; infer.block[0] = 256;
    s = graph_.add_stage(infer);
    if (!ok(s)) return s;

    scheduler::StageContract postprocess;
    postprocess.name = "postprocess";
    postprocess.module = "nexusrt.kernels.postprocess";
    postprocess.function = "detokenize_format";
    postprocess.token_budget = 1024;
    postprocess.sm_footprint_mb = 16;
    postprocess.mem_footprint_mb = 64;
    postprocess.depends_on = {"infer"};
    postprocess.grid[0] = 8; postprocess.block[0] = 128;
    s = graph_.add_stage(postprocess);
    if (!ok(s)) return s;

    s = graph_.validate();
    if (!ok(s)) return s;
    built_ = true;
    return Status::Ok;
}

Status Pipeline::run() {
    if (!built_) {
        auto s = build();
        if (!ok(s)) return s;
    }
    if (cfg_.dry_run) return Status::Ok;
    return graph_.run();
}

Status Pipeline::run_preprocess(const std::string& corpus_path) {
    // Allocate a small HBM buffer and stream the corpus header in via GDS.
    memory::Allocation buf;
    auto s = ctx_.memory().alloc(1 << 20, {}, buf);
    if (!ok(s)) return s;
    s = ctx_.dma().gds_read(corpus_path, 0, buf.hbm_ptr, 1 << 20);
    ctx_.memory().free(buf);
    return s;
}

Status Pipeline::run_train(uint32_t n_steps) {
    for (uint32_t i = 0; i < n_steps; ++i) {
        // In production: enqueue fwd+bwd kernels, issue NCCL allreduce,
        // wait barrier. Here we sync the firmware layer as a smoke test.
        auto s = ctx_.microkernel().sync_all();
        if (!ok(s)) return s;
    }
    return Status::Ok;
}

Status Pipeline::run_infer(const std::vector<int32_t>& prompt,
                           uint32_t max_new_tokens,
                           std::vector<int32_t>& out) {
    out.clear();
    out.reserve(prompt.size() + max_new_tokens);
    out = prompt;
    for (uint32_t i = 0; i < max_new_tokens; ++i) {
        // The autoregressive loop: schedule one decode step, manage the
        // KV-cache via token_opt, append the new token.
        scope_.step(out.size());
        int32_t next = static_cast<int32_t>(out.size());  // placeholder
        out.push_back(next);
    }
    return Status::Ok;
}

Status Pipeline::run_postprocess(const std::vector<int32_t>& tokens,
                                 std::string& out) {
    // Detokenize: map each token id to a printable character. This is a
    // stub; production uses a GPU-resident vocab table.
    std::ostringstream os;
    for (auto t : tokens) {
        if (t >= 32 && t < 127) os << static_cast<char>(t);
        else os << ' ';
    }
    out = os.str();
    return Status::Ok;
}

std::string Pipeline::metrics_json() const {
    std::ostringstream os;
    os << "{\"stages\":[";
    bool first = true;
    for (auto const& name : graph_.stage_names()) {
        if (!first) os << ",";
        first = false;
        auto n = graph_.find(name);
        os << "{\"name\":\"" << name << "\","
           << "\"state\":" << (n ? static_cast<int>(n->state) : 0) << ","
           << "\"last_status\":" << (n ? n->last_status : 0) << "}";
    }
    os << "]}";
    return os.str();
}

} // namespace pipeline
} // namespace nexusrt
