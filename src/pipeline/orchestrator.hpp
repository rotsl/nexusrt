// =============================================================================
// NexusRT — src/pipeline/orchestrator.hpp
// End-to-end LLM pipeline orchestrator. NO PyTorch / TF / JAX imports.
//
// The orchestrator wires together the four canonical stages:
//   preprocess → train → infer → postprocess
// using scheduler::TaskGraph + firmware::DmaEngine + token_opt::ContextScope.
//
// Each stage maps to a stage contract (see config/pipeline.yaml). The
// orchestrator reads the contract catalog at construction time and exposes
// a single run() entry point.
// =============================================================================
#pragma once

#include "firmware/types.hpp"
#include "firmware/context.hpp"
#include "scheduler/graph.hpp"
#include "token_opt/scope.hpp"

#include <memory>
#include <string>
#include <vector>

namespace nexusrt {
namespace pipeline {

struct PipelineConfig {
    std::string contracts_yaml = "config/pipeline.yaml";
    std::string runtime_yaml   = "config/runtime.yaml";
    std::string hardware_yaml  = "config/hardware.yaml";
    bool        dry_run        = false;
    bool        enable_review_gates = false;
};

class Pipeline {
public:
    Pipeline(FirmwareContext& ctx, PipelineConfig const& cfg);
    ~Pipeline();

    // Build the task graph from the contract catalog.
    Status build();

    // Execute the pipeline end-to-end.
    Status run();

    // Per-stage entry points (used by tests and notebooks).
    Status run_preprocess(const std::string& corpus_path);
    Status run_train(uint32_t n_steps);
    Status run_infer(const std::vector<int32_t>& prompt_tokens,
                     uint32_t max_new_tokens,
                     std::vector<int32_t>& out_tokens);
    Status run_postprocess(const std::vector<int32_t>& token_ids,
                           std::string& out_text);

    // Diagnostics — returns a JSON snapshot.
    std::string metrics_json() const;

private:
    FirmwareContext&         ctx_;
    PipelineConfig           cfg_;
    scheduler::TaskGraph     graph_;
    token_opt::ContextScope  scope_;
    bool                     built_ = false;
};

} // namespace pipeline
} // namespace nexusrt
