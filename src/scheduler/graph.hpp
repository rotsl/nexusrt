// =============================================================================
// NexusRT — src/scheduler/graph.hpp
// Async task graph executor with explicit dependency tracking and
// warp-specialized concurrent compute + DMA.
//
// Each node in the graph is a "Stage" — a kernel launch with declared
// inputs, outputs, memory footprint, and token budget. The scheduler
// enforces stage contracts before launching, picks a stream from the
// firmware micro-kernel's pool, and uses events to enforce DAG ordering.
// =============================================================================
#pragma once

#include "firmware/types.hpp"
#include "firmware/context.hpp"
#include "firmware/microkernel.hpp"
#include "firmware/dma_engine.hpp"

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace nexusrt {
namespace scheduler {

// StageContract — declared by the caller, enforced by the scheduler.
struct StageContract {
    std::string name;
    std::string module;             // firmware::KernelModule name
    std::string function;           // kernel entrypoint name

    std::vector<void*> inputs;
    std::vector<void*> outputs;

    uint32_t token_budget       = 0;     // hard cap; enforced by token_opt
    uint32_t sm_footprint_mb    = 0;
    uint32_t mem_footprint_mb   = 0;

    uint32_t grid[3]  = {1,1,1};
    uint32_t block[3] = {1,1,1};
    uint32_t shared_mem_bytes = 0;

    std::vector<std::string> depends_on;  // names of upstream stages

    // Kernel arguments (raw void*). Caller must keep alive until wait().
    std::vector<void*> kernel_args;

    // Warp specialization policy override (defaults to firmware ctx's policy).
    uint32_t producer_warps = 0;          // 0 = use default
    uint32_t consumer_warps = 0;
};

enum class StageState : uint8_t {
    Pending   = 0,
    Ready     = 1,   // all deps satisfied
    Running   = 2,
    Completed = 3,
    Failed    = 4,
};

struct StageNode {
    StageContract  contract;
    StageState     state = StageState::Pending;
    void*          stream = nullptr;
    void*          done_event = nullptr;
    uint64_t       submit_ts_ns = 0;
    uint64_t       complete_ts_ns = 0;
    int32_t        last_status = 0;
};

class TaskGraph {
public:
    explicit TaskGraph(FirmwareContext& ctx) : ctx_(ctx) {}
    ~TaskGraph();

    // Register a stage. Returns its handle (the stage name, unique).
    Status add_stage(StageContract const& c);

    // Validate the DAG (no cycles, all dependencies exist, all contracts
    // satisfiable). Must be called before run().
    Status validate();

    // Submit every ready stage; returns immediately. Use wait() to join.
    Status run();

    // Wait for a specific stage to complete.
    Status wait(const std::string& stage_name, uint32_t timeout_ms);

    // Wait for all stages.
    Status wait_all(uint32_t timeout_ms);

    // Enable / disable stream overlap between two stages. When enabled, the
    // two stages will be scheduled on different streams (compute vs dma)
    // and may run concurrently.
    Status set_overlap(const std::string& a, const std::string& b, bool enable);

    // Diagnostics
    StageNode const* find(const std::string& name) const;
    std::vector<std::string> stage_names() const;

private:
    bool ready_to_run(StageNode const& n) const;
    Status launch(StageNode& n);

    FirmwareContext& ctx_;
    mutable std::mutex mtx_;
    std::unordered_map<std::string, std::shared_ptr<StageNode>> nodes_;
    std::map<std::pair<std::string,std::string>, bool> overlap_;
    std::atomic<bool> validated_{false};
};

} // namespace scheduler
} // namespace nexusrt
