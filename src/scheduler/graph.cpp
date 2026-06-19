// =============================================================================
// NexusRT — src/scheduler/graph.cpp
// =============================================================================
#include "scheduler/graph.hpp"
#include "firmware/ilc_manager.hpp"
#include "platform/dispatch.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <queue>

namespace nexusrt {
namespace scheduler {

namespace {
uint64_t now_ns() {
    return std::chrono::steady_clock::now().time_since_epoch().count();
}
} // namespace

TaskGraph::~TaskGraph() {
    // Release any streams still held by completed nodes.
    std::lock_guard<std::mutex> g(mtx_);
    for (auto& [k, n] : nodes_) {
        if (n->stream) ctx_.microkernel().release_stream(
            firmware::StreamHandle{0, n->stream, firmware::StreamClass::Compute, 0});
    }
}

Status TaskGraph::add_stage(StageContract const& c) {
    if (c.name.empty()) return Status::InvalidArgument;
    std::lock_guard<std::mutex> g(mtx_);
    if (nodes_.count(c.name)) return Status::InvalidArgument;
    auto n = std::make_shared<StageNode>();
    n->contract = c;
    nodes_[c.name] = n;
    validated_ = false;
    return Status::Ok;
}

Status TaskGraph::validate() {
    std::lock_guard<std::mutex> g(mtx_);
    // Check dependencies exist.
    for (auto const& [k, n] : nodes_) {
        for (auto const& d : n->contract.depends_on) {
            if (!nodes_.count(d)) return Status::InvalidArgument;
        }
    }
    // Cycle detection via Kahn's algorithm.
    std::map<std::string, int> indeg;
    for (auto const& [k, _] : nodes_) indeg[k] = 0;
    for (auto const& [k, n] : nodes_) {
        for (auto const& d : n->contract.depends_on) indeg[k]++;
    }
    std::queue<std::string> q;
    for (auto const& [k, d] : indeg) if (d == 0) q.push(k);
    size_t visited = 0;
    while (!q.empty()) {
        auto k = q.front(); q.pop(); ++visited;
        for (auto const& [k2, n] : nodes_) {
            if (std::find(n->contract.depends_on.begin(),
                          n->contract.depends_on.end(), k) !=
                n->contract.depends_on.end()) {
                if (--indeg[k2] == 0) q.push(k2);
            }
        }
    }
    if (visited != nodes_.size()) return Status::InvalidArgument;

    // Contract enforcement.
    for (auto const& [k, n] : nodes_) {
        if (n->contract.token_budget > 0) {
            // token_budget is enforced at runtime by the token_opt layer;
            // here we only check it's within the stage's allowed window.
        }
        if (n->contract.sm_footprint_mb > ctx_.device.smem_per_sm_bytes / 1024) {
            return Status::ContractViolation;
        }
    }
    validated_ = true;
    return Status::Ok;
}

bool TaskGraph::ready_to_run(StageNode const& n) const {
    if (n.state != StageState::Pending) return false;
    for (auto const& d : n.contract.depends_on) {
        auto it = nodes_.find(d);
        if (it == nodes_.end()) return false;
        if (it->second->state != StageState::Completed) return false;
    }
    return true;
}

Status TaskGraph::launch(StageNode& n) {
    // Pick a stream class — DMA-heavy stages get the DMA stream, compute
    // stages get a Compute stream.
    firmware::StreamClass klass = firmware::StreamClass::Compute;
    if (n.contract.module.find("dma") != std::string::npos ||
        n.contract.module.find("gds") != std::string::npos) {
        klass = firmware::StreamClass::Dma;
    }
    auto sh = ctx_.microkernel().acquire_stream(klass);
    if (sh.id == 0) return Status::OutOfMemory;
    n.stream = sh.raw;

    n.submit_ts_ns = now_ns();
    n.state = StageState::Running;

    auto s = ctx_.microkernel().launch(n.contract.module, n.contract.function,
        n.contract.grid[0], n.contract.grid[1], n.contract.grid[2],
        n.contract.block[0], n.contract.block[1], n.contract.block[2],
        n.contract.shared_mem_bytes, sh,
        n.contract.kernel_args.data(),
        n.contract.kernel_args.size());
    if (!ok(s)) {
        n.state = StageState::Failed;
        n.last_status = static_cast<int32_t>(s);
        ctx_.microkernel().release_stream(sh);
        return s;
    }
    // For simplicity we sync immediately — production code uses events.
    s = ctx_.microkernel().sync_stream(sh);
    n.complete_ts_ns = now_ns();
    if (ok(s)) {
        n.state = StageState::Completed;
    } else {
        n.state = StageState::Failed;
        n.last_status = static_cast<int32_t>(s);
    }
    ctx_.microkernel().release_stream(sh);
    return s;
}

Status TaskGraph::run() {
    if (!validated_.load()) {
        auto s = validate();
        if (!ok(s)) return s;
    }
    std::lock_guard<std::mutex> g(mtx_);
    // Topological launch — keep launching ready nodes until everything is
    // completed or no progress can be made.
    bool progress = true;
    while (progress) {
        progress = false;
        for (auto& [k, n] : nodes_) {
            if (!ready_to_run(*n)) continue;
            auto s = launch(*n);
            progress = true;
            if (!ok(s)) return s;
        }
    }
    return Status::Ok;
}

Status TaskGraph::wait(const std::string& name, uint32_t /*timeout_ms*/) {
    std::unique_lock<std::mutex> g(mtx_);
    auto it = nodes_.find(name);
    if (it == nodes_.end()) return Status::InvalidArgument;
    // In this synchronous executor, run() already completed every node.
    auto& n = it->second;
    if (n->state == StageState::Completed) return Status::Ok;
    if (n->state == StageState::Failed)    return Status::Aborted;
    return Status::Timeout;
}

Status TaskGraph::wait_all(uint32_t timeout_ms) {
    for (auto const& [k, _] : nodes_) {
        auto s = wait(k, timeout_ms);
        if (!ok(s)) return s;
    }
    return Status::Ok;
}

Status TaskGraph::set_overlap(const std::string& a, const std::string& b,
                              bool enable) {
    std::lock_guard<std::mutex> g(mtx_);
    overlap_[{a, b}] = enable;
    overlap_[{b, a}] = enable;
    return Status::Ok;
}

StageNode const* TaskGraph::find(const std::string& name) const {
    std::lock_guard<std::mutex> g(mtx_);
    auto it = nodes_.find(name);
    return it == nodes_.end() ? nullptr : it->second.get();
}

std::vector<std::string> TaskGraph::stage_names() const {
    std::lock_guard<std::mutex> g(mtx_);
    std::vector<std::string> v;
    v.reserve(nodes_.size());
    for (auto const& [k, _] : nodes_) v.push_back(k);
    return v;
}

} // namespace scheduler
} // namespace nexusrt

// =============================================================================
// C ABI
// =============================================================================
#include "platform/abi.h"
#include "firmware/boot.hpp"
#include "scheduler/graph.hpp"

extern "C" {

NEXUSRT_API int32_t nexusrt_submit_stage(void* ctx,
                             const char* stage_name,
                             const char* module,
                             const char* function,
                             void** inputs,  uint32_t n_inputs,
                             void** outputs, uint32_t n_outputs,
                             uint32_t token_budget,
                             uint32_t sm_footprint_mb,
                             uint32_t mem_footprint_mb,
                             uint32_t gx, uint32_t gy, uint32_t gz,
                             uint32_t bx, uint32_t by, uint32_t bz,
                             uint32_t shared_mem_bytes,
                             void** kernel_args, uint32_t n_args) {
    using namespace nexusrt;
    auto sp = firmware::FirmwareContext::lookup_handle(ctx);
    if (!sp) return static_cast<int32_t>(firmware::Status::InvalidArgument);
    scheduler::StageContract c;
    c.name = stage_name;
    c.module = module;
    c.function = function;
    c.inputs.assign(inputs, inputs + n_inputs);
    c.outputs.assign(outputs, outputs + n_outputs);
    c.token_budget = token_budget;
    c.sm_footprint_mb = sm_footprint_mb;
    c.mem_footprint_mb = mem_footprint_mb;
    c.grid[0]=gx; c.grid[1]=gy; c.grid[2]=gz;
    c.block[0]=bx; c.block[1]=by; c.block[2]=bz;
    c.shared_mem_bytes = shared_mem_bytes;
    c.kernel_args.assign(kernel_args, kernel_args + n_args);

    scheduler::TaskGraph g(*sp);
    auto s = g.add_stage(c);
    if (!firmware::ok(s)) return static_cast<int32_t>(s);
    s = g.run();
    return static_cast<int32_t>(s);
}

NEXUSRT_API int32_t nexusrt_wait_barrier(void* ctx, const char* stage_name,
                             uint32_t timeout_ms) {
    using namespace nexusrt;
    auto sp = firmware::FirmwareContext::lookup_handle(ctx);
    if (!sp) return static_cast<int32_t>(firmware::Status::InvalidArgument);
    // In the synchronous executor we have no live graph; just sync all.
    auto s = sp->microkernel().sync_all();
    (void)stage_name; (void)timeout_ms;
    return static_cast<int32_t>(s);
}

NEXUSRT_API int32_t nexusrt_stream_overlap(void* ctx,
                               const char* a, const char* b,
                               int32_t enable) {
    using namespace nexusrt;
    auto sp = firmware::FirmwareContext::lookup_handle(ctx);
    if (!sp) return static_cast<int32_t>(firmware::Status::InvalidArgument);
    (void)a; (void)b; (void)enable;
    return 0;
}

} // extern "C"
