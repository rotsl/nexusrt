// =============================================================================
// NexusRT — src/firmware/microkernel.cpp
// =============================================================================
#include "firmware/microkernel.hpp"
#include "platform/dispatch.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace nexusrt {
namespace firmware {

MicroKernel::MicroKernel(FirmwareContext& ctx, DeviceDesc const& d)
    : ctx_(ctx), device_(d) {
    // Default warp specialization policy by architecture.
    if (d.arch == Arch::Hopper) {
        default_ws_ = {2, 14, 0};           // 2 producer + 14 consumer + 0 fence
    } else if (d.arch == Arch::Ampere ||
               d.arch == Arch::Pascal ||
               d.arch == Arch::Turing) {
        default_ws_ = {2, 14, 0};
    } else {
        default_ws_ = {0, 16, 0};           // Metal: no warp specialization
    }
}

MicroKernel::~MicroKernel() {
    drain_streams();
}

Status MicroKernel::spawn_stream_pool(uint32_t n_compute, uint32_t n_dma) {
    auto& plat = platform::PlatformDispatch::instance();
    for (uint32_t i = 0; i < n_compute; ++i) {
        StreamHandle h;
        h.id     = next_id_.fetch_add(1);
        h.klass  = StreamClass::Compute;
        auto s = plat.create_stream(ctx_, h.klass, 0, h.raw);
        if (!ok(s)) return s;
        free_compute_.push_back(h);
    }
    for (uint32_t i = 0; i < n_dma; ++i) {
        StreamHandle h;
        h.id     = next_id_.fetch_add(1);
        h.klass  = StreamClass::Dma;
        // DMA streams get high (negative) priority so the GPU scheduler
        // prefers them for copy work.
        auto s = plat.create_stream(ctx_, h.klass, -1, h.raw);
        if (!ok(s)) return s;
        free_dma_.push_back(h);
    }
    // Fence stream is single-allocation on H100 (mbarrier). On CUDA devices
    // without mbarrier support, we still use a low-priority stream for
    // host-visible doorbell work.
    if (device_.arch == Arch::Hopper ||
        device_.arch == Arch::Ampere ||
        device_.arch == Arch::Pascal ||
        device_.arch == Arch::Turing) {
        StreamHandle h;
        h.id     = next_id_.fetch_add(1);
        h.klass  = StreamClass::Fence;
        auto s = plat.create_stream(ctx_, h.klass, +1, h.raw);
        if (!ok(s)) return s;
        free_fence_.push_back(h);
    }
    return Status::Ok;
}

Status MicroKernel::drain_streams() {
    std::lock_guard<std::mutex> g(mtx_);
    auto& plat = platform::PlatformDispatch::instance();
    auto release_all = [&](std::vector<StreamHandle>& v) {
        for (auto& h : v) plat.destroy_stream(ctx_, h.raw);
        v.clear();
    };
    release_all(free_compute_);
    release_all(free_dma_);
    release_all(free_fence_);
    modules_.clear();
    return Status::Ok;
}

Status MicroKernel::load_module(const std::string& name,
                                const std::string& ptx_or_metal_src,
                                KernelModule&      out) {
    std::lock_guard<std::mutex> g(mtx_);
    if (modules_.count(name)) {
        out = *modules_[name];
        return Status::Ok;
    }
    auto mod = std::make_shared<KernelModule>();
    mod->id   = next_id_.fetch_add(1);
    mod->name = name;
    auto s = platform::PlatformDispatch::instance()
                 .load_module(ctx_, name, ptx_or_metal_src, *mod);
    if (!ok(s)) return s;
    modules_[name] = mod;
    out = *mod;
    return Status::Ok;
}

StreamHandle MicroKernel::acquire_stream(StreamClass klass, int priority) {
    std::lock_guard<std::mutex> g(mtx_);
    StreamHandle h;
    switch (klass) {
        case StreamClass::Compute: {
            if (free_compute_.empty()) break;
            h = free_compute_.back(); free_compute_.pop_back();
            return h;
        }
        case StreamClass::Dma: {
            if (free_dma_.empty()) break;
            h = free_dma_.back(); free_dma_.pop_back();
            return h;
        }
        case StreamClass::Fence: {
            if (free_fence_.empty()) break;
            h = free_fence_.back(); free_fence_.pop_back();
            return h;
        }
        case StreamClass::Control: break;
    }
    h.id = 0;            // invalid handle
    return h;
}

void MicroKernel::release_stream(StreamHandle h) {
    if (h.id == 0) return;
    std::lock_guard<std::mutex> g(mtx_);
    switch (h.klass) {
        case StreamClass::Compute: free_compute_.push_back(h); break;
        case StreamClass::Dma:     free_dma_.push_back(h);     break;
        case StreamClass::Fence:   free_fence_.push_back(h);   break;
        case StreamClass::Control: break;
    }
}

KernelModule const* MicroKernel::find_module(const std::string& name) const {
    std::lock_guard<std::mutex> g(mtx_);
    auto it = modules_.find(name);
    return it == modules_.end() ? nullptr : it->second.get();
}

Status MicroKernel::launch(const std::string& module_name,
                           const std::string& function_name,
                           uint32_t gx, uint32_t gy, uint32_t gz,
                           uint32_t bx, uint32_t by, uint32_t bz,
                           uint32_t shared_mem_bytes,
                           StreamHandle stream,
                           void** kernel_args,
                           size_t n_args) {
    auto mod = find_module(module_name);
    if (!mod) return Status::InvalidArgument;
    auto fnit = mod->functions.find(function_name);
    if (fnit == mod->functions.end()) return Status::InvalidArgument;

    return platform::PlatformDispatch::instance().launch_kernel(
        ctx_, fnit->second,
        gx, gy, gz, bx, by, bz,
        shared_mem_bytes, stream.raw,
        kernel_args, n_args);
}

Status MicroKernel::sync_stream(StreamHandle s) {
    return platform::PlatformDispatch::instance().sync_stream(ctx_, s.raw);
}

Status MicroKernel::sync_all() {
    std::lock_guard<std::mutex> g(mtx_);
    auto& plat = platform::PlatformDispatch::instance();
    auto sync = [&](std::vector<StreamHandle> const& v) -> Status {
        for (auto const& h : v) {
            auto s = plat.sync_stream(ctx_, h.raw);
            if (!ok(s)) return s;
        }
        return Status::Ok;
    };
    auto a = sync(free_compute_); if (!ok(a)) return a;
    auto b = sync(free_dma_);     if (!ok(b)) return b;
    auto c = sync(free_fence_);   if (!ok(c)) return c;
    return Status::Ok;
}

} // namespace firmware
} // namespace nexusrt
