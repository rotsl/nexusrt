// =============================================================================
// NexusRT — src/scheduler/stream_pool.hpp
// Stream pool wrapper around firmware::MicroKernel. Provides scoped
// acquire / release via RAII.
// =============================================================================
#pragma once

#include "firmware/types.hpp"
#include "firmware/context.hpp"
#include "firmware/microkernel.hpp"

namespace nexusrt {
namespace scheduler {

class ScopedStream {
public:
    ScopedStream(FirmwareContext& ctx, firmware::StreamClass klass, int prio = 0)
        : ctx_(ctx), h_(ctx.microkernel().acquire_stream(klass, prio)) {}
    ~ScopedStream() { if (h_.id) ctx_.microkernel().release_stream(h_); }
    ScopedStream(ScopedStream const&)            = delete;
    ScopedStream& operator=(ScopedStream const&) = delete;
    ScopedStream(ScopedStream&& o) noexcept : ctx_(o.ctx_), h_(o.h_) { o.h_.id = 0; }

    firmware::StreamHandle handle() const { return h_; }
    void*                  raw()    const { return h_.raw; }
    explicit operator bool() const { return h_.id != 0; }

    Status sync() { return ctx_.microkernel().sync_stream(h_); }

private:
    FirmwareContext&       ctx_;
    firmware::StreamHandle h_;
};

} // namespace scheduler
} // namespace nexusrt
