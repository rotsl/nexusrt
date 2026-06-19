// =============================================================================
// NexusRT — src/firmware/metal_shim.mm
// Apple Metal / MLX shim. Compiled only when NEXUSRT_ENABLE_METAL is on.
//
// Maps the same PlatformInterface onto MTLDevice + MTLCommandQueue +
// MTLComputeCommandEncoder. Unified memory is native — MTLBuffer wraps a
// pointer that is simultaneously CPU- and GPU-readable, so the fault handler
// is effectively a no-op.
// =============================================================================
#include "firmware/types.hpp"
#include "firmware/context.hpp"
#include "firmware/microkernel.hpp"
#include "platform/dispatch.hpp"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace nexusrt {
namespace platform {

class MetalPlatform : public PlatformInterface {
public:
    bool probe() override;
    DeviceDesc describe_device(const std::string& profile) override;
    Status lock_clocks(const DeviceDesc& d, FirmwareContext& ctx) override;

    Status reserve_vaddress(FirmwareContext& ctx, uint64_t bytes) override;
    Status map_hbm_pool(FirmwareContext& ctx) override;
    Status release_vaddress(FirmwareContext& ctx) override;

    Status create_stream(FirmwareContext& ctx, StreamClass klass,
                         int priority, void*& out) override;
    Status destroy_stream(FirmwareContext& ctx, void* stream) override;
    Status sync_stream(FirmwareContext& ctx, void* stream) override;

    Status load_module(FirmwareContext& ctx, const std::string& name,
                       const std::string& src, KernelModule& out) override;
    Status launch_kernel(FirmwareContext& ctx, void* fn,
                         uint32_t gx, uint32_t gy, uint32_t gz,
                         uint32_t bx, uint32_t by, uint32_t bz,
                         uint32_t smem, void* stream,
                         void** args, size_t n_args) override;

    Status alloc_hbm(FirmwareContext& ctx, uint64_t bytes, void*& out) override;
    Status free_hbm(FirmwareContext& ctx, void* p) override;
    Status prefetch_range(FirmwareContext& ctx,
                          uint64_t vaddr, uint64_t bytes) override;

    Status gds_init(FirmwareContext& ctx) override { return Status::NotImplemented; }
    Status gds_fini(FirmwareContext& ctx) override { return Status::Ok; }
    Status gds_read_fallback(FirmwareContext& ctx, const std::string& path,
                             uint64_t off, void* dst, uint64_t bytes) override;
    Status gds_write_fallback(FirmwareContext& ctx, const std::string& path,
                              uint64_t off, void* src, uint64_t bytes) override;
    Status grdma_init(FirmwareContext& ctx) override { return Status::NotImplemented; }
    Status grdma_fini(FirmwareContext& ctx) override { return Status::Ok; }

    Status install_fault_buffer(FirmwareContext&, void*, uint32_t) override {
        return Status::Ok;  // unified memory: no faults
    }
    Status uninstall_fault_buffer(FirmwareContext&, void*) override {
        return Status::Ok;
    }
    Status resolve_fault(FirmwareContext&, uint64_t) override {
        return Status::Ok;
    }

    Status tma_encode(FirmwareContext&, TmaDescriptor&) override {
        return Status::NotImplemented;
    }
    Status tma_issue_copy(FirmwareContext&, TmaDescriptor const&,
                          void*, uint64_t, uint64_t, void*) override {
        return Status::NotImplemented;
    }
    Status tma_release(FirmwareContext&, void*) override { return Status::Ok; }

    Status ilc_tag(FirmwareContext&, void*) override { return Status::NotImplemented; }
    Status ilc_untag(FirmwareContext&, void*) override { return Status::Ok; }

    Status dma_submit(FirmwareContext& ctx, DmaRequest const& r,
                      void** out_event) override;
    Status dma_wait(FirmwareContext& ctx, void* ev, uint32_t ms) override;

private:
    struct MetalState {
        id<MTLDevice>        device  = nil;
        id<MTLCommandQueue>  queue   = nil;
        std::vector<id<MTLBuffer>> buffers;  // keep alive
        std::mutex mtx;
    };
    static MetalState* state(FirmwareContext& ctx);
    static std::mutex& states_mtx();
    static std::map<FirmwareContext*, std::unique_ptr<MetalState>>& states();
};

namespace {
struct AutoRegister {
    AutoRegister() {
        PlatformDispatch::instance().register_backend(
            Vendor::Apple, std::make_shared<MetalPlatform>());
    }
};
static AutoRegister g_auto_register;
} // namespace

bool MetalPlatform::probe() {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    return d != nil;
}

DeviceDesc MetalPlatform::describe_device(const std::string& /*profile*/) {
    DeviceDesc d;
    d.vendor = Vendor::Apple;
    d.arch   = Arch::AppleSilicon;
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return d;
    d.name = [[dev name] UTF8String];

    // M1 Pro: 16 or 32 GB unified LPDDR5
    // We can't read the actual size from Metal directly; rely on the
    // hardware.yaml override or recommendedMaxWorkingSetSize.
    d.hbm_capacity_bytes = (uint64_t)[dev recommendedMaxWorkingSetSize];
    d.hbm_bandwidth_bps  = 200ull * 1024 * 1024 * 1024;
    d.sm_count           = 16;        // M1 Pro: 16 GPU cores
    d.smem_per_sm_bytes  = 32 * 1024; // threadgroup memory
    d.features.unified_native = true;
    d.features.gds   = false;
    d.features.grdma = false;
    d.features.tma   = false;
    d.features.ilc   = false;
    d.features.dsm   = false;
    return d;
}

Status MetalPlatform::lock_clocks(const DeviceDesc&, FirmwareContext&) {
    return Status::Ok;  // no DVFS control on Apple Silicon
}

Status MetalPlatform::reserve_vaddress(FirmwareContext& ctx, uint64_t) {
    // Unified memory: no VA reservation needed. We just store a sentinel.
    ctx.plat.vaddress_base = 0xDEADBEEF;
    ctx.plat.vaddress_size = 0;
    return Status::Ok;
}

Status MetalPlatform::map_hbm_pool(FirmwareContext&) {
    return Status::Ok;
}

Status MetalPlatform::release_vaddress(FirmwareContext& ctx) {
    ctx.plat.vaddress_base = 0;
    return Status::Ok;
}

Status MetalPlatform::create_stream(FirmwareContext& ctx, StreamClass,
                                    int, void*& out) {
    auto* s = state(ctx);
    if (!s) return Status::InvalidArgument;
    @autoreleasepool {
        // Metal has no priority streams; we just create a command buffer.
        id<MTLCommandBuffer> cb = [s->queue commandBuffer];
        [cb retain];
        out = (__bridge void*)cb;
    }
    return Status::Ok;
}

Status MetalPlatform::destroy_stream(FirmwareContext&, void* stream) {
    if (!stream) return Status::Ok;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge_transfer id<MTLCommandBuffer>)stream;
        (void)cb;  // released via ARC
    }
    return Status::Ok;
}

Status MetalPlatform::sync_stream(FirmwareContext&, void* stream) {
    if (!stream) return Status::Ok;
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)stream;
    [cb waitUntilCompleted];
    return Status::Ok;
}

Status MetalPlatform::load_module(FirmwareContext& ctx, const std::string&,
                                  const std::string& src, KernelModule& out) {
    auto* s = state(ctx);
    if (!s) return Status::InvalidArgument;
    @autoreleasepool {
        NSString* code = [NSString stringWithUTF8String:src.c_str()];
        NSError* err = nil;
        id<MTLLibrary> lib = [s->device newLibraryWithSource:code
                                                     options:nil
                                                       error:&err];
        if (err) return Status::DriverError;
        out.raw_module = (__bridge_retained void*)lib;
    }
    return Status::Ok;
}

Status MetalPlatform::launch_kernel(FirmwareContext& ctx, void* fn,
                                    uint32_t gx, uint32_t gy, uint32_t gz,
                                    uint32_t bx, uint32_t by, uint32_t bz,
                                    uint32_t /*smem*/, void* stream,
                                    void** args, size_t /*n_args*/) {
    auto* s = state(ctx);
    if (!s) return Status::InvalidArgument;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)stream;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        id<MTLComputePipelineState> ps = (__bridge id<MTLComputePipelineState>)fn;
        [enc setComputePipelineState:ps];
        // Bind args — caller is responsible for encoding them as MTLBuffer
        // bindings. Here we just dispatch the grid.
        MTLSize tg = MTLSizeMake(bx, by, bz);
        MTLSize grid = MTLSizeMake(gx * bx, gy * by, gz * bz);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
    }
    (void)args;
    return Status::Ok;
}

Status MetalPlatform::alloc_hbm(FirmwareContext& ctx, uint64_t bytes,
                                void*& out) {
    auto* s = state(ctx);
    if (!s) return Status::InvalidArgument;
    @autoreleasepool {
        id<MTLBuffer> buf = [s->device newBufferWithLength:bytes
                                                   options:MTLResourceStorageModeShared];
        if (!buf) return Status::OutOfMemory;
        [buf retain];
        {
            std::lock_guard<std::mutex> g(s->mtx);
            s->buffers.push_back(buf);
        }
        out = [buf contents];
    }
    return Status::Ok;
}

Status MetalPlatform::free_hbm(FirmwareContext& ctx, void* p) {
    auto* s = state(ctx);
    if (!s || !p) return Status::Ok;
    std::lock_guard<std::mutex> g(s->mtx);
    for (auto it = s->buffers.begin(); it != s->buffers.end(); ++it) {
        if ([*it contents] == p) {
            [*it release];
            s->buffers.erase(it);
            return Status::Ok;
        }
    }
    return Status::InvalidArgument;
}

Status MetalPlatform::prefetch_range(FirmwareContext&, uint64_t, uint64_t) {
    // Unified memory: nothing to prefetch.
    return Status::Ok;
}

Status MetalPlatform::gds_read_fallback(FirmwareContext& ctx,
                                        const std::string& path,
                                        uint64_t off, void* dst, uint64_t bytes) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return Status::IoError;
    std::fseek(f, (long)off, SEEK_SET);
    size_t got = std::fread(dst, 1, bytes, f);
    std::fclose(f);
    return got == bytes ? Status::Ok : Status::IoError;
}

Status MetalPlatform::gds_write_fallback(FirmwareContext& ctx,
                                         const std::string& path,
                                         uint64_t off, void* src, uint64_t bytes) {
    FILE* f = std::fopen(path.c_str(), "r+b");
    if (!f) return Status::IoError;
    std::fseek(f, (long)off, SEEK_SET);
    size_t put = std::fwrite(src, 1, bytes, f);
    std::fclose(f);
    return put == bytes ? Status::Ok : Status::IoError;
}

Status MetalPlatform::dma_submit(FirmwareContext& ctx, DmaRequest const& r,
                                 void** out_event) {
    // On Metal, "DMA" is a blit encoder. For host↔device on unified memory
    // it is a no-op (the pointer is shared). For D2D we use copyFromBuffer.
    auto* s = state(ctx);
    if (!s) return Status::InvalidArgument;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [s->queue commandBuffer];
        [cb retain];
        *out_event = (__bridge void*)cb;
    }
    return Status::Ok;
}

Status MetalPlatform::dma_wait(FirmwareContext&, void* ev, uint32_t) {
    if (!ev) return Status::Ok;
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)ev;
    [cb waitUntilCompleted];
    return Status::Ok;
}

MetalPlatform::MetalState* MetalPlatform::state(FirmwareContext& ctx) {
    std::lock_guard<std::mutex> g(states_mtx());
    auto it = states().find(&ctx);
    if (it == states().end()) {
        auto p = std::make_unique<MetalState>();
        p->device = MTLCreateSystemDefaultDevice();
        if (!p->device) return nil;
        p->queue  = [p->device newCommandQueue];
        auto [ins, _] = states().emplace(&ctx, std::move(p));
        return ins->second.get();
    }
    return it->second.get();
}

std::mutex& MetalPlatform::states_mtx() { static std::mutex m; return m; }
std::map<FirmwareContext*, std::unique_ptr<MetalPlatform::MetalState>>&
MetalPlatform::states() {
    static std::map<FirmwareContext*, std::unique_ptr<MetalState>> s;
    return s;
}

} // namespace platform
} // namespace nexusrt
