// =============================================================================
// NexusRT — src/firmware/context.cpp
// =============================================================================
#include "firmware/context.hpp"
#include "memory/manager.hpp"

namespace nexusrt {
namespace firmware {

::nexusrt::memory::MemoryManager& FirmwareContext::memory() {
    std::lock_guard<std::mutex> g(registry_mutex());  // reuse existing mutex
    if (!mem_) {
        mem_ = std::make_shared<::nexusrt::memory::MemoryManager>(*this);
    }
    return *mem_;
}

FirmwareContext::~FirmwareContext() {
    // RAII: even if the caller forgot nexusrt_firmware_shutdown(), we still
    // release GPU resources here. Subsystem shared_ptrs are released in
    // reverse-of-declaration order automatically.
}

std::mutex& FirmwareContext::registry_mutex() {
    static std::mutex m;
    return m;
}

std::unordered_map<void*, std::shared_ptr<FirmwareContext>>&
FirmwareContext::registry() {
    static std::unordered_map<void*, std::shared_ptr<FirmwareContext>> r;
    return r;
}

void FirmwareContext::register_handle(void* k, std::shared_ptr<FirmwareContext> sp) {
    std::lock_guard<std::mutex> g(registry_mutex());
    registry()[k] = std::move(sp);
}

std::shared_ptr<FirmwareContext> FirmwareContext::lookup_handle(void* k) {
    std::lock_guard<std::mutex> g(registry_mutex());
    auto it = registry().find(k);
    return it == registry().end() ? nullptr : it->second;
}

std::shared_ptr<FirmwareContext> FirmwareContext::take_handle(void* k) {
    std::lock_guard<std::mutex> g(registry_mutex());
    auto it = registry().find(k);
    if (it == registry().end()) return nullptr;
    auto sp = std::move(it->second);
    registry().erase(it);
    return sp;
}

} // namespace firmware
} // namespace nexusrt
