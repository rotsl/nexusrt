// =============================================================================
// NexusRT — src/platform/abi_stubs.cpp
// Translation unit that pulls in all the C ABI symbols so the shared library
// has a single entry point for linking.
// =============================================================================
#include "platform/abi.h"

// The C ABI functions are defined in their respective module .cpp files:
//   - nexusrt_firmware_*       : src/firmware/boot.cpp, src/firmware/fault_handler.cpp
//   - nexusrt_mem_*             : src/memory/manager.cpp
//   - nexusrt_submit_stage etc  : src/scheduler/graph.cpp
//   - nexusrt_context_scope etc : src/token_opt/scope.cpp
//
// This file is intentionally empty — it exists only to ensure the shared
// library has at least one translation unit when those modules are linked
// in statically.

extern "C" {

NEXUSRT_API int32_t nexusrt_version(int32_t* out_major,
                                    int32_t* out_minor,
                                    int32_t* out_patch) {
    if (out_major) *out_major = 0;
    if (out_minor) *out_minor = 1;
    if (out_patch) *out_patch = 0;
    return 0;
}

NEXUSRT_API int32_t nexusrt_metrics_dump(void* /*ctx*/, char* out_json, size_t cap) {
    static constexpr const char* kMetrics = "{\"status\":\"not_implemented\"}";
    if (!out_json || cap == 0) return -1;
    size_t i = 0;
    for (; i + 1 < cap && kMetrics[i] != '\0'; ++i) {
        out_json[i] = kMetrics[i];
    }
    out_json[i] = '\0';
    return 0;
}

NEXUSRT_API int nexusrt_abi_stubs_anchor(void) { return 0; }

} // extern "C"
