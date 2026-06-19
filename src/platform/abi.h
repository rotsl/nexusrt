// =============================================================================
// NexusRT — src/platform/abi.h
// Unified C ABI for cross-platform binding. This header is the *only*
// interface exposed to Python (via cffi/ctypes) and to other language
// bindings.
//
// All functions return int32_t. Non-negative values are success; negative
// values are nexusrt::firmware::Status codes.
//
// Lifetime: every handle returned by nexusrt_*_create() must be released
// by the matching nexusrt_*_destroy(). RAII is enforced in the C++ layer;
// the C ABI is a thin wrapper.
// =============================================================================
#pragma once

#include <stdint.h>
#include <stddef.h>

#ifndef NEXUSRT_API
#if defined(_WIN32)
#define NEXUSRT_API __declspec(dllexport)
#else
#define NEXUSRT_API __attribute__((visibility("default")))
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ---- Version ---------------------------------------------------------------
NEXUSRT_API int32_t nexusrt_version(int32_t* out_major,
                                    int32_t* out_minor,
                                    int32_t* out_patch);

// ---- Firmware (see src/firmware/boot.hpp) ----------------------------------
NEXUSRT_API int32_t nexusrt_firmware_init(const char* profile, void** out_ctx);
NEXUSRT_API int32_t nexusrt_firmware_shutdown(void* ctx);
NEXUSRT_API int32_t nexusrt_firmware_device_desc(void* ctx,
                                                 int32_t*  out_vendor,
                                                 int32_t*  out_arch,
                                                 char*     out_name, size_t name_cap,
                                                 uint64_t* out_hbm_bytes,
                                                 uint32_t* out_sm_count,
                                                 uint32_t* out_features_bits);
NEXUSRT_API int32_t nexusrt_firmware_task_submit(void* ctx,
                                                 const char* module,
                                                 const char* function,
                                                 uint32_t gx, uint32_t gy, uint32_t gz,
                                                 uint32_t bx, uint32_t by, uint32_t bz,
                                                 uint32_t shared_mem_bytes,
                                                 void** args, uint32_t n_args,
                                                 int32_t  stream_class,
                                                 int32_t  stream_priority);
NEXUSRT_API int32_t nexusrt_firmware_fault_handler(void* ctx,
                                                   uint64_t faulting_addr,
                                                   int32_t  timeout_ms);

// ---- Memory (see src/memory/manager.hpp) -----------------------------------
NEXUSRT_API int32_t nexusrt_mem_alloc(void* ctx,
                                      uint64_t bytes,
                                      uint32_t flags,   // AllocHints packed
                                      void**   out_ptr);
NEXUSRT_API int32_t nexusrt_mem_free(void* ctx, void* ptr);
NEXUSRT_API int32_t nexusrt_mem_prefetch(void* ctx, void* ptr, uint64_t bytes);
NEXUSRT_API int32_t nexusrt_mem_advise_read_mostly(void* ctx, void* ptr, uint64_t bytes);

// ---- Scheduler (see src/scheduler/graph.hpp) -------------------------------
NEXUSRT_API int32_t nexusrt_submit_stage(void*       ctx,
                                         const char* stage_name,
                                         const char* module,
                                         const char* function,
                                         void**      inputs,  uint32_t n_inputs,
                                         void**      outputs, uint32_t n_outputs,
                                         uint32_t    token_budget,
                                         uint32_t    sm_footprint_mb,
                                         uint32_t    mem_footprint_mb,
                                         uint32_t    grid_x, uint32_t grid_y, uint32_t grid_z,
                                         uint32_t    block_x, uint32_t block_y, uint32_t block_z,
                                         uint32_t    shared_mem_bytes,
                                         void**      kernel_args, uint32_t n_args);
NEXUSRT_API int32_t nexusrt_wait_barrier(void* ctx, const char* stage_name, uint32_t timeout_ms);
NEXUSRT_API int32_t nexusrt_stream_overlap(void* ctx,
                                           const char* a, const char* b,
                                           int32_t enable);

// ---- Token optimization ----------------------------------------------------
NEXUSRT_API int32_t nexusrt_context_scope(void* ctx, const char* stage,
                                          uint32_t layer_mask,
                                          uint64_t* out_token_budget);
NEXUSRT_API int32_t nexusrt_prefetch_attention(void* ctx, void* kv_cache,
                                               uint64_t n_slots,
                                               const uint32_t* topk_indices);
NEXUSRT_API int32_t nexusrt_token_prune(void* ctx, void* kv_cache,
                                        uint32_t max_resident);

// ---- Diagnostics -----------------------------------------------------------
NEXUSRT_API int32_t nexusrt_metrics_dump(void* ctx, char* out_json, size_t cap);

#ifdef __cplusplus
} // extern "C"
#endif
