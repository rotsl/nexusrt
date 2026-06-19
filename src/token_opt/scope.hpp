// =============================================================================
// NexusRT — src/token_opt/scope.hpp
// ICM-inspired layered context delivery.
//
// Context is divided into 5 layers (per the ICM methodology):
//   L0_system       — system prompt, pinned in HBM
//   L1_persona      — persona / role definition, pinned in HBM
//   L2_instructions — task instructions, HBM-resident
//   L3_reference    — reference docs / RAG chunks, HBM-paged
//   L4_working      — current turn artifacts (KV cache), HBM-working
//
// Each stage declares which layers it may access (config/pipeline.yaml:
// context_routing.icm_layers). The scheduler enforces the routing rules.
// =============================================================================
#pragma once

#include "firmware/types.hpp"
#include "firmware/context.hpp"
#include "memory/page_table.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace nexusrt {
namespace token_opt {

enum class IcmLayer : uint8_t {
    L0_system       = 0,
    L1_persona      = 1,
    L2_instructions = 2,
    L3_reference    = 3,
    L4_working      = 4,
    Count           = 5,
};

constexpr uint32_t kLayerCount = static_cast<uint32_t>(IcmLayer::Count);

struct LayerSpec {
    uint32_t    max_tokens = 0;
    std::string residency;          // pinned | hbm | hbm_paged | hbm_working
    uint64_t    hbm_base  = 0;      // HBM-resident base address (if pinned)
    uint64_t    bytes_used = 0;
};

// Default per-layer token budgets. Overridable via runtime.yaml:token_opt.icm_layers.
inline std::array<LayerSpec, kLayerCount> default_layers() {
    std::array<LayerSpec, kLayerCount> L{};
    L[0] = {  256, "pinned",      0, 0};
    L[1] = {  512, "pinned",      0, 0};
    L[2] = { 1024, "hbm",         0, 0};
    L[3] = { 8192, "hbm_paged",   0, 0};
    L[4] = { 4096, "hbm_working", 0, 0};
    return L;
}

// ContextScope — one per active inference. Tracks token usage per layer
// and exposes the attention prefetcher + KV-cache pruner.
class ContextScope {
public:
    explicit ContextScope(FirmwareContext& ctx);

    // Set the per-layer budget. Called once at scope construction.
    void set_layer(IcmLayer l, LayerSpec const& s);

    // Per-step entry — advances the autoregressive counter, triggers the
    // attention prefetcher, and prunes the KV cache if over budget.
    Status step(uint32_t generated_so_far);

    // Bind a KV cache region. The prefetcher and pruner operate on this.
    void bind_kv_cache(void* kv_hbm, uint64_t bytes, uint32_t max_resident);

    // Hint that the next N tokens will attend to the given top-k slots.
    // The prefetcher issues HBM prefetches for those slots.
    Status prefetch_attention(const uint32_t* topk_indices, uint32_t k);

    // Prune the KV cache down to max_resident tokens.
    Status prune(uint32_t max_resident);

    // Token budget for this scope across all layers.
    uint64_t total_token_budget() const;
    uint64_t total_bytes_used()   const;

    // Per-layer diagnostics.
    LayerSpec const& layer(IcmLayer l) const { return layers_[static_cast<size_t>(l)]; }

private:
    FirmwareContext& ctx_;
    std::array<LayerSpec, kLayerCount> layers_;
    mutable std::mutex mtx_;
    void*    kv_hbm_         = nullptr;
    uint64_t kv_bytes_       = 0;
    uint32_t max_resident_   = 0;
    uint32_t current_resident_ = 0;
    std::vector<float> attention_history_;
};

} // namespace token_opt
} // namespace nexusrt
