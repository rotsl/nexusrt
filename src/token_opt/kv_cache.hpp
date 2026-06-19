// =============================================================================
// NexusRT — src/token_opt/kv_cache.hpp
// KV cache manager with attention-weighted LRU eviction.
//
// The cache is organized as PagedAttention-style token pages: each page
// holds `page_size_tokens` (default 16) tokens. Pages are evicted in bands
// of `eviction_band` (default 2) to amortize TLB cost.
// =============================================================================
#pragma once

#include "firmware/types.hpp"
#include "firmware/context.hpp"
#include "token_opt/scope.hpp"

#include <map>
#include <mutex>
#include <vector>

namespace nexusrt {
namespace token_opt {

struct KvPage {
    uint32_t    first_token_index = 0;
    uint32_t    refcount          = 0;
    float       last_attention    = 0.0f;
    bool        resident          = false;
};

class KvCache {
public:
    KvCache(FirmwareContext& ctx, ContextScope& scope,
            uint32_t max_resident, uint32_t page_size_tokens,
            uint32_t eviction_band)
        : ctx_(ctx), scope_(scope),
          max_resident_(max_resident),
          page_size_tokens_(page_size_tokens),
          eviction_band_(eviction_band) {}

    // Mark a page as accessed at the given attention weight. Pages whose
    // attention weight is below `score_threshold` (runtime.yaml) are
    // eligible for eviction.
    void touch(uint32_t token_index, float attention_weight);

    // Evict up to `n` pages. Returns the number actually evicted.
    uint32_t evict(uint32_t n);

    // Pin / unpin a page (refcount). Pinned pages are never evicted.
    void pin  (uint32_t token_index);
    void unpin(uint32_t token_index);

    uint32_t resident_pages()   const { return resident_; }
    uint32_t max_resident()     const { return max_resident_; }

private:
    FirmwareContext& ctx_;
    ContextScope&    scope_;
    uint32_t         max_resident_;
    uint32_t         page_size_tokens_;
    uint32_t         eviction_band_;
    uint32_t         resident_ = 0;
    mutable std::mutex mtx_;
    std::map<uint32_t, KvPage> pages_;   // keyed by page-aligned token index
};

} // namespace token_opt
} // namespace nexusrt
