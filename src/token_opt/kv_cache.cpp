// =============================================================================
// NexusRT — src/token_opt/kv_cache.cpp
// =============================================================================
#include "token_opt/kv_cache.hpp"
#include <algorithm>

namespace nexusrt {
namespace token_opt {

void KvCache::touch(uint32_t token_index, float attention_weight) {
    std::lock_guard<std::mutex> g(mtx_);
    uint32_t page = token_index & ~(page_size_tokens_ - 1);
    auto& p = pages_[page];
    p.first_token_index = page;
    p.last_attention = std::max(p.last_attention, attention_weight);
    if (!p.resident) {
        if (resident_ >= max_resident_) {
            // Trigger eviction in bands.
            evict(eviction_band_);
        }
        p.resident = true;
        ++resident_;
    }
}

uint32_t KvCache::evict(uint32_t n) {
    std::lock_guard<std::mutex> g(mtx_);
    // Build a sorted list of eviction candidates (lowest attention first,
    // refcount == 0 only).
    std::vector<std::pair<float, uint32_t>> cands;
    for (auto const& [idx, p] : pages_) {
        if (p.resident && p.refcount == 0) {
            cands.emplace_back(p.last_attention, idx);
        }
    }
    std::sort(cands.begin(), cands.end());
    uint32_t evicted = 0;
    for (uint32_t i = 0; i < n && i < cands.size(); ++i) {
        auto& p = pages_[cands[i].second];
        p.resident = false;
        p.last_attention = 0.0f;
        --resident_;
        ++evicted;
    }
    return evicted;
}

void KvCache::pin(uint32_t token_index) {
    std::lock_guard<std::mutex> g(mtx_);
    uint32_t page = token_index & ~(page_size_tokens_ - 1);
    pages_[page].refcount++;
}

void KvCache::unpin(uint32_t token_index) {
    std::lock_guard<std::mutex> g(mtx_);
    uint32_t page = token_index & ~(page_size_tokens_ - 1);
    auto it = pages_.find(page);
    if (it == pages_.end()) return;
    if (it->second.refcount > 0) it->second.refcount--;
}

} // namespace token_opt
} // namespace nexusrt
