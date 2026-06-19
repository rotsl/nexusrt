// =============================================================================
// NexusRT — src/token_opt/prefetcher.cpp
// =============================================================================
#include "token_opt/prefetcher.hpp"
#include <algorithm>
#include <numeric>

namespace nexusrt {
namespace token_opt {

void AttentionPrefetcher::observe(const std::vector<float>& row) {
    std::lock_guard<std::mutex> g(mtx_);
    history_.push_back(row);
    while (history_.size() > history_window_) history_.pop_front();
}

Status AttentionPrefetcher::prefetch_next(uint32_t k) {
    std::lock_guard<std::mutex> g(mtx_);
    if (history_.empty()) return Status::Ok;
    // Average the last N rows to get a smoother prediction.
    size_t n_slots = history_.back().size();
    std::vector<float> score(n_slots, 0.0f);
    for (auto const& r : history_) {
        for (size_t i = 0; i < r.size() && i < n_slots; ++i) score[i] += r[i];
    }
    // Argmax top-k.
    std::vector<uint32_t> idx(n_slots);
    std::iota(idx.begin(), idx.end(), 0u);
    std::partial_sort(idx.begin(), idx.begin() + std::min<size_t>(k, n_slots),
                      idx.end(),
                      [&](uint32_t a, uint32_t b) { return score[a] > score[b]; });
    idx.resize(std::min<size_t>(k, n_slots));
    return scope_.prefetch_attention(idx.data(), static_cast<uint32_t>(idx.size()));
}

} // namespace token_opt
} // namespace nexusrt
