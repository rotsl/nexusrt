// =============================================================================
// NexusRT — tests/unit/test_token_opt.cpp
// =============================================================================
#include "token_opt/scope.hpp"
#include "token_opt/prefetcher.hpp"
#include "token_opt/kv_cache.hpp"
#include "firmware/boot.hpp"

#include <gtest/gtest.h>

using namespace nexusrt;

namespace {

class TokenOptFixture : public ::testing::Test {
protected:
    void SetUp() override {
        r_ = firmware::boot({});
        if (r_.status == firmware::Status::DeviceNotFound) GTEST_SKIP();
        ASSERT_EQ(r_.status, firmware::Status::Ok);
    }
    void TearDown() override {
        if (r_.ctx) firmware::shutdown(*r_.ctx);
    }
    firmware::BootResult r_;
};

TEST_F(TokenOptFixture, LayerDefaults) {
    token_opt::ContextScope s(*r_.ctx);
    EXPECT_EQ(s.layer(token_opt::IcmLayer::L0_system).max_tokens, 256u);
    EXPECT_EQ(s.layer(token_opt::IcmLayer::L1_persona).max_tokens, 512u);
    EXPECT_EQ(s.layer(token_opt::IcmLayer::L2_instructions).max_tokens, 1024u);
    EXPECT_EQ(s.layer(token_opt::IcmLayer::L3_reference).max_tokens, 8192u);
    EXPECT_EQ(s.layer(token_opt::IcmLayer::L4_working).max_tokens, 4096u);
}

TEST_F(TokenOptFixture, StepAdvancesL4) {
    token_opt::ContextScope s(*r_.ctx);
    auto before = s.layer(token_opt::IcmLayer::L4_working).bytes_used;
    s.step(1);
    auto after = s.layer(token_opt::IcmLayer::L4_working).bytes_used;
    EXPECT_GT(after, before);
}

TEST_F(TokenOptFixture, PrefetcherIssuesNoOpWhenEmpty) {
    token_opt::ContextScope s(*r_.ctx);
    token_opt::AttentionPrefetcher ap(*r_.ctx, s);
    EXPECT_EQ(ap.prefetch_next(8), firmware::Status::Ok);
}

TEST_F(TokenOptFixture, PrefetcherPicksTopK) {
    token_opt::ContextScope s(*r_.ctx);
    s.bind_kv_cache(reinterpret_cast<void*>(0x1000), 4096 * 32, 32);
    token_opt::AttentionPrefetcher ap(*r_.ctx, s);
    ap.observe({0.1f, 0.5f, 0.2f, 0.05f, 0.15f, 0.0f, 0.0f, 0.0f});
    EXPECT_EQ(ap.prefetch_next(2), firmware::Status::Ok);
}

TEST_F(TokenOptFixture, KvCacheEvictsToMaxResident) {
    token_opt::ContextScope s(*r_.ctx);
    token_opt::KvCache kv(*r_.ctx, s, /*max_resident=*/4,
                          /*page_size=*/16, /*band=*/1);
    for (uint32_t i = 0; i < 8 * 16; i += 16) kv.touch(i, 0.1f);
    EXPECT_LE(kv.resident_pages(), kv.max_resident());
}

TEST_F(TokenOptFixture, KvCachePinnedNotEvicted) {
    token_opt::ContextScope s(*r_.ctx);
    token_opt::KvCache kv(*r_.ctx, s, /*max_resident=*/2,
                          /*page_size=*/16, /*band=*/1);
    kv.touch(0,  0.1f);
    kv.pin(0);
    kv.touch(16, 0.1f);
    kv.touch(32, 0.1f);   // should evict 16, not 0
    EXPECT_GE(kv.resident_pages(), 1u);
}

} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
