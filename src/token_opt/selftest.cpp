// =============================================================================
// NexusRT — src/token_opt/selftest.cpp
// =============================================================================
#include "token_opt/scope.hpp"
#include "token_opt/prefetcher.hpp"
#include "token_opt/kv_cache.hpp"
#include "firmware/boot.hpp"
#include <gtest/gtest.h>
using namespace nexusrt;

TEST(TokenOpt, LayerDefaults) {
    auto r = firmware::boot({});
    if (r.status == firmware::Status::DeviceNotFound) GTEST_SKIP();
    ASSERT_EQ(r.status, firmware::Status::Ok);
    token_opt::ContextScope scope(*r.ctx);
    EXPECT_EQ(scope.layer(token_opt::IcmLayer::L0_system).max_tokens, 256u);
    EXPECT_EQ(scope.layer(token_opt::IcmLayer::L4_working).max_tokens, 4096u);
    firmware::shutdown(*r.ctx);
}

TEST(TokenOpt, StepAdvancesL4) {
    auto r = firmware::boot({});
    if (r.status == firmware::Status::DeviceNotFound) GTEST_SKIP();
    ASSERT_EQ(r.status, firmware::Status::Ok);
    token_opt::ContextScope scope(*r.ctx);
    auto before = scope.layer(token_opt::IcmLayer::L4_working).bytes_used;
    scope.step(1);
    auto after = scope.layer(token_opt::IcmLayer::L4_working).bytes_used;
    EXPECT_GT(after, before);
    firmware::shutdown(*r.ctx);
}

TEST(TokenOpt, KvCacheEvict) {
    auto r = firmware::boot({});
    if (r.status == firmware::Status::DeviceNotFound) GTEST_SKIP();
    ASSERT_EQ(r.status, firmware::Status::Ok);
    token_opt::ContextScope scope(*r.ctx);
    token_opt::KvCache kv(*r.ctx, scope, /*max_resident=*/4,
                          /*page_size=*/16, /*band=*/1);
    // Touch 8 pages — should trigger eviction.
    for (uint32_t i = 0; i < 8 * 16; i += 16) kv.touch(i, 0.1f);
    EXPECT_LE(kv.resident_pages(), kv.max_resident());
    firmware::shutdown(*r.ctx);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
