// =============================================================================
// NexusRT — src/pipeline/selftest.cpp
// =============================================================================
#include "pipeline/orchestrator.hpp"
#include "firmware/boot.hpp"
#include <gtest/gtest.h>
using namespace nexusrt;

TEST(Pipeline, BuildValidate) {
    auto r = firmware::boot({});
    if (r.status == firmware::Status::DeviceNotFound) GTEST_SKIP();
    ASSERT_EQ(r.status, firmware::Status::Ok);
    pipeline::PipelineConfig cfg; cfg.dry_run = true;
    pipeline::Pipeline p(*r.ctx, cfg);
    EXPECT_EQ(p.build(), firmware::Status::Ok);
    EXPECT_EQ(p.run(),   firmware::Status::Ok);
    firmware::shutdown(*r.ctx);
}

TEST(Pipeline, InferLoopProducesTokens) {
    auto r = firmware::boot({});
    if (r.status == firmware::Status::DeviceNotFound) GTEST_SKIP();
    ASSERT_EQ(r.status, firmware::Status::Ok);
    pipeline::Pipeline p(*r.ctx, {});
    std::vector<int32_t> out;
    EXPECT_EQ(p.run_infer({1,2,3}, 5, out), firmware::Status::Ok);
    EXPECT_EQ(out.size(), 8u);
    firmware::shutdown(*r.ctx);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
