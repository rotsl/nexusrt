// =============================================================================
// NexusRT — tests/unit/test_scheduler.cpp
// =============================================================================
#include "scheduler/graph.hpp"
#include "scheduler/warp_specialization.hpp"
#include "scheduler/stream_pool.hpp"
#include "firmware/boot.hpp"

#include <gtest/gtest.h>

using namespace nexusrt;

namespace {

class SchedFixture : public ::testing::Test {
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

TEST_F(SchedFixture, AddStageAndValidate) {
    scheduler::TaskGraph g(*r_.ctx);
    scheduler::StageContract a; a.name = "a"; a.module = "m"; a.function = "f";
    scheduler::StageContract b; b.name = "b"; b.module = "m"; b.function = "f";
    b.depends_on = {"a"};
    EXPECT_EQ(g.add_stage(a), firmware::Status::Ok);
    EXPECT_EQ(g.add_stage(b), firmware::Status::Ok);
    EXPECT_EQ(g.validate(), firmware::Status::Ok);
}

TEST_F(SchedFixture, RejectCycle) {
    scheduler::TaskGraph g(*r_.ctx);
    scheduler::StageContract a; a.name = "a"; a.depends_on = {"b"};
    scheduler::StageContract b; b.name = "b"; b.depends_on = {"a"};
    g.add_stage(a); g.add_stage(b);
    EXPECT_NE(g.validate(), firmware::Status::Ok);
}

TEST_F(SchedFixture, RejectsDuplicateName) {
    scheduler::TaskGraph g(*r_.ctx);
    scheduler::StageContract a; a.name = "a"; a.module = "m"; a.function = "f";
    EXPECT_EQ(g.add_stage(a), firmware::Status::Ok);
    EXPECT_NE(g.add_stage(a), firmware::Status::Ok);
}

TEST_F(SchedFixture, ScopedStreamAcquiresAndReleases) {
    scheduler::ScopedStream s(*r_.ctx, firmware::StreamClass::Compute);
    EXPECT_TRUE(static_cast<bool>(s));
    EXPECT_NE(s.raw(), nullptr);
}

TEST_F(SchedFixture, SetOverlap) {
    scheduler::TaskGraph g(*r_.ctx);
    scheduler::StageContract a; a.name = "a"; a.module = "m"; a.function = "f";
    scheduler::StageContract b; b.name = "b"; a.module = "m"; b.function = "f";
    g.add_stage(a); g.add_stage(b);
    EXPECT_EQ(g.set_overlap("a", "b", true), firmware::Status::Ok);
}

TEST(WarpSpec, HopperDefaults) {
    auto ws = scheduler::WarpSpecPolicy::for_arch(firmware::Arch::Hopper);
    EXPECT_TRUE(ws.use_tma);
    EXPECT_EQ(ws.cluster_size, 8u);
    EXPECT_EQ(ws.total_warps(), 16u);
}

TEST(WarpSpec, AmpereDefaults) {
    auto ws = scheduler::WarpSpecPolicy::for_arch(firmware::Arch::Ampere);
    EXPECT_FALSE(ws.use_tma);
    EXPECT_FALSE(ws.use_dsm);
    EXPECT_EQ(ws.total_warps(), 16u);
}

TEST(WarpSpec, AppleDefaults) {
    auto ws = scheduler::WarpSpecPolicy::for_arch(firmware::Arch::AppleSilicon);
    EXPECT_EQ(ws.producer_warps, 0u);
    EXPECT_EQ(ws.consumer_warps, 16u);
}

} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
