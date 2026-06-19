// =============================================================================
// NexusRT — src/scheduler/selftest.cpp
// =============================================================================
#include "scheduler/graph.hpp"
#include "scheduler/warp_specialization.hpp"
#include "firmware/boot.hpp"

#include <gtest/gtest.h>

using namespace nexusrt;

TEST(Scheduler, WarpSpecHopper) {
    auto ws = scheduler::WarpSpecPolicy::for_arch(firmware::Arch::Hopper);
    EXPECT_TRUE(ws.use_tma);
    EXPECT_TRUE(ws.use_dsm);
    EXPECT_EQ(ws.cluster_size, 8u);
    EXPECT_EQ(ws.total_warps(), 16u);
}

TEST(Scheduler, WarpSpecAmpere) {
    auto ws = scheduler::WarpSpecPolicy::for_arch(firmware::Arch::Ampere);
    EXPECT_FALSE(ws.use_tma);
    EXPECT_EQ(ws.total_warps(), 16u);
}

TEST(Scheduler, TaskGraphAddValidate) {
    auto r = firmware::boot({});
    if (r.status == firmware::Status::DeviceNotFound) GTEST_SKIP();
    ASSERT_EQ(r.status, firmware::Status::Ok);
    scheduler::TaskGraph g(*r.ctx);
    scheduler::StageContract a; a.name = "a"; a.module = "m"; a.function = "f";
    scheduler::StageContract b; b.name = "b"; b.module = "m"; b.function = "f";
    b.depends_on = {"a"};
    EXPECT_EQ(g.add_stage(a), firmware::Status::Ok);
    EXPECT_EQ(g.add_stage(b), firmware::Status::Ok);
    EXPECT_EQ(g.validate(), firmware::Status::Ok);
    firmware::shutdown(*r.ctx);
}

TEST(Scheduler, TaskGraphRejectsCycle) {
    auto r = firmware::boot({});
    if (r.status == firmware::Status::DeviceNotFound) GTEST_SKIP();
    ASSERT_EQ(r.status, firmware::Status::Ok);
    scheduler::TaskGraph g(*r.ctx);
    scheduler::StageContract a; a.name = "a"; a.depends_on = {"b"};
    scheduler::StageContract b; b.name = "b"; b.depends_on = {"a"};
    g.add_stage(a); g.add_stage(b);
    EXPECT_NE(g.validate(), firmware::Status::Ok);
    firmware::shutdown(*r.ctx);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
