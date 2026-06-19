// =============================================================================
// NexusRT — src/firmware/selftest.cpp
// GoogleTest-based self-test for the firmware-equivalent layer.
// =============================================================================
#include "firmware/boot.hpp"
#include "firmware/context.hpp"
#include "firmware/fault_handler.hpp"
#include "firmware/microkernel.hpp"

#include <gtest/gtest.h>
#include <cstring>

using namespace nexusrt::firmware;

TEST(Firmware, BootDetectsDevice) {
    BootOptions opts;
    opts.profile = "auto";
    auto r = boot(opts);
    if (!ok(r.status)) {
        if (r.status == Status::DeviceNotFound) {
            GTEST_SKIP() << "No GPU detected — skipping on this host";
        }
        FAIL() << "boot failed: " << status_string(r.status);
    }
    ASSERT_NE(r.device.vendor, Vendor::Unknown);
    ASSERT_GT(r.device.hbm_capacity_bytes, 0u);
    EXPECT_EQ(r.ctx.use_count(), 1);
    EXPECT_EQ(shutdown(*r.ctx), Status::Ok);
}

TEST(Firmware, BootEmitsEvents) {
    BootOptions opts;
    std::vector<BootPhase> seen;
    opts.on_event = [&](BootEvent const& e) { seen.push_back(e.phase); };
    auto r = boot(opts);
    if (r.status == Status::DeviceNotFound) GTEST_SKIP();
    ASSERT_EQ(r.status, Status::Ok);
    EXPECT_EQ(seen.front(), BootPhase::PreInit);
    EXPECT_EQ(seen.back(),  BootPhase::Ready);
    shutdown(*r.ctx);
}

TEST(Firmware, StreamPoolAcquireRelease) {
    auto r = boot({});
    if (r.status == Status::DeviceNotFound) GTEST_SKIP();
    ASSERT_EQ(r.status, Status::Ok);
    auto& mk = r.ctx->microkernel();
    auto s1 = mk.acquire_stream(StreamClass::Compute);
    ASSERT_NE(s1.id, 0u);
    auto s2 = mk.acquire_stream(StreamClass::Compute);
    ASSERT_NE(s2.id, 0u);
    ASSERT_NE(s1.id, s2.id);
    mk.release_stream(s1);
    mk.release_stream(s2);
    // Acquire again — should reuse a freed slot.
    auto s3 = mk.acquire_stream(StreamClass::Compute);
    ASSERT_NE(s3.id, 0u);
    mk.release_stream(s3);
    shutdown(*r.ctx);
}

TEST(Firmware, WarpSpecDefaults) {
    auto r = boot({});
    if (r.status == Status::DeviceNotFound) GTEST_SKIP();
    ASSERT_EQ(r.status, Status::Ok);
    auto const& ws = r.ctx->microkernel().default_warp_spec();
    EXPECT_EQ(ws.total_warps(), 16u);
    shutdown(*r.ctx);
}

TEST(Firmware, FaultHandlerPolls) {
    BootOptions opts;
    opts.fault_buffer_slots = 16;
    auto r = boot(opts);
    if (r.status == Status::DeviceNotFound) GTEST_SKIP();
    ASSERT_EQ(r.status, Status::Ok);
    auto& fh = r.ctx->faults();
    EXPECT_EQ(fh.inject_fetch_hint(0, 4096), Status::Ok);
    shutdown(*r.ctx);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
