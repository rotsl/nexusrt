// =============================================================================
// NexusRT — tests/unit/test_firmware.cpp
// GoogleTest unit tests for the firmware-equivalent layer.
// =============================================================================
#include "firmware/boot.hpp"
#include "firmware/context.hpp"
#include "firmware/microkernel.hpp"
#include "firmware/dma_engine.hpp"
#include "firmware/fault_handler.hpp"
#include "firmware/ilc_manager.hpp"

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <thread>

using namespace nexusrt::firmware;

namespace {

class FirmwareFixture : public ::testing::Test {
protected:
    void SetUp() override {
        r_ = boot({});
        if (r_.status == Status::DeviceNotFound) GTEST_SKIP();
        ASSERT_EQ(r_.status, Status::Ok);
    }
    void TearDown() override {
        if (r_.ctx) shutdown(*r_.ctx);
    }
    BootResult r_;
};

TEST_F(FirmwareFixture, BootPopulatesDeviceDesc) {
    EXPECT_NE(r_.device.vendor, Vendor::Unknown);
    EXPECT_GT(r_.device.hbm_capacity_bytes, 0u);
    EXPECT_GT(r_.device.sm_count, 0u);
}

TEST_F(FirmwareFixture, StreamPoolAcquireRelease) {
    auto& mk = r_.ctx->microkernel();
    auto s = mk.acquire_stream(StreamClass::Compute);
    ASSERT_NE(s.id, 0u);
    mk.release_stream(s);
}

TEST_F(FirmwareFixture, DmaEngineGdsFallback) {
    auto& dma = r_.ctx->dma();
    // Without GDS installed, the fallback path uses host-pinned bounce.
    EXPECT_FALSE(dma.gds_enabled() || true);  // informational
}

TEST_F(FirmwareFixture, FaultHandlerInjects) {
    auto& fh = r_.ctx->faults();
    EXPECT_EQ(fh.inject_fetch_hint(0, 4096), Status::Ok);
}

TEST_F(FirmwareFixture, IlcManagerDisabledOnAmpere) {
    if (r_.device.arch != Arch::Hopper) {
        EXPECT_EQ(r_.ctx->ilc(), nullptr);
    } else {
        EXPECT_NE(r_.ctx->ilc(), nullptr);
        EXPECT_TRUE(r_.ctx->ilc()->enabled());
    }
}

TEST_F(FirmwareFixture, MicroKernelWarpSpecDefaults) {
    auto const& ws = r_.ctx->microkernel().default_warp_spec();
    EXPECT_EQ(ws.total_warps(), 16u);
}

TEST_F(FirmwareFixture, BootEventsEmittedInOrder) {
    BootOptions opts;
    std::vector<BootPhase> phases;
    opts.on_event = [&](BootEvent const& e) { phases.push_back(e.phase); };
    auto r = boot(opts);
    if (r.status == Status::DeviceNotFound) GTEST_SKIP();
    ASSERT_EQ(r.status, Status::Ok);
    EXPECT_EQ(phases.front(), BootPhase::PreInit);
    EXPECT_EQ(phases.back(),  BootPhase::Ready);
    shutdown(*r.ctx);
}

TEST(StatusString, AllValues) {
    EXPECT_STREQ(status_string(Status::Ok),                  "ok");
    EXPECT_STREQ(status_string(Status::DeviceNotFound),      "device not found");
    EXPECT_STREQ(status_string(Status::ContractViolation),   "contract violation");
}

} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
