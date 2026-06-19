// =============================================================================
// NexusRT — tests/unit/test_memory.cpp
// GoogleTest unit tests for the memory layer.
// =============================================================================
#include "memory/page_table.hpp"
#include "memory/manager.hpp"
#include "memory/coalescer.hpp"
#include "firmware/boot.hpp"

#include <gtest/gtest.h>

using namespace nexusrt;

namespace {

class MemoryFixture : public ::testing::Test {
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

TEST_F(MemoryFixture, AllocFreeRoundtrip) {
    auto& mem = r_.ctx->memory();
    memory::Allocation a;
    ASSERT_EQ(mem.alloc(1 << 16, {}, a), firmware::Status::Ok);
    EXPECT_NE(a.hbm_ptr, nullptr);
    EXPECT_EQ(mem.free(a), firmware::Status::Ok);
}

TEST_F(MemoryFixture, PageTableIncrefDecref) {
    auto& mem = r_.ctx->memory();
    memory::Allocation a;
    ASSERT_EQ(mem.alloc(1 << 16, {}, a), firmware::Status::Ok);
    auto& pt = mem.page_table();
    EXPECT_EQ(pt.incref(a.vaddr), firmware::Status::Ok);
    EXPECT_EQ(pt.decref(a.vaddr), firmware::Status::Ok);
    EXPECT_EQ(mem.free(a), firmware::Status::Ok);
}

TEST_F(MemoryFixture, PageTableEvictionRespectsRefcount) {
    auto& mem = r_.ctx->memory();
    memory::Allocation a;
    ASSERT_EQ(mem.alloc(1 << 16, {}, a), firmware::Status::Ok);
    auto& pt = mem.page_table();
    pt.incref(a.vaddr);
    uint32_t evicted = pt.evict_until(95, 50, memory::EvictionPolicy::LruRefcount);
    EXPECT_EQ(evicted, 0u);
    pt.decref(a.vaddr);
    EXPECT_EQ(mem.free(a), firmware::Status::Ok);
}

TEST_F(MemoryFixture, CoalescerRuns) {
    auto& mem = r_.ctx->memory();
    uint32_t n = 0;
    EXPECT_EQ(mem.coalesce(&n), firmware::Status::Ok);
}

TEST_F(MemoryFixture, PrefetchNoop) {
    auto& mem = r_.ctx->memory();
    memory::Allocation a;
    ASSERT_EQ(mem.alloc(1 << 16, {}, a), firmware::Status::Ok);
    EXPECT_EQ(mem.prefetch(a.hbm_ptr, 1 << 16), firmware::Status::Ok);
    EXPECT_EQ(mem.free(a), firmware::Status::Ok);
}

TEST(CoalescerAlign, WarpBoundary) {
    EXPECT_EQ(memory::Coalescer::align_to_warp(0,   firmware::Arch::Hopper), 0ull);
    EXPECT_EQ(memory::Coalescer::align_to_warp(1,   firmware::Arch::Hopper), 128ull);
    EXPECT_EQ(memory::Coalescer::align_to_warp(127, firmware::Arch::Hopper), 128ull);
    EXPECT_EQ(memory::Coalescer::align_to_warp(128, firmware::Arch::Hopper), 128ull);
    EXPECT_EQ(memory::Coalescer::align_to_warp(129, firmware::Arch::Hopper), 256ull);
}

} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
