// =============================================================================
// NexusRT — src/memory/selftest.cpp
// =============================================================================
#include "memory/page_table.hpp"
#include "memory/manager.hpp"
#include "firmware/boot.hpp"

#include <gtest/gtest.h>

using namespace nexusrt;

TEST(Memory, AllocFree) {
    auto r = firmware::boot({});
    if (r.status == firmware::Status::DeviceNotFound) GTEST_SKIP();
    ASSERT_EQ(r.status, firmware::Status::Ok);
    auto& mem = r.ctx->memory();
    memory::Allocation a;
    ASSERT_EQ(mem.alloc(1 << 20, {}, a), firmware::Status::Ok);
    EXPECT_NE(a.hbm_ptr, nullptr);
    EXPECT_EQ(mem.free(a), firmware::Status::Ok);
    firmware::shutdown(*r.ctx);
}

TEST(Memory, PageTableIncrefDecref) {
    auto r = firmware::boot({});
    if (r.status == firmware::Status::DeviceNotFound) GTEST_SKIP();
    ASSERT_EQ(r.status, firmware::Status::Ok);
    auto& mem = r.ctx->memory();
    memory::Allocation a;
    ASSERT_EQ(mem.alloc(1 << 20, {}, a), firmware::Status::Ok);
    auto& pt = mem.page_table();
    EXPECT_EQ(pt.incref(a.vaddr), firmware::Status::Ok);
    EXPECT_EQ(pt.decref(a.vaddr), firmware::Status::Ok);
    EXPECT_EQ(mem.free(a), firmware::Status::Ok);
    firmware::shutdown(*r.ctx);
}

TEST(Memory, PrefetchNoop) {
    auto r = firmware::boot({});
    if (r.status == firmware::Status::DeviceNotFound) GTEST_SKIP();
    ASSERT_EQ(r.status, firmware::Status::Ok);
    auto& mem = r.ctx->memory();
    memory::Allocation a;
    ASSERT_EQ(mem.alloc(1 << 20, {}, a), firmware::Status::Ok);
    EXPECT_EQ(mem.prefetch(a.hbm_ptr, 1 << 20), firmware::Status::Ok);
    EXPECT_EQ(mem.free(a), firmware::Status::Ok);
    firmware::shutdown(*r.ctx);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
