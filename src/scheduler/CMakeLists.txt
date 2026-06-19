# src/scheduler/CMakeLists.txt
cmake_minimum_required(VERSION 3.24)

set(NEXUSRT_SCHEDULER_SOURCES
    graph.cpp
    warp_specialization.cpp
    stream_pool.cpp
)

add_library(nexusrt_scheduler STATIC ${NEXUSRT_SCHEDULER_SOURCES})
target_compile_features(nexusrt_scheduler PUBLIC cxx_std_17)
target_include_directories(nexusrt_scheduler
    PUBLIC ${NEXUSRT_REPO_ROOT}/src ${NEXUSRT_REPO_ROOT}/include)
target_link_libraries(nexusrt_scheduler
    PUBLIC nexusrt_firmware nexusrt_memory nexusrt_platform)

if(NEXUSRT_BUILD_TESTS)
    add_executable(nexusrt_scheduler_selftest selftest.cpp)
    target_link_libraries(nexusrt_scheduler_selftest
        PRIVATE nexusrt_scheduler GTest::gtest_main)
    gtest_discover_tests(nexusrt_scheduler_selftest)
endif()
