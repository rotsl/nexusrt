# src/memory/CMakeLists.txt
cmake_minimum_required(VERSION 3.24)

set(NEXUSRT_MEMORY_SOURCES
    page_table.cpp
    manager.cpp
    coalescer.cpp
    gds_router.cpp
    grdma_router.cpp
)

add_library(nexusrt_memory STATIC ${NEXUSRT_MEMORY_SOURCES})
target_compile_features(nexusrt_memory PUBLIC cxx_std_17)
target_include_directories(nexusrt_memory
    PUBLIC ${NEXUSRT_REPO_ROOT}/src ${NEXUSRT_REPO_ROOT}/include)
target_link_libraries(nexusrt_memory
    PUBLIC nexusrt_firmware nexusrt_platform)

if(NEXUSRT_BUILD_TESTS)
    add_executable(nexusrt_memory_selftest selftest.cpp)
    target_link_libraries(nexusrt_memory_selftest
        PRIVATE nexusrt_memory GTest::gtest_main)
    gtest_discover_tests(nexusrt_memory_selftest)
endif()
