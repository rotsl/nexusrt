# =============================================================================
# NexusRT — tests/unit/CMakeLists.txt
# =============================================================================
cmake_minimum_required(VERSION 3.24)

# C++ unit tests — one binary per module. Each links against the matching
# static library + GoogleTest.

set(NEXUSRT_UNIT_TESTS
    test_firmware
    test_memory
    test_scheduler
    test_token_opt
)

foreach(t ${NEXUSRT_UNIT_TESTS})
    add_executable(${t} ${t}.cpp)
    target_link_libraries(${t}
        PRIVATE
            nexusrt_firmware
            nexusrt_memory
            nexusrt_scheduler
            nexusrt_token_opt
            nexusrt_pipeline
            nexusrt_platform
            GTest::gtest_main)
    gtest_discover_tests(${t})
endforeach()

# Coverage threshold enforcement (≥85% for core modules).
# We require gcov and run lcov after `ctest`; the threshold is checked in
# packaging/setup_hooks/post_test_coverage.sh.
