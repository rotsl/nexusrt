# src/token_opt/CMakeLists.txt
cmake_minimum_required(VERSION 3.24)

set(NEXUSRT_TOKEN_OPT_SOURCES
    scope.cpp
    prefetcher.cpp
    kv_cache.cpp
)

add_library(nexusrt_token_opt STATIC ${NEXUSRT_TOKEN_OPT_SOURCES})
target_compile_features(nexusrt_token_opt PUBLIC cxx_std_17)
target_include_directories(nexusrt_token_opt
    PUBLIC ${NEXUSRT_REPO_ROOT}/src ${NEXUSRT_REPO_ROOT}/include)
target_link_libraries(nexusrt_token_opt
    PUBLIC nexusrt_firmware nexusrt_memory)

if(NEXUSRT_BUILD_TESTS)
    add_executable(nexusrt_token_opt_selftest selftest.cpp)
    target_link_libraries(nexusrt_token_opt_selftest
        PRIVATE nexusrt_token_opt GTest::gtest_main)
    gtest_discover_tests(nexusrt_token_opt_selftest)
endif()
