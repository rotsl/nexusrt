# src/platform/CMakeLists.txt
cmake_minimum_required(VERSION 3.24)

set(NEXUSRT_PLATFORM_SOURCES
    dispatch.cpp
)

if(NEXUSRT_ENABLE_CUDA)
    list(APPEND NEXUSRT_PLATFORM_SOURCES cuda_driver_loader.cpp)
endif()
if(NEXUSRT_ENABLE_METAL)
    list(APPEND NEXUSRT_PLATFORM_SOURCES metal_loader.mm)
endif()

add_library(nexusrt_platform STATIC ${NEXUSRT_PLATFORM_SOURCES})
target_compile_features(nexusrt_platform PUBLIC cxx_std_17)
target_include_directories(nexusrt_platform
    PUBLIC ${NEXUSRT_REPO_ROOT}/src ${NEXUSRT_REPO_ROOT}/include)
