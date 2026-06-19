# =============================================================================
# NexusRT — src/firmware/CMakeLists.txt
# Build rules for the firmware-equivalent micro-kernel layer.
# =============================================================================
cmake_minimum_required(VERSION 3.24)

# This subdirectory builds the firmware-equivalent micro-kernel as a static
# library that is consumed by the scheduler, memory, and pipeline layers.
#
# The micro-kernel is *not* a kernel module — it is a user-space + GPU-resident
# code bundle that uses the lowest publicly available CUDA Driver APIs to
# emulate bare-metal firmware behavior. See docs/architecture.md §Firmware
# Boundary.

set(NEXUSRT_FIRMWARE_SOURCES
    boot.cpp
    microkernel.cpp
    tma_engine.cpp
    dma_engine.cpp
    fault_handler.cpp
    ilc_manager.cpp
    context.cpp
)

# CUDA Driver wrappers — only compiled when NexusRT is built with CUDA support.
if(NEXUSRT_ENABLE_CUDA)
    enable_language(CUDA)
    list(APPEND NEXUSRT_FIRMWARE_SOURCES
        cuda_driver_shim.cu
        tma_descriptor.cu
        ilc_allocations.cu
    )
endif()

# Metal wrappers — only compiled on Apple platforms.
if(NEXUSRT_ENABLE_METAL)
    enable_language(OBJCXX)
    list(APPEND NEXUSRT_FIRMWARE_SOURCES
        metal_shim.mm
    )
endif()

add_library(nexusrt_firmware STATIC ${NEXUSRT_FIRMWARE_SOURCES})
target_compile_features(nexusrt_firmware PUBLIC cxx_std_17)

target_include_directories(nexusrt_firmware
    PUBLIC
        ${NEXUSRT_REPO_ROOT}/src
        ${NEXUSRT_REPO_ROOT}/include
)

if(NEXUSRT_ENABLE_CUDA)
    if(TARGET CUDA::cuda_driver)
        target_link_libraries(nexusrt_firmware PUBLIC CUDA::cuda_driver)
    else()
        find_library(NEXUSRT_CUDA_DRIVER_LIBRARY
            NAMES cuda
            PATHS
                ${CUDAToolkit_LIBRARY_DIR}
                ${CUDAToolkit_LIBRARY_ROOT}/lib64
                /usr/local/cuda/lib64
                /usr/local/cuda/lib64/stubs
                /usr/local/cuda/compat
                /usr/local/nvidia/lib
                /usr/local/nvidia/lib64
                /usr/lib64
                /usr/lib/x86_64-linux-gnu
            NO_DEFAULT_PATH)
        if(NOT NEXUSRT_CUDA_DRIVER_LIBRARY)
            find_library(NEXUSRT_CUDA_DRIVER_LIBRARY NAMES cuda)
        endif()
        if(NOT NEXUSRT_CUDA_DRIVER_LIBRARY)
            message(FATAL_ERROR
                "NexusRT: CUDA driver library libcuda was not found. "
                "Checked CUDA toolkit, /usr/local/nvidia, /usr/local/cuda, "
                "and system library paths.")
        endif()
        target_link_libraries(nexusrt_firmware PUBLIC ${NEXUSRT_CUDA_DRIVER_LIBRARY})
    endif()
    target_compile_definitions(nexusrt_firmware PUBLIC NEXUSRT_HAVE_CUDA=1)
endif()

if(NEXUSRT_ENABLE_METAL)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
    target_link_libraries(nexusrt_firmware
        PUBLIC ${METAL_FRAMEWORK} ${METALKIT_FRAMEWORK})
    target_compile_definitions(nexusrt_firmware PUBLIC NEXUSRT_HAVE_METAL=1)
endif()

# ---- Unit tests -------------------------------------------------------------
if(NEXUSRT_BUILD_TESTS)
    add_executable(nexusrt_firmware_selftest selftest.cpp)
    target_link_libraries(nexusrt_firmware_selftest
        PRIVATE
            nexusrt_firmware
            nexusrt_platform
            nexusrt_memory
            nexusrt_scheduler
            nexusrt_token_opt
            nexusrt_pipeline
            GTest::gtest_main)
    gtest_discover_tests(nexusrt_firmware_selftest)
endif()
