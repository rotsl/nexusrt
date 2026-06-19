# =============================================================================
# NexusRT — top-level CMakeLists.txt
# Builds the C++17 core + optional CUDA / Metal backends + Python bindings.
# =============================================================================
cmake_minimum_required(VERSION 3.24)

project(nexusrt
    VERSION 0.1.0
    LANGUAGES C CXX
    DESCRIPTION "Firmware-centric, OS-bypass runtime for end-to-end LLM pipelines"
    HOMEPAGE_URL "https://github.com/nexusrt/nexusrt"
)

get_filename_component(NEXUSRT_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# ---- Options ---------------------------------------------------------------
option(NEXUSRT_ENABLE_CUDA   "Enable NVIDIA CUDA backend"              AUTO)
option(NEXUSRT_ENABLE_METAL  "Enable Apple Metal backend (macOS only)" AUTO)
option(NEXUSRT_BUILD_TESTS   "Build unit tests (requires GoogleTest)"  OFF)
option(NEXUSRT_BUILD_BENCHMARKS "Build benchmarks"                     OFF)
option(NEXUSRT_BUILD_DOCS    "Build documentation (requires Sphinx)"   OFF)
option(NEXUSRT_ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)
option(NEXUSRT_ENABLE_LTO    "Enable link-time optimization"           OFF)

# ---- C++ standard ----------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NEXUSRT_ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ---- Default build type ----------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---- Output dirs -----------------------------------------------------------
set(NEXUSRT_OUTPUT_DIR ${CMAKE_BINARY_DIR}/outputs)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${NEXUSRT_OUTPUT_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${NEXUSRT_OUTPUT_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${NEXUSRT_OUTPUT_DIR}/bin)

# ---- Include paths ---------------------------------------------------------
include_directories(${NEXUSRT_REPO_ROOT}/src)
include_directories(${NEXUSRT_REPO_ROOT}/include)

# ---- CUDA detection --------------------------------------------------------
if(NEXUSRT_ENABLE_CUDA)
    find_package(CUDAToolkit 12.0 QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "NexusRT: CUDA ${CUDAToolkit_VERSION} found")
        set(NEXUSRT_ENABLE_CUDA ON CACHE BOOL "" FORCE)
        enable_language(CUDA)
        # Kaggle commonly exposes P100 (sm_60) or T4 (sm_75); local validation
        # may also use A100 (sm_80) or H100 (sm_90).
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "60;75;80;90")
        endif()
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        add_compile_definitions(NEXUSRT_HAVE_CUDA=1)
        include_directories(${CUDAToolkit_INCLUDE_DIRS})
    elseif(NEXUSRT_ENABLE_CUDA STREQUAL "AUTO")
        message(STATUS "NexusRT: CUDA not found — building CPU/Metal-only")
        set(NEXUSRT_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
    else()
        message(FATAL_ERROR "NexusRT: NEXUSRT_ENABLE_CUDA=ON but CUDA not found")
    endif()
endif()

# ---- Metal detection -------------------------------------------------------
if(APPLE AND NEXUSRT_ENABLE_METAL)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(METALKIT_FRAMEWORK MetalKit REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    message(STATUS "NexusRT: Metal framework found")
    set(NEXUSRT_ENABLE_METAL ON CACHE BOOL "" FORCE)
    enable_language(OBJCXX)
    add_compile_definitions(NEXUSRT_HAVE_METAL=1)
elseif(NEXUSRT_ENABLE_METAL STREQUAL "AUTO")
    set(NEXUSRT_ENABLE_METAL OFF CACHE BOOL "" FORCE)
endif()

# ---- Coverage --------------------------------------------------------------
if(NEXUSRT_ENABLE_COVERAGE)
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
endif()

# ---- GoogleTest ------------------------------------------------------------
if(NEXUSRT_BUILD_TESTS)
    # FetchContent is preferred; fall back to system-installed gtest.
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    )
    set(gtest_force_shared_rt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    include(GoogleTest)
    enable_testing()
endif()

# ---- Subdirectories --------------------------------------------------------
add_subdirectory(${NEXUSRT_REPO_ROOT}/src/firmware   ${CMAKE_BINARY_DIR}/firmware)
add_subdirectory(${NEXUSRT_REPO_ROOT}/src/platform   ${CMAKE_BINARY_DIR}/platform)
add_subdirectory(${NEXUSRT_REPO_ROOT}/src/memory     ${CMAKE_BINARY_DIR}/memory)
add_subdirectory(${NEXUSRT_REPO_ROOT}/src/scheduler  ${CMAKE_BINARY_DIR}/scheduler)
add_subdirectory(${NEXUSRT_REPO_ROOT}/src/token_opt  ${CMAKE_BINARY_DIR}/token_opt)
add_subdirectory(${NEXUSRT_REPO_ROOT}/src/pipeline   ${CMAKE_BINARY_DIR}/pipeline)

if(NEXUSRT_BUILD_TESTS)
    add_subdirectory(${NEXUSRT_REPO_ROOT}/tests/unit ${CMAKE_BINARY_DIR}/tests/unit)
endif()

# GNU ld scans static archives from left to right, while these core libraries
# intentionally reference each other across firmware, platform, and memory
# boundaries. Use a linker rescan group where available, with a repeated-list
# fallback for linkers that do not expose start/end-group semantics.
set(NEXUSRT_CORE_STATIC_LIBS
    nexusrt_firmware
    nexusrt_platform
    nexusrt_memory
    nexusrt_scheduler
    nexusrt_token_opt
    nexusrt_pipeline
)
list(JOIN NEXUSRT_CORE_STATIC_LIBS "," NEXUSRT_CORE_LINK_GROUP_ITEMS)

function(nexusrt_link_core_static target)
    if(NOT TARGET ${target})
        return()
    endif()
    if(APPLE)
        return()
    elseif(CMAKE_CXX_LINK_GROUP_USING_RESCAN_SUPPORTED)
        target_link_libraries(${target}
            PRIVATE "$<LINK_GROUP:RESCAN,${NEXUSRT_CORE_LINK_GROUP_ITEMS}>")
    else()
        target_link_libraries(${target}
            PRIVATE ${NEXUSRT_CORE_STATIC_LIBS} ${NEXUSRT_CORE_STATIC_LIBS})
    endif()
endfunction()

if(NEXUSRT_BUILD_TESTS)
    foreach(t
        nexusrt_firmware_selftest
        nexusrt_memory_selftest
        nexusrt_scheduler_selftest
        nexusrt_token_opt_selftest
        nexusrt_pipeline_selftest
        test_firmware
        test_memory
        test_scheduler
        test_token_opt)
        nexusrt_link_core_static(${t})
    endforeach()
endif()

# ---- Shared library target -------------------------------------------------
add_library(nexusrt SHARED
    ${NEXUSRT_REPO_ROOT}/src/platform/abi_stubs.cpp
    ${NEXUSRT_REPO_ROOT}/src/firmware/boot.cpp
    ${NEXUSRT_REPO_ROOT}/src/firmware/fault_handler.cpp
    ${NEXUSRT_REPO_ROOT}/src/memory/manager.cpp
    ${NEXUSRT_REPO_ROOT}/src/scheduler/graph.cpp
    ${NEXUSRT_REPO_ROOT}/src/token_opt/scope.cpp
)
if(NEXUSRT_ENABLE_CUDA)
    target_sources(nexusrt PRIVATE
        ${NEXUSRT_REPO_ROOT}/src/firmware/cuda_driver_shim.cu
    )
endif()
target_link_libraries(nexusrt PUBLIC
    nexusrt_platform
    nexusrt_firmware
    nexusrt_memory
    nexusrt_scheduler
    nexusrt_token_opt
    nexusrt_pipeline
)
nexusrt_link_core_static(nexusrt)
set_target_properties(nexusrt PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)

# ---- RPATH -----------------------------------------------------------------
# On Linux, install the .so to lib/ and set RPATH to $ORIGIN.
set_target_properties(nexusrt PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

# ---- Install rules ---------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS nexusrt
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY ${NEXUSRT_REPO_ROOT}/src/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nexusrt
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h" PATTERN "*.cuh")
install(DIRECTORY ${NEXUSRT_REPO_ROOT}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nexusrt)
install(FILES ${NEXUSRT_REPO_ROOT}/config/hardware.yaml
              ${NEXUSRT_REPO_ROOT}/config/runtime.yaml
              ${NEXUSRT_REPO_ROOT}/config/pipeline.yaml
        DESTINATION ${CMAKE_INSTALL_SYSCONFDIR}/nexusrt)

# ---- pkg-config file -------------------------------------------------------
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/nexusrt.pc.in
    ${CMAKE_BINARY_DIR}/nexusrt.pc
    @ONLY
)
install(FILES ${CMAKE_BINARY_DIR}/nexusrt.pc
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

# ---- Print summary ---------------------------------------------------------
message(STATUS "")
message(STATUS "=== NexusRT ${PROJECT_VERSION} configuration summary ===")
message(STATUS "  Build type         : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA backend       : ${NEXUSRT_ENABLE_CUDA}")
if(NEXUSRT_ENABLE_CUDA)
    message(STATUS "  CUDA version       : ${CUDAToolkit_VERSION}")
    message(STATUS "  CUDA architectures : ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  Metal backend      : ${NEXUSRT_ENABLE_METAL}")
message(STATUS "  Build tests        : ${NEXUSRT_BUILD_TESTS}")
message(STATUS "  Build benchmarks   : ${NEXUSRT_BUILD_BENCHMARKS}")
message(STATUS "  Build docs         : ${NEXUSRT_BUILD_DOCS}")
message(STATUS "  Coverage           : ${NEXUSRT_ENABLE_COVERAGE}")
message(STATUS "  LTO                : ${NEXUSRT_ENABLE_LTO}")
message(STATUS "  Output dir         : ${NEXUSRT_OUTPUT_DIR}")
message(STATUS "=======================================================")
