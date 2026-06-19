# src/pipeline/CMakeLists.txt
cmake_minimum_required(VERSION 3.24)

set(NEXUSRT_PIPELINE_SOURCES
    orchestrator.cpp
    preprocess.cpp
    train.cpp
    infer.cpp
    postprocess.cpp
)

if(NEXUSRT_ENABLE_CUDA)
    list(APPEND NEXUSRT_PIPELINE_SOURCES
        kernels/tokenize_kernel.cu
        kernels/transformer_kernel.cu
        kernels/decode_kernel.cu
        kernels/detokenize_kernel.cu
    )
endif()

add_library(nexusrt_pipeline STATIC ${NEXUSRT_PIPELINE_SOURCES})
target_compile_features(nexusrt_pipeline PUBLIC cxx_std_17)
target_include_directories(nexusrt_pipeline
    PUBLIC ${NEXUSRT_REPO_ROOT}/src ${NEXUSRT_REPO_ROOT}/include)
target_link_libraries(nexusrt_pipeline
    PUBLIC nexusrt_firmware nexusrt_memory nexusrt_scheduler
           nexusrt_platform nexusrt_token_opt)

if(NEXUSRT_ENABLE_CUDA)
    set_source_files_properties(kernels/tokenize_kernel.cu
                                kernels/transformer_kernel.cu
                                kernels/decode_kernel.cu
                                kernels/detokenize_kernel.cu
        PROPERTIES LANGUAGE CUDA)
endif()

if(NEXUSRT_BUILD_TESTS)
    add_executable(nexusrt_pipeline_selftest selftest.cpp)
    target_link_libraries(nexusrt_pipeline_selftest
        PRIVATE nexusrt_pipeline GTest::gtest_main)
    gtest_discover_tests(nexusrt_pipeline_selftest)
endif()
