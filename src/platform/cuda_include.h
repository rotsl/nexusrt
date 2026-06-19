// =============================================================================
// NexusRT — src/platform/cuda_include.h
// Central CUDA header include guard. Allows the rest of the codebase to
// compile without CUDA headers installed (the Metal path uses this file as
// a no-op).
// =============================================================================
#pragma once

#if defined(NEXUSRT_HAVE_CUDA) && NEXUSRT_HAVE_CUDA
  #include <cuda.h>
  #define NEXUSRT_HAVE_CUDA_HOPPER 1
#else
  // Stubs so non-CUDA compiles succeed.
  typedef int CUresult;
  typedef struct CUstream_st* CUstream;
  typedef struct CUcontext_st* CUcontext;
  typedef struct CUmodule_st* CUmodule;
  typedef struct CUfunction_st* CUfunction;
  typedef struct CUmemGenericAllocationHandle_st* CUmemGenericAllocationHandle;
  typedef struct CUtensorMap_st* CUtensorMap;
  typedef unsigned long long CUdeviceptr;
  typedef int CUdevice;
  enum { CUDA_SUCCESS = 0 };
  #define CUDA_ERROR_OUT_OF_MEMORY 1
  #define CUDA_ERROR_INVALID_VALUE 2
  #define CUDA_ERROR_NOT_FOUND 3
  #define CUDA_ERROR_NOT_SUPPORTED 4
  #define NEXUSRT_HAVE_CUDA_HOPPER 0
#endif
