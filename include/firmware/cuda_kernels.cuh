// =============================================================================
// NexusRT — include/firmware/cuda_kernels.cuh
// Shared header for all NexusRT CUDA kernels. Guards against non-CUDA builds.
// =============================================================================
#pragma once

#ifdef __CUDACC__
  #include <cuda_runtime.h>
  #include <cuda_fp16.h>
#else
  // Stubs so the headers compile without nvcc.
  #include <cstdint>
  typedef short __half;
  static inline __half __float2half(float) { return 0; }
#endif
