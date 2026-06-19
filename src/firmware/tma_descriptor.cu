// =============================================================================
// NexusRT — src/firmware/tma_descriptor.cu
// Device-side TMA descriptor helpers for Hopper.
// Encodes CUtensorMap descriptors and exposes a tiny launcher that issues
// cp.async.bulk.tensor from host code via the Driver API.
// =============================================================================
#include "firmware/types.hpp"
#include "firmware/tma_engine.hpp"
#include "platform/cuda_include.h"

#include <cuda.h>

namespace nexusrt {
namespace firmware {
namespace tma_detail {

// Encode a 2-D tiled TMA descriptor for a HBM-resident tensor.
// H100 supports up to 5-D; we expose 2-D because LLM attention and MLP
// weight matrices are 2-D.
extern "C" int nexusrt_tma_encode_2d(
    void*       host_ptr,
    uint64_t    rows,
    uint64_t    cols,
    uint32_t    elem_size,        // 2 = bf16, 4 = fp32
    uint32_t    tile_rows,
    uint32_t    tile_cols,
    int         swizzle,          // 0=None,1=B32,2=B64,3=B128
    void**      out_descriptor)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    CUtensorMap m = {};
    CUtensorMapDataType dtype =
        (elem_size == 2) ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 :
        (elem_size == 4) ? CU_TENSOR_MAP_DATA_TYPE_FLOAT32 :
                            CU_TENSOR_MAP_DATA_TYPE_UINT16;
    const uint64_t shape[2]  = {cols, rows};
    const uint64_t stride[1] = {cols * elem_size};   // bytes between rows
    const uint32_t box[2]    = {tile_cols, tile_rows};
    const uint32_t elem[2]   = {1, 1};
    CUresult r = cuTensorMapEncodeTiled(
        &m, dtype, 2,
        host_ptr,
        shape, stride,
        box, elem,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        (CUtensorMapSwizzle)swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS) return static_cast<int>(r);
    *out_descriptor = new CUtensorMap(m);
    return 0;
#else
    (void)host_ptr; (void)rows; (void)cols; (void)elem_size;
    (void)tile_rows; (void)tile_cols; (void)swizzle; (void)out_descriptor;
    return -1;   // not Hopper
#endif
}

extern "C" void nexusrt_tma_release(void* descriptor) {
    if (!descriptor) return;
    delete static_cast<CUtensorMap*>(descriptor);
}

} // namespace tma_detail
} // namespace firmware
} // namespace nexusrt
