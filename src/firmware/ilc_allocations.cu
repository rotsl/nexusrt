// =============================================================================
// NexusRT — src/firmware/ilc_allocations.cu
// Inline Compression allocation wrappers — H100 only.
// =============================================================================
#include "firmware/types.hpp"
#include "firmware/context.hpp"
#include "platform/cuda_include.h"

#include <cuda.h>

namespace nexusrt {
namespace firmware {

// Allocate a compressed HBM region. On Hopper this is done by setting
// CU_MEM_ALLOCATION_PROP_COMPRESSED in the CUmemAllocationProp::allocFlags
// at cuMemCreate time.
extern "C" int nexusrt_ilc_alloc(void*    ctx_v,
                                 uint64_t bytes,
                                 void**   out_ptr)
{
    auto ctx = reinterpret_cast<FirmwareContext*>(ctx_v);
    if (!ctx) return -1;

    CUmemGenericAllocationHandle h;
    CUmemAllocationProp prop = {};
    prop.type                  = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type         = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id           = 0;
    prop.allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
    CUresult r = cuMemCreate(&h, bytes, &prop, 0);
    if (r != CUDA_SUCCESS) return static_cast<int>(r);

    CUdeviceptr addr = 0;
    r = cuMemAddressReserve(&addr, bytes, 0, 0, 0);
    if (r != CUDA_SUCCESS) { cuMemRelease(h); return static_cast<int>(r); }
    r = cuMemMap(addr, bytes, 0, h, 0);
    if (r != CUDA_SUCCESS) {
        cuMemAddressFree(addr, bytes);
        cuMemRelease(h);
        return static_cast<int>(r);
    }
    CUmemAccessDesc acc = {};
    acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    acc.location.id   = 0;
    acc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    cuMemSetAccess(addr, bytes, &acc, 1);
    *out_ptr = (void*)addr;
    return 0;
}

extern "C" int nexusrt_ilc_free(void* ptr, uint64_t bytes) {
    if (!ptr) return 0;
    cuMemUnmap((CUdeviceptr)ptr, bytes);
    cuMemAddressFree((CUdeviceptr)ptr, bytes);
    // Note: the allocation handle is leaked here; in production we would
    // track it in the IlcManager's tagged_ set and release on untag.
    return 0;
}

} // namespace firmware
} // namespace nexusrt
