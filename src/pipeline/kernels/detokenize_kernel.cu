// =============================================================================
// NexusRT — src/pipeline/kernels/detokenize_kernel.cu
// Detokenize: map int32 token IDs back to bytes. Output is routed to host
// only on the final answer (config/pipeline.yaml: postprocess.host_copy_only_on_final).
// =============================================================================
#include "firmware/cuda_kernels.cuh"

#include <cstdint>

namespace nexusrt { namespace kernels {

__global__ void detokenize_format(
    const int32_t* __restrict__ tokens,
    uint8_t*       __restrict__ out_bytes,
    uint64_t                    n_tokens)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_tokens) return;
    int32_t t = tokens[i];
    out_bytes[i] = (t >= 32 && t < 127) ? (uint8_t)t : (uint8_t)' ';
}

}} // namespace

extern "C" void nexusrt_launch_detokenize_format(
    void* stream, void* tokens, void* out_bytes,
    uint64_t n_tokens, uint32_t grid, uint32_t block)
{
    using namespace nexusrt::kernels;
    detokenize_format<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const int32_t*)tokens, (uint8_t*)out_bytes, n_tokens);
}
