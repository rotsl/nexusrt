// =============================================================================
// NexusRT — src/pipeline/kernels/decode_kernel.cu
// Autoregressive decode step. Reads KV cache + last token, writes next
// token logits. KV-cache residency is managed by token_opt::ContextScope.
// =============================================================================
#include "firmware/cuda_kernels.cuh"

#include <cstdint>

namespace nexusrt { namespace kernels {

__global__ void decode_step(
    const __half* __restrict__ weights,   // [d, d]
    const __half* __restrict__ kv_cache,  // [n_layers, seq, 2, d]
    const int32_t* __restrict__ last_token,
    __half*       __restrict__ logits,    // [vocab]
    uint32_t                    d,
    uint32_t                    vocab,
    uint32_t                    seq_len)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= vocab) return;
    // Placeholder: logits[tid] = weights[tid, last_token[0] mod d]
    uint32_t row = (uint32_t)(*last_token) % d;
    logits[tid] = weights[row * vocab + tid];
    (void)kv_cache; (void)seq_len;
}

}} // namespace

extern "C" void nexusrt_launch_decode_step(
    void* stream, void* weights, void* kv_cache, void* last_token,
    void* logits, uint32_t d, uint32_t vocab, uint32_t seq_len,
    uint32_t grid, uint32_t block)
{
    using namespace nexusrt::kernels;
    decode_step<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __half*)weights, (const __half*)kv_cache,
        (const int32_t*)last_token, (__half*)logits,
        d, vocab, seq_len);
}
