// =============================================================================
// NexusRT — src/pipeline/kernels/tokenize_kernel.cu
// GPU BPE tokenizer kernel. Reads raw bytes from HBM, writes int32 token IDs
// to a separate HBM buffer. Designed to be launched by the scheduler with
// warp-specialized producer (GDS) + consumer (tokenize) warps.
// =============================================================================
#include "firmware/cuda_kernels.cuh"

#include <cstdint>

namespace nexusrt { namespace kernels {

// Minimal byte-pair tokenizer: emits one token per byte for the smoke-test
// build. Production replaces this with a full BPE merge table lookup.
__global__ void tokenize_bpe_u8(
    const uint8_t* __restrict__ raw,
    int32_t*       __restrict__ tokens,
    uint64_t                    n_bytes,
    uint32_t                    vocab_size)
{
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bytes) return;
    uint8_t b = raw[i];
    // Map byte → token id by simple modulo. Real BPE would use a merges
    // table indexed in shared memory.
    tokens[i] = static_cast<int32_t>(b) % vocab_size;
}

}} // namespace

// Host-side launcher consumed by scheduler::TaskGraph.
extern "C" void nexusrt_launch_tokenize_bpe_u8(
    void* stream, void* raw, void* tokens,
    uint64_t n_bytes, uint32_t vocab_size,
    uint32_t grid, uint32_t block)
{
    using namespace nexusrt::kernels;
    tokenize_bpe_u8<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)raw, (int32_t*)tokens, n_bytes, vocab_size);
}
