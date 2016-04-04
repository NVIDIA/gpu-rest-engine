#include "kernel.h"

#include <cuda_runtime.h>
#include <stdexcept>

__global__ void empty_kernel()
{
}

void kernel_wrapper(cudaStream_t stream)
{
    empty_kernel<<<1, 1, 0, stream>>>();
    cudaError_t st = cudaStreamSynchronize(stream);
    if (st != cudaSuccess)
        throw std::invalid_argument("could not launch CUDA kernel");
}
