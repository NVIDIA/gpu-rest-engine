#include "gpu_allocator.h"

#include <stdexcept>
#include <cstring>
#include <cuda_runtime.h>

// Page offset on Kepler/Maxwell
#define ALIGNMENT (128*1024)

GPUAllocator::GPUAllocator(size_t size)
    : total_size_(size),
      current_size_(0)
{
    cudaError_t rc = cudaMalloc(&base_ptr_, total_size_);
    if (rc != cudaSuccess)
        throw std::runtime_error("Could not allocate GPU memory");

    current_ptr_ = base_ptr_;
}

GPUAllocator::~GPUAllocator()
{
    cudaFree(base_ptr_);
}

static int align_up(unsigned int v, unsigned int alignment)
{
    return ((v + alignment - 1) / alignment) * alignment;
}

cudaError_t GPUAllocator::slabAllocate(void** dev_ptr, size_t size)
{
    if (current_size_ + size >= total_size_)
	return cudaErrorMemoryAllocation;

    *dev_ptr = current_ptr_;
    size_t aligned_size = align_up(size, ALIGNMENT);
    current_ptr_ = (uint8_t*)current_ptr_ + aligned_size;
    current_size_ += aligned_size;

    return cudaSuccess;
}

void GPUAllocator::reset()
{
    current_ptr_ = base_ptr_;
    current_size_ = 0;
}

bool GPUAllocator::allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize)
{
    int padded_width  = align_up(cols, 16);
    int padded_height = align_up(rows, 16);
    int total_size = elemSize * padded_width * padded_height;

    cudaError_t status = slabAllocate((void**)&mat->data, total_size);
    if (status != cudaSuccess)
        return false;

    mat->step = padded_width * elemSize;
    mat->refcount = new int;

    return true;
}

void GPUAllocator::free(cv::cuda::GpuMat* mat)
{
    delete mat->refcount;
}
