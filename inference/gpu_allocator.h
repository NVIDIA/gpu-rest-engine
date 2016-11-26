#ifndef GPU_ALLOCATOR_H
#define GPU_ALLOCATOR_H

#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>

using GpuMat = cv::cuda::GpuMat;
using namespace cv;

/* A simple linear allocator class to allocate storage for cv::GpuMat objects.
   This feature was added in OpenCV 3.0. */
class GPUAllocator : public GpuMat::Allocator
{
public:
    GPUAllocator(size_t size);

    ~GPUAllocator();

    void reset();

public: /* GpuMat::Allocator interface */
    bool allocate(GpuMat* mat, int rows, int cols, size_t elemSize);

    void free(GpuMat* mat);

private:
    cudaError_t grow(void** dev_ptr, size_t size);

private:
    void* base_ptr_;
    void* current_ptr_;
    size_t total_size_;
    size_t current_size_;
};

#endif // GPU_ALLOCATOR_H
