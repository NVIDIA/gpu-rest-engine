#include "benchmark.h"

#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "common.h"
#include "kernel.h"

class BenchmarkContext
{
public:
    friend ScopedContext<BenchmarkContext>;

    static bool IsCompatible(int device)
    {
        cudaError_t st = cudaSetDevice(device);
        if (st != cudaSuccess)
            return false;

        return true;
    }

    BenchmarkContext(int device)
        : device_(device)
    {
        cudaError_t st = cudaSetDevice(device_);
        if (st != cudaSuccess)
            throw std::invalid_argument("could not set CUDA device");

        st = cudaStreamCreate(&stream_);
        if (st != cudaSuccess)
            throw std::invalid_argument("could not create CUDA stream");
    }

    ~BenchmarkContext()
    {
        cudaStreamDestroy(stream_);
    }

    cudaStream_t CUDAStream()
    {
        return stream_;
    }

private:
    void Activate()
    {
        cudaError_t st = cudaSetDevice(device_);
        if (st != cudaSuccess)
            throw std::invalid_argument("could not set CUDA device");
    }

    void Deactivate()
    {
    }

private:
    int device_;
    cudaStream_t stream_;
};

struct benchmark_ctx
{
    ContextPool<BenchmarkContext> pool;
};

constexpr static int kContextsPerDevice = 4;

benchmark_ctx* benchmark_initialize()
{
    try
    {
        int device_count;
        cudaError_t st = cudaGetDeviceCount(&device_count);
        if (st != cudaSuccess)
            throw std::invalid_argument("could not list CUDA devices");

        ContextPool<BenchmarkContext> pool;
        for (int dev = 0; dev < device_count; ++dev)
        {
            if (!BenchmarkContext::IsCompatible(dev))
            {
                std::cerr << "Skipping device: " << dev << std::endl;
                continue;
            }

            for (int i = 0; i < kContextsPerDevice; ++i)
            {
                std::unique_ptr<BenchmarkContext> context(new BenchmarkContext(dev));
                pool.Push(std::move(context));
            }
        }

        if (pool.Size() == 0)
            throw std::invalid_argument("no suitable CUDA device");

        benchmark_ctx* ctx = new benchmark_ctx{std::move(pool)};
        errno = 0;
        return ctx;
    }
    catch (const std::invalid_argument& ex)
    {
        errno = EINVAL;
        return nullptr;
    }
}

void benchmark_execute(benchmark_ctx* ctx)
{
    try
    {
        ScopedContext<BenchmarkContext> context(ctx->pool);
        cudaStream_t stream = context->CUDAStream();
        kernel_wrapper(stream);
        errno = 0;
    }
    catch (const std::invalid_argument&)
    {
        errno = EINVAL;
    }
}

void benchmark_destroy(benchmark_ctx* ctx)
{
    delete ctx;
}
