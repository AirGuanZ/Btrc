#pragma once

#include <cuda_runtime.h>
#include <optix_stubs.h>

#include <btrc/core/common.h>

BTRC_CORE_BEGIN

inline void throw_on_error(cudaError err)
{
    if(err != cudaSuccess)
        throw BtrcException(cudaGetErrorString(err));
}

inline void throw_on_error(CUresult result)
{
    if(result != CUDA_SUCCESS)
    {
        const char *err_str;
        cuGetErrorString(result, &err_str);
        throw BtrcException(err_str);
    }
}

inline void throw_on_error(OptixResult result)
{
    if(result != OPTIX_SUCCESS)
        throw BtrcException(optixGetErrorString(result));
}

BTRC_CORE_END
