#pragma once

#include <memory>
#include <stdexcept>

#ifdef __CUDACC__
    #define BTRC_IS_CUDACC 1
#else
    #define BTRC_IS_CUDACC 0
#endif

#if defined(DEBUG) || defined(_DEBUG)
    #define BTRC_IS_DEBUG 1
#else
    #define BTRC_IS_DEBUG 0
#endif

#ifdef __CUDA_ARCH__
    #define BTRC_IS_CUDA_DEVICE_CODE 1
#else
    #define BTRC_IS_CUDA_DEVICE_CODE 0
#endif

#define BTRC_CORE_BEGIN namespace btrc::core {
#define BTRC_CORE_END   }

#define BTRC_WAVEFRONT_BEGIN BTRC_CORE_BEGIN namespace wf {
#define BTRC_WAVEFRONT_END   } BTRC_CORE_END

#define BTRC_OPTIX_BEGIN BTRC_CORE_BEGIN namespace optix {
#define BTRC_OPTIX_END   } BTRC_CORE_END

#define BTRC_CUDA_BEGIN BTRC_CORE_BEGIN namespace cuda {
#define BTRC_CUDA_END   } BTRC_CORE_END

#if BTRC_IS_CUDACC
    #define BTRC_CPU         __host__
    #define BTRC_GPU         __device__
    #define BTRC_XPU         __host__ __device__
    #define BTRC_KERNEL      __global__
    #define BTRC_FORCEINLINE __forceinline__
#else
    #define BTRC_CPU
    #define BTRC_GPU
    #define BTRC_XPU
    #define BTRC_KERNEL
    #ifdef _MSC_VER
        #define BTRC_FORCEINLINE __forceinline
    #else
        #define BTRC_FORCEINLINE __attribute__((always_inline))
    #endif
#endif

BTRC_CORE_BEGIN

class BtrcException : public std::runtime_error
{
public:

    using runtime_error::runtime_error;
};

template<typename T, typename I>
BTRC_XPU constexpr T up_align(T val, I align)
{
    return (val + align - 1) / align * align;
}

template<typename T, typename I>
BTRC_XPU constexpr T down_align(T val, I align)
{
    return val / align * align;
}

template<typename T>
using RC = std::shared_ptr<T>;

template<typename T>
using Box = std::unique_ptr<T>;

template<typename T, typename...Args>
BTRC_CPU auto newRC(Args &&...args)
{
    return std::make_shared<T>(std::forward<Args>(args)...);
}

template<typename T, typename...Args>
BTRC_CPU auto newBox(Args &&...args)
{
    return std::make_unique<T>(std::forward<Args>(args)...);
}

BTRC_CORE_END
