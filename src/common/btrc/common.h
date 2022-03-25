#pragma once

#include <memory>
#include <stdexcept>

#include <cuj.h>

#if defined(DEBUG) || defined(_DEBUG)
#define BTRC_IS_DEBUG 1
#else
#define BTRC_IS_DEBUG 0
#endif

#define BTRC_BEGIN namespace btrc {
#define BTRC_END   }

#define BTRC_FACTORY_BEGIN BTRC_BEGIN namespace factory {
#define BTRC_FACTORY_END   } BTRC_END

#define BTRC_BUILTIN_BEGIN BTRC_BEGIN namespace builtin {
#define BTRC_BUILTIN_END   } BTRC_END

#define BTRC_CUDA_BEGIN BTRC_BEGIN namespace cuda {
#define BTRC_CUDA_END   } BTRC_END

#define BTRC_OPTIX_BEGIN BTRC_BEGIN namespace optix {
#define BTRC_OPTIX_END   } BTRC_END

BTRC_BEGIN

class BtrcException : public std::runtime_error
{
public:

    using runtime_error::runtime_error;
};

template<typename T, typename I>
constexpr T up_align(T val, I align)
{
    return (val + align - 1) / align * align;
}

template<typename T, typename I>
constexpr T down_align(T val, I align)
{
    return val / align * align;
}

template<typename T>
using RC = std::shared_ptr<T>;

template<typename T>
using Box = std::unique_ptr<T>;

template<typename T, typename...Args>
auto newRC(Args &&...args)
{
    return std::make_shared<T>(std::forward<Args>(args)...);
}

template<typename T, typename...Args>
auto newBox(Args &&...args)
{
    return std::make_unique<T>(std::forward<Args>(args)...);
}

BTRC_END
