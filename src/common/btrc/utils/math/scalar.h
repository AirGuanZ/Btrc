#pragma once

#include <cmath>

#include <btrc/common.h>

BTRC_BEGIN

constexpr float btrc_max_float = (std::numeric_limits<float>::max)();
constexpr float btrc_pi  = 3.1415926535f;

BTRC_XPU inline float btrc_sqrt(float x)
{
#if BTRC_IS_CUDA_DEVICE_CODE
    return sqrtf(x);
#else
    return std::sqrt(x);
#endif
}

BTRC_XPU inline float btrc_pow(float x, float y)
{
#if BTRC_IS_CUDA_DEVICE_CODE
    return powf(x, y);
#else
    return std::pow(x, y);
#endif
}

BTRC_XPU inline float btrc_sin(float x)
{
#if BTRC_IS_CUDA_DEVICE_CODE
    return sinf(x);
#else
    return std::sin(x);
#endif
}

BTRC_XPU inline float btrc_cos(float x)
{
#if BTRC_IS_CUDA_DEVICE_CODE
    return cosf(x);
#else
    return std::cos(x);
#endif
}

BTRC_XPU inline float btrc_acos(float x)
{
#if BTRC_IS_CUDA_DEVICE_CODE
    return acosf(x);
#else
    return std::acos(x);
#endif
}

BTRC_END
