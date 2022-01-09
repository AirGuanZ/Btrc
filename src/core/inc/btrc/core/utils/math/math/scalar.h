#pragma once

#include <cmath>

#include <btrc/core/common.h>

BTRC_CORE_BEGIN

constexpr float btrc_inf = INFINITY;
constexpr float btrc_pi  = 3.1415926535f;

BTRC_XPU inline float btrc_sqrt(float x)
{
#if BTRC_IS_CUDA_DEVICE_CODE
    return sqrtf(x);
#else
    return std::sqrt(x);
#endif
}

BTRC_CORE_END
