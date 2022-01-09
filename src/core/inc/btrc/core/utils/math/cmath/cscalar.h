#pragma once

#include <cuj.h>

#include <btrc/core/common.h>

BTRC_CORE_BEGIN

using cuj::ref;

using cuj::f32;
using cuj::u32;

namespace cstd = cuj::cstd;

inline f32 lerp(f32 a, f32 b, f32 t)
{
    return a * (1.0f - t) + b * t;
}

BTRC_CORE_END
