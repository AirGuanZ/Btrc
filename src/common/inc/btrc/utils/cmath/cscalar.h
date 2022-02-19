#pragma once

#include <cuj.h>

#include <btrc/common.h>

BTRC_BEGIN

using cuj::ptr;
using cuj::ref;
using cuj::var;

using cuj::boolean;
using cuj::f32;
using cuj::i32;
using cuj::u32;
using cuj::u64;

namespace cstd = cuj::cstd;

inline f32 lerp(f32 a, f32 b, f32 t)
{
    return a * (1.0f - t) + b * t;
}

BTRC_END
