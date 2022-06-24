#pragma once

#include <cuj.h>

#include <btrc/common.h>

BTRC_BEGIN

using cuj::ptr;
using cuj::ref;
using cuj::var;

using cuj::boolean;
using cuj::i64;
using cuj::f32;
using cuj::i32;
using cuj::u32;
using cuj::u64;
using cuj::u8;

template<size_t N>
using Sam = cuj::arr<f32, N>;

using Sam1 = Sam<1>;
using Sam2 = Sam<2>;
using Sam3 = Sam<3>;
using Sam4 = Sam<4>;
using Sam5 = Sam<5>;

template<typename...Args>
auto make_sample(Args...args)
{
    Sam<sizeof...(args)> ret;
    int index = 0;
    ((ret[index++] = args), ...);
    return ret;
}

namespace cstd = cuj::cstd;

inline f32 lerp(f32 a, f32 b, f32 t)
{
    return a * (1.0f - t) + b * t;
}

BTRC_END
