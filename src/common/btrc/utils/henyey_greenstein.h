#pragma once

#include <btrc/utils/cmath/cmath.h>

BTRC_BEGIN

inline f32 henyey_greenstein(f32 g, f32 u)
{
    static auto func = cuj::function_contextless([](f32 g, f32 u)
    {
        var g2 = g * g;
        var dem = 1.0f + g2 - 2.0f * (g * u);
        return (1.0f - g2) / (4.0f * btrc_pi * dem * cstd::sqrt(dem));
    });
    return func(g, u);
}

BTRC_END
