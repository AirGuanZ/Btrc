#pragma once

#include <btrc/utils/math/vec3.h>

BTRC_BEGIN

struct Ray
{
    Vec3f o;
    Vec3f d;
    float t;

    Ray()
        : Ray({}, { 1, 0, 0 })
    {
        
    }

    Ray(const Vec3f &o, const Vec3f &d, float t = btrc_max_float)
        : o(o), d(d), t(t)
    {
        
    }
};

BTRC_END
