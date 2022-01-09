#pragma once

#include <btrc/core/utils/math/math/vec3.h>

BTRC_CORE_BEGIN

class Ray
{
public:

    using Mask = uint32_t;

    Vec3f o;
    Vec3f d;
    float t0;
    float t1;
    float time;
    Mask  mask;

    Ray();

    Ray(
        const Vec3f &o,
        const Vec3f &d,
        float        t0   = 0,
        float        t1   = btrc_inf,
        float        time = 0,
        Mask         mask = 0xffffffff);
};

// ========================== impl ==========================

inline Ray::Ray()
    : Ray({}, {})
{
    
}

inline Ray::Ray(
    const Vec3f &o, const Vec3f &d, float t0, float t1, float time, Mask mask)
    : o(o), d(d), t0(t0), t1(t1), time(time), mask(mask)
{
    
}

BTRC_CORE_END
