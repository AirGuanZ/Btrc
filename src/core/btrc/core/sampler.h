#pragma once

#include <btrc/utils/cmath/cmath.h>

BTRC_BEGIN

class Sampler
{
public:

    virtual ~Sampler() = default;

    virtual f32 get1d() = 0;

    virtual Sam2 get2d()
    {
        f32 x = get1d();
        f32 y = get1d();
        return make_sample(x, y);
    }

    virtual Sam3 get3d()
    {
        f32 x = get1d();
        f32 y = get1d();
        f32 z = get1d();
        return make_sample(x, y, z);
    }

    virtual Sam4 get4d()
    {
        f32 x = get1d();
        f32 y = get1d();
        f32 z = get1d();
        f32 w = get1d();
        return make_sample(x, y, z, w);
    }

    virtual Sam5 get5d()
    {
        f32 x = get1d();
        f32 y = get1d();
        f32 z = get1d();
        f32 w = get1d();
        f32 u = get1d();
        return make_sample(x, y, z, w, u);
    }
};

BTRC_END
