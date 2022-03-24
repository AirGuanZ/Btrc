#pragma once

#include <btrc/utils/cmath/cmath.h>

BTRC_BEGIN

class Sampler
{
public:

    virtual ~Sampler() = default;

    virtual f32 get1d() = 0;

    virtual CVec2f get2d() = 0;

    virtual CVec3f get3d() = 0;
};

BTRC_END
