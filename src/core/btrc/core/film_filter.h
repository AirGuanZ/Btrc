#pragma once

#include <btrc/core/context.h>
#include <btrc/utils/cmath/cmath.h>

BTRC_BEGIN

class FilmFilter : public Object
{
public:

    virtual CVec2f sample(ref<CRNG> rng) const = 0;
};

BTRC_END
