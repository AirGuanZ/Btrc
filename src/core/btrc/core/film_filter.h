#pragma once

#include <btrc/core/context.h>
#include <btrc/core/sampler.h>
#include <btrc/utils/cmath/cmath.h>

BTRC_BEGIN

class FilmFilter : public Object
{
public:

    virtual CVec2f sample(Sampler &sampler) const = 0;
};

BTRC_END
