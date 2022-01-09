#pragma once

#include <btrc/core/utils/math/cmath.h>

BTRC_CORE_BEGIN

class Spectrum
{
public:

    float r, g, b;
};

class CSpectrum
{
public:

    static CSpectrum zeros();

    static CSpectrum ones();

    explicit CSpectrum(f32 v = 0);

    f32 r, g, b;
};

BTRC_CORE_END
