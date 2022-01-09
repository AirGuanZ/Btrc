#include <btrc/core/spectrum/spectrum.h>

BTRC_CORE_BEGIN

CSpectrum CSpectrum::zeros()
{
    return CSpectrum(0);
}

CSpectrum CSpectrum::ones()
{
    return CSpectrum(1);
}

CSpectrum::CSpectrum(f32 v)
{
    r = v;
    g = v;
    b = v;
}

BTRC_CORE_END
