#include <btrc/core/texture2d.h>

BTRC_BEGIN

Constant2D::Constant2D()
{
    set_value(0.0f);
}

void Constant2D::set_value(float value)
{
    value_ = Spectrum::from_rgb(value, value, value);
}

void Constant2D::set_value(const Spectrum &value)
{
    value_ = value;
}

CSpectrum Constant2D::sample_spectrum_inline(CompileContext &cc, ref<CVec2f> uv) const
{
    return value_.read(cc);
}

f32 Constant2D::sample_float_inline(CompileContext &cc, ref<CVec2f> uv) const
{
    return value_.read(cc).r;
}

BTRC_END
