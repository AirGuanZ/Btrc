#include <btrc/core/texture3d.h>

BTRC_BEGIN

Constant3D::Constant3D()
{
    set_value(0.0f);
}

void Constant3D::set_value(float value)
{
    set_value(Spectrum::from_rgb(value, value, value));
}

void Constant3D::set_value(const Spectrum &value)
{
    value_ = value;
}

f32 Constant3D::sample_float_inline(CompileContext &cc, ref<CVec3f> uvw) const
{
    return value_.read(cc).r;
}

CSpectrum Constant3D::sample_spectrum_inline(CompileContext &cc, ref<CVec3f> uvw) const
{
    return value_.read(cc);
}

CSpectrum Constant3D::get_max_spectrum(CompileContext &cc) const
{
    return value_.read(cc);
}

BTRC_END
