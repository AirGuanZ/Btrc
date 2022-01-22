#include <btrc/core/light/gradient_sky.h>

BTRC_CORE_BEGIN

GradientSky::GradientSky()
{
    up_ = { 0, 0, 1 };
}

void GradientSky::set_lower(const Spectrum &lower)
{
    lower_ = lower;
}

void GradientSky::set_upper(const Spectrum &upper)
{
    upper_ = upper;
}

void GradientSky::set_up(const Vec3f &up)
{
    up_ = up;
}

CSpectrum GradientSky::eval_le(const CVec3f &to_light) const
{
    var cos_theta = dot(CVec3f(up_), normalize(to_light));
    var s = cstd::saturate(0.5f * (cos_theta + 1.0f));
    return lower_.to_cspectrum() * (1.0f - s) + upper_.to_cspectrum() * s;
}

EnvirLight::SampleLiResult GradientSky::sample_li(const CVec3f &sam) const
{
    CFrame frame = CFrame::from_z(up_);
    var local_dir = sample_sphere_uniform(sam.x, sam.y);
    var global_dir = frame.local_to_global(local_dir);

    SampleLiResult result;
    result.direction_to_light = global_dir;
    result.radiance = eval_le(global_dir);
    result.pdf = pdf_sample_sphere_uniform();

    return result;
}

f32 GradientSky::pdf_li(const CVec3f &to_light) const
{
    return pdf_sample_sphere_uniform();
}

BTRC_CORE_END
