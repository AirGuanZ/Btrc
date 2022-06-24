#include <btrc/builtin/light/gradient_sky.h>

BTRC_BUILTIN_BEGIN

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
    up_ = normalize(up);
}

CSpectrum GradientSky::eval_le_inline(CompileContext &cc, ref<CVec3f> to_light) const
{
    var cos_theta = dot(up_, normalize(to_light));
    var s = cstd::saturate(0.5f * (cos_theta + 1.0f));
    return CSpectrum(lower_) * (1.0f - s) + CSpectrum(upper_) * s;
}

EnvirLight::SampleLiResult GradientSky::sample_li_inline(CompileContext &cc, ref<Sam3> sam) const
{
    CFrame frame = CFrame::from_z(up_);
    var local_dir = sample_sphere_uniform(sam[0], sam[1]);
    var global_dir = frame.local_to_global(local_dir);

    SampleLiResult result;
    result.direction_to_light = global_dir;
    result.radiance = eval_le(cc, global_dir);
    result.pdf = pdf_sample_sphere_uniform();

    return result;
}

f32 GradientSky::pdf_li_inline(CompileContext &cc, ref<CVec3f> to_light) const
{
    return pdf_sample_sphere_uniform();
}

EnvirLight::SampleEmitResult GradientSky::sample_emit_inline(CompileContext &cc, ref<Sam3> sam) const
{
    // TODO
    return {};
}

f32 GradientSky::pdf_emit_inline(CompileContext &cc, ref<CVec3f> dir) const
{
    // TODO
    return {};
}

RC<Light> GradientSkyCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    const Spectrum upper = node->parse_child<Spectrum>("upper");
    const Spectrum lower = node->parse_child<Spectrum>("lower");
    const Vec3f up = node->parse_child<Vec3f>("up");
    auto sky = newRC<GradientSky>();
    sky->set_upper(upper);
    sky->set_lower(lower);
    sky->set_up(up);
    return sky;
}

BTRC_BUILTIN_END
