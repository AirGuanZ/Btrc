#include <btrc/core/medium.h>
#include <btrc/utils/henyey_greenstein.h>
#include <btrc/utils/local_angle.h>

BTRC_BEGIN

namespace
{

    f32 sqr(f32 x)
    {
        return x * x;
    }

    f32 mul2(f32 x)
    {
        return x + x;
    }

} // namespace anonymous

void HenyeyGreensteinPhaseShader::set_g(f32 g)
{
    g_ = g;
}

void HenyeyGreensteinPhaseShader::set_color(ref<CSpectrum> color)
{
    color_ = color;
}

PhaseShader::SampleResult HenyeyGreensteinPhaseShader::sample(CompileContext &cc, ref<CVec3f> wo, ref<CVec3f> sam) const
{
    static auto sample_func = cuj::function_contextless([](
        f32 g, ref<CSpectrum> color, ref<CVec3f> wo, ref<CVec3f> sam)
        {
            var s = sam.x + sam.x - 1;
            f32 u;
            $if(cstd::abs(g) < 0.001f)
            {
                u = s;
            }
            $else
            {
                var g2 = g * g;
                u = (1.0f + g2 - sqr((1.0f - g2) / (1.0f + g * s))) / mul2(g);
            };

            var cos_theta = -u;
            var sin_theta = local_angle::cos2sin(cos_theta);
            var phi = 2 * btrc_pi * sam.y;

            var local_wi = CVec3f(sin_theta * cstd::sin(phi), sin_theta * cstd::cos(phi), cos_theta);
            var phase_val = henyey_greenstein(g, u);

            SampleResult result;
            result.phase = color * CSpectrum::from_rgb(phase_val, phase_val, phase_val);
            result.dir = CFrame::from_z(wo).local_to_global(local_wi);
            result.pdf = phase_val;

            return result;
        });
    return sample_func(g_, color_, wo, sam);
}

CSpectrum HenyeyGreensteinPhaseShader::eval(CompileContext &cc, ref<CVec3f> wi, ref<CVec3f> wo) const
{
    var u = -cos(wi, wo);
    var f = henyey_greenstein(g_, u);
    return color_ * CSpectrum::from_rgb(f, f, f);
}

f32 HenyeyGreensteinPhaseShader::pdf(CompileContext &cc, ref<CVec3f> wi, ref<CVec3f> wo) const
{
    return henyey_greenstein(g_, -cos(wi, wo));
}

Medium::SampleResult Medium::sample(
    CompileContext &cc, ref<CVec3f> a, ref<CVec3f> b, Sampler &sampler) const
{
    return sample(cc, a, b, a, b, sampler);
}

CSpectrum Medium::tr(
    CompileContext &cc, ref<CVec3f> a, ref<CVec3f> b, Sampler &sampler) const
{
    return tr(cc, a, b, a, b, sampler);
}

void TransformMedium::set_tranformed(RC<Medium> transformed)
{
    transformed_ = std::move(transformed);
}

void TransformMedium::set_transform(const Transform3D &world_to_local)
{
    world_to_local_ = world_to_local;
}

Medium::SampleResult TransformMedium::sample(
    CompileContext &cc,
    ref<CVec3f>     a,
    ref<CVec3f>     b,
    ref<CVec3f>     uvw_a,
    ref<CVec3f>     uvw_b,
    Sampler        &sampler) const
{
    var world_to_local = CTransform3D(world_to_local_);
    var local_uvw_a = world_to_local.apply_to_point(uvw_a);
    var local_uvw_b = world_to_local.apply_to_point(uvw_b);
    return transformed_->sample(cc, a, b, local_uvw_a, local_uvw_b, sampler);
}

CSpectrum TransformMedium::tr(
    CompileContext &cc,
    ref<CVec3f>     a,
    ref<CVec3f>     b,
    ref<CVec3f>     uvw_a,
    ref<CVec3f>     uvw_b,
    Sampler        &sampler) const
{
    var world_to_local = CTransform3D(world_to_local_);
    var local_uvw_a = world_to_local.apply_to_point(uvw_a);
    var local_uvw_b = world_to_local.apply_to_point(uvw_b);
    return transformed_->tr(cc, a, b, local_uvw_a, local_uvw_b, sampler);
}

float TransformMedium::get_priority() const
{
    return transformed_->get_priority();
}

BTRC_END
