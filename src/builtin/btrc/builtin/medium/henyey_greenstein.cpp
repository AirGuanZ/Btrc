#include <btrc/builtin/medium/henyey_greenstein.h>
#include <btrc/utils/local_angle.h>

BTRC_BUILTIN_BEGIN

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

    f32 henyey_greenstein(f32 g, f32 u)
    {
        var g2 = g * g;
        var dem = 1.0f + g2 - mul2(g * u);
        return (1.0f - g2) / (4.0f * btrc_pi * dem * cstd::sqrt(dem));
    }

} // namespace anonymous

void HenyeyGreensteinPhaseShader::set_g(f32 g)
{
    g_ = g;
}

PhaseShader::SampleResult HenyeyGreensteinPhaseShader::sample(CompileContext &cc, ref<CVec3f> wo, ref<CVec3f> sam) const
{
    static auto sample_func = cuj::function_contextless([](f32 g, ref<CVec3f> wo, ref<CVec3f> sam)
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
        result.phase = CSpectrum::from_rgb(phase_val, phase_val, phase_val);
        result.dir = CFrame::from_z(wo).local_to_global(local_wi);
        result.pdf = phase_val;

        return result;
    });
    return sample_func(g_, wo, sam);
}

CSpectrum HenyeyGreensteinPhaseShader::eval(CompileContext &cc, ref<CVec3f> wi, ref<CVec3f> wo) const
{
    static auto eval_func = cuj::function_contextless(
        [](f32 g, ref<CVec3f> wi, ref<CVec3f> wo)
    {
        var u = -cos(wi, wo);
        var f = henyey_greenstein(g, u);
        return CSpectrum::from_rgb(f, f, f);
    });
    return eval_func(g_, wi, wo);
}

f32 HenyeyGreensteinPhaseShader::pdf(CompileContext &cc, ref<CVec3f> wi, ref<CVec3f> wo) const
{
    static auto pdf_func = cuj::function_contextless(
        [](f32 g, ref<CVec3f> wi, ref<CVec3f> wo)
    {
        return henyey_greenstein(g, -cos(wi, wo));
    });
    return pdf_func(g_, wi, wo);
}

BTRC_BUILTIN_END
