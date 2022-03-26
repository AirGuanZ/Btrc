#include <btrc/builtin/material/component/diffuse.h>

BTRC_BUILTIN_BEGIN

Shader::SampleResult DiffuseComponentImpl::sample(ref<CVec3f> lwo, ref<CVec3f> sam, TransportMode mode) const
{
    Shader::SampleResult result;
    $if(lwo.z <= 0)
    {
        result.clear();
    }
    $else
    {
        var lwi = sample_hemisphere_zweighted(sam.x, sam.y);
        result.bsdf = albedo_value / btrc_pi;
        result.dir = lwi;
        result.pdf = pdf_sample_hemisphere_zweighted(lwi);
        result.is_delta = false;
    };
    return result;
}

CSpectrum DiffuseComponentImpl::eval(ref<CVec3f> lwi, ref<CVec3f> lwo, TransportMode mode) const
{
    CSpectrum result;
    $if(lwi.z <= 0 | lwo.z <= 0)
    {
        result = CSpectrum::zero();
    }
    $else
    {
        result = albedo_value / btrc_pi;
    };
    return result;
}

f32 DiffuseComponentImpl::pdf(ref<CVec3f> lwi, ref<CVec3f> lwo, TransportMode mode) const
{
    f32 result;
    $if(lwi.z <= 0 | lwo.z <= 0)
    {
        result = 0;
    }
    $else
    {
        result = pdf_sample_hemisphere_zweighted(lwi);
    };
    return result;
}

CSpectrum DiffuseComponentImpl::albedo() const
{
    return albedo_value;
}

BTRC_BUILTIN_END
