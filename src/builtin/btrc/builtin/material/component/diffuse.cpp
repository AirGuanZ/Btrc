#include <btrc/builtin/material/component/diffuse.h>

BTRC_BUILTIN_BEGIN

BSDFComponent::SampleResult DiffuseComponentImpl::sample(ref<CVec3f> lwo, ref<Sam3> sam, TransportMode mode) const
{
    return BSDFComponent::discard_pdf_rev(sample_bidir(lwo, sam, mode));
}

BSDFComponent::SampleBidirResult DiffuseComponentImpl::sample_bidir(
    ref<CVec3f> lwo, ref<Sam3> sam, TransportMode mode) const
{
    BSDFComponent::SampleBidirResult result;
    $if(lwo.z <= 0)
    {
        result.clear();
    }
    $else
    {
        var lwi = sample_hemisphere_zweighted(sam[0], sam[1]);
        result.bsdf     = albedo_value / btrc_pi;
        result.dir      = lwi;
        result.pdf      = pdf_sample_hemisphere_zweighted(lwi);
        result.pdf_rev  = pdf_sample_hemisphere_zweighted(lwo);
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
