#include <btrc/core/material/material.h>

BTRC_CORE_BEGIN

boolean BSDFWithBlackFringesHandling::is_black_fringes(ref<CVec3f> v) const
{
    var geometry_upper = geometry_frame_.in_positive_z_hemisphere(v);
    var shading_upper = shading_frame_.in_positive_z_hemisphere(v);
    return geometry_upper != shading_upper;
}

boolean BSDFWithBlackFringesHandling::is_black_fringes(
    ref<CVec3f> w1, ref<CVec3f> w2) const
{
    return is_black_fringes(w1) | is_black_fringes(w2);
}

BSDF::SampleResult BSDFWithBlackFringesHandling::sample_black_fringes(
    const CVec3f &wo, const CVec3f &sam) const
{
    var wo_upper = geometry_frame_.in_positive_z_hemisphere(wo);
    var local_wi = sample_hemisphere_zweighted(sam.x, sam.y);
    $if(!wo_upper)
    {
        local_wi.z = -local_wi.z;
    };
    SampleResult result;
    result.bsdf = albedo() / btrc_pi;
    result.dir = geometry_frame_.local_to_global(local_wi);
    result.pdf = pdf_sample_hemisphere_zweighted(local_wi);
    return result;
}

CSpectrum BSDFWithBlackFringesHandling::eval_black_fringes(
    const CVec3f &wi, const CVec3f &wo) const
{
    var wo_upper = geometry_frame_.in_positive_z_hemisphere(wo);
    var wi_upper = geometry_frame_.in_positive_z_hemisphere(wi);
    CSpectrum result = albedo().get_type()->create_czero();
    $if(wo_upper == wi_upper)
    {
        result = albedo() / btrc_pi;
    };
    return result;
}

f32 BSDFWithBlackFringesHandling::pdf_black_fringes(
    const CVec3f &wi, const CVec3f &wo) const
{
    var wo_upper = geometry_frame_.in_positive_z_hemisphere(wo);
    var wi_upper = geometry_frame_.in_positive_z_hemisphere(wi);
    f32 result;
    $if(wo_upper != wi_upper)
    {
        result = 0;
    }
    $else
    {
        var local_wi = normalize(geometry_frame_.global_to_local(wi));
        result = pdf_sample_hemisphere_zweighted(local_wi);
    };
    return result;
}

BSDFWithBlackFringesHandling::BSDFWithBlackFringesHandling(
    CFrame geometry_frame, CFrame shading_frame)
    : geometry_frame_(geometry_frame), shading_frame_(shading_frame)
{
    
}

BSDF::SampleResult BSDFWithBlackFringesHandling::sample(
    const CVec3f &wo, const CVec3f &sam, TransportMode mode) const
{
    SampleResult result;
    $if(is_black_fringes(wo))
    {
        result = sample_black_fringes(wo, sam);
    }
    $else
    {
        result = sample_impl(wo, sam, mode);
        $if(result.pdf >= 0)
        {
            var down = cstd::abs(dot(geometry_frame_.z, result.dir));
            $if(down > 1e-4f)
            {
                var up = cstd::abs(dot(shading_frame_.z, result.dir));
                result.bsdf = (up / down) * result.bsdf;
            };
        };
    };
    return result;
}

CSpectrum BSDFWithBlackFringesHandling::eval(
    const CVec3f &wi, const CVec3f &wo, TransportMode mode) const
{
    CSpectrum result;
    $if(is_black_fringes(wi, wo))
    {
        result = eval_black_fringes(wi, wo);
    }
    $else
    {
        result = eval_impl(wi, wo, mode);
        var down = cstd::abs(dot(geometry_frame_.z, wi));
        $if(down > 1e-4f)
        {
            var up = cstd::abs(dot(shading_frame_.z, wi));
            result = (up / down) * result;
        };
    };
    return result;
}

f32 BSDFWithBlackFringesHandling::pdf(
    const CVec3f &wi, const CVec3f &wo, TransportMode mode) const
{
    f32 pdf;
    $if(is_black_fringes(wi, wo))
    {
        pdf = pdf_black_fringes(wi, wo);
    }
    $else
    {
        pdf = pdf_impl(wi, wo, mode);
    };
    return pdf;
}

BTRC_CORE_END
