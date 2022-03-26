#include <btrc/builtin/material/utils/shader_frame.h>

BTRC_BEGIN

boolean ShaderFrame::is_black_fringes(ref<CVec3f> w) const
{
    var geometry_upper = geometry.in_positive_z_hemisphere(w);
    var shading_upper = shading.in_positive_z_hemisphere(w);
    return geometry_upper != shading_upper;
}

boolean ShaderFrame::is_black_fringes(ref<CVec3f> w1, ref<CVec3f> w2) const
{
    return is_black_fringes(w1) | is_black_fringes(w2);
}

Shader::SampleResult ShaderFrame::sample_black_fringes(
    ref<CVec3f> wo, ref<CVec3f> sam, ref<CSpectrum> albedo) const
{
    var wo_upper = geometry.in_positive_z_hemisphere(wo);
    var local_wi = sample_hemisphere_zweighted(sam.x, sam.y);
    $if(!wo_upper)
    {
        local_wi.z = -local_wi.z;
    };
    Shader::SampleResult result;
    result.bsdf = albedo / btrc_pi;
    result.dir = geometry.local_to_global(local_wi);
    result.pdf = pdf_sample_hemisphere_zweighted(local_wi);
    result.is_delta = false;
    return result;
}

CSpectrum ShaderFrame::eval_black_fringes(
    ref<CVec3f> wi, ref<CVec3f> wo, ref<CSpectrum> albedo) const
{
    var wo_upper = geometry.in_positive_z_hemisphere(wo);
    var wi_upper = geometry.in_positive_z_hemisphere(wi);
    var result = CSpectrum::zero();
    $if(wo_upper == wi_upper)
    {
        result = albedo / btrc_pi;
    };
    return result;
}

f32 ShaderFrame::pdf_black_fringes(ref<CVec3f> wi, ref<CVec3f> wo) const
{
    var wo_upper = geometry.in_positive_z_hemisphere(wo);
    var wi_upper = geometry.in_positive_z_hemisphere(wi);
    f32 result;
    $if(wo_upper != wi_upper)
    {
        result = 0;
    }
    $else
    {
        var local_wi = normalize(geometry.global_to_local(wi));
        result = pdf_sample_hemisphere_zweighted(local_wi);
    };
    return result;
}

f32 ShaderFrame::correct_shading_energy(ref<CVec3f> wi) const
{
    f32 ret;
    var down = cstd::abs(dot(geometry.z, wi));
    $if(down > 1e-4f)
    {
        var up = cstd::abs(dot(shading.z, wi));
        ret = up / down;
    }
    $else
    {
        ret = 1;
    };
    return ret;
}

BTRC_END
