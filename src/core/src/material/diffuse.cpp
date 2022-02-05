#include <btrc/core/material/diffuse.h>
#include <btrc/core/material/shader_closure.h>
#include <btrc/core/material/shader_frame.h>

BTRC_CORE_BEGIN

CUJ_CLASS_BEGIN(DiffuseShaderImpl)

    CUJ_MEMBER_VARIABLE(ShaderFrame, frame)
    CUJ_MEMBER_VARIABLE(CSpectrum,   albedo_val)

    Shader::SampleResult sample(ref<CVec3f> wo, ref<CVec3f> sam, TransportMode mode) const
    {
        $declare_scope;
        Shader::SampleResult result;

        $if(frame.is_black_fringes(wo))
        {
            result = frame.sample_black_fringes(wo, sam, albedo_val);
            $exit_scope;
        };

        $if(dot(wo, frame.shading.z) <= 0)
        {
            result.bsdf = CSpectrum::zero();
            result.dir = CVec3f();
            result.pdf = 0;
            $exit_scope;
        };

        var local_wi = sample_hemisphere_zweighted(sam.x, sam.y);
        result.bsdf = albedo_val / btrc_pi;
        result.dir = frame.shading.local_to_global(local_wi);
        result.pdf = pdf_sample_hemisphere_zweighted(local_wi);
        result.bsdf = frame.correct_shading_energy(result.dir) * result.bsdf;
        return result;
    }

    CSpectrum eval(ref<CVec3f> wi, ref<CVec3f> wo, TransportMode mode) const
    {
        $declare_scope;
        CSpectrum result;

        $if(frame.is_black_fringes(wi, wo))
        {
            result = frame.eval_black_fringes(wi, wo, albedo_val);
            $exit_scope;
        };
        
        $if(dot(wi, frame.shading.z) <= 0 | dot(wo, frame.shading.z) <= 0)
        {
            result = CSpectrum::zero();
            $exit_scope;
        };

        result = albedo_val / btrc_pi;
        result = frame.correct_shading_energy(wi) * result;
        return result;
    }

    f32 pdf(ref<CVec3f> wi, ref<CVec3f> wo, TransportMode mode) const
    {
        $declare_scope;
        f32 result;

        $if(frame.is_black_fringes(wi, wo))
        {
            result = frame.pdf_black_fringes(wi, wo);
            $exit_scope;
        };

        $if(dot(wi, frame.shading.z) <= 0 | dot(wo, frame.shading.z) <= 0)
        {
            result = 0;
            $exit_scope;
        };

        var local_wi = normalize(frame.shading.global_to_local(wi));
        result = pdf_sample_hemisphere_zweighted(local_wi);
        return result;
    }

    CSpectrum albedo() const
    {
        return albedo_val;
    }

    CVec3f normal() const
    {
        return frame.shading.z;
    }

    boolean is_delta() const
    {
        return false;
    }

CUJ_CLASS_END

void Diffuse::set_albedo(const Spectrum &albedo)
{
    albedo_ = albedo;
}

RC<Shader> Diffuse::create_shader(const CIntersection &inct) const
{
    DiffuseShaderImpl impl;
    impl.frame.geometry = inct.frame;
    impl.frame.shading = inct.frame.rotate_to_new_z(inct.interp_normal);
    impl.albedo_val = CSpectrum(albedo_);
    return newRC<ShaderClosure<DiffuseShaderImpl>>(as_shared(), impl);
}

BTRC_CORE_END
