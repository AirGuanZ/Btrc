#pragma once

#include <btrc/builtin/material/utils/aggregate.h>
#include <btrc/builtin/material/utils/fresnel.h>
#include <btrc/builtin/material/utils/microfacet.h>
#include <btrc/utils/local_angle.h>

BTRC_BUILTIN_BEGIN

CUJ_TEMPLATE_CLASS_BEGIN(MicrofacetReflectionComponentImpl, FresnelPoint)

    CUJ_MEMBER_VARIABLE(FresnelPoint, fresnel)
    CUJ_MEMBER_VARIABLE(f32, ax)
    CUJ_MEMBER_VARIABLE(f32, ay)

    MicrofacetReflectionComponentImpl(FresnelPoint _fresnel, f32 roughness, f32 anisotropic)
    {
        fresnel = _fresnel;
        f32 aspect = 1.0f;
        $if(anisotropic > 0)
        {
            aspect = cstd::sqrt(1.0f - 0.9f * anisotropic);
        };
        ax = (cstd::max)(f32(0.001f), roughness * roughness / aspect);
        ay = (cstd::max)(f32(0.001f), roughness * roughness * aspect);
    }

    BSDFComponent::SampleResult sample(ref<CVec3f> lwo, ref<Sam3> sam, TransportMode mode) const
    {
        return BSDFComponent::discard_pdf_rev(sample_bidir(lwo, sam, mode));
    }

    BSDFComponent::SampleBidirResult sample_bidir(ref<CVec3f> lwo, ref<Sam3> sam, TransportMode mode) const
    {
        BSDFComponent::SampleBidirResult result;
        result.clear();
        $if(lwo.z > 0)
        {
            var lwh = normalize(microfacet::sample_anisotropic_gtr2_vnor(lwo, ax, ay, make_sample(sam[1], sam[2])));
            $if(lwh.z > 0)
            {
                var lwi = normalize(2.0f * dot(lwo, lwh) * lwh - lwo);
                $if(lwi.z > 0)
                {
                    result.dir = lwi;
                    this->eval_and_pdf(lwi, lwo, result.bsdf, result.pdf, result.pdf_rev);
                };
            };
        };
        return result;
    }

    CSpectrum eval(ref<CVec3f> lwi, ref<CVec3f> lwo, TransportMode mode) const
    {
        CSpectrum result; f32 pdf, pdf_rev;
        eval_and_pdf(lwi, lwo, result, pdf, pdf_rev);
        return result;
    }

    f32 pdf(ref<CVec3f> lwi, ref<CVec3f> lwo, TransportMode mode) const
    {
        CSpectrum bsdf; f32 pdf, pdf_rev;
        eval_and_pdf(lwi, lwo, bsdf, pdf, pdf_rev);
        return pdf;
    }

    CSpectrum albedo() const
    {
        return fresnel.eval(1);
    }

    void eval_and_pdf(
        ref<CVec3f>    lwi,
        ref<CVec3f>    lwo,
        ref<CSpectrum> eval_output,
        ref<f32>       pdf_output,
        ref<f32>       pdf_rev_output) const
    {
        $if(lwi.z <= 0 | lwo.z <= 0)
        {
            eval_output = CSpectrum::zero();
            pdf_output = 0;
            pdf_rev_output = 0;
        }
        $else
        {
            f32 cos_theta_i = local_angle::cos_theta(lwi);
            f32 cos_theta_o = local_angle::cos_theta(lwo);

            var lwh = normalize(lwi + lwo);
            var cos_theta_d = dot(lwi, lwh);

            var phi_h = local_angle::phi(lwh);
            var sin_phi_h = cstd::sin(phi_h);
            var cos_phi_h = cstd::cos(phi_h);
            var cos_theta_h = local_angle::cos_theta(lwh);
            var sin_theta_h = local_angle::cos2sin(cos_theta_h);
            var D = microfacet::anisotropic_gtr2(sin_phi_h, cos_phi_h, sin_theta_h, cos_theta_h, ax, ay);

            var phi_i = local_angle::phi(lwi);
            var phi_o = local_angle::phi(lwo);
            var sin_phi_i = cstd::sin(phi_i), cos_phi_i = cstd::cos(phi_i);
            var sin_phi_o = cstd::sin(phi_o), cos_phi_o = cstd::cos(phi_o);
            var tan_theta_i = local_angle::tan_theta(lwi);
            var tan_theta_o = local_angle::tan_theta(lwo);

            var Go = microfacet::smith_anisotropic_gtr2(
                cos_phi_o, sin_phi_o, ax, ay, tan_theta_o);
            var Gi = microfacet::smith_anisotropic_gtr2(
                cos_phi_i, sin_phi_i, ax, ay, tan_theta_i);
            var G = Gi * Go;

            var F = fresnel.eval(cos_theta_d);

            eval_output = F * D * G / cstd::abs(4.0f * cos_theta_i * cos_theta_o);
            pdf_output = Go * D / (4.0f * lwo.z);
            pdf_rev_output = Gi * D / (4.0f * lwi.z);
        };
    }

CUJ_CLASS_END

BTRC_BUILTIN_END
