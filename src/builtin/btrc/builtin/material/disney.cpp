#include <btrc/builtin/material/disney.h>
#include <btrc/builtin/material/utils/aggregate.h>
#include <btrc/builtin/material/utils/fresnel.h>
#include <btrc/builtin/material/utils/microfacet.h>
#include <btrc/utils/local_angle.h>

BTRC_BUILTIN_BEGIN

class DisneyBSDF : public BSDFComponent
{
    CSpectrum C_;
    CSpectrum C_tint_;
    f32 metallic_;
    f32 roughness_;
    f32 specular_;
    f32 specular_tint_;
    f32 anisotropic_;
    f32 sheen_;
    f32 sheen_tint_;
    f32 clearcoat_;
    f32 clearcoat_roughness_;
    f32 transmission_;
    f32 ior_;
    f32 transmission_roughness_;
    f32 trans_ax_;
    f32 trans_ay_;
    f32 ax_;
    f32 ay_;
    f32 diffuse_weight_;
    f32 specular_weight_;
    f32 clearcoat_weight_;
    f32 transmission_weight_;

    static f32 sqr(f32 x)
    {
        return x * x;
    }

    static f32 one_minus_5(f32 x)
    {
        var t = 1.0f - x;
        var t2 = t * t;
        return t2 * t2 * t;
    }

    static f32 eta_to_r0(f32 eta)
    {
        return sqr(eta - 1) / sqr(eta + 1);
    }

    static CSpectrum to_tint(const CSpectrum &color)
    {
        var lum = color.get_lum();
        return cstd::select(lum > 0, color / lum, CSpectrum::one());
    }

    static boolean refract(const CVec3f &w, const CVec3f &n, f32 eta, ref<CVec3f> output)
    {
        boolean result;
        var cos_theta_i = dot(w, n);
        var sin2_theta_t = sqr(eta) * (1.0f - sqr(cos_theta_i));
        var cos2_theta_t = 1.0f - sin2_theta_t;
        $if(cos2_theta_t <= 0)
        {
            result = false;
        }
        $else
        {
            var cos_theta_t = cstd::sqrt(cos2_theta_t);
            output = normalize((eta * cos_theta_i - cos_theta_t) * n - eta * w);
            result = true;
        };
        return result;
    }

    CSpectrum eval_diffuse(f32 cos_theta_i, f32 cos_theta_o, f32 cos_theta_d) const
    {
        var lambert = C_ / btrc_pi;
        var fl = one_minus_5(cos_theta_i);
        var fv = one_minus_5(cos_theta_o);
        var rr = 2.0f * roughness_ * cos_theta_d * cos_theta_d;
        var retro_refl = C_ / btrc_pi * rr * (fl + fv + fl * fv * (rr - 1));
        return lambert * (1.0f - 0.5f * fl) * (1.0f - 0.5f * fv) + retro_refl;
    }

    CSpectrum eval_sheen(f32 cos_theta_d) const
    {
        return 4.0f * sheen_ * lerp(CSpectrum::one(), C_tint_, sheen_tint_)
                             * one_minus_5(cos_theta_d);
    }

    CSpectrum eval_clearcoat(
        f32 cos_theta_i, f32 cos_theta_o, f32 tan_theta_i, f32 tan_theta_o,
        f32 sin_theta_h, f32 cos_theta_h, f32 cos_theta_d) const
    {
        CUJ_ASSERT(cos_theta_i > 0 & cos_theta_o > 0);
        var D = microfacet::gtr1(sin_theta_h, cos_theta_h, clearcoat_roughness_);
        var F = schlick_approx(0.04f, cos_theta_d);
        var G = microfacet::smith_gtr2(tan_theta_i, 0.25f)
              * microfacet::smith_gtr2(tan_theta_o, 0.25f);
        var val = clearcoat_ * D * F * G / cstd::abs(4.0f * cos_theta_i * cos_theta_o);
        return CSpectrum::from_rgb(val, val, val);
    }

    CSpectrum eval_trans(const CVec3f &lwi, const CVec3f &lwo, TransportMode mode) const
    {
        CUJ_ASSERT(lwi.z * lwo.z < 0);

        var cos_theta_i = local_angle::cos_theta(lwi);
        var cos_theta_o = local_angle::cos_theta(lwo);

        var eta = cstd::select(cos_theta_o > 0, ior_, 1.0f / ior_);
        var lwh = normalize(lwo + eta * lwi);
        $if(lwh.z < 0)
        {
            lwh = -lwh;
        };
        var cos_theta_d = dot(lwo, lwh);
        var F = dielectric_fresnel(ior_, 1, cos_theta_d);

        var phi_h = local_angle::phi(lwh);
        var sin_phi_h = cstd::sin(phi_h);
        var cos_phi_h = cstd::cos(phi_h);
        var cos_theta_h = local_angle::cos_theta(lwh);
        var sin_theta_h = local_angle::cos2sin(cos_theta_h);
        var D = microfacet::anisotropic_gtr2(
            sin_phi_h, cos_phi_h, sin_theta_h, cos_theta_h, trans_ax_, trans_ay_);

        var phi_i = local_angle::phi(lwi);
        var phi_o = local_angle::phi(lwo);
        var sin_phi_i = cstd::sin(phi_i), cos_phi_i = cstd::cos(phi_i);
        var sin_phi_o = cstd::sin(phi_o), cos_phi_o = cstd::cos(phi_o);
        var tan_theta_i = local_angle::tan_theta(lwi);
        var tan_theta_o = local_angle::tan_theta(lwo);
        var G = microfacet::smith_anisotropic_gtr2(
                    cos_phi_i, sin_phi_i, trans_ax_, trans_ay_, tan_theta_i)
              * microfacet::smith_anisotropic_gtr2(
                    cos_phi_o, sin_phi_o, trans_ax_, trans_ay_, tan_theta_o);

        var sdem = cos_theta_d + eta * dot(lwi, lwh);
        var corr_factor = mode == TransportMode::Radiance ? 1.0f / eta : 1.0f;

        var sqrtC = CSpectrum::from_rgb(cstd::sqrt(C_.r), cstd::sqrt(C_.g), cstd::sqrt(C_.b));

        var val = (1.0f - F) * D * G * eta * eta
                * dot(lwi, lwh) * dot(lwo, lwh)
                * corr_factor * corr_factor
                / (cos_theta_i * cos_theta_o * sdem * sdem);

        var trans_factor = cstd::select(cos_theta_o > 0, transmission_, f32(1.0f));
        return (1.0f - metallic_) * trans_factor * sqrtC * cstd::abs(val);
    }

    CSpectrum eval_inner_refl(const CVec3f &lwi, const CVec3f &lwo) const
    {
        CUJ_ASSERT(lwi.z < 0 & lwo.z < 0);

        var lwh = -normalize(lwi + lwo);
        CUJ_ASSERT(lwh.z > 0);

        var cos_theta_d = dot(lwo, lwh);
        var F = dielectric_fresnel(ior_, 1, cos_theta_d);

        var phi_h = local_angle::phi(lwh);
        var sin_phi_h = cstd::sin(phi_h);
        var cos_phi_h = cstd::cos(phi_h);
        var cos_theta_h = local_angle::cos_theta(lwh);
        var sin_theta_h = local_angle::cos2sin(cos_theta_h);
        var D = microfacet::anisotropic_gtr2(
            sin_phi_h, cos_phi_h, sin_theta_h, cos_theta_h, trans_ax_, trans_ay_);

        var phi_i = local_angle::phi(lwi);
        var phi_o = local_angle::phi(lwo);
        var sin_phi_i = cstd::sin(phi_i);
        var cos_phi_i = cstd::cos(phi_i);
        var sin_phi_o = cstd::sin(phi_o);
        var cos_phi_o = cstd::cos(phi_o);
        var tan_theta_i = local_angle::tan_theta(lwi);
        var tan_theta_o = local_angle::tan_theta(lwo);
        var G = microfacet::smith_anisotropic_gtr2(
                    cos_phi_i, sin_phi_i, trans_ax_, trans_ay_, tan_theta_i)
              * microfacet::smith_anisotropic_gtr2(
                    cos_phi_o, sin_phi_o, trans_ax_, trans_ay_, tan_theta_o);

        return transmission_ * C_ * cstd::abs(F * D * G / (4.0f * lwi.z * lwo.z));
    }

    CSpectrum eval_specular(const CVec3f &lwi, const CVec3f &lwo) const
    {
        CUJ_ASSERT(lwi.z > 0 & lwo.z > 0);

        var cos_theta_i = local_angle::cos_theta(lwi);
        var cos_theta_o = local_angle::cos_theta(lwo);

        var lwh = normalize(lwi + lwo);
        var cos_theta_d = dot(lwi, lwh);

        var Cspec = lerp(lerp(CSpectrum::one(), C_tint_, specular_tint_), C_, metallic_);
        var dielectric_fresnel = Cspec * builtin::dielectric_fresnel(ior_, 1.0f, cos_theta_d);
        var conductor_fresnel = schlick_approx(Cspec, cos_theta_d);
        var F = lerp(specular_ * dielectric_fresnel, conductor_fresnel, metallic_);

        var phi_h = local_angle::phi(lwh);
        var sin_phi_h = cstd::sin(phi_h);
        var cos_phi_h = cstd::cos(phi_h);
        var cos_theta_h = local_angle::cos_theta(lwh);
        var sin_theta_h = local_angle::cos2sin(cos_theta_h);
        var D = microfacet::anisotropic_gtr2(
            sin_phi_h, cos_phi_h, sin_theta_h, cos_theta_h, ax_, ay_);

        var phi_i = local_angle::phi(lwi);
        var phi_o = local_angle::phi(lwo);
        var sin_phi_i = cstd::sin(phi_i), cos_phi_i = cstd::cos(phi_i);
        var sin_phi_o = cstd::sin(phi_o), cos_phi_o = cstd::cos(phi_o);
        var tan_theta_i = local_angle::tan_theta(lwi);
        var tan_theta_o = local_angle::tan_theta(lwo);
        var G = microfacet::smith_anisotropic_gtr2(
                    cos_phi_i, sin_phi_i, ax_, ay_, tan_theta_i)
              * microfacet::smith_anisotropic_gtr2(
                    cos_phi_o, sin_phi_o, ax_, ay_, tan_theta_o);

        return F * D * G / cstd::abs(4.0f * cos_theta_i * cos_theta_o);
    }

    CVec3f sample_diffuse(const CVec2f &sam) const noexcept
    {
        return sample_hemisphere_zweighted(sam.x, sam.y);
    }

    CVec3f sample_specular(const CVec3f &lwo, const CVec2f &sam) const
    {
        $declare_scope;
        CVec3f lwi;

        var lwh = microfacet::sample_anisotropic_gtr2_vnor(
            lwo, ax_, ay_, sam);
        $if(lwh.z <= 0)
        {
            lwi = CVec3f(0);
            $exit_scope;
        };

        lwi = 2.0f * dot(lwo, lwh) * lwh - lwo;
        $if(lwi.z <= 0)
        {
            lwi = CVec3f(0);
            $exit_scope;
        };
        lwi = normalize(lwi);
        return lwi;
    }

    CVec3f sample_clearcoat(const CVec3f &lwo, const CVec2f &sam) const
    {
        $declare_scope;
        CVec3f lwi;

        var lwh = microfacet::sample_gtr1(clearcoat_roughness_, sam);
        $if(lwh.z <= 0)
        {
            lwi = CVec3f(0);
            $exit_scope;
        };

        lwi = 2.0f * dot(lwo, lwh) * lwh - lwo;
        $if(lwi.z <= 0)
        {
            lwi = CVec3f(0);
            $exit_scope;
        };
        lwi = normalize(lwi);
        return lwi;
    }

    CVec3f sample_transmission(const CVec3f &lwo, const CVec2f &sam) const
    {
        $declare_scope;
        CVec3f lwi;
        var lwh = microfacet::sample_anisotropic_gtr2(trans_ax_, trans_ay_, sam);
        $if(lwh.z <= 0)
        {
            lwi = CVec3f(0);
            $exit_scope;
        };
        $if((lwo.z > 0) != (dot(lwh, lwo) > 0))
        {
            lwi = CVec3f(0);
            $exit_scope;
        };
        var eta = cstd::select(lwo.z > 0, 1.0f / ior_, ior_);
        var owh = cstd::select(dot(lwh, lwo) > 0, CVec3f(lwh), -lwh);
        $if(!refract(lwo, owh, eta, lwi))
        {
            lwi = CVec3f(0);
            $exit_scope;
        };
        $if(lwi.z * lwo.z > 0 | ((lwi.z > 0) != (dot(lwh, lwi) > 0)))
        {
            lwi = CVec3f(0);
            $exit_scope;
        };
        lwi = normalize(lwi);
        return lwi;
    }

    CVec3f sample_inner_refl(const CVec3f &lwo, const CVec2f &sam) const
    {
        CUJ_ASSERT(lwo.z < 0);
        $declare_scope;
        CVec3f lwi;
        var lwh = microfacet::sample_anisotropic_gtr2(trans_ax_, trans_ay_, sam);
        $if(lwh.z <= 0)
        {
            lwi = CVec3f(0);
            $exit_scope;
        };
        lwi = 2.0f * dot(lwo, lwh) * lwh - lwo;
        $if(lwi.z > 0)
        {
            lwi = CVec3f(0);
            $exit_scope;
        };
        lwi = normalize(lwi);
        return lwi;
    }

    f32 pdf_diffuse(const CVec3f &lwi, const CVec3f &lwo) const
    {
        CUJ_ASSERT(lwi.z > 0 & lwo.z > 0);
        return pdf_sample_hemisphere_zweighted(lwi);
    }

    std::pair<f32, f32> pdf_specular_clearcoat(const CVec3f &lwi, const CVec3f &lwo) const
    {
        CUJ_ASSERT(lwi.z > 0 & lwo.z > 0);

        var lwh = normalize(lwi + lwo);
        var phi_h = local_angle::phi(lwh);
        var sin_phi_h = cstd::sin(phi_h);
        var cos_phi_h = cstd::cos(phi_h);
        var cos_theta_h = local_angle::cos_theta(lwh);
        var sin_theta_h = local_angle::cos2sin(cos_theta_h);
        var cos_theta_d = dot(lwi, lwh);

        var cos_phi_o = cstd::cos(local_angle::phi(lwo));
        var sin_phi_o = local_angle::cos2sin(cos_phi_o);
        var tan_theta_o = local_angle::tan_theta(lwo);

        var specular_D = microfacet::anisotropic_gtr2(
            sin_phi_h, cos_phi_h, sin_theta_h, cos_theta_h, ax_, ay_);

        var pdf_specular = microfacet::smith_anisotropic_gtr2(
                            cos_phi_o, sin_phi_o, ax_, ay_, tan_theta_o)
                         * specular_D / (4.0f * lwo.z);

        var clearcoat_D = microfacet::gtr1(
            sin_theta_h, cos_theta_h, clearcoat_roughness_);
        var pdf_clearcoat = cos_theta_h * clearcoat_D / (4.0f * cos_theta_d);

        return { pdf_specular, pdf_clearcoat };
    }

    f32 pdf_transmission(const CVec3f &lwi, const CVec3f &lwo) const
    {
        CUJ_ASSERT(lwi.z * lwo.z < 0);

        $declare_scope;
        f32 result;

        var eta = cstd::select(lwo.z > 0, ior_, 1.0f / ior_);
        var lwh = normalize(lwo + eta * lwi);
        $if(lwh.z < 0)
        {
            lwh = -lwh;
        };

        $if(((lwo.z > 0) != (dot(lwh, lwo) > 0)) |
            ((lwi.z > 0) != (dot(lwh, lwi) > 0)))
        {
            result = 0;
            $exit_scope;
        };

        var sdem = dot(lwo, lwh) + eta * dot(lwi, lwh);
        var dwh_to_dwi = eta * eta * dot(lwi, lwh) / (sdem * sdem);

        var phi_h = local_angle::phi(lwh);
        var sin_phi_h = cstd::sin(phi_h);
        var cos_phi_h = cstd::cos(phi_h);
        var cos_theta_h = local_angle::cos_theta(lwh);
        var sin_theta_h = local_angle::cos2sin(cos_theta_h);

        var D = microfacet::anisotropic_gtr2(
            sin_phi_h, cos_phi_h, sin_theta_h, cos_theta_h, trans_ax_, trans_ay_);
        result = cstd::abs(dot(lwi, lwh) * D * dwh_to_dwi);

        return result;
    }

    f32 pdf_inner_refl(const CVec3f &lwi, const CVec3f &lwo) const
    {
        CUJ_ASSERT(lwi.z < 0 & lwo.z < 0);

        var lwh = -normalize(lwi + lwo);
        var phi_h = local_angle::phi(lwh);
        var sin_phi_h = cstd::sin(phi_h);
        var cos_phi_h = cstd::cos(phi_h);
        var cos_theta_h = local_angle::cos_theta(lwh);
        var sin_theta_h = local_angle::cos2sin(cos_theta_h);
        var cos_theta_d = dot(lwi, lwh);

        var D = microfacet::anisotropic_gtr2(
            sin_phi_h, cos_phi_h, sin_theta_h, cos_theta_h, trans_ax_, trans_ay_);

        return cstd::abs(cos_theta_h * D / (4.0f * cos_theta_d));
    }

public:

    DisneyBSDF(
        const CSpectrum &base_color,
        f32 metallic,
        f32 roughness,
        f32 specular,
        f32 specular_tint,
        f32 anisotropic,
        f32 sheen,
        f32 sheen_tint,
        f32 clearcoat,
        f32 clearcoat_gloss,
        f32 transmission,
        f32 transmission_roughness,
        f32 ior)
    {
        C_ = base_color;
        C_tint_ = to_tint(C_);
        metallic_ = metallic;
        roughness_ = roughness;
        specular_ = specular;
        specular_tint_ = specular_tint;
        anisotropic_ = anisotropic;
        sheen_ = sheen;
        sheen_tint_ = sheen_tint;
        transmission_ = transmission;
        transmission_roughness_ = transmission_roughness;
        ior_ = cstd::max(1.01f, ior);

        var aspect = 1.0f;
        $if(anisotropic > 0)
        {
            aspect = cstd::sqrt(1.0f - 0.9f * anisotropic);
        };
        ax_ = cstd::max(0.001f, sqr(roughness) / aspect);
        ay_ = cstd::max(0.001f, sqr(roughness) * aspect);

        trans_ax_ = cstd::max(0.001f, sqr(transmission_roughness) / aspect);
        trans_ay_ = cstd::max(0.001f, sqr(transmission_roughness) * aspect);

        clearcoat_ = clearcoat;
        clearcoat_roughness_ = cstd::max(0.0001f, sqr(lerp(0.1f, 0.0f, clearcoat_gloss)));

        var A = cstd::clamp(base_color.get_lum() * (1.0f - metallic), 0.3f, 0.7f);
        var B = 1.0f - A;

        diffuse_weight_      = A * (1.0f - transmission);
        transmission_weight_ = A * transmission;
        specular_weight_     = B * 2.0f / (2.0f + clearcoat);
        clearcoat_weight_    = B * clearcoat / (2.0f + clearcoat);
    }

    CSpectrum eval(CompileContext &cc, ref<CVec3f> lwi, ref<CVec3f> lwo, TransportMode mode) const override
    {
        $declare_scope;
        CSpectrum result;

        $if(cstd::abs(lwi.z) < 1e-4f | cstd::abs(lwo.z) < 1e-4f)
        {
            result = CSpectrum::zero();
            $exit_scope;
        };

        // transmission

        $if(lwi.z * lwo.z < 0)
        {
            $if(transmission_ == 0.0f)
            {
                result = CSpectrum::zero();
                $exit_scope;
            };

            result = eval_trans(lwi, lwo, mode);
            $exit_scope;
        };

        // inner refl

        $if(lwi.z < 0 & lwo.z < 0)
        {
            $if(transmission_ == 0.0f)
            {
                result = CSpectrum::zero();
                $exit_scope;
            };

            result = eval_inner_refl(lwi, lwo);
            $exit_scope;
        };

        // reflection

        $if(lwi.z <= 0 | lwo.z <= 0)
        {
            result = CSpectrum::zero();
            $exit_scope;
        };

        var cos_theta_i = local_angle::cos_theta(lwi);
        var cos_theta_o = local_angle::cos_theta(lwo);

        var lwh = normalize(lwi + lwo);
        var cos_theta_d = dot(lwi, lwh);

        var diffuse = CSpectrum::zero(), sheen = CSpectrum::zero();
        $if(metallic_ < 1.0f)
        {
            diffuse = eval_diffuse(cos_theta_i, cos_theta_o, cos_theta_d);
            $if(sheen_ > 0.0f)
            {
                sheen = eval_sheen(cos_theta_d);
            };
        };

        var specular = eval_specular(lwi, lwo);

        var clearcoat = CSpectrum::zero();
        $if(clearcoat_ > 0.0f)
        {
            var tan_theta_i = local_angle::tan_theta(lwi);
            var tan_theta_o = local_angle::tan_theta(lwo);
            var cos_theta_h = local_angle::cos_theta(lwh);
            var sin_theta_h = local_angle::cos2sin(cos_theta_h);
            clearcoat = eval_clearcoat(
                cos_theta_i, cos_theta_o, tan_theta_i, tan_theta_o,
                sin_theta_h, cos_theta_h, cos_theta_d);
        };

        result = (1.0f - metallic_) * (1.0f - transmission_)
                  * (diffuse + sheen) + specular + clearcoat;
        return result;
    }

    Shader::SampleResult sample(CompileContext &cc, ref<CVec3f> lwo, ref<CVec3f> sam, TransportMode mode) const override
    {
        $declare_scope;
        Shader::SampleResult result;

        auto handle_bad_sample = [&]
        {
            var v = result.bsdf * cstd::abs(result.dir.z) / result.pdf;
            $if(cstd::abs(cstd::max(cstd::max(v.r, v.g), v.b)) > 10)
            {
                result.clear();
            };
        };

        $if(cstd::abs(lwo.z) < 1e-4f)
        {
            result.clear();
            $exit_scope;
        };

        // transmission and inner refl

        $if(lwo.z < 0)
        {
            $if(transmission_ == 0.0f)
            {
                result.clear();
                $exit_scope;
            };

            CVec3f lwi;
            f32 macro_F = dielectric_fresnel(ior_, 1, lwo.z);
            macro_F = cstd::clamp(macro_F, 0.1f, 0.9f);
            $if(sam.x >= macro_F)
            {
                lwi = sample_transmission(lwo, CVec2f(sam.y, sam.z));
            }
            $else
            {
                lwi = sample_inner_refl(lwo, CVec2f(sam.y, sam.z));
            };

            $if(lwi.x == 0 & lwi.y == 0 & lwi.z == 0)
            {
                result.clear();
                $exit_scope;
            };

            result.dir = lwi;
            result.bsdf = eval(cc, lwi, lwo, mode);
            result.pdf = pdf(cc, lwi, lwo, mode);
            handle_bad_sample();
            $exit_scope;
        };

        // reflection + transmission

        var sam_selector = sam.x;
        var new_sam = CVec2f(sam.y, sam.z);

        CVec3f lwi;

        $if(sam_selector < diffuse_weight_)
        {
            lwi = sample_diffuse(new_sam);
        }
        $else
        {
            sam_selector = sam_selector - diffuse_weight_;
            $if(sam_selector < transmission_weight_)
            {
                lwi = sample_transmission(lwo, new_sam);
            }
            $else
            {
                sam_selector = sam_selector - transmission_weight_;
                $if(sam_selector < specular_weight_)
                {
                    lwi = sample_specular(lwo, new_sam);
                }
                $else
                {
                    lwi = sample_clearcoat(lwo, new_sam);
                };
            };
        };

        $if(lwi.x == 0 & lwi.y == 0 & lwi.z == 0)
        {
            result.clear();
            $exit_scope;
        };

        result.dir = lwi;
        result.bsdf = eval(cc, lwi, lwo, mode);
        result.pdf = pdf(cc, lwi, lwo, mode);
        handle_bad_sample();
        return result;
    }

    f32 pdf(CompileContext &cc, ref<CVec3f> lwi, ref<CVec3f> lwo, TransportMode mode) const override
    {
        $declare_scope;
        f32 result;

        $if(cstd::abs(lwi.z) < 1e-4f | cstd::abs(lwo.z) < 1e-4f)
        {
            result = 0;
            $exit_scope;
        };

        // transmission and inner refl

        $if(lwo.z < 0)
        {
            $if(transmission_ == 0.0f)
            {
                result = 0;
                $exit_scope;
            };
            var macro_F = dielectric_fresnel(ior_, 1, lwo.z);
            macro_F = cstd::clamp(macro_F, 0.1f, 0.9f);
            $if(lwi.z > 0)
            {
                result = (1.0f - macro_F) * pdf_transmission(lwi, lwo);
                $exit_scope;
            };
            result = macro_F * pdf_inner_refl(lwi, lwo);
            $exit_scope;
        };

        // transmission and refl

        $if(lwi.z < 0)
        {
            result = transmission_weight_ * pdf_transmission(lwi, lwo);
            $exit_scope;
        };

        var diffuse = pdf_diffuse(lwi, lwo);

        auto [specular, clearcoat] = pdf_specular_clearcoat(lwi, lwo);

        result = diffuse_weight_ * diffuse
               + specular_weight_ * specular
               + clearcoat_weight_ * clearcoat;
        return result;
    }

    CSpectrum albedo(CompileContext &cc) const override
    {
        return C_;
    }
};

void DisneyMaterial::set_base_color(RC<Texture2D> tex)
{
    base_color_ = std::move(tex);
}

void DisneyMaterial::set_metallic(RC<Texture2D> tex)
{
    metallic_ = std::move(tex);
}

void DisneyMaterial::set_roughness(RC<Texture2D> tex)
{
    roughness_ = std::move(tex);
}

void DisneyMaterial::set_specular(RC<Texture2D> tex)
{
    specular_ = std::move(tex);
}

void DisneyMaterial::set_specular_tint(RC<Texture2D> tex)
{
    specular_tint_ = std::move(tex);
}

void DisneyMaterial::set_anisotropic(RC<Texture2D> tex)
{
    anisotropic_ = std::move(tex);
}

void DisneyMaterial::set_sheen(RC<Texture2D> tex)
{
    sheen_ = std::move(tex);
}

void DisneyMaterial::set_sheen_tint(RC<Texture2D> tex)
{
    sheen_tint_ = std::move(tex);
}

void DisneyMaterial::set_clearcoat(RC<Texture2D> tex)
{
    clearcoat_ = std::move(tex);
}

void DisneyMaterial::set_clearcoat_gloss(RC<Texture2D> tex)
{
    clearcoat_gloss_ = std::move(tex);
}

void DisneyMaterial::set_transmission(RC<Texture2D> tex)
{
    transmission_ = std::move(tex);
}

void DisneyMaterial::set_transmission_roughness(RC<Texture2D> tex)
{
    transmission_roughness_ = std::move(tex);
}

void DisneyMaterial::set_ior(RC<Texture2D> tex)
{
    ior_ = std::move(tex);
}

void DisneyMaterial::set_normal(RC<NormalMap> normal)
{
    normal_ = std::move(normal);
}

RC<Shader> DisneyMaterial::create_shader(CompileContext &cc, const SurfacePoint &inct) const
{
    ShaderFrame frame;
    frame.geometry = inct.frame;
    frame.shading = inct.frame.rotate_to_new_z(inct.interp_z);
    frame.shading = normal_->adjust_frame(cc, inct, frame.shading);

    var base_color             = base_color_->sample_spectrum(cc, inct);
    var metallic               = metallic_->sample_float(cc, inct);
    var roughness              = roughness_->sample_float(cc, inct);
    var transmission           = transmission_->sample_float(cc, inct);
    var transmission_roughness = transmission_roughness_->sample_float(cc, inct);
    var ior                    = ior_->sample_float(cc, inct);
    var specular               = specular_->sample_float(cc, inct);
    var specular_tint          = specular_tint_->sample_float(cc, inct);
    var anisotropic            = anisotropic_->sample_float(cc, inct);
    var sheen                  = sheen_->sample_float(cc, inct);
    var sheen_tint             = sheen_tint_->sample_float(cc, inct);
    var clearcoat              = clearcoat_->sample_float(cc, inct);
    var clearcoat_gloss        = clearcoat_gloss_->sample_float(cc, inct);

    auto comp = newBox<DisneyBSDF>(
        base_color, metallic, roughness, specular, specular_tint,
        anisotropic, sheen, sheen_tint, clearcoat, clearcoat_gloss,
        transmission, transmission_roughness, ior);

    auto shader = newRC<BSDFAggregate>(as_shared(), false, frame);
    shader->add_component(1, std::move(comp));
    return shader;
}

RC<Material> DisneyMaterialCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    auto get_or_constant = [&](std::string_view name, float default_value)->RC<Texture2D>
    {
        if(auto n = node->find_child_node(name))
            return context.create<Texture2D>(n);
        auto result = newRC<Constant2D>();
        result->set_value(default_value);
        return result;
    };

    auto base_color      = context.create<Texture2D>(node->child_node("color"));
    auto metallic        = context.create<Texture2D>(node->child_node("metallic"));
    auto roughness       = context.create<Texture2D>(node->child_node("roughness"));
    auto transmission    = get_or_constant("transmission", 0);
    auto ior             = get_or_constant("ior", 1.5f);
    auto specular        = get_or_constant("specular", 1);
    auto specular_tint   = get_or_constant("specular_tint", 0);
    auto anisotropic     = get_or_constant("anisotropic", 0);
    auto sheen           = get_or_constant("sheen", 0);
    auto sheen_tint      = get_or_constant("sheen_tint", 0);
    auto clearcoat       = get_or_constant("clearcoat", 0);
    auto clearcoat_gloss = get_or_constant("clearcoat_gloss", 1);

    auto transmission_roughness = roughness;
    if(auto n = node->find_child_node("transmission_roughness"))
        transmission_roughness = context.create<Texture2D>(n);

    auto normal = newRC<NormalMap>();
    normal->load(node, context);

    auto result = newRC<DisneyMaterial>();
    result->set_base_color(std::move(base_color));
    result->set_metallic(std::move(metallic));
    result->set_roughness(std::move(roughness));
    result->set_transmission(std::move(transmission));
    result->set_ior(std::move(ior));
    result->set_specular(std::move(specular));
    result->set_specular_tint(std::move(specular_tint));
    result->set_anisotropic(std::move(anisotropic));
    result->set_sheen(std::move(sheen));
    result->set_sheen_tint(std::move(sheen_tint));
    result->set_clearcoat(std::move(clearcoat));
    result->set_clearcoat_gloss(std::move(clearcoat_gloss));
    result->set_transmission_roughness(std::move(transmission_roughness));
    result->set_normal(std::move(normal));
    return result;
}

BTRC_BUILTIN_END
