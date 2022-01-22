#include <btrc/core/material/diffuse.h>

BTRC_CORE_BEGIN

class DiffuseBSDF : public BSDFWithBlackFringesHandling
{
    CSpectrum albedo_;

public:

    DiffuseBSDF(
        const CFrame &geometry_frame,
        const CFrame &shading_frame,
        const CSpectrum &albedo)
        : BSDFWithBlackFringesHandling(geometry_frame, shading_frame),
          albedo_(albedo)
    {
        
    }

    SampleResult sample_impl(
        const CVec3f &wo, const CVec3f &sam, TransportMode mode) const override
    {
        $declare_scope;
        SampleResult result;

        $if(dot(wo, shading_frame_.z) <= 0)
        {
            result.bsdf = albedo_.get_type()->create_czero();
            result.dir = CVec3f();
            result.pdf = 0;
            $exit_scope;
        };

        var local_wi = sample_hemisphere_zweighted(sam.x, sam.y);
        result.bsdf = albedo_ / btrc_pi;
        result.dir = shading_frame_.local_to_global(local_wi);
        result.pdf = pdf_sample_hemisphere_zweighted(local_wi);

        return result;
    }

    CSpectrum eval_impl(
        const CVec3f &wi, const CVec3f &wo, TransportMode mode) const override
    {
        CSpectrum result;
        $if(dot(wi, shading_frame_.z) <= 0 | dot(wo, shading_frame_.z) <= 0)
        {
            result = albedo_.get_type()->create_czero();
        }
        $else
        {
            result = albedo_ / btrc_pi;
        };
        return result;
    }

    f32 pdf_impl(
        const CVec3f &wi, const CVec3f &wo, TransportMode mode) const override
    {
        f32 result;
        $if(dot(wi, shading_frame_.z) <= 0 | dot(wo, shading_frame_.z) <= 0)
        {
            result = 0;
        }
        $else
        {
            var local_wi = normalize(shading_frame_.global_to_local(wi));
            result = pdf_sample_hemisphere_zweighted(local_wi);
        };
        return result;
    }

    CSpectrum albedo() const override
    {
        return albedo_;
    }

    CVec3f normal() const override
    {
        return shading_frame_.z;
    }

    bool is_delta() const override
    {
        return false;
    }
};

void Diffuse::set_albedo(const Spectrum &albedo)
{
    albedo_ = albedo;
}

Box<BSDF> Diffuse::create_bsdf(const CIntersection &inct) const
{
    return newBox<DiffuseBSDF>(
        inct.frame, inct.frame.rotate_to_new_z(inct.interp_normal), albedo_);
}

BTRC_CORE_END
