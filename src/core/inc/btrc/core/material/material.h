#pragma once

#include <btrc/core/spectrum/spectrum.h>

BTRC_CORE_BEGIN

struct CIntersection
{
    CVec3f position;
    CFrame frame;
    CVec3f interp_normal;
    CVec2f uv;
    CVec2f tex_coord;
};

enum class TransportMode
{
    Radiance, Importance
};

class BSDF
{
public:

    struct SampleResult
    {
        CSpectrum bsdf;
        CVec3f    dir;
        f32       pdf;
    };

    virtual ~BSDF() = default;

    virtual SampleResult sample(
        const CVec3f &wo, const CVec3f &sam, TransportMode mode) const = 0;

    virtual CSpectrum eval(
        const CVec3f &wi, const CVec3f &wo, TransportMode mode) const = 0;

    virtual f32 pdf(
        const CVec3f &wi, const CVec3f &wo, TransportMode mode) const = 0;

    virtual CSpectrum albedo() const = 0;

    virtual CVec3f normal() const = 0;

    virtual bool is_delta() const = 0;
};

class BSDFWithBlackFringesHandling : public BSDF
{
protected:

    CFrame geometry_frame_;
    CFrame shading_frame_;

private:

    boolean is_black_fringes(ref<CVec3f> v) const;

    boolean is_black_fringes(ref<CVec3f> w1, ref<CVec3f> w2) const;

    SampleResult sample_black_fringes(const CVec3f &wo, const CVec3f &sam) const;

    CSpectrum eval_black_fringes(const CVec3f &wi, const CVec3f &wo) const;

    f32 pdf_black_fringes(const CVec3f &wi, const CVec3f &wo) const;

public:

    BSDFWithBlackFringesHandling(CFrame geometry_frame, CFrame shading_frame);

    virtual SampleResult sample_impl(
        const CVec3f &wo, const CVec3f &sam, TransportMode mode) const = 0;

    virtual CSpectrum eval_impl(
        const CVec3f &wi, const CVec3f &wo, TransportMode mode) const = 0;

    virtual f32 pdf_impl(
        const CVec3f &wi, const CVec3f &wo, TransportMode mode) const = 0;

    SampleResult sample(
        const CVec3f &wo, const CVec3f &sam, TransportMode mode) const final;

    CSpectrum eval(
        const CVec3f &wi, const CVec3f &wo, TransportMode mode) const final;

    f32 pdf(
        const CVec3f &wi, const CVec3f &wo, TransportMode mode) const final;
};

class Material
{
public:

    virtual ~Material() = default;

    virtual Box<BSDF> create_bsdf(const CIntersection &inct) const = 0;
};

BTRC_CORE_END
