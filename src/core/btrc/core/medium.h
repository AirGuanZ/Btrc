#pragma once

#include <btrc/core/context.h>
#include <btrc/core/sampler.h>
#include <btrc/core/spectrum.h>

BTRC_BEGIN

class PhaseShader
{
public:

    CUJ_CLASS_BEGIN(SampleResult)

        CUJ_MEMBER_VARIABLE(CSpectrum, phase)
        CUJ_MEMBER_VARIABLE(CVec3f,    dir)
        CUJ_MEMBER_VARIABLE(f32,       pdf)

        void clear()
        {
            phase = CSpectrum::zero();
            dir = CVec3f(0);
            pdf = 0;
        }

    CUJ_CLASS_END

    virtual ~PhaseShader() = default;

    virtual SampleResult sample(CompileContext &cc, ref<CVec3f> wo, ref<Sam3> sam) const = 0;

    virtual CSpectrum eval(CompileContext &cc, ref<CVec3f> wi, ref<CVec3f> wo) const = 0;

    virtual f32 pdf(CompileContext &cc, ref<CVec3f> wi, ref<CVec3f> wo) const = 0;
};

class HenyeyGreensteinPhaseShader : public PhaseShader
{
public:

    void set_g(f32 g);

    void set_color(ref<CSpectrum> color);

    SampleResult sample(CompileContext &cc, ref<CVec3f> wo, ref<Sam3> sam) const override;

    CSpectrum eval(CompileContext &cc, ref<CVec3f> wi, ref<CVec3f> wo) const override;

    f32 pdf(CompileContext &cc, ref<CVec3f> wi, ref<CVec3f> wo) const override;

private:

    f32 g_;
    CSpectrum color_;
};

using MediumID = uint32_t;
using CMediumID = cuj::cxx<MediumID>;

class Medium : public Object
{
public:

    struct SampleResult
    {
        boolean         scattered;
        CSpectrum       throughput;
        CVec3f          position;
        RC<PhaseShader> shader;
    };

    virtual SampleResult sample(
        CompileContext &cc,
        ref<CVec3f>     a,
        ref<CVec3f>     b,
        ref<CVec3f>     uvw_a,
        ref<CVec3f>     uvw_b,
        Sampler        &sampler) const = 0;

    virtual CSpectrum tr(
        CompileContext &cc,
        ref<CVec3f>     a,
        ref<CVec3f>     b,
        ref<CVec3f>     uvw_a,
        ref<CVec3f>     uvw_b,
        Sampler        &sampler) const = 0;

    SampleResult sample(
        CompileContext &cc,
        ref<CVec3f>     a,
        ref<CVec3f>     b,
        Sampler        &sampler) const;

    CSpectrum tr(
        CompileContext &cc,
        ref<CVec3f>     a,
        ref<CVec3f>     b,
        Sampler        &sampler) const;

    virtual float get_priority() const = 0;
};

class TransformMedium : public Medium
{
public:

    void set_tranformed(RC<Medium> transformed);

    void set_transform(const Transform3D &world_to_local);

    SampleResult sample(
        CompileContext &cc,
        ref<CVec3f>     a,
        ref<CVec3f>     b,
        ref<CVec3f>     uvw_a,
        ref<CVec3f>     uvw_b,
        Sampler        &sampler) const override;

    CSpectrum tr(
        CompileContext &cc,
        ref<CVec3f>     a,
        ref<CVec3f>     b,
        ref<CVec3f>     uvw_a,
        ref<CVec3f>     uvw_b,
        Sampler        &sampler) const override;

    float get_priority() const override;

private:
    
    Transform3D world_to_local_;
    BTRC_OBJECT(Medium, transformed_);
};

BTRC_END
