#pragma once

#include <btrc/core/context.h>
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

    virtual SampleResult sample(
        CompileContext &cc, ref<CVec3f> wo, ref<CVec3f> sam) const = 0;

    virtual CSpectrum eval(
        CompileContext &cc, ref<CVec3f> wi, ref<CVec3f> wo) const = 0;

    virtual f32 pdf(
        CompileContext &cc, ref<CVec3f> wi, ref<CVec3f> wo) const = 0;
};

using MediumID = uint32_t;
using CMediumID = cuj::cxx<MediumID>;

constexpr MediumID MEDIUM_ID_VOID = MediumID(-1);

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

    virtual SampleResult sample(CompileContext &cc, ref<CVec3f> a, ref<CVec3f> b, ref<CRNG> rng) const = 0;

    virtual CSpectrum tr(CompileContext &cc, ref<CVec3f> a, ref<CVec3f> b, ref<CRNG> rng) const = 0;

    virtual float get_priority() const = 0;
};

BTRC_END
