#pragma once

#include <btrc/core/context.h>
#include <btrc/core/geometry.h>
#include <btrc/core/spectrum.h>

BTRC_BEGIN

enum class TransportMode
{
    Radiance, Importance
};

class Shader
{
public:

    CUJ_CLASS_BEGIN(SampleResult)

        CUJ_MEMBER_VARIABLE(CSpectrum, bsdf)
        CUJ_MEMBER_VARIABLE(CVec3f,    dir)
        CUJ_MEMBER_VARIABLE(f32,       pdf)
        CUJ_MEMBER_VARIABLE(boolean,   is_delta)

        void clear()
        {
            bsdf = CSpectrum::zero();
            dir = CVec3f();
            pdf = 0;
            is_delta = false;
        }

    CUJ_CLASS_END

    CUJ_CLASS_BEGIN(SampleBidirResult)
        
        CUJ_MEMBER_VARIABLE(CSpectrum, bsdf)
        CUJ_MEMBER_VARIABLE(CVec3f,    dir)
        CUJ_MEMBER_VARIABLE(f32,       pdf)
        CUJ_MEMBER_VARIABLE(f32,       pdf_rev)
        CUJ_MEMBER_VARIABLE(boolean,   is_delta)
    
        void clear()
        {
            bsdf = CSpectrum::zero();
            dir = CVec3f();
            pdf = 0;
            pdf_rev = 0;
            is_delta = false;
        }

    CUJ_CLASS_END

    virtual ~Shader() = default;

    virtual SampleResult sample(
        CompileContext &cc,
        ref<CVec3f>     wo,
        ref<CVec3f>     sam,
        TransportMode   mode) const = 0;

    virtual SampleBidirResult sample_bidir(
        CompileContext &cc,
        ref<CVec3f>     wo,
        ref<CVec3f>     sam,
        TransportMode   mode) const = 0;

    virtual CSpectrum eval(
        CompileContext &cc,
        ref<CVec3f>     wi,
        ref<CVec3f>     wo,
        TransportMode   mode) const = 0;

    virtual f32 pdf(
        CompileContext &cc,
        ref<CVec3f>     wi,
        ref<CVec3f>     wo,
        TransportMode   mode) const = 0;

    virtual CSpectrum albedo(CompileContext &cc) const = 0;

    virtual CVec3f normal(CompileContext &cc) const = 0;

    static SampleResult discard_pdf_rev(const SampleBidirResult &result);
};

class Material : public Object
{
public:

    virtual RC<Shader> create_shader(CompileContext &cc, const SurfacePoint &inct) const = 0;
};

BTRC_END
