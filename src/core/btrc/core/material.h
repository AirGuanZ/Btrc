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

        void clear()
        {
            bsdf = CSpectrum::zero();
            dir = CVec3f();
            pdf = 0;
        }

    CUJ_CLASS_END

    virtual ~Shader() = default;

    virtual SampleResult sample(
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

    virtual boolean is_delta(CompileContext &cc) const = 0;
};

class Material : public Object
{
public:

    virtual ~Material() = default;

    virtual RC<Shader> create_shader(CompileContext &cc, const SurfacePoint &inct) const = 0;
};

BTRC_END
