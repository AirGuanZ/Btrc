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
        ref<CVec3f>   wo,
        ref<CVec3f>   sam,
        TransportMode mode) const = 0;

    virtual CSpectrum eval(
        ref<CVec3f>   wi,
        ref<CVec3f>   wo,
        TransportMode mode) const = 0;

    virtual f32 pdf(
        ref<CVec3f>   wi,
        ref<CVec3f>   wo,
        TransportMode mode) const = 0;

    virtual CSpectrum albedo() const = 0;

    virtual CVec3f normal() const = 0;

    virtual boolean is_delta() const = 0;
};

class Material : public Object
{
public:

    virtual ~Material() = default;

    virtual RC<Shader> create_shader(const SurfacePoint &inct) const = 0;
};

BTRC_END
