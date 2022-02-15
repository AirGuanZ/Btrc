#include <btrc/builtin/material/black.h>

BTRC_BUILTIN_BEGIN

class BlackShader : public Shader
{
    CVec3f normal_;

public:

    explicit BlackShader(ref<CVec3f> normal)
    {
        normal_ = normal;
    }

    SampleResult sample(
        ref<CVec3f>   wo,
        ref<CVec3f>   sam,
        TransportMode mode) const override
    {
        SampleResult result;
        result.bsdf = CSpectrum::zero();
        return result;
    }

    CSpectrum eval(
        ref<CVec3f>   wi,
        ref<CVec3f>   wo,
        TransportMode mode) const override
    {
        return CSpectrum::zero();
    }

    f32 pdf(
        ref<CVec3f>   wi,
        ref<CVec3f>   wo,
        TransportMode mode) const override
    {
        return 0;
    }

    CSpectrum albedo() const override
    {
        return CSpectrum::zero();
    }

    CVec3f normal() const override
    {
        return normal_;
    }

    boolean is_delta() const override
    {
        return false;
    }
};

RC<Shader> Black::create_shader(const SurfacePoint &inct) const
{
    return newRC<BlackShader>(inct.frame.z);
}

BTRC_BUILTIN_END
