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
        CompileContext &cc,
        ref<CVec3f>     wo,
        ref<CVec3f>     sam,
        TransportMode   mode) const override
    {
        SampleResult result;
        result.clear();
        return result;
    }

    CSpectrum eval(
        CompileContext &cc,
        ref<CVec3f>     wi,
        ref<CVec3f>     wo,
        TransportMode   mode) const override
    {
        return CSpectrum::zero();
    }

    f32 pdf(
        CompileContext &cc,
        ref<CVec3f>     wi,
        ref<CVec3f>     wo,
        TransportMode mode) const override
    {
        return 0;
    }

    CSpectrum albedo(CompileContext &cc) const override
    {
        return CSpectrum::zero();
    }

    CVec3f normal(CompileContext &cc) const override
    {
        return normal_;
    }
};

RC<Shader> Black::create_shader(CompileContext &cc, const SurfacePoint &inct) const
{
    return newRC<BlackShader>(inct.frame.z);
}

RC<Material> BlackCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    return newRC<Black>();
}

BTRC_BUILTIN_END
