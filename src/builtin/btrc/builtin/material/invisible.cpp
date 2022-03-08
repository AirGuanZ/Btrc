#include <btrc/builtin/material/invisible.h>
#include <btrc/builtin/material/utils/shader_closure.h>

BTRC_BUILTIN_BEGIN

class InvisibleShader : public Shader
{
public:

    CVec3f normal_;

    SampleResult sample(CompileContext &cc, ref<CVec3f> wo, ref<CVec3f> sam, TransportMode mode) const override
    {
        SampleResult result;
        result.bsdf = CSpectrum::one() / cstd::abs(dot(normal_, normalize(wo)));
        result.dir = -wo;
        result.pdf = 1;
        return result;
    }

    CSpectrum eval(CompileContext &cc, ref<CVec3f> wi, ref<CVec3f> wo, TransportMode mode) const override
    {
        return CSpectrum::zero();
    }

    f32 pdf(CompileContext &cc, ref<CVec3f> wi, ref<CVec3f> wo, TransportMode mode) const override
    {
        return 0.0f;
    }

    CSpectrum albedo(CompileContext &cc) const override
    {
        return CSpectrum::one();
    }

    CVec3f normal(CompileContext &cc) const override
    {
        return normal_;
    }

    boolean is_delta(CompileContext &cc) const override
    {
        return true;
    }
};

RC<Shader> InvisibleSurface::create_shader(CompileContext &cc, const SurfacePoint &inct) const
{
    auto shader = newRC<InvisibleShader>();
    shader->normal_ = inct.frame.z;
    return shader;
}

RC<Material> InvisibleSurfaceCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    return newRC<InvisibleSurface>();
}

BTRC_BUILTIN_END
