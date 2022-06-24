#include <btrc/builtin/material/invisible.h>
#include <btrc/builtin/material/utils/shader_closure.h>

BTRC_BUILTIN_BEGIN

class InvisibleShader : public Shader
{
public:

    CVec3f normal_;

    SampleResult sample(CompileContext &cc, ref<CVec3f> wo, ref<Sam3> sam, TransportMode mode) const override
    {
        return discard_pdf_rev(sample_bidir(cc, wo, sam, mode));
    }
    
    SampleBidirResult sample_bidir(CompileContext &cc, ref<CVec3f> wo, ref<Sam3> sam, TransportMode mode) const override
    {
        SampleBidirResult result;
        result.bsdf = CSpectrum::one() / cstd::abs(dot(normal_, normalize(wo)));
        result.dir = -wo;
        result.pdf = 1;
        result.pdf_rev = 1;
        result.is_delta = true;
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
