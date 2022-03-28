#include <btrc/builtin/material/glass.h>
#include <btrc/builtin/material/utils/fresnel.h>
#include <btrc/builtin/material/utils/shader_closure.h>
#include <btrc/builtin/material/utils/shader_frame.h>

BTRC_BUILTIN_BEGIN

namespace
{

    boolean refr(ref<CVec3f> nwo, ref<CVec3f> nor, f32 eta, ref<CVec3f> output)
    {
        boolean result;
        f32 cos_theta_i = cstd::abs(nwo.z);
        f32 sin_theta_i_2 = (cstd::max)(f32(0), 1.0f - cos_theta_i * cos_theta_i);
        f32 sin_theta_t_2 = eta * eta * sin_theta_i_2;
        $if(sin_theta_t_2 >= 1)
        {
            result = false;
        }
        $else
        {
            f32 cosThetaT = cstd::sqrt(1.0f - sin_theta_t_2);
            output = normalize((eta * cos_theta_i - cosThetaT) * nor - eta * nwo);
            result = true;
        };
        return result;
    }

} // namespace anonymous

CUJ_CLASS_BEGIN(GlassShaderImpl)

    CUJ_MEMBER_VARIABLE(ShaderFrame, raw_frame)
    CUJ_MEMBER_VARIABLE(CSpectrum,   color)
    CUJ_MEMBER_VARIABLE(f32,         ior)

    Shader::SampleResult sample(ref<CVec3f> wo, ref<CVec3f> sam, TransportMode mode) const
    {
        $declare_scope;
        Shader::SampleResult result;

        var frame = raw_frame.flip_for_black_fringes(wo);
        var nwo = normalize(frame.shading.global_to_local(wo));
        var fr = dielectric_fresnel(ior, 1, nwo.z);

        $if(sam.x < fr)
        {
            var lwi = CVec3f(-nwo.x, -nwo.y, nwo.z);
            var wi = frame.shading.local_to_global(lwi);
            $if(frame.is_black_fringes(wi))
            {
                result.clear();
                $exit_scope;
            };
            var bsdf = color * fr / cstd::abs(lwi.z);
            var norm = frame.correct_shading_energy(wi);
            result.dir = wi;
            result.bsdf = bsdf * norm;
            result.pdf = fr;
            result.is_delta = true;
            $exit_scope;
        };

        var nor = CVec3f(0, 0, cstd::select(nwo.z > 0, f32(1), f32(-1)));
        var eta = cstd::select(nwo.z > 0, 1.0f / ior, f32(ior));
        CVec3f nwi;
        $if(!refr(nwo, nor, eta, nwi))
        {
            result.clear();
            $exit_scope;
        };
        var wi = frame.shading.local_to_global(nwi);
        $if(frame.is_black_fringes(wi))
        {
            result.clear();
            $exit_scope;
        };
        var corr = mode == TransportMode::Radiance ? eta * eta : f32(1);
        var f = corr * color * (1.0f - fr) / cstd::abs(nwi.z);
        var pdf = 1.0f - fr;
        var norm = frame.correct_shading_energy(wi);
        result.bsdf = f * norm;
        result.dir = wi;
        result.pdf = pdf;
        result.is_delta = true;
        return result;
    }

    CSpectrum eval(ref<CVec3f> wi, ref<CVec3f> wo, TransportMode mode) const
    {
        return CSpectrum::zero();
    }

    f32 pdf(ref<CVec3f> wi, ref<CVec3f> wo, TransportMode mode) const
    {
        return 0;
    }

    CSpectrum albedo() const
    {
        return color;
    }

    CVec3f normal() const
    {
        return raw_frame.shading.z;
    }

CUJ_CLASS_END

void Glass::set_color(RC<Texture2D> color)
{
    color_ = std::move(color);
}

void Glass::set_ior(RC<Texture2D> ior)
{
    ior_ = std::move(ior);
}

void Glass::set_normal(RC<NormalMap> normal)
{
    normal_ = std::move(normal);
}

RC<Shader> Glass::create_shader(CompileContext &cc, const SurfacePoint &inct) const
{
    GlassShaderImpl impl;
    impl.raw_frame.geometry = inct.frame;
    impl.raw_frame.shading = inct.frame.rotate_to_new_z(inct.interp_z);
    impl.raw_frame.shading = normal_->adjust_frame(cc, inct, impl.raw_frame.shading);
    impl.color = color_->sample_spectrum(cc, inct);
    impl.ior = ior_->sample_float(cc, inct);
    return newRC<ShaderClosure<GlassShaderImpl>>(as_shared(), impl);
}

RC<Material> GlassCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    auto color = context.create<Texture2D>(node->child_node("color"));
    auto ior = context.create<Texture2D>(node->child_node("ior"));
    auto normal = newRC<NormalMap>();
    normal->load(node, context);
    auto glass = newRC<Glass>();
    glass->set_color(std::move(color));
    glass->set_ior(std::move(ior));
    glass->set_normal(std::move(normal));
    return glass;
}

BTRC_BUILTIN_END
