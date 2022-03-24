#include <btrc/builtin/material/mirror.h>
#include <btrc/builtin/material/utils/fresnel.h>
#include <btrc/builtin/material/utils/shader_closure.h>
#include <btrc/builtin/material/utils/shader_frame.h>

BTRC_BUILTIN_BEGIN

CUJ_CLASS_BEGIN(MirrorShaderImpl)

    CUJ_MEMBER_VARIABLE(ShaderFrame, frame)
    CUJ_MEMBER_VARIABLE(CSpectrum,   color)

    Shader::SampleResult sample(ref<CVec3f> wo, ref<CVec3f> sam, TransportMode mode) const
    {
        $declare_scope;
        Shader::SampleResult result;

        $if(frame.is_black_fringes(wo))
        {
            result = frame.sample_black_fringes(wo, sam, color);
            $exit_scope;
        };

        var lwo = frame.shading.global_to_local(wo);
        $if(lwo.z <= 0)
        {
            result.clear();
            $exit_scope;
        };
        lwo = normalize(lwo);

        var lwi = CVec3f(-lwo.x, -lwo.y, lwo.z);
        var wi = frame.shading.local_to_global(lwi);
        var fr = schlick_approx(color, lwo.z);
        var bsdf = fr / lwo.z;
        var norm_factor = frame.correct_shading_energy(wi);

        result.bsdf = bsdf * norm_factor;
        result.dir = wi;
        result.pdf = 1;
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
        return frame.shading.z;
    }

    boolean is_delta() const
    {
        return true;
    }

CUJ_CLASS_END

void Mirror::set_color(RC<Texture2D> color)
{
    color_ = std::move(color);
}

void Mirror::set_normal(RC<NormalMap> normal)
{
    normal_ = std::move(normal);
}

RC<Shader> Mirror::create_shader(CompileContext &cc, const SurfacePoint &inct) const
{
    MirrorShaderImpl impl;
    impl.frame.geometry = inct.frame;
    impl.frame.shading = inct.frame.rotate_to_new_z(inct.interp_z);
    impl.frame.shading = normal_->adjust_frame(cc, inct, impl.frame.shading);
    impl.color = color_->sample_spectrum(cc, inct);
    return newRC<ShaderClosure<MirrorShaderImpl>>(as_shared(), impl);
}

RC<Material> MirrorCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    auto color = context.create<Texture2D>(node->child_node("color"));
    auto normal = newRC<NormalMap>();
    normal->load(node, context);
    auto mirror = newRC<Mirror>();
    mirror->set_color(std::move(color));
    mirror->set_normal(std::move(normal));
    return mirror;
}

BTRC_BUILTIN_END
