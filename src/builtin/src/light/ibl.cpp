#include <btrc/builtin/light/ibl.h>
#include <btrc/utils/local_angle.h>

BTRC_BUILTIN_BEGIN

void IBL::set_texture(RC<const Texture2D> tex)
{
    tex_ = std::move(tex);
}

void IBL::set_up(const Vec3f &up)
{
    frame_ = Frame::from_z(up);
}

CSpectrum IBL::eval_le_inline(ref<CVec3f> to_light) const
{
    var frame = CFrame(frame_);
    var dir = normalize(frame.global_to_local(to_light));
    var phi = local_angle::phi(dir);
    var theta = local_angle::theta(dir);
    var u = cstd::saturate(1 / (2 * btrc_pi) * phi);
    var v = cstd::saturate(1 / btrc_pi * theta);
    return tex_->sample_spectrum(CVec2f(u, v));
}

EnvirLight::SampleLiResult IBL::sample_li_inline(ref<CVec3f> sam) const
{
    // TODO: importance sampling

    CFrame frame = CFrame(frame_);
    var local_dir = sample_sphere_uniform(sam.x, sam.y);
    var global_dir = frame.local_to_global(local_dir);
    SampleLiResult result;
    result.direction_to_light = global_dir;
    result.radiance = eval_le(global_dir);
    result.pdf = pdf_sample_sphere_uniform();
    return result;
}

f32 IBL::pdf_li_inline(ref<CVec3f> to_light) const
{
    return pdf_sample_sphere_uniform();
}

RC<Light> IBLCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    auto tex = context.create<Texture2D>(node->child_node("tex"));
    auto up = node->parse_child_or("up", Vec3f(0, 0, 1));
    auto result = newRC<IBL>();
    result->set_texture(std::move(tex));
    result->set_up(up);
    return result;
}

BTRC_BUILTIN_END
