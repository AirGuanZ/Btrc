#include <btrc/builtin/light/ibl.h>
#include <btrc/utils/local_angle.h>

BTRC_BUILTIN_BEGIN

void IBL::set_texture(RC<Texture2D> tex)
{
    tex_ = std::move(tex);
}

void IBL::set_up(const Vec3f &up)
{
    frame_ = Frame::from_z(up);
}

void IBL::set_lut_res(const Vec2i &lut_res)
{
    lut_res_ = lut_res;
}

void IBL::commit()
{
    sampler_ = newBox<EnvirLightSampler>();
    sampler_->preprocess(tex_.get(), lut_res_, 256);
}

CSpectrum IBL::eval_le_inline(CompileContext &cc, ref<CVec3f> to_light) const
{
    var dir = normalize(CFrame(frame_).global_to_local(to_light));
    return eval_local(cc, dir);
}

EnvirLight::SampleLiResult IBL::sample_li_inline(CompileContext &cc, ref<Sam3> sam) const
{
    var local_sample = sampler_->sample(sam);
    var local_dir = local_sample.to_light;
    var global_dir = CFrame(frame_).local_to_global(local_dir);
    SampleLiResult result;
    result.direction_to_light = global_dir;
    result.radiance = eval_local(cc, local_dir);
    result.pdf = local_sample.pdf;
    return result;
}

f32 IBL::pdf_li_inline(CompileContext &cc, ref<CVec3f> to_light) const
{
    var dir = CFrame(frame_).global_to_local(to_light);
    return sampler_->pdf(dir);
}

EnvirLight::SampleEmitResult IBL::sample_emit_inline(CompileContext &cc, ref<Sam<3>> sam) const
{
    var local_sample = sampler_->sample(sam);
    var local_dir = local_sample.to_light;
    var global_dir = CFrame(frame_).local_to_global(local_dir);

    SampleEmitResult result;
    result.direction = -global_dir;
    result.radiance = eval_local(cc, local_dir);
    result.pdf_dir = local_sample.pdf;

    return result;
}

f32 IBL::pdf_emit_inline(CompileContext &cc, ref<CVec3f> dir) const
{
    var local_dir = CFrame(frame_).global_to_local(dir);
    return sampler_->pdf(-local_dir);
}

CSpectrum IBL::eval_local(CompileContext &cc, ref<CVec3f> normalized_to_light) const
{
    var phi = local_angle::phi(normalized_to_light);
    var theta = local_angle::theta(normalized_to_light);
    var u = cstd::saturate(1 / (2 * btrc_pi) * phi);
    var v = cstd::saturate(1 / btrc_pi * theta);
    return tex_->sample_spectrum(cc, CVec2f(u, v));
}

RC<Light> IBLCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    auto tex = context.create<Texture2D>(node->child_node("tex"));
    auto up = node->parse_child_or("up", Vec3f(0, 0, 1));
    const int lut_res_x = node->parse_child_or("lut_width", 400);
    const int lut_res_y = node->parse_child_or("lut_height", 200);
    auto result = newRC<IBL>();
    result->set_texture(std::move(tex));
    result->set_up(up);
    result->set_lut_res({ lut_res_x, lut_res_y });
    return result;
}

BTRC_BUILTIN_END
