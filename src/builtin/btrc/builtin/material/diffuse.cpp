#include <btrc/builtin/material/component/diffuse.h>
#include <btrc/builtin/material/diffuse.h>

BTRC_BUILTIN_BEGIN

void Diffuse::set_albedo(RC<Texture2D> albedo)
{
    albedo_ = std::move(albedo);
}

void Diffuse::set_normal(RC<NormalMap> normal)
{
    normal_ = std::move(normal);
}

RC<Shader> Diffuse::create_shader(CompileContext &cc, const SurfacePoint &inct) const
{
    ShaderFrame frame;
    frame.geometry = inct.frame;
    frame.shading = inct.frame.rotate_to_new_z(inct.interp_z);
    frame.shading = normal_->adjust_frame(cc, inct, frame.shading);

    DiffuseComponentImpl diffuse_closure;
    diffuse_closure.albedo_value = albedo_->sample_spectrum(cc, inct);

    auto shader = newRC<BSDFAggregate>(as_shared(), false, frame);
    shader->add_closure(1, "diffuse", diffuse_closure);
    return shader;
}

RC<Material> DiffuseCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    auto albedo = context.create<Texture2D>(node->child_node("color"));
    auto normal = newRC<NormalMap>();
    normal->load(node, context);
    auto ret = newRC<Diffuse>();
    ret->set_albedo(std::move(albedo));
    ret->set_normal(std::move(normal));
    return ret;
}

BTRC_BUILTIN_END
