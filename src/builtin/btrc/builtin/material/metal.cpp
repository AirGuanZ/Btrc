#include <btrc/builtin/material/component/microfacet_reflection.h>
#include <btrc/builtin/material/metal.h>

BTRC_BUILTIN_BEGIN

void Metal::set_shadow_terminator_term(bool enable)
{
    shadow_terminator_term_ = enable;
}

void Metal::set_r0(RC<Texture2D> R0)
{
    R0_ = std::move(R0);
}

void Metal::set_roughness(RC<Texture2D> roughness)
{
    roughness_ = std::move(roughness);
}

void Metal::set_anisotropic(RC<Texture2D> anisoropic)
{
    anisotropic_ = std::move(anisoropic);
}

void Metal::set_normal(RC<NormalMap> normal)
{
    normal_ = std::move(normal);
}

RC<Shader> Metal::create_shader(CompileContext &cc, const SurfacePoint &inct) const
{
    ShaderFrame frame;
    frame.geometry = inct.frame;
    frame.shading = inct.frame.rotate_to_new_z(inct.interp_z);
    frame.shading = normal_->adjust_frame(cc, inct, frame.shading);

    ConductorFresnelPoint fresnel;
    fresnel.R0 = R0_->sample_spectrum(cc, inct);

    var roughness = roughness_->sample_float(cc, inct);
    var anisotropic = anisotropic_->sample_float(cc, inct);
    MicrofacetReflectionComponentImpl closure(fresnel, roughness, anisotropic);

    auto shader = newRC<BSDFAggregate>(cc, as_shared(), frame, shadow_terminator_term_);
    shader->add_closure(1, "specular", closure);
    return shader;
}

RC<Material> MetalCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    const bool shadow_terminator_term = node->parse_child_or("shadow_terminator_term", true);

    auto r0 = context.create<Texture2D>(node->child_node("color"));
    auto roughness = context.create<Texture2D>(node->child_node("roughness"));

    RC<Texture2D> anisotropic;
    if(auto n = node->find_child_node("anisotropic"))
        anisotropic = context.create<Texture2D>(n);
    else
    {
        auto c = newRC<Constant2D>();
        c->set_value(0);
        anisotropic = std::move(c);
    }

    auto normal = newRC<NormalMap>();
    normal->load(node, context);

    auto result = newRC<Metal>();
    result->set_shadow_terminator_term(shadow_terminator_term);
    result->set_r0(std::move(r0));
    result->set_roughness(std::move(roughness));
    result->set_anisotropic(std::move(anisotropic));
    result->set_normal(std::move(normal));
    return result;
}

BTRC_BUILTIN_END
