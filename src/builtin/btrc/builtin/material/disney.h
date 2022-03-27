#pragma once

#include <btrc/builtin/material/utils/normal_map.h>
#include <btrc/core/material.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class DisneyMaterial : public Material
{
public:

    void set_shadow_terminator_term(bool enable);

    void set_base_color(RC<Texture2D> tex);

    void set_metallic(RC<Texture2D> tex);

    void set_roughness(RC<Texture2D> tex);

    void set_specular(RC<Texture2D> tex);

    void set_specular_tint(RC<Texture2D> tex);

    void set_anisotropic(RC<Texture2D> tex);

    void set_sheen(RC<Texture2D> tex);

    void set_sheen_tint(RC<Texture2D> tex);

    void set_clearcoat(RC<Texture2D> tex);

    void set_clearcoat_gloss(RC<Texture2D> tex);

    void set_transmission(RC<Texture2D> tex);

    void set_transmission_roughness(RC<Texture2D> tex);

    void set_ior(RC<Texture2D> tex);

    void set_normal(RC<NormalMap> normal);

    RC<Shader> create_shader(CompileContext &cc, const SurfacePoint &inct) const override;

private:

    bool shadow_terminator_term_ = true;
    BTRC_OBJECT(Texture2D, base_color_);
    BTRC_OBJECT(Texture2D, metallic_);
    BTRC_OBJECT(Texture2D, roughness_);
    BTRC_OBJECT(Texture2D, specular_);
    BTRC_OBJECT(Texture2D, specular_tint_);
    BTRC_OBJECT(Texture2D, anisotropic_);
    BTRC_OBJECT(Texture2D, sheen_);
    BTRC_OBJECT(Texture2D, sheen_tint_);
    BTRC_OBJECT(Texture2D, clearcoat_);
    BTRC_OBJECT(Texture2D, clearcoat_gloss_);
    BTRC_OBJECT(Texture2D, transmission_);
    BTRC_OBJECT(Texture2D, transmission_roughness_);
    BTRC_OBJECT(Texture2D, ior_);
    BTRC_OBJECT(NormalMap, normal_);
};

class DisneyMaterialCreator : public factory::Creator<Material>
{
public:

    std::string get_name() const override { return "disney"; }

    RC<Material> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
