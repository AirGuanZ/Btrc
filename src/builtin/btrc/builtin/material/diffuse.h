#pragma once

#include <btrc/builtin/material/utils/normal_map.h>
#include <btrc/core/material.h>
#include <btrc/core/texture2d.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class Diffuse : public Material
{
public:

    void set_shadow_terminator_term(bool enable);

    void set_albedo(RC<Texture2D> albedo);

    void set_normal(RC<NormalMap> normal);

    RC<Shader> create_shader(CompileContext &cc, const SurfacePoint &inct) const override;

private:

    bool shadow_terminator_term_ = true;
    BTRC_OBJECT(Texture2D, albedo_);
    BTRC_OBJECT(NormalMap, normal_);
};

class DiffuseCreator : public factory::Creator<Material>
{
public:

    std::string get_name() const override { return "diffuse"; }

    RC<Material> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
