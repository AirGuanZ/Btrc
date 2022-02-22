#pragma once

#include <btrc/core/material.h>
#include <btrc/core/texture2d.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class Metal : public Material
{
public:

    void set_r0(RC<Texture2D> R0);

    void set_roughness(RC<Texture2D> roughness);

    void set_anisotropic(RC<Texture2D> anisoropic);

    RC<Shader> create_shader(CompileContext &cc, const SurfacePoint &inct) const override;

private:

    BTRC_OBJECT(Texture2D, R0_);
    BTRC_OBJECT(Texture2D, roughness_);
    BTRC_OBJECT(Texture2D, anisotropic_);
};

class MetalCreator : public factory::Creator<Material>
{
public:

    std::string get_name() const override { return "metal"; }

    RC<Material> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
