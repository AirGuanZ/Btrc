#pragma once

#include <btrc/core/material.h>
#include <btrc/core/texture2d.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class Metal : public Material
{
public:

    void set_r0(RC<const Texture2D> R0);

    void set_roughness(RC<const Texture2D> roughness);

    void set_anisotropic(RC<const Texture2D> anisoropic);

    RC<Shader> create_shader(CompileContext &cc, const SurfacePoint &inct) const override;

private:

    RC<const Texture2D> R0_;
    RC<const Texture2D> roughness_;
    RC<const Texture2D> anisotropic_;
};

class MetalCreator : public factory::Creator<Material>
{
public:

    std::string get_name() const override { return "metal"; }

    RC<Material> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
