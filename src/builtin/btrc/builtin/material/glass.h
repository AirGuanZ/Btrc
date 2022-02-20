#pragma once

#include <btrc/core/material.h>
#include <btrc/core/texture2d.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class Glass : public Material
{
public:

    void set_color(RC<const Texture2D> color);

    void set_ior(RC<const Texture2D> ior);

    RC<Shader> create_shader(const SurfacePoint &inct) const override;

private:

    RC<const Texture2D> color_;
    RC<const Texture2D> ior_;
};

class GlassCreator : public factory::Creator<Material>
{
public:

    std::string get_name() const override { return "glass"; }

    RC<Material> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
