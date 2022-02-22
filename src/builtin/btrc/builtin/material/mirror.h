#pragma once

#include <btrc/core/material.h>
#include <btrc/core/texture2d.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class Mirror : public Material
{
public:

    void set_color(RC<Texture2D> color);

    RC<Shader> create_shader(CompileContext &cc, const SurfacePoint &inct) const override;

private:

    BTRC_OBJECT(Texture2D, color_);
};

class MirrorCreator : public factory::Creator<Material>
{
public:

    std::string get_name() const override { return "mirror"; }

    RC<Material> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
