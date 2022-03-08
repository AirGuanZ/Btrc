#pragma once

#include <btrc/core/material.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class InvisibleSurface : public Material
{
public:

    RC<Shader> create_shader(CompileContext &cc, const SurfacePoint &inct) const override;
};

class InvisibleSurfaceCreator : public factory::Creator<Material>
{
public:

    std::string get_name() const override { return "invisible"; }

    RC<Material> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
