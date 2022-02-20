#pragma once

#include <btrc/core/material.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class Black : public Material
{
public:

    RC<Shader> create_shader(const SurfacePoint &inct) const override;
};

class BlackCreator : public factory::Creator<Material>
{
public:

    std::string get_name() const override { return "black"; }

    RC<Material> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
