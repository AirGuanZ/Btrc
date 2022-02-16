#pragma once

#include <btrc/core/material.h>
#include <btrc/core/texture2d.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class Diffuse : public Material
{
public:

    void set_albedo(RC<const Texture2D> albedo);
    
    RC<Shader> create_shader(const SurfacePoint &inct) const override;

private:

    RC<const Texture2D> albedo_;
};

class DiffuseCreator : public factory::Creator<Material>
{
public:

    std::string get_name() const override { return "diffuse"; }

    RC<Material> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
