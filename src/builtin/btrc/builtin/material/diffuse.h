#pragma once

#include <btrc/core/material.h>
#include <btrc/core/texture2d.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class Diffuse : public Material
{
public:

    void set_albedo(RC<Texture2D> albedo);

    std::vector<RC<Object>> get_dependent_objects() override;

    RC<Shader> create_shader(CompileContext &cc, const SurfacePoint &inct) const override;

private:

    RC<Texture2D> albedo_;
};

class DiffuseCreator : public factory::Creator<Material>
{
public:

    std::string get_name() const override { return "diffuse"; }

    RC<Material> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
