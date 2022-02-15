#pragma once

#include <btrc/core/material.h>
#include <btrc/core/texture2d.h>

BTRC_BUILTIN_BEGIN

class Diffuse : public Material
{
public:

    void set_albedo(RC<const Texture2D> albedo);
    
    RC<Shader> create_shader(const SurfacePoint &inct) const override;

private:

    RC<const Texture2D> albedo_;
};

BTRC_BUILTIN_END
