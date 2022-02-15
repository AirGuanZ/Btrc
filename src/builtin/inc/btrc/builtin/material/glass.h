#pragma once

#include <btrc/core/material.h>
#include <btrc/core/texture2d.h>

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

BTRC_BUILTIN_END
