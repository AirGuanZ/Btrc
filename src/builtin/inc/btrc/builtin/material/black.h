#pragma once

#include <btrc/core/material.h>

BTRC_BUILTIN_BEGIN

class Black : public Material
{
public:

    RC<Shader> create_shader(const SurfacePoint &inct) const override;
};

BTRC_BUILTIN_END
