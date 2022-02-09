#pragma once

#include <btrc/core/material/material.h>

BTRC_CORE_BEGIN

class Black : public Material
{
public:

    RC<Shader> create_shader(const CIntersection &inct) const override;
};

BTRC_CORE_END
