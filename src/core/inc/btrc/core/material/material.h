#pragma once

#include <btrc/core/common/context.h>
#include <btrc/core/material/shader.h>
#include <btrc/core/geometry/geometry.h>

BTRC_CORE_BEGIN

class Material : public Object
{
public:

    virtual ~Material() = default;

    virtual RC<Shader> create_shader(const SurfacePoint &inct) const = 0;
};

BTRC_CORE_END
