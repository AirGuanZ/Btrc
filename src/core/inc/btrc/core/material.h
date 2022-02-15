#pragma once

#include <btrc/core/context.h>
#include <btrc/core/geometry.h>
#include <btrc/core/shader/shader.h>

BTRC_BEGIN

class Material : public Object
{
public:

    virtual ~Material() = default;

    virtual RC<Shader> create_shader(const SurfacePoint &inct) const = 0;
};

BTRC_END
