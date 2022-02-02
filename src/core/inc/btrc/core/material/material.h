#pragma once

#include <btrc/core/compile/object.h>
#include <btrc/core/material/shader.h>

BTRC_CORE_BEGIN

CUJ_CLASS_BEGIN(CIntersection)
    CUJ_MEMBER_VARIABLE(CVec3f, position)
    CUJ_MEMBER_VARIABLE(CFrame, frame)
    CUJ_MEMBER_VARIABLE(CVec3f, interp_normal)
    CUJ_MEMBER_VARIABLE(CVec2f, uv)
    CUJ_MEMBER_VARIABLE(CVec2f, tex_coord)
CUJ_CLASS_END

class Material : public Object<Material>
{
public:

    virtual ~Material() = default;

    virtual RC<Shader> create_shader(const CIntersection &inct) const = 0;
};

BTRC_CORE_END
