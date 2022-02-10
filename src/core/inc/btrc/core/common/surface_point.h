#pragma once

#include <btrc/core/utils/cmath/cmath.h>

BTRC_CORE_BEGIN

CUJ_CLASS_BEGIN(SurfacePoint)
    CUJ_MEMBER_VARIABLE(CVec3f, position)
    CUJ_MEMBER_VARIABLE(CFrame, frame)
    CUJ_MEMBER_VARIABLE(CVec3f, interp_z)
    CUJ_MEMBER_VARIABLE(CVec2f, uv)
    CUJ_MEMBER_VARIABLE(CVec2f, tex_coord)
CUJ_CLASS_END

BTRC_CORE_END
