#pragma once

#include <btrc/utils/cmath/cmath.h>

BTRC_BEGIN

CUJ_CLASS_BEGIN(SurfacePoint)
    CUJ_MEMBER_VARIABLE(CVec3f, position)
    CUJ_MEMBER_VARIABLE(CFrame, frame)
    CUJ_MEMBER_VARIABLE(CVec3f, interp_z)
    CUJ_MEMBER_VARIABLE(CVec2f, uv)
    CUJ_MEMBER_VARIABLE(CVec2f, tex_coord)
CUJ_CLASS_END

inline void apply(const CTransform3D &transform, ref<SurfacePoint> spt)
{
    spt.position = transform.apply_to_point(spt.position);
    spt.frame.x = normalize(transform.apply_to_vector(spt.frame.x));
    spt.frame.y = normalize(transform.apply_to_vector(spt.frame.y));
    spt.frame.z = normalize(transform.apply_to_normal(spt.frame.z));
    spt.interp_z = normalize(transform.apply_to_normal(spt.interp_z));
}

BTRC_END
