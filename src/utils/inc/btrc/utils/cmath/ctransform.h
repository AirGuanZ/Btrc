#pragma once

#include <btrc/utils/cmath/cquaterion.h>
#include <btrc/utils/math/transform.h>

BTRC_BEGIN

CUJ_PROXY_CLASS_EX(CTransform, Transform, translate, scale, rotate)
{
    CUJ_BASE_CONSTRUCTORS

    CTransform(f32 scale, const CQuaterion &rotate, const CVec3f &translate);

    CTransform(const Transform &t);

    CVec3f apply_to_point(const CVec3f &p) const;

    CVec3f apply_to_vector(const CVec3f &v) const;
};

BTRC_END
