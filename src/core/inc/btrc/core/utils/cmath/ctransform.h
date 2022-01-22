#pragma once

#include <btrc/core/utils/cmath/cquaterion.h>
#include <btrc/core/utils/cmath/cvec3.h>

BTRC_CORE_BEGIN

struct Transform
{
    float     scale;
    Quaterion rotate;
    Vec3f     translate;
};

CUJ_PROXY_CLASS_EX(CTransform, Transform, scale, rotate, translate)
{
    CUJ_BASE_CONSTRUCTORS

    CTransform(f32 scale, const CQuaterion &rotate, const CVec3f &translate);

    CVec3f apply_to_point(const CVec3f &p) const;

    CVec3f apply_to_vector(const CVec3f &v) const;
};

BTRC_CORE_END
