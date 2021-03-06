#pragma once

#include <btrc/utils/cmath/cmat3.h>
#include <btrc/utils/cmath/cmat4.h>
#include <btrc/utils/cmath/cvec2.h>
#include <btrc/utils/cmath/cvec3.h>
#include <btrc/utils/math/transform.h>

BTRC_BEGIN

CUJ_PROXY_CLASS_EX(CTransform2D, Transform2D, mat, inv)
{
    CUJ_BASE_CONSTRUCTORS

    CTransform2D(const Transform2D &t);

    CTransform2D(const CMat3 &mat, const CMat3 &inv);

    CVec2f apply_to_point(const CVec2f &p) const;
};

CUJ_PROXY_CLASS_EX(CTransform3D, Transform3D, mat, inv)
{
    CUJ_BASE_CONSTRUCTORS

    CTransform3D(const Transform3D &t);

    CTransform3D(const CMat4 &mat, const CMat4 &inv);

    CVec3f apply_to_point(const CVec3f &p) const;

    CVec3f apply_to_vector(const CVec3f &v) const;

    CVec3f apply_to_normal(const CVec3f &n) const;
};

BTRC_END
