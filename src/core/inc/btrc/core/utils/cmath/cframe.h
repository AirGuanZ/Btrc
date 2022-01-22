#pragma once

#include <btrc/core/utils/cmath/cvec3.h>

BTRC_CORE_BEGIN

class CFrame
{
public:

    CVec3f x, y, z;

    CFrame();

    CFrame(const CVec3f &x, const CVec3f &y, const CVec3f &z);

    static CFrame from_x(const CVec3f &x);

    static CFrame from_y(const CVec3f &y);

    static CFrame from_z(const CVec3f &z);

    CVec3f local_to_global(const CVec3f &local) const;

    CVec3f global_to_local(const CVec3f &global) const;

    CFrame rotate_to_new_z(const CVec3f &new_z) const;

    boolean in_positive_z_hemisphere(const CVec3f &v) const;
};

BTRC_CORE_END
