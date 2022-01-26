#include <btrc/core/utils/cmath/cframe.h>

BTRC_CORE_BEGIN

CFrame::CFrame()
    : CFrame(CVec3f(1, 0, 0), CVec3f(0, 1, 0), CVec3f(0, 0, 1))
{
    
}

CFrame::CFrame(const CVec3f &_x, const CVec3f &_y, const CVec3f &_z)
{
    x = _x;
    y = _y;
    z = _z;
}

CFrame CFrame::from_x(const CVec3f &_x)
{
    CVec3f new_x = normalize(_x), new_z;
    $if(cstd::abs(cstd::abs(new_x.y) - 1) < 0.1f)
    {
        new_z = cross(new_x, CVec3f(0, 0, 1));
    }
    $else
    {
        new_z = cross(new_x, CVec3f(0, 1, 0));
    };
    return CFrame(new_x, cross(new_z, new_x), new_z);
}

CFrame CFrame::from_y(const CVec3f &_y)
{
    CVec3f new_y = normalize(_y), new_x;
    $if(cstd::abs(cstd::abs(new_y.z) - 1) < 0.1f)
    {
        new_x = cross(new_y, CVec3f(1, 0, 0));
    }
    $else
    {
        new_x = cross(new_y, CVec3f(0, 0, 1));
    };
    return CFrame(new_x, new_y, cross(new_x, new_y));
}

CFrame CFrame::from_z(const CVec3f &_z)
{
    CVec3f new_z = normalize(_z), new_y;
    $if(cstd::abs(cstd::abs(new_z.x) - 1) < 0.1f)
    {
        new_y = cross(new_z, CVec3f(0, 1, 0));
    }
    $else
    {
        new_y = cross(new_z, CVec3f(1, 0, 0));
    };
    return CFrame(cross(new_z, new_y), new_y, new_z);
}

CVec3f CFrame::local_to_global(const CVec3f &local) const
{
    return x * local.x + y * local.y + z * local.z;
}

CVec3f CFrame::global_to_local(const CVec3f &global) const
{
    return CVec3f(dot(global, x), dot(global, y), dot(global, z));
}

CFrame CFrame::rotate_to_new_z(const CVec3f &new_z) const
{
    var new_x = cross(y, new_z);
    var new_y = cross(new_z, new_x);
    return CFrame(new_x, new_y, new_z);
}

boolean CFrame::in_positive_z_hemisphere(const CVec3f &v) const
{
    return dot(v, z) > 0;
}

BTRC_CORE_END
