#include <btrc/utils/math/frame.h>

BTRC_BEGIN

Frame::Frame()
    : Frame(Vec3f(1, 0, 0), Vec3f(0, 1, 0), Vec3f(0, 0, 1))
{
    
}

Frame::Frame(const Vec3f &x, const Vec3f &y, const Vec3f &z)
    : x(normalize(x)), y(normalize(y)), z(normalize(z))
{
    
}

Frame Frame::from_x(const Vec3f &x)
{
    Vec3f new_x = normalize(x), new_z;
    if(1 - std::abs(new_x.y) < 0.1f)
    {
        new_z = cross(new_x, Vec3f(0, 0, 1));
    }
    else
    {
        new_z = cross(new_x, Vec3f(0, 1, 0));
    }
    return Frame(new_x, cross(new_z, new_x), new_z);
}

Frame Frame::from_y(const Vec3f &y)
{
    Vec3f new_y = normalize(y), new_x;
    if(1 - std::abs(new_y.z) < 0.1f)
    {
        new_x = cross(new_y, Vec3f(1, 0, 0));
    }
    else
    {
        new_x = cross(new_y, Vec3f(0, 0, 1));
    }
    return Frame(new_x, new_y, cross(new_x, new_y));
}

Frame Frame::from_z(const Vec3f &z)
{
    Vec3f new_z = normalize(z), new_y;
    if(1 - std::abs(new_z.x) < 0.1f)
    {
        new_y = cross(new_z, Vec3f(0, 1, 0));
    }
    else
    {
        new_y = cross(new_z, Vec3f(1, 0, 0));
    }
    return Frame(cross(new_z, new_y), new_y, new_z);
}

Vec3f Frame::local_to_global(const Vec3f &local) const
{
    return x * local.x + y * local.y + z * local.z;
}

Vec3f Frame::global_to_local(const Vec3f &global) const
{
    return Vec3f(dot(global, x), dot(global, y), dot(global, z));
}

Frame Frame::rotate_to_new_z(const Vec3f &new_z) const
{
    const Vec3f new_x = cross(y, new_z);
    const Vec3f new_y = cross(new_z, new_x);
    return Frame(new_x, new_y, new_z);
}

bool Frame::in_positive_z_hemisphere(const Vec3f &v) const
{
    return dot(v, z) > 0;
}

BTRC_END
