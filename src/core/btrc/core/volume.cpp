#include <btrc/core/volume.h>

BTRC_BEGIN

namespace
{

    float min(float a, float b, float c, float d)
    {
        return std::min(std::min(a, b), std::min(c, d));
    }

    float max(float a, float b, float c, float d)
    {
        return std::max(std::max(a, b), std::max(c, d));
    }

} // namespace anonymous

void VolumePrimitive::set_geometry(const Vec3f &o, const Vec3f &x, const Vec3f &y, const Vec3f &z)
{
    o_ = o;
    x_ = x;
    y_ = y;
    z_ = z;
}

void VolumePrimitive::set_sigma_t(RC<Texture3D> sigma_t)
{
    sigma_t_ = std::move(sigma_t);
}

void VolumePrimitive::set_albedo(RC<Texture3D> albedo)
{
    albedo_ = std::move(albedo);
}

VolumePrimitive::VolumeGeometryInfo VolumePrimitive::get_geometry_info() const
{
    return { o_, x_, y_, z_ };
}

CVec3f VolumePrimitive::world_pos_to_uvw(ref<CVec3f> p) const
{
    var op = p - o_;
    var u = dot(op, x_) * (1.0f / length_square(x_));
    var v = dot(op, y_) * (1.0f / length_square(y_));
    var w = dot(op, z_) * (1.0f / length_square(z_));
    return CVec3f(u, v, w);
}

AABB3f VolumePrimitive::get_bounding_box() const
{
    const Vec3f o = o_, x = x_, y = y_, z = z_;
    const Vec3f lower = {
        min(o.x, o.x + x.x, o.x + y.x, o.x + z.x),
        min(o.y, o.y + x.y, o.y + y.y, o.y + z.y),
        min(o.y, o.z + x.z, o.z + y.z, o.z + z.z),
    };
    const Vec3f upper = {
        max(o.x, o.x + x.x, o.x + y.x, o.x + z.x),
        max(o.y, o.y + x.y, o.y + y.y, o.y + z.y),
        max(o.y, o.z + x.z, o.z + y.z, o.z + z.z),
    };
    return { lower, upper };
}

RC<Texture3D> VolumePrimitive::get_sigma_t() const
{
    return sigma_t_.get();
}

RC<Texture3D> VolumePrimitive::get_albedo() const
{
    return albedo_.get();
}

BTRC_END
