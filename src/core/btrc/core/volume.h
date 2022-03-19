#pragma once

#include <btrc/core/texture3d.h>
#include <btrc/utils/math/aabb.h>

BTRC_BEGIN

class VolumePrimitive : public Object
{
public:

    struct VolumeGeometryInfo
    {
        Vec3f o, x, y, z;
    };

    void set_geometry(const Vec3f &o, const Vec3f &x, const Vec3f &y, const Vec3f &z);

    void set_sigma_t(RC<Texture3D> sigma_t);

    void set_albedo(RC<Texture3D> albedo);

    VolumeGeometryInfo get_geometry_info() const;

    CVec3f world_pos_to_uvw(ref<CVec3f> p) const;

    AABB3f get_bounding_box() const;

    RC<Texture3D> get_sigma_t() const;

    RC<Texture3D> get_albedo() const;

private:

    Vec3f o_;
    Vec3f x_;
    Vec3f y_;
    Vec3f z_;

    BTRC_OBJECT(Texture3D, sigma_t_);
    BTRC_OBJECT(Texture3D, albedo_);
};

BTRC_END
