#pragma once

#include <btrc/utils/math/vec3.h>

BTRC_BEGIN

class Frame
{
public:

    Vec3f x, y, z;

    Frame();

    Frame(const Vec3f &x, const Vec3f &y, const Vec3f &z);

    static Frame from_x(const Vec3f &x);

    static Frame from_y(const Vec3f &y);

    static Frame from_z(const Vec3f &z);

    Vec3f local_to_global(const Vec3f &local) const;

    Vec3f global_to_local(const Vec3f &global) const;

    Frame rotate_to_new_z(const Vec3f &new_z) const;

    bool in_positive_z_hemisphere(const Vec3f &v) const;
};

BTRC_END
