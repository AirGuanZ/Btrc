#pragma once

#include <array>

#include <btrc/utils/math/mat4.h>
#include <btrc/utils/math/quaterion.h>

BTRC_BEGIN

struct Transform
{
    Vec3f     translate;
    Vec3f     scale  = Vec3f(1);
    Quaterion rotate = Quaterion(Vec3f(1, 0, 0), 0);

    std::array<float, 12> to_transform_matrix() const;

    Transform inverse() const;

    Vec3f apply_to_point(const Vec3f &p) const;
};

BTRC_END
