#pragma once

#include <array>

#include <btrc/utils/math/quaterion.h>

BTRC_BEGIN

struct Transform
{
    Vec3f     translate;
    float     scale   = 1;
    Quaterion rotate  = Quaterion(Vec3f(1, 0, 0), 0);

    std::array<float, 12> to_transform_matrix() const;

    Transform inverse() const;
};

// ========================== impl ==========================

inline std::array<float, 12> Transform::to_transform_matrix() const
{
    std::array<std::array<float, 3>, 3> rotation;
    rotate.to_rotation_matrix(&rotation[0][0]);
    const std::array result = {
        scale * rotation[0][0], scale * rotation[0][1], scale * rotation[0][2], translate.x,
        scale * rotation[1][0], scale * rotation[1][1], scale * rotation[1][2], translate.y,
        scale * rotation[2][0], scale * rotation[2][1], scale * rotation[2][2], translate.z
    };
    return result;
}

inline Transform Transform::inverse() const
{
    std::array<std::array<float, 3>, 3> rotation;
    rotate.to_rotation_matrix(&rotation[0][0]);
    return Transform{
        .translate = Vec3f(
            translate.x * rotation[0][0] + translate.y * rotation[0][1] + translate.z * rotation[0][2],
            translate.x * rotation[1][0] + translate.y * rotation[1][1] + translate.z * rotation[1][2],
            translate.x * rotation[2][0] + translate.y * rotation[2][1] + translate.z * rotation[2][2]),
        .scale = 1.0f / scale,
        .rotate = conjugate(rotate)
    };
}

BTRC_END
