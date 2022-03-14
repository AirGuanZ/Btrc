#include <btrc/utils/math/transform.h>

BTRC_BEGIN

std::array<float, 12> Transform::to_transform_matrix() const
{
    std::array<std::array<float, 3>, 3> rotation;
    rotate.to_rotation_matrix(&rotation[0][0]);
    const std::array result = {
        scale.x * rotation[0][0], scale.y * rotation[0][1], scale.z * rotation[0][2], translate.x,
        scale.x * rotation[1][0], scale.y * rotation[1][1], scale.z * rotation[1][2], translate.y,
        scale.x * rotation[2][0], scale.y * rotation[2][1], scale.z * rotation[2][2], translate.z
    };
    return result;
}

Transform Transform::inverse() const
{
    std::array<std::array<float, 3>, 3> rotation;
    rotate.to_rotation_matrix(&rotation[0][0]);

    const Mat4 m(
        scale.x * rotation[0][0], scale.y * rotation[0][1], scale.z * rotation[0][2], translate.x,
        scale.x * rotation[1][0], scale.y * rotation[1][1], scale.z * rotation[1][2], translate.y,
        scale.x * rotation[2][0], scale.y * rotation[2][1], scale.z * rotation[2][2], translate.z,
        0, 0, 0, 1);
    const Mat4 invm = m.inverse();

    const float a = invm.data[0][0], b = invm.data[1][0], c = invm.data[2][0], d = invm.data[3][0];
    const float e = invm.data[0][1], f = invm.data[1][1], g = invm.data[2][1], h = invm.data[3][1];
    const float i = invm.data[0][2], j = invm.data[1][2], k = invm.data[2][2], l = invm.data[3][2];

    const Vec3f inv_translate = Vec3f(d, h, l);

    const Vec3f inv_scale = Vec3f(
        length(Vec3f(a, e, i)),
        length(Vec3f(b, f, j)),
        length(Vec3f(c, g, k)));

    const Quaterion inv_rotate = conjugate(rotate);

    return Transform{
        .translate = inv_translate,
        .scale = inv_scale,
        .rotate = inv_rotate
    };
}

Vec3f Transform::apply_to_point(const Vec3f &p) const
{
    return rotate.apply_to_vector(scale * p) + translate;
}

BTRC_END
