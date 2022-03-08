#pragma once

#include <btrc/utils/math/vec4.h>

BTRC_BEGIN

struct Mat4
{
    Vec4f data[4];

    Mat4() = default;

    Mat4(
        float m00, float m01, float m02, float m03,
        float m10, float m11, float m12, float m13,
        float m20, float m21, float m22, float m23,
        float m30, float m31, float m32, float m33);

    Mat4(
        const Vec4f &c0,
        const Vec4f &c1,
        const Vec4f &c2,
        const Vec4f &c3);

    Mat4 inverse() const;
};

BTRC_END
