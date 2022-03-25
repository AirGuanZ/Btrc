#pragma once

#include <btrc/utils/math/vec4.h>

BTRC_BEGIN

struct Mat4
{
    Vec4f data[4];

    Mat4();

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

    Mat4 transpose() const;

    float at(int r, int c) const;

    float &at(int r, int c);

    static Mat4 translate(float x, float y, float z);

    static Mat4 rotate(const Vec3f &axis, float rad);

    static Mat4 rotate_x(float rad);

    static Mat4 rotate_y(float rad);

    static Mat4 rotate_z(float rad);

    static Mat4 scale(float x, float y, float z);
};

Vec4f operator*(const Mat4 &m, const Vec4f &v);

Mat4 operator*(const Mat4 &a, const Mat4 &b);

BTRC_END
