#pragma once

#include <btrc/utils/math/vec3.h>

BTRC_BEGIN

struct Mat3
{
    Vec3f data[3];

    Mat3();

    Mat3(
        float m00, float m01, float m02,
        float m10, float m11, float m12,
        float m20, float m21, float m22);

    Mat3(
        const Vec3f &c0,
        const Vec3f &c1,
        const Vec3f &c2);

    Mat3 inverse() const;

    Mat3 transpose() const;

    float at(int r, int c) const;

    float &at(int r, int c);

    static Mat3 translate(float x, float y);

    static Mat3 rotate(float rad);

    static Mat3 scale(float x, float y);
};

Mat3 operator*(const Mat3 &a, const Mat3 &b);

Mat3 operator/(const Mat3 &a, float b);

BTRC_END
