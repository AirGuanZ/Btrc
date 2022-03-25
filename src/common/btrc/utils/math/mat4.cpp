#include <btrc/utils/math/mat4.h>

BTRC_BEGIN

Mat4::Mat4()
    : Mat4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1)
{
    
}

Mat4::Mat4(
    float m00, float m01, float m02, float m03,
    float m10, float m11, float m12, float m13,
    float m20, float m21, float m22, float m23,
    float m30, float m31, float m32, float m33)
    : data{ { m00, m10, m20, m30 },
            { m01, m11, m21, m31 },
            { m02, m12, m22, m32 },
            { m03, m13, m23, m33 } }
{
    
}

Mat4::Mat4(const Vec4f &c0, const Vec4f &c1, const Vec4f &c2, const Vec4f &c3)
    : data{ c0, c1, c2, c3 }
{
    
}

Mat4 Mat4::inverse() const
{
    const float coef00 = data[2][2] * data[3][3] - data[3][2] * data[2][3];
    const float coef02 = data[1][2] * data[3][3] - data[3][2] * data[1][3];
    const float coef03 = data[1][2] * data[2][3] - data[2][2] * data[1][3];
    const float coef04 = data[2][1] * data[3][3] - data[3][1] * data[2][3];
    const float coef06 = data[1][1] * data[3][3] - data[3][1] * data[1][3];
    const float coef07 = data[1][1] * data[2][3] - data[2][1] * data[1][3];
    const float coef08 = data[2][1] * data[3][2] - data[3][1] * data[2][2];
    const float coef10 = data[1][1] * data[3][2] - data[3][1] * data[1][2];
    const float coef11 = data[1][1] * data[2][2] - data[2][1] * data[1][2];
    const float coef12 = data[2][0] * data[3][3] - data[3][0] * data[2][3];
    const float coef14 = data[1][0] * data[3][3] - data[3][0] * data[1][3];
    const float coef15 = data[1][0] * data[2][3] - data[2][0] * data[1][3];
    const float coef16 = data[2][0] * data[3][2] - data[3][0] * data[2][2];
    const float coef18 = data[1][0] * data[3][2] - data[3][0] * data[1][2];
    const float coef19 = data[1][0] * data[2][2] - data[2][0] * data[1][2];
    const float coef20 = data[2][0] * data[3][1] - data[3][0] * data[2][1];
    const float coef22 = data[1][0] * data[3][1] - data[3][0] * data[1][1];
    const float coef23 = data[1][0] * data[2][1] - data[2][0] * data[1][1];

    const Vec4f fac0(coef00, coef00, coef02, coef03);
    const Vec4f fac1(coef04, coef04, coef06, coef07);
    const Vec4f fac2(coef08, coef08, coef10, coef11);
    const Vec4f fac3(coef12, coef12, coef14, coef15);
    const Vec4f fac4(coef16, coef16, coef18, coef19);
    const Vec4f fac5(coef20, coef20, coef22, coef23);

    const Vec4f vec0(data[1][0], data[0][0], data[0][0], data[0][0]);
    const Vec4f vec1(data[1][1], data[0][1], data[0][1], data[0][1]);
    const Vec4f vec2(data[1][2], data[0][2], data[0][2], data[0][2]);
    const Vec4f vec3(data[1][3], data[0][3], data[0][3], data[0][3]);

    const Vec4f inv0(vec1 * fac0 - vec2 * fac1 + vec3 * fac2);
    const Vec4f inv1(vec0 * fac0 - vec2 * fac3 + vec3 * fac4);
    const Vec4f inv2(vec0 * fac1 - vec1 * fac3 + vec3 * fac5);
    const Vec4f inv3(vec0 * fac2 - vec1 * fac4 + vec2 * fac5);

    const Vec4f sign_a(+1, -1, +1, -1);
    const Vec4f sign_b(-1, +1, -1, +1);

    const Mat4 inverse = Mat4(inv0 * sign_a, inv1 * sign_b, inv2 * sign_a, inv3 * sign_b);
    const Vec4f row0(inverse.data[0][0], inverse.data[1][0], inverse.data[2][0], inverse.data[3][0]);

    const Vec4f dot0(data[0] * row0);
    const float dot1 = (dot0.x + dot0.y) + (dot0.z + dot0.w);
    const float inv_det = 1 / dot1;

    return Mat4(
        inverse.data[0] * inv_det,
        inverse.data[1] * inv_det,
        inverse.data[2] * inv_det,
        inverse.data[3] * inv_det);
}

Mat4 Mat4::transpose() const
{
    return Mat4(
        at(0, 0), at(1, 0), at(2, 0), at(3, 0),
        at(0, 1), at(1, 1), at(2, 1), at(3, 1),
        at(0, 2), at(1, 2), at(2, 2), at(3, 2),
        at(0, 3), at(1, 3), at(2, 3), at(3, 3));
}

float Mat4::at(int r, int c) const
{
    return data[c][r];
}

float &Mat4::at(int r, int c)
{
    return data[c][r];
}

Mat4 Mat4::translate(float x, float y, float z)
{
    return Mat4(
        1, 0, 0, x,
        0, 1, 0, y,
        0, 0, 1, z,
        0, 0, 0, 1);
}

Mat4 Mat4::rotate(const Vec3f &axis, float rad)
{
    const auto a = normalize(axis);
    const float sinv = std::sin(rad), cosv = std::cos(rad);

    Mat4 ret;

    ret.data[0][0] = a.x * a.x + (1 - a.x * a.x) * cosv;
    ret.data[0][1] = a.x * a.y * (1 - cosv) - a.z * sinv;
    ret.data[0][2] = a.x * a.z * (1 - cosv) + a.y * sinv;
    ret.data[0][3] = 0;

    ret.data[1][0] = a.x * a.y * (1 - cosv) + a.z * sinv;
    ret.data[1][1] = a.y * a.y + (1 - a.y * a.y) * cosv;
    ret.data[1][2] = a.y * a.z * (1 - cosv) - a.x * sinv;
    ret.data[1][3] = 0;

    ret.data[2][0] = a.x * a.z * (1 - cosv) - a.y * sinv;
    ret.data[2][1] = a.y * a.z * (1 - cosv) + a.x * sinv;
    ret.data[2][2] = a.z * a.z + (1 - a.z * a.z) * cosv;
    ret.data[2][3] = 0;

    ret.data[3][0] = 0;
    ret.data[3][1] = 0;
    ret.data[3][2] = 0;
    ret.data[3][3] = 1;

    return ret;
}

Mat4 Mat4::rotate_x(float rad)
{
    const auto S = std::sin(rad), C = std::cos(rad);
    return Mat4(
        1, 0, 0, 0,
        0, C, -S, 0,
        0, S, C, 0,
        0, 0, 0, 1);
}

Mat4 Mat4::rotate_y(float rad)
{
    const auto S = std::sin(rad), C = std::cos(rad);
    return Mat4(
        C, 0, S, 0,
        0, 1, 0, 0,
        -S, 0, C, 0,
        0, 0, 0, 1);
}

Mat4 Mat4::rotate_z(float rad)
{
    const auto S = std::sin(rad), C = std::cos(rad);
    return Mat4(
        C, -S, 0, 0,
        S, C, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1);
}

Mat4 Mat4::scale(float x, float y, float z)
{
    return Mat4(
        x, 0, 0, 0,
        0, y, 0, 0,
        0, 0, z, 0,
        0, 0, 0, 1);
}

Mat4 operator*(const Mat4 &a, const Mat4 &b)
{
    Mat4 ret;
    for(int r = 0; r != 4; ++r)
    {
        for(int c = 0; c != 4; ++c)
        {
            ret.at(r, c) =
                a.at(r, 0) * b.at(0, c) +
                a.at(r, 1) * b.at(1, c) +
                a.at(r, 2) * b.at(2, c) +
                a.at(r, 3) * b.at(3, c);
        }
    }
    return ret;
}

Vec4f operator*(const Mat4 &m, const Vec4f &v)
{
    Vec4f ret;
    for(int r = 0; r < 4; ++r)
    {
        ret[r] =
            m.at(r, 0) * v.x +
            m.at(r, 1) * v.y +
            m.at(r, 2) * v.z +
            m.at(r, 3) * v.w;
    }
    return ret;
}

BTRC_END
