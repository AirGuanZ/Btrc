#include <btrc/utils/math/mat4.h>

BTRC_BEGIN

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

BTRC_END
