#include <btrc/utils/math/mat3.h>

BTRC_BEGIN

namespace
{

    float det(float m00, float m01, float m10, float m11)
    {
        return m00 * m10 - m01 * m11;
    }

    Mat3 adj(const Mat3 &m)
    {
        Mat3 adj;
        adj.data[0][0] = +det(m.data[1][1], m.data[2][1], m.data[1][2], m.data[2][2]);
        adj.data[0][1] = -det(m.data[1][0], m.data[2][0], m.data[1][2], m.data[2][2]);
        adj.data[0][2] = +det(m.data[1][0], m.data[2][0], m.data[1][1], m.data[2][1]);
        adj.data[1][0] = -det(m.data[0][1], m.data[2][1], m.data[0][2], m.data[2][2]);
        adj.data[1][1] = +det(m.data[0][0], m.data[2][0], m.data[0][2], m.data[2][2]);
        adj.data[1][2] = -det(m.data[0][0], m.data[2][0], m.data[0][1], m.data[2][1]);
        adj.data[2][0] = +det(m.data[0][1], m.data[1][1], m.data[0][2], m.data[1][2]);
        adj.data[2][1] = -det(m.data[0][0], m.data[1][0], m.data[0][2], m.data[1][2]);
        adj.data[2][2] = +det(m.data[0][0], m.data[1][0], m.data[0][1], m.data[1][1]);
        return adj.transpose();
    }

} // namespace anonymous

Mat3::Mat3()
    : Mat3(
        1, 0, 0,
        0, 1, 0,
        0, 0, 1)
{
    
}

Mat3::Mat3(
    float m00, float m01, float m02,
    float m10, float m11, float m12,
    float m20, float m21, float m22)
    : Mat3(
        { m00, m10, m20 },
        { m01, m11, m21 },
        { m02, m12, m22 })
{
    
}

Mat3::Mat3(const Vec3f &c0, const Vec3f &c1, const Vec3f &c2)
    : data{ c0, c1, c2 }
{
    
}

Mat3 Mat3::inverse() const
{
    // return adj / dot(data[0], adj.get_row(0));
    const Mat3 a = adj(*this);
    const Vec3f r0 = { a.data[0][0], a.data[1][0], a.data[2][0] };
    return a / dot(data[0], r0);
}

Mat3 Mat3::transpose() const
{
    return Mat3(
        at(0, 0), at(1, 0), at(2, 0),
        at(0, 1), at(1, 1), at(2, 1),
        at(0, 2), at(1, 2), at(2, 2));
}

float Mat3::at(int r, int c) const
{
    return data[c][r];
}

float &Mat3::at(int r, int c)
{
    return data[c][r];
}

Mat3 Mat3::translate(float x, float y)
{
    return Mat3(
        1, 0, x,
        0, 1, y,
        0, 0, 1);
}

Mat3 Mat3::rotate(float rad)
{
    const auto S = std::sin(rad), C = std::cos(rad);
    return Mat3(
        C, -S, 0,
        S, C, 0,
        0, 0, 1);
}

Mat3 Mat3::scale(float x, float y)
{
    return Mat3(
        x, 0, 0,
        0, y, 0,
        0, 0, 1);
}

Mat3 operator*(const Mat3 &a, const Mat3 &b)
{
    Mat3 ret;
    for(int r = 0; r < 3; ++r)
    {
        for(int c = 0; c < 3; ++c)
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

Mat3 operator/(const Mat3 &a, float b)
{
    Mat3 ret;
    for(int c = 0; c < 3; ++c)
    {
        for(int r = 0; r < 3; ++r)
            ret.data[c][r] = a.data[c][r] / b;
    }
    return ret;
}

BTRC_END
