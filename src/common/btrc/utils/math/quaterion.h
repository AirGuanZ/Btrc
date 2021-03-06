#pragma once

#include <btrc/utils/math/vec3.h>

BTRC_BEGIN

class Quaterion
{
public:

    float w, x, y, z;

    Quaterion();

    Quaterion(float w, float x, float y, float z);

    Quaterion(const Vec3f &axis, float rad);

    Vec3f apply_to_vector(const Vec3f &v) const;

    Quaterion operator*(const Quaterion &rhs) const;

    // m: row-major 3x3 matrix
    void to_rotation_matrix(float *m) const;
};

inline Quaterion normalize(const Quaterion &q);

inline Quaterion conjugate(const Quaterion &q);

inline Quaterion slerp(const Quaterion &lhs, const Quaterion &rhs, float t);

// ========================== impl ==========================

inline Quaterion::Quaterion()
    : Quaterion(1, 0, 0, 0)
{
    
}

inline Quaterion::Quaterion(float w, float x, float y, float z)
    : w(w), x(x), y(y), z(z)
{
    
}

inline Quaterion::Quaterion(const Vec3f &axis, float rad)
{
    const Vec3f naxis = normalize(axis);
    const float half_theta = 0.5f * rad;
    const float sin_angle = std::sin(half_theta);
    const float cos_angle = std::cos(half_theta);
    x = sin_angle * naxis.x;
    y = sin_angle * naxis.y;
    z = sin_angle * naxis.z;
    w = cos_angle;
}

inline Vec3f Quaterion::apply_to_vector(const Vec3f &v) const
{
    const Quaterion vq(0, v.x, v.y, v.z);
    const Quaterion inv_this = conjugate(*this);
    const Quaterion result = *this * vq * inv_this;
    return Vec3f(result.x, result.y, result.z);
}

inline Quaterion Quaterion::operator*(const Quaterion &rhs) const
{
    return Quaterion(
        w * rhs.w - x * rhs.x - y * rhs.y - z * rhs.z,
        w * rhs.x + x * rhs.w + y * rhs.z - z * rhs.y,
        w * rhs.y + y * rhs.w + z * rhs.x - x * rhs.z,
        w * rhs.z + z * rhs.w + x * rhs.y - y * rhs.x);
}

inline void Quaterion::to_rotation_matrix(float *m) const
{
    const auto [qr, qi, qj, qk] = normalize(*this);
    //const float qr = w, qi = x, qj = y, qk = z;
    m[0] = 1 - 2 * (qj * qj + qk * qk); m[1] = 2 * (qi * qj - qk * qr);     m[2] = 2 * (qi * qk + qj * qr);
    m[3] = 2 * (qi * qj + qk * qr);     m[4] = 1 - 2 * (qi * qi + qk * qk); m[5] = 2 * (qj * qk - qi * qr);
    m[6] = 2 * (qi * qk - qj * qr);     m[7] = 2 * (qj * qk + qi * qr);     m[8] = 1 - 2 * (qi * qi + qj * qj);
}

inline Quaterion normalize(const Quaterion &q)
{
    const float len = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    const float inv_len = 1 / len;
    return Quaterion(q.w * inv_len, q.x * inv_len, q.y * inv_len, q.z * inv_len);
}

inline Quaterion conjugate(const Quaterion &q)
{
    return Quaterion(q.w, -q.x, -q.y, -q.z);
}

inline Quaterion slerp(
    const Quaterion &lhs, const Quaterion &rhs, float t)
{
    float cos_theta = lhs.x * rhs.x
                    + lhs.y * rhs.y
                    + lhs.z * rhs.z
                    + lhs.w * rhs.w;
    Quaterion real_rhs = rhs;
    if(cos_theta < 0)
    {
        cos_theta = -cos_theta;
        real_rhs.x = -real_rhs.x;
        real_rhs.y = -real_rhs.y;
        real_rhs.z = -real_rhs.z;
        real_rhs.w = -real_rhs.w;
    }

    float lhs_w, rhs_w;
    if(1 - cos_theta > 1e-4f)
    {
        const float theta = std::acos(cos_theta);
        const float sin_theta = std::sin(theta);
        lhs_w = std::sin((1 - t) * theta) / sin_theta;
        rhs_w = std::sin(t * theta) / sin_theta;
    }
    else
    {
        lhs_w = 1 - t;
        rhs_w = t;
    }

    return Quaterion(
        lhs_w * lhs.w + rhs_w * real_rhs.w,
        lhs_w * lhs.x + rhs_w * real_rhs.x,
        lhs_w * lhs.y + rhs_w * real_rhs.y,
        lhs_w * lhs.z + rhs_w * real_rhs.z);
}

BTRC_END
