#pragma once

#include <btrc/core/utils/cmath/cvec3.h>
#include <btrc/core/utils/math/quaterion.h>

BTRC_CORE_BEGIN

CUJ_PROXY_CLASS_EX(CQuaterion, Quaterion, w, x, y, z)
{
    CUJ_BASE_CONSTRUCTORS

    CQuaterion();

    CQuaterion(f32 w, f32 x, f32 y, f32 z);

    CQuaterion(const CVec3f &axis, f32 rad);

    CQuaterion(const Quaterion &q);

    CVec3f apply_to_vector(const CVec3f &v) const;

    CQuaterion operator*(const CQuaterion &rhs) const;
};

inline CQuaterion normalize(const CQuaterion &q);

inline CQuaterion conjugate(const CQuaterion &q);

inline CQuaterion slerp(const CQuaterion &lhs, const CQuaterion &rhs, f32 t);

// ========================== impl ==========================

inline CQuaterion::CQuaterion()
    : CQuaterion(1, 0, 0, 0)
{
    
}

inline CQuaterion::CQuaterion(f32 _w, f32 _x, f32 _y, f32 _z)
{
    w = _w;
    x = _x;
    y = _y;
    z = _z;
}

inline CQuaterion::CQuaterion(const CVec3f &axis, f32 rad)
{
    var naxis = normalize(axis);
    var half_theta = 0.5f * rad;
    var sin_angle = cstd::sin(half_theta);
    var cos_angle = cstd::cos(half_theta);
    x = sin_angle * naxis.x;
    y = sin_angle * naxis.y;
    z = sin_angle * naxis.z;
    w = cos_angle;
}

inline CQuaterion::CQuaterion(const Quaterion &q)
    : CQuaterion(q.w, q.x, q.y, q.z)
{
    
}

inline CVec3f CQuaterion::apply_to_vector(const CVec3f &v) const
{
    CQuaterion vq(0, v.x, v.y, v.z);
    CQuaterion inv_this = conjugate(*this);
    CQuaterion result = *this * vq * inv_this;
    return CVec3f(result.x, result.y, result.z);
}

inline CQuaterion CQuaterion::operator*(const CQuaterion &rhs) const
{
    return CQuaterion(
        w * rhs.w - x * rhs.x - y * rhs.y - z * rhs.z,
        w * rhs.x + x * rhs.w + y * rhs.z - z * rhs.y,
        w * rhs.y + y * rhs.w + z * rhs.x - x * rhs.z,
        w * rhs.z + z * rhs.w + x * rhs.y - y * rhs.x);
}

inline CQuaterion normalize(const CQuaterion &q)
{
    var len = cstd::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    var inv_len = 1.0f / len;
    return CQuaterion(q.w * inv_len, q.x * inv_len, q.y * inv_len, q.z * inv_len);
}

inline CQuaterion conjugate(const CQuaterion &q)
{
    return CQuaterion(q.w, -q.x, -q.y, -q.z);
}

inline CQuaterion slerp(const CQuaterion &lhs, const CQuaterion &rhs, f32 t)
{
    var cos_theta = lhs.x * rhs.x
                  + lhs.y * rhs.y
                  + lhs.z * rhs.z
                  + lhs.w * rhs.w;
    CQuaterion real_rhs = rhs;
    $if(cos_theta < 0)
    {
        cos_theta = -cos_theta;
        real_rhs.x = -real_rhs.x;
        real_rhs.y = -real_rhs.y;
        real_rhs.z = -real_rhs.z;
        real_rhs.w = -real_rhs.w;
    };

    f32 lhs_w, rhs_w;
    $if(1.0f - cos_theta > 1e-4f)
    {
        var theta = cstd::acos(cos_theta);
        var sin_theta = cstd::sin(theta);
        lhs_w = cstd::sin((1.0f - t) * theta) / sin_theta;
        rhs_w = cstd::sin(t * theta) / sin_theta;
    }
    $else
    {
        lhs_w = 1.0f - t;
        rhs_w = t;
    };

    return CQuaterion(
        lhs_w * lhs.w + rhs_w * real_rhs.w,
        lhs_w * lhs.x + rhs_w * real_rhs.x,
        lhs_w * lhs.y + rhs_w * real_rhs.y,
        lhs_w * lhs.z + rhs_w * real_rhs.z);
}

BTRC_CORE_END
