#pragma once

#include <btrc/core/utils/cmath/cscalar.h>
#include <btrc/core/utils/math/vec3.h>

BTRC_CORE_BEGIN

CUJ_PROXY_CLASS_EX(CVec3f, Vec3f, x, y, z)
{
    CUJ_BASE_CONSTRUCTORS

    explicit CVec3f(f32 v = 0);

    CVec3f(f32 _x, f32 _y, f32 _z);

    CVec3f(const Vec3f &v);

    explicit CVec3f(ref<cstd::LCG>);
};

inline f32 length_square(const CVec3f &v);

inline f32 length(const CVec3f &v);

inline CVec3f normalize(const CVec3f &v);

inline f32 dot(const CVec3f &a, const CVec3f &b);

inline CVec3f cross(const CVec3f &a, const CVec3f &b);

inline CVec3f operator+(const CVec3f &a, const CVec3f &b);
inline CVec3f operator-(const CVec3f &a, const CVec3f &b);
inline CVec3f operator*(const CVec3f &a, const CVec3f &b);
inline CVec3f operator/(const CVec3f &a, const CVec3f &b);

inline CVec3f operator+(f32 a, const CVec3f &b);
inline CVec3f operator-(f32 a, const CVec3f &b);
inline CVec3f operator*(f32 a, const CVec3f &b);
inline CVec3f operator/(f32 a, const CVec3f &b);

inline CVec3f operator+(const CVec3f &a, f32 b);
inline CVec3f operator-(const CVec3f &a, f32 b);
inline CVec3f operator*(const CVec3f &a, f32 b);
inline CVec3f operator/(const CVec3f &a, f32 b);

inline CVec3f operator-(const CVec3f &v);

// ========================== impl ==========================

inline CVec3f::CVec3f(f32 v)
    : CVec3f(v, v, v)
{
    
}

inline CVec3f::CVec3f(f32 _x, f32 _y, f32 _z)
{
    x = _x;
    y = _y;
    z = _z;
}

inline CVec3f::CVec3f(const Vec3f &v)
    : CVec3f(v.x, v.y, v.z)
{
    
}

inline CVec3f::CVec3f(ref<cstd::LCG> rng)
{
    x = rng.uniform_float();
    y = rng.uniform_float();
    z = rng.uniform_float();
}

inline f32 length_square(const CVec3f &v)
{
    return dot(v, v);
}

inline f32 length(const CVec3f &v)
{
    return cstd::sqrt(length_square(v));
}

inline CVec3f normalize(const CVec3f &v)
{
    f32 inv_len = cstd::rsqrt(length_square(v));
    return inv_len * v;
}

inline f32 dot(const CVec3f &a, const CVec3f &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline CVec3f cross(const CVec3f &a, const CVec3f &b)
{
    return CVec3f(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

inline CVec3f operator+(const CVec3f &a, const CVec3f &b)
{
    return CVec3f(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline CVec3f operator-(const CVec3f &a, const CVec3f &b)
{
    return CVec3f(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline CVec3f operator*(const CVec3f &a, const CVec3f &b)
{
    return CVec3f(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline CVec3f operator/(const CVec3f &a, const CVec3f &b)
{
    return CVec3f(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline CVec3f operator+(f32 a, const CVec3f &b)
{
    return CVec3f(a + b.x, a + b.y, a + b.z);
}

inline CVec3f operator-(f32 a, const CVec3f &b)
{
    return CVec3f(a - b.x, a - b.y, a - b.z);
}

inline CVec3f operator*(f32 a, const CVec3f &b)
{
    return CVec3f(a * b.x, a * b.y, a * b.z);
}

inline CVec3f operator/(f32 a, const CVec3f &b)
{
    return CVec3f(a / b.x, a / b.y, a / b.z);
}

inline CVec3f operator+(const CVec3f &a, f32 b)
{
    return CVec3f(a.x + b, a.y + b, a.z + b);
}

inline CVec3f operator-(const CVec3f &a, f32 b)
{
    return CVec3f(a.x - b, a.y - b, a.z - b);
}

inline CVec3f operator*(const CVec3f &a, f32 b)
{
    return CVec3f(a.x * b, a.y * b, a.z * b);
}

inline CVec3f operator/(const CVec3f &a, f32 b)
{
    return CVec3f(a.x / b, a.y / b, a.z / b);
}

inline CVec3f operator-(const CVec3f &v)
{
    return CVec3f(-v.x, -v.y, -v.z);
}

BTRC_CORE_END
