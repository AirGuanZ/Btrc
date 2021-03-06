#pragma once

#include <btrc/utils/cmath/cscalar.h>
#include <btrc/utils/math/vec3.h>

BTRC_BEGIN

CUJ_PROXY_CLASS_EX(CVec3f, Vec3f, x, y, z)
{
    CUJ_BASE_CONSTRUCTORS

    explicit CVec3f(f32 v = 0);

    CVec3f(f32 _x, f32 _y, f32 _z);

    CVec3f(const Vec3f &v);

    explicit CVec3f(ref<cstd::LCG>);

    explicit CVec3f(ref<cstd::PCG>);

    ref<f32> operator[](i32 i) const;
};

CUJ_PROXY_CLASS_EX(CVec3i, Vec3i, x, y, z)
{
    CUJ_BASE_CONSTRUCTORS

    explicit CVec3i(i32 v = 0);

    CVec3i(i32 _x, i32 _y, i32 _z);

    CVec3i(const Vec3i &v);

    ref<i32> operator[](i32 i) const;
};

inline CVec3f load_aligned(ptr<CVec3f> addr);

inline void save_aligned(ref<CVec3f> val, ptr<CVec3f> addr);

inline f32 length_square(const CVec3f &v);

inline f32 length(const CVec3f &v);

inline CVec3f normalize(const CVec3f &v);

inline f32 dot(const CVec3f &a, const CVec3f &b);

inline CVec3f cross(const CVec3f &a, const CVec3f &b);

inline f32 cos(const CVec3f &a, const CVec3f &b);

inline boolean isfinite(const CVec3f &v);

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

inline CVec3i operator+(const CVec3i &a, const CVec3i &b);
inline CVec3i operator+(i32 a, const CVec3i &b);
inline CVec3i operator+(const CVec3i &a, i32 b);

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

inline CVec3f::CVec3f(ref<cstd::PCG> rng)
{
    x = rng.uniform_float();
    y = rng.uniform_float();
    z = rng.uniform_float();
}

inline ref<f32> CVec3f::operator[](i32 i) const
{
    return *(x.address() + i);
}

inline CVec3i::CVec3i(i32 v)
    : CVec3i(v, v, v)
{
    
}

inline CVec3i::CVec3i(i32 _x, i32 _y, i32 _z)
{
    x = _x;
    y = _y;
    z = _z;
}

inline CVec3i::CVec3i(const Vec3i &v)
    : CVec3i(v.x, v.y, v.z)
{
    
}

inline ref<i32> CVec3i::operator[](i32 i) const
{
    return *(x.address() + i);
}

inline CVec3f load_aligned(ptr<CVec3f> addr)
{
    f32 x, y, z;
    cstd::load_f32x3(cuj::bitcast<ptr<f32>>(addr), x, y, z);
    return CVec3f(x, y, z);
}

inline void save_aligned(ref<CVec3f> val, ptr<CVec3f> addr)
{
    cstd::store_f32x3(cuj::bitcast<ptr<f32>>(addr), val.x, val.y, val.z);
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

inline f32 cos(const CVec3f &a, const CVec3f &b)
{
    return dot(normalize(a), normalize(b));
}

inline boolean isfinite(const CVec3f &v)
{
    return cstd::isfinite(v.x) & cstd::isfinite(v.y) & cstd::isfinite(v.z);
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

inline CVec3i operator+(const CVec3i &a, const CVec3i &b)
{
    return CVec3i(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline CVec3i operator+(i32 a, const CVec3i &b)
{
    return CVec3i(a) + b;
}

inline CVec3i operator+(const CVec3i &a, i32 b)
{
    return a + CVec3i(b);
}

BTRC_END
