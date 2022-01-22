#pragma once

#include <btrc/core/utils/cmath/cscalar.h>
#include <btrc/core/utils/cmath/cvec3.h>
#include <btrc/core/utils/math/vec4.h>

BTRC_CORE_BEGIN

CUJ_PROXY_CLASS_EX(CVec4f, Vec4f, x, y, z, w)
{
    CUJ_BASE_CONSTRUCTORS
    CUJ_NONE_TRIVIALLY_COPYABLE
    explicit CVec4f(f32 v = 0);
    CVec4f(f32 _x, f32 _y, f32 _z, f32 _w);
    CVec4f(const Vec4f &v);
    CVec4f(const CVec4f &other) : CVec4f() { *this = other; }
    CVec4f(const CVec3f &xyz, f32 _w);
    CVec4f &operator=(const CVec4f &other);
    CVec3f xyz() const;
};

CUJ_PROXY_CLASS_EX(CVec4u, Vec4u, x, y, z, w)
{
    CUJ_BASE_CONSTRUCTORS
    CUJ_NONE_TRIVIALLY_COPYABLE
    explicit CVec4u(u32 v = 0);
    CVec4u(u32 _x, u32 _y, u32 _z, u32 _w);
    CVec4u(const Vec4u &v);
    CVec4u(const CVec4u &other) : CVec4u() { *this = other; }
    CVec4u &operator=(const CVec4u &other);
};

inline f32 length_square(const CVec4f &v);
inline f32 length(const CVec4f &v);
inline f32 dot(const CVec4f &a, const CVec4f &b);
inline CVec4f normalize(const CVec4f &v);

inline CVec4f operator+(const CVec4f &a, const CVec4f &b);
inline CVec4f operator-(const CVec4f &a, const CVec4f &b);
inline CVec4f operator*(const CVec4f &a, const CVec4f &b);
inline CVec4f operator/(const CVec4f &a, const CVec4f &b);

inline CVec4f operator+(f32 a, const CVec4f &b);
inline CVec4f operator-(f32 a, const CVec4f &b);
inline CVec4f operator*(f32 a, const CVec4f &b);
inline CVec4f operator/(f32 a, const CVec4f &b);

inline CVec4f operator+(const CVec4f &a, f32 b);
inline CVec4f operator-(const CVec4f &a, f32 b);
inline CVec4f operator*(const CVec4f &a, f32 b);
inline CVec4f operator/(const CVec4f &a, f32 b);

// ========================== impl ==========================

inline CVec4f::CVec4f(f32 v)
    : CVec4f(v, v, v, v)
{
    
}

inline CVec4f::CVec4f(f32 _x, f32 _y, f32 _z, f32 _w)
{
    x = _x;
    y = _y;
    z = _z;
    w = _w;
}

inline CVec4f::CVec4f(const Vec4f &v)
    : CVec4f(v.x, v.y, v.z, v.w)
{
    
}

inline CVec4f::CVec4f(const CVec3f &xyz, f32 _w)
    : CVec4f(xyz.x, xyz.y, xyz.z, _w)
{
    
}

inline CVec4f &CVec4f::operator=(const CVec4f &other)
{
    f32 _x, _y, _z, _w;
    cstd::load_f32x4(other.x.address(), _x, _y, _z, _w);
    cstd::store_f32x4(x.address(), _x, _y, _z, _w);
    return *this;
}

inline CVec3f CVec4f::xyz() const
{
    return CVec3f(x, y, z);
}

inline CVec4u::CVec4u(u32 v)
    : CVec4u(v, v, v, v)
{

}

inline CVec4u::CVec4u(u32 _x, u32 _y, u32 _z, u32 _w)
{
    x = _x;
    y = _y;
    z = _z;
    w = _w;
}

inline CVec4u::CVec4u(const Vec4u &v)
    : CVec4u(v.x, v.y, v.z, v.w)
{

}

inline CVec4u &CVec4u::operator=(const CVec4u &other)
{
    u32 _x, _y, _z, _w;
    cstd::load_u32x4(other.x.address(), _x, _y, _z, _w);
    cstd::store_u32x4(x.address(), _x, _y, _z, _w);
    return *this;
}

inline f32 length_square(const CVec4f &v)
{
    return dot(v, v);
}

inline f32 length(const CVec4f &v)
{
    return cstd::sqrt(length_square(v));
}

inline CVec4f normalize(const CVec4f &v)
{
    f32 inv_len = cstd::rsqrt(length_square(v));
    return inv_len * v;
}

inline f32 dot(const CVec4f &a, const CVec4f &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline CVec4f operator+(const CVec4f &a, const CVec4f &b)
{
    return CVec4f(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline CVec4f operator-(const CVec4f &a, const CVec4f &b)
{
    return CVec4f(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline CVec4f operator*(const CVec4f &a, const CVec4f &b)
{
    return CVec4f(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline CVec4f operator/(const CVec4f &a, const CVec4f &b)
{
    return CVec4f(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

inline CVec4f operator+(f32 a, const CVec4f &b)
{
    return CVec4f(a + b.x, a + b.y, a + b.z, a + b.w);
}

inline CVec4f operator-(f32 a, const CVec4f &b)
{
    return CVec4f(a - b.x, a - b.y, a - b.z, a - b.w);
}

inline CVec4f operator*(f32 a, const CVec4f &b)
{
    return CVec4f(a * b.x, a * b.y, a * b.z, a * b.w);
}

inline CVec4f operator/(f32 a, const CVec4f &b)
{
    return CVec4f(a / b.x, a / b.y, a / b.z, a / b.w);
}

inline CVec4f operator+(const CVec4f &a, f32 b)
{
    return CVec4f(a.x + b, a.y + b, a.z + b, a.w + b);
}

inline CVec4f operator-(const CVec4f &a, f32 b)
{
    return CVec4f(a.x - b, a.y - b, a.z - b, a.w - b);
}

inline CVec4f operator*(const CVec4f &a, f32 b)
{
    return CVec4f(a.x * b, a.y * b, a.z * b, a.w * b);
}

inline CVec4f operator/(const CVec4f &a, f32 b)
{
    return CVec4f(a.x / b, a.y / b, a.z / b, a.w / b);
}

BTRC_CORE_END
