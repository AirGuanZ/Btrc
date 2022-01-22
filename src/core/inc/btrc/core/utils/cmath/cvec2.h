#pragma once

#include <btrc/core/utils/cmath/cscalar.h>
#include <btrc/core/utils/math/vec2.h>

BTRC_CORE_BEGIN

CUJ_PROXY_CLASS_EX(CVec2f, Vec2f, x, y)
{
    CUJ_BASE_CONSTRUCTORS
    CUJ_NONE_TRIVIALLY_COPYABLE
    explicit CVec2f(f32 v = 0);
    CVec2f(f32 _x, f32 _y);
    CVec2f(const Vec2f &v);
    CVec2f(const CVec2f &other) : CVec2f() { *this = other; }
    CVec2f &operator=(const CVec2f &other);
    explicit CVec2f(ref<cstd::LCG> rng);
};

CUJ_PROXY_CLASS_EX(CVec2u, Vec2u, x, y)
{
    CUJ_BASE_CONSTRUCTORS
    CUJ_NONE_TRIVIALLY_COPYABLE
    explicit CVec2u(u32 v = 0);
    CVec2u(u32 _x, u32 _y);
    CVec2u(const Vec2u &v);
    CVec2u(const CVec2u &other): CVec2u() { *this = other; }
    CVec2u &operator=(const CVec2u &other)
    {
        u32 _x, _y;
        cstd::load_u32x2(other.x.address(), _x, _y);
        cstd::store_u32x2(x.address(), _x, _y);
        return *this;
    }
};

inline f32 length_square(const CVec2f &v);
inline f32 length(const CVec2f &v);
inline f32 dot(const CVec2f &a, const CVec2f &b);
inline CVec2f normalize(const CVec2f &v);

inline CVec2f operator+(const CVec2f &a, const CVec2f &b);
inline CVec2f operator-(const CVec2f &a, const CVec2f &b);
inline CVec2f operator*(const CVec2f &a, const CVec2f &b);
inline CVec2f operator/(const CVec2f &a, const CVec2f &b);

inline CVec2f operator+(f32 a, const CVec2f &b);
inline CVec2f operator-(f32 a, const CVec2f &b);
inline CVec2f operator*(f32 a, const CVec2f &b);
inline CVec2f operator/(f32 a, const CVec2f &b);

inline CVec2f operator+(const CVec2f &a, f32 b);
inline CVec2f operator-(const CVec2f &a, f32 b);
inline CVec2f operator*(const CVec2f &a, f32 b);
inline CVec2f operator/(const CVec2f &a, f32 b);

// ========================== impl ==========================

inline CVec2f::CVec2f(f32 v)
    : CVec2f(v, v)
{
    
}

inline CVec2f::CVec2f(f32 _x, f32 _y)
{
    x = _x;
    y = _y;
}

inline CVec2f::CVec2f(const Vec2f &v)
    : CVec2f(v.x, v.y)
{
    
}

inline CVec2f &CVec2f::operator=(const CVec2f &other)
{
    f32 _x, _y;
    cstd::load_f32x2(other.x.address(), _x, _y);
    cstd::store_f32x2(x.address(), _x, _y);
    return *this;
}

inline CVec2f::CVec2f(ref<cstd::LCG> rng)
{
    x = rng.uniform_float();
    y = rng.uniform_float();
}

inline CVec2u::CVec2u(u32 v)
    : CVec2u(v, v)
{
    
}

inline CVec2u::CVec2u(u32 _x, u32 _y)
{
    x = _x;
    y = _y;
}

inline CVec2u::CVec2u(const Vec2u &v)
    : CVec2u(v.x, v.y)
{
    
}

inline f32 length_square(const CVec2f &v)
{
    return dot(v, v);
}

inline f32 length(const CVec2f &v)
{
    return cstd::sqrt(length_square(v));
}

inline CVec2f normalize(const CVec2f &v)
{
    f32 inv_len = cstd::rsqrt(length_square(v));
    return inv_len * v;
}

inline f32 dot(const CVec2f &a, const CVec2f &b)
{
    return a.x * b.x + a.y * b.y;
}

inline CVec2f operator+(const CVec2f &a, const CVec2f &b)
{
    return CVec2f(a.x + b.x, a.y + b.y);
}

inline CVec2f operator-(const CVec2f &a, const CVec2f &b)
{
    return CVec2f(a.x - b.x, a.y - b.y);
}

inline CVec2f operator*(const CVec2f &a, const CVec2f &b)
{
    return CVec2f(a.x * b.x, a.y * b.y);
}

inline CVec2f operator/(const CVec2f &a, const CVec2f &b)
{
    return CVec2f(a.x / b.x, a.y / b.y);
}

inline CVec2f operator+(f32 a, const CVec2f &b)
{
    return CVec2f(a + b.x, a + b.y);
}

inline CVec2f operator-(f32 a, const CVec2f &b)
{
    return CVec2f(a - b.x, a - b.y);
}

inline CVec2f operator*(f32 a, const CVec2f &b)
{
    return CVec2f(a * b.x, a * b.y);
}

inline CVec2f operator/(f32 a, const CVec2f &b)
{
    return CVec2f(a / b.x, a / b.y);
}

inline CVec2f operator+(const CVec2f &a, f32 b)
{
    return CVec2f(a.x + b, a.y + b);
}

inline CVec2f operator-(const CVec2f &a, f32 b)
{
    return CVec2f(a.x - b, a.y - b);
}

inline CVec2f operator*(const CVec2f &a, f32 b)
{
    return CVec2f(a.x * b, a.y * b);
}

inline CVec2f operator/(const CVec2f &a, f32 b)
{
    return CVec2f(a.x / b, a.y / b);
}

BTRC_CORE_END
