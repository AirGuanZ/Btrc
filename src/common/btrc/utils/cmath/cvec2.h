#pragma once

#include <btrc/utils/cmath/cscalar.h>
#include <btrc/utils/math/vec2.h>

BTRC_BEGIN

CUJ_PROXY_CLASS_EX(CVec2f, Vec2f, x, y)
{
    CUJ_BASE_CONSTRUCTORS
    explicit CVec2f(f32 v = 0);
    CVec2f(f32 _x, f32 _y);
    CVec2f(const Vec2f &v);
    explicit CVec2f(ref<cstd::LCG> rng);
};

CUJ_PROXY_CLASS_EX(CVec2u, Vec2u, x, y)
{
    CUJ_BASE_CONSTRUCTORS
    explicit CVec2u(u32 v = 0);
    CVec2u(u32 _x, u32 _y);
    CVec2u(const Vec2u &v);
};

inline CVec2f load_aligned(ptr<CVec2f> addr);
inline CVec2u load_aligned(ptr<CVec2u> addr);

inline void save_aligned(ref<CVec2f> val, ptr<CVec2f> addr);
inline void save_aligned(ref<CVec2u> val, ptr<CVec2u> addr);

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

inline CVec2f load_aligned(ptr<CVec2f> addr)
{
    f32 x, y;
    cstd::load_f32x2(cuj::bitcast<ptr<f32>>(addr), x, y);
    return CVec2f(x, y);
}

inline CVec2u load_aligned(ptr<CVec2u> addr)
{
    u32 x, y;
    cstd::load_u32x2(cuj::bitcast<ptr<u32>>(addr), x, y);
    return CVec2u(x, y);
}

inline void save_aligned(ref<CVec2f> val, ptr<CVec2f> addr)
{
    cstd::store_f32x2(cuj::bitcast<ptr<f32>>(addr), val.x, val.y);
}

inline void save_aligned(ref<CVec2u> val, ptr<CVec2u> addr)
{
    cstd::store_u32x2(cuj::bitcast<ptr<u32>>(addr), val.x, val.y);
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

BTRC_END
