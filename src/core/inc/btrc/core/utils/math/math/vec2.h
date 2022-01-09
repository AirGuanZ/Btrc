#pragma once

#include <btrc/core/utils/math/math/scalar.h>

BTRC_CORE_BEGIN

template<typename T>
class Vec2
{
public:

    using ElementType = T;

    T x, y;

    Vec2();

    explicit Vec2(T v);

    Vec2(T x, T y);
};

using Vec2f = Vec2<float>;
using Vec2i = Vec2<int32_t>;
using Vec2u = Vec2<uint32_t>;

template<typename T>
T length_square(const Vec2<T> &v);

template<typename T>
T length(const Vec2<T> &v);

template<typename T>
Vec2<T> normalize(const Vec2<T> &v);

template<typename T>
T dot(const Vec2<T> &a, const Vec2<T> &b);

template<typename T> Vec2<T> operator+(const Vec2<T> &a, const Vec2<T> &b);
template<typename T> Vec2<T> operator-(const Vec2<T> &a, const Vec2<T> &b);
template<typename T> Vec2<T> operator*(const Vec2<T> &a, const Vec2<T> &b);
template<typename T> Vec2<T> operator/(const Vec2<T> &a, const Vec2<T> &b);

template<typename T> Vec2<T> operator+(T a, const Vec2<T> &b);
template<typename T> Vec2<T> operator-(T a, const Vec2<T> &b);
template<typename T> Vec2<T> operator*(T a, const Vec2<T> &b);
template<typename T> Vec2<T> operator/(T a, const Vec2<T> &b);

template<typename T> Vec2<T> operator+(const Vec2<T> &a, T b);
template<typename T> Vec2<T> operator-(const Vec2<T> &a, T b);
template<typename T> Vec2<T> operator*(const Vec2<T> &a, T b);
template<typename T> Vec2<T> operator/(const Vec2<T> &a, T b);

// ========================== impl ==========================

template<typename T>
Vec2<T>::Vec2()
    : Vec2(T(0))
{
    
}

template<typename T>
Vec2<T>::Vec2(T v)
    : Vec2(v, v)
{
    
}

template<typename T>
Vec2<T>::Vec2(T x, T y)
    : x(x), y(y)
{
    
}

template<typename T>
T length_square(const Vec2<T> &v)
{
    return dot(v, v);
}

template<typename T>
T length(const Vec2<T> &v)
{
    static_assert(std::is_floating_point_v<T>);
    return btrc_sqrt(length_square(v));
}

template<typename T>
Vec2<T> normalize(const Vec2<T> &v)
{
    static_assert(std::is_floating_point_v<T>);
    return T(1) / length(v) * v;
}

template<typename T>
T dot(const Vec2<T> &a, const Vec2<T> &b)
{
    return a.x * b.x + a.y * b.y;
}

template<typename T>
Vec2<T> operator+(const Vec2<T> &a, const Vec2<T> &b)
{
    return Vec2<T>(a.x + b.x, a.y + b.y);
}

template<typename T>
Vec2<T> operator-(const Vec2<T> &a, const Vec2<T> &b)
{
    return Vec2<T>(a.x - b.x, a.y - b.y);
}

template<typename T>
Vec2<T> operator*(const Vec2<T> &a, const Vec2<T> &b)
{
    return Vec2<T>(a.x * b.x, a.y * b.y);
}

template<typename T>
Vec2<T> operator/(const Vec2<T> &a, const Vec2<T> &b)
{
    return Vec2<T>(a.x / b.x, a.y / b.y);
}

template<typename T>
Vec2<T> operator+(T a, const Vec2<T> &b)
{
    return Vec2<T>(a + b.x, a + b.y);
}

template<typename T>
Vec2<T> operator-(T a, const Vec2<T> &b)
{
    return Vec2<T>(a - b.x, a - b.y);
}

template<typename T>
Vec2<T> operator*(T a, const Vec2<T> &b)
{
    return Vec2<T>(a * b.x, a * b.y);
}

template<typename T>
Vec2<T> operator/(T a, const Vec2<T> &b)
{
    return Vec2<T>(a / b.x, a / b.y);
}

template<typename T>
Vec2<T> operator+(const Vec2<T> &a, T b)
{
    return Vec2<T>(a.x + b, a.y + b);
}

template<typename T>
Vec2<T> operator-(const Vec2<T> &a, T b)
{
    return Vec2<T>(a.x - b, a.y - b);
}

template<typename T>
Vec2<T> operator*(const Vec2<T> &a, T b)
{
    return Vec2<T>(a.x * b, a.y * b);
}

template<typename T>
Vec2<T> operator/(const Vec2<T> &a, T b)
{
    return Vec2<T>(a.x / b, a.y / b);
}

BTRC_CORE_END
