#pragma once

#include <btrc/core/utils/math/vec3.h>

BTRC_CORE_BEGIN

template<typename T>
class Vec4
{
public:

    using ElementType = T;

    T x, y, z, w;

    BTRC_XPU Vec4();

    BTRC_XPU explicit Vec4(T v);

    BTRC_XPU Vec4(T x, T y, T z, T w);

    BTRC_XPU Vec4(const Vec3<T> &xyz, T w);
};

using Vec4f = Vec4<float>;
using Vec4i = Vec4<int32_t>;
using Vec4u = Vec4<uint32_t>;
using Vec4b = Vec4<uint8_t>;

template<typename T>
BTRC_XPU T length_square(const Vec4<T> &v);

template<typename T>
BTRC_XPU T length(const Vec4<T> &v);

template<typename T>
BTRC_XPU Vec4<T> normalize(const Vec4<T> &v);

template<typename T>
BTRC_XPU T dot(const Vec4<T> &a, const Vec4<T> &b);

template<typename T> BTRC_XPU Vec4<T> operator+(const Vec4<T> &a, const Vec4<T> &b);
template<typename T> BTRC_XPU Vec4<T> operator-(const Vec4<T> &a, const Vec4<T> &b);
template<typename T> BTRC_XPU Vec4<T> operator*(const Vec4<T> &a, const Vec4<T> &b);
template<typename T> BTRC_XPU Vec4<T> operator/(const Vec4<T> &a, const Vec4<T> &b);

template<typename T> BTRC_XPU Vec4<T> operator+(T a, const Vec4<T> &b);
template<typename T> BTRC_XPU Vec4<T> operator-(T a, const Vec4<T> &b);
template<typename T> BTRC_XPU Vec4<T> operator*(T a, const Vec4<T> &b);
template<typename T> BTRC_XPU Vec4<T> operator/(T a, const Vec4<T> &b);

template<typename T> BTRC_XPU Vec4<T> operator+(const Vec4<T> &a, T b);
template<typename T> BTRC_XPU Vec4<T> operator-(const Vec4<T> &a, T b);
template<typename T> BTRC_XPU Vec4<T> operator*(const Vec4<T> &a, T b);
template<typename T> BTRC_XPU Vec4<T> operator/(const Vec4<T> &a, T b);

// ========================== impl ==========================

template<typename T>
BTRC_XPU Vec4<T>::Vec4()
    : Vec4(T(0))
{
    
}

template<typename T>
BTRC_XPU Vec4<T>::Vec4(T v)
    : Vec4(v, v, v, v)
{
    
}

template<typename T>
BTRC_XPU Vec4<T>::Vec4(T x, T y, T z, T w)
    : x(x), y(y), z(z), w(w)
{
    
}

template<typename T>
BTRC_XPU Vec4<T>::Vec4(const Vec3<T> &xyz, T w)
    : Vec4(xyz.x, xyz.y, xyz.z, w)
{
    
}

template<typename T>
BTRC_XPU T length_square(const Vec4<T> &v)
{
    return dot(v, v);
}

template<typename T>
BTRC_XPU T length(const Vec4<T> &v)
{
    static_assert(std::is_floating_point_v<T>);
    return btrc_sqrt(length_square(v));
}

template<typename T>
BTRC_XPU Vec4<T> normalize(const Vec4<T> &v)
{
    static_assert(std::is_floating_point_v<T>);
    return T(1) / length(v) * v;
}

template<typename T>
BTRC_XPU T dot(const Vec4<T> &a, const Vec4<T> &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

template<typename T>
BTRC_XPU Vec4<T> operator+(const Vec4<T> &a, const Vec4<T> &b)
{
    return Vec4<T>(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

template<typename T>
BTRC_XPU Vec4<T> operator-(const Vec4<T> &a, const Vec4<T> &b)
{
    return Vec4<T>(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

template<typename T>
BTRC_XPU Vec4<T> operator*(const Vec4<T> &a, const Vec4<T> &b)
{
    return Vec4<T>(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

template<typename T>
BTRC_XPU Vec4<T> operator/(const Vec4<T> &a, const Vec4<T> &b)
{
    return Vec4<T>(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

template<typename T>
BTRC_XPU Vec4<T> operator+(T a, const Vec4<T> &b)
{
    return Vec4<T>(a + b.x, a + b.y, a + b.z, a + b.w);
}

template<typename T>
BTRC_XPU Vec4<T> operator-(T a, const Vec4<T> &b)
{
    return Vec4<T>(a - b.x, a - b.y, a - b.z, a - b.w);
}

template<typename T>
BTRC_XPU Vec4<T> operator*(T a, const Vec4<T> &b)
{
    return Vec4<T>(a * b.x, a * b.y, a * b.z, a * b.w);
}

template<typename T>
BTRC_XPU Vec4<T> operator/(T a, const Vec4<T> &b)
{
    return Vec4<T>(a / b.x, a / b.y, a / b.z, a / b.w);
}

template<typename T>
BTRC_XPU Vec4<T> operator+(const Vec4<T> &a, T b)
{
    return Vec4<T>(a.x + b, a.y + b, a.z + b, a.w + b);
}

template<typename T>
BTRC_XPU Vec4<T> operator-(const Vec4<T> &a, T b)
{
    return Vec4<T>(a.x - b, a.y - b, a.z - b, a.w - b);
}

template<typename T>
BTRC_XPU Vec4<T> operator*(const Vec4<T> &a, T b)
{
    return Vec4<T>(a.x * b, a.y * b, a.z * b, a.w * b);
}

template<typename T>
BTRC_XPU Vec4<T> operator/(const Vec4<T> &a, T b)
{
    return Vec4<T>(a.x / b, a.y / b, a.z / b, a.w / b);
}

BTRC_CORE_END
