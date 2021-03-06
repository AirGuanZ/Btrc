#pragma once

#include <btrc/utils/math/scalar.h>

BTRC_BEGIN

template<typename T>
class Vec3
{
public:

    using ElementType = T;

    T x, y, z;

    Vec3();

    explicit Vec3(T v);

    Vec3(T x, T y, T z);

    float operator[](size_t i) const;

    float &operator[](size_t i);
};

using Vec3f = Vec3<float>;
using Vec3i = Vec3<int>;
using Vec3b = Vec3<uint8_t>;

template<typename T>
T length_square(const Vec3<T> &v);

template<typename T>
T length(const Vec3<T> &v);

template<typename T>
Vec3<T> normalize(const Vec3<T> &v);

template<typename T>
T dot(const Vec3<T> &a, const Vec3<T> &b);

template<typename T>
Vec3<T> cross(const Vec3<T> &a, const Vec3<T> &b);

template<typename T> Vec3<T> operator+(const Vec3<T> &a, const Vec3<T> &b);
template<typename T> Vec3<T> operator-(const Vec3<T> &a, const Vec3<T> &b);
template<typename T> Vec3<T> operator*(const Vec3<T> &a, const Vec3<T> &b);
template<typename T> Vec3<T> operator/(const Vec3<T> &a, const Vec3<T> &b);

template<typename T> Vec3<T> operator+(T a, const Vec3<T> &b);
template<typename T> Vec3<T> operator-(T a, const Vec3<T> &b);
template<typename T> Vec3<T> operator*(T a, const Vec3<T> &b);
template<typename T> Vec3<T> operator/(T a, const Vec3<T> &b);

template<typename T> Vec3<T> operator+(const Vec3<T> &a, T b);
template<typename T> Vec3<T> operator-(const Vec3<T> &a, T b);
template<typename T> Vec3<T> operator*(const Vec3<T> &a, T b);
template<typename T> Vec3<T> operator/(const Vec3<T> &a, T b);

// ========================== impl ==========================

template<typename T>
Vec3<T>::Vec3()
    : Vec3(T(0))
{
    
}

template<typename T>
Vec3<T>::Vec3(T v)
    : Vec3(v, v, v)
{
    
}

template<typename T>
Vec3<T>::Vec3(T x, T y, T z)
    : x(x), y(y), z(z)
{
    
}

template<typename T>
float Vec3<T>::operator[](size_t i) const
{
    return *(&x + i);
}

template<typename T>
float &Vec3<T>::operator[](size_t i)
{
    return *(&x + i);
}

template<typename T>
T length_square(const Vec3<T> &v)
{
    return dot(v, v);
}

template<typename T>
T length(const Vec3<T> &v)
{
    static_assert(std::is_floating_point_v<T>);
    return std::sqrt(length_square(v));
}

template<typename T>
Vec3<T> normalize(const Vec3<T> &v)
{
    static_assert(std::is_floating_point_v<T>);
    return T(1) / length(v) * v;
}

template<typename T>
T dot(const Vec3<T> &a, const Vec3<T> &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template<typename T>
Vec3<T> cross(const Vec3<T> &a, const Vec3<T> &b)
{
    return Vec3<T>(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

template<typename T>
Vec3<T> operator+(const Vec3<T> &a, const Vec3<T> &b)
{
    return Vec3<T>(a.x + b.x, a.y + b.y, a.z + b.z);
}

template<typename T>
Vec3<T> operator-(const Vec3<T> &a, const Vec3<T> &b)
{
    return Vec3<T>(a.x - b.x, a.y - b.y, a.z - b.z);
}

template<typename T>
Vec3<T> operator*(const Vec3<T> &a, const Vec3<T> &b)
{
    return Vec3<T>(a.x * b.x, a.y * b.y, a.z * b.z);
}

template<typename T>
Vec3<T> operator/(const Vec3<T> &a, const Vec3<T> &b)
{
    return Vec3<T>(a.x / b.x, a.y / b.y, a.z / b.z);
}

template<typename T>
Vec3<T> operator+(T a, const Vec3<T> &b)
{
    return Vec3<T>(a + b.x, a + b.y, a + b.z);
}

template<typename T>
Vec3<T> operator-(T a, const Vec3<T> &b)
{
    return Vec3<T>(a - b.x, a - b.y, a - b.z);
}

template<typename T>
Vec3<T> operator*(T a, const Vec3<T> &b)
{
    return Vec3<T>(a * b.x, a * b.y, a * b.z);
}

template<typename T>
Vec3<T> operator/(T a, const Vec3<T> &b)
{
    return Vec3<T>(a / b.x, a / b.y, a / b.z);
}

template<typename T>
Vec3<T> operator+(const Vec3<T> &a, T b)
{
    return Vec3<T>(a.x + b, a.y + b, a.z + b);
}

template<typename T>
Vec3<T> operator-(const Vec3<T> &a, T b)
{
    return Vec3<T>(a.x - b, a.y - b, a.z - b);
}

template<typename T>
Vec3<T> operator*(const Vec3<T> &a, T b)
{
    return Vec3<T>(a.x * b, a.y * b, a.z * b);
}

template<typename T>
Vec3<T> operator/(const Vec3<T> &a, T b)
{
    return Vec3<T>(a.x / b, a.y / b, a.z / b);
}

BTRC_END
