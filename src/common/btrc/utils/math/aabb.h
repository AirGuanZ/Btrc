#pragma once

#include <btrc/utils/math/vec3.h>

BTRC_BEGIN

template<typename T>
struct AABB3
{
    Vec3<T> lower = Vec3<T>((std::numeric_limits<T>::max)());
    Vec3<T> upper = Vec3<T>(std::numeric_limits<T>::lowest());

    AABB3() = default;

    AABB3(const Vec3<T> &lower, const Vec3<T> &upper);

    bool empty() const;
};

using AABB3f = AABB3<float>;

template<typename T>
AABB3<T> union_aabb(const AABB3<T> &a, const AABB3<T> &b);

template<typename T>
AABB3<T> intersect_aabb(const AABB3<T> &a, const AABB3<T> &b);

// ========================== impl ==========================

template<typename T>
AABB3<T>::AABB3(const Vec3<T> &lower, const Vec3<T> &upper)
    : lower(lower), upper(upper)
{
    
}

template<typename T>
AABB3<T> union_aabb(const AABB3<T> &a, const AABB3<T> &b)
{
    return AABB3<T>(
        Vec3<T>((std::min)(a.lower.x, b.lower.x),
                (std::min)(a.lower.y, b.lower.y), 
                (std::min)(a.lower.z, b.lower.z)),
        Vec3<T>((std::max)(a.upper.x, b.upper.x),
                (std::max)(a.upper.y, b.upper.y), 
                (std::max)(a.upper.z, b.upper.z)));
}

template<typename T>
AABB3<T> intersect_aabb(const AABB3<T> &a, const AABB3<T> &b)
{
    return AABB3<T>(
        Vec3<T>((std::max)(a.lower.x, b.lower.x),
                (std::max)(a.lower.y, b.lower.y), 
                (std::max)(a.lower.z, b.lower.z)),
        Vec3<T>((std::min)(a.upper.x, b.upper.x),
                (std::min)(a.upper.y, b.upper.y), 
                (std::min)(a.upper.z, b.upper.z)));
}

template<typename T>
bool AABB3<T>::empty() const
{
    return lower.x < upper.x && lower.y < upper.y && lower.z < upper.z;
}

BTRC_END
