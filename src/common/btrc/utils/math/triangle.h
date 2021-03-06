#pragma once

#include <btrc/utils/math/vec2.h>
#include <btrc/utils/math/frame.h>

BTRC_BEGIN

inline Vec3f triangle_dpdu(
    const Vec3f &B_A,
    const Vec3f &C_A,
    const Vec2f &b_a,
    const Vec2f &c_a,
    const Vec3f &nor)
{
    const float m00 = b_a.x, m01 = b_a.y;
    const float m10 = c_a.x, m11 = c_a.y;
    const float det = m00 * m11 - m01 * m10;
    if(det == 0.0f)
        return Frame::from_z(nor).x;
    const float inv_det = 1 / det;
    return normalize(m11 * inv_det * B_A - m01 * inv_det * C_A);
}

inline float triangle_area(const Vec3f &b_a, const Vec3f &c_a) noexcept
{
    return 0.5f * length(cross(b_a, c_a));
}

inline float triangle_area(const Vec3f &a, const Vec3f &b, const Vec3f &c)
{
    return triangle_area(b - a, c - a);
}

BTRC_END
