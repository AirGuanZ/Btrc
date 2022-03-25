#pragma once

#include <btrc/utils/math/aabb.h>
#include <btrc/utils/math/mat3.h>
#include <btrc/utils/math/mat4.h>
#include <btrc/utils/math/vec2.h>
#include <btrc/utils/math/vec3.h>

BTRC_BEGIN

struct Transform2D
{
    Mat3 mat;
    Mat3 inv;

    Transform2D() = default;

    explicit Transform2D(const Mat3 &mat);

    Transform2D(const Mat3 &mat, const Mat3 &inv);

    Transform2D inverse() const;

    Vec2f apply_to_point(const Vec2f &p) const;

    static Transform2D translate(float x, float y);

    static Transform2D rotate(float rad);

    static Transform2D scale(float x, float y);
};

Transform2D operator*(const Transform2D &a, const Transform2D &b);

struct Transform3D
{
    Mat4 mat;
    Mat4 inv;

    Transform3D() = default;

    explicit Transform3D(const Mat4 &mat);

    Transform3D(const Mat4 &mat, const Mat4 &inv);

    Transform3D inverse() const;

    Vec3f apply_to_point(const Vec3f &p) const;

    AABB3f apply_to_aabb(const AABB3f &bbox) const;

    static Transform3D translate(float x, float y, float z);

    static Transform3D rotate(const Vec3f &axis, float rad);

    static Transform3D rotate_x(float rad);

    static Transform3D rotate_y(float rad);

    static Transform3D rotate_z(float rad);

    static Transform3D scale(float x, float y, float z);
};

Transform3D operator*(const Transform3D &a, const Transform3D &b);

BTRC_END
