#include <btrc/utils/math/mat4.h>
#include <btrc/utils/math/transform.h>

BTRC_BEGIN

Transform3D::Transform3D(const Mat4 &mat)
    : mat(mat), inv(mat.inverse())
{
    
}

Transform3D::Transform3D(const Mat4 &mat, const Mat4 &inv)
    : mat(mat), inv(inv)
{
    
}

Transform3D Transform3D::inverse() const
{
    return Transform3D(inv, mat);
}

Vec3f Transform3D::apply_to_point(const Vec3f &p) const
{
    return Vec3f(
        mat.at(0, 0) * p.x + mat.at(0, 1) * p.y + mat.at(0, 2) * p.z + mat.at(0, 3),
        mat.at(1, 0) * p.x + mat.at(1, 1) * p.y + mat.at(1, 2) * p.z + mat.at(1, 3),
        mat.at(2, 0) * p.x + mat.at(2, 1) * p.y + mat.at(2, 2) * p.z + mat.at(2, 3));
}

AABB3f Transform3D::apply_to_aabb(const AABB3f &bbox) const
{
    AABB3f result;
    for(auto &p : {
        Vec3f(bbox.lower.x, bbox.lower.y, bbox.lower.z),
        Vec3f(bbox.lower.x, bbox.lower.y, bbox.upper.z),
        Vec3f(bbox.lower.x, bbox.upper.y, bbox.lower.z),
        Vec3f(bbox.lower.x, bbox.upper.y, bbox.upper.z),
        Vec3f(bbox.upper.x, bbox.lower.y, bbox.lower.z),
        Vec3f(bbox.upper.x, bbox.lower.y, bbox.upper.z),
        Vec3f(bbox.upper.x, bbox.upper.y, bbox.lower.z),
        Vec3f(bbox.upper.x, bbox.upper.y, bbox.upper.z) })
        result = union_aabb(result, apply_to_point(p));
    return result;
}

Transform3D Transform3D::translate(float x, float y, float z)
{
    return Transform3D(Mat4::translate(x, y, z), Mat4::translate(-x, -y, -z));
}

Transform3D Transform3D::rotate(const Vec3f &axis, float rad)
{
    return Transform3D(Mat4::rotate(axis, rad), Mat4::rotate(axis, -rad));
}

Transform3D Transform3D::rotate_x(float rad)
{
    return Transform3D(Mat4::rotate_x(rad), Mat4::rotate_x(-rad));
}

Transform3D Transform3D::rotate_y(float rad)
{
    return Transform3D(Mat4::rotate_y(rad), Mat4::rotate_y(-rad));
}

Transform3D Transform3D::rotate_z(float rad)
{
    return Transform3D(Mat4::rotate_z(-rad), Mat4::rotate_z(-rad));
}

Transform3D Transform3D::scale(float x, float y, float z)
{
    return Transform3D(Mat4::scale(x, y, z), Mat4::scale(1 / x, 1 / y, 1 / z));
}

Transform3D operator*(const Transform3D &a, const Transform3D &b)
{
    return Transform3D(a.mat * b.mat, b.inv * a.inv);
}

BTRC_END
