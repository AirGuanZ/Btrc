#include <btrc/utils/cmath/ctransform.h>

BTRC_BEGIN

CTransform2D::CTransform2D(const Transform2D &t)
{
    mat = CMat3(t.mat);
    inv = CMat3(t.inv);
}

CTransform2D::CTransform2D(const CMat3 &_mat, const CMat3 &_inv)
{
    mat = _mat;
    inv = _inv;
}

CVec2f CTransform2D::apply_to_point(const CVec2f &p) const
{
    return CVec2f(
        mat.at(0, 0) * p.x + mat.at(0, 1) * p.y + mat.at(0, 2),
        mat.at(1, 0) * p.x + mat.at(1, 1) * p.y + mat.at(1, 2));
}

CTransform3D::CTransform3D(const Transform3D &t)
{
    mat = CMat4(t.mat);
    inv = CMat4(t.inv);
}

CTransform3D::CTransform3D(const CMat4 &_mat, const CMat4 &_inv)
{
    mat = _mat;
    inv = _inv;
}

CVec3f CTransform3D::apply_to_point(const CVec3f &p) const
{
    return CVec3f(
        mat.at(0, 0) * p.x + mat.at(0, 1) * p.y + mat.at(0, 2) * p.z + mat.at(0, 3),
        mat.at(1, 0) * p.x + mat.at(1, 1) * p.y + mat.at(1, 2) * p.z + mat.at(1, 3),
        mat.at(2, 0) * p.x + mat.at(2, 1) * p.y + mat.at(2, 2) * p.z + mat.at(2, 3));
}

CVec3f CTransform3D::apply_to_vector(const CVec3f &v) const
{
    return CVec3f(
        mat.at(0, 0) * v.x + mat.at(0, 1) * v.y + mat.at(0, 2) * v.z,
        mat.at(1, 0) * v.x + mat.at(1, 1) * v.y + mat.at(1, 2) * v.z,
        mat.at(2, 0) * v.x + mat.at(2, 1) * v.y + mat.at(2, 2) * v.z);
}

CVec3f CTransform3D::apply_to_normal(const CVec3f &n) const
{
    return CVec3f(
        inv.at(0, 0) * n.x + inv.at(1, 0) * n.y + inv.at(2, 0) * n.z,
        inv.at(0, 1) * n.x + inv.at(1, 1) * n.y + inv.at(2, 1) * n.z,
        inv.at(0, 2) * n.x + inv.at(1, 2) * n.y + inv.at(2, 2) * n.z);
}

BTRC_END
