#pragma once

#include <btrc/core/context.h>
#include <btrc/core/surface_point.h>
#include <btrc/utils/cmath/cmath.h>
#include <btrc/utils/optix/context.h>

BTRC_BEGIN

struct GeometryInfo
{
    Vec4f *geometry_ex_tex_coord_u_a;
    Vec4f *geometry_ey_tex_coord_u_ba;
    Vec4f *geometry_ez_tex_coord_u_ca;

    Vec4f *shading_normal_tex_coord_v_a;
    Vec4f *shading_normal_tex_coord_v_ba;
    Vec4f *shading_normal_tex_coord_v_ca;
};

CUJ_PROXY_CLASS(
    CGeometryInfo,
    GeometryInfo,
    geometry_ex_tex_coord_u_a,
    geometry_ey_tex_coord_u_ba,
    geometry_ez_tex_coord_u_ca,
    shading_normal_tex_coord_v_a,
    shading_normal_tex_coord_v_ba,
    shading_normal_tex_coord_v_ca);

class Geometry : public Object
{
public:

    CUJ_CLASS_BEGIN(SampleResult)
        CUJ_MEMBER_VARIABLE(SurfacePoint, point)
        CUJ_MEMBER_VARIABLE(f32,          pdf)
    CUJ_CLASS_END

    virtual OptixTraversableHandle get_blas() const = 0;

    virtual const GeometryInfo &get_geometry_info() const = 0;

    virtual SampleResult sample(ref<CVec3f> sam) const = 0;

    virtual f32 pdf(ref<CVec3f> pos) const = 0;

    virtual SampleResult sample(ref<CVec3f> dst_pos, ref<CVec3f> sam) const = 0;

    virtual f32 pdf(ref<CVec3f> dst_pos, ref<CVec3f> pos) const = 0;
};

BTRC_END
