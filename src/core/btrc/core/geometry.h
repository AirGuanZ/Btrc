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

    virtual SampleResult sample_inline(ref<CVec3f> sam) const = 0;

    virtual f32 pdf_inline(ref<CVec3f> pos) const = 0;

    virtual SampleResult sample_inline(ref<CVec3f> dst_pos, ref<CVec3f> sam) const = 0;

    virtual f32 pdf_inline(ref<CVec3f> dst_pos, ref<CVec3f> pos) const = 0;

    SampleResult sample(CompileContext &cc, ref<CVec3f> sam) const
    {
        auto action = [this](ref<CVec3f> _sam) { return sample_inline(_sam); };
        return cc.record_object_action(as_shared(), "sample", action, sam);
    }

    f32 pdf(CompileContext &cc, ref<CVec3f> pos) const
    {
        auto action = [this](ref<CVec3f> _pos) { return pdf_inline(_pos); };
        return cc.record_object_action(as_shared(), "pdf", action, pos);
    }

    SampleResult sample(CompileContext &cc, ref<CVec3f> dst_pos, ref<CVec3f> sam) const
    {
        auto action = [this](ref<CVec3f> dst_pos, ref<CVec3f> sam) { return sample_inline(dst_pos, sam); };
        return cc.record_object_action(as_shared(), "sample_dst", action, dst_pos, sam);
    }

    f32 pdf(CompileContext &cc, ref<CVec3f> dst_pos, ref<CVec3f> pos) const
    {
        auto action = [this](ref<CVec3f> dst_pos, ref<CVec3f> pos) { return pdf_inline(dst_pos, pos); };
        return cc.record_object_action(as_shared(), "pdf_dst", action, dst_pos, pos);
    }
};

BTRC_END
