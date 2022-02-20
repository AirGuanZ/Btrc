#pragma once

#include <btrc/core/geometry.h>
#include <btrc/factory/context.h>
#include <btrc/utils/cmath/calias.h>

BTRC_BUILTIN_BEGIN

class TriangleMesh : public Geometry
{
public:

    TriangleMesh(
        optix::Context &optix_ctx,
        const std::string &filename,
        bool transform_to_unit_cube);

    OptixTraversableHandle get_blas() const override;

    const GeometryInfo &get_geometry_info() const override;

    SampleResult sample(ref<CVec3f> sam) const override;

    f32 pdf(ref<CVec3f> pos) const override;

    SampleResult sample(ref<CVec3f> dst_pos, ref<CVec3f> sam) const override;

    f32 pdf(ref<CVec3f> dst_pos, ref<CVec3f> pos) const override;

private:

    cuda::CUDABuffer<Vec4f> geo_info_buf_;
    GeometryInfo            geo_info_ = {};
    optix::TriangleAS       as_;

    // { ax, ay, az, bax, bay, baz, cax, cay, caz } * triangle_count
    cuda::CUDABuffer<float> positions_;

    CAliasTable alias_table_;
    float       total_area_ = 0;
};

class TriangleMeshCreator : public factory::Creator<Geometry>
{
public:

    std::string get_name() const override { return "triangle_mesh"; }

    RC<Geometry> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
