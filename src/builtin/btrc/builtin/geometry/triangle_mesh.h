#pragma once

#include <btrc/core/geometry.h>
#include <btrc/factory/context.h>
#include <btrc/utils/cmath/calias.h>

BTRC_BUILTIN_BEGIN

class TriangleMesh : public Geometry
{
public:

    void set_optix_context(optix::Context &optix_ctx);

    void set_filename(std::string filename);

    void set_transform_to_unit_cube(bool transform);

    void commit() override;

    OptixTraversableHandle get_blas() const override;

    const GeometryInfo &get_geometry_info() const override;

    AABB3f get_bounding_box() const override;

    SampleResult sample_inline(ref<Sam3> sam) const override;

    f32 pdf_inline(ref<CVec3f> pos) const override;

    SampleResult sample_inline(ref<CVec3f> dst_pos, ref<Sam3> sam) const override;

    f32 pdf_inline(ref<CVec3f> dst_pos, ref<CVec3f> pos) const override;

private:

    optix::Context *optix_ctx_ = nullptr;
    std::string filename_;
    bool transform_to_unit_cube_ = false;

    cuda::Buffer<Vec4f> geo_info_buf_;
    GeometryInfo        geo_info_ = {};
    optix::TriangleAS   as_;

    // { ax, ay, az, bax, bay, baz, cax, cay, caz } * triangle_count
    cuda::Buffer<float> positions_;

    CAliasTable alias_table_;
    float       total_area_ = 0;
    AABB3f      bbox_;
};

class TriangleMeshCreator : public factory::Creator<Geometry>
{
public:

    std::string get_name() const override { return "triangle_mesh"; }

    RC<Geometry> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
