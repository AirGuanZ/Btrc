#pragma once

#include <btrc/core/geometry.h>
#include <btrc/core/light.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class MeshLight : public AreaLight
{
public:

    void set_intensity(const Spectrum &intensity);

    void set_geometry(RC<Geometry> geometry, const Transform3D &local_to_world) override;

    CSpectrum eval_le_inline(CompileContext &cc, ref<SurfacePoint> spt, ref<CVec3f> wr) const override;

    SampleLiResult sample_li_inline(CompileContext &cc, ref<CVec3f> ref_pos, ref<Sam3> sam) const override;

    f32 pdf_li_inline(CompileContext &cc, ref<CVec3f> ref_pos, ref<CVec3f> pos, ref<CVec3f> nor) const override;

    SampleEmitResult sample_emit_inline(CompileContext &cc, ref<Sam<5>> sam) const override;

    PdfEmitResult pdf_emit_inline(CompileContext &cc, ref<SurfacePoint> spt, ref<CVec3f> wr) const override;

private:

    BTRC_OBJECT(Geometry, geometry_);
    float scale_ = 1;
    Transform3D local_to_world_;
    Spectrum  intensity_;
};

class MeshLightCreator : public factory::Creator<Light>
{
public:

    std::string get_name() const override { return "mesh_light"; }

    RC<Light> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
