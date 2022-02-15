#pragma once

#include <btrc/core/geometry.h>
#include <btrc/core/light.h>

BTRC_BUILTIN_BEGIN

class MeshLight : public AreaLight
{
public:

    explicit MeshLight(
        RC<const Geometry> geometry,
        const Transform   &local_to_world,
        const Spectrum    &intensity);

    CSpectrum eval_le_inline(
        ref<CVec3f> pos,
        ref<CVec3f> nor,
        ref<CVec2f> uv,
        ref<CVec2f> tex_coord,
        ref<CVec3f> wr) const override;

    SampleLiResult sample_li_inline(
        ref<CVec3f> ref_pos,
        ref<CVec3f> sam) const override;

    f32 pdf_li_inline(
        ref<CVec3f> ref_pos,
        ref<CVec3f> pos,
        ref<CVec3f> nor) const override;

private:

    RC<const Geometry> geometry_;
    Transform          local_to_world_;
    Spectrum           intensity_;
};

BTRC_BUILTIN_END
