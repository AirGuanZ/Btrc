#pragma once

#include <btrc/core/light/light.h>

BTRC_CORE_BEGIN

class GradientSky : public EnvirLight
{
    Spectrum lower_ = Spectrum::from_rgb(0, 0, 0);
    Spectrum upper_ = Spectrum::from_rgb(1, 1, 1);
    Vec3f    up_    = { 0, 0, 1 };

public:

    using EnvirLight::EnvirLight;

    void set_lower(const Spectrum &lower);

    void set_upper(const Spectrum &upper);

    void set_up(const Vec3f &up);

    CSpectrum eval_le_inline(ref<CVec3f> to_light) const override;

    SampleLiResult sample_li_inline(ref<CVec3f> sam) const override;

    f32 pdf_li_inline(ref<CVec3f> to_light) const override;
};

BTRC_CORE_END
