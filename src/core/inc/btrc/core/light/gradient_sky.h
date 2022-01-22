#pragma once

#include <btrc/core/light/light.h>

BTRC_CORE_BEGIN

class GradientSky : public EnvirLight
{
    Spectrum lower_;
    Spectrum upper_;
    Vec3f    up_;

public:

    GradientSky();

    void set_lower(const Spectrum &lower);

    void set_upper(const Spectrum &upper);

    void set_up(const Vec3f &up);

    CSpectrum eval_le(const CVec3f &to_light) const override;

    SampleLiResult sample_li(const CVec3f &sam) const override;

    f32 pdf_li(const CVec3f &to_light) const override;
};

BTRC_CORE_END
