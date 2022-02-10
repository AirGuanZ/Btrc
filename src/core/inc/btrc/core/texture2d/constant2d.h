#pragma once

#include <btrc/core/texture2d/texture2d.h>
#include <btrc/core/utils/variant.h>

BTRC_CORE_BEGIN

class Constant2D : public Texture2D
{
    Variant<float, Spectrum> value_;

public:

    Constant2D();

    void set_value(float value);

    void set_value(const Spectrum &value);

    f32 sample_float_inline(ref<CVec2f> uv) const override;

    CSpectrum sample_spectrum_inline(ref<CVec2f> uv) const override;
};

BTRC_CORE_END
