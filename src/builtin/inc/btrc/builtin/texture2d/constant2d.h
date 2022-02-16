#pragma once

#include <btrc/core/texture2d.h>
#include <btrc/factory/context.h>
#include <btrc/utils/variant.h>

BTRC_BUILTIN_BEGIN

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

class Constant2DCreator : public factory::Creator<Texture2D>
{
public:

    std::string get_name() const override { return "constant"; }

    RC<Texture2D> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
