#include <btrc/builtin/texture2d/constant2d.h>

BTRC_BUILTIN_BEGIN

Constant2D::Constant2D()
{
    set_value(0.0f);
}

void Constant2D::set_value(float value)
{
    value_ = value;
}

void Constant2D::set_value(const Spectrum &value)
{
    value_ = value;
}

CSpectrum Constant2D::sample_spectrum_inline(ref<CVec2f> uv) const
{
    return value_.match(
        [](float v) { return CSpectrum::from_rgb(v, v, v); },
        [](const Spectrum &v) { return CSpectrum(v); });
}

f32 Constant2D::sample_float_inline(ref<CVec2f> uv) const
{
    return value_.match(
        [](float v) { return f32(v); },
        [](const Spectrum &v) { return f32(v.get_lum()); });
}

RC<Texture2D> Constant2DCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    auto value = node->parse_child<Spectrum>("value");
    auto result = newRC<Constant2D>();
    result->set_value(value);
    return result;
}

BTRC_BUILTIN_END
