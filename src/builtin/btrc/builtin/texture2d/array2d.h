#pragma once

#include <btrc/core/texture2d.h>
#include <btrc/factory/context.h>
#include <btrc/utils/cuda/texture.h>

BTRC_BUILTIN_BEGIN

class Array2D : public Texture2D, public Uncopyable
{
public:

    void initialize(RC<const Texture> cuda_texture);

    void initialize(const std::string &filename, const Texture::Description &desc);

    CSpectrum sample_spectrum_inline(ref<CVec2f> uv) const override;

    f32 sample_float_inline(ref<CVec2f> uv) const override;

private:

    RC<const Texture> tex_;
};

class Array2DCreator : public factory::Creator<Texture2D>
{
public:

    std::string get_name() const override { return "array"; }

    RC<Texture2D> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
