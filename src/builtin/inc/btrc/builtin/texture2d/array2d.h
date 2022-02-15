#pragma once

#include <btrc/core/texture2d.h>
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

BTRC_BUILTIN_END
