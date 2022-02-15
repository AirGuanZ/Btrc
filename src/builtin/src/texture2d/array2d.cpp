#include <btrc/builtin/texture2d/array2d.h>

BTRC_BUILTIN_BEGIN

void Array2D::initialize(RC<const Texture> cuda_texture)
{
    tex_ = std::move(cuda_texture);
}

void Array2D::initialize(const std::string &filename, const Texture::Description &desc)
{
    auto tex = newRC<Texture>();
    tex->initialize(filename, desc);
    tex_ = std::move(tex);
}

CSpectrum Array2D::sample_spectrum_inline(ref<CVec2f> uv) const
{
    f32 r, g, b;
    cstd::sample_texture2d_3f(u64(tex_->get_tex()), uv.x, uv.y, r, g, b);
    return CSpectrum::from_rgb(r, g, b);
}

f32 Array2D::sample_float_inline(ref<CVec2f> uv) const
{
    f32 r;
    cstd::sample_texture2d_1f(u64(tex_->get_tex()), uv.x, uv.y, r);
    return r;
}

BTRC_BUILTIN_END
