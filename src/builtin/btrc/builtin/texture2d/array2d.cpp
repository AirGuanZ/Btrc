#include <btrc/builtin/texture2d/array2d.h>
#include <btrc/builtin/texture2d/description.h>

BTRC_BUILTIN_BEGIN

void Array2D::initialize(RC<const cuda::Texture> cuda_texture)
{
    tex_ = std::move(cuda_texture);
}

void Array2D::initialize(const std::string &filename, const cuda::Texture::Description &desc)
{
    auto tex = newRC<cuda::Texture>();
    tex->initialize(filename, desc);
    tex_ = std::move(tex);
}

CSpectrum Array2D::sample_spectrum_inline(CompileContext &cc, ref<CVec2f> uv) const
{
    f32 r, g, b;
    cstd::sample_texture2d_3f(u64(tex_->get_tex()), uv.x, uv.y, r, g, b);
    return CSpectrum::from_rgb(r, g, b);
}

f32 Array2D::sample_float_inline(CompileContext &cc, ref<CVec2f> uv) const
{
    f32 r;
    cstd::sample_texture2d_1f(u64(tex_->get_tex()), uv.x, uv.y, r);
    return r;
}

RC<Texture2D> Array2DCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    const auto filename = context.resolve_path(node->parse_child<std::string>("filename")).string();
    const auto desc = parse_texture_desc(node);
    auto result = newRC<Array2D>();
    result->initialize(filename, desc);
    return result;
}

BTRC_BUILTIN_END
