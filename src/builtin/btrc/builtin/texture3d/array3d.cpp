#include <btrc/builtin/texture2d/description.h>
#include <btrc/builtin/texture3d/array3d.h>

BTRC_BUILTIN_BEGIN

void Array3D::initialize(RC<const cuda::Texture> cuda_texture)
{
    tex_ = std::move(cuda_texture);
    set_recompile();
}

void Array3D::initialize_from_text(const std::string &text_filename, const cuda::Texture::Description &desc)
{
    auto arr = newRC<cuda::Array>();
    arr->load_from_text(text_filename);
    auto tex = newRC<cuda::Texture>();
    tex->initialize(std::move(arr), desc);
    initialize(std::move(tex));
}

void Array3D::initialize_from_images(const std::vector<std::string> &image_filenames, const cuda::Texture::Description &desc)
{
    auto arr = newRC<cuda::Array>();
    arr->load_from_images(image_filenames);
    auto tex = newRC<cuda::Texture>();
    tex->initialize(std::move(arr), desc);
    initialize(std::move(tex));
}

CSpectrum Array3D::sample_spectrum_inline(CompileContext &cc, ref<CVec3f> uvw) const
{
    f32 r, g, b;
    cstd::sample_texture3d_3f(u64(tex_->get_tex()), uvw.x, uvw.y, uvw.z, r, g, b);
    return CSpectrum::from_rgb(r, g, b);
}

f32 Array3D::sample_float_inline(CompileContext &cc, ref<CVec3f> uvw) const
{
    f32 r;
    cstd::sample_texture3d_1f(u64(tex_->get_tex()), uvw.x, uvw.y, uvw.z, r);
    return r;
}

CSpectrum Array3D::get_max_spectrum(CompileContext &cc) const
{
    auto v = tex_->get_max_value();
    return CSpectrum::from_rgb(v.x, v.y, v.z);
}

CSpectrum Array3D::get_min_spectrum(CompileContext &cc) const
{
    auto v = tex_->get_min_value();
    return CSpectrum::from_rgb(v.x, v.y, v.z);
}

RC<Texture3D> Array3DCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    const auto text_filename = context.resolve_path(node->parse_child<std::string>("text"));
    const auto desc = parse_texture_desc(node);
    auto result = newRC<Array3D>();
    result->initialize_from_text(text_filename.string(), desc);
    return result;
}

BTRC_BUILTIN_END
