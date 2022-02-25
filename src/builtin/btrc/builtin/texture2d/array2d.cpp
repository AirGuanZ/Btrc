#include <btrc/builtin/texture2d/array2d.h>

BTRC_BUILTIN_BEGIN

namespace
{
    Texture::AddressMode string_to_address_mode(std::string_view str)
    {
        if(str == "clamp")
            return Texture::AddressMode::Clamp;
        if(str == "mirror")
            return Texture::AddressMode::Mirror;
        if(str == "wrap")
            return Texture::AddressMode::Wrap;
        if(str == "border")
            return Texture::AddressMode::Border;
        throw BtrcException(fmt::format("unknown address mode: {}", str));
    }

    Texture::FilterMode string_to_filter_mode(std::string_view str)
    {
        if(str == "point")
            return Texture::FilterMode::Point;
        if(str == "linear")
            return Texture::FilterMode::Linear;
        throw BtrcException(fmt::format("unknown filter mode: {}", str));
    }

    Texture::Description parse_texture2d_desc(const RC<const factory::Node> &node)
    {
        auto address_u = node->parse_child_or<std::string>("address_mode_u", "clamp");
        auto address_v = node->parse_child_or<std::string>("address_mode_v", "clamp");
        auto filter = node->parse_child_or<std::string>("filter", "linear");

        Texture::Description desc;
        desc.address_modes[0] = string_to_address_mode(address_u);
        desc.address_modes[1] = string_to_address_mode(address_v);
        desc.address_modes[2] = desc.address_modes[0];
        desc.filter_mode = string_to_filter_mode(filter);
        desc.srgb_to_linear = node->parse_child_or("srgb_to_linear", false);

        return desc;
    }
}

void Array2D::initialize(RC<const Texture> cuda_texture)
{
    tex_ = std::move(cuda_texture);
    set_recompile();
}

void Array2D::initialize(const std::string &filename, const Texture::Description &desc)
{
    auto tex = newRC<Texture>();
    tex->initialize(filename, desc);
    tex_ = std::move(tex);
    set_recompile();
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
    const auto desc = parse_texture2d_desc(node);
    auto result = newRC<Array2D>();
    result->initialize(filename, desc);
    return result;
}

BTRC_BUILTIN_END
