#include <btrc/builtin/texture2d/description.h>
#include <btrc/utils/exception.h>

BTRC_BUILTIN_BEGIN

cuda::Texture::AddressMode string_to_address_mode(std::string_view str)
{
    if(str == "clamp")
        return cuda::Texture::AddressMode::Clamp;
    if(str == "mirror")
        return cuda::Texture::AddressMode::Mirror;
    if(str == "wrap")
        return cuda::Texture::AddressMode::Wrap;
    if(str == "border")
        return cuda::Texture::AddressMode::Border;
    throw BtrcException(fmt::format("unknown address mode: {}", str));
}

cuda::Texture::FilterMode string_to_filter_mode(std::string_view str)
{
    if(str == "point")
        return cuda::Texture::FilterMode::Point;
    if(str == "linear")
        return cuda::Texture::FilterMode::Linear;
    throw BtrcException(fmt::format("unknown filter mode: {}", str));
}

cuda::Texture::Description parse_texture_desc(const RC<const factory::Node> &node)
{
    BTRC_HI_TRY

    auto address_u = node->parse_child_or<std::string>("address_mode_u", "clamp");
    auto address_v = node->parse_child_or<std::string>("address_mode_v", "clamp");
    auto address_w = node->parse_child_or<std::string>("address_mode_w", "clamp");
    auto filter = node->parse_child_or<std::string>("filter", "linear");

    cuda::Texture::Description desc;
    desc.address_modes[0] = string_to_address_mode(address_u);
    desc.address_modes[1] = string_to_address_mode(address_v);
    desc.address_modes[2] = string_to_address_mode(address_w);
    desc.filter_mode = string_to_filter_mode(filter);
    desc.srgb_to_linear = node->parse_child_or("srgb_to_linear", false);

    if(auto n = node->find_child_node("border"))
    {
        if(auto a = n->as_array())
        {
            if(a->get_size() == 1)
            {
                const float v = a->get_element(0)->parse<float>();
                desc.border_value[0] = v;
                desc.border_value[1] = v;
                desc.border_value[2] = v;
                desc.border_value[3] = v;
            }
            else if(a->get_size() == 3)
            {
                desc.border_value[0] = a->get_element(0)->parse<float>();
                desc.border_value[1] = a->get_element(1)->parse<float>();
                desc.border_value[2] = a->get_element(2)->parse<float>();
            }
            else
                throw BtrcException("invalid border array size");
        }
        else if(auto v = n->as_value())
        {
            const float val = v->parse<float>();
            desc.border_value[0] = val;
            desc.border_value[1] = val;
            desc.border_value[2] = val;
            desc.border_value[3] = val;
        }
        else
            throw BtrcException("border: expect array or value node");
    }

    return desc;

    BTRC_HI_WRAP("in parsing texture description")
}

BTRC_BUILTIN_END
