#pragma once

#include <btrc/utils/cuda/texture.h>
#include <btrc/factory/node/node.h>

BTRC_BUILTIN_BEGIN

cuda::Texture::AddressMode string_to_address_mode(std::string_view str);

cuda::Texture::FilterMode string_to_filter_mode(std::string_view str);

cuda::Texture::Description parse_texture_desc(const RC<const factory::Node> &node);

BTRC_BUILTIN_END
