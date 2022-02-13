#include <btrc/core/utils/cuda/error.h>
#include <btrc/core/utils/cuda/texture.h>
#include <btrc/core/utils/scope_guard.h>
#include <btrc/core/utils/unreachable.h>

BTRC_CORE_BEGIN

namespace
{

    cudaTextureAddressMode to_cuda_address_mode(Texture::AddressMode mode)
    {
        switch(mode)
        {
        case Texture::AddressMode::Wrap:
            return cudaAddressModeWrap;
        case Texture::AddressMode::Clamp:
            return cudaAddressModeClamp;
        case Texture::AddressMode::Mirror:
            return cudaAddressModeMirror;
        case Texture::AddressMode::Border:
            return cudaAddressModeBorder;
        }
        unreachable();
    }

} // namespace anonymous

Texture::Texture()
    : tex_(0)
{

}

Texture::Texture(Texture &&other) noexcept
    : Texture()
{
    swap(other);
}

Texture &Texture::operator=(Texture &&other) noexcept
{
    swap(other);
    return *this;
}

Texture::~Texture()
{
    destroy();
}

void Texture::swap(Texture &other) noexcept
{
    std::swap(arr_, other.arr_);
    std::swap(tex_, other.tex_);
}

Texture::operator bool() const
{
    return arr_ != nullptr;
}

void Texture::initialize(RC<const Array> arr, const Description &desc)
{
    destroy();
    arr_ = std::move(arr);
    BTRC_SCOPE_FAIL{ arr_ = {}; tex_ = 0; };

    cudaTextureDesc cu_desc = {};
    cu_desc.addressMode[0] = to_cuda_address_mode(desc.address_modes[0]);
    cu_desc.addressMode[1] = to_cuda_address_mode(desc.address_modes[1]);
    cu_desc.addressMode[2] = to_cuda_address_mode(desc.address_modes[2]);
    cu_desc.filterMode =
        desc.filter_mode == FilterMode::Point ?
        cudaFilterModePoint : cudaFilterModeLinear;
    cu_desc.normalizedCoords = 1;

    const auto format = arr_->get_format();
    if(format == Array::Format::UNorm8x1 ||
       format == Array::Format::UNorm8x2 ||
       format == Array::Format::UNorm8x4 ||
       format == Array::Format::SNorm8x1 ||
       format == Array::Format::SNorm8x2 ||
       format == Array::Format::SNorm8x4 ||
       format == Array::Format::UNorm16x1 ||
       format == Array::Format::UNorm16x2 ||
       format == Array::Format::UNorm16x4 ||
       format == Array::Format::SNorm16x1 ||
       format == Array::Format::SNorm16x2 ||
       format == Array::Format::SNorm16x4)
        cu_desc.readMode = cudaReadModeNormalizedFloat;
    else
        cu_desc.readMode = cudaReadModeElementType;

    cudaResourceDesc rsc;
    rsc.resType = cudaResourceTypeArray;
    rsc.res.array.array = arr_->get_arr();

    throw_on_error(cudaCreateTextureObject(&tex_, &rsc, &cu_desc, nullptr));
}

void Texture::initialize(const std::string &filename, const Description &desc)
{
    auto arr = newRC<Array>();
    arr->load_from_image(filename);
    initialize(std::move(arr), desc);
}

cudaTextureObject_t Texture::get_tex() const
{
    return tex_;
}

void Texture::destroy()
{
    if(tex_)
    {
        cudaDestroyTextureObject(tex_);
        tex_ = 0;
    }
    arr_ = {};
}

BTRC_CORE_END
