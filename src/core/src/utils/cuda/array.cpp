#include <cassert>

#include <btrc/core/utils/cuda/array.h>
#include <btrc/core/utils/cuda/error.h>
#include <btrc/core/utils/scope_guard.h>
#include <btrc/core/utils/unreachable.h>

BTRC_CORE_BEGIN

namespace
{

    struct FormatDesc
    {
        cudaChannelFormatDesc channel;
        size_t                width_bytes;
    };

    FormatDesc get_format_desc(Array::Format format, size_t width)
    {
        switch(format)
        {
        case Array::Format::S8x1:  return { cudaCreateChannelDesc<char1>(),   1 * width * 1 };
        case Array::Format::S8x2:  return { cudaCreateChannelDesc<char2>(),   1 * width * 2 };
        case Array::Format::S8x4:  return { cudaCreateChannelDesc<char4>(),   1 * width * 4 };
        case Array::Format::U8x1:  return { cudaCreateChannelDesc<uchar1>(),  1 * width * 1 };
        case Array::Format::U8x2:  return { cudaCreateChannelDesc<uchar2>(),  1 * width * 2 };
        case Array::Format::U8x4:  return { cudaCreateChannelDesc<uchar4>(),  1 * width * 4 };
        case Array::Format::S16x1: return { cudaCreateChannelDesc<short1>(),  2 * width * 1 };
        case Array::Format::S16x2: return { cudaCreateChannelDesc<short2>(),  2 * width * 2 };
        case Array::Format::S16x4: return { cudaCreateChannelDesc<short4>(),  2 * width * 4 };
        case Array::Format::U16x1: return { cudaCreateChannelDesc<ushort1>(), 2 * width * 1 };
        case Array::Format::U16x2: return { cudaCreateChannelDesc<ushort2>(), 2 * width * 2 };
        case Array::Format::U16x4: return { cudaCreateChannelDesc<ushort4>(), 2 * width * 4 };
        case Array::Format::F16x1: return { cudaCreateChannelDescHalf1(),     2 * width * 1 };
        case Array::Format::F16x2: return { cudaCreateChannelDescHalf2(),     2 * width * 2 };
        case Array::Format::F16x4: return { cudaCreateChannelDescHalf4(),     2 * width * 4 };
        case Array::Format::S32x1: return { cudaCreateChannelDesc<int1>(),    4 * width * 1 };
        case Array::Format::S32x2: return { cudaCreateChannelDesc<int2>(),    4 * width * 2 };
        case Array::Format::S32x4: return { cudaCreateChannelDesc<int4>(),    4 * width * 4 };
        case Array::Format::U32x1: return { cudaCreateChannelDesc<uint1>(),   4 * width * 1 };
        case Array::Format::U32x2: return { cudaCreateChannelDesc<uint2>(),   4 * width * 2 };
        case Array::Format::U32x4: return { cudaCreateChannelDesc<uint4>(),   4 * width * 4 };
        case Array::Format::F32x1: return { cudaCreateChannelDesc<float1>(),  4 * width * 1 };
        case Array::Format::F32x2: return { cudaCreateChannelDesc<float2>(),  4 * width * 2 };
        case Array::Format::F32x4: return { cudaCreateChannelDesc<float4>(),  4 * width * 4 };
        case Array::Format::UNorm8x1:  return { cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized8X1>(),  1 * width * 1 };
        case Array::Format::UNorm8x2:  return { cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized8X2>(),  1 * width * 2 };
        case Array::Format::UNorm8x4:  return { cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized8X4>(),  1 * width * 4 };
        case Array::Format::SNorm8x1:  return { cudaCreateChannelDesc<cudaChannelFormatKindSignedNormalized8X1>(),    1 * width * 1 };
        case Array::Format::SNorm8x2:  return { cudaCreateChannelDesc<cudaChannelFormatKindSignedNormalized8X2>(),    1 * width * 2 };
        case Array::Format::SNorm8x4:  return { cudaCreateChannelDesc<cudaChannelFormatKindSignedNormalized8X4>(),    1 * width * 4 };
        case Array::Format::UNorm16x1: return { cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized16X1>(), 2 * width * 1 };
        case Array::Format::UNorm16x2: return { cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized16X2>(), 2 * width * 2 };
        case Array::Format::UNorm16x4: return { cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized16X4>(), 2 * width * 4 };
        case Array::Format::SNorm16x1: return { cudaCreateChannelDesc<cudaChannelFormatKindSignedNormalized16X1>(),   2 * width * 1 };
        case Array::Format::SNorm16x2: return { cudaCreateChannelDesc<cudaChannelFormatKindSignedNormalized16X2>(),   2 * width * 2 };
        case Array::Format::SNorm16x4: return { cudaCreateChannelDesc<cudaChannelFormatKindSignedNormalized16X4>(),   2 * width * 4 };
        }
        unreachable();
    }

} // namespace anonymous

Array::Array()
    : width_(0), height_(0), format_{}, arr_(nullptr)
{
    
}

Array::~Array()
{
    if(arr_)
        cudaFreeArray(arr_);
}

void Array::load_from_memory(
    int         width,
    int         height,
    Format      format,
    const void *linear_data)
{
    assert(linear_data);

    cudaArray_t new_arr = nullptr;
    BTRC_SCOPE_SUCCESS
    {
        if(arr_)
            cudaFreeArray(arr_);
        width_ = width;
        height_ = height;
        format_ = format;
        arr_ = new_arr;
    };
    BTRC_SCOPE_FAIL
    {
        if(new_arr) cudaFreeArray(new_arr);
    };

    const auto &format_desc = get_format_desc(format, width);

    throw_on_error(cudaMallocArray(&new_arr, &format_desc.channel, width, height));
    throw_on_error(cudaMemcpy2DToArray(
        new_arr, 0, 0, linear_data,
        format_desc.width_bytes, format_desc.width_bytes, height,
        cudaMemcpyHostToDevice));
}

void Array::load_from_memory(const Image<uint8_t> &image)
{
    load_from_memory(image.width(), image.height(), Format::UNorm8x1, image.data());
}

void Array::load_from_memory(const Image<Vec3b> &image)
{
    load_from_memory(image.to<Vec4b>());
}

void Array::load_from_memory(const Image<Vec4b> &image)
{
    load_from_memory(image.width(), image.height(), Format::UNorm8x4, image.data());
}

void Array::load_from_memory(const Image<float> &image)
{
    load_from_memory(image.width(), image.height(), Format::F32x1, image.data());
}

void Array::load_from_memory(const Image<Vec3f> &image)
{
    load_from_memory(image.to<Vec4f>());
}

void Array::load_from_memory(const Image<Vec4f> &image)
{
    load_from_memory(image.width(), image.height(), Format::F32x4, image.data());
}

void Array::load_from_image(const std::string &filename)
{
    auto image_dynamic = ImageDynamic::load(filename);
    image_dynamic.match(
        [](std::monostate) { unreachable(); },
        [&](auto &image) { this->load_from_memory(image); });
}

cudaArray_t Array::get_arr() const
{
    return arr_;
}

int Array::get_width() const
{
    return width_;
}

int Array::get_height() const
{
    return height_;
}

Array::Format Array::get_format() const
{
    return format_;
}

BTRC_CORE_END
