#include <cassert>
#include <fstream>

#include <btrc/utils/cuda/array.h>
#include <btrc/utils/cuda/error.h>
#include <btrc/utils/scope_guard.h>
#include <btrc/utils/unreachable.h>

BTRC_CUDA_BEGIN

namespace
{

    struct FormatDesc
    {
        cudaChannelFormatDesc channel;
        size_t                width_bytes;
    };

    template<typename T>
    struct ImageToFormatAux;

#define ADD_IMAGE_TO_FORMAT(TEXEL, DST_TEXEL, FORMAT)                  \
    template<>                                                         \
    struct ImageToFormatAux<Image<TEXEL>>                              \
    {                                                                  \
        using DstTexel = DST_TEXEL;                                    \
        static constexpr Array::Format format = Array::Format::FORMAT; \
    };

    ADD_IMAGE_TO_FORMAT(uint8_t, uint8_t, SNorm8x1)
    ADD_IMAGE_TO_FORMAT(Vec3b,   Vec4b,   SNorm8x4)
    ADD_IMAGE_TO_FORMAT(Vec4b,   Vec4b,   SNorm8x4)
    ADD_IMAGE_TO_FORMAT(float,   float,   F32x1)
    ADD_IMAGE_TO_FORMAT(Vec3f,   Vec4f,   F32x4)
    ADD_IMAGE_TO_FORMAT(Vec4f,   Vec4f,   F32x4)

#undef ADD_IMAGE_TO_FORMAT

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
        case Array::Format::UNorm8x1:  return { cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned),  1 * width * 1 };
        case Array::Format::UNorm8x2:  return { cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned),  1 * width * 2 };
        case Array::Format::UNorm8x4:  return { cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned),  1 * width * 4 };
        case Array::Format::SNorm8x1:  return { cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindSigned),    1 * width * 1 };
        case Array::Format::SNorm8x2:  return { cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindSigned),    1 * width * 2 };
        case Array::Format::SNorm8x4:  return { cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindSigned),    1 * width * 4 };
        case Array::Format::UNorm16x1: return { cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned), 2 * width * 1 };
        case Array::Format::UNorm16x2: return { cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindUnsigned), 2 * width * 2 };
        case Array::Format::UNorm16x4: return { cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned), 2 * width * 4 };
        case Array::Format::SNorm16x1: return { cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindSigned),   2 * width * 1 };
        case Array::Format::SNorm16x2: return { cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindSigned),   2 * width * 2 };
        case Array::Format::SNorm16x4: return { cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindSigned),   2 * width * 4 };
        case Array::Format::UNorm32x1: return { cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned), 4 * width * 1 };
        case Array::Format::UNorm32x2: return { cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindUnsigned), 4 * width * 2 };
        case Array::Format::UNorm32x4: return { cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned), 4 * width * 4 };
        case Array::Format::SNorm32x1: return { cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindSigned),   4 * width * 1 };
        case Array::Format::SNorm32x2: return { cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindSigned),   4 * width * 2 };
        case Array::Format::SNorm32x4: return { cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindSigned),   4 * width * 4 };
        }
        throw BtrcException("unsupported channel format");
    }

} // namespace anonymous

Array::Array()
    : width_(0), height_(0), depth_(0), format_{}, arr_(nullptr)
{
    
}

Array::~Array()
{
    if(arr_)
        cudaFreeArray(arr_);
}

void Array::load_from_memory(int width, int height, Format format, const void *linear_data)
{
    assert(linear_data);

    cudaArray_t new_arr = nullptr;
    BTRC_SCOPE_SUCCESS
    {
        if(arr_)
            cudaFreeArray(arr_);
        width_ = width;
        height_ = height;
        depth_ = 0;
        format_ = format;
        arr_ = new_arr;
    };
    BTRC_SCOPE_FAIL
    {
        if(new_arr)
            cudaFreeArray(new_arr);
    };

    const auto &format_desc = get_format_desc(format, width);

    throw_on_error(cudaMallocArray(&new_arr, &format_desc.channel, width, height));
    throw_on_error(cudaMemcpy2DToArray(
        new_arr, 0, 0, linear_data,
        format_desc.width_bytes, format_desc.width_bytes, height,
        cudaMemcpyHostToDevice));

    std::tie(min_value_, max_value_) = find_minmax_values(width, height, 1, format, linear_data);
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

void Array::load_from_memory(int width, int height, int depth, Format format, const void *linear_data)
{
    assert(linear_data);

    cudaArray_t new_arr = nullptr;
    BTRC_SCOPE_SUCCESS
    {
        if(arr_)
            cudaFreeArray(arr_);
        width_ = width;
        height_ = height;
        depth_ = depth;
        format_ = format;
        arr_ = new_arr;
    };
    BTRC_SCOPE_FAIL
    {
        if(new_arr)
            cudaFreeArray(new_arr);
    };

    const auto &format_desc = get_format_desc(format, width);
    const auto extent = cudaExtent{
        static_cast<size_t>(width),
        static_cast<size_t>(height),
        static_cast<size_t>(depth)
    };

    throw_on_error(cudaMalloc3DArray(&new_arr, &format_desc.channel, extent));
    const cudaMemcpy3DParms copy_params = {
        .srcArray = nullptr,
        .srcPos = cudaPos{ 0, 0, 0 },
        .srcPtr = cudaPitchedPtr{
            .ptr = const_cast<void *>(linear_data),
            .pitch = format_desc.width_bytes,
            .xsize = static_cast<size_t>(width),
            .ysize = static_cast<size_t>(height)
        },
        .dstArray = new_arr,
        .dstPos = cudaPos{ 0, 0, 0 },
        .dstPtr = { nullptr, 0, 0, 0 },
        .extent = extent,
        .kind = cudaMemcpyHostToDevice
    };
    throw_on_error(cudaMemcpy3D(&copy_params));

    std::tie(min_value_, max_value_) = find_minmax_values(width, height, depth, format, linear_data);
}

void Array::load_from_images(const std::vector<std::string> &filenames)
{
    auto image0 = ImageDynamic::load(filenames[0]);
    image0.match(
        [](std::monostate) { unreachable(); },
        [&](const auto &concret_image)
    {
        using ConcretImage = std::remove_cvref_t<decltype(concret_image)>;
        using SrcTexel = typename ConcretImage::Texel;
        using DstTexel = typename ImageToFormatAux<ConcretImage>::DstTexel;

        std::vector<DstTexel> linear_data;
        const int texel_count = concret_image.width() * concret_image.height();
        if constexpr(std::is_same_v<SrcTexel, DstTexel>)
        {
            std::copy(
                concret_image.data(),
                concret_image.data() + texel_count,
                std::back_inserter(linear_data));
        }
        else
        {
            auto converted_image = concret_image.template to<DstTexel>();
            std::copy(
                converted_image.data(),
                converted_image.data() + texel_count,
                std::back_inserter(linear_data));
        }

        for(size_t i = 1; i < filenames.size(); ++i)
        {
            const auto imagei = ImageDynamic::load(filenames[i]);
            imagei.match(
                [](std::monostate) { unreachable(); },
                [&](const auto &concret_imagei)
            {
                using ConcretImageI = std::remove_cvref_t<decltype(concret_imagei)>;
                using SrcTexelI = typename ConcretImageI::Texel;
                using DstTexelI = typename ImageToFormatAux<ConcretImageI>::DstTexel;
                constexpr Format formati = ImageToFormatAux<ConcretImageI>::format;

                if constexpr(formati != ImageToFormatAux<ConcretImage>::format)
                {
                    throw BtrcException(
                        "Array::load_from_images: image formats doesn't match");
                }
                else
                {
                    static_assert(std::is_same_v<DstTexel, DstTexelI>);

                    if(concret_imagei.width() != concret_image.width())
                    {
                        throw BtrcException(
                            "Array::load_from_images: image widths doesn't match");
                    }

                    if(concret_imagei.height() != concret_image.height())
                    {
                        throw BtrcException(
                            "Array::load_from_images: image heights doesn't match");
                    }

                    if constexpr(std::is_same_v<SrcTexelI, DstTexelI>)
                    {
                        std::copy(
                            concret_imagei.data(),
                            concret_imagei.data() + texel_count,
                            std::back_inserter(linear_data));
                    }
                    else
                    {
                        auto converted_imagei = concret_imagei.template to<DstTexelI>();
                        std::copy(
                            converted_imagei.data(),
                            converted_imagei.data() + texel_count,
                            std::back_inserter(linear_data));
                    }
                }
            });
        }

        constexpr Format format = ImageToFormatAux<ConcretImage>::format;
        this->load_from_memory(
            concret_image.width(),
            concret_image.height(),
            static_cast<int>(filenames.size()),
            format, linear_data.data());
    });
}

void Array::load_from_text(const std::string &filename)
{
    std::ifstream fin(filename, std::ifstream::in);
    if(!fin)
        throw BtrcException("failed to open file: " + filename);

    std::string component;
    fin >> component;
    int channels, width, height, depth;
    fin >> channels >> width >> height >> depth;
    if(!fin)
        throw BtrcException("Array::load_from_text: failed to parse file head");

    if(channels != 1 && channels != 3 && channels != 4)
        throw BtrcException("unsupported channel count: " + std::to_string(channels));

    if(component == "byte")
    {
        std::vector<uint8_t> linear_data;
        for(int z = 0; z < depth; ++z)
        {
            for(int y = 0; y < height; ++y)
            {
                for(int x = 0; x < width; ++x)
                {
                    uint8_t comp;
                    for(int c = 0; c < channels; ++c)
                    {
                        fin >> comp;
                        linear_data.push_back(comp);
                    }
                    if(channels == 3)
                        linear_data.push_back(255);
                }
            }
        }
        if(!fin)
            throw BtrcException("failed to load " + filename);
        if(channels == 1)
            load_from_memory(width, height, depth, Format::SNorm8x1, linear_data.data());
        else
            load_from_memory(width, height, depth, Format::SNorm8x4, linear_data.data());
    }
    else if(component == "float")
    {
        std::vector<float> linear_data;
        for(int z = 0; z < depth; ++z)
        {
            for(int y = 0; y < height; ++y)
            {
                for(int x = 0; x < width; ++x)
                {
                    float comp;
                    for(int c = 0; c < channels; ++c)
                    {
                        fin >> comp;
                        linear_data.push_back(comp);
                    }
                    if(channels == 3)
                        linear_data.push_back(1.0f);
                }
            }
        }
        if(!fin)
            throw BtrcException("failed to load " + filename);
        if(channels == 1)
            load_from_memory(width, height, depth, Format::F32x1, linear_data.data());
        else
            load_from_memory(width, height, depth, Format::F32x4, linear_data.data());
    }
    else
        throw BtrcException("unknown component type: " + component);
}

bool Array::is_2d() const
{
    return depth_ == 0;
}

bool Array::is_3d() const
{
    return !is_2d();
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

int Array::get_depth() const
{
    return depth_;
}

Array::Format Array::get_format() const
{
    return format_;
}

const Vec3f &Array::get_min_value() const
{
    return min_value_;
}

const Vec3f &Array::get_max_value() const
{
    return max_value_;
}

namespace
{

    template<typename Component, int ComponentCount, bool Normalize>
    std::pair<Vec3f, Vec3f> find_minmax_impl(
        int width, int height, int depth, const void *data)
    {
        Vec3<Component> minv(std::numeric_limits<Component>::max());
        Vec3<Component> maxv(std::numeric_limits<Component>::lowest());

        const size_t texel_count = static_cast<size_t>(width) * height * depth;
        auto comp = static_cast<const Component *>(data);

        for(size_t i = 0; i < texel_count; ++i)
        {
            Vec3<Component> texel;
            texel.x = *comp++;
            if constexpr(ComponentCount >= 2)
                texel.y = *comp++;
            if constexpr(ComponentCount == 4)
            {
                texel.z = *comp++;
                ++comp;
            }

            minv.x = (std::min)(minv.x, texel.x);
            minv.y = (std::min)(minv.y, texel.y);
            minv.z = (std::min)(minv.z, texel.z);

            maxv.x = (std::max)(maxv.x, texel.x);
            maxv.y = (std::max)(maxv.y, texel.y);
            maxv.z = (std::max)(maxv.z, texel.z);
        }

#define CONVERT_COMP(TYPE)                                                             \
        if constexpr(std::is_same_v<Component, TYPE>)                                  \
        {                                                                              \
            if constexpr(Normalize)                                                    \
            {                                                                          \
                constexpr float factor = static_cast<float>(                           \
                    1.0 / std::numeric_limits<Component>::max());                      \
                const Vec3f minr = {                                                   \
                    static_cast<float>(minv.x) * factor,                               \
                    static_cast<float>(minv.y) * factor,                               \
                    static_cast<float>(minv.z) * factor                                \
                };                                                                     \
                const Vec3f maxr = {                                                   \
                    static_cast<float>(maxv.x) * factor,                               \
                    static_cast<float>(maxv.y) * factor,                               \
                    static_cast<float>(maxv.z) * factor                                \
                };                                                                     \
                return { minr, maxr };                                                 \
            }                                                                          \
            else                                                                       \
            {                                                                          \
                const Vec3f minr = {                                                   \
                    static_cast<float>(minv.x),                                        \
                    static_cast<float>(minv.y),                                        \
                    static_cast<float>(minv.z)                                         \
                };                                                                     \
                const Vec3f maxr = {                                                   \
                    static_cast<float>(maxv.x),                                        \
                    static_cast<float>(maxv.y),                                        \
                    static_cast<float>(maxv.z)                                         \
                };                                                                     \
                return { minr, maxr };                                                 \
            }                                                                          \
        }


        if constexpr(std::is_same_v<Component, float>)
            return { minv, maxv };
        CONVERT_COMP(int32_t)
        CONVERT_COMP(uint32_t)
        CONVERT_COMP(int16_t)
        CONVERT_COMP(uint16_t)
        CONVERT_COMP(int8_t)
        CONVERT_COMP(uint8_t)
        throw BtrcException("Array::find_minmax_values: unsupported format");
    }

} // namespace anonymous

std::pair<Vec3f, Vec3f> Array::find_minmax_values(
    int width, int height, int depth,
    Format format, const void *data)
{
    switch(format)
    {
    case Format::S8x1:      return find_minmax_impl<int8_t, 1, false>(width, height, depth, data);
    case Format::S8x2:      return find_minmax_impl<int8_t, 2, false>(width, height, depth, data);
    case Format::S8x4:      return find_minmax_impl<int8_t, 4, false>(width, height, depth, data);
    case Format::U8x1:      return find_minmax_impl<uint8_t, 1, false>(width, height, depth, data);
    case Format::U8x2:      return find_minmax_impl<uint8_t, 2, false>(width, height, depth, data);
    case Format::U8x4:      return find_minmax_impl<uint8_t, 4, false>(width, height, depth, data);
    case Format::S16x1:     return find_minmax_impl<int16_t, 1, false>(width, height, depth, data);
    case Format::S16x2:     return find_minmax_impl<int16_t, 2, false>(width, height, depth, data);
    case Format::S16x4:     return find_minmax_impl<int16_t, 4, false>(width, height, depth, data);
    case Format::U16x1:     return find_minmax_impl<uint16_t, 1, false>(width, height, depth, data);
    case Format::U16x2:     return find_minmax_impl<uint16_t, 2, false>(width, height, depth, data);
    case Format::U16x4:     return find_minmax_impl<uint16_t, 4, false>(width, height, depth, data);
    case Format::S32x1:     return find_minmax_impl<int32_t, 1, false>(width, height, depth, data);
    case Format::S32x2:     return find_minmax_impl<int32_t, 2, false>(width, height, depth, data);
    case Format::S32x4:     return find_minmax_impl<int32_t, 4, false>(width, height, depth, data);
    case Format::U32x1:     return find_minmax_impl<uint32_t, 1, false>(width, height, depth, data);
    case Format::U32x2:     return find_minmax_impl<uint32_t, 2, false>(width, height, depth, data);
    case Format::U32x4:     return find_minmax_impl<uint32_t, 4, false>(width, height, depth, data);
    case Format::F32x1:     return find_minmax_impl<float, 1, false>(width, height, depth, data);
    case Format::F32x2:     return find_minmax_impl<float, 2, false>(width, height, depth, data);
    case Format::F32x4:     return find_minmax_impl<float, 4, false>(width, height, depth, data);
    case Format::SNorm8x1:  return find_minmax_impl<int8_t, 1, true>(width, height, depth, data);
    case Format::SNorm8x2:  return find_minmax_impl<int8_t, 2, true>(width, height, depth, data);
    case Format::SNorm8x4:  return find_minmax_impl<int8_t, 4, true>(width, height, depth, data);
    case Format::UNorm8x1:  return find_minmax_impl<uint8_t, 1, true>(width, height, depth, data);
    case Format::UNorm8x2:  return find_minmax_impl<uint8_t, 2, true>(width, height, depth, data);
    case Format::UNorm8x4:  return find_minmax_impl<uint8_t, 4, true>(width, height, depth, data);
    case Format::SNorm16x1: return find_minmax_impl<int16_t, 1, true>(width, height, depth, data);
    case Format::SNorm16x2: return find_minmax_impl<int16_t, 2, true>(width, height, depth, data);
    case Format::SNorm16x4: return find_minmax_impl<int16_t, 4, true>(width, height, depth, data);
    case Format::UNorm16x1: return find_minmax_impl<uint16_t, 1, true>(width, height, depth, data);
    case Format::UNorm16x2: return find_minmax_impl<uint16_t, 2, true>(width, height, depth, data);
    case Format::UNorm16x4: return find_minmax_impl<uint16_t, 4, true>(width, height, depth, data);
    case Format::SNorm32x1: return find_minmax_impl<int32_t, 1, true>(width, height, depth, data);
    case Format::SNorm32x2: return find_minmax_impl<int32_t, 2, true>(width, height, depth, data);
    case Format::SNorm32x4: return find_minmax_impl<int32_t, 4, true>(width, height, depth, data);
    case Format::UNorm32x1: return find_minmax_impl<uint32_t, 1, true>(width, height, depth, data);
    case Format::UNorm32x2: return find_minmax_impl<uint32_t, 2, true>(width, height, depth, data);
    case Format::UNorm32x4: return find_minmax_impl<uint32_t, 4, true>(width, height, depth, data);
    case Format::F16x1:
    case Format::F16x2:
    case Format::F16x4:
        throw BtrcException("unimplemented");
    }
    unreachable();
}

BTRC_CUDA_END
