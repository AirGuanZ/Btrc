#pragma once

#include <array>
#include <cassert>
#include <filesystem>
#include <vector>

#include <btrc/utils/math/math.h>
#include <btrc/utils/unreachable.h>
#include <btrc/utils/variant.h>

BTRC_BEGIN

enum class ImageFormat
{
    Auto,
    PNG,
    JPG,
    HDR,
    EXR
};

template<typename T>
class Image
{
public:

    using Texel = T;

    Image();

    Image(int width, int height);

    Image(const Image &other);

    Image(Image &&other) noexcept;

    Image &operator=(const Image &other);

    Image &operator=(Image &&other) noexcept;

    ~Image() = default;

    void swap(Image &other) noexcept;

    operator bool() const;

    int width() const;

    int height() const;

    Texel &operator()(int x, int y);

    const Texel &operator()(int x, int y) const;

    template<typename U>
    Image<U> to() const;

    void pow_(float v);

    Image pow(float v) const;

    auto begin() { return data_.begin(); }
    auto end()   { return data_.end(); }

    auto begin() const { return data_.begin(); }
    auto end()   const { return data_.end(); }

    auto data()       { return data_.data(); }
    auto data() const { return data_.data(); }

    void save(
        const std::string &filename,
        ImageFormat        format = ImageFormat::Auto) const;

    static Image load(
        const std::string &filename,
        ImageFormat        format = ImageFormat::Auto);

private:

    int width_;
    int height_;
    std::vector<Texel> data_;
};

class ImageDynamic
{
public:

    enum TexelType
    {
        U8x1,
        U8x3,
        U8x4,
        F32x1,
        F32x3,
        F32x4
    };

    ImageDynamic() = default;

    template<typename T>
    ImageDynamic(const Image<T> &image);

    template<typename T>
    ImageDynamic(Image<T> &&image);

    void swap(ImageDynamic &other) noexcept;

    operator bool() const;

    template<typename T>
    bool is() const;

    template<typename T>
    T &as();

    template<typename T>
    const T &as() const;

    template<typename T>
    T *as_if();

    template<typename T>
    const T *as_if() const;

    ImageDynamic to(TexelType new_texel_type) const;

    template<typename T>
    ImageDynamic to() const;

    int width() const;

    int height() const;

    template<typename...Vs>
    auto match(Vs &&...vs) const;

    template<typename...Vs>
    auto match(Vs &&...vs);

    void save(
        const std::string &filename,
        ImageFormat        format = ImageFormat::Auto) const;

    static ImageDynamic load(
        const std::string &filename,
        ImageFormat        format = ImageFormat::Auto);

private:

    Variant<
        std::monostate,
        Image<uint8_t>,
        Image<Vec3b>,
        Image<Vec4b>,
        Image<float>,
        Image<Vec3f>,
        Image<Vec4f>> image_ = std::monostate{};
};

// ========================== impl ==========================

namespace image_detail
{
    
    template<typename T>
    struct Trait;
    template<>
    struct Trait<uint8_t>
    {
        using Component = uint8_t;
        static constexpr int ComponentCount = 1;
    };
    template<>
    struct Trait<Vec3b>
    {
        using Component = uint8_t;
        static constexpr int ComponentCount = 3;
    };
    template<>
    struct Trait<Vec4b>
    {
        using Component = uint8_t;
        static constexpr int ComponentCount = 4;
    };
    template<>
    struct Trait<float>
    {
        using Component = float;
        static constexpr int ComponentCount = 1;
    };
    template<>
    struct Trait<Vec3f>
    {
        using Component = float;
        static constexpr int ComponentCount = 3;
    };
    template<>
    struct Trait<Vec4f>
    {
        using Component = float;
        static constexpr int ComponentCount = 4;
    };

    template<typename To, typename From>
    To to_component(From from)
    {
        if constexpr(std::is_same_v<To, From>)
            return from;
        else if constexpr(std::is_same_v<To, uint8_t>)
        {
            static_assert(std::is_same_v<From, float>);
            return static_cast<uint8_t>(
                (std::min)(static_cast<int>(from * 256), 255));
        }
        else
        {
            static_assert(std::is_same_v<To, float>);
            static_assert(std::is_same_v<From, uint8_t>);
            return from / 255.0f;
        }
    }

    template<typename T, int SrcComps, int DstComps>
    std::array<T, DstComps> convert_comps(const T *src)
    {
        constexpr T DEFAULT_ALPHA =
            static_cast<T>(std::is_same_v<T, uint8_t> ? 255 : 1);

        std::array<T, DstComps> ret;
        if constexpr(SrcComps == DstComps)
        {
            for(int i = 0; i < SrcComps; ++i)
                ret[i] = src[i];
        }
        else if constexpr(SrcComps == 1)
        {
            static_assert(DstComps == 3 || DstComps == 4);
            ret[0] = src[0];
            ret[1] = src[0];
            ret[2] = src[0];
            if constexpr(DstComps == 4)
                ret[3] = DEFAULT_ALPHA;
        }
        else if constexpr(SrcComps == 3)
        {
            if constexpr(DstComps == 1)
                ret[0] = src[0];
            else
            {
                static_assert(DstComps == 4);
                ret[0] = src[0];
                ret[1] = src[1];
                ret[2] = src[2];
                ret[3] = DEFAULT_ALPHA;
            }
        }
        else
        {
            static_assert(SrcComps == 4);
            static_assert(DstComps == 1 || DstComps == 3);
            ret[0] = src[0];
            if constexpr(DstComps == 3)
            {
                ret[1] = src[1];
                ret[2] = src[2];
            }
        }

        return ret;
    }

    template<typename To, typename From>
    To to(const From &from)
    {
        if constexpr(std::is_same_v<To, From>)
            return from;

        constexpr int FromComps = Trait<From>::ComponentCount;
        constexpr int ToComps = Trait<To>::ComponentCount;

        using FromComp = typename Trait<From>::Component;
        using ToComp = typename Trait<To>::Component;

        std::array<FromComp, ToComps> converted_channels =
            convert_comps<FromComp, FromComps, ToComps>(
                reinterpret_cast<const FromComp *>(&from));

        std::array<ToComp, ToComps> result_channels;
        for(int i = 0; i < ToComps; ++i)
            result_channels[i] = to_component<ToComp>(converted_channels[i]);

        To result;
        std::memcpy(&result, &result_channels, sizeof(To));
        return result;
    }

    inline float pow(float a, float b)
    {
        return std::pow(a, b);
    }

    inline Vec3f pow(const Vec3f &a, float b)
    {
        return Vec3f(std::pow(a.x, b), std::pow(a.y, b), std::pow(a.z, b));
    }

    inline Vec4f pow(const Vec4f &a, float b)
    {
        return Vec4(
            std::pow(a.x, b), std::pow(a.y, b), std::pow(a.z, b), std::pow(a.w, b));
    }

    inline uint8_t pow(uint8_t a, float b)
    {
        return to<uint8_t>(pow(to<float>(a), b));
    }

    inline Vec3b pow(const Vec3b &a, float b)
    {
        return to<Vec3b>(pow(to<Vec3f>(a), b));
    }

    inline Vec4b pow(const Vec4b &a, float b)
    {
        return to<Vec4b>(pow(to<Vec4f>(a), b));
    }
    
    void save_png(
        const std::string &filename,
        int                width,
        int                height,
        int                data_channels,
        const uint8_t     *data);
    
    void save_jpg(
        const std::string &filename,
        int                width,
        int                height,
        int                data_channels,
        const uint8_t     *data);
    
    void save_hdr(
        const std::string &filename,
        int                width,
        int                height,
        const float       *data);

    void save_exr(
        const std::string &filename,
        int                width,
        int                height,
        const float       *data);

    std::vector<uint8_t> load_png(
        const std::string &filename,
        int               *width,
        int               *height,
        int               *channels);

    std::vector<uint8_t> load_jpg(
        const std::string &filename,
        int               *width,
        int               *height,
        int               *channels);

    std::vector<Vec3f> load_hdr(
        const std::string &filename,
        int               *width,
        int               *height);

    ImageFormat infer_format(const std::string &filename);

} // namespace image_detail

template<typename T>
Image<T>::Image()
    : width_(0), height_(0)
{

}

template<typename T>
Image<T>::Image(int width, int height)
    : width_(width), height_(height)
{
    assert(width > 0 && height > 0);
    data_.resize(width * height);
}

template<typename T>
Image<T>::Image(const Image &other)
    : width_(other.width_), height_(other.height_)
{
    data_ = other.data_;
}

template<typename T>
Image<T>::Image(Image &&other) noexcept
    : Image()
{
    swap(other);
}

template<typename T>
Image<T> &Image<T>::operator=(const Image &other)
{
    Image t(other);
    swap(t);
    return *this;
}

template<typename T>
Image<T> &Image<T>::operator=(Image &&other) noexcept
{
    swap(other);
    return *this;
}

template<typename T>
void Image<T>::swap(Image &other) noexcept
{
    std::swap(width_, other.width_);
    std::swap(height_, other.height_);
    data_.swap(other.data_);
}

template<typename T>
Image<T>::operator bool() const
{
    assert(width_ > 0 == height_ > 0);
    assert(width_ > 0 == !data_.empty());
    return width_ > 0;
}

template<typename T>
int Image<T>::width() const
{
    return width_;
}

template<typename T>
int Image<T>::height() const
{
    return height_;
}

template<typename T>
typename Image<T>::Texel &Image<T>::operator()(int x, int y)
{
    assert(0 <= x && x < width_);
    assert(0 <= y && y < height_);
    const int idx = y * width_ + x;
    return data_[idx];
}

template<typename T>
const typename Image<T>::Texel &Image<T>::operator()(int x, int y) const
{
    assert(0 <= x && x < width_);
    assert(0 <= y && y < height_);
    const int idx = y * width_ + x;
    return data_[idx];
}

template<typename T>
template<typename U>
Image<U> Image<T>::to() const
{
    assert(!!*this);
    Image<U> result(width_, height_);
    for(int i = 0; i < width_ * height_; ++i)
        result.data()[i] = image_detail::to<U>(data_[i]);
    return result;
}

template<typename T>
Image<T> Image<T>::pow(float v) const
{
    Image ret = *this;
    ret.pow_(v);
    return ret;
}

template<typename T>
void Image<T>::pow_(float v)
{
    for(auto &d : data_)
        d = image_detail::pow(d, v);
}

template<typename T>
void Image<T>::save(const std::string &filename, ImageFormat format) const
{
    using namespace image_detail;

    if(format == ImageFormat::Auto)
        format = infer_format(filename);

    switch(format)
    {
    case ImageFormat::PNG:
    case ImageFormat::JPG:
    {
        auto save_func = format == ImageFormat::PNG ? &save_png : &save_jpg;
        if constexpr(std::is_same_v<T, float>)
        {
            to<uint8_t>().save(filename, format);
        }
        else if constexpr(std::is_same_v<T, Vec3f>)
        {
            to<Vec3b>().save(filename, format);
        }
        else if constexpr(std::is_same_v<T, Vec4f>)
        {
            to<Vec4b>().save(filename, format);
        }
        else if constexpr(std::is_same_v<T, uint8_t>)
        {
            save_func(filename, width_, height_, 1, &data_[0]);
        }
        else if constexpr(std::is_same_v<T, Vec3b>)
        {
            save_func(filename, width_, height_, 3, &data_[0].x);
        }
        else
        {
            static_assert(std::is_same_v<T, Vec4b>);
            save_func(filename, width_, height_, 4, &data_[0].x);
        }
        break;
    }
    case ImageFormat::HDR:
    case ImageFormat::EXR:
    {
        auto save_func = format == ImageFormat::HDR ? &save_hdr : &save_exr;
        if constexpr(std::is_same_v<T, Vec3f>)
        {
            save_func(filename, width_, height_, &data_[0].x);
        }
        else
        {
            to<Vec3f>().save(filename, format);
        }
        break;
    }
    default:
        unreachable();
    }
}

template<typename T>
Image<T> Image<T>::load(const std::string &filename, ImageFormat format)
{
    auto image_dynamic = ImageDynamic::load(filename, format);
    if(auto result = image_dynamic.as_if<Image<T>>())
        return std::move(*result);
    return image_dynamic.match(
        [&](std::monostate) -> Image<T>
        {
            throw BtrcException("failed to load image from " + filename);
        },
        [](auto &image)
        {
            return image.template to<T>();
        });
}

template<typename T>
ImageDynamic::ImageDynamic(const Image<T> &image)
    : image_(image)
{
    
}

template<typename T>
ImageDynamic::ImageDynamic(Image<T> &&image)
    : image_(std::move(image))
{
    
}

template<typename...Vs>
auto ImageDynamic::match(Vs &&...vs) const
{
    return image_.match(std::forward<Vs>(vs)...);
}

template<typename ... Vs>
auto ImageDynamic::match(Vs &&... vs)
{
    return image_.match(std::forward<Vs>(vs)...);
}

BTRC_END
