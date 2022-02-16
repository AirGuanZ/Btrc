#include <array>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <tinyexr.h>

#include <btrc/utils/image.h>
#include <btrc/utils/scope_guard.h>
#include <btrc/utils/string.h>

BTRC_BEGIN

namespace image_detail
{

    void save_png(
        const std::string &filename,
        int                width,
        int                height,
        int                data_channels,
        const uint8_t     *data)
    {
        if(!stbi_write_png(
            filename.c_str(), width, height, data_channels, data, 0))
            throw BtrcException("failed to write png file: " + filename);
    }
    
    void save_jpg(
        const std::string &filename,
        int                width,
        int                height,
        int                data_channels,
        const uint8_t     *data)
    {
        if(!stbi_write_jpg(
            filename.c_str(), width, height, data_channels, data, 0))
            throw BtrcException("failed to write jpg file: " + filename);
    }
    
    void save_hdr(
        const std::string &filename,
        int                width,
        int                height,
        const float       *data)
    {
        if(!stbi_write_hdr(
            filename.c_str(), width, height, 3, data))
            throw BtrcException("failed to write hdr file: " + filename);
    }

    void save_exr(
        const std::string &filename,
        int                width,
        int                height,
        const float       *data)
    {
        EXRHeader header;
        InitEXRHeader(&header);

        EXRImage image;
        InitEXRImage(&image);

        image.num_channels = 3;
        std::vector channels(3, std::vector<float>(width * height));
        for(int i = 0; i < width * height; ++i)
        {
            channels[0][i] = data[3 * i + 0];
            channels[1][i] = data[3 * i + 1];
            channels[2][i] = data[3 * i + 2];
        }

        std::array channel_ptrs = {
            channels[2].data(),
            channels[1].data(),
            channels[0].data()
        };

        image.images = reinterpret_cast<unsigned char **>(channel_ptrs.data());
        image.width = width;
        image.height = height;

        header.num_channels = 3;

        std::vector<EXRChannelInfo> header_channels(3);
        header.channels = header_channels.data();
        strcpy_s(header.channels[0].name, 255, "B");
        strcpy_s(header.channels[1].name, 255, "G");
        strcpy_s(header.channels[2].name, 255, "R");

        std::vector<int> header_pixel_types(3);
        std::vector<int> header_requested_pixel_types(3);
        header.pixel_types = header_pixel_types.data();
        header.requested_pixel_types = header_requested_pixel_types.data();
        for(int i = 0; i < 3; ++i)
        {
            header.pixel_types[i]           = TINYEXR_PIXELTYPE_FLOAT;
            header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
        }

        const char *err;
        const int ret = SaveEXRImageToFile(
            &image, &header, filename.c_str(), &err);
        if(ret != TINYEXR_SUCCESS)
        {
            std::string msg = "failed to save exr file: " + filename;
            if(err)
            {
                msg += std::string(". ") + err;
                FreeEXRErrorMessage(err);
            }
            throw BtrcException(msg);
        }
    }
    
    std::vector<uint8_t> load_png(
        const std::string &filename,
        int               *width,
        int               *height,
        int               *channels)
    {
        std::FILE *file = std::fopen(filename.c_str(), "rb");
        if(!file)
            throw BtrcException("failed to open file: " + filename);
        BTRC_SCOPE_EXIT{ std::fclose(file); };

        int x, y, c;
        auto bytes = stbi_load_from_file(file, &x, &y, &c, 0);
        if(!bytes)
            throw BtrcException("failed to load image: " + filename);
        BTRC_SCOPE_EXIT{ stbi_image_free(bytes); };

        if(width)
            *width = x;
        if(height)
            *height = y;
        if(channels)
            *channels = c;
        return std::vector(bytes, bytes + x * y * c);
    }

    std::vector<uint8_t> load_jpg(
        const std::string &filename,
        int               *width,
        int               *height,
        int               *channels)
    {
        return load_png(filename, width, height, channels);
    }

    std::vector<Vec3f> load_hdr(
        const std::string &filename,
        int               *width,
        int               *height)
    {
        std::FILE *file = std::fopen(filename.c_str(), "rb");
        if(!file)
            throw BtrcException("failed to open file: " + filename);
        BTRC_SCOPE_EXIT{ std::fclose(file); };

        int x, y, c;
        auto floats = stbi_loadf_from_file(file, &x, &y, &c, 3);
        if(!floats)
            throw BtrcException("failed to load image: " + filename);
        BTRC_SCOPE_EXIT{ stbi_image_free(floats); };
        
        assert(x > 0 && y > 0);
        if(width)
            *width = x;
        if(height)
            *height = y;

        std::vector<Vec3f> result(x * y * 3);
        std::memcpy(result.data(), floats, sizeof(float) * x * y * 3);
        return result;
    }

    ImageFormat infer_format(const std::string &filename)
    {
        auto ext = std::filesystem::path(filename).extension().string();
        to_upper_(ext);
        if(ext == ".PNG")
            return ImageFormat::PNG;
        if(ext == ".JPG" || ext == ".JPEG")
            return ImageFormat::JPG;
        if(ext == ".HDR")
            return ImageFormat::HDR;
        if(ext == ".EXR")
            return ImageFormat::EXR;
        throw BtrcException("unknown image format: " + ext);
    }

} // namespace image_detail

void ImageDynamic::swap(ImageDynamic &other) noexcept
{
    image_.swap(other.image_);
}

ImageDynamic::operator bool() const
{
    return image_.match(
        [](std::monostate) { return false; },
        [](auto &image) { return static_cast<bool>(image); });
}

template<typename T>
bool ImageDynamic::is() const
{
    return image_.is<T>();
}

template<typename T>
T &ImageDynamic::as()
{
    return *image_.as_if<T>();
}

template<typename T>
const T &ImageDynamic::as() const
{
    return *image_.as_if<T>();
}

template<typename T>
T *ImageDynamic::as_if()
{
    return image_.as_if<T>();
}

template<typename T>
const T *ImageDynamic::as_if() const
{
    return image_.as_if<T>();
}

ImageDynamic ImageDynamic::to(TexelType new_texel_type) const
{
    return image_.match(
        [](std::monostate) -> ImageDynamic
    {
        throw BtrcException("ImageDynamic::to: empty source image");
    },
        [&](auto &image)
    {
        switch(new_texel_type)
        {
        case U8x1:  return ImageDynamic(image.template to<uint8_t>());
        case U8x3:  return ImageDynamic(image.template to<Vec3b>());
        case U8x4:  return ImageDynamic(image.template to<Vec4b>());
        case F32x1: return ImageDynamic(image.template to<float>());
        case F32x3: return ImageDynamic(image.template to<Vec3f>());
        case F32x4: return ImageDynamic(image.template to<Vec4f>());
        }
        unreachable();
    });
}

template<typename T>
ImageDynamic ImageDynamic::to() const
{
    return image_.match(
        [](std::monostate) -> ImageDynamic
    {
        throw BtrcException("ImageDynamic::to: empty source image");
    },
        [](auto &image)
    {
        return ImageDynamic(image.template to<T>());
    });
}

int ImageDynamic::width() const
{
    return image_.match(
        [](std::monostate) { return 0; },
        [](auto &image) { return image.width(); });
}

int ImageDynamic::height() const
{
    return image_.match(
        [](std::monostate) { return 0; },
        [](auto &image) { return image.height(); });
}

void ImageDynamic::save(const std::string &filename, ImageFormat format) const
{
    image_.match(
        [](std::monostate)
    {
        throw BtrcException("ImageDynamic::save: empty image");
    },
        [&](auto &image)
    {
        image.save(filename, format);
    });
}

ImageDynamic ImageDynamic::load(const std::string &filename, ImageFormat format)
{
    using namespace image_detail;

    if(format == ImageFormat::Auto)
        format = infer_format(filename);

    switch(format)
    {
    case ImageFormat::PNG:
    case ImageFormat::JPG:
    {
        int width, height, channels;
        const auto data = format == ImageFormat::PNG ?
            load_png(filename, &width, &height, &channels) :
            load_jpg(filename, &width, &height, &channels);
        if(channels == 1)
        {
            auto image = Image<uint8_t>(width, height);
            std::memcpy(image.data(), data.data(), sizeof(uint8_t) * data.size());
            return ImageDynamic(std::move(image));
        }
        if(channels == 3)
        {
            auto image = Image<Vec3b>(width, height);
            std::memcpy(image.data(), data.data(), sizeof(uint8_t) * data.size());
            return ImageDynamic(std::move(image));
        }
        if(channels == 4)
        {
            auto image = Image<Vec4b>(width, height);
            std::memcpy(image.data(), data.data(), sizeof(uint8_t) * data.size());
            return ImageDynamic(std::move(image));
        }
        throw BtrcException(
            "ImageDynamic::load: unsupported channels: " + std::to_string(channels));
    }
    case ImageFormat::HDR:
    {
        int width, height;
        auto data = load_hdr(filename, &width, &height);
        auto image = Image<Vec3f>(width, height);
        std::memcpy(image.data(), data.data(), sizeof(Vec3f) * data.size());
        return ImageDynamic(image);
    }
    case ImageFormat::EXR:
        throw BtrcException("ImageDynamic::load: EXR is not supported");
    case ImageFormat::Auto:
        unreachable();
    }
    unreachable();
}

BTRC_END
