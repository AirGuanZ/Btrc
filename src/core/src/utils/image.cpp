#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#include <tinyexr.h>

#include <btrc/core/utils/image.h>

BTRC_CORE_BEGIN

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

} // namespace image_detail

BTRC_CORE_END
