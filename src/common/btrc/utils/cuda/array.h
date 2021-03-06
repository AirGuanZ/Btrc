#pragma once

#include <cuda_runtime.h>

#include <btrc/utils/image.h>
#include <btrc/utils/uncopyable.h>

BTRC_CUDA_BEGIN

class Array : public Uncopyable
{
public:

    enum class Format
    {
        S8x1,
        S8x2,
        S8x4,
        U8x1,
        U8x2,
        U8x4,
        S16x1,
        S16x2,
        S16x4,
        U16x1,
        U16x2,
        U16x4,
        F16x1,
        F16x2,
        F16x4,
        S32x1,
        S32x2,
        S32x4,
        U32x1,
        U32x2,
        U32x4,
        F32x1,
        F32x2,
        F32x4,
        UNorm8x1,
        UNorm8x2,
        UNorm8x4,
        SNorm8x1,
        SNorm8x2,
        SNorm8x4,
        UNorm16x1,
        UNorm16x2,
        UNorm16x4,
        SNorm16x1,
        SNorm16x2,
        SNorm16x4,
        UNorm32x1,
        UNorm32x2,
        UNorm32x4,
        SNorm32x1,
        SNorm32x2,
        SNorm32x4
    };

    Array();

    ~Array();

    // init 2d
    
    void load_from_memory(
        int         width,
        int         height,
        Format      format,
        const void *linear_data);

    void load_from_memory(const Image<uint8_t> &image);
    void load_from_memory(const Image<Vec3b>   &image);
    void load_from_memory(const Image<Vec4b>   &image);
    void load_from_memory(const Image<float>   &image);
    void load_from_memory(const Image<Vec3f>   &image);
    void load_from_memory(const Image<Vec4f>   &image);

    void load_from_image(const std::string &filename);

    // init3d

    void load_from_memory(
        int         width,
        int         height,
        int         depth,
        Format      format,
        const void *linear_data);

    void load_from_images(const std::vector<std::string> &filenames);

    void load_from_text(const std::string &filename);

    // query

    bool is_2d() const;

    bool is_3d() const;

    cudaArray_t get_arr() const;

    int get_width() const;

    int get_height() const;

    int get_depth() const;

    Format get_format() const;

    const Vec3f &get_min_value() const;

    const Vec3f &get_max_value() const;

private:

    static std::pair<Vec3f, Vec3f> find_minmax_values(
        int width, int height, int depth,
        Format format, const void *data);

    int         width_;
    int         height_;
    int         depth_;
    Format      format_;
    cudaArray_t arr_;
    Vec3f       max_value_;
    Vec3f       min_value_;
};

BTRC_CUDA_END
