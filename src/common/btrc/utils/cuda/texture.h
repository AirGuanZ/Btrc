#pragma once

#include <cuda_runtime.h>

#include <btrc/utils/cuda/array.h>

BTRC_CUDA_BEGIN

class Texture : public Uncopyable
{
public:

    enum class AddressMode
    {
        Wrap, Clamp, Mirror, Border
    };

    enum class FilterMode
    {
        Point, Linear
    };

    struct Description
    {
        AddressMode address_modes[3] = { AddressMode::Wrap, AddressMode::Wrap, AddressMode::Wrap };
        FilterMode  filter_mode = FilterMode::Point;
        bool        srgb_to_linear = false;
        float       border_value[4] = { 0, 0, 0, 0 };
    };

    Texture();

    Texture(Texture &&other) noexcept;

    Texture &operator=(Texture &&other) noexcept;

    ~Texture();

    void swap(Texture &other) noexcept;

    operator bool() const;
    
    void initialize(RC<const Array> arr, const Description &desc);

    void initialize(const std::string &filename, const Description &desc);

    cudaTextureObject_t get_tex() const;

    Vec3f get_min_value() const;

    Vec3f get_max_value() const;

private:

    void destroy();

    RC<const Array>     arr_;
    cudaTextureObject_t tex_;
};

BTRC_CUDA_END
