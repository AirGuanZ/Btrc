#pragma once

#include <cuda_runtime.h>

#include <btrc/utils/cuda/array.h>

BTRC_BEGIN

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
        AddressMode address_modes[3];
        FilterMode  filter_mode;
        bool        srgb_to_linear;
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

private:

    void destroy();

    RC<const Array>     arr_;
    cudaTextureObject_t tex_;
};

BTRC_END
