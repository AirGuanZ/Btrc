#pragma once

#include <btrc/utils/optix/context.h>
#include <btrc/utils/uncopyable.h>

BTRC_BUILTIN_BEGIN

class OptixAIDenoiser : public Uncopyable
{
public:

    OptixAIDenoiser() = default;

    OptixAIDenoiser(
        optix::Context &context,
        bool            albedo,
        bool            normal,
        int             width,
        int             height);

    OptixAIDenoiser(OptixAIDenoiser &&other) noexcept;

    OptixAIDenoiser &operator=(OptixAIDenoiser &&other) noexcept;

    ~OptixAIDenoiser();

    void swap(OptixAIDenoiser &other) noexcept;

    void denoise(const Vec4f *color, const Vec4f *albedo, const Vec4f *normal) const;

    const Vec4f *get_output_buffer() const;

private:

    struct Impl;

    Impl *impl_ = nullptr;
};

BTRC_BUILTIN_END
