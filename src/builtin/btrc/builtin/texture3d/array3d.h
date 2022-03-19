#pragma once

#include <btrc/core/texture3d.h>
#include <btrc/factory/context.h>
#include <btrc/utils/cuda/texture.h>

BTRC_BUILTIN_BEGIN

class Array3D : public Texture3D
{
public:

    void initialize(RC<const cuda::Texture> cuda_texture);

    void initialize_from_text(const std::string &text_filename, const cuda::Texture::Description &desc);

    void initialize_from_images(const std::vector<std::string> &image_filenames, const cuda::Texture::Description &desc);

    CSpectrum sample_spectrum_inline(CompileContext &cc, ref<CVec3f> uvw) const override;

    f32 sample_float_inline(CompileContext &cc, ref<CVec3f> uvw) const override;

    Spectrum get_max_spectrum() const override;

    Spectrum get_min_spectrum() const override;

private:

    RC<const cuda::Texture> tex_;
};

class Array3DCreator : public factory::Creator<Texture3D>
{
public:

    std::string get_name() const override { return "array"; }

    RC<Texture3D> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
