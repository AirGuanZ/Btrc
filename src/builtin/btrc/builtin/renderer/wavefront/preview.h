#pragma once

#include <btrc/utils/cuda/module.h>
#include <btrc/utils/math/vec4.h>
#include <btrc/utils/uncopyable.h>

#include "./common.h"

BTRC_WFPT_BEGIN

class PreviewImageGenerator : public Uncopyable
{
public:

    PreviewImageGenerator();

    void generate(
        int          width,
        int          height,
        const Vec4f *value_buffer,
        const float *weight_buffer,
        Vec4f       *output_buffer) const;
    
    void generate_albedo(
        int          width,
        int          height,
        const Vec4f *value_buffer,
        const float *weight_buffer,
        Vec4f       *output_buffer) const;
    
    void generate_normal(
        int          width,
        int          height,
        const Vec4f *value_buffer,
        const float *weight_buffer,
        Vec4f       *output_buffer) const;

private:

    cuda::Module cuda_module_;
};

BTRC_WFPT_END
