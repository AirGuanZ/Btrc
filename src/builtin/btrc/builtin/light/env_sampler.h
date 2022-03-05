#pragma once

#include <btrc/core/texture2d.h>
#include <btrc/utils/cuda/buffer.h>

BTRC_BUILTIN_BEGIN

class EnvirLightSampler
{
public:

    CUJ_CLASS_BEGIN(SampleResult)
        CUJ_MEMBER_VARIABLE(CVec3f, to_light)
        CUJ_MEMBER_VARIABLE(f32, pdf)
    CUJ_CLASS_END

    void preprocess(const RC<const Texture2D> &tex, const Vec2i &lut_res, int n_samples);

    SampleResult sample(ref<CVec3f> sam) const;

    f32 pdf(ref<CVec3f> to_light) const;

private:

    Vec2i               lut_res_;
    cuda::Buffer<float> tile_probs_;
    CAliasTable         tile_alias_;
};

BTRC_BUILTIN_END
