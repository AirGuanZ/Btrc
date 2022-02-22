#pragma once

#include <btrc/core/light.h>

BTRC_BEGIN

class LightSampler : public Object
{
public:

    CUJ_CLASS_BEGIN(SampleResult)
        CUJ_MEMBER_VARIABLE(i32, light_idx)
        CUJ_MEMBER_VARIABLE(f32, pdf)
    CUJ_CLASS_END

    virtual ~LightSampler() = default;

    virtual void clear() = 0;

    virtual void add_light(RC<Light> light) = 0;

    virtual SampleResult sample(const CVec3f &ref, f32 time, f32 sam) const = 0;

    virtual f32 pdf(const CVec3f &ref, f32 time, i32 light_index) const = 0;

    virtual int get_light_count() const = 0;

    virtual RC<const Light> get_light(int index) const = 0;

    virtual RC<const EnvirLight> get_envir_light() const = 0;

    virtual int get_envir_light_index() const = 0;
};

BTRC_END
