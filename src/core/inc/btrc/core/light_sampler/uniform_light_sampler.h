#pragma once

#include <btrc/core/light_sampler/light_sampler.h>

BTRC_CORE_BEGIN

class UniformLightSampler : public LightSampler
{
public:

    void add_light(RC<const Light> light);

    SampleResult sample(const CVec3f &ref, f32 time, f32 sam) const override;

    f32 pdf(const CVec3f &ref, f32 time, i32 light_index) const override;

    int get_light_count() const override;

    RC<const Light> get_light(int index) const override;

    RC<const EnvirLight> get_envir_light() const override;

    int get_envir_light_index() const override;

private:

    std::vector<RC<const Light>> lights_;

    int envir_light_index_ = -1;
    RC<const EnvirLight> envir_light_;
};

BTRC_CORE_END
