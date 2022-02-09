#include <btrc/core/light_sampler/uniform_light_sampler.h>

BTRC_CORE_BEGIN

void UniformLightSampler::add_light(RC<const Light> light)
{
    if(!light->is_area())
    {
        assert(!envir_light_);
        envir_light_ = std::dynamic_pointer_cast<const EnvirLight>(light);
        envir_light_index_ = static_cast<int>(lights_.size());
    }
    lights_.push_back(std::move(light));
}

LightSampler::SampleResult UniformLightSampler::sample(
    const CVec3f &ref, f32 time, f32 sam) const
{
    if(lights_.empty())
    {
        SampleResult result;
        result.light_idx = -1;
        result.pdf = 0;
        return result;
    }

    var idx = cstd::min(
        i32(sam * static_cast<int>(lights_.size())),
        i32(lights_.size()) - 1);
    var pdf = 1.0f / f32(lights_.size());

    SampleResult result;
    result.light_idx = idx;
    result.pdf = pdf;
    return result;
}

f32 UniformLightSampler::pdf(const CVec3f &ref, f32 time, i32 light_index) const
{
    return 1.0f / f32(lights_.size());
}

int UniformLightSampler::get_light_count() const
{
    return static_cast<int>(lights_.size());
}

RC<const Light> UniformLightSampler::get_light(int index) const
{
    return lights_[index];
}

RC<const EnvirLight> UniformLightSampler::get_envir_light() const
{
    return envir_light_;
}

int UniformLightSampler::get_envir_light_index() const
{
    return envir_light_index_;
}

BTRC_CORE_END
