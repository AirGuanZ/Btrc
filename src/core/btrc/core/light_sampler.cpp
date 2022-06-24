#include <btrc/core/light_sampler.h>

BTRC_BEGIN

void UniformLightSampler::clear()
{
    lights_ = {};
    envir_light_index_ = -1;
    envir_light_ = {};
}

void UniformLightSampler::add_light(RC<Light> light)
{
    if(!light->is_area())
    {
        assert(!envir_light_);
        envir_light_ = std::dynamic_pointer_cast<EnvirLight>(light);
        envir_light_index_ = static_cast<int>(lights_.size());
    }
    lights_.push_back(std::move(light));
}

UniformLightSampler::SampleResult UniformLightSampler::sample(const CVec3f &ref, f32 sam) const
{
    if(lights_.empty())
    {
        SampleResult result;
        result.light_idx = -1;
        result.pdf = 0;
        return result;
    }

    var idx = cstd::min(i32(sam * static_cast<int>(lights_.size())), i32(static_cast<int>(lights_.size())) - 1);
    var pdf = 1.0f / f32(lights_.size());

    SampleResult result;
    result.light_idx = idx;
    result.pdf = pdf;
    return result;
}

f32 UniformLightSampler::pdf(const CVec3f &ref, i32 light_index) const
{
    return 1.0f / f32(lights_.size());
}

LightSampler::SampleEmitResult UniformLightSampler::sample_emit(f32 sam) const
{
    return sample(CVec3f(0), sam);
}

f32 UniformLightSampler::pdf_emit(i32 light_index) const
{
    return pdf(CVec3f(0), light_index);
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

BTRC_END
