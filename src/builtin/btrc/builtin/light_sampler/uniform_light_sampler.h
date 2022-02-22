#pragma once

#include <btrc/core/light_sampler.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class UniformLightSampler : public LightSampler
{
public:

    void clear() override;

    void add_light(RC<Light> light) override;

    std::vector<RC<Object>> get_dependent_objects() override;

    SampleResult sample(const CVec3f &ref, f32 time, f32 sam) const override;

    f32 pdf(const CVec3f &ref, f32 time, i32 light_index) const override;

    int get_light_count() const override;

    RC<const Light> get_light(int index) const override;

    RC<const EnvirLight> get_envir_light() const override;

    int get_envir_light_index() const override;

private:

    std::vector<RC<Light>> lights_;

    int envir_light_index_ = -1;
    RC<EnvirLight> envir_light_;
};

class UniformLightSamplerCreator : public factory::Creator<LightSampler>
{
public:

    std::string get_name() const override { return "uniform"; }

    RC<LightSampler> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
