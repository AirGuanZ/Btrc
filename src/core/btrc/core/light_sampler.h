#pragma once

#include <btrc/core/light.h>

BTRC_BEGIN

class LightSampler
{
public:

    CUJ_CLASS_BEGIN(SampleResult)
        CUJ_MEMBER_VARIABLE(i32, light_idx)
        CUJ_MEMBER_VARIABLE(f32, pdf)
    CUJ_CLASS_END

    virtual ~LightSampler() = default;

    virtual void clear() = 0;

    virtual void add_light(RC<Light> light) = 0;

    virtual SampleResult sample(const CVec3f &ref, f32 sam) const = 0;

    virtual f32 pdf(const CVec3f &ref, i32 light_index) const = 0;

    virtual int get_light_count() const = 0;

    virtual RC<const Light> get_light(int index) const = 0;

    virtual RC<const EnvirLight> get_envir_light() const = 0;

    virtual int get_envir_light_index() const = 0;
};

class UniformLightSampler : public LightSampler
{
public:

    void clear() override;

    void add_light(RC<Light> light) override;

    SampleResult sample(const CVec3f &ref, f32 sam) const override;

    f32 pdf(const CVec3f &ref, i32 light_index) const override;

    int get_light_count() const override;

    RC<const Light> get_light(int index) const override;

    RC<const EnvirLight> get_envir_light() const override;

    int get_envir_light_index() const override;

private:

    std::vector<RC<Light>> lights_;

    int envir_light_index_ = -1;
    RC<EnvirLight> envir_light_;
};

BTRC_END
