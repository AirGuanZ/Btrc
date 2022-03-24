#pragma once

#include <btrc/core/medium.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class HomogeneousMedium : public Medium
{
public:

    void set_priority(float priority);

    void set_sigma_t(float sigma_t);

    void set_albedo(const Spectrum &albedo);

    void set_g(float g);

    SampleResult sample(
        CompileContext &cc,
        ref<CVec3f>     a,
        ref<CVec3f>     b,
        ref<CVec3f>     uvw_a,
        ref<CVec3f>     uvw_b,
        Sampler        &sampler) const override;

    CSpectrum tr(
        CompileContext &cc,
        ref<CVec3f>     a,
        ref<CVec3f>     b,
        ref<CVec3f>     uvw_a,
        ref<CVec3f>     uvw_b,
        Sampler        &sampler) const override;

    float get_priority() const override;

private:

    float priority_ = 0.0f;
    float sigma_t_ = 1;
    Spectrum albedo_ = Spectrum::from_rgb(0.6f, 0.6f, 0.6f);
    float g_ = 0;
};

class HomogeneousMediumCreator : public factory::Creator<Medium>
{
public:

    std::string get_name() const override { return "homogeneous"; }

    RC<Medium> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
