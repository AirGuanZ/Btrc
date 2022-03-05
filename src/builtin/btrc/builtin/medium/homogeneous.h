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

    SampleResult sample(CompileContext &cc, ref<CVec3f> a, ref<CVec3f> b, ref<cstd::LCG> rng) const override;

    CSpectrum tr(CompileContext &cc, ref<CVec3f> a, ref<CVec3f> b) const override;

    float get_priority() const override;

private:

    float priority_ = 0.0f;
    BTRC_PROPERTY(float, sigma_t_);
    BTRC_PROPERTY(Spectrum, albedo_);
    BTRC_PROPERTY(float, g_);
};

class HomogeneousMediumCreator : public factory::Creator<Medium>
{
public:

    std::string get_name() const override { return "homogeneous"; }

    RC<Medium> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
