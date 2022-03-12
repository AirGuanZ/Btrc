#pragma once

#include <btrc/core/medium.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class HetergeneousMedium : public Medium
{
public:

    void set_priority(float priority);

    void set_sigma_t(RC<Texture3D> sigma_t);

    void set_albedo(RC<Texture3D> albedo);

    void set_g(RC<Texture3D> g);

    SampleResult sample(
        CompileContext &cc,
        ref<CVec3f>     a,
        ref<CVec3f>     b,
        ref<CVec3f>     uvw_a,
        ref<CVec3f>     uvw_b,
        ref<CRNG>       rng) const override;

    CSpectrum tr(
        CompileContext &cc,
        ref<CVec3f>     a,
        ref<CVec3f>     b,
        ref<CVec3f>     uvw_a,
        ref<CVec3f>     uvw_b,
        ref<CRNG>       rng) const override;

    float get_priority() const override;

private:

    float priority_ = 0.0f;
    BTRC_OBJECT(Texture3D, sigma_t_);
    BTRC_OBJECT(Texture3D, albedo_);
    BTRC_OBJECT(Texture3D, g_);
};

class HetergeneousMediumCreator : public factory::Creator<Medium>
{
public:

    std::string get_name() const override { return "hetergeneous"; }

    RC<Medium> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
