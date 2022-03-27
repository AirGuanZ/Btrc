#pragma once

#include <btrc/core/sampler.h>

BTRC_BUILTIN_BEGIN

namespace halton_sampler_detail
{

    struct State
    {
        uint32_t dimension;
        uint64_t halton_index;
    };

    CUJ_PROXY_CLASS(CState, State, dimension, halton_index);

} // namespace halton_sampler_detail

class HaltonSampler : public Sampler
{
public:

    using State = halton_sampler_detail::State;
    using CState = halton_sampler_detail::CState;

    HaltonSampler(const Vec2i &res, const CVec2u &pixel, i32 sample_index);

    HaltonSampler(const Vec2i &res, const CState &state);

    void save(ptr<CState> output);

    f32 get1d() override;

    CVec2f get2d() override;

    CVec3f get3d() override;

private:

    void initialize_consts(const Vec2i &res);

    Vec2i base_scales_;
    Vec2i base_exponents_;
    int mult_inverse_[2];
    CState state_;
};

BTRC_BUILTIN_END
