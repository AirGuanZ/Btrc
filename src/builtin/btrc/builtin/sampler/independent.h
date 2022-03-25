#pragma once

#include <btrc/core/sampler.h>

BTRC_BUILTIN_BEGIN

namespace independent_sampler_detail
{

    struct State
    {
        cstd::PCG::Data rng;
    };

    CUJ_PROXY_CLASS(CState, State, rng);

} // namespace independent_sampler_detail

class IndependentSampler : public Sampler
{
public:

    using State = independent_sampler_detail::State;
    using CState = independent_sampler_detail::CState;

    explicit IndependentSampler(ref<CState> state);

    IndependentSampler(u64 pixel_index, u64 sample_index);

    void save(ptr<CState> p_state) const;

    f32 get1d() override;

    CVec2f get2d() override;

    CVec3f get3d() override;

private:

    CState state_;
};

BTRC_BUILTIN_END
