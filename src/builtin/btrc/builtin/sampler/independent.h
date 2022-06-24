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

    IndependentSampler(const Vec2i &res, ref<CState> state);

    IndependentSampler(const Vec2i &res, const CVec2u &pixel, i32 sample_index);

    void save(ptr<CState> p_state) const;

    f32 get1d() override;

private:

    CState state_;
};

BTRC_BUILTIN_END
