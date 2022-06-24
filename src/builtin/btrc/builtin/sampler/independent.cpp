#include <btrc/builtin/sampler/independent.h>
#include <btrc/utils/hash.h>

BTRC_BUILTIN_BEGIN

IndependentSampler::IndependentSampler(const Vec2i &res, ref<CState> state)
{
    state_ = state;
}

IndependentSampler::IndependentSampler(const Vec2i &res, const CVec2u &pixel, i32 sample_index)
{
    state_.rng = cstd::PCG(hash::hash(pixel.x, pixel.y));
    state_.rng.advance(i64(sample_index) * 65536u);
}

void IndependentSampler::save(ptr<CState> p_state) const
{
    *p_state = state_;
}

f32 IndependentSampler::get1d()
{
    return state_.rng.uniform_float();
}

BTRC_BUILTIN_END
