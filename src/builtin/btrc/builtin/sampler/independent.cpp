#include <btrc/builtin/sampler/independent.h>

BTRC_BUILTIN_BEGIN

IndependentSampler::IndependentSampler(ref<CState> state)
{
    state_ = state;
}

IndependentSampler::IndependentSampler(u64 pixel_index, u64 sample_index)
{
    state_.rng = cstd::PCG(pixel_index);
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

CVec2f IndependentSampler::get2d()
{
    var x = get1d();
    var y = get1d();
    return CVec2f(x, y);
}

CVec3f IndependentSampler::get3d()
{
    var x = get1d();
    var y = get1d();
    var z = get1d();
    return CVec3f(x, y, z);
}

BTRC_BUILTIN_END
