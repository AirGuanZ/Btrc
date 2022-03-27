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
