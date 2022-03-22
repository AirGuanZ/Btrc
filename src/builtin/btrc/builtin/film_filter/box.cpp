#include <btrc/builtin/film_filter/box.h>

BTRC_BUILTIN_BEGIN

CVec2f BoxFilter::sample(ref<CRNG> rng) const
{
    var x = rng.uniform_float() - 0.5f;
    var y = rng.uniform_float() - 0.5f;
    return CVec2f(x, y);
}

RC<FilmFilter> BoxFilterCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    return newRC<BoxFilter>();
}

BTRC_BUILTIN_END
