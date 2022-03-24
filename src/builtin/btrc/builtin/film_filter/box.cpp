#include <btrc/builtin/film_filter/box.h>

BTRC_BUILTIN_BEGIN

CVec2f BoxFilter::sample(Sampler &sampler) const
{
    var x = sampler.get1d() - 0.5f;
    var y = sampler.get1d() - 0.5f;
    return CVec2f(x, y);
}

RC<FilmFilter> BoxFilterCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    return newRC<BoxFilter>();
}

BTRC_BUILTIN_END
