#pragma once

#include <btrc/core/film_filter.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class BoxFilter : public FilmFilter
{
public:

    CVec2f sample(Sampler &sampler) const override;
};

class BoxFilterCreator : public factory::Creator<FilmFilter>
{
public:

    std::string get_name() const override { return "box"; }

    RC<FilmFilter> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
