#pragma once

#include <btrc/core/film_filter.h>
#include <btrc/factory/context.h>

BTRC_BUILTIN_BEGIN

class GaussianFilter : public FilmFilter
{
public:

    void set_radius(float radius);

    void set_alpha(float alpha);

    void commit() override;

    CVec2f sample(ref<CRNG> rng) const override;

private:

    float radius_ = 0;
    float alpha_ = 0;
    float expv_ = 0;
    CAliasTable alias_table_;
};

class GaussianFilterCreator : public factory::Creator<FilmFilter>
{
public:

    std::string get_name() const override { return "gaussian"; }

    RC<FilmFilter> create(RC<const factory::Node> node, factory::Context &context) override;
};

BTRC_BUILTIN_END
