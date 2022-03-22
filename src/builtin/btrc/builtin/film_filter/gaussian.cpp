#include <numeric>

#include <btrc/builtin/film_filter/gaussian.h>

BTRC_BUILTIN_BEGIN

namespace
{

    constexpr int GRID_SIZE = 256;

    float gaussian(float d, float expv, float alpha)
    {
        return (std::max)(0.0f, std::exp(-alpha * d * d) - expv);
    }

} // namespace anonymous

void GaussianFilter::set_radius(float radius)
{
    radius_ = radius;
    set_need_commit();
}

void GaussianFilter::set_alpha(float alpha)
{
    alpha_ = alpha;
    set_need_commit();
}

void GaussianFilter::commit()
{
    expv_ = std::exp(-alpha_ * radius_ * radius_);

    std::vector value_table(GRID_SIZE * GRID_SIZE, 0.0f);

    for(int y = 0; y < GRID_SIZE; ++y)
    {
        const float yf = 2 * radius_ * (y + 0.5f) / GRID_SIZE - radius_;
        for(int x = 0; x < GRID_SIZE; ++x)
        {
            const float xf = 2 * radius_ * (x + 0.5f) / GRID_SIZE - radius_;
            const float d2 = xf * xf + yf * yf;
            if(d2 < radius_ * radius_)
            {
                const float gx = gaussian(xf, expv_, alpha_);
                const float gy = gaussian(yf, expv_, alpha_);
                value_table[y * GRID_SIZE + x] = gx * gy;
            }
        }
    }

    AliasTable alias_table;
    alias_table.initialize(value_table);
    alias_table_ = CAliasTable(alias_table);
}

CVec2f GaussianFilter::sample(ref<CRNG> rng) const
{
    var grid_index = alias_table_.sample(rng.uniform_float());
    var grid_y = grid_index / GRID_SIZE;
    var grid_x = grid_index % GRID_SIZE;
    var x_beg = -radius_ + 2 * radius_ / GRID_SIZE * f32(grid_x);
    var x_end = -radius_ + 2 * radius_ / GRID_SIZE * f32(grid_x + 1);
    var y_beg = -radius_ + 2 * radius_ / GRID_SIZE * f32(grid_y);
    var y_end = -radius_ + 2 * radius_ / GRID_SIZE * f32(grid_y + 1);
    var xf = lerp(x_beg, x_end, rng.uniform_float());
    var yf = lerp(y_beg, y_end, rng.uniform_float());
    return CVec2f(xf, yf);
}

RC<FilmFilter> GaussianFilterCreator::create(RC<const factory::Node> node, factory::Context &context)
{
    const float radius = node->parse_child_or("radius", 1.3f);
    const float alpha = node->parse_child_or("alpha", 1.6f);
    auto result = newRC<GaussianFilter>();
    result->set_radius(radius);
    result->set_alpha(alpha);
    return result;
}

BTRC_BUILTIN_END
