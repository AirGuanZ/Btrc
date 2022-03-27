#include <btrc/builtin/sampler/halton.h>
#include <btrc/utils/low_discrepancy.h>
#include <btrc/utils/prime.h>

BTRC_BUILTIN_BEGIN

namespace
{

    constexpr int MAX_HALTON_RESOLUTION = 128;

    int64_t mod(int64_t a, int64_t b)
    {
        const int64_t result = a - a / b * b;
        return result < 0 ? result + b : result;
    }

    void extended_gcd(uint64_t a, uint64_t b, int64_t *x, int64_t *y)
    {
        if(b == 0)
        {
            *x = 1;
            *y = 0;
            return;
        }
        int64_t d = a / b, xp, yp;
        extended_gcd(b, a % b, &xp, &yp);
        *x = yp;
        *y = xp - (d * yp);
    }

    uint64_t multiplicative_inverse(int64_t a, int64_t n)
    {
        int64_t x, y;
        extended_gcd(a, n, &x, &y);
        return mod(x, n);
    }

    f32 sample_halton(u32 dim, u64 halton_index)
    {
        return owen_scrambled_radical_inverse(i32(dim), halton_index, mix_bits(u64(1u + (dim << 4))));
    }

} // namespace anonymous

HaltonSampler::HaltonSampler(const Vec2i &res, const CVec2u &pixel, i32 sample_index)
{
    initialize_consts(res);

    state_.halton_index = 0;
    const int sample_stride = base_scales_[0] * base_scales_[1];
    if(sample_stride > 1)
    {
        var pm = CVec2u(pixel.x % MAX_HALTON_RESOLUTION, pixel.y % MAX_HALTON_RESOLUTION);
        for(int i = 0; i < 2; ++i)
        {
            var dim_offset = (i == 0) ? inverse_radical_inverse(u64(pm[i]), 2, base_exponents_[i])
                                      : inverse_radical_inverse(u64(pm[i]), 3, base_exponents_[i]);
            state_.halton_index = state_.halton_index + dim_offset * (sample_stride / base_scales_[i]) * mult_inverse_[i];
        }
        state_.halton_index = state_.halton_index % sample_stride;
    }
    state_.halton_index = state_.halton_index + u64(sample_index * sample_stride);

    // skip the first 2 dimensions due to filter importance sampling
    state_.dimension = 2;
}

HaltonSampler::HaltonSampler(const Vec2i &res, const CState &state)
{
    initialize_consts(res);
    state_ = state;
}

void HaltonSampler::save(ptr<CState> output)
{
    *output = state_;
}

f32 HaltonSampler::get1d()
{
    $if(state_.dimension >= PRIME_TABLE_SIZE)
    {
        state_.dimension = 2;
    };
    var result = sample_halton(state_.dimension, state_.halton_index);
    state_.dimension = state_.dimension + 1;
    return result;
}

CVec2f HaltonSampler::get2d()
{
    $if(state_.dimension + 1 >= PRIME_TABLE_SIZE)
    {
        state_.dimension = 2;
    };
    var x = sample_halton(state_.dimension, state_.halton_index);
    var y = sample_halton(state_.dimension + 1, state_.halton_index);
    state_.dimension = state_.dimension + 2;
    return CVec2f(x, y);
}

CVec3f HaltonSampler::get3d()
{
    $if(state_.dimension + 2 >= PRIME_TABLE_SIZE)
    {
        state_.dimension = 2;
    };
    var x = sample_halton(state_.dimension, state_.halton_index);
    var y = sample_halton(state_.dimension + 1, state_.halton_index);
    var z = sample_halton(state_.dimension + 2, state_.halton_index);
    state_.dimension = state_.dimension + 3;
    return CVec3f(x, y, z);
}

void HaltonSampler::initialize_consts(const Vec2i &res)
{
    for(int i = 0; i < 2; ++i)
    {
        const int base = (i == 0) ? 2 : 3;
        int scale = 1, exp = 0;
        while(scale < std::min(res[i], MAX_HALTON_RESOLUTION))
        {
            scale *= base;
            ++exp;
        }
        base_scales_[i] = scale;
        base_exponents_[i] = exp;
    }

    mult_inverse_[0] = static_cast<int>(multiplicative_inverse(base_scales_[1], base_scales_[0]));
    mult_inverse_[1] = static_cast<int>(multiplicative_inverse(base_scales_[0], base_scales_[1]));
}

BTRC_BUILTIN_END
