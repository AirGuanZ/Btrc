#include <btrc/utils/low_discrepancy.h>
#include <btrc/utils/prime.h>

// ref: pbrt-v4

BTRC_BEGIN

namespace
{
    i32 permutation_element(u32 i, u32 l, u32 p)
    {
        u32 w = l - 1;
        w = w | (w >> 1);
        w = w | (w >> 2);
        w = w | (w >> 4);
        w = w | (w >> 8);
        w = w | (w >> 16);
        $loop
        {
            i = i ^  (p);
            i = i *  (0xe170893d);
            i = i ^  (p >> 16);
            i = i ^  ((i & w) >> 4);
            i = i ^  (p >> 8);
            i = i *  (0x0929eb3f);
            i = i ^  (p >> 23);
            i = i ^  ((i & w) >> 1);
            i = i *  (1u | p >> 27);
            i = i *  (0x6935fa69);
            i = i ^  ((i & w) >> 11);
            i = i *  (0x74dcb303);
            i = i ^  ((i & w) >> 2);
            i = i *  (0x9e501cc3);
            i = i ^  ((i & w) >> 2);
            i = i *  (0xc860a3df);
            i = i &  (w);
            i = i ^  (i >> 5);
            $if(i < l)
            {
                $break;
            };
        };
        return i32((i + p) % l);
    }

} // namespace anonymous

u64 mix_bits(u64 v)
{
    v = v ^ (v >> 31);
    v = v * 0x7fb5d329728ea185;
    v = v ^ (v >> 27);
    v = v * 0x81dadef4bc2dd44d;
    v = v ^ (v >> 33);
    return v;
}

f32 radical_inverse(i32 base_index, u64 a)
{
    static auto func = cuj::function_contextless(
        [](i32 base_index, u64 a)
    {
        var prims = cuj::const_data(std::span<const int>(PRIME_TABLE));
        var base = u64(prims[base_index]);
        var inv_base = 1.0f / f32(base), inv_base_n = 1.0f;
        u64 reversed_digits = 0;
        $while(a != 0)
        {
            var next = a / base;
            var digit = a - next * base;
            reversed_digits = reversed_digits * base + digit;
            inv_base_n = inv_base_n * inv_base;
            a = next;
        };
        return cstd::min(f32(reversed_digits) * inv_base_n, 0x1.fffffep-1f);
    });
    return func(base_index, a);
}

f32 owen_scrambled_radical_inverse(i32 base_index, u64 a, u64 hash)
{
    var prims = cuj::const_data(std::span<const int>(PRIME_TABLE));
    var base = u64(prims[base_index]);
    var inv_base = 1.0f / f32(base), inv_base_m = 1.0f;
    u64 reversed_digits = 0;
    var digit_index = 0;
    $while(1.0f - inv_base_m < 1)
    {
        var next = a / base;
        var digit_value = a - next * base;
        var digit_hash = mix_bits(hash ^ reversed_digits);
        digit_value = u64(permutation_element(u32(digit_value), u32(base), u32(digit_hash)));
        reversed_digits = reversed_digits * base + digit_value;
        inv_base_m = inv_base_m * inv_base;
        digit_index = digit_index + 1;
        a = next;
    };
    return cstd::min(inv_base_m * f32(reversed_digits), 0x1.fffffep-1f);
}

u64 inverse_radical_inverse(u64 inverse, i32 base, i32 digits)
{
    u64 index = 0;
    $forrange(i, 0, digits)
    {
        (void)i;
        var digit = inverse % u64(base);
        inverse = inverse / u64(base);
        index = index * u64(base) + digit;
    };
    return index;
}

BTRC_END
