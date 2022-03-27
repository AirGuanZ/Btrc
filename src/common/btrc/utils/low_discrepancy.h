#pragma once

#include <btrc/utils/cmath/cmath.h>

BTRC_BEGIN

u64 mix_bits(u64 v);

f32 radical_inverse(i32 base_index, u64 a);

u64 inverse_radical_inverse(u64 inverse, i32 base, i32 digits);

f32 owen_scrambled_radical_inverse(i32 base_index, u64 a, u64 hash);

BTRC_END
