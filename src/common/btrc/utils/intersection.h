#pragma once

#include <btrc/utils/cmath/cmath.h>

BTRC_BEGIN

inline CVec3f intersection_offset(ref<CVec3f> inct_pos, ref<CVec3f> inct_nor)
{
    static auto func = cuj::function_contextless(
        "btrc_intersection_offset", [](ref<CVec3f> pos, ref<CVec3f> nor)
    {
        constexpr float ORIGIN      = 1.0f / 32.0f;
        constexpr float FLOAT_SCALE = 1.0f / 65536.0f;
        constexpr float INT_SCALE   = 256.0f;

        var of_i_x = i32(INT_SCALE * nor.x);
        var of_i_y = i32(INT_SCALE * nor.y);
        var of_i_z = i32(INT_SCALE * nor.z);

        var p_i_x = cuj::bitcast<f32>(cuj::bitcast<i32>(pos.x) + cstd::select(pos.x < 0, -of_i_x, i32(of_i_x)));
        var p_i_y = cuj::bitcast<f32>(cuj::bitcast<i32>(pos.y) + cstd::select(pos.y < 0, -of_i_y, i32(of_i_y)));
        var p_i_z = cuj::bitcast<f32>(cuj::bitcast<i32>(pos.z) + cstd::select(pos.z < 0, -of_i_z, i32(of_i_z)));

        var r_x = cstd::select(cstd::abs(pos.x) < ORIGIN, pos.x + FLOAT_SCALE * nor.x, f32(p_i_x));
        var r_y = cstd::select(cstd::abs(pos.y) < ORIGIN, pos.y + FLOAT_SCALE * nor.y, f32(p_i_y));
        var r_z = cstd::select(cstd::abs(pos.z) < ORIGIN, pos.z + FLOAT_SCALE * nor.z, f32(p_i_z));

        return CVec3f(r_x, r_y, r_z);
    });
    return func(inct_pos, inct_nor);
}

inline CVec3f intersection_offset(ref<CVec3f> inct_pos, ref<CVec3f> inct_nor, ref<CVec3f> next_dir)
{
    return intersection_offset(
        inct_pos, cstd::select(dot(inct_nor, next_dir) > 0, inct_nor, ref(-inct_nor)));
}

BTRC_END
