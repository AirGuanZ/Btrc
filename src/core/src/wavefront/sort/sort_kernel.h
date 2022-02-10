#pragma once

#include <btrc/core/utils/math/math.h>

BTRC_WAVEFRONT_BEGIN

BTRC_CPU void sort_states(
    int          total_state_count,
    const float *inct_t,
    const Vec4u *inct_uv_id,
    int32_t     *output_active_state_index);

BTRC_WAVEFRONT_END
