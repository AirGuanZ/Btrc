#pragma once

#include <btrc/core/common.h>

BTRC_WAVEFRONT_BEGIN

BTRC_CPU void sort_states(
    int          total_state_count,
    const float *inct_t,
    int32_t     *output_active_state_index,
    int32_t     *active_state_counter,
    int32_t     *inactive_state_counter);

BTRC_WAVEFRONT_END
