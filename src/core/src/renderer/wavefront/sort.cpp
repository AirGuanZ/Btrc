#include <btrc/core/renderer/wavefront/sort.h>

#include "sort/sort_kernel.h"

BTRC_WAVEFRONT_BEGIN

void SortPipeline::sort(
    int          current_active_state_count,
    const float *inct_t,
    const Vec4u *inct_uv_id,
    int32_t     *output_active_state_index)
{
    sort_states(
        current_active_state_count, inct_t,
        inct_uv_id, output_active_state_index);
}

BTRC_WAVEFRONT_END
