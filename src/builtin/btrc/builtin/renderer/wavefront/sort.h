#pragma once

#include <btrc/utils/math/math.h>

#include "./common.h"

BTRC_WFPT_BEGIN

class SortPipeline
{
public:

    void sort(
        int          current_active_state_count,
        const float *inct_t,
        const Vec4u *inct_uv_id,
        int32_t     *output_active_state_index);
};

BTRC_WFPT_END
