#pragma once

#include <btrc/utils/math/vec3.h>
#include <btrc/utils/math/vec4.h>

#include "./common.h"

BTRC_WFPT_BEGIN

BTRC_CPU void compute_preview_image(
    int          width,
    int          height,
    const Vec4f *value_buffer,
    const float *weight_buffer,
    Vec4f       *output_buffer);

BTRC_WFPT_END
