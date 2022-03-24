#pragma once

#include <btrc/utils/cmath/cmath.h>

BTRC_BEGIN

CVec3f intersection_offset(ref<CVec3f> inct_pos, ref<CVec3f> inct_nor);

CVec3f intersection_offset(ref<CVec3f> inct_pos, ref<CVec3f> inct_nor, ref<CVec3f> next_dir);

BTRC_END
